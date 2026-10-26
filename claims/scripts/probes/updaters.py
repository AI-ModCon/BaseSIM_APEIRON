#!/usr/bin/env python3
"""Behaviour probe: apeiron.training.updater, off the pinned set.

This is the layer the campaign has the least confidence in. A12 recorded that
the updater math is "nearly criterion-free", and A13 that the JVP/SAM updater
is interchangeable with a different algorithm as far as the tests can tell --
a reconstruction can implement something else entirely and pass. A33 puts the
kept-ratio at 61% with a spread of 5.

So this probe does not ask whether the hooks exist. It drives each updater
through a real forward/backward and compares the RESULTING WEIGHTS, exactly.
With SGD and no momentum one step is a closed-form function of the gradient,
so two implementations of the same mathematics land on identical numbers and
two implementations of different mathematics cannot. That makes "the gradient
this updater produces" an observable, which is the one thing A13 says the
criteria never check.

Also probed: the factory's dispatch over every documented mode and some
undocumented ones, the historical-data mixing contract (which promises the
per-step sample count is unchanged, and tops up from the current batch when
history is short), gradient accumulation scaling, and the regularisation
hooks' return values.

Determinism: fixed weights, fixed data, SGD without momentum, no sampling.
Losses and parameters are rounded at five decimals, far finer than any
mathematical difference between two updaters.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import torch  # noqa: E402

from _fixtures import (  # noqa: E402
    attempt, emit, fixed_loader, make_cfg, make_harness, params, show,
)

from apeiron.training.updater.create_updater import create_updater  # noqa: E402

MODES = ("base", "ewc_online", "kfac_online", "jvp_reg", "none")


def batch(n: int = 4, offset: float = 0.0):
    loader = fixed_loader(n, n, offset=offset)
    return next(iter(loader))


def built(mode: str, extra: str = "", overrides=()):
    cfg = make_cfg(extra=f"\n[continual_learning]\nupdate_mode = \"{mode}\"\n"
                         + extra, overrides=overrides)
    harness = make_harness(cfg)
    return create_updater(cfg, harness), harness


def one_step(mode: str, extra: str = "", hist: bool = False,
             overrides=()) -> str:
    """Run a single forward/backward/step and report loss plus final weights."""
    updater, harness = built(mode, extra, overrides)
    optimizer = harness.get_optimizer()
    updater.cl_preprocessing()
    updater.update_pre_fwd_bwd()
    optimizer.zero_grad()
    loss = updater.fwd_bwd(batch(), batch(offset=-0.5) if hist else None)
    penalty = updater.update_post_fwd_bwd()
    optimizer.step()
    updater.update_post_optimizer_call()
    updater.cl_postprocessing()
    return show({"loss": round(float(loss), 5),
                 "penalty": show(penalty),
                 "weights": params(harness.model)})


def after_cycle(mode: str, overrides=(), report: str = "weights") -> str:
    """One step taken AFTER a complete CL cycle has closed.

    This is the state the regularisers are built for: `cl_preprocessing`
    opens an accumulator, each step feeds it, and `cl_postprocessing` folds
    it into the Fisher estimate and anchors the reference weights. Only then
    does a penalty term have anything to multiply, so only then can a lambda
    make a difference. Probing before this point measures the plain SGD step
    that every mode shares.
    """
    updater, harness = built(mode, overrides=overrides)
    optimizer = harness.get_optimizer()

    # cycle one: populate whatever state the updater keeps
    updater.cl_preprocessing()
    for i in range(2):
        updater.update_pre_fwd_bwd()
        optimizer.zero_grad()
        updater.fwd_bwd(batch(offset=0.1 * i), batch(offset=-0.5))
        updater.update_post_fwd_bwd()
        optimizer.step()
        updater.update_post_optimizer_call()
    updater.cl_postprocessing()

    # Cycle two, and it must be MORE THAN ONE STEP. `cl_postprocessing`
    # re-anchors the reference weights to the current ones (theta_star <-
    # theta_final), so on the first step of the next cycle `p - theta_star`
    # is exactly zero and the penalty is zero for every lambda. It can only
    # bite once the weights have moved away from the anchor. Reporting after
    # a single step is what made the earlier version of this probe call
    # ewc_lambda and kfac_lambda dead.
    updater.cl_preprocessing()
    losses, penalties = [], []
    for i in range(3):
        updater.update_pre_fwd_bwd()
        optimizer.zero_grad()
        losses.append(round(float(
            updater.fwd_bwd(batch(offset=0.3 + 0.1 * i),
                            batch(offset=-0.5))), 5))
        penalties.append(show(updater.update_post_fwd_bwd()))
        if report == "grads" and i == 2:
            return show([[name, show(p.grad)] for name, p in
                         sorted(harness.model.named_parameters())])
        optimizer.step()
        updater.update_post_optimizer_call()
    updater.cl_postprocessing()
    return show({"losses": losses, "penalties": penalties,
                 "weights": params(harness.model)})


def grads(mode: str, extra: str = "", hist: bool = False) -> str:
    """The gradient itself -- the quantity A13 says nothing pins."""
    updater, harness = built(mode, extra)
    harness.model.zero_grad()
    updater.cl_preprocessing()
    updater.update_pre_fwd_bwd()
    updater.fwd_bwd(batch(), batch(offset=-0.5) if hist else None)
    updater.update_post_fwd_bwd()
    return show([[name, show(p.grad)] for name, p in
                 sorted(harness.model.named_parameters())])


def main() -> int:
    out = {}

    # --- 1. the factory's dispatch ----------------------------------------
    for mode in MODES:
        out[f"create[{mode}]"] = attempt(
            lambda m=mode: type(built(m)[0]).__name__)
    for mode in ("Base", "BASE", "", " base ", "ewc", "jvp", "nope", "None"):
        out[f"create[{mode!r}]"] = attempt(
            lambda m=mode: type(built(m)[0]).__name__)

    # --- 2. one step, weights compared exactly ----------------------------
    for mode in MODES:
        out[f"step[{mode}]"] = attempt(lambda m=mode: one_step(m))
        out[f"grads[{mode}]"] = attempt(lambda m=mode: grads(m))

    # --- 3. the regularisation knobs actually reaching the math -----------
    # A12/A13 ask whether these knobs are wired in at all. Answering that
    # needs the state in which they APPLY, and the first version of this
    # probe did not reach it: EWC's and KFAC's Fisher estimates are all
    # zeros until a CL cycle has closed, so `lambda * 0 * diff` is zero for
    # every lambda, and JVP's own docstring says that with no historical
    # batch it is "a plain current-batch step". The probe therefore reported
    # every knob as dead and every mode as identical -- a false finding about
    # APEIRON produced entirely by driving the trivial case.
    for mode, knob in (("ewc_online", "ewc_lambda"),
                       ("kfac_online", "kfac_lambda")):
        for value in ("0.0", "1.0", "1000.0"):
            out[f"step[{mode},{knob}={value},cold]"] = attempt(
                lambda m=mode, k=knob, v=value: one_step(
                    m, overrides=["--set", f"continual_learning.{k}={v}"]))
            out[f"step[{mode},{knob}={value},after-cycle]"] = attempt(
                lambda m=mode, k=knob, v=value: after_cycle(
                    m, overrides=["--set", f"continual_learning.{k}={v}"]))
    for knob in ("jvp_rho_theta", "jvp_rho_x", "jvp_data_sign"):
        for value in ("0.0", "0.05", "1.0", "-1.0"):
            # with a historical batch, which is the only path where the SAM
            # perturbation is taken at all
            out[f"step[jvp_reg,{knob}={value},with-hist]"] = attempt(
                lambda k=knob, v=value: one_step(
                    "jvp_reg", hist=True,
                    overrides=["--set", f"continual_learning.{k}={v}"]))
            out[f"step[jvp_reg,{knob}={value},no-hist]"] = attempt(
                lambda k=knob, v=value: one_step(
                    "jvp_reg",
                    overrides=["--set", f"continual_learning.{k}={v}"]))
    # and the modes compared against each other in the warm state
    for mode in MODES:
        out[f"after-cycle[{mode}]"] = attempt(lambda m=mode: after_cycle(m))
        out[f"after-cycle-grads[{mode}]"] = attempt(
            lambda m=mode: after_cycle(m, report="grads"))

    # --- 4. the historical-mixing contract --------------------------------
    # The docstring promises the per-step sample count is unchanged and that
    # a short historical batch is topped up from the current one. Both are
    # observable through the loss and the resulting weights.
    for mode in MODES:
        for mix in ("true", "false"):
            out[f"mix[{mode},enabled={mix},hist-given]"] = attempt(
                lambda m=mode, x=mix: one_step(
                    m, hist=True,
                    overrides=["--set",
                               f"continual_learning.mix_historic_data={x}"]))
            out[f"mix[{mode},enabled={mix},no-hist]"] = attempt(
                lambda m=mode, x=mix: one_step(
                    m, hist=False,
                    overrides=["--set",
                               f"continual_learning.mix_historic_data={x}"]))

    def sized_mix(n_cur: int, n_hist: int) -> str:
        """Short, equal and oversized histories against a fixed current batch."""
        updater, harness = built(
            "base", overrides=["--set",
                               "continual_learning.mix_historic_data=true"])
        harness.model.zero_grad()
        loss = updater.fwd_bwd(batch(n_cur), batch(n_hist, offset=-0.5))
        return show({"loss": round(float(loss), 5),
                     "weights": params(harness.model)})

    for n_cur, n_hist in ((4, 4), (4, 1), (4, 0), (4, 8), (1, 1), (2, 3)):
        out[f"mix[sizes cur={n_cur},hist={n_hist}]"] = attempt(
            lambda a=n_cur, b=n_hist: sized_mix(a, b))

    # --- 5. gradient accumulation scaling ---------------------------------
    for steps in ("1", "2", "4"):
        out[f"step[base,grad_accum={steps}]"] = attempt(
            lambda s=steps: one_step(
                "base", overrides=["--set", f"train.grad_accumulation_steps={s}"]))
    out["step[base,grad_accum=0]"] = attempt(
        lambda: one_step("base",
                         overrides=["--set", "train.grad_accumulation_steps=0"]))

    # --- 6. degenerate batches --------------------------------------------
    def raw_fwd(mode: str, b) -> str:
        updater, harness = built(mode)
        harness.model.zero_grad()
        return show(round(float(updater.fwd_bwd(b)), 5))

    for mode in ("base", "jvp_reg", "none"):
        out[f"fwd_bwd[{mode},single-sample]"] = attempt(
            lambda m=mode: raw_fwd(m, batch(1)))
        out[f"fwd_bwd[{mode},empty]"] = attempt(
            lambda m=mode: raw_fwd(m, (torch.zeros(0, 4),
                                       torch.zeros(0, dtype=torch.long))))
        out[f"fwd_bwd[{mode},mismatched]"] = attempt(
            lambda m=mode: raw_fwd(m, (torch.zeros(4, 4),
                                       torch.zeros(2, dtype=torch.long))))

    # --- 7. hook return values and idempotence ----------------------------
    for mode in MODES:
        updater, harness = built(mode)
        out[f"hooks[{mode},preprocessing]"] = attempt(
            lambda u=updater: show(u.cl_preprocessing()))
        out[f"hooks[{mode},pre-fwd]"] = attempt(
            lambda u=updater: show(u.update_pre_fwd_bwd()))
        out[f"hooks[{mode},post-fwd-before-any-step]"] = attempt(
            lambda u=updater: show(u.update_post_fwd_bwd()))
        out[f"hooks[{mode},post-optimizer]"] = attempt(
            lambda u=updater: show(u.update_post_optimizer_call()))
        out[f"hooks[{mode},postprocessing]"] = attempt(
            lambda u=updater: show(u.cl_postprocessing()))
        out[f"hooks[{mode},weights-untouched-by-hooks]"] = show(
            params(harness.model))

    # --- 8. two steps, to catch state that should or should not persist ---
    def two_steps(mode: str) -> str:
        updater, harness = built(mode)
        optimizer = harness.get_optimizer()
        seen = []
        for i in range(2):
            updater.cl_preprocessing()
            updater.update_pre_fwd_bwd()
            optimizer.zero_grad()
            loss = updater.fwd_bwd(batch(offset=0.1 * i))
            updater.update_post_fwd_bwd()
            optimizer.step()
            updater.update_post_optimizer_call()
            updater.cl_postprocessing()
            seen.append(round(float(loss), 5))
        return show({"losses": seen, "weights": params(harness.model)})

    for mode in MODES:
        out[f"two-steps[{mode}]"] = attempt(lambda m=mode: two_steps(m))

    emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
