#!/usr/bin/env python3
"""Behaviour probe: apeiron.model.torch_model_harness, off the pinned set.

This layer has the highest kept-ratio of the ten (A33: 81%) and the thinnest
mutation score of the ten (A10: 16%) -- a reconstruction that comes back
almost whole, judged by criteria that kill almost nothing. That combination
is exactly what a behaviour probe is for.

Where it aims:

  the task registry   register_task / eval_past_tasks / task_diagonals are
                      the machinery behind the transfer metrics a continual
                      learning paper reports (A30). They freeze a window,
                      keep records index-aligned, and EVICT the oldest past a
                      cap. Nothing pins the eviction, the alignment, or what
                      an empty registry returns -- and a misalignment here
                      silently corrupts every forgetting number downstream.
  the scalar coercion _to_scalar turns a tensor of any rank, or a float, into
                      one number by taking a mean. Rank, dtype and degenerate
                      shapes are all unpinned.
  batch unpacking     _unpack is documented as the override point for a
                      subclassed dataloader, so its behaviour on anything
                      other than a 2-tuple is part of the contract callers
                      extend.
  the legacy alias    __init_subclass__ bridges the misspelled `get_optmizer`
                      onto `get_optimizer` and warns. Which spelling wins
                      when a subclass defines both, and whether the warning
                      fires, is behaviour a subclass author depends on.

A concrete harness over fixed tensors comes from _fixtures, identical on both
sides, so what differs is only the base class under it.
"""
import sys
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import pathlib  # noqa: E402

import torch  # noqa: E402

from _fixtures import (  # noqa: E402
    Tiny, attempt, emit, fixed_loader, make_cfg, make_harness, show,
    tmpdir,
)

from apeiron.model.torch_model_harness import BaseModelHarness  # noqa: E402


def main() -> int:
    out = {}

    # --- 1. _to_scalar over every shape a metric might arrive as ----------
    for name, value in (
        ("float", 1.5),
        ("int", 2),
        ("bool", True),
        ("rank0", torch.tensor(3.5)),
        ("rank1", torch.tensor([1.0, 2.0, 3.0])),
        ("rank2", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("empty", torch.zeros(0)),
        ("long-dtype", torch.tensor([1, 2, 3])),
        ("bool-dtype", torch.tensor([True, False])),
        ("nan", torch.tensor([float("nan"), 1.0])),
        ("inf", torch.tensor([float("inf"), 1.0])),
        ("requires-grad", torch.tensor([1.0, 2.0], requires_grad=True)),
        ("string", "nope"),
        ("none", None),
        ("list", [1.0, 2.0]),
    ):
        out[f"_to_scalar[{name}]"] = attempt(BaseModelHarness._to_scalar, value)

    # --- 2. _unpack, the documented override point ------------------------
    harness = make_harness()
    for name, batch in (
        ("pair", (torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))),
        ("list-pair", [torch.zeros(2, 4), torch.zeros(2, dtype=torch.long)]),
        ("triple", (torch.zeros(2, 4), torch.zeros(2), torch.zeros(2))),
        ("single", (torch.zeros(2, 4),)),
        ("empty", ()),
        ("dict", {"x": torch.zeros(2, 4), "y": torch.zeros(2)}),
        ("nested", ((torch.zeros(2, 4), torch.zeros(2)), torch.zeros(2))),
    ):
        out[f"_unpack[{name}]"] = attempt(
            lambda b=batch: [show(t) for t in harness._unpack(b)])

    # --- 3. eval and the loaders ------------------------------------------
    out["eval[fixed-stream]"] = attempt(lambda: make_harness().eval())
    out["history_eval[fixed]"] = attempt(lambda: make_harness().history_eval())
    out["_eval_loader[empty-loader]"] = attempt(
        lambda: make_harness()._eval_loader(fixed_loader(0, 4)))
    out["_eval_loader[single-sample]"] = attempt(
        lambda: make_harness()._eval_loader(fixed_loader(1, 1)))
    out["_eval_loader[batch-larger-than-data]"] = attempt(
        lambda: make_harness()._eval_loader(fixed_loader(3, 16)))
    out["_eval_loader[ragged-batches]"] = attempt(
        lambda: make_harness()._eval_loader(fixed_loader(7, 3)))
    # does evaluating leave the model in eval mode?
    left = make_harness()
    left.model.train()
    left.eval()
    out["eval[leaves-model-in-eval]"] = show(not left.model.training)

    # --- 4. the task registry: alignment, eviction, empty -----------------
    out["registry[empty-past]"] = attempt(lambda: make_harness().eval_past_tasks())
    out["registry[empty-diagonals]"] = attempt(
        lambda: make_harness().task_diagonals)

    def registered(count: int, cap=None, diagonals=None):
        h = make_harness()
        if cap is not None:
            h.max_task_records = cap
        for i in range(count):
            h.register_task(diagonals[i] if diagonals else [float(i)])
        return h

    for count in (1, 2, 3):
        h = registered(count)
        out[f"registry[{count}-tasks,diagonals]"] = show(h.task_diagonals)
        out[f"registry[{count}-tasks,past-len]"] = show(len(h.eval_past_tasks()))
        out[f"registry[{count}-tasks,aligned]"] = show(
            len(h.eval_past_tasks()) == len(h.task_diagonals))
        out[f"registry[{count}-tasks,past-values]"] = show(h.eval_past_tasks())

    # eviction: which records survive the cap, and do the diagonals follow?
    for cap in (1, 2, 5):
        h = registered(4, cap=cap, diagonals=[[10.0], [20.0], [30.0], [40.0]])
        out[f"registry[cap={cap},diagonals]"] = show(h.task_diagonals)
        out[f"registry[cap={cap},aligned]"] = show(
            len(h.eval_past_tasks()) == len(h.task_diagonals))
    out["registry[cap=0]"] = attempt(
        lambda: registered(2, cap=0).task_diagonals)
    out["registry[default-cap]"] = attempt(
        lambda: make_harness().max_task_records)

    # a diagonal of the wrong width, and a non-list
    out["registry[empty-diagonal]"] = attempt(
        lambda: registered(1, diagonals=[[]]).task_diagonals)
    out["registry[wide-diagonal]"] = attempt(
        lambda: registered(1, diagonals=[[1.0, 2.0, 3.0]]).task_diagonals)
    out["registry[scalar-diagonal]"] = attempt(
        lambda: registered(1, diagonals=[1.0]).task_diagonals)
    # registering freezes a COPY: advancing the stream must not change it
    moved = registered(1, diagonals=[[1.0]])
    before = moved.eval_past_tasks()
    moved.update_data_stream()
    out["registry[frozen-against-stream]"] = show(
        before == moved.eval_past_tasks())

    # --- 5. the deprecated optimizer spelling -----------------------------
    def subclass(body: str):
        """Build a harness subclass that defines the named methods."""
        ns = {}
        exec(  # noqa: S102 — the probe is the test
            "import torch\n"
            "from apeiron.model.torch_model_harness import BaseModelHarness\n"
            "class Sub(BaseModelHarness):\n"
            "    def get_stream_dataloader(self): return None\n"
            "    def get_train_dataloaders(self): return None, None\n"
            "    def get_hist_dataloaders(self): return None, None\n"
            "    def update_data_stream(self): pass\n"
            "    def get_criterion(self): return None\n"
            + body, ns)
        return ns["Sub"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = attempt(
            subclass, "    def get_optmizer(self): return 'legacy'\n")
        out["alias[legacy-only,built]"] = legacy
        out["alias[legacy-only,warned]"] = show(
            [str(w.category.__name__) for w in caught])
    out["alias[legacy-only,resolves]"] = attempt(
        lambda: subclass("    def get_optmizer(self): return 'legacy'\n")
        .get_optimizer(None))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        subclass("    def get_optimizer(self): return 'new'\n")
        out["alias[new-only,warned]"] = show(
            [str(w.category.__name__) for w in caught])

    both = ("    def get_optimizer(self): return 'new'\n"
            "    def get_optmizer(self): return 'legacy'\n")
    out["alias[both,resolves]"] = attempt(
        lambda: subclass(both).get_optimizer(None))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        subclass(both)
        out["alias[both,warned]"] = show(
            [str(w.category.__name__) for w in caught])
    out["alias[neither]"] = attempt(lambda: subclass("    pass\n"))

    # --- 6. checkpoints ---------------------------------------------------
    # `save_ckpt` writes to `Path(cfg.model.ckpts_path)`, and the default for
    # that field is "" -- which resolves to the PROCESS WORKING DIRECTORY. An
    # earlier version of this probe therefore wrote a `latest` pointer file
    # into the repository root. Every checkpoint case below is pointed at a
    # temporary directory, and the default's behaviour is recorded by reading
    # the resolved path rather than by writing to it.
    ckpt_dir = tmpdir()
    out["ckpts[disabled-by-default]"] = attempt(
        lambda: make_harness().ckpts_enabled)
    out["ckpts[default-path-resolves-to]"] = attempt(
        lambda: str(pathlib.Path(make_harness().cfg.model.ckpts_path)))
    enabled = make_harness(make_cfg(overrides=[
        "--set", "model.max_ckpts=2",
        "--set", f"model.ckpts_path={ckpt_dir}"]))
    out["ckpts[enabled-when-max-set]"] = attempt(lambda: enabled.ckpts_enabled)
    out["ckpts[payload-keys]"] = attempt(
        lambda: sorted(make_harness().build_checkpoint_payload().keys())
        if isinstance(make_harness().build_checkpoint_payload(), dict)
        else type(make_harness().build_checkpoint_payload()).__name__)
    out["ckpts[save-when-disabled]"] = attempt(
        lambda: make_harness(make_cfg(overrides=[
            "--set", f"model.ckpts_path={ckpt_dir}"])).save_ckpt(0))
    out["ckpts[save-when-enabled]"] = attempt(lambda: enabled.save_ckpt(0))
    out["ckpts[files-written]"] = attempt(
        lambda: sorted(p.name for p in ckpt_dir.iterdir()))

    # --- 7. construction surface ------------------------------------------
    out["init[abstract-direct]"] = attempt(
        lambda: BaseModelHarness(make_cfg(), Tiny()))
    out["init[eval_metrics-present]"] = attempt(
        lambda: sorted(make_harness().eval_metrics))
    out["init[cfg-device]"] = attempt(lambda: str(make_harness().cfg.device))

    emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
