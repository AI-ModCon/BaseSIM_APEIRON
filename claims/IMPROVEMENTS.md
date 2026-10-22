# From findings to improvements

How the reconstruction evidence in [LOG.md](LOG.md) converts into APEIRON
changes. Four channels, ordered by effort-to-payoff; item numbers refer to
log entries.

## 1. The quick-fix batch (mechanical, PR-ready)

- **Lazy `apeiron/__init__.py`** (A1): PEP 562 module `__getattr__` so a
  layer imports without the whole dependency tree. The enabler for
  everything below.
- **Dead surface** (A3): `evaluation.py` has no callers and a wrong
  docstring; `accuracy_topk` has no callers and no tests. Delete or
  specify.
- **Leaky evidently guard** (A7): `test_ensemble` and
  `test_ensemble_without_sub_detectors_raises` NameError instead of
  skipping where evidently is absent; `HAS_EVIDENTLY` actually measures
  pandas.
- **Free `EnsembleDetector`** (A6): pure voting logic living in the
  pandas-coupled module.
- **`.gitignore` blanket `*.toml` / `*.json`** (A4): silently drops new
  fixtures and configs.
- **river 0.23 vs Python 3.14** (A8): declared support is wider than the
  installable reality; bump river or narrow `requires-python`.

## 2. The test backlog, ranked by measured consequence

**Round one is done** (see LOG A14, A17). Mutation by layer, before and
after: config 0.36 → 0.68, model-harness 0.16 → 0.56, updaters 0.12 →
0.37, drift-detection 0.23 → 0.30; evaluation 0.76 and logger 0.32
untouched so far. The entry points went from no tests at all to end-to-end
coverage, and the Logger facade from none to full. The confirmation that
matters: regrown implementations now land on the documented algorithms
instead of plausible substitutes.

**Round two (the mechanical sweep) is done.** Line coverage of `src/`
went 86% to **97%** (2,201 statements, 70 uncovered), 280 to 459 tests.
Newly covered: the mlflow backend (25% to 100%), the wandb run-attached
paths (88% to 100%), the CL trigger schedules, FIFO checkpointing, KFAC's
Conv2d branches, the optimizer profiler and ATen FLOPs estimate, the ABC
contracts and updater guard clauses, the full adaptation cycle
(checkpoint, detector reset, profiled steps), both runners' options, and
device selection. `ContinuousMonitor`, `ContinuousTrainer`,
`torch_model_harness`, `logger.py`, `console_logger`, `wandb_logger`,
`mlflow_logger`, `count_flops` and the updater package are at or near
100%.

What is left, and why:

**Round three: the evidently pipeline is done too.** `src/` is at **99%**
(2,201 statements, 10 uncovered), 501 tests.
`ModelPerformanceDetector` went 64% to 100% in two tiers — real evidently
reports checking the contract with the installed library, and a stubbed
`Report` for the four result-shape refusals and the regime boundaries
real data cannot place reliably. Pinned into the claim, drift-detection
mutation went **0.30 to 0.60**, with comparison mutations rising from 0
of 4 to 4 of 4 across 222 sites and arithmetic from 0 to 2 of 2.

What is left:

- **10 structurally unreachable statements**, which is the ceiling: the
  Python 3.10 `tomli` import fallback (dead on a 3.13+ project), three
  `if __name__ == "__main__"` guards, one `torch.cuda.empty_cache()`, and
  two dead lines in `evaluation.py` whose condition the function already
  returned on. GPU-only code is covered by patching the probes rather
  than needing hardware.
- **Still unclaimed layers**: trainer, monitor, and profilers have the
  coverage but no reticuli claim, so their constraint is unmeasured.
- `accuracy_topk` and `evaluation.py` await the delete-or-specify
  decision — see A19, where testing `evaluation.py` turned up a real
  6x loss error.

The original backlog, for reference:

1. **Updater numerics** (A12, A13 — highest value). The codex regrowth
   proved a different algorithm passes: the regrown JVP updater perturbs
   on the combined gradient where the Bertsimas formulation requires the
   current-only direction, and drops `jvp_rho_x` entirely. Fix: fixed
   tiny model, fixed seed, assert actual update directions and penalty
   values for EWC, KFAC, and JVP — including a test that distinguishes
   current-only from combined-gradient perturbation.
2. **Drift severity boundaries** (A9): comparison mutations 0 of 222
   sites; assert the severity-to-regime mapping at its thresholds.
3. **Config field assertions** (A2, A5): a 148-of-403-line regrowth
   passes; composition pins the detector fields, direct assertions must
   pin the rest (`LoggingCfg`, `verbosity`, updater hyperparameter
   defaults).
4. **Harness transfer metrics** (A10): `register_task`,
   `eval_past_tasks`, `task_diagonals` — documented, zero criteria.
5. **Entry points**: `main.py`, `drift_only.py`, `cl_only.py` have no
   tests at all, so no claim can cover them; tests come first.

The loop is measurable: write tests → `ret assess` (mutation climbs) →
codex regrowth (divergence shrinks toward the real implementation).
Progress is a number moving, not a feeling.

## 3. Claims as standing CI

`ret verify` makes any edit to pinned criteria a visible identity event in
review; `ret audit` re-earns every verdict cold in a hermetic,
hash-pinned room on each merge. This is the mechanism that would have
caught the MNIST golden reference drifting stale on `main`.

## 4. The measured dependency map fixes packaging

The claim rooms established each layer's true import surface: config and
evaluation are torch-only; drift is river+numpy, with pandas/evidently
needed only for `ModelPerformanceDetector`; the loggers need
wandb+pandas. That is a ready-made `[project.optional-dependencies]`
design (`apeiron[drift]`, `apeiron[wandb]`, …), which matters for the HPC
deployments where evidently wheels are already a known gap. Requires the
lazy `__init__` first.

## Limits, stated

Passing a gate is not substitutability — measured twice here (a
gate-passing config broke the drift factory; a gate-passing updater was a
different algorithm). Every regrowth divergence needs a human verdict:
granted freedom, or hole to pin. Mutation scores are not comparable
across programs; they are compared before/after on the same layer.
