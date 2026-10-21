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

Mutation by layer: evaluation 0.76, config 0.36, logger 0.32, drift 0.23,
harness 0.16, updaters 0.12 — criteria strength falls as the code gets
closer to the science.

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
