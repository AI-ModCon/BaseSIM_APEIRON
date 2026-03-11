# BaseSim Documentation

This directory contains the detailed reference docs for the framework's three main extension points:

- `model_harness.md`: model + data-stream integration contract
- `drift_detectors.md`: detector classes, detector config, and detector wiring
- `continuous_learning.md`: continual-learning trainer, updater modes, and training config

## Read Order

1. Start with `model_harness.md` to understand how models and stream loaders are exposed.
2. Read `drift_detectors.md` to see how monitoring decisions are made.
3. Read `continuous_learning.md` to understand what happens after drift is detected.

## Runtime Flow

1. `src/main.py` builds `Config` from TOML, env vars, and CLI overrides.
2. `examples/utils.py` selects a concrete `BaseModelHarness` by `cfg.data.name`.
3. `src/driver/continuous_monitor.py` evaluates streaming batches and calls a detector at intervals.
4. On drift, `src/training/continuous_trainer.py` runs a CL loop with an updater from `src/training/updater/create_updater.py`.
5. Logging is stage-aware (`eval`, `drift`, `cl`) via `src/logger/`.

## Config Sections

The parser expects these TOML sections:

- Required: `[model]`, `[data]`, `[train]`, `[drift_detection]`
- Optional: `[continual_learning]`, `[visualization]`
- Required top-level keys: `seed`, `device`, `multi_gpu`
- Common top-level optional key: `verbosity`

See each doc file for field-by-field meaning and defaults.
