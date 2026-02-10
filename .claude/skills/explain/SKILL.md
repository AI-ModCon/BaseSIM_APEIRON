---
name: explain
description: |
  Explain parts of the BaseSim framework architecture, code flow, or concepts.
  Use when the user asks how the framework works, what a module does, how drift
  detection or continual learning is implemented, or wants a codebase overview.
argument-hint: "[topic: architecture|drift|updaters|config|monitoring|harness|profiling|logging|pipeline]"
user-invocable: true
context: fork
agent: Explore
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash
---

Explain the BaseSim (SIM: Self Improving Model) framework architecture and internals.

## Arguments
- `$ARGUMENTS`: Optional topic to focus on. If empty, provide a high-level architecture overview.

## Available Topics

- **architecture** -- Full system overview, data flow, component interactions
- **drift** -- Drift detection algorithms, DriftSignal, LearningRegime, detector lifecycle
- **updaters** -- CL update strategies, BaseUpdater hook lifecycle, EWC/JVP/KFAC internals
- **config** -- TOML configuration system, dataclass hierarchy, override mechanisms
- **monitoring** -- ContinuousMonitor loop, batch processing, drift checking, stream extension
- **harness** -- BaseModelHarness ABC, how to implement, data stream pattern
- **profiling** -- FLOPSProfiler, what is measured, limitations
- **logging** -- Logger stages, WandB integration, CSV output, console verbosity
- **pipeline** -- End-to-end flow from `main.py` through monitoring, drift, training, and back

## Key Source File Locations

- Entry point: `src/main.py`
- Config system: `src/config/configuration.py`
- Model harness ABC: `src/model/torch_model_harness.py`
- Continuous monitor: `src/driver/continuous_monitor.py`
- Continuous trainer: `src/training/continuous_trainer.py`
- Updater base + implementations: `src/training/updater/`
- Drift detector base + implementations: `src/drift_detection/detectors/`
- Drift detector loader: `src/drift_detection/load_drift_detector.py`
- Example harnesses: `examples/mnist/model.py`, `examples/cifar/model.py`, `examples/imagenet/model.py`
- Example factory: `examples/utils.py`
- Logger: `src/logger/`
- Profiler: `src/profilers/`
- Visualization: `src/visualization/`

## Procedure

1. If a topic is specified in `$ARGUMENTS`, read the relevant source files and provide a detailed explanation.
2. If no topic is specified, provide the full architecture overview covering all components.
3. Include in your explanation:
   - What each component does and why
   - How components interact (caller/callee relationships)
   - Key classes, their responsibilities, and their abstract interfaces
   - Data flow through the system
   - Extension points for customization
4. Reference specific file paths and line numbers when explaining internals.
5. For algorithmic explanations (drift detection, EWC, JVP, KFAC), explain the underlying concept and how it maps to the code.
6. Keep explanations grounded in the actual source code -- read files before explaining them.
