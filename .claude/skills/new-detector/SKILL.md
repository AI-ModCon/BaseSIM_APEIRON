---
name: new-detector
description: |
  Create a new drift detector for the BaseSim framework. Use when the user
  wants to implement a custom drift detection algorithm beyond the built-in
  ADWIN, KSWIN, PageHinkley, ModelPerformance, ModelEval, and Ensemble detectors.
argument-hint: "<DetectorClassName>"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
  - Grep
---

Scaffold a new drift detector for the BaseSim framework.

## Arguments
- `$0`: Class name for the new detector (e.g., "CUSUMDetector", "DDMDetector", "HDDMDetector")

## Reference: Base Class and Existing Implementations

### Base class interface (BaseDriftDetector, DriftSignal, LearningRegime)
!`cat src/drift_detection/detectors/base.py`

### Statistical detector implementations (ADWIN, KSWIN, PageHinkley)
!`cat src/drift_detection/detectors/statistical_detectors.py`

### Detector factory/loader
!`cat src/drift_detection/load_drift_detector.py`

### Module exports
!`cat src/drift_detection/__init__.py`

### Config dataclass for detector parameters
!`grep -A 30 "class DriftDetectionCfg" src/config/configuration.py`

## Required Interface

Every detector must subclass `BaseDriftDetector` and implement:

```python
from drift_detection.detectors.base import BaseDriftDetector, DriftSignal, LearningRegime

class $0(BaseDriftDetector):
    def __init__(self, <hyperparams>, name: str = "<name>"):
        super().__init__(name)
        self._is_initialized = True
        # Store hyperparams, initialize internal state

    def update(self, value: float, **kwargs) -> DriftSignal:
        """Process new metric value, return drift signal.

        Must return DriftSignal with:
          - regime: LearningRegime (STABLE, CONTINUAL_LEARNING, FINE_TUNING, RETRAIN)
          - drift_detected: bool
          - drift_score: float (0-1, higher = more drift)
          - confidence: Optional[float] (0-1)
          - metadata: Optional[dict] (extra info)
        """

    def reset(self) -> None:
        """Reset to initial state (called after CL if reset_after_learning=true)."""
```

## Files to Create/Modify

### 1. Create detector implementation
Either add to `src/drift_detection/detectors/statistical_detectors.py` if it's a simple statistical detector, or create a new file `src/drift_detection/detectors/<name>.py` for complex detectors.

### 2. Update `src/drift_detection/load_drift_detector.py`
Add a new `elif detector_name == "$0":` branch in the `load_drift_detector()` factory function. The branch should:
- Extract relevant hyperparameters from `cfg.drift_detection`
- Construct and return an instance of the new detector

### 3. Update `src/drift_detection/__init__.py`
Add the new detector class to the imports and `__all__` list.

### 4. Update `src/config/configuration.py`
Add any new hyperparameters to `DriftDetectionCfg` with sensible defaults. Follow the naming convention of existing params (prefix with detector abbreviation, e.g., `adwin_`, `kswin_`, `ph_`).

## Procedure

1. Ask the user about the detection algorithm they want to implement:
   - What statistical test or method does it use?
   - What are its hyperparameters?
   - When should it signal CONTINUAL_LEARNING vs FINE_TUNING vs RETRAIN?
2. Read the existing detector implementations for pattern reference.
3. Create the detector class following the established patterns:
   - Constructor stores hyperparams and initializes state
   - `update()` processes values incrementally and returns `DriftSignal`
   - `reset()` clears state completely
4. Wire it into the loader factory in `load_drift_detector.py`.
5. Add config parameters to `DriftDetectionCfg`.
6. Update `__init__.py` exports.
7. Verify imports work:
   ```bash
   cd /home/user/BaseSim_Framework && poetry run python -c "from drift_detection import $0; print('Import OK:', $0)"
   ```

## Design Guidelines
- Detectors should be **stateful** and process one value at a time via `update()`
- The `update()` method must return a `DriftSignal` every call (even when no drift)
- Use `LearningRegime.STABLE` for no-drift signals
- `drift_score` should be normalized to 0-1 where possible
- Support `reset()` for the `reset_after_learning` config option
- Keep external dependencies minimal (prefer `river` for streaming algorithms)
