# API Reference

Generated from the docstrings in `src/apeiron/`. The heavy runtime dependencies
are mocked during the docs build, so signatures involving `torch` types render
as plain names.

## Top-level package

Everything below is re-exported from `apeiron` itself:

```python
from apeiron import (
    Config, ModelCfg, DataCfg, TrainCfg, ContinualLearningCfg,
    DriftDetectionCfg, VisualizationCfg, LoggingCfg, build_config,
    BaseModelHarness, ContinuousMonitor, ContinuousTrainer, BaseUpdater,
    BaseDriftDetector, DriftSignal, LearningRegime,
    ADWINDetector, KSWINDetector, PageHinkleyDetector,
    ModelPerformanceDetector, ModelEvalDetector, EnsembleDetector,
    Logger, get_logger,
)
```

```{toctree}
:maxdepth: 2

config
model
driver
drift_detection
training
evaluation
logger
profilers
```
