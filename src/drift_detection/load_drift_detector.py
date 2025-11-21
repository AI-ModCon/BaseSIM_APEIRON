from src.config.configuration import Config
from src.drift_detection.detectors.base import BaseDriftDetector


def load_drift_detector(cfg: Config) -> BaseDriftDetector:
    """Dynamically load and instantiate a drift detector based on its name.

    Args:
        detector_name (str): Name of the drift detector class to load.
        cfg: Configuration object containing parameters for the detector.

    Returns:
        BaseDriftDetector: An instance of the specified drift detector.
    """
    detector_name = cfg.drift_detection.detector_name

    if detector_name == "ADWINDetector":
        from src.drift_detection.detectors.statistical_detectors import ADWINDetector

        detector_instance = ADWINDetector()
    elif detector_name == "KSWINDetector":
        from src.drift_detection.detectors.statistical_detectors import KSWINDetector

        detector_instance = KSWINDetector()
    elif detector_name == "PageHinkleyDetector":
        from src.drift_detection.detectors.statistical_detectors import (
            PageHinkleyDetector,
        )

        detector_instance = PageHinkleyDetector()
    elif detector_name == "ModelPerformanceDetector":
        from src.drift_detection.detectors.model_performance_detector import (
            ModelPerformanceDetector,
        )

        detector_instance = ModelPerformanceDetector()
    elif detector_name == "EnsembleDetector":
        from src.drift_detection.detectors.model_performance_detector import (
            EnsembleDetector,
        )

        detector_instance = EnsembleDetector()
    else:
        raise ValueError(f"Unknown drift detector: {detector_name}")

    return detector_instance
