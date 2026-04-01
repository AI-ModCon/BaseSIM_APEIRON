from apeiron.config.configuration import Config
from apeiron.drift_detection.detectors.base import BaseDriftDetector


def load_drift_detector(cfg: Config) -> BaseDriftDetector:
    """Dynamically load and instantiate a drift detector based on its name.

    Args:
        detector_name (str): Name of the drift detector class to load.
        cfg: Configuration object containing parameters for the detector.

    Returns:
        BaseDriftDetector: An instance of the specified drift detector.
    """
    detector_name = cfg.drift_detection.detector_name

    detector_instance: BaseDriftDetector
    if detector_name == "ADWINDetector":
        from apeiron.drift_detection.detectors.statistical_detectors import ADWINDetector

        detector_instance = ADWINDetector(
            delta=cfg.drift_detection.adwin_delta,
            minor_threshold=cfg.drift_detection.adwin_minor_threshold,
            moderate_threshold=cfg.drift_detection.adwin_moderate_threshold,
        )
    elif detector_name == "KSWINDetector":
        from apeiron.drift_detection.detectors.statistical_detectors import KSWINDetector

        detector_instance = KSWINDetector(
            alpha=cfg.drift_detection.kswin_alpha,
            window_size=cfg.drift_detection.kswin_window_size,
            stat_size=cfg.drift_detection.kswin_stat_size,
        )
    elif detector_name == "PageHinkleyDetector":
        from apeiron.drift_detection.detectors.statistical_detectors import (
            PageHinkleyDetector,
        )

        detector_instance = PageHinkleyDetector(
            min_instances=cfg.drift_detection.ph_min_instances,
            delta=cfg.drift_detection.ph_delta,
            threshold=cfg.drift_detection.ph_threshold,
            alpha=cfg.drift_detection.ph_alpha,
        )
    elif detector_name == "ModelPerformanceDetector":
        from apeiron.drift_detection.detectors.model_performance_detector import (
            ModelPerformanceDetector,
        )

        detector_instance = ModelPerformanceDetector()
    elif detector_name == "EnsembleDetector":
        raise NotImplementedError(
            "EnsembleDetector requires configuration of sub-detectors, "
            "which is not yet implemented. Use ADWINDetector, KSWINDetector, "
            "PageHinkleyDetector, or ModelPerformanceDetector instead."
        )

        # from apeiron.drift_detection.detectors.model_performance_detector import (
        #    EnsembleDetector,
        # )

        # detector_instance = EnsembleDetector()

    elif detector_name == "EvalDetector":
        from apeiron.drift_detection.detectors.model_performance_detector import (
            ModelEvalDetector,
        )

        detector_instance = ModelEvalDetector()
    else:
        raise ValueError(f"Unknown drift detector: {detector_name}")

    return detector_instance
