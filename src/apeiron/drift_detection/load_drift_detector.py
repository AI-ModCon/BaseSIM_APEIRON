from apeiron.config.configuration import Config
from apeiron.drift_detection.detectors.base import BaseDriftDetector


def _build_detector(detector_name: str, cfg: Config) -> BaseDriftDetector:
    """Instantiate a single (non-ensemble) drift detector from the config.

    Args:
        detector_name (str): Name of the drift detector class to build.
        cfg: Configuration object containing parameters for the detector.

    Returns:
        BaseDriftDetector: An instance of the specified drift detector.
    """
    detector_instance: BaseDriftDetector
    if detector_name == "ADWINDetector":
        from apeiron.drift_detection.detectors.statistical_detectors import (
            ADWINDetector,
        )

        detector_instance = ADWINDetector(
            delta=cfg.drift_detection.adwin_delta,
            minor_threshold=cfg.drift_detection.adwin_minor_threshold,
            moderate_threshold=cfg.drift_detection.adwin_moderate_threshold,
        )
    elif detector_name == "KSWINDetector":
        from apeiron.drift_detection.detectors.statistical_detectors import (
            KSWINDetector,
        )

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
    elif detector_name == "EvalDetector":
        from apeiron.drift_detection.detectors.model_performance_detector import (
            ModelEvalDetector,
        )

        detector_instance = ModelEvalDetector()
    else:
        raise ValueError(f"Unknown drift detector: {detector_name}")

    return detector_instance


def load_drift_detector(cfg: Config) -> BaseDriftDetector:
    """Dynamically load and instantiate a drift detector based on its name.

    Args:
        cfg: Configuration object containing parameters for the detector.

    Returns:
        BaseDriftDetector: An instance of the specified drift detector.
    """
    detector_name = cfg.drift_detection.detector_name

    if detector_name != "EnsembleDetector":
        return _build_detector(detector_name, cfg)

    from apeiron.drift_detection.detectors.model_performance_detector import (
        EnsembleDetector,
    )

    sub_names = cfg.drift_detection.ensemble_detectors
    if not sub_names:
        raise ValueError(
            "EnsembleDetector requires [drift_detection] ensemble_detectors to list "
            "at least one sub-detector, e.g. "
            'ensemble_detectors = ["ADWINDetector", "KSWINDetector"]'
        )
    if "EnsembleDetector" in sub_names:
        raise ValueError("EnsembleDetector cannot be nested inside itself")

    return EnsembleDetector(
        detectors=[_build_detector(name, cfg) for name in sub_names],
        voting=cfg.drift_detection.ensemble_voting,
    )
