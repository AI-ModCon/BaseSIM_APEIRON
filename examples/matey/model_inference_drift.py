from __future__ import annotations

from pathlib import Path

from config.configuration import Config
from examples.matey.model import MATEYHarness
from logger import get_logger


class MATEYInferenceDriftHarness(MATEYHarness):
    """Real MATEY ViT inference with optional cross-domain data streams.

    Extends the full ViT harness so ContinuousMonitor evaluates NRMSE on real
    model forward passes. On each stream update, alternates between:

    - ``data.path`` (baseline domain — e.g. training machine/shots)
    - ``data.alt_path`` (shift domain — e.g. different machine/shots)

    When ``alt_path`` is empty, every stream uses ``path`` (same as ``matey.toml``).

    Requires ``model.pretrained_path`` for meaningful inference metrics.
    """

    def __init__(self, cfg: Config):
        self._baseline_root = self._resolve_cfg_path(cfg.data.path)
        alt_raw = str(getattr(cfg.data, "alt_path", "") or "").strip()
        self._alt_root = self._resolve_cfg_path(alt_raw) if alt_raw else None
        if self._alt_root is not None:
            self._validate_domain_root(self._alt_root, label="alt_path")

        super().__init__(cfg)

        pretrained = str(cfg.model.pretrained_path).strip()
        if not pretrained:
            get_logger().warning(
                "matey_inference_drift: model.pretrained_path is empty — "
                "ViT uses random weights; set a checkpoint for real inference.",
                level=0,
            )
        elif self._alt_root is not None:
            get_logger().info(
                "==== MATEY inference drift: domain shift enabled ====",
                level=0,
            )
            get_logger().info(f"\tBaseline domain: {self._baseline_root}", level=1)
            get_logger().info(f"\tShift domain:    {self._alt_root}", level=1)

    @staticmethod
    def _resolve_cfg_path(raw: str) -> Path:
        path = Path(raw.strip())
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @staticmethod
    def _validate_domain_root(data_root: Path, *, label: str) -> None:
        if not data_root.exists():
            raise FileNotFoundError(
                f"Matey {label} does not exist: {data_root}. "
                "Set [data].alt_path to a SOLPS root with train/ and valid/."
            )
        if not data_root.is_dir():
            raise NotADirectoryError(f"Matey {label} is not a directory: {data_root}")

    def _active_domain_label(self) -> str:
        if self._alt_root is None:
            return "baseline"
        return "baseline" if self.task_counter % 2 == 0 else "shift"

    def _active_data_root(self) -> Path:
        if self._alt_root is None or self.task_counter % 2 == 0:
            return self._baseline_root
        return self._alt_root

    def update_data_stream(self) -> None:
        domain = self._active_domain_label()
        self._data_root = self._active_data_root()
        # Force SOLPS split cache rebuild for the active domain root.
        self._solps_split = None
        self._configure_user_data_paths(self._params)
        self._configure_solps_staged_pool(self._params)

        logger = get_logger()
        logger.info(
            f"==== MATEY inference stream #{self.task_counter + 1} "
            f"({domain} domain) ====",
            level=0,
        )
        logger.info(f"\tData root: {self._data_root}", level=1)

        super().update_data_stream()
