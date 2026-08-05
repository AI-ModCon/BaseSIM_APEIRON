"""Harness that streams a *sequence* of SOLPS simulations, for Figure 2.

``MATEYInferenceDriftHarness`` alternates between exactly two roots, which is
enough to make a single change point but cannot express "simulations keep
arriving, and part way through they start coming from a different machine".
That is what Figure 2 needs, so this harness walks an ordered list of staged
arrivals instead.

The order and each arrival's metadata come from ``stream_manifest.json``, read
from ``data.path``; see ``examples/matey/README.md`` for its schema and for how
to build a stream root from SOLPS output. Keeping it in the data
root rather than in the config means this needs no new keys in
``apeiron.config.configuration`` -- the framework config is shared with other
users of APEIRON and should not grow a SOLPS-specific vocabulary. The dataset
type and field labels sit beside it in ``matey_settings.json``, for the same
reason (see ``examples/matey/solps/settings.py``).

What each stream update does:

* point the loaders at the next arrival's bundle;
* remember the most recent arrival of the stream's *first case*, and expose it
  through ``get_hist_dataloaders()``. That is what makes forgetting measurable:
  after continual learning adapts to newly arrived data,
  ``BaseModelHarness.history_eval()`` reports error back on the original case.

Monitoring stops when the arrivals are exhausted. ``drift_detection.
max_stream_updates`` must be ``n_arrivals - 1``: ContinuousMonitor calls
``update_data_stream()`` once before its loop and once per extension.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apeiron.config.configuration import Config
from apeiron.logger import get_logger
from examples.matey.model import MATEYHarness

MANIFEST_NAME = "stream_manifest.json"


class MATEYStreamHarness(MATEYHarness):
    """Walk an ordered sequence of staged SOLPS arrivals."""

    def __init__(self, cfg: Config):
        self._stream_root = Path(str(cfg.data.path).strip()).resolve()
        self._manifest = self._load_manifest(self._stream_root)
        self._arrivals: list[dict] = self._manifest["arrivals"]
        if not self._arrivals:
            raise ValueError(f"{MANIFEST_NAME} lists no arrivals: {self._stream_root}")

        # "Historical" is keyed on the starting *case*, not the starting
        # machine. Tying it to the machine made get_hist_dataloaders() return
        # (None, None) throughout the held-out DIII-D scenario -- the same
        # machine as the baseline -- so forgetting went unmeasured for exactly
        # the drift events that matter. The baseline is the data the surrogate
        # was doing well on, which is the first case.
        self._baseline_case = self._arrivals[0].get("case")
        self._first_machine = self._arrivals[0].get("machine")
        # Loaders from the most recent baseline-case arrival, so adaptation to
        # any later case can be scored against the original one.
        self._hist_loaders: tuple[Any, Any] | None = None
        self._current_arrival: dict = self._arrivals[0]

        super().__init__(cfg)

        logger = get_logger()
        logger.info("==== MATEY sequential simulation stream ====", level=0)
        logger.info(f"\tRoot:      {self._stream_root}", level=1)
        logger.info(f"\tArrivals:  {len(self._arrivals)}", level=1)
        logger.info(f"\tMachines:  {' -> '.join(self._machine_run_lengths())}", level=1)
        logger.info(
            f"\tMachine change points (arrival index): "
            f"{self._manifest.get('machine_change_points')}",
            level=1,
        )
        if not str(cfg.model.pretrained_path).strip():
            logger.warning(
                "matey_stream: model.pretrained_path is empty -- the ViT has "
                "random weights, so the error ladder is meaningless."
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_manifest(root: Path) -> dict:
        path = root / MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"No {MANIFEST_NAME} at {root}. See examples/matey/README.md "
                f"for the stream-root layout and how to build one."
            )
        with path.open() as fh:
            return json.load(fh)

    def _machine_run_lengths(self) -> list[str]:
        """Compact 'DIII-D x16, KSTAR x8' style summary for the run log."""
        out: list[str] = []
        for arrival in self._arrivals:
            machine = str(arrival.get("machine"))
            if out and out[-1].startswith(machine + " x"):
                count = int(out[-1].split(" x")[1]) + 1
                out[-1] = f"{machine} x{count}"
            else:
                out.append(f"{machine} x1")
        return out

    @property
    def n_arrivals(self) -> int:
        return len(self._arrivals)

    def current_arrival(self) -> dict:
        """Metadata for the arrival currently being streamed."""
        return dict(self._current_arrival)

    # ------------------------------------------------------------------
    def update_data_stream(self) -> None:
        idx = self.task_counter
        if idx >= len(self._arrivals):
            # Deliberately NOT StopIteration. ContinuousMonitor calls
            # _extend_stream() from inside its `except StopIteration` handler, so
            # a StopIteration raised here escapes start() entirely -- the run
            # dies without its closing log lines instead of stopping cleanly.
            # Note the monitor calls update_data_stream() once before the loop
            # and once per extension, so it consumes max_stream_updates + 1
            # arrivals: set max_stream_updates to n_arrivals - 1.
            raise RuntimeError(
                f"Simulation stream exhausted: {len(self._arrivals)} arrivals "
                f"staged, arrival index {idx} requested. Set "
                f"drift_detection.max_stream_updates to "
                f"{len(self._arrivals) - 1} (n_arrivals - 1), or stage more."
            )

        arrival = self._arrivals[idx]
        self._current_arrival = arrival
        self._data_root = self._stream_root / arrival["dir"]
        # _configure_user_data_paths() returns quietly when neither train/ nor
        # valid/ is present, to stay usable with non-SOLPS fixtures. In stream
        # mode that silence is dangerous: the loaders would keep pointing at the
        # PREVIOUS arrival while the log announces this one, so a missing bundle
        # reads as a flat regime rather than an error -- and those stale loaders
        # would then be latched as the forgetting baseline.
        if not any((self._data_root / sub).is_dir() for sub in ("train", "valid")):
            raise FileNotFoundError(
                f"Arrival {idx} ({arrival['dir']!r}) has no train/ or valid/ "
                f"directory under {self._data_root}. Re-stage the stream; "
                f"continuing would silently re-serve the previous arrival."
            )
        # Force the SOLPS split cache to rebuild against the new root.
        self._solps_split = None
        self._configure_user_data_paths(self._params, self.cfg)
        if not self._settings.use_step_inference:
            self._configure_solps_staged_pool(self._params, self.cfg)

        machine = arrival.get("machine")
        is_change = bool(idx > 0 and machine != self._arrivals[idx - 1].get("machine"))

        logger = get_logger()
        logger.info(
            f"==== arrival {idx + 1}/{len(self._arrivals)}: "
            f"{arrival.get('case')} seg {arrival.get('segment')} "
            f"[{machine}]{'  <-- MACHINE CHANGE' if is_change else ''} ====",
            level=0,
        )
        logger.info(
            f"\tframes {arrival.get('time_range')}  "
            f"train {arrival.get('train_range')}  valid {arrival.get('valid_range')}",
            level=1,
        )

        is_baseline = arrival.get("case") == self._baseline_case
        self._current_stream_domain = "baseline" if is_baseline else "shift"
        super().update_data_stream()

        if is_baseline:
            self._hist_loaders = (self._cur_train_loader, self._cur_val_loader)

    def get_hist_dataloaders(self):
        """Loaders from the starting case, for the forgetting axis.

        ``(None, None)`` while the stream is still on the starting case --
        historical and current data would be the same set, so the comparison
        would be vacuous.
        """
        if self._hist_loaders is None:
            return None, None
        if self._current_arrival.get("case") == self._baseline_case:
            return None, None
        return self._hist_loaders
