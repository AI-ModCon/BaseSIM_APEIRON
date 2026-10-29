"""The ordered sequence of rows one run records -- the file's whole shape.

Every criterion in this directory that touches logging pins one property of
one row. The metrics file is a SEQUENCE, and the run claim digests each
series over its `(step, value)` pairs, where `step` is a global counter the
logger advances per row (A68). So the shape of the file couples every row
to every other: one extra, missing or reordered row shifts the step label
of everything after it, and series whose values are byte-identical fail
their digest.

That coupling made the one-property-at-a-time approach a long tail. Three
successive rebuilds each fixed the property that had been pinned and
differed somewhere new, and the count of mismatched series went 17, 16,
then back up to 22 -- not because the rebuilds got worse, but because each
was a fresh implementation free wherever nothing was asserted.

This pins the shape in one place: the exact ordered sequence of
`(stage, sorted metric names)` a short run records, under a seeded
fixture with drift forced so the adaptation path is included. Anything
that adds, drops, renames or reorders a row fails here with the index of
the first difference, rather than as a list of digests that disagree for
reasons several frames away.

Measured against `src/` and verified to repeat across runs before being
written down (R26). The configuration is small on purpose -- one stream
update, a check every four batches, Page-Hinkley at a near-zero threshold
so every check fires -- which gives 38 rows: 26 evaluations, 6 drift
checks and 6 continual-learning rows.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import TensorDataset

from apeiron.config.configuration import DriftDetectionCfg
from apeiron.driver.continuous_monitor import ContinuousMonitor

#: (stage, sorted metric names) for every row, in order. Re-derivable by
#: running the loop below; see the module docstring for the configuration.
EXPECTED_ROWS = [
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('drift', ('confidence', 'cperf_detector_flop', 'cperf_detector_flops', 'cperf_detector_time', 'cperf_infer_flop', 'cperf_infer_flops', 'cperf_infer_time', 'metric_0', 'regime', 'score')),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('drift', ('confidence', 'cperf_detector_flop', 'cperf_detector_flops', 'cperf_detector_time', 'cperf_infer_flop', 'cperf_infer_flops', 'cperf_infer_time', 'detected', 'metric_0', 'regime', 'score')),
    ('cl', ('drift_event_id', 'jvp_reg_forgetting_loss', 'jvp_reg_generation_loss', 'jvp_reg_total_loss')),
    ('cl', ('drift_event_id', 'jvp_reg_forgetting_loss', 'jvp_reg_generation_loss', 'jvp_reg_total_loss')),
    ('eval', ('drift_event_id', 'val_post_cur_accuracy', 'val_post_hist_accuracy')),
    ('eval', ('fwt', 'test_curr_acc', 'test_hist_acc', 'test_pre_cl_acc')),
    ('cl', ('cperf_detector_flop', 'cperf_detector_flops', 'cperf_detector_time', 'cperf_infer_flop', 'cperf_infer_flops', 'cperf_infer_time')),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('drift', ('confidence', 'cperf_detector_flop', 'cperf_detector_flops', 'cperf_detector_time', 'cperf_infer_flop', 'cperf_infer_flops', 'cperf_infer_time', 'metric_0', 'regime', 'score')),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('drift', ('confidence', 'cperf_detector_flop', 'cperf_detector_flops', 'cperf_detector_time', 'cperf_infer_flop', 'cperf_infer_flops', 'cperf_infer_time', 'metric_0', 'regime', 'score')),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('drift', ('confidence', 'cperf_detector_flop', 'cperf_detector_flops', 'cperf_detector_time', 'cperf_infer_flop', 'cperf_infer_flops', 'cperf_infer_time', 'detected', 'metric_0', 'regime', 'score')),
    ('cl', ('drift_event_id', 'jvp_reg_forgetting_loss', 'jvp_reg_generation_loss', 'jvp_reg_total_loss')),
    ('cl', ('drift_event_id', 'jvp_reg_forgetting_loss', 'jvp_reg_generation_loss', 'jvp_reg_total_loss')),
    ('eval', ('drift_event_id', 'val_post_cur_accuracy', 'val_post_hist_accuracy')),
    ('eval', ('bwt', 'fwt', 'test_curr_acc', 'test_hist_acc', 'test_pre_cl_acc')),
    ('cl', ('cperf_detector_flop', 'cperf_detector_flops', 'cperf_detector_time', 'cperf_infer_flop', 'cperf_infer_flops', 'cperf_infer_time')),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('eval', ('accuracy',)),
    ('drift', ('confidence', 'cperf_detector_flop', 'cperf_detector_flops', 'cperf_detector_time', 'cperf_infer_flop', 'cperf_infer_flops', 'cperf_infer_time', 'metric_0', 'regime', 'score')),
]

#: Page-Hinkley at a near-zero threshold fires on essentially every check,
#: so the continual-learning rows are part of the sequence.
FIRES = {"detector_name": "PageHinkleyDetector", "ph_threshold": 0.0001,
         "ph_min_instances": 1, "ph_delta": 0.0, "detection_interval": 4,
         "max_stream_updates": 1}


def _fixed(seed: int, n: int = 24, in_f: int = 4, nc: int = 3) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, in_f, generator=generator),
                         torch.randint(0, nc, (n,), generator=generator))


def _recorded_rows(default_cfg):
    """Every row a run logs, as (stage, sorted metric names), in order."""
    from conftest import DummyHarness

    np.random.seed(5)
    torch.manual_seed(5)
    cfg = replace(default_cfg,
                  drift_detection=replace(DriftDetectionCfg(), **FIRES))
    harness = DummyHarness(cfg, train_data=_fixed(1), val_data=_fixed(2),
                           hist_train_data=_fixed(3), hist_val_data=_fixed(4))
    with torch.no_grad():                      # pin the weights outright
        for index, parameter in enumerate(harness.model.parameters()):
            parameter.copy_(torch.full_like(parameter, 0.01 * (index + 1)))

    monitor = ContinuousMonitor(cfg=cfg, modelHarness=harness)
    rows: list[tuple[str, tuple[str, ...]]] = []
    stage = {"name": None}
    real_stage = monitor.logger.stage

    def stage_hook(name):
        stage["name"] = name
        return real_stage(name)

    monitor.logger.stage = stage_hook
    monitor.logger.log = lambda metrics, *a, **kw: rows.append(
        (stage["name"], tuple(sorted(metrics)))
    )
    monitor.run()
    return rows


class TestTheRecordedSequence:
    def test_the_row_count(self, default_cfg):
        rows = _recorded_rows(default_cfg)
        assert len(rows) == len(EXPECTED_ROWS), (
            f"the run recorded {len(rows)} rows, expected "
            f"{len(EXPECTED_ROWS)}; every extra or missing row shifts the "
            f"global step of everything after it"
        )

    def test_the_stage_sequence(self, default_cfg):
        """Order matters on its own: the same rows in a different order
        pair different values with the same step."""
        rows = _recorded_rows(default_cfg)
        assert [s for s, _n in rows] == [s for s, _n in EXPECTED_ROWS]

    def test_every_row_matches(self, default_cfg):
        rows = _recorded_rows(default_cfg)
        for index, (got, want) in enumerate(zip(rows, EXPECTED_ROWS)):
            assert got == want, (
                f"row {index} differs\n"
                f"  expected {want[0]}: {list(want[1])}\n"
                f"  recorded {got[0]}: {list(got[1])}"
            )

    def test_the_sequence_repeats(self, default_cfg):
        """The fixture is seeded, so two runs must agree -- without this the
        table above would be one draw from a distribution (R26)."""
        assert _recorded_rows(default_cfg) == _recorded_rows(default_cfg)


class TestTheSequenceIsWorthPinning:
    """Guards, so the table cannot be satisfied trivially."""

    def test_all_three_stages_appear(self):
        stages = {s for s, _n in EXPECTED_ROWS}
        assert stages == {"eval", "drift", "cl"}, (
            f"the pinned sequence only covers {sorted(stages)}; a run that "
            f"never adapts would not exercise the cl rows"
        )

    def test_the_adaptation_path_is_included(self):
        assert sum(1 for s, _n in EXPECTED_ROWS if s == "cl") > 0

    def test_the_rows_are_not_all_identical(self):
        assert len({(s, n) for s, n in EXPECTED_ROWS}) > 1
