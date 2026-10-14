"""Parsing helpers behind the retrospective evaluation.

These carry the failure modes that would be invisible in the output: an
off-by-one between the run log's 1-based arrival banner and the 0-based index
used everywhere else would attribute every drift event to the wrong arrival, and
the resulting figure would look entirely reasonable.
"""

from __future__ import annotations

import sys
from pathlib import Path

from examples.matey.eval_retrospective import (
    event_to_arrival,
    find_checkpoints,
    parse_arrivals,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))


class TestParseArrivals:
    def test_all(self):
        assert parse_arrivals("all", 4) == [0, 1, 2, 3]

    def test_range(self):
        assert parse_arrivals("0-3", 32) == [0, 1, 2, 3]

    def test_list(self):
        assert parse_arrivals("0,4,8", 32) == [0, 4, 8]

    def test_out_of_range_is_dropped(self):
        assert parse_arrivals("0,99", 4) == [0]


class TestEventToArrival:
    LOG = """
==== arrival 1/12: baseline_d3d seg 0 [DIII-D] ====
some noise
==== arrival 2/12: baseline_d3d seg 1 [DIII-D] ====
==== DRIFT DETECTED (Event #1)! ====
==== arrival 3/12: ood_d3d seg 0 [DIII-D] ====
==== DRIFT DETECTED (Event #2)! ====
==== DRIFT DETECTED (Event #3)! ====
"""

    def test_events_map_to_the_arrival_above_them(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text(self.LOG)
        # The banner is 1-based; every other index in the pipeline is 0-based.
        assert event_to_arrival(str(log)) == {1: 1, 2: 2, 3: 2}

    def test_missing_log_is_not_fatal(self, tmp_path):
        assert event_to_arrival(str(tmp_path / "nope.log")) == {}

    def test_no_log_requested(self):
        assert event_to_arrival("") == {}


class TestFindCheckpoints:
    def test_sorted_numerically_not_lexically(self, tmp_path):
        for n in (1, 2, 10, 11):
            (tmp_path / f"drift_adaptation_{n}.pt").write_text("x")
        (tmp_path / "latest").write_text("drift_adaptation_11.pt")
        assert [e for e, _ in find_checkpoints(tmp_path)] == [1, 2, 10, 11]

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "best_ckpt.tar").write_text("x")
        assert find_checkpoints(tmp_path) == []
