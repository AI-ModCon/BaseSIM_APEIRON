"""Tests for cl_only.py's trigger schedules and their CLI construction.

The schedules are the experimental controls for the accuracy-cost frontier —
the frozen-model lower bound, the adaptation upper bound, the rate-matched
null, and the per-trigger attribution replays — so their firing patterns and
their refusals are the contract the experiments rest on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cl_only import (  # noqa: E402
    FixedSchedule,
    NeverSchedule,
    PeriodicSchedule,
    RandomSchedule,
    TriggerSchedule,
    build_parser,
    make_schedule,
)


def _fires(schedule: TriggerSchedule, windows: int = 10) -> list[int]:
    return [i for i in range(windows) if schedule.should_fire(i)]


class TestTriggerScheduleBase:
    def test_base_should_fire_is_abstract(self):
        with pytest.raises(NotImplementedError):
            TriggerSchedule("bare").should_fire(0)

    def test_base_describe_is_the_name(self):
        assert TriggerSchedule("bare").describe() == "bare"


class TestNeverSchedule:
    def test_never_fires(self):
        assert _fires(NeverSchedule()) == []

    def test_describe(self):
        assert NeverSchedule().describe() == "never"


class TestPeriodicSchedule:
    def test_period_one_fires_every_window(self):
        assert _fires(PeriodicSchedule(1)) == list(range(10))

    def test_fires_at_the_end_of_each_period(self):
        # offset to the END of the first period, so period N fires exactly
        # floor(windows / N) times
        assert _fires(PeriodicSchedule(3), windows=10) == [2, 5, 8]
        assert len(_fires(PeriodicSchedule(4), windows=12)) == 12 // 4

    def test_period_below_one_refuses(self):
        for bad in (0, -1):
            with pytest.raises(ValueError, match="--period must be >= 1"):
                PeriodicSchedule(bad)

    def test_describe(self):
        assert PeriodicSchedule(5).describe() == "periodic(every 5 window(s))"


class TestFixedSchedule:
    def test_fires_only_at_the_listed_windows(self):
        assert _fires(FixedSchedule([1, 4, 9])) == [1, 4, 9]

    def test_duplicates_collapse_and_describe_is_sorted(self):
        s = FixedSchedule([9, 1, 4, 1])
        assert s.fire_at == {1, 4, 9}
        assert s.describe() == "fixed([1,4,9])"

    def test_empty_list_never_fires(self):
        assert _fires(FixedSchedule([])) == []


class TestRandomSchedule:
    def test_budget_fires_exactly_budget_times_within_horizon(self):
        s = RandomSchedule(seed=7, budget=3, horizon=10)
        fired = _fires(s, windows=10)
        assert len(fired) == 3
        assert all(0 <= i < 10 for i in fired)

    def test_budget_is_reproducible_from_the_seed(self):
        a = RandomSchedule(seed=7, budget=3, horizon=10)
        b = RandomSchedule(seed=7, budget=3, horizon=10)
        assert _fires(a) == _fires(b)

    def test_different_seeds_place_differently(self):
        placements = {
            tuple(_fires(RandomSchedule(seed=s, budget=3, horizon=20)))
            for s in range(8)
        }
        assert len(placements) > 1, "every seed produced the same placement"

    def test_budget_without_horizon_refuses(self):
        with pytest.raises(ValueError, match="positive horizon"):
            RandomSchedule(seed=1, budget=3, horizon=0)

    def test_budget_larger_than_horizon_refuses(self):
        with pytest.raises(ValueError, match="exceeds --horizon"):
            RandomSchedule(seed=1, budget=11, horizon=10)

    def test_neither_prob_nor_budget_refuses(self):
        with pytest.raises(ValueError, match="needs --prob or --budget"):
            RandomSchedule(seed=1)

    def test_probability_one_fires_every_window(self):
        assert _fires(RandomSchedule(seed=1, prob=1.0)) == list(range(10))

    def test_probability_is_bernoulli_per_window(self):
        # p = 0.5 over many windows lands well inside the tail bounds, and
        # the draws are consumed per call rather than precomputed
        fired = _fires(RandomSchedule(seed=3, prob=0.5), windows=400)
        assert 150 < len(fired) < 250

    def test_describe_names_the_parametrization(self):
        budgeted = RandomSchedule(seed=7, budget=2, horizon=5).describe()
        assert budgeted.startswith("random(seed=7, exactly 2 of 5: [")
        assert RandomSchedule(seed=7, prob=0.25).describe() == "random(seed=7, p=0.25)"


class TestMakeSchedule:
    def _args(self, argv: list[str]):
        args, _ = build_parser().parse_known_args(argv)
        return args

    def test_never(self, default_cfg):
        s = make_schedule(self._args(["--schedule", "never"]), default_cfg)
        assert isinstance(s, NeverSchedule)

    def test_periodic_from_period(self, default_cfg):
        s = make_schedule(
            self._args(["--schedule", "periodic", "--period", "3"]), default_cfg
        )
        assert isinstance(s, PeriodicSchedule) and s.period == 3

    def test_periodic_from_budget_divides_the_horizon(self, default_cfg):
        # horizon defaults to drift_detection.max_stream_updates (20 by
        # default), so a budget of 5 becomes a period of 4
        s = make_schedule(
            self._args(["--schedule", "periodic", "--budget", "5"]), default_cfg
        )
        assert isinstance(s, PeriodicSchedule) and s.period == 4

    def test_periodic_without_period_or_budget_refuses(self, default_cfg):
        with pytest.raises(ValueError, match="--period or --budget"):
            make_schedule(self._args(["--schedule", "periodic"]), default_cfg)

    def test_fixed_parses_the_trigger_list(self, default_cfg):
        s = make_schedule(
            self._args(["--schedule", "fixed", "--trigger-at", "7,19,29"]), default_cfg
        )
        assert isinstance(s, FixedSchedule) and s.fire_at == {7, 19, 29}

    def test_fixed_tolerates_blank_entries(self, default_cfg):
        s = make_schedule(
            self._args(["--schedule", "fixed", "--trigger-at", "1,,2,"]), default_cfg
        )
        assert s.fire_at == {1, 2}

    def test_fixed_without_trigger_at_refuses(self, default_cfg):
        with pytest.raises(ValueError, match="needs --trigger-at"):
            make_schedule(self._args(["--schedule", "fixed"]), default_cfg)

    def test_random_defaults_its_horizon_to_the_window_count(self, default_cfg):
        s = make_schedule(
            self._args(["--schedule", "random", "--budget", "4"]), default_cfg
        )
        assert isinstance(s, RandomSchedule)
        assert s.horizon == default_cfg.drift_detection.max_stream_updates

    def test_explicit_horizon_wins(self, default_cfg):
        s = make_schedule(
            self._args(["--schedule", "random", "--budget", "4", "--horizon", "9"]),
            default_cfg,
        )
        assert s.horizon == 9

    def test_schedule_seed_overrides_the_run_seed(self, default_cfg):
        # the point of --schedule-seed: vary placements while training stays
        # reproducible under cfg.seed
        plain = make_schedule(
            self._args(["--schedule", "random", "--budget", "4"]), default_cfg
        )
        assert plain.seed == default_cfg.seed
        seeded = make_schedule(
            self._args(
                ["--schedule", "random", "--budget", "4", "--schedule-seed", "123"]
            ),
            default_cfg,
        )
        assert seeded.seed == 123

    def test_unknown_schedule_refuses(self, default_cfg):
        args = self._args(["--schedule", "never"])
        args.schedule = "telepathy"  # argparse choices would reject it first
        with pytest.raises(ValueError, match="Unknown schedule"):
            make_schedule(args, default_cfg)
