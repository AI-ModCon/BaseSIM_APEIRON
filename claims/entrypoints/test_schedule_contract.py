"""The scheduled-CL side runner's schedules, and the CLI defaults behind them.

`cl_only.py` exists to answer an experimental question: how much of
drift-triggered adaptation's benefit is the *timing* and how much is just
the number of adaptations? Its schedules are the controls for that --
`never` as the frozen lower bound, `periodic` and `random` as rate-matched
nulls -- so WHICH windows a schedule fires on is the experiment's
independent variable. Nothing asserted it.

Both reconstructions broke it, in ways that would have produced a
plausible-looking but wrong comparison:

  * `RandomSchedule(prob=0.5)` fired on NO window. The two
    parametrizations were collapsed into one and `prob` was ignored, so
    the rate-matched null became the frozen baseline -- and the headline
    "drift-triggered beats random" would have been measured against
    nothing happening at all.
  * The same schedule produced an IDENTICAL pattern for different seeds,
    so repeated runs of the null would agree with each other and the
    variance of the control would read as zero.
  * `make_schedule` rejected `--budget` without `--period`, which is the
    budget-matching path (`cl_only.py:213`): the period is DERIVED as
    `max(1, round(horizon / budget))`. Refusing it removes the only
    schedule that holds the adaptation count fixed while varying
    placement, which is the comparison the module documents as preferred.
  * `make_schedule` rejected a non-positive `--horizon`, where the source
    falls back to `cfg.drift_detection.max_stream_updates` (`:204`). With
    the CLI's default of 0, that fallback is the normal path, not an edge
    case.

The firing patterns are pinned exactly. `RandomSchedule` draws from
`np.random.default_rng(seed)` and the claim pins numpy in its lock, so for
a given seed the pattern is a computation rather than a recording -- it can
be re-derived, and a reconstruction that samples differently cannot match
it by luck.

The CLI defaults are pinned with them because the two interact: the parser
defaults `period`, `prob`, `budget` and `horizon` to numeric zeros rather
than to `None` (`:425`-`:463`), and every branch in `make_schedule` tests
them with `>` comparisons. A rebuild that defaulted them to `None` turned
each of those tests into a TypeError.

Left free: the runners' `label` strings. `ScheduledCLRunner` and
`DriftOnlyMonitor` set `self.label` for the run announcement, A48 pins
that the attribute exists, and the exact words are display -- the rebuilds
wrote "Scheduled CL" and "Drift-only monitoring" and nothing reads them.
"""

from __future__ import annotations

import pytest

import cl_only
from cl_only import (
    FixedSchedule,
    NeverSchedule,
    PeriodicSchedule,
    RandomSchedule,
    build_parser,
    make_schedule,
)

HORIZON = 24


def pattern(schedule, n: int = HORIZON) -> str:
    """`#` where the schedule fires, `.` where it does not."""
    return "".join("#" if schedule.should_fire(i) else "." for i in range(n))


class TestParserDefaults:
    """Numeric zeros, not None -- `make_schedule` compares them with `>`."""

    def test_the_schedule_defaults_to_periodic(self):
        args = build_parser().parse_args([])
        assert args.schedule == "periodic"

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("period", 0), ("budget", 0), ("horizon", 0), ("prob", 0.0),
         ("trigger_at", "")],
    )
    def test_the_numeric_defaults_are_zero_not_none(self, name, expected):
        args = build_parser().parse_args([])
        got = getattr(args, name)
        assert got == expected
        assert got is not None

    def test_the_schedule_seed_defaults_to_none(self):
        """This one IS None, because `make_schedule` distinguishes "no seed
        given" from "seed 0" and falls back to `cfg.seed` only for the
        former."""
        args = build_parser().parse_args([])
        assert args.schedule_seed is None


class TestRandomScheduleProb:
    """Independent Bernoulli(p) per window."""

    @pytest.mark.parametrize(
        ("seed", "prob", "expected"),
        [
            (0, 0.5, ".###.......#.#.#..####.."),
            (1, 0.5, "..#.##.#.#..#.######.##."),
            (7, 0.25, "...#..#.............##.#"),
            (0, 1.0, "########################"),
        ],
    )
    def test_the_pattern_is_exact(self, seed, prob, expected):
        schedule = RandomSchedule(seed=seed, prob=prob)
        assert pattern(schedule) == expected

    def test_different_seeds_give_different_patterns(self):
        """A null whose variance is zero is not a null. One rebuild
        produced the same pattern for every seed."""
        assert pattern(RandomSchedule(seed=0, prob=0.5)) != \
            pattern(RandomSchedule(seed=1, prob=0.5))

    def test_the_same_seed_repeats(self):
        assert pattern(RandomSchedule(seed=3, prob=0.5)) == \
            pattern(RandomSchedule(seed=3, prob=0.5))

    def test_describe_names_the_probability(self):
        assert RandomSchedule(seed=1, prob=0.5).describe() == \
            "random(seed=1, p=0.5)"


class TestRandomScheduleBudget:
    """Exactly `budget` windows, sampled without replacement."""

    @pytest.mark.parametrize(
        ("seed", "budget", "expected"),
        [
            (0, 3, "............#.#...#....."),
            (1, 3, "..........##......#....."),
            (0, HORIZON, "########################"),
        ],
    )
    def test_the_pattern_is_exact(self, seed, budget, expected):
        schedule = RandomSchedule(seed=seed, budget=budget, horizon=HORIZON)
        assert pattern(schedule) == expected

    def test_exactly_the_budget_fires(self):
        """The whole point of the budget form: the count is held fixed and
        only the placement varies."""
        for seed in (0, 1, 2, 5):
            schedule = RandomSchedule(seed=seed, budget=7, horizon=HORIZON)
            assert pattern(schedule).count("#") == 7

    def test_describe_lists_the_chosen_windows(self):
        assert RandomSchedule(seed=0, budget=3, horizon=HORIZON).describe() == \
            "random(seed=0, exactly 3 of 24: [12,14,18])"


class TestRandomScheduleRefusals:
    """Parametrizations that cannot be honoured."""

    def test_neither_prob_nor_budget_raises(self):
        with pytest.raises(ValueError):
            RandomSchedule(seed=0)

    def test_a_budget_without_a_horizon_raises(self):
        with pytest.raises(ValueError):
            RandomSchedule(seed=0, budget=3, horizon=0)

    def test_a_budget_larger_than_the_horizon_raises(self):
        with pytest.raises(ValueError):
            RandomSchedule(seed=0, budget=30, horizon=HORIZON)


class TestMakeSchedule:
    """The CLI-to-schedule mapping, including both fallbacks."""

    @pytest.fixture()
    def cfg(self, default_cfg):
        return default_cfg

    def _args(self, **overrides):
        args = build_parser().parse_args([])
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_never_is_never(self, cfg):
        schedule = make_schedule(self._args(schedule="never"), cfg)
        assert isinstance(schedule, NeverSchedule)
        assert pattern(schedule) == "." * HORIZON

    def test_periodic_takes_an_explicit_period(self, cfg):
        schedule = make_schedule(self._args(schedule="periodic", period=4), cfg)
        assert isinstance(schedule, PeriodicSchedule)
        assert pattern(schedule) == "...#...#...#...#...#...#"

    @pytest.mark.parametrize(
        ("budget", "period", "expected"),
        [
            (4, 6, ".....#.....#.....#.....#"),
            (3, 8, ".......#.......#.......#"),
            (6, 4, "...#...#...#...#...#...#"),
            (24, 1, "########################"),
        ],
    )
    def test_periodic_derives_its_period_from_a_budget(
        self, cfg, budget, period, expected
    ):
        """Budget matching: `max(1, round(horizon / budget))`. A rebuild
        refused this outright, which removes the count-matched control --
        the one the module documents as preferred, because it holds the
        adaptation count fixed and varies only the placement."""
        schedule = make_schedule(
            self._args(schedule="periodic", period=0, budget=budget,
                       horizon=HORIZON), cfg)
        assert schedule.period == period
        assert pattern(schedule) == expected

    def test_periodic_needs_one_of_the_two(self, cfg):
        with pytest.raises(ValueError):
            make_schedule(self._args(schedule="periodic", period=0, budget=0),
                          cfg)

    def test_a_non_positive_horizon_falls_back_to_the_config(self, cfg):
        """The CLI defaults `--horizon` to 0, so this fallback is the normal
        path. Both rebuilds raised instead."""
        schedule = make_schedule(
            self._args(schedule="random", horizon=0, budget=3), cfg)
        assert pattern(schedule).count("#") <= 3

    def test_the_schedule_seed_overrides_the_config_seed(self, cfg):
        with_seed = make_schedule(
            self._args(schedule="random", prob=0.5, schedule_seed=7), cfg)
        assert pattern(with_seed) == pattern(RandomSchedule(seed=7, prob=0.5))

    def test_fixed_needs_its_trigger_list(self, cfg):
        with pytest.raises(ValueError):
            make_schedule(self._args(schedule="fixed", trigger_at=""), cfg)

    def test_fixed_fires_exactly_where_told(self, cfg):
        schedule = make_schedule(
            self._args(schedule="fixed", trigger_at="1,2,3"), cfg)
        assert isinstance(schedule, FixedSchedule)
        assert pattern(schedule, 6) == ".###.."

    def test_an_unknown_schedule_raises(self, cfg):
        with pytest.raises(ValueError):
            make_schedule(self._args(schedule="nope"), cfg)


class TestMainSignature:
    """Declared `-> int`, and the project type-checks (CLAUDE.md)."""

    def test_main_accepts_an_argv_list_and_is_annotated(self):
        """Both halves of the signature, because `mypy .` checks both.

        The first version of this asserted only the RETURN annotation, and
        a rebuild satisfied it with `(argv=None) -> int` -- an unannotated
        parameter, which mypy reads as implicitly `Any` and which drops the
        documentation of what `main` may be handed.
        """
        import inspect
        signature = inspect.signature(cl_only.main)
        assert "argv" in signature.parameters
        assert signature.return_annotation is not inspect.Signature.empty
        argv = signature.parameters["argv"]
        assert argv.annotation is not inspect.Parameter.empty
        assert argv.default is None
