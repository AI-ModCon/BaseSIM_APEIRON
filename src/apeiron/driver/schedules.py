"""Trigger schedules: decide fire / no-fire per decision point, ignoring metrics.

A schedule is the control-arm counterpart to a drift detector -- it fires
continual-learning updates on a fixed rule rather than on detected drift, so a
schedule run is a budget-matched baseline for judging how good a detector
actually is. Paired with :class:`~apeiron.driver.trigger_policy.SchedulePolicy`
it drops straight into the unified :class:`~apeiron.driver.stream_engine.StreamEngine`.

One decision point occurs per stream window, so ``decision_idx`` is the index of
the window that just finished streaming.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TriggerSchedule:
    """Decides fire / no-fire at each decision point, ignoring model metrics.

    One decision point occurs per stream window, so ``decision_idx`` is the
    index of the window that just finished streaming and a run has exactly
    ``drift_detection.max_stream_updates`` of them.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def should_fire(self, decision_idx: int) -> bool:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class NeverSchedule(TriggerSchedule):
    """Never adapt. Frozen-model lower bound for the accuracy-cost frontier."""

    def __init__(self) -> None:
        super().__init__("never")

    def should_fire(self, decision_idx: int) -> bool:
        return False


class PeriodicSchedule(TriggerSchedule):
    """Fire every ``period`` windows, starting at window index ``period - 1``.

    ``period = 1`` adapts after every window -- the adaptation upper bound.
    Firing is offset to the *end* of the first period rather than at index 0
    so that ``period = N`` fires exactly ``floor(windows / N)`` times.
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError(f"--period must be >= 1, got {period}")
        super().__init__("periodic")
        self.period = period

    def should_fire(self, decision_idx: int) -> bool:
        return (decision_idx + 1) % self.period == 0

    def describe(self) -> str:
        return f"periodic(every {self.period} window(s))"


class RandomSchedule(TriggerSchedule):
    """Fire at random windows -- the rate-matched null.

    Two parametrizations:

    * ``prob``: independent Bernoulli(p) per window. The realized trigger
      count varies run to run, which is the honest null if you want a
      distribution over counts as well as placements.
    * ``budget``: exactly ``budget`` windows sampled uniformly without
      replacement from ``[0, horizon)``, where ``horizon`` defaults to the
      run's window count. Preferred for budget matching, since it holds the
      count fixed and varies only the placement -- the thing being controlled
      for.
    """

    def __init__(
        self,
        seed: int,
        prob: float = 0.0,
        budget: int = 0,
        horizon: int = 0,
    ) -> None:
        super().__init__("random")
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.prob = prob
        self.budget = budget
        self.horizon = horizon
        self.fire_at: Optional[set[int]] = None

        if budget > 0:
            if horizon <= 0:
                raise ValueError("--budget requires a positive horizon")
            if budget > horizon:
                raise ValueError(f"--budget {budget} exceeds --horizon {horizon}")
            self.fire_at = set(
                int(i) for i in self.rng.choice(horizon, size=budget, replace=False)
            )
        elif prob <= 0.0:
            raise ValueError("--schedule random needs --prob or --budget")

    def should_fire(self, decision_idx: int) -> bool:
        if self.fire_at is not None:
            return decision_idx in self.fire_at
        return bool(self.rng.random() < self.prob)

    def describe(self) -> str:
        if self.fire_at is not None:
            pts = ",".join(str(i) for i in sorted(self.fire_at))
            return f"random(seed={self.seed}, exactly {self.budget} of {self.horizon}: [{pts}])"
        return f"random(seed={self.seed}, p={self.prob})"


class FixedSchedule(TriggerSchedule):
    """Fire at an explicit list of window indices.

    Drives the per-trigger attribution experiments: replay a detector's own
    firing windows with one dropped (how much was that trigger worth?) or with
    all of them shifted by a few windows (how much does timing precision
    matter?).
    """

    def __init__(self, trigger_at: list[int]) -> None:
        super().__init__("fixed")
        self.fire_at = set(trigger_at)

    def should_fire(self, decision_idx: int) -> bool:
        return decision_idx in self.fire_at

    def describe(self) -> str:
        pts = ",".join(str(i) for i in sorted(self.fire_at))
        return f"fixed([{pts}])"
