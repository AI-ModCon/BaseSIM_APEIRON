"""Page-Hinkley change detection over a stream of scalars.

Two one-sided CUSUM accumulators run over the deviation of each sample from
the running mean, one biased down by `delta` and one up. The test statistic
is each accumulator's distance from its own running extremum -- the upward
sum's distance above its minimum, the downward sum's distance below its
maximum -- and a change is reported when either exceeds `threshold`.

The accumulators are deliberately NOT clamped at zero. Clamping is the other
common spelling of this test and it is not equivalent: it discards a negative
excursion, so a statistic that dips and then recovers is measured from zero
rather than from the bottom of the dip, and the detector fires late. The two
forms agree on any stream that rises from a floor, which is why the
difference is invisible until a dip-and-recover stream is tested.
"""

INITIAL_MIN_INCREASE = float("inf")
#: Not -inf and not 0.0. The downward statistic is measured as the distance
#: below the running maximum of a sum that starts at zero, and seeding that
#: maximum at -1 lets the first sample register a deviation of up to 1.0
#: before the maximum catches up. It is an arbitrary seed rather than a
#: derived one, and it is observable, so it is part of the behaviour.
INITIAL_MAX_DECREASE = -1.0


class PageHinkley:
    def __init__(self, min_instances=30, delta=0.005, threshold=50.0,
                 alpha=1 - 0.0001):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._reset()

    def _reset(self):
        self._n = 0
        self._mean = 0.0
        self._sum_increase = 0.0
        self._sum_decrease = 0.0
        self._min_increase = INITIAL_MIN_INCREASE
        self._max_decrease = INITIAL_MAX_DECREASE
        self.drift_detected = False

    @property
    def width(self):
        """Samples in the current window, which a detection restarts."""
        return self._n

    def update(self, value):
        if self.drift_detected:
            self._reset()

        self._n += 1
        self._mean += (value - self._mean) / self._n
        deviation = value - self._mean

        self._sum_increase = self.alpha * self._sum_increase + deviation - self.delta
        self._sum_decrease = self.alpha * self._sum_decrease + deviation + self.delta

        if self._sum_increase < self._min_increase:
            self._min_increase = self._sum_increase
        if self._sum_decrease > self._max_decrease:
            self._max_decrease = self._sum_decrease

        if self._n >= self.min_instances:
            rose = self._sum_increase - self._min_increase
            fell = self._max_decrease - self._sum_decrease
            self.drift_detected = rose > self.threshold or fell > self.threshold

        return self
