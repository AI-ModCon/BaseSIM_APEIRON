"""Standard-library Kolmogorov-Smirnov windowing drift detector."""

from collections import deque
from functools import lru_cache
from math import comb
import random


@lru_cache(maxsize=1024)
def _ks_tail(n, m, distance):
    """Exact two-sided P(D >= distance / (n*m)) under the KS null.

    Every ordering of the two sample labels is a lattice path with equal
    probability. Count paths strictly inside the observed distance, then
    subtract their integer count from the total before dividing. This avoids
    cancellation for very small tail probabilities.
    """
    if distance == 0:
        return 1.0
    paths = [0] * (m + 1)
    for i in range(n + 1):
        for j in range(m + 1):
            if abs(i * m - j * n) >= distance:
                paths[j] = 0
            elif i == 0 and j == 0:
                paths[j] = 1
            else:
                paths[j] += paths[j - 1] if j else 0
    total = comb(n + m, n)
    return (total - paths[m]) / total


def _ks_2samp(first, second):
    """Return the empirical KS distance and its exact two-sided p-value.

    Tied observations advance both CDFs together. As in the usual exact KS
    test, the null probability uses the continuous-distribution assumption.
    """
    first, second = sorted(first), sorted(second)
    n, m = len(first), len(second)
    if not n or not m:
        raise ValueError("KS samples must be nonempty")
    i = j = distance = 0
    while i < n or j < m:
        if j == m or (i < n and first[i] <= second[j]):
            value = first[i]
        else:
            value = second[j]
        while i < n and first[i] <= value:
            i += 1
        while j < m and second[j] <= value:
            j += 1
        distance = max(distance, abs(i * m - j * n))
    return distance / (n * m), _ks_tail(n, m, distance)


class KSWIN:
    """Compare recent observations with a random sample of older observations.

    ``stat_size`` observations are sampled without replacement from the older
    part of a bounded window. Drift requires p <= alpha and KS distance > 0.1.
    A detected drift is exposed until the next update, which starts a fresh
    window and restarts the seeded random generator (River 0.23.0 semantics).
    """

    def __init__(self, alpha=0.005, window_size=100, stat_size=30,
                 seed=None, window=None):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(stat_size, int) or not 0 < stat_size <= window_size // 2:
            raise ValueError("stat_size must be positive and at most half window_size")
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self._reset()
        if window is not None:
            self.window.extend(window)

    @property
    def drift_detected(self):
        return self._drift_detected

    def _reset(self):
        self._drift_detected = False
        self.p_value = 0
        self.n = 0
        self.window = deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)

    def update(self, x):
        """Consume one observation and update ``drift_detected``."""
        if self.drift_detected:
            self._reset()
        self.n += 1
        self.window.append(x)
        if len(self.window) < self.window_size:
            self._drift_detected = False
            return
        split = self.window_size - self.stat_size
        reference = [self.window[i] for i in
                     self._rng.sample(range(split), self.stat_size)]
        recent = list(self.window)[split:]
        statistic, self.p_value = _ks_2samp(reference, recent)
        self._drift_detected = self.p_value <= self.alpha and statistic > 0.1
        if self.drift_detected:
            self.window = deque(recent, maxlen=self.window_size)
