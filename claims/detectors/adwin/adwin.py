"""Standard-library ADWIN2 with River 0.23.0 windowing semantics.

Buckets summarize contiguous blocks by their sum and squared deviations.
Rows hold blocks of size 2**row, ordered oldest first within each row.
See https://github.com/online-ml/river/tree/0.23.0/river/drift.
"""

from collections import deque
from math import log, sqrt


class ADWIN:
    """Detect changes in a stream's mean using an adaptive window.

    ``clock`` controls the interval between checks. ``max_buckets`` limits
    summaries per block size, and ``min_window_length`` bounds each side of
    a candidate cut. Checks begin once the window exceeds ``grace_period``.
    A detected change is exposed until the next update, which starts a fresh
    window, matching River's detector wrapper.
    """

    def __init__(self, delta=0.002, clock=32, max_buckets=5,
                 min_window_length=5, grace_period=10):
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        self._reset()

    def _reset(self):
        self._rows = [deque()]
        self._width = 0
        self._total = 0.0
        self._variance = 0.0
        self._tick = 0
        self._n_detections = 0
        self._drift_detected = False

    @property
    def drift_detected(self):
        return self._drift_detected

    @property
    def width(self):
        return self._width

    @property
    def total(self):
        return self._total

    @property
    def variance(self):
        """Sum of squared deviations in the retained window (as in River)."""
        return self._variance

    @property
    def estimation(self):
        return self._total / self._width if self._width else 0.0

    @property
    def n_detections(self):
        return self._n_detections

    def update(self, x):
        """Consume one value and update ``drift_detected``."""
        if self._drift_detected:
            self._reset()
        x = float(x)
        self._rows[0].append((x, 0.0))
        self._width += 1
        if self._width > 1:
            diff = x - self._total / (self._width - 1)
            self._variance += (self._width - 1) * diff * diff / self._width
        self._total += x
        self._compress()
        self._tick += 1
        if self._tick % self.clock == 0 and self._width > self.grace_period:
            while self._has_significant_cut():
                self._drift_detected = True
                self._delete_oldest()
        if self._drift_detected:
            self._n_detections += 1

    def _compress(self):
        row = 0
        while len(self._rows[row]) > self.max_buckets:
            if row + 1 == len(self._rows):
                self._rows.append(deque())
            total0, var0 = self._rows[row].popleft()
            total1, var1 = self._rows[row].popleft()
            size = float(1 << row)
            diff = total0 / size - total1 / size
            variance = var0 + var1 + size * size * diff * diff / (size + size)
            self._rows[row + 1].append((total0 + total1, variance))
            row += 1

    def _delete_oldest(self):
        size = float(1 << (len(self._rows) - 1))
        total, variance = self._rows[-1].popleft()
        self._width -= int(size)
        self._total -= total
        diff = total / size - self._total / self._width
        self._variance -= (
            variance + size * self._width * diff * diff / (size + self._width)
        )
        if not self._rows[-1]:
            self._rows.pop()

    def _has_significant_cut(self):
        # Traverse oldest to newest. Only bucket boundaries are candidates;
        # after a significant cut, remove one oldest bucket and scan again.
        n0, n1 = 0.0, float(self._width)
        total0, total1 = 0.0, self._total
        for row in range(len(self._rows) - 1, -1, -1):
            size = float(1 << row)
            for k, (total, _) in enumerate(self._rows[row]):
                n0 += size
                n1 -= size
                total0 += total
                total1 -= total
                if row == 0 and k == len(self._rows[row]) - 1:
                    return False
                if n0 >= self.min_window_length and n1 >= self.min_window_length:
                    delta_prime = log(2 * log(self._width) / self.delta)
                    m_recip = (1.0 / (n0 - self.min_window_length + 1)
                               + 1.0 / (n1 - self.min_window_length + 1))
                    epsilon = (
                        sqrt(2 * m_recip * (self._variance / self._width) * delta_prime)
                        + 2 / 3 * delta_prime * m_recip
                    )
                    if abs(total0 / n0 - total1 / n1) > epsilon:
                        return True
        return False
