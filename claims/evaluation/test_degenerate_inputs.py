"""What `accuracy` does with input it cannot score.

The suite covers correct answers on well-formed input. It says nothing
about malformed input, and the three cases below are the ones a caller can
actually produce: a model whose output has a sequence dimension, a loader
that yields an empty final batch, and a `topk` tuple built from a config
list that came out empty.

On all three the source RAISES and both reconstructions returned a number
instead:

    accuracy(3-D output)      source RuntimeError   rebuild [0.0]
    accuracy(empty batch)     source ZeroDivision   rebuild [NaN]
    accuracy_topk(topk=())    source ValueError     rebuild []

That direction matters more than it looks. These values do not stay local:
`eval_metrics` carries them into the logger and the metrics CSV, and
`ContinuousMonitor` buffers them and feeds the aggregate to a drift
detector. A reconstruction that answers `0.0` for a 3-D output reports 0%
accuracy for a sequence model -- which reads as catastrophic model failure
rather than as a shape mistake -- and one that answers `NaN` poisons the
mean of the metric buffer, so the detector sees NaN and the drift decision
becomes undefined. Failing loudly is the behaviour worth keeping, which is
why this file pins the exception rather than the error text.

The exception TYPES are deliberately not asserted. `ZeroDivisionError`
from an empty batch and `RuntimeError` from torch's own `t()` are both
incidental -- the source does not raise them on purpose, it simply does not
guard -- so pinning the type would freeze an accident. What is pinned is
that the call does not return a value, which is the part that is not in
question. If the team later adds explicit guards with a chosen exception
type, these cases keep passing.
"""

from __future__ import annotations

import pytest
import torch

from apeiron.evaluation.metrics import accuracy, accuracy_topk


def _well_formed():
    return torch.tensor([[0.1, 0.9], [0.8, 0.2]]), torch.tensor([1, 0])


class TestItStillScoresWellFormedInput:
    """The controls, and they have to be wide.

    The first version of this file asserted only that a 2x2 batch scores
    100%, and that was not enough. A rebuild took "degenerate input must
    raise" and over-applied it, rejecting `topk=(0,)` -- which the source
    ACCEPTS, returning 0.0. The probe then reported MORE divergences after
    the pin than before it. A criterion that says "refuse this" needs a
    matching set saying "accept that", measured from the source, or it
    licenses a stricter implementation than the one it was written from.
    """

    def test_a_perfect_batch_is_one_hundred_percent(self):
        output, target = _well_formed()
        assert float(accuracy(output, target)) == pytest.approx(100.0)

    def test_topk_returns_one_value_per_k(self):
        output, target = _well_formed()
        assert len(accuracy_topk(output, target, topk=(1, 2))) == 2

    @pytest.mark.parametrize(
        ("topk", "expected"),
        [
            ((1,), [100.0]),
            ((2,), [100.0]),
            # top-0 is accepted and scores zero: nothing can be in an empty
            # set of predictions. Vacuous, but it is the source's answer and
            # a config that computes `topk=range(0, n)` reaches it.
            ((0,), [0.0]),
            ((1, 0), [100.0, 0.0]),
            ((0, 1, 2), [0.0, 100.0, 100.0]),
        ],
    )
    def test_the_accepted_topk_values(self, topk, expected):
        output, target = _well_formed()
        got = [float(v) for v in accuracy_topk(output, target, topk=topk)]
        assert got == pytest.approx(expected)

    def test_a_single_class_batch_scores(self):
        """One column is a legitimate degenerate-but-scoreable case."""
        output = torch.tensor([[0.5], [0.5]])
        target = torch.tensor([0, 0])
        assert float(accuracy(output, target)) == pytest.approx(100.0)

    def test_a_single_sample_batch_scores(self):
        output = torch.tensor([[0.1, 0.9]])
        target = torch.tensor([1])
        assert float(accuracy(output, target)) == pytest.approx(100.0)


class TestDegenerateInputRaises:
    """It must not answer a number it cannot compute."""

    def test_a_three_dimensional_output_raises(self):
        """A sequence model's logits are (batch, time, classes). Answering
        0.0 here reports total failure for what is really a shape
        mismatch, and that number reaches the CSV and the drift detector."""
        output = torch.zeros(2, 3, 4)
        target = torch.zeros(2, dtype=torch.long)
        with pytest.raises(Exception):      # noqa: B017 — type is incidental
            accuracy(output, target)

    def test_an_empty_batch_raises(self):
        """Dividing by a batch size of zero. A NaN here propagates through
        the metric buffer's mean into the drift detector's input."""
        output = torch.zeros(0, 3)
        target = torch.zeros(0, dtype=torch.long)
        with pytest.raises(Exception):      # noqa: B017 — type is incidental
            accuracy(output, target)

    def test_an_empty_topk_raises(self):
        """`max(())` has no answer, so there is no k to score."""
        output, target = _well_formed()
        with pytest.raises(Exception):      # noqa: B017 — type is incidental
            accuracy_topk(output, target, topk=())

    def test_a_k_larger_than_the_class_count_raises(self):
        """Already the source's behaviour, and worth holding: top-5
        accuracy over two classes is not 100%, it is a configuration
        error."""
        output, target = _well_formed()
        with pytest.raises(Exception):      # noqa: B017 — type is incidental
            accuracy_topk(output, target, topk=(5,))
