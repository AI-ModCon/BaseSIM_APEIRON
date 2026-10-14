"""Slicing and joining of MATEY's structured batches, which replay needs.

``BaseUpdater`` builds a replay step by taking half of the current batch and
half of a historical one. On a Tensor that is slicing and ``torch.cat``; on
these dataclasses it needs ``len()``, ``[]``, and a probe saying whether the two
can be a single forward pass at all. They frequently cannot: arrivals from
different machines sit on different spatial grids, so the two halves have to be
run as separate weighted sub-batches instead.

No MATEY install is needed -- ``matey_batches`` imports only torch and the
standard library at module scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # repo root, so `examples.matey` imports
    sys.path.append(str(_ROOT))

from examples.matey.solps.matey_batches import (  # noqa: E402
    MateyInputBatch,
    MateyTargetBatch,
)


def _batch(n: int = 4, width: int = 98, tkhead: str = "tk-2D") -> MateyInputBatch:
    return MateyInputBatch(
        input=torch.randn(n, 1, 3, 1, 38, width),
        field_labels=torch.arange(n * 3).reshape(n, 3),
        bcs=torch.zeros(n, 2),
        leadtime=torch.ones(n, 1),
        cond_input=torch.randn(n, 2),
        tkhead_name=tkhead,
        blockdict={"Ind_dim": torch.tensor([1, 38, width])},
    )


class TestLength:
    def test_counts_samples(self):
        assert len(_batch(6)) == 6

    def test_graph_batch_has_no_sample_axis(self):
        graph = MateyInputBatch(graph=object(), is_graph=True)
        with pytest.raises(TypeError, match="no sample axis"):
            len(graph)


class TestSlicing:
    def test_per_sample_fields_are_sliced_together(self):
        b = _batch(8)
        half = b[:3]
        assert len(half) == 3
        for name in ("input", "field_labels", "bcs", "leadtime", "cond_input"):
            assert getattr(half, name).shape[0] == 3
            assert torch.equal(getattr(half, name), getattr(b, name)[:3])

    def test_global_leadtime_is_passed_through(self):
        """The adapter's default leadtime is [1, 1] regardless of batch size;
        slicing it to the sample count would corrupt it."""
        b = MateyInputBatch(input=torch.randn(4, 2), leadtime=torch.ones(1, 1))
        assert torch.equal(b[:2].leadtime, b.leadtime)

    def test_batch_level_metadata_survives(self):
        b = _batch(8)
        assert b[:3].tkhead_name == b.tkhead_name
        assert b[:3].blockdict is b.blockdict

    def test_target_batch_slices(self):
        t = MateyTargetBatch(target=torch.randn(8, 3, 38, 98))
        assert len(t[:5]) == 5 and t[:5].shape[0] == 5


class TestCanCatWith:
    def test_same_geometry_joins(self):
        assert _batch(4).can_cat_with(_batch(2))

    def test_different_grid_refuses(self):
        """The cross-machine case: two devices whose SOLPS grids differ."""
        assert not _batch(4, width=98).can_cat_with(_batch(2, width=170))

    def test_different_tokenizer_head_refuses(self):
        assert not _batch(4).can_cat_with(_batch(2, tkhead="tk-3D"))

    def test_graph_batch_refuses(self):
        graph = MateyInputBatch(graph=object(), is_graph=True)
        assert not _batch(4).can_cat_with(graph)

    def test_tensor_blockdict_does_not_raise(self):
        """Ind_dim holds tensors, and `==` on those returns a tensor whose
        truth value is ambiguous -- so the key must be plain ints."""
        assert isinstance(_batch(4)._geometry_key()[1], tuple)


class TestCat:
    def test_matches_fieldwise_concatenation(self):
        a, b = _batch(4), _batch(2)
        joined = MateyInputBatch.cat(a, b)
        assert len(joined) == 6
        for name in ("input", "field_labels", "bcs", "cond_input"):
            expected = torch.cat([getattr(a, name), getattr(b, name)], dim=0)
            assert torch.equal(getattr(joined, name), expected)

    def test_targets_concatenate(self):
        a = MateyTargetBatch(target=torch.randn(4, 3))
        b = MateyTargetBatch(target=torch.randn(2, 3))
        assert len(MateyTargetBatch.cat(a, b)) == 6
