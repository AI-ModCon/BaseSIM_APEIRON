"""What `get_performance()` reports, which is what the metrics CSV carries.

The profiler's numbers are not only read by humans. `get_performance()`
returns a flat dict and three call sites splat it straight into a logged
metric row --

    continuous_trainer.py:255   **{f"cperf_{k}": v for k, v in flops_perf.items()}
    continuous_monitor.py:376
    drift_only.py:151, cl_only.py:369

-- so every key and every value in that dict becomes a column and a cell in
the CSV at `[logging] metrics_output_path`, which is the artifact external
analysis reads. The key scheme and the value types are therefore contract,
and `test_profiler.py` asserts neither: it checks that the keys exist, not
what the values are, so a reconstruction reporting `768` where the original
reports `768.0` passes the suite and changes the CSV.

The FLOP figures are pinned as exact integers against a model with fixed
shapes, so the assertion is about the computation rather than about its
type alone: 2 * (8*16 + 16*4) = 384 per sample under the convention
torch's FlopCounterMode uses, and the count has to scale with the batch.
A reconstruction that returned a plausible constant would satisfy a
type-only assertion and fail this one. (A25 recorded that every FLOPs
figure the framework reports is unverified; the ATen-table path behind
`measure_flops_optimizer` is covered by `test_profiler.py`, and this file
covers the FlopCounterMode path it does not.)

TWO DIVERGENCES ARE DELIBERATELY LEFT UNPINNED, and both choices are
judgements rather than oversights:

1. The formatters' precision and unit thresholds. `_format_flops`,
   `_format_time` and `_format_throughput` turn a float into a display
   string; nothing in `src/` parses their output, and the only existing
   assertions are substring checks on the unit name. Two reconstructions
   differ from the original on about a hundred values here -- `'1 FLOPs'`
   against `'1.00 FLOPs'`, `'1.0000 s'` against `'1.00 s'` -- and that is
   latitude a test suite should leave alone. Pinning it would buy a worse
   suite: more criteria mass, no protection.

2. What `profiles` holds after a measured block raises. The original keeps
   the tag, with `flop` recorded as 0, so `get_performance()` reports the
   failed block as a completed measurement of zero FLOPs and the CSV gets
   a `cperf_<tag>_flop = 0.0` cell indistinguishable from a real zero. Both
   reconstructions drop the tag instead. Which is right is a question for
   whoever owns the metrics, not something to settle by pinning the
   current behaviour into a gate, so this file asserts only the part that
   is not in question: the exception propagates and `tag` is reset, so the
   next measurement is not recorded under the failed one's name.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from apeiron.profilers.count_flops import FLOPSProfiler


class Tiny(nn.Module):
    """Fixed shapes, so the FLOP count is a property of the graph alone."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture()
def tiny() -> Tiny:
    return Tiny()


def _measured(model: Tiny, batch: int = 2, tag: str = "fwd",
              times: int = 1) -> FLOPSProfiler:
    profiler = FLOPSProfiler(warmup_iters=0)
    x = torch.ones(batch, 8)
    for _ in range(times):
        with profiler.measure_flops(tag):
            model(x)
    return profiler


class TestReportedKeys:
    """The dict's shape, because its keys become CSV column names."""

    def test_three_keys_per_tag_and_nothing_else(self, tiny):
        perf = _measured(tiny).get_performance()
        assert set(perf) == {"fwd_flop", "fwd_time", "fwd_flops"}

    def test_two_tags_report_independently(self, tiny):
        profiler = FLOPSProfiler(warmup_iters=0)
        x = torch.ones(2, 8)
        for tag in ("infer", "update_fwd_bwd"):
            with profiler.measure_flops(tag):
                tiny(x)
        assert set(profiler.get_performance()) == {
            "infer_flop", "infer_time", "infer_flops",
            "update_fwd_bwd_flop", "update_fwd_bwd_time",
            "update_fwd_bwd_flops",
        }

    def test_a_fresh_profiler_reports_nothing(self):
        assert FLOPSProfiler().get_performance() == {}


class TestReportedValues:
    """The values, because they become CSV cells."""

    def test_flop_is_a_float(self, tiny):
        """`768` and `768.0` are the same number and different cells."""
        value = _measured(tiny).get_performance()["fwd_flop"]
        assert isinstance(value, float)
        assert not isinstance(value, bool)

    @pytest.mark.parametrize(
        ("batch", "expected"),
        [(1, 384.0), (2, 768.0), (4, 1536.0), (8, 3072.0)],
    )
    def test_flop_count_is_exact_and_scales_with_the_batch(
        self, tiny, batch, expected
    ):
        perf = _measured(tiny, batch=batch).get_performance()
        assert perf["fwd_flop"] == expected

    def test_throughput_is_the_count_over_the_duration(self, tiny):
        """`{tag}_flops` is a RATE and `{tag}_flop` a COUNT -- one trailing
        letter apart, which is its own hazard for anyone reading the CSV.
        The duration is not asserted (it is a clock) but the relationship
        between the three is."""
        perf = _measured(tiny).get_performance()
        assert perf["fwd_flops"] == pytest.approx(
            perf["fwd_flop"] / perf["fwd_time"]
        )

    def test_repeated_measurement_averages_rather_than_replaces(self, tiny):
        """Three identical passes report the same figure one does, so the
        reported count is a per-measurement average and not a running
        total -- the difference is a factor of three in the CSV."""
        once = _measured(tiny, times=1).get_performance()["fwd_flop"]
        thrice = _measured(tiny, times=3).get_performance()["fwd_flop"]
        assert once == 768.0
        assert thrice == once


class TestProfileStore:
    """`profiles[tag]`, which `get_performance` reduces."""

    def test_records_a_count_and_a_duration_per_measurement(self, tiny):
        rec = _measured(tiny, times=3).profiles["fwd"]
        assert set(rec) == {"flop", "time"}
        assert rec["flop"] == [768, 768, 768]
        assert len(rec["time"]) == 3


class TestFailedMeasurement:
    """Only the part that is not an open question (see the module docstring)."""

    def test_the_exception_propagates(self, tiny):
        profiler = FLOPSProfiler(warmup_iters=0)
        with pytest.raises(RuntimeError, match="planted"), \
                profiler.measure_flops("boom"):
            raise RuntimeError("planted")

    def test_the_tag_is_released(self, tiny):
        """Otherwise the next measurement lands under the failed one's
        name, and two unrelated blocks are averaged into one CSV cell."""
        profiler = FLOPSProfiler(warmup_iters=0)
        try:
            with profiler.measure_flops("boom"):
                raise RuntimeError("planted")
        except RuntimeError:
            pass
        assert profiler.tag is None

    def test_a_later_measurement_is_recorded_under_its_own_tag(self, tiny):
        profiler = FLOPSProfiler(warmup_iters=0)
        try:
            with profiler.measure_flops("boom"):
                raise RuntimeError("planted")
        except RuntimeError:
            pass
        with profiler.measure_flops("fwd"):
            tiny(torch.ones(2, 8))
        assert profiler.get_performance()["fwd_flop"] == 768.0
