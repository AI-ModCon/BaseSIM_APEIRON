"""Tests for src/apeiron/profilers/count_flops.py"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from apeiron.profilers.count_flops import FLOPSProfiler


# TODO: throughout this, mostly checking that the outputs are of a form that we expect. We should also verify results
class TestFLOPSProfilerInit:
    def test_default_warmup(self):
        p = FLOPSProfiler()
        assert p.warmup_iters == 1
        assert p.profiles == {}

    def test_custom_warmup(self):
        p = FLOPSProfiler(warmup_iters=5)
        assert p.warmup_iters == 5


class TestMeasureFlops:
    def test_records_flops_and_time(self):
        p = FLOPSProfiler()
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        with p.measure_flops(tag="test"):
            model(x)
        assert "test" in p.profiles
        assert len(p.profiles["test"]["flop"]) == 1
        assert len(p.profiles["test"]["time"]) == 1
        assert p.profiles["test"]["flop"][0] >= 0
        assert p.profiles["test"]["time"][0] > 0

    def test_multiple_measurements(self):
        p = FLOPSProfiler()
        model = nn.Linear(10, 5)
        for _ in range(3):
            with p.measure_flops(tag="fwd"):
                model(torch.randn(4, 10))
        assert len(p.profiles["fwd"]["flop"]) == 3

    def test_different_tags(self):
        p = FLOPSProfiler()
        model = nn.Linear(10, 5)
        with p.measure_flops(tag="a"):
            model(torch.randn(2, 10))
        with p.measure_flops(tag="b"):
            model(torch.randn(2, 10))
        assert "a" in p.profiles
        assert "b" in p.profiles


class TestGetPerformance:
    def test_empty_profiles(self):
        p = FLOPSProfiler()
        assert p.get_performance() == {}

    def test_returns_avg_metrics(self):
        p = FLOPSProfiler()
        model = nn.Linear(10, 5)
        for _ in range(3):
            with p.measure_flops(tag="test"):
                model(torch.randn(4, 10))
        perf = p.get_performance()
        assert "test_flop" in perf
        assert "test_time" in perf
        assert "test_flops" in perf  # throughput
        assert perf["test_time"] > 0
        assert perf["test_flops"] > 0


class TestFormatHelpers:
    @pytest.fixture()
    def profiler(self):
        return FLOPSProfiler()

    def test_format_flops_tera(self, profiler):
        assert "TFLOP" in profiler._format_flops(1.5e12)

    def test_format_flops_giga(self, profiler):
        assert "GFLOP" in profiler._format_flops(2.5e9)

    def test_format_flops_mega(self, profiler):
        assert "MFLOP" in profiler._format_flops(3.5e6)

    def test_format_flops_kilo(self, profiler):
        assert "KFLOP" in profiler._format_flops(1500)

    def test_format_flops_small(self, profiler):
        assert "FLOPs" in profiler._format_flops(100)

    def test_format_time_seconds(self, profiler):
        assert " s" in profiler._format_time(2.5)

    def test_format_time_milliseconds(self, profiler):
        assert "ms" in profiler._format_time(0.005)

    def test_format_time_microseconds(self, profiler):
        result = profiler._format_time(0.000005)
        assert "μs" in result

    def test_format_time_nanoseconds(self, profiler):
        assert "ns" in profiler._format_time(0.0000000005)

    def test_format_throughput_tera(self, profiler):
        assert "TFLOP/s" in profiler._format_throughput(1.5e12)

    def test_format_throughput_giga(self, profiler):
        assert "GFLOP/s" in profiler._format_throughput(2e9)

    def test_format_throughput_mega(self, profiler):
        assert "MFLOP/s" in profiler._format_throughput(5e6)

    def test_format_throughput_kilo(self, profiler):
        assert "KFLOP/s" in profiler._format_throughput(5000)

    def test_format_throughput_small(self, profiler):
        assert "FLOP/s" in profiler._format_throughput(500)


class TestMeasureFlopsOptimizer:
    def test_parameter_free_model_records_zero_flops(self):
        class NoParamModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.sum(dim=-1, keepdim=True)

        p = FLOPSProfiler()
        model = NoParamModel()
        with p.measure_flops_optimizer(tag="optimizer", model=model, device="cpu"):
            pass
        assert p.profiles["optimizer"]["flop"] == [0]
        assert p.profiles["optimizer"]["time"][0] >= 0


class TestPrintPerformance:
    def test_no_data(self, capsys):
        p = FLOPSProfiler()
        p.print_performance()
        captured = capsys.readouterr()
        assert "No performance data" in captured.out

    def test_with_data(self, capsys):
        p = FLOPSProfiler()
        model = nn.Linear(10, 5)
        with p.measure_flops(tag="forward"):
            model(torch.randn(4, 10))
        p.print_performance()
        captured = capsys.readouterr()
        assert "forward" in captured.out
        assert "TOTAL" in captured.out


# ---------------------------------------------------------------------------
# measure_flops_optimizer and the ATen-based FLOPs estimate
# ---------------------------------------------------------------------------
class TestOptimizerProfiling:
    """The optimizer path, which FlopCounterMode cannot see.

    `measure_flops_optimizer` runs a torch.profiler session and estimates
    total FLOPs as (FLOPs per parameter element) x (trainable parameters),
    mapping each observed ATen operation through ATEN_FLOPS_PER_ELEMENT.
    """

    def _model_and_optimizer(self):
        import torch.nn as nn
        from torch.optim import SGD

        torch.manual_seed(0)
        model = nn.Linear(4, 3)
        loss = model(
            torch.randn(8, 4, generator=torch.Generator().manual_seed(1))
        ).sum()
        loss.backward()
        return model, SGD(model.parameters(), lr=0.1)

    def test_records_a_measurement_for_the_tag(self):
        p = FLOPSProfiler()
        model, optimizer = self._model_and_optimizer()
        with p.measure_flops_optimizer(model=model, device="cpu", tag="optimizer"):
            optimizer.step()

        assert "optimizer" in p.profiles
        assert len(p.profiles["optimizer"]["flop"]) == 1
        assert len(p.profiles["optimizer"]["time"]) == 1
        assert p.profiles["optimizer"]["time"][0] >= 0.0
        # the estimate scales by trainable parameter count, so it cannot be
        # negative and the tag is released afterwards
        assert p.profiles["optimizer"]["flop"][0] >= 0
        assert p.tag is None
        assert p.start_time is None

    def test_repeated_measurements_accumulate(self):
        p = FLOPSProfiler()
        model, optimizer = self._model_and_optimizer()
        for _ in range(2):
            with p.measure_flops_optimizer(model=model, device="cpu"):
                optimizer.step()
        assert len(p.profiles["optimizer"]["flop"]) == 2

    def test_performance_summary_includes_the_optimizer_tag(self):
        p = FLOPSProfiler()
        model, optimizer = self._model_and_optimizer()
        with p.measure_flops_optimizer(model=model, device="cpu"):
            optimizer.step()
        performance = p.get_performance()
        assert any("optimizer" in key for key in performance)


class TestFlopsPerElementEstimate:
    def test_an_empty_profile_estimates_zero(self):
        p = FLOPSProfiler()

        class _NoEvents:
            def key_averages(self):
                return []

        assert p._estimate_flops_per_elem(_NoEvents()) == 0

    def test_known_operations_are_multiplied_by_their_call_count(self):
        from apeiron.profilers.aten_flops_map import ATEN_FLOPS_PER_ELEMENT

        operation, per_element = next(iter(ATEN_FLOPS_PER_ELEMENT.items()))

        class _Event:
            key = operation
            flops = 0
            count = 3
            cpu_time_total = 10
            self_cpu_time_total = 10
            input_shapes = [[4, 4]]

        class _Events:
            def key_averages(self):
                return [_Event()]

        p = FLOPSProfiler()
        assert p._estimate_flops_per_elem(_Events()) == per_element * 3

    def test_unknown_operations_contribute_nothing(self):
        class _Event:
            key = "aten::not_a_real_operation"
            flops = 0
            count = 7
            cpu_time_total = 10
            self_cpu_time_total = 10
            input_shapes = []

        class _Events:
            def key_averages(self):
                return [_Event()]

        assert FLOPSProfiler()._estimate_flops_per_elem(_Events()) == 0

    def test_debug_prints_the_breakdown(self, capsys):
        from apeiron.profilers.aten_flops_map import ATEN_FLOPS_PER_ELEMENT

        operation = next(iter(ATEN_FLOPS_PER_ELEMENT))

        class _Event:
            key = operation
            flops = 0
            count = 1
            cpu_time_total = 10
            self_cpu_time_total = 10
            input_shapes = [[2, 2]]

        class _Events:
            def key_averages(self):
                return [_Event()]

        FLOPSProfiler()._estimate_flops_per_elem(_Events(), debug=True)
        out = capsys.readouterr().out
        assert "All Function Calls" in out
        assert "Estimated FLOPs per param" in out
