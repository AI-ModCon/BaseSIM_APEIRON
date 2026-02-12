"""Tests for src/profilers/count_flops.py"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from profilers.count_flops import FLOPSProfiler


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
