"""Compute performance using PyTorch FlopCounterMode with timing.

FlopCounterMode Limitations & Coverage:
========================================

COVERED OPERATIONS (with FLOP counting):
- Linear layers (nn.Linear, F.linear): matmul + bias
- Convolution (nn.Conv1d/2d/3d): conv operations
- Matrix multiplication (torch.matmul, @, torch.mm, torch.bmm)
- Element-wise ops: add, sub, mul, div (count as FLOPs)
- Activation functions: ReLU, GELU, SiLU, etc. (counted as element-wise)
- Normalization: LayerNorm, BatchNorm (counted)
- Loss functions: CrossEntropyLoss, MSELoss (counted as element-wise + reductions)

NOT COVERED or LIMITATIONS:
- Indexing, slicing, reshape, transpose (NO FLOPs, just memory ops)
- Custom autograd Functions - need explicit FLOP annotations
- Some advanced operations - check flop_counter.py for full list
Reference: https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py

- Relying on the accuracy of torch's flop_counter for general use.

- Optimizers require FLOP estimation as flop_counter does not automatically count all non-module operations.
- For profiling the optimizer step, user must measure_flops_optimizer(tag="opt", model=model, device=cfg.device).


"""

# -
import time
import pandas as pd
from contextlib import contextmanager
from typing import Dict, List, Optional, Generator

import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.flop_counter import FlopCounterMode

from profilers.aten_flops_map import ATEN_FLOPS_PER_ELEMENT


# -
class FLOPSProfiler:
    """Profiler for measuring FLOPs and execution time of PyTorch operations.

    This class uses PyTorch's FlopCounterMode to automatically count FLOPs for
    most operations. For optimizer steps, use measure_flops_optimizer() which
    employs torch.profiler for accurate FLOP estimation.

    Attributes:
        warmup_iters: Number of warmup iterations before profiling
        profiles: Dictionary storing FLOP and time measurements per tag
        tag: Current measurement tag (set during active profiling)
        start_time: Start time of current measurement
        flop_counter: FlopCounterMode instance for current measurement
    """

    def __init__(
        self,
        warmup_iters: int = 1,
    ) -> None:
        """Initialize the FLOPSProfiler.

        Args:
            warmup_iters: Number of warmup iterations (default: 1)
        """
        self.warmup_iters: int = warmup_iters
        self.tag: Optional[str] = None
        self.start_time: Optional[float] = None
        self.flop_counter: Optional[FlopCounterMode] = None
        self.profiles: Dict[str, Dict[str, List[float]]] = {}

    @contextmanager
    def measure_flops(
        self, tag: str = "default"
    ) -> Generator["FLOPSProfiler", None, None]:
        """Context manager for measuring FLOPs and time for a code block.

        Uses FlopCounterMode to automatically count FLOPs for supported operations.
        Best for forward/backward passes of neural network modules.

        Args:
            tag: Identifier for this measurement session (default: "default")

        Yields:
            self: The profiler instance

        Example:
            >>> profiler = FLOPSProfiler()
            >>> with profiler.measure_flops("forward"):
            ...     output = model(input)
        """
        self.tag = tag
        if self.tag not in self.profiles:
            self._add_profile(self.tag)

        self.flop_counter = FlopCounterMode(display=False, depth=None)  # type: ignore[arg-type]
        self.flop_counter.__enter__()
        self.start_time = time.perf_counter()

        try:
            yield self
        finally:
            elapsed_time = time.perf_counter() - self.start_time
            total_flops = self.flop_counter.get_total_flops()
            self.flop_counter.__exit__(None, None, None)

            # Combine automatic and manual FLOPs
            self.profiles[self.tag]["flop"].append(total_flops)
            self.profiles[self.tag]["time"].append(elapsed_time)

            self.tag = None
            self.start_time = None
            self.flop_counter = None

    @contextmanager
    def measure_flops_optimizer(
        self, model: nn.Module, device: str, tag: str = "optimizer"
    ) -> Generator["FLOPSProfiler", None, None]:
        """Context manager for measuring FLOPs and time for optimizer step.

        Uses torch.profiler to estimate FLOPs for optimizer operations,
        which are not captured by FlopCounterMode. Estimates total FLOPs by
        multiplying per-element operations by the number of trainable parameters.

        Args:
            model: The model being optimized (used to count parameters)
            device: Device type ('cuda' or 'cpu')
            tag: Identifier for this measurement session (default: "optimizer")

        Yields:
            self: The profiler instance

        Example:
            >>> profiler = FLOPSProfiler()
            >>> with profiler.measure_flops_optimizer(model, "cuda", "optimizer"):
            ...     optimizer.step()
        """
        self.tag = tag
        if self.tag not in self.profiles:
            self._add_profile(self.tag)

        self.start_time = time.perf_counter()

        # Use torch profiler for optimizer operations
        with profile(
            activities=(
                [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                if device == "cuda"
                else [ProfilerActivity.CPU]
            ),
            with_flops=True,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            try:
                yield self
            finally:
                pass

        elapsed_time = time.perf_counter() - self.start_time

        # Estimate FLOPs using the profiler
        flops_per_elem = self._estimate_flops_per_elem(prof)
        total_params_require_grad = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_flops = flops_per_elem * total_params_require_grad

        self.profiles[self.tag]["flop"].append(total_flops)
        self.profiles[self.tag]["time"].append(elapsed_time)

        self.tag = None
        self.start_time = None

    def _add_profile(self, tag: str) -> None:
        """Initialize a new profile entry for a given tag.

        Args:
            tag: Identifier for the profile entry
        """
        self.profiles[tag] = {}
        self.profiles[tag]["flop"] = []
        self.profiles[tag]["time"] = []

    def _estimate_flops_per_elem(self, prof: profile, debug: bool = False) -> int:
        """Estimate FLOPs per parameter element from profiler data.

        Analyzes torch.profiler events to estimate FLOPs per parameter by mapping
        ATen operations to their FLOP counts and multiplying by call counts.

        Args:
            prof: torch.profiler.profile instance with recorded operations
            debug: If True, print detailed profiling information (default: False)

        Returns:
            Estimated FLOPs per parameter element
        """
        # Get aggregated statistics
        events = prof.key_averages()

        # Extract data into lists
        data = []
        for event in events:
            data.append(
                {
                    "operation": event.key,
                    "flops": event.flops,
                    "count": event.count,
                    "cpu_time_us": event.cpu_time_total,
                    "cpu_time_ms": event.cpu_time_total / 1000,
                    "cuda_time_us": (
                        event.cuda_time_total
                        if hasattr(event, "cuda_time_total")
                        else 0
                    ),
                    "self_cpu_time_us": event.self_cpu_time_total,
                    "input_shapes": (
                        str(event.input_shapes) if event.input_shapes else ""
                    ),
                }
            )

        # Create DataFrame
        df = pd.DataFrame(data)

        # Multiply FLOPs per element by the number of times each operation was called
        df["est_flops_per_param"] = df.apply(
            lambda row: ATEN_FLOPS_PER_ELEMENT.get(row.operation, 0) * row["count"],
            axis=1,
        )

        # - Profiler Probe
        # - May need to look here later to see if there are any missing computations in the lookup.
        if debug:
            df = df.sort_values("self_cpu_time_us", ascending=False).reset_index(
                drop=True
            )
            print("All Function Calls:\n", df)
            print("Compute Function Calls:\n", df[df.est_flops_per_param > 0])
            print("Estimated FLOPs per param:", int(df.est_flops_per_param.sum()))

        # -
        return int(df.est_flops_per_param.sum())

    def get_performance(self) -> Dict[str, float]:
        """Calculate average performance metrics across all measurements.

        Computes the mean FLOPs, time, and throughput (FLOP/s) for each tag.

        Returns:
            Dictionary containing averaged metrics with keys:
                - {tag}_flop: Average FLOPs per operation
                - {tag}_time: Average time per operation (seconds)
                - {tag}_flops: Average throughput (FLOP/s)
        """
        perf = {}
        for tag in self.profiles:
            flop_ = self.profiles[tag]["flop"]
            time_ = self.profiles[tag]["time"]
            avg_flop_ = sum(flop_) / len(flop_)
            avg_time_ = sum(time_) / len(time_)
            perf[f"{tag}_flop"] = avg_flop_
            perf[f"{tag}_time"] = avg_time_
            perf[f"{tag}_flops"] = avg_flop_ / avg_time_

        return perf

    def _format_flops(self, flops: float) -> str:
        """Format FLOPs in human-readable form.

        Args:
            flops: Number of FLOPs to format

        Returns:
            Formatted string with appropriate unit (TFLOPs, GFLOPs, MFLOPs, KFLOPs, or FLOPs)
        """
        if flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops / 1e3:.2f} KFLOPs"
        else:
            return f"{flops:.0f} FLOPs"

    def _format_time(self, time_sec: float) -> str:
        """Format time in human-readable form.

        Args:
            time_sec: Time in seconds

        Returns:
            Formatted string with appropriate unit (s, ms, μs, or ns)
        """
        if time_sec >= 1.0:
            return f"{time_sec:.4f} s"
        elif time_sec >= 1e-3:
            return f"{time_sec * 1e3:.2f} ms"
        elif time_sec >= 1e-6:
            return f"{time_sec * 1e6:.2f} μs"
        else:
            return f"{time_sec * 1e9:.2f} ns"

    def _format_throughput(self, flops_per_sec: float) -> str:
        """Format throughput (FLOP/s) in human-readable form.

        Args:
            flops_per_sec: Throughput in FLOPs per second

        Returns:
            Formatted string with appropriate unit (TFLOP/s, GFLOP/s, MFLOP/s, KFLOP/s, or FLOP/s)
        """
        if flops_per_sec >= 1e12:
            return f"{flops_per_sec / 1e12:.2f} TFLOP/s"
        elif flops_per_sec >= 1e9:
            return f"{flops_per_sec / 1e9:.2f} GFLOP/s"
        elif flops_per_sec >= 1e6:
            return f"{flops_per_sec / 1e6:.2f} MFLOP/s"
        elif flops_per_sec >= 1e3:
            return f"{flops_per_sec / 1e3:.2f} KFLOP/s"
        else:
            return f"{flops_per_sec:.0f} FLOP/s"

    def print_performance(self) -> None:
        """Pretty print the performance metrics (averaged per update).

        Displays a formatted table showing FLOPs, time, and throughput for each
        profiled operation tag, along with totals.
        """
        perf = self.get_performance()

        if not perf:
            print("No performance data collected yet.")
            return

        # Extract unique tags
        tags = sorted(set(key.rsplit("_", 1)[0] for key in perf.keys()))

        # Print header
        print("\n" + "=" * 75)
        print("Compute Performance Metrics (Averaged per Update)")
        print("=" * 75)
        print(f"{'Operation':<15} {'FLOPs':<18} {'Time':<15} {'Throughput':<20}")
        print("-" * 75)

        # Track totals
        total_flops: float = 0
        total_time: float = 0

        # Print each tag's metrics
        for tag in tags:
            flop_key = f"{tag}_flop"
            time_key = f"{tag}_time"
            flops_key = f"{tag}_flops"

            if flop_key in perf and time_key in perf and flops_key in perf:
                flop_str = self._format_flops(perf[flop_key])
                time_str = self._format_time(perf[time_key])
                throughput_str = self._format_throughput(perf[flops_key])

                print(f"{tag:<15} {flop_str:<18} {time_str:<15} {throughput_str:<20}")

                total_flops += perf[flop_key]
                total_time += perf[time_key]

        # Print total row
        if total_flops > 0 and total_time > 0:
            print("-" * 75)
            total_throughput = total_flops / total_time
            total_flop_str = self._format_flops(total_flops)
            total_time_str = self._format_time(total_time)
            total_throughput_str = self._format_throughput(total_throughput)

            print(
                f"{'TOTAL':<15} {total_flop_str:<18} {total_time_str:<15} {total_throughput_str:<20}"
            )

        print("=" * 75 + "\n")
