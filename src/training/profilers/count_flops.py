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

- Optimizers require FLOP estimation as Pytorch does not automatically count operations.

Current FLOP ESTIMATIONs:
- Adam Optimizer


"""

#-
import torch, time
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from contextlib import contextmanager
from typing import Dict, List, Optional, Generator, Any

#-
class FLOPSProfiler:
    """Profiler for measuring FLOPs and execution time of PyTorch operations.

    This class uses PyTorch's FlopCounterMode to automatically count FLOPs for
    most operations, and allows manual FLOP injection for operations not covered
    (like optimizer steps).

    Attributes:
        warmup_iters: Number of warmup iterations before profiling
        profiles: Dictionary storing FLOP and time measurements per tag
        manual_flops: Counter for manually injected FLOPs
    """

    def __init__(
        self,
        warmup_iters: int = 1,
    ) -> None:
        """Initialize the FLOPSProfiler.

        Args:
            warmup_iters: Number of warmup iterations (default: 1)
            reference_forward_flops: Optional reference FLOP count for validation
        """
        self.warmup_iters: int = warmup_iters
        self.tag: Optional[str] = None
        self.start_time: Optional[float] = None
        self.flop_counter: Optional[FlopCounterMode] = None
        self.profiles: Dict[str, Dict[str, List[float]]] = {}
        self.manual_flops: int = 0  # For manual FLOP injection

    @contextmanager
    def measure_flops(self, tag: str = "default") -> Generator['FLOPSProfiler', None, None]:
        """Context manager for measuring FLOPs and time for a code block.

        Args:
            tag: Identifier for this measurement session (default: "default")

        Yields:
            self: The profiler instance for manual FLOP injection

        Example:
            >>> profiler = FLOPSProfiler()
            >>> with profiler.measure_flops("forward"):
            ...     output = model(input)
            ...     profiler.add_flops(custom_op_flops)
        """
        self.tag = tag
        if self.tag not in self.profiles:
            self._add_profile(self.tag)

        self.manual_flops = 0  # Reset manual FLOPs for this measurement
        self.flop_counter = FlopCounterMode(display=False, depth=None)
        self.flop_counter.__enter__()
        self.start_time = time.perf_counter()

        try:
            yield self
        finally:
            elapsed_time = time.perf_counter() - self.start_time
            auto_flops = self.flop_counter.get_total_flops()
            self.flop_counter.__exit__(None, None, None)

            # Combine automatic and manual FLOPs
            total_flops = auto_flops + self.manual_flops

            self.profiles[self.tag]["flop"].append(total_flops)
            self.profiles[self.tag]["time"].append(elapsed_time)

            self.tag = None
            self.start_time = None
            self.flop_counter = None
            self.manual_flops = 0

    def _add_profile(self, tag: str) -> None:
        """Initialize a new profile entry for a given tag.

        Args:
            tag: Identifier for the profile entry
        """
        self.profiles[tag] = {}
        self.profiles[tag]["flop"] = []
        self.profiles[tag]["time"] = []

    def add_flops(self, flops: int) -> None:
        """Manually add FLOPs to the current measurement.

        Use this when FlopCounterMode doesn't capture certain operations.
        Must be called within a measure_flops() context.

        Args:
            flops: Number of FLOPs to add

        """
        self.manual_flops += flops

    def count_adam_step(self, params_dict: Dict[str, torch.Tensor]) -> int:
        """Estimate FLOPs for an Adam optimizer step.

        Adam performs these operations per parameter element:
        - m = b1*m + (1-b1)*g         -> 3 FLOPs (2 mul, 1 add)
        - v = b2*v + (1-b2)*(g*g)     -> 4 FLOPs (3 mul, 1 add)
        - m_hat = m / (1 - b1**t)     -> 1 FLOP (1 div)
        - v_hat = v / (1 - b2**t)     -> 1 FLOP (1 div)
        - sqrt(v_hat)                 -> 1 FLOP
        - sqrt + eps                  -> 1 FLOP (1 add)
        - m_hat / (sqrt + eps)        -> 1 FLOP (1 div)
        - lr * ...                    -> 1 FLOP (1 mul)
        - w - ...                     -> 1 FLOP (1 sub)
        Total: ~14 FLOPs per parameter

        Args:
            params_dict: Dictionary of parameters {name: tensor}

        Returns:
            Estimated number of FLOPs for the Adam step
        """
        total_params = sum(p.numel() for p in params_dict.values())
        flops = total_params * 14
        self.add_flops(flops)
        return flops

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
           flop_ = int(sum(flop_) / len(flop_))
           time_ = sum(time_) / len(time_)
           perf[f"{tag}_flop"] = flop_
           perf[f"{tag}_time"] = time_
           perf[f"{tag}_flops"] = flop_ / time_

        return perf

    def _format_flops(self, flops: float) -> str:
        """Format FLOPs in human-readable form.

        Args:
            flops: Number of FLOPs to format

        Returns:
            Formatted string with appropriate unit (TFLOPs, GFLOPs, MFLOPs, KFLOPs, or FLOPs)
        """
        if flops >= 1e12:  return f"{flops/1e12:.2f} TFLOPs"
        elif flops >= 1e9: return f"{flops/1e9:.2f} GFLOPs"
        elif flops >= 1e6: return f"{flops/1e6:.2f} MFLOPs"
        elif flops >= 1e3: return f"{flops/1e3:.2f} KFLOPs"
        else:              return f"{flops:.0f} FLOPs"

    def _format_time(self, time_sec: float) -> str:
        """Format time in human-readable form.

        Args:
            time_sec: Time in seconds

        Returns:
            Formatted string with appropriate unit (s, ms, μs, or ns)
        """
        if time_sec >= 1.0:    return f"{time_sec:.4f} s"
        elif time_sec >= 1e-3: return f"{time_sec*1e3:.2f} ms"
        elif time_sec >= 1e-6: return f"{time_sec*1e6:.2f} μs"
        else:                  return f"{time_sec*1e9:.2f} ns"

    def _format_throughput(self, flops_per_sec: float) -> str:
        """Format throughput (FLOP/s) in human-readable form.

        Args:
            flops_per_sec: Throughput in FLOPs per second

        Returns:
            Formatted string with appropriate unit (TFLOP/s, GFLOP/s, MFLOP/s, KFLOP/s, or FLOP/s)
        """
        if flops_per_sec >= 1e12:  return f"{flops_per_sec/1e12:.2f} TFLOP/s"
        elif flops_per_sec >= 1e9: return f"{flops_per_sec/1e9:.2f} GFLOP/s"
        elif flops_per_sec >= 1e6: return f"{flops_per_sec/1e6:.2f} MFLOP/s"
        elif flops_per_sec >= 1e3: return f"{flops_per_sec/1e3:.2f} KFLOP/s"
        else:                      return f"{flops_per_sec:.0f} FLOP/s"

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
        tags = sorted(set(key.rsplit('_', 1)[0] for key in perf.keys()))

        # Print header
        print("\n" + "="*75)
        print("Compute Performance Metrics (Averaged per Update)")
        print("="*75)
        print(f"{'Operation':<15} {'FLOPs':<18} {'Time':<15} {'Throughput':<20}")
        print("-"*75)

        # Track totals
        total_flops = 0
        total_time = 0

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
            print("-"*75)
            total_throughput = total_flops / total_time
            total_flop_str = self._format_flops(total_flops)
            total_time_str = self._format_time(total_time)
            total_throughput_str = self._format_throughput(total_throughput)

            print(f"{'TOTAL':<15} {total_flop_str:<18} {total_time_str:<15} {total_throughput_str:<20}")

        print("="*75 + "\n")
