# FLOPS Profiler Guide

# Overview
The FLOPS Profiler is a lightweight performance analysis tool to track operations of PyTorch training workflows 
by allowing users to tag and measure specific function calls or subroutines. Built on PyTorch's `FlopCounterMode`, 
it automatically counts floating-point operations (FLOPs) and execution time for tagged code blocks, calculating 
computational throughput (FLOP/s) for each operation. For operations not automatically captured by `FlopCounterMode` 
(such as during the optimizer step), the profiler drops back to PyTorch's native profiler to trace the function calls 
of the optimizer step and reference a lookup table for the number of FLOPs expended per parameter.

# Basic Usage

```python
from training.profilers import FLOPSProfiler

# Initialize profiler
profiler = FLOPSProfiler(warmup_iters=10)

# Use in training loop
for iter in range(num_iters):

    # Warm up is needed on GPUs
    if profiler and iter > profiler.warmup_iters:

        with profiler.measure_flops(tag="forward"):
            output = model(input)
            loss = criterion(output, target)

        with profiler.measure_flops(tag="backward"):
            loss.backward()

        with profiler.measure_flops_optimizer(tag="optimizer", model=model, device=cfg.device):
            optimizer.step()

    else:
        # Regular training without profiling
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Print results
profiler.print_performance()
```

## Example output:
```
===========================================================================
Compute Performance Metrics (Averaged per Update)
===========================================================================
Operation           FLOPs              Time            Throughput
---------------------------------------------------------------------------
backward            3.15 GFLOPs        1.51 ms         2.09 TFLOP/s
forward             1.62 GFLOPs        1.83 ms         885.94 GFLOP/s
optimizer           31.92 MFLOPs       2.87 ms         11.14 GFLOP/s
---------------------------------------------------------------------------
TOTAL               4.81 GFLOPs        6.21 ms         774.45 GFLOP/s
===========================================================================
```

