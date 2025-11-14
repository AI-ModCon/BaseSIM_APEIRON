# FLOPS Profiler Guide

# Overview
The FLOPS Profiler is a lightweight performance analysis tool to track operations of PyTorch training workflows 
by allowing users to tag and measure specific function calls or subroutines. Built on PyTorch's `FlopCounterMode`, 
it automatically counts floating-point operations (FLOPs) and execution time for tagged code blocks, calculating 
computational throughput (FLOP/s) for each operation. For operations not automatically captured by PyTorch 
(such as optimizer steps or custom functions), the profiler provides utilities to manually estimate
and inject FLOP counts, ensuring comprehensive performance analysis across the entire training pipeline.

# Basic Usage

```python
from src.training.profilers import FLOPSProfiler

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

        with profiler.measure_flops(tag="optimizer"):
            optimizer.step()
            # Manually count optimizer FLOPs
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            profiler.count_adam_step(params)
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
Operation       FLOPs              Time            Throughput
---------------------------------------------------------------------------
backward        4.82 GFLOPs        15.23 ms        316.48 GFLOP/s
forward         2.41 GFLOPs        7.12 ms         338.48 GFLOP/s
optimizer       67.20 MFLOPs       1.85 ms         36.32 MFLOP/s
---------------------------------------------------------------------------
TOTAL           7.30 GFLOPs        24.20 ms        301.65 GFLOP/s
===========================================================================
```

