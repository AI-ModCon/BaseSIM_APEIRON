# FLOPS Profiler

```{include} ../src/apeiron/profilers/README.md
:start-line: 2
```

## Where the profiler shows up in a run

`ContinuousMonitor` and `ContinuousTrainer` are constructed with a
`FLOPSProfiler` and emit its measurements as stage-namespaced metrics, so they
land in the CSV and in your metrics backend alongside accuracy and loss:

| Metric family | Emitted during | Meaning |
| --- | --- | --- |
| `*/cperf_infer_flop`, `_time`, `_flops` | eval, drift | Forward pass over stream batches. |
| `*/cperf_detector_flop`, `_time`, `_flops` | drift | Cost of the detector `update(...)` call itself. |
| `*/cperf_update_fwd_bwd_flop`, `_time`, `_flops` | cl | Forward + backward inside the CL loop. |
| `*/cperf_optimizer_flop`, `_time`, `_flops` | cl | The optimizer step. |

See {doc}`configurations` for the full metric list written to the CSV.

```{note}
GPU measurements need a warm-up. `FLOPSProfiler(warmup_iters=N)` skips the first
`N` iterations so kernel autotuning and allocator warm-up do not distort the
timings.
```

## API

Full class and method documentation lives in {doc}`api/profilers`.
