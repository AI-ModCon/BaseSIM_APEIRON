#!/usr/bin/env python3
"""Minimal falsification: an APEIRON run does not reproduce itself.

`mnist.toml` sets `seed = 1337`. Nothing under `src/` ever calls
`torch.manual_seed`, and the MNIST harness builds three of its loaders with
`shuffle=True` and no `generator=`:

    ds_train_tf,   bs,        shuffle=True, ...   # train
    ds_val_tf,     stream_bs, shuffle=True, ...   # the monitored stream
    ds_hist_train, bs,        shuffle=True, ...   # historical replay

`DataLoader(shuffle=True)` without a generator draws its permutation from the
global torch RNG. Unseeded, that permutation differs on every process, so the
batch order differs, so the gradient sequence differs, so the weights differ,
so every metric downstream differs.

What IS reproducible, and what makes this worth fixing rather than hopeless:
`examples/mnist/mnist.pth` exists, so initial weights are loaded rather than
sampled; and the affine drift transform is seeded off `cfg.seed +
task_counter`, so the drift schedule is identical run to run. Only the
shuffling is loose.

Measured on this checkout at seed 1337, device cpu, num_workers 0, over a
two-window run: of ~2,307 comparable (step, metric) pairs excluding
wall-clock fields, about 46% match, and `eval/accuracy` differs by tens of
percentage points between two runs of the same config. A handful of (step,
metric) pairs appear in one run and not the other, so even the set of
recorded keys is unstable.

The magnitude is quoted as a range because the defect is nondeterministic and
so is its size: two measurements of the same comparison gave 45.7% / 37.5
points and 45.6% / 31.25 points. The stable facts are that agreement sits
near half and that the accuracy gap is far larger than any tolerance a result
would be reported to -- not any particular figure.

Seeding the global RNG before the run makes it exact: 2,311 of 2,311
non-wall-clock pairs identical, accuracy gap 0.000. The only residue is
wall-clock -- `*_time` and the `*_flops` throughputs derived from it -- which
no fix can make identical.

One line covers the whole surface. Measured with --device and --workers:

    device  workers  seeded   agreement
    cpu     0        no       ~46%  (the defect)
    cpu     0        yes      100%
    cpu     4        yes      100%
    mps     0        yes      100%
    mps     4        yes      100%

Workers need no `generator=`: RandomSampler computes the permutation in the
parent process from the global CPU RNG, worker seeds derive from that same
generator, and the MNIST transforms carry no per-sample randomness. And
`torch.manual_seed` covers the mps backend, so no `torch.mps.manual_seed` is
required. Across cpu and mps at one seed, `eval/accuracy` and
`drift/metric_0` are identical and `eval/loss` agrees to 2e-6 -- float
arithmetic, not behaviour.

Not established: this is a short run in which no drift ever fired, so
whether a 2e-6 difference compounds over the shipped max_iter = 600 and flips
a detection at a threshold is untested; and
`torch.use_deterministic_algorithms` is never set anywhere, so a
nondeterministic kernel remains a possible third source.

    python3 claims/scripts/falsify_run_reproducibility.py \\
        --python .venv/bin/python --config examples/mnist/mnist.toml

Exits 0 if the runs differ (the defect reproduces), 1 if they match. Pass
`--seeded` to run both through a wrapper that seeds first, which should flip
the verdict and is how the proposed fix was verified.

Why this matters more than it looks: the framework exists to produce
continual-learning results. A reported BWT, a forgetting curve, an accuracy
trace -- none of them can be checked by anyone, including their author,
because rerunning the same config yields different numbers while the config
says `seed = 1337`. The seed field is an assurance the code does not keep.
"""
import argparse
import csv
import pathlib
import subprocess
import sys
import tempfile

#: Wall-clock fields and anything derived from them. These can never match
#: between two processes and are not what the comparison is about.
TIMED = ("_time", "_flops", "throughput")

SEEDING_WRAPPER = '''
"""Seed the global RNG, then hand off to APEIRON's own main.

The __main__ guard is not decoration. macOS starts DataLoader workers with
`spawn`, so every worker re-imports this module; without the guard each one
would start its own training run and the parent would die with "DataLoader
worker exited unexpectedly, details are lost due to multiprocessing". That
is a defect in this wrapper, not in APEIRON, and it is exactly how the
num_workers > 0 cell of the determinism matrix first failed.
"""
import sys


def _seed_and_run() -> int:
    import random

    import torch
    from apeiron.config.configuration import build_config

    argv = sys.argv[1:]
    cfg = build_config(argv)
    # torch.manual_seed covers every backend including mps, verified
    # separately; no torch.mps.manual_seed call is needed.
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    try:
        import numpy as np
        np.random.seed(cfg.seed)
    except ImportError:
        pass

    from src.main import main
    return main(argv)


if __name__ == "__main__":
    sys.exit(_seed_and_run())
'''


def keyed(path: pathlib.Path) -> dict:
    """{(step, metric): value}, wall-clock fields dropped."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 3:
                continue
            step, metric, value = row
            if any(t in metric for t in TIMED):
                continue
            out[(step, metric)] = value
    return out


def run(python: str, config: str, out_csv: pathlib.Path, root: pathlib.Path,
        wrapper: pathlib.Path | None, device: str = "cpu",
        workers: int = 0) -> None:
    argv = ([python, str(wrapper)] if wrapper else [python, "-m", "src.main"])
    argv += ["--config", config,
             "--set", f"logging.metrics_output_path={out_csv}",
             "--set", f"device={device}",
             "--set", f"train.num_workers={workers}",
             "--set", "logging.backend=none"]
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(root),
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(root),
                               "HOME": str(pathlib.Path.home())})
    if proc.returncode != 0:
        sys.exit(f"the run failed:\n{proc.stderr.strip()[-3000:]}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=".venv/bin/python",
                    help="interpreter with torch, torchvision and the deps")
    ap.add_argument("--config", default="examples/mnist/mnist.toml")
    ap.add_argument("--seeded", action="store_true",
                    help="seed the global RNG first -- verifies the fix")
    ap.add_argument("--root", default=".",
                    help="repository root; main.py resolves paths against it")
    ap.add_argument("--device", default="cpu",
                    help="cpu, mps or cuda. The shipped config says 'auto', "
                         "which selects mps on Apple hardware, so cpu here is "
                         "a deliberately favourable setting")
    ap.add_argument("--workers", type=int, default=0,
                    help="train.num_workers. The shipped config says 4; 0 "
                         "removes worker RNG and ordering from the picture")
    ap.add_argument("--keep", default=None,
                    help="directory to leave the two metric CSVs in, for "
                         "cross-configuration comparison")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root).resolve()

    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        wrapper = None
        if a.seeded:
            wrapper = d / "seeded_main.py"
            wrapper.write_text(SEEDING_WRAPPER)
        keep = pathlib.Path(a.keep).resolve() if a.keep else d
        keep.mkdir(parents=True, exist_ok=True)
        tag = f"{a.device}-w{a.workers}{'-seeded' if a.seeded else ''}"
        first, second = keep / f"{tag}-1.csv", keep / f"{tag}-2.csv"
        for target in (first, second):
            run(a.python, a.config, target, root, wrapper,
                device=a.device, workers=a.workers)

        A, B = keyed(first), keyed(second)

    shared = set(A) & set(B)
    if not shared:
        sys.exit("no comparable metrics were written; check the config's "
                 "metrics_output_path handling")
    same = sum(1 for k in shared if A[k] == B[k])
    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))

    def gap(metric: str) -> float:
        vals = []
        for k in shared:
            if k[1] == metric:
                try:
                    vals.append(abs(float(A[k]) - float(B[k])))
                except ValueError:
                    pass
        return max(vals, default=0.0)

    mode = "seeded" if a.seeded else "as shipped"
    print(f"two runs of {a.config} ({mode}), seed unchanged, "
          f"device {a.device}, num_workers {a.workers}\n")
    print(f"  comparable (step, metric) pairs : {len(shared)}")
    print(f"  identical                       : {same}/{len(shared)} "
          f"({100 * same / len(shared):.1f}%)")
    print(f"  pairs recorded by only one run  : {only_a} / {only_b}")
    print(f"  worst eval/accuracy gap         : {gap('eval/accuracy'):.3f} points")
    print(f"  worst eval/loss gap             : {gap('eval/loss'):.4f}")

    if same == len(shared) and not (only_a or only_b):
        print("\nthe run reproduces itself on every non-wall-clock metric")
        return 1
    print(f"\nREPRODUCED: {len(shared) - same} of {len(shared)} metrics differ "
          f"between two runs of one config.\nThe three shuffled loaders draw "
          f"from an unseeded global RNG; `seed` in the config never reaches "
          f"it.\nVerified fix: `torch.manual_seed(cfg.seed)` before the "
          f"harness is built --\nrerun with --seeded to see 100% agreement.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
