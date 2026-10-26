"""Acceptance criteria for a reproducible APEIRON training run on MNIST.

Every other claim in this ladder pins a layer's TESTS. This one pins a
RESULT: the config in `mnist-run.toml`, driven through `main.py` over the
pinned MNIST subset, must produce the metrics trace in
`reference_trace.csv` -- exactly, on every value that is not a wall-clock
reading.

That is a much stronger acceptance boundary than a unit test, and it is the
boundary the next stage of the campaign needs: a reconstruction of APEIRON
either reproduces this training result or it does not.

WHY THIS CLAIM IS REDUCED, and in which three ways. The shipped
`examples/mnist/mnist.toml` could not be pinned as it stands:

  * `max_stream_updates = 20` -> 6. `update_data_stream()` extends the
    stream each window and every window re-evaluates the whole grown
    stream, so the cost is quadratic in window count: 18s at two windows,
    63s at four, 81s at six. Twenty windows did not finish in the time
    available. Six keeps the gate near half a minute.
  * `max_iter = 600` -> 50. The per-event CL training, cut for the same
    reason.
  * the dataset -> the first 2,000 train and 2,000 validation examples of
    real MNIST, rewritten in the same idx-ubyte format and pinned here as
    claim inputs (3MB). The full 64MB dataset is too large to pin, and
    leaving it as an external precondition would make the claim
    reproducible only on a machine that already has it at the right path.
    A subset is real data, small enough to carry, and travels.

Also pinned away from the example's defaults, and not for cost:
`device = "cpu"` and `num_workers = 0` remove the platform from the
comparison (both were measured not to matter -- A41 -- but a claim should
not depend on that holding); `backend = "none"` keeps the gate off the
network; and `reset_after_learning = true` because with it `false` the
detector fires on roughly half of all checks, and a saturated detector
makes the drift assertions below say nothing.

So: this is a reproducible APEIRON result, not a scientific one. The numbers
are what this configuration produces, and the model reaches 100% on a
2,000-image subset, which no paper would report. What the claim establishes
is that the framework can produce a training trace that is reproducible to
the last digit, and that a regrowth of it has to land on the same trace.

THE CLAIM CANNOT PASS WITHOUT THE DETERMINISM FIX. `main.py` is a generated
file here, and the trace is unreachable unless it seeds the global RNG:
three of the harness's loaders use `shuffle=True` with no generator, so
unseeded the batch order differs per process and roughly half the metrics
move (A41). The gate is therefore also the argument for that one-line
upstream change -- the fix is not asserted in prose, it is required in order
to pass.
"""

import csv
import hashlib
import pathlib
import struct
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
CONFIG = "mnist-run.toml"
REFERENCE = HERE / "reference_trace.csv"
PRODUCED = HERE / "run_metrics.csv"

#: Wall-clock readings and anything derived from them. These cannot be equal
#: between two processes and are excluded from the comparison; everything
#: else is compared exactly.
TIMED = ("_time", "_flops", "throughput")

#: The result, stated rather than only hashed. A reader should be able to see
#: what this run does without opening the reference trace.
EXPECTED_DRIFT_EVENTS = [350]
EXPECTED_FINAL_ACCURACY = 100.0
EXPECTED_EVAL_POINTS = 376
EXPECTED_PAIRS = 1665

#: The pinned subset, asserted by shape so a swapped dataset is refused
#: rather than silently producing a different trace.
DATASET = {
    "train-images-idx3-ubyte": (0x803, 2000, 28, 28),
    "train-labels-idx1-ubyte": (0x801, 2000),
    "t10k-images-idx3-ubyte": (0x803, 2000, 28, 28),
    "t10k-labels-idx1-ubyte": (0x801, 2000),
}
#: sha256 of the four files above, concatenated in sorted name order.
DATASET_DIGEST = "acbc708aaefa70fe0f06d2d7182ea6d7c201a7ae0224423b69626360c7c61c6b"


def check_dataset() -> None:
    raw = HERE / "data" / "MNIST" / "raw"
    hasher = hashlib.sha256()
    for name in sorted(DATASET):
        path = raw / name
        assert path.is_file(), f"the pinned dataset is missing {name}"
        head = path.read_bytes()[:16]
        if DATASET[name][0] == 0x803:
            got = struct.unpack(">IIII", head[:16])
        else:
            got = struct.unpack(">II", head[:8])
        assert got == DATASET[name], (
            f"{name}: header {got}, expected {DATASET[name]} -- this is not "
            f"the pinned subset, and the trace below would not apply to it")
        hasher.update(path.read_bytes())
    digest = hasher.hexdigest()
    if DATASET_DIGEST != "PLACEHOLDER":
        assert digest == DATASET_DIGEST, (
            f"dataset digest {digest}, expected {DATASET_DIGEST}")


def trace(path: pathlib.Path) -> tuple[dict, list, list]:
    rows = [r for r in csv.reader(open(path, newline="")) if len(r) == 3][1:]
    keyed = {(s, m): v for s, m, v in rows
             if not any(t in m for t in TIMED)}
    fired = [int(s) for s, m, v in rows
             if m == "drift/score" and float(v) > 0]
    accuracy = [float(v) for s, m, v in rows if m == "eval/accuracy"]
    return keyed, fired, accuracy


def run() -> None:
    PRODUCED.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "main", "--config", CONFIG,
         "--set", f"logging.metrics_output_path={PRODUCED}"],
        cwd=str(HERE), capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(HERE),
             "HOME": str(pathlib.Path.home())})
    assert proc.returncode == 0, (
        f"the training run failed ({proc.returncode}):\n"
        f"{proc.stderr.strip()[-3000:]}")
    assert PRODUCED.is_file(), "the run wrote no metrics file"


def main() -> int:
    check_dataset()
    run()

    want, want_events, want_accuracy = trace(REFERENCE)
    got, got_events, got_accuracy = trace(PRODUCED)

    # 1. the result as a readable statement
    assert got_events == EXPECTED_DRIFT_EVENTS, (
        f"drift fires at {got_events}, expected {EXPECTED_DRIFT_EVENTS}")
    assert got_accuracy, "the run recorded no accuracy"
    assert abs(got_accuracy[-1] - EXPECTED_FINAL_ACCURACY) < 1e-9, (
        f"final accuracy {got_accuracy[-1]}, expected "
        f"{EXPECTED_FINAL_ACCURACY}")
    assert len(got_accuracy) == EXPECTED_EVAL_POINTS, (
        f"{len(got_accuracy)} eval points, expected {EXPECTED_EVAL_POINTS}")

    # 2. and the whole trace, to the last digit
    assert len(got) == EXPECTED_PAIRS, (
        f"{len(got)} comparable (step, metric) pairs, expected "
        f"{EXPECTED_PAIRS} -- the shape of the run changed, not just its "
        f"numbers")
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    assert not missing, f"the run did not record {len(missing)}: {missing[:6]}"
    assert not extra, f"the run recorded {len(extra)} it should not: {extra[:6]}"

    differ = [k for k in want if want[k] != got[k]]
    assert not differ, (
        f"{len(differ)} of {len(want)} metrics differ from the pinned "
        f"trace.\nFirst few: "
        + "; ".join(f"{k[1]}@{k[0]}: {want[k]} -> {got[k]}"
                    for k in differ[:5])
        + ("\n\nIf about half the trace differs, the run is not seeded: "
           "main.py must call torch.manual_seed(cfg.seed) before the harness "
           "is built (A41)." if len(differ) > len(want) // 4 else ""))

    (HERE / "RUN_OK").write_text("ok\n")
    print(f"mnist run reproduced: {len(want)} metrics exact, "
          f"drift at {got_events}, final accuracy {got_accuracy[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
