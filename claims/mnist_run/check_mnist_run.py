"""Acceptance criteria for a reproducible APEIRON training run on MNIST.

Every other claim in this ladder pins a layer's TESTS. This one pins a
RESULT: the configuration in `mnist-run.toml`, driven through `main.py` over
the pinned MNIST subset, must produce a particular training trace and a
particular set of trained weights.

HOW THIS CLAIM WAS BROKEN, AND WHY IT NOW LOOKS LIKE THIS. The first version
pinned `reference_trace.csv` as an input and compared the run's output
against it. Both files sat in the room, so the shortest path to passing was
to copy the answer -- and a blind reconstruction took it, writing a
45-line `RecordedMNISTRun` that read the reference and wrote it to the
output path, importing no torch at all. It reached the source root. It was
also scrupulously honest about itself: "A recorded run, not a reconstructed
neural network or training harness ... it makes no claim to perform
inference or optimization." The producer behaved correctly; the criteria
were circular (claims/LOG.md A43).

Two changes close it.

  1. THE EXPECTED TRACE IS NOT IN THE ROOM. What is pinned is a sha256 per
     metric over that metric's (step, value) sequence. A digest cannot be
     inverted into a trace, so matching it requires computing the trace.
     Per-metric rather than one digest over everything, so a failure still
     says WHICH metric moved without revealing what it should have been.

  2. THE TRAINED WEIGHTS ARE PINNED. The run writes a checkpoint on its
     drift event; this check loads it and digests the parameter values. The
     room contains `examples/mnist/mnist.pth`, the INITIAL weights, and
     nothing else from which trained weights could be derived. Copying is
     not available: you cannot produce the state of a trained model without
     training it.

Generating and checking share one code path -- `python3 check_mnist_run.py
--emit` prints the constants below -- so the pinned values cannot drift from
how they are verified.

WHY THIS CLAIM IS REDUCED, in three ways. The shipped
`examples/mnist/mnist.toml` could not be pinned as it stands:

  * `max_stream_updates = 20` -> 6. `update_data_stream()` extends the
    stream each window and every window re-evaluates the whole grown
    stream, so cost is quadratic in window count: 18s at two windows, 63s
    at four, 81s at six. Twenty did not finish in the time available.
  * `max_iter = 600` -> 50. The per-event CL training, cut for the same
    reason.
  * the dataset -> the first 2,000 train and 2,000 validation examples of
    real MNIST, rewritten in the same idx-ubyte format and pinned here
    (3MB). The full 64MB cannot be pinned, and an external dataset would
    make the claim reproducible only where that dataset already sits.

Also pinned away from the example's defaults, and not for cost: `device =
"cpu"` and `num_workers = 0` keep the platform out of the comparison;
`backend = "none"` keeps the gate off the network; `reset_after_learning =
true` because with it `false` the detector fires on roughly half of all
checks and a saturated detector makes the drift assertion say nothing; and
`max_ckpts`/`ckpts_path` are set so the run writes the weights this check
needs.

So this is a reproducible result, not a scientific one. The model reaches
100% on 2,000 images, which no paper would report. What the claim
establishes is that the framework CAN produce a training trace reproducible
to the last digit, and that a regrowth has to actually train to land on it.

THE CLAIM CANNOT PASS UNSEEDED. `main.py` is generated here, and two
unseeded RNGs decide what the run does: torch (three loaders use
shuffle=True with no generator, so batch order varies and about half the
metrics move) and numpy (the monitor samples `detected=0` rows at 10%, so
the trace's shape varies). Stripping both seed calls fails this gate. The
one-line upstream fix A41 asks for is not argued for in prose here; it is
required in order to pass.
"""

import csv
import hashlib
import json
import pathlib
import struct
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
CONFIG = "mnist-run.toml"
PRODUCED = HERE / "run_metrics.csv"
CKPT_DIR = HERE / "ckpts"

#: Wall-clock readings and anything derived from them cannot be equal between
#: two processes and are excluded. Everything else is compared by digest.
TIMED = ("_time", "_flops", "throughput")

#: sha256 per metric over its (step, value) sequence. Not invertible, so
#: matching these requires computing the trace rather than copying it.
TRACE_DIGESTS = {
    "eval/accuracy": "7ad11fcfe683bcdb931622820c3171ff51c9037610d0e4473db2bca77aeeed30",
    "eval/loss": "f89ac5fcff560444a9e6016f7df465582545ad77c437210041f82abfe588992f",
    "eval/step": "0ccde54bc62b86d87557513d4eac2484e564ea19c870ad3f99642031ac26d0e7",
    "drift/score": "95f6d550bf0803530e5a637480d9f9f035ff2c46b63d0c7bfb3621956f28ef0d",
    "drift/regime": "941f191ecf704020c8122ca944b1441fb85a720903970108702f9deaedfb44b1",
    "drift/confidence": "fcdf00ef4eddae9fcdbf2891256df85b1d6f1a7186ef5d78023a60b8ce0f4c35",
    "drift/metric_0": "cf19f3fe610a2dc29bc8dfd6d73c76d824a5d6676990ab2014b4b5ce85c0a35e",
    "drift/cperf_infer_flop": "a5054c62a1c35d6c7a1c1241e52ec71ef922007d7a53830f7f076a4b836d1d9d",
    "drift/cperf_detector_flop": "cd7329a424eafa05f5df09edcf94c13f5de0321780ff677ee6b8323db9307f5e",
    "drift/step": "4f6ff4c6cedd0f02ad0a9e107a3fe4a60cb85e4383857a4301e06509c1a53ad3",
    "drift/detected": "3df0ea056c6043612b050dfb2c1e569669e0e5243f682bf294ede24d1e9009d2",
    "cl/jvp_reg_total_loss": "6d0b41030cfb1bd3af08cf3141f9a524d86114b09e9ff03e11e50aa297cc8624",
    "cl/jvp_reg_forgetting_loss": "6fb09c6afd53efbcf080e0765cb9283dd6e83671b12d7d839cac0f7e81a05100",
    "cl/jvp_reg_generation_loss": "6d0b41030cfb1bd3af08cf3141f9a524d86114b09e9ff03e11e50aa297cc8624",
    "cl/drift_event_id": "40480c678dc9016f1363d56eb1ac0a56342402835700b4f13cbe146794483554",
    "cl/step": "cc779eb7ddf8db635778a99eaa0a4eb50447fdf1b2c2d5c588b7beaf2a480ee1",
    "eval/drift_event_id": "75a37546f0f7237d772342c0b898566282e6e0cbdf73358c37b6b6350811f3d0",
    "eval/val_post_cur_accuracy": "99348991beb0f1b16d9d4a05105d22f662e96cfd650770ef0c92c6966f12c454",
    "eval/val_post_cur_loss": "ff1ca8801762c4825aea52f1943b3de634cec86fdf5af428f2e1195acf8c092f",
    "eval/val_post_hist_accuracy": "b7005c7520e1d57bd951c4c154b39575cecd5e56e234d06a8b8034e622c11201",
    "eval/val_post_hist_loss": "edf48b8abd0b33d987c66f02930391c0ad390f12fbf8f50cce2a4ae9d5e77868",
    "eval/test_curr_acc": "88a21b2136a854d859c1477438dc0abfd9256ad3141cf2b4c1a7e7f1daa1b6fa",
    "eval/test_pre_cl_acc": "cca30caa4969ac25cba82f6dbcb7ef55b4d553e3641be4a30ad049e58a5f7532",
    "eval/fwt": "ed9937ccb69e925d5e0ef3f4e6b226dfe32d28dc7caa8d7e5269298a4a688d6f",
    "eval/test_hist_acc": "0b6e5c9fe274b3554af708e9803456cc922468300830e0eabe4955dff0b9725e",
    "cl/cperf_infer_flop": "811c5d0b872b802d58ab3f09e36a9f953cec02e93ae17280c9da6bf0e29897ce",
    "cl/cperf_detector_flop": "45787f5337e197061563e15797d39f0317af8741001db169becf60d14495c2e7",
    "cl/cperf_update_fwd_bwd_flop": "f262805ab2867f24d93e6d0cae405a541bd84386ec575baad9ee937c1ac7dc64",
    "cl/cperf_optimizer_flop": "09448f7e1fbb2ae6427e6654511b4ac5f37ecd2efaf97c7ce33b745b888f6d6d",
    "drift/cperf_update_fwd_bwd_flop": "bcdd63f14c792101ad484f1cd57a6e131f5d7ebbd41ef99bf7b61732d740b44f",
    "drift/cperf_optimizer_flop": "92b01be909b18a70bbc478e4d94db17e21ab830030bfd93f23e2b8d68133c5d8"
}

#: sha256 over the trained parameter values, in sorted key order, rounded to
#: six decimals. Unforgeable from the room: only the INITIAL weights are here.
WEIGHTS_DIGEST = "3bcb11418faa7099430af11d040cd2ccbebf0bd4e4ff829d894ffdb02a5f5651"

#: Readable statements of the result. Not sufficient to synthesise a trace,
#: and kept so a human can see what this run does.
EXPECTED_DRIFT_EVENTS = [350]
EXPECTED_FINAL_ACCURACY = 100.0
EXPECTED_EVAL_POINTS = 376
EXPECTED_METRIC_NAMES = 31

DATASET = {
    "train-images-idx3-ubyte": (0x803, 2000, 28, 28),
    "train-labels-idx1-ubyte": (0x801, 2000),
    "t10k-images-idx3-ubyte": (0x803, 2000, 28, 28),
    "t10k-labels-idx1-ubyte": (0x801, 2000),
}


def check_dataset() -> None:
    raw = HERE / "data" / "MNIST" / "raw"
    for name, header in sorted(DATASET.items()):
        path = raw / name
        assert path.is_file(), f"the pinned dataset is missing {name}"
        head = path.read_bytes()[:16]
        got = (struct.unpack(">IIII", head[:16]) if header[0] == 0x803
               else struct.unpack(">II", head[:8]))
        assert got == header, (
            f"{name}: header {got}, expected {header} -- this is not the "
            f"pinned subset, and the pinned result would not apply to it")


def run() -> None:
    PRODUCED.unlink(missing_ok=True)
    for stale in CKPT_DIR.glob("*"):
        stale.unlink()
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


def trace_digests(path: pathlib.Path) -> dict:
    """One digest per metric, over its (step, value) sequence in file order."""
    series: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 3:
                continue
            step, metric, value = row
            if any(t in metric for t in TIMED):
                continue
            series.setdefault(metric, []).append(f"{step}={value}")
    return {metric: hashlib.sha256("|".join(rows).encode()).hexdigest()
            for metric, rows in series.items()}


def weights_digest() -> str:
    """Digest the trained parameters the run checkpointed.

    Values rather than file bytes: a torch.save of the same tensors need not
    be byte-identical, but the numbers must be.
    """
    import torch

    pts = sorted(CKPT_DIR.glob("drift_adaptation_*.pt"))
    assert pts, (
        f"the run wrote no checkpoint under {CKPT_DIR.name}/. This claim pins "
        f"the TRAINED weights, so the run must checkpoint them -- "
        f"model.max_ckpts and model.ckpts_path are set in the config for "
        f"exactly this reason. An implementation that does not train cannot "
        f"satisfy this.")
    payload = torch.load(pts[-1], map_location="cpu", weights_only=True)
    assert isinstance(payload, dict) and payload, (
        f"the checkpoint is not a state dict: {type(payload).__name__}")
    hasher = hashlib.sha256()
    for key in sorted(payload):
        tensor = payload[key]
        hasher.update(key.encode())
        flat = tensor.detach().flatten().tolist()
        hasher.update(",".join(f"{v:.6f}" for v in flat).encode())
    return hasher.hexdigest()


def observe() -> tuple[dict, str, list, list]:
    rows = [r for r in csv.reader(open(PRODUCED, newline="")) if len(r) == 3][1:]
    events = [int(s) for s, m, v in rows
              if m == "drift/score" and float(v) > 0]
    accuracy = [float(v) for s, m, v in rows if m == "eval/accuracy"]
    return trace_digests(PRODUCED), weights_digest(), events, accuracy


def main() -> int:
    check_dataset()
    run()
    digests, weights, events, accuracy = observe()

    if "--emit" in sys.argv:
        print(json.dumps({"TRACE_DIGESTS": digests,
                          "WEIGHTS_DIGEST": weights,
                          "EXPECTED_DRIFT_EVENTS": events,
                          "EXPECTED_FINAL_ACCURACY": accuracy[-1],
                          "EXPECTED_EVAL_POINTS": len(accuracy),
                          "EXPECTED_METRIC_NAMES": len(digests)}, indent=1))
        return 0

    # 1. the readable statements
    assert events == EXPECTED_DRIFT_EVENTS, (
        f"drift fires at {events}, expected {EXPECTED_DRIFT_EVENTS}")
    assert accuracy, "the run recorded no accuracy"
    assert abs(accuracy[-1] - EXPECTED_FINAL_ACCURACY) < 1e-9, (
        f"final accuracy {accuracy[-1]}, expected {EXPECTED_FINAL_ACCURACY}")
    assert len(accuracy) == EXPECTED_EVAL_POINTS, (
        f"{len(accuracy)} eval points, expected {EXPECTED_EVAL_POINTS}")
    assert len(digests) == EXPECTED_METRIC_NAMES, (
        f"{len(digests)} metric names, expected {EXPECTED_METRIC_NAMES}")

    # 2. the trace, by digest, metric by metric
    missing = sorted(set(TRACE_DIGESTS) - set(digests))
    extra = sorted(set(digests) - set(TRACE_DIGESTS))
    assert not missing, f"the run recorded no {missing}"
    assert not extra, f"the run recorded unexpected metrics {extra}"
    wrong = sorted(m for m in TRACE_DIGESTS if TRACE_DIGESTS[m] != digests[m])
    assert not wrong, (
        f"{len(wrong)} of {len(TRACE_DIGESTS)} metric series do not match the "
        f"pinned run: {wrong}\n"
        f"If most of them differ, the run is probably not seeded: main.py "
        f"must call torch.manual_seed(cfg.seed) AND np.random.seed(cfg.seed) "
        f"before the harness is built (A41).")

    # 3. the trained weights, which cannot be copied from anything here
    assert weights == WEIGHTS_DIGEST, (
        f"the checkpointed parameters digest to {weights[:16]}..., expected "
        f"{WEIGHTS_DIGEST[:16]}... -- the model this run produced is not the "
        f"model the claim pins")

    (HERE / "RUN_OK").write_text("ok\n")
    print(f"mnist run reproduced: {len(digests)} metric series by digest, "
          f"trained weights match, drift at {events}, "
          f"final accuracy {accuracy[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
