"""Score every saved adaptation checkpoint against every arrival of the stream.

The streamed run answers "how well does the model do on the simulation that just
arrived". It cannot answer "and what did adapting to it cost on the data we had
already learned", because by the time a later arrival is streamed the earlier one
is gone. This replays them: for each checkpoint APEIRON wrote at a drift event,
evaluate it on every arrival, including the ones that came before it.

The result is the continual-learning R-matrix, ``R[event][arrival]``, from which
backward transfer follows directly -- the error on early arrivals as a function
of how much adaptation has happened since.

Two things to be careful about, both of which this script records rather than
resolves:

* "Historical" in the streamed run means the most recent arrival of the stream's
  *first case*, not MATEY's pre-training corpus. Forgetting measured here is
  forgetting of the starting case.
* An arrival that a CL round trained on is not evidence of backward transfer --
  the model has seen it. ``adapted_arrivals`` in the sidecar marks those so the
  figure can exclude them.

Usage::

    python examples/matey/eval_retrospective.py \\
        --config examples/matey/matey_stream.toml \\
        --arm base --ckpts $OUTDIR/ckpts_base --run-log $OUTDIR/run_base.log \\
        --set data.path=$STREAM --set model.pretrained_path=$CKPT
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from apeiron.config.configuration import build_config  # noqa: E402
from apeiron.logger import get_logger  # noqa: E402
from examples.matey.model import MATEYHarness  # noqa: E402
from examples.matey.model_stream import MATEYStreamHarness  # noqa: E402

ARRIVAL_RE = re.compile(r"==== arrival (\d+)/\d+:")
DRIFT_RE = re.compile(r"==== DRIFT DETECTED \(Event #(\d+)\)")
CKPT_RE = re.compile(r"drift_adaptation_(\d+)\.pt$")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--arm", required=True, help="label for the run being scored")
    p.add_argument("--ckpts", required=True, help="directory of drift_adaptation_*.pt")
    p.add_argument("--run-log", default="", help="run log, for the event->arrival map")
    p.add_argument("--arrivals", default="all", help="'all', '0-7', or '0,4,8'")
    p.add_argument("--out", default="", help="output CSV (default: <ckpts>/../retro)")
    return p.parse_known_args(argv)


def parse_arrivals(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def event_to_arrival(run_log: str) -> dict[int, int]:
    """Map each drift event to the arrival it fired on.

    Both markers are printed by the same run in file order, so the arrival is
    simply the most recent one announced above the event.
    """
    mapping: dict[int, int] = {}
    if not run_log or not Path(run_log).is_file():
        return mapping
    current = -1
    for line in Path(run_log).read_text(errors="ignore").splitlines():
        arrival = ARRIVAL_RE.search(line)
        if arrival:
            current = int(arrival.group(1)) - 1  # the log is 1-based
            continue
        drift = DRIFT_RE.search(line)
        if drift:
            mapping[int(drift.group(1))] = current
    return mapping


def _windows(harness):
    """Yield one metric list per monitoring window, instead of their mean.

    ``BaseModelHarness.eval()`` returns a single batch-weighted average over the
    arrival's loader. The figure needs the same per-window resolution the online
    run records, so the loop is repeated here rather than collapsed.
    """
    import torch

    harness.model.eval()
    with torch.no_grad():
        for batch in harness.get_train_dataloaders()[1]:
            x, y = harness._unpack(batch)
            x, y = x.to(harness.cfg.device), y.to(harness.cfg.device)
            y_hat = harness.model(x)
            yield [
                harness._to_scalar(m(y_hat, y)) for m in harness.eval_metrics.values()
            ]


def find_checkpoints(ckpts: Path) -> list[tuple[int, Path]]:
    found = []
    for path in ckpts.glob("drift_adaptation_*.pt"):
        m = CKPT_RE.search(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    cfg = build_config(["--config", args.config, *passthrough])
    logger = get_logger()

    ckpts = Path(args.ckpts)
    events = find_checkpoints(ckpts)
    fired_at = event_to_arrival(args.run_log)
    if fired_at and len(fired_at) != len(events):
        # Catches both FIFO eviction (max_ckpts too small) and two arms sharing
        # one ckpts_path, either of which silently scores the wrong weights.
        raise SystemExit(
            f"{len(events)} checkpoints in {ckpts} but {len(fired_at)} drift events "
            f"in {args.run_log}. Raise model.max_ckpts, or give each arm its own "
            f"model.ckpts_path -- the numbers must agree or the mapping is guesswork."
        )

    # One harness, built from model.pretrained_path: that checkpoint has the
    # hyperparams.yaml the architecture is rebuilt from, and the adaptation
    # snapshots under ckpts_path do not. Only state_dicts are swapped below.
    harness = MATEYStreamHarness(cfg)
    inner = harness._adapter_model.matey_model
    metrics = list(harness.eval_metrics)

    arrivals = parse_arrivals(args.arrivals, harness.n_arrivals)
    # Event 0 is the un-adapted model, the reference every other row is read against.
    todo = [(0, str(cfg.model.pretrained_path))] + [(e, str(p)) for e, p in events]

    out_path = Path(args.out) if args.out else ckpts.parent / f"retro_{args.arm}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "arm",
                "event_id",
                "ckpt",
                "fired_at_arrival",
                "eval_arrival",
                "window",
                "case",
                "machine",
                "in_pretraining",
                "held_out",
                "metric",
                "value",
            ]
        )
        # Arrivals outer: rebuilding a loader costs seconds, swapping a
        # state_dict costs a fraction of one.
        for arrival_idx in arrivals:
            harness.task_counter = arrival_idx
            harness.update_data_stream()
            meta = harness.current_arrival()
            for event_id, ckpt_path in todo:
                MATEYHarness._load_pretrained_weights_if_available(inner, ckpt_path)
                for window, values in enumerate(_windows(harness)):
                    for name, value in zip(metrics, values):
                        writer.writerow(
                            [
                                args.arm,
                                event_id,
                                Path(ckpt_path).name,
                                fired_at.get(event_id, -1),
                                arrival_idx,
                                window,
                                meta.get("case"),
                                meta.get("machine"),
                                meta.get("in_pretraining"),
                                meta.get("held_out"),
                                name,
                                float(value),
                            ]
                        )
                        rows += 1
            logger.info(
                f"retrospective: arrival {arrival_idx} scored by {len(todo)} "
                f"checkpoints",
                level=1,
            )

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "arm": args.arm,
                "stream_root": str(cfg.data.path),
                "pretrained": str(cfg.model.pretrained_path),
                "baseline_case": harness._baseline_case,
                "n_arrivals": harness.n_arrivals,
                "machine_change_points": harness._manifest.get("machine_change_points"),
                # Arrivals a CL round trained on. The model has seen these, so
                # they cannot evidence backward transfer.
                "adapted_arrivals": sorted(set(fired_at.values()) - {-1}),
                "events": [
                    {
                        "event_id": e,
                        "ckpt": Path(p).name,
                        "fired_at": fired_at.get(e, -1),
                    }
                    for e, p in todo
                ],
            },
            indent=2,
        )
    )
    logger.info(f"wrote {rows} rows to {out_path} and {sidecar}", level=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
