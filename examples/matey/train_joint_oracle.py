#!/usr/bin/env python3
"""Jointly fine-tune the pre-trained checkpoint on *every* arrival in the stream.

This builds the upper-bound arm of Figure 3. The comparison it completes is the
standard one for an online method:

  frozen   the pre-trained checkpoint, never updated       (lower bound)
  CL       APEIRON adapts online, causally, on drift       (the method)
  oracle   one model fine-tuned offline on the train
           splits of all N arrivals, then frozen and
           streamed exactly like the other two             (upper bound)

The oracle sees the whole campaign at once, including the arrivals the online
method had not met yet when it was scored on them. That non-causality is the
point: it brackets how much of the remaining error is reachable by *any*
adaptation schedule, as opposed to how much is simply MATEY's capacity on this
data.

What this is NOT
----------------
It is **not** a from-scratch re-pre-training of MATEY on all the data. That is a
multi-node pre-training job against the full ``Datasets_pretraining`` tree and is
out of scope here. Everything below starts from the same ``leadtime_1``
checkpoint the other two arms start from and fine-tunes it. Describe it as
"jointly fine-tuned on all arrivals", never as "trained on all the data from
scratch" -- and note the checkpoint had already seen three of the four cases in
pre-training, so on those the oracle is refining, not learning.

Honesty of the split
--------------------
Training touches only each arrival's ``train_range``. The staging step keeps
``train_range`` and ``valid_range`` disjoint with a 5-frame gap, and the stream
run scores on ``valid_range``, so the oracle is never trained on the frames it
is scored against. Verified per arrival before the first gradient step; the run
aborts if the manifest says otherwise.

Usage
-----
    python examples/matey/train_joint_oracle.py \\
        --config examples/matey/matey_stream.toml \\
        --out /lustre/.../mateydata/models/joint_oracle \\
        --epochs 3 --steps-per-arrival 40

Then stream it as a third arm:

    python -m src.main --config examples/matey/matey_stream.toml \\
        --set model.pretrained_path=<out>/best_ckpt.tar \\
        --set continual_learning.update_mode=none \\
        --set visualization.input=<run>/stream_oracle.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

from apeiron.config.configuration import build_config
from apeiron.logger import get_logger
from examples.matey.model_stream import MATEYStreamHarness


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--out", required=True, help="directory for best_ckpt.tar + hyperparams.yaml"
    )
    ap.add_argument(
        "--epochs", type=int, default=3, help="passes over the full set of arrivals"
    )
    ap.add_argument(
        "--steps-per-arrival",
        type=int,
        default=40,
        help="gradient steps per arrival per epoch",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument(
        "--max-arrivals",
        type=int,
        default=0,
        help="debug: use only the first N arrivals (0 = all). A "
        "checkpoint written with this set is NOT an oracle",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="debug: build everything and take a few steps, but do "
        "not write a checkpoint",
    )
    # Everything else is forwarded to build_config as --set overrides.
    return ap.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(list(sys.argv[1:] if argv is None else argv))

    cfg = build_config(["--config", args.config, *passthrough])
    logger = get_logger(verbosity=cfg.verbosity)

    # update_mode is irrelevant here -- this script owns its own loop -- but the
    # harness asserts on it, and "none" is always accepted.
    harness = MATEYStreamHarness(cfg)
    device = torch.device(cfg.device)

    arrivals = harness._arrivals  # noqa: SLF001 -- the manifest is the contract
    if args.max_arrivals > 0:
        arrivals = arrivals[: args.max_arrivals]
        logger.warning(
            f"--max-arrivals={args.max_arrivals}: training on a PREFIX of the "
            "stream. This is a debug mode -- the result is not an upper bound "
            "and must not be plotted as the oracle arm."
        )
    n = len(arrivals)

    # The whole value of this arm is that it is an *honest* upper bound. If any
    # arrival's train and valid ranges overlap, the number it produces is
    # memorisation and the figure would be wrong in the flattering direction.
    bad = [a["index"] for a in arrivals if not a.get("disjoint", False)]
    if bad:
        raise SystemExit(
            f"arrivals {bad} have overlapping train/valid ranges; re-stage with "
            "stage_solps_stream.py before training an oracle on them"
        )
    logger.info(
        f"All {n} arrivals have disjoint train/valid ranges "
        f"(gap {arrivals[0].get('gap')} frames)",
        level=0,
    )

    optimizer = harness.get_optmizer()
    criterion = harness.get_criterion()
    generator = torch.Generator().manual_seed(args.seed)

    total_planned = args.epochs * n * args.steps_per_arrival
    logger.info("==== joint fine-tune (oracle arm) ====", level=0)
    logger.info(f"\tarrivals:          {n}", level=1)
    logger.info(f"\tepochs:            {args.epochs}", level=1)
    logger.info(f"\tsteps/arrival:     {args.steps_per_arrival}", level=1)
    logger.info(f"\tplanned steps:     {total_planned}", level=1)
    logger.info(f"\tlr:                {cfg.train.init_lr:g}", level=1)
    logger.info(f"\tdevice:            {device}", level=1)

    history: list[dict] = []
    step_total = 0
    t0 = time.time()

    harness.model.train()
    for epoch in range(args.epochs):
        # Shuffle the arrival order every epoch. Walking them in stream order
        # would end every epoch on the last case, and the final steps bias the
        # weights toward whatever was seen last -- exactly the recency effect
        # the oracle exists to be free of.
        order = torch.randperm(n, generator=generator).tolist()
        epoch_loss, epoch_steps = 0.0, 0

        for arrival_idx in order:
            # MATEYStreamHarness.update_data_stream() reads task_counter as the
            # arrival index and then increments it, so setting it here is how a
            # caller selects an arrival out of order.
            harness.task_counter = arrival_idx
            harness.update_data_stream()
            train_loader, _ = harness.get_train_dataloaders()

            steps_here = 0
            for batch in train_loader:
                if steps_here >= args.steps_per_arrival:
                    break
                x, y = harness._unpack(batch)  # noqa: SLF001
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(harness.model(x), y)
                loss.backward()
                optimizer.step()

                value = float(loss.item())
                epoch_loss += value
                epoch_steps += 1
                steps_here += 1
                step_total += 1
                if step_total % args.log_every == 0:
                    logger.info(
                        f"\tepoch {epoch + 1}/{args.epochs}  step {step_total}"
                        f"/{total_planned}  loss {value:.5f}  "
                        f"({time.time() - t0:.0f}s)",
                        level=1,
                    )

            if steps_here == 0:
                logger.warning(
                    f"arrival {arrival_idx} yielded no training batches -- "
                    "check its train_range staged correctly"
                )

        mean_loss = epoch_loss / max(1, epoch_steps)
        history.append(
            {"epoch": epoch + 1, "steps": epoch_steps, "mean_loss": mean_loss}
        )
        logger.info(
            f"epoch {epoch + 1}/{args.epochs} done: {epoch_steps} steps, "
            f"mean loss {mean_loss:.5f}",
            level=0,
        )

    # ---- save ------------------------------------------------------------
    if args.dry_run:
        logger.info("--dry-run: not writing a checkpoint", level=0)
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_ckpt.tar"

    # Save the *inner* MATEY module: _load_pretrained_weights_if_available()
    # loads into the model built by _build_matey_model, not into the adapter, so
    # adapter-prefixed keys would fail every one of its five key transforms.
    state = harness._adapter_model.matey_model.state_dict()  # noqa: SLF001
    torch.save({"model_state": state}, ckpt_path)

    # The harness reads hyperparams.yaml from beside the checkpoint (or one level
    # up) for architecture. Without it the oracle checkpoint cannot be rebuilt.
    src_yaml = harness._resolve_checkpoint_hyperparams_yaml(  # noqa: SLF001
        cfg.model.pretrained_path
    )
    if src_yaml is not None:
        shutil.copy2(src_yaml, out_dir / "hyperparams.yaml")
        logger.info(f"copied {src_yaml.name} -> {out_dir}", level=1)
    else:
        logger.warning(
            "no hyperparams.yaml found beside the source checkpoint; the oracle "
            "checkpoint may not rebuild"
        )

    meta = {
        "source_checkpoint": str(cfg.model.pretrained_path),
        "stream_root": str(harness._stream_root),  # noqa: SLF001
        "n_arrivals": n,
        "epochs": args.epochs,
        "steps_per_arrival": args.steps_per_arrival,
        "total_steps": step_total,
        "init_lr": float(cfg.train.init_lr),
        "seed": args.seed,
        "trained_on": "train_range of every arrival (valid_range never touched)",
        "history": history,
        "wall_seconds": round(time.time() - t0, 1),
        "note": (
            "Fine-tuned from the pre-trained checkpoint on all arrivals jointly. "
            "NOT a from-scratch re-pre-training of MATEY."
        ),
    }
    with (out_dir / "joint_oracle.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(f"wrote {ckpt_path}", level=0)
    logger.info(f"wrote {out_dir / 'joint_oracle.json'}", level=0)
    logger.info(f"total {step_total} steps in {time.time() - t0:.0f}s", level=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
