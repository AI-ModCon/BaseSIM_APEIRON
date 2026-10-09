# Lab 4: Your own problem

**Self-paced.** Aim to leave with your own model running inside the monitor → detect → adapt loop, even on a small slice of your data.

## Start with five questions

Answer these before writing code. They decide every setting in the config.

1. **What is a stream window for you?** A shift, a run, a simulation campaign batch, a day of beam time, a new site. This is what `update_data_stream()` returns.
2. **What can you score, and when?** Drift detection here watches a performance metric, so you need ground truth sometime: measurements, delayed labels, a trusted simulator. If labels arrive late, your windows should follow their cadence.
3. **What does a missed drift cost, and what does an unneeded update cost?** That sets detector sensitivity (Lab 2).
4. **Must the model keep working on old regimes?** If yes, use replay and a stability-preserving update such as EWC. If not, plain fine-tuning on recent data may be best (Lab 3).
5. **What is your adaptation budget?** Seconds on a GPU or minutes on a CPU per update sets `max_iter`, and decides whether you adapt on detection or on a schedule.

## Path A: Start from the harness template (recommended)

`tutorials/a6/scripts/harness_template.py` is a complete, working harness for a synthetic regression stream that drifts in two ways: the inputs move (covariate shift) and the input-to-output relationship rotates (concept drift). Run it first, unchanged:

```bash
python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_custom.toml \
    --harness tutorials/a6/scripts/harness_template.py:DriftingRegressionHarness
python tutorials/a6/scripts/plot_run.py output/a6/custom.csv
```

Expected: [[MEASURE: lab4 template result from verify report]]. The detector watches MAE, where lower is better, so drift shows up as the line going *up*.

Then copy the file and change the three places marked `>>> YOUR PROBLEM <<<`:

| Where | What to put there |
|---|---|
| `build_model()` | your PyTorch model; load your trained checkpoint here and delete `_pretrain()` |
| `load_window(t, n)` | read window `t` of your data and return `(inputs, targets)` tensors |
| `eval_metrics` / `higher_is_better` | the scores to monitor. The first entry is `metric_index = 0`. |

Point the config at your copy (`--harness my_harness.py:MyHarness`) and adjust `a6_custom.toml`:

- `detection_interval`: aim for at least 10–30 checks per window (stream samples ÷ `data.batch_size` ÷ interval).
- Detector thresholds are in your metric's units. For Page-Hinkley, start `ph_threshold` near a few times the metric's typical batch-to-batch noise.
- `max_iter`, `init_lr`: the adaptation budget and step size. Start small and increase.
- `max_ckpts` / `ckpts_path`: every adapted model is saved so you can redeploy or roll back.

The same harness works with `python -m src.drift_only` and `python -m src.cl_only` once you register it in `examples/utils.py:get_example`; see `docs/model_harness.md` in the repo.

## Path B: Let an agent scaffold it

If you use an agentic coding tool, the Apeiron skills (see [Setup](00_setup.md#optional-agent-skills)) do this interactively:

- **`custom-experiment`** scaffolds a harness, data utilities and a TOML config for your dataset and model, registers it, and runs a smoke test. Example request: *"Create an Apeiron experiment for my dataset in data/train.csv with the model in models/surrogate.py."*
- **`integrate-apeiron`** reads an existing training script and adds the lightest adapter that fits.
- **`choose-detector`** asks about your metric and the drift you expect, and writes the `[drift_detection]` block.

Review what the agent writes. Check the gotchas below.

## Path C: Just add a detector to a training loop you already have

You do not need the harness to get value. The detectors work on any number you already compute:

```python
from apeiron.drift_detection import ADWINDetector

detector = ADWINDetector(delta=0.002)

for step, batch in enumerate(incoming_data):          # your existing loop
    score = evaluate(model, batch)                     # e.g. validation MAE on the new data
    signal = detector.update(score)
    if signal.drift_detected:
        print(f"drift at step {step}: regime={signal.regime.value}, score={signal.drift_score:.3f}")
        # alert, pause the model, retrain, or call Apeiron's ContinuousTrainer
```

`signal.regime` suggests a response (`continual_learning`, `fine_tuning` or `retrain`) based on how severe the change looks.

## Gotchas when wrapping your own model

- **Full training batches.** During adaptation, Apeiron only uses training batches with at least `[train] batch_size` samples. A window smaller than one batch makes it wait forever. Use `drop_last=True` and make sure each window holds at least one full batch (the template asserts this).
- **Metric direction.** Detectors fire on changes in *either* direction. Set `higher_is_better` correctly, and read `fwt`/`bwt` with your metric's direction in mind (for an error metric, positive BWT means forgetting).
- **Method name.** The optimizer hook is `get_optimizer`. Older harnesses spelled it `get_optmizer`; that still works but prints a deprecation warning.
- **Workers and devices.** Keep `num_workers = 0` until everything works. Move to GPU with `--set device=auto`.
- **Unknown config keys** inside a section stop the run at start-up. That is intentional, to catch typos.

## Where to go next

- `docs/model_harness.md`, `docs/choosing_a_detector.md`, `docs/continuous_learning.md` and `docs/tracking.md` in the repo
- Documentation site: https://basesim-apeiron.readthedocs.io
- GPU-scale examples: `examples/cifar/` (a ViT and a VGG on CIFAR-10)
