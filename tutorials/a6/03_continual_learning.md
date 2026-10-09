# Lab 3: Choosing an adaptation strategy

**22 minutes.** You compare how the model is updated once drift is detected, and check whether it forgets what it knew.

## Theory in two minutes

Adapting a deployed model is a balancing act between **plasticity** (learn the new regime quickly) and **stability** (do not overwrite the old regimes). Push too far toward plasticity and you get *catastrophic forgetting*: the model fits today's data and breaks on yesterday's. Push too far toward stability and the model stops absorbing anything new.

Apeiron separates two choices you make independently:

**How the weights are updated**, set by `[continual_learning] update_mode`:

| `update_mode` | Idea |
|---|---|
| `none` | never change the weights (the "static model" baseline) |
| `base` | plain fine-tuning on the current window |
| `ewc_online` | Elastic Weight Consolidation: penalize moving the weights that mattered for earlier windows (`ewc_lambda` sets how strongly) |
| `kfac_online` | a curvature-aware relative of EWC that also models interactions between weights (`kfac_lambda`) |
| `jvp_reg` | a robustness-oriented update (sharpness-aware) that always combines current and past data |

**What data the update sees**, set by `mix_historic_data`. With `true`, half of each adaptation batch is replayed from past windows. Replay is often the single most effective defence against forgetting, and it is independent of the update rule.

How to read the results:

- `stream_mean`: average accuracy the deployed model actually delivered over the whole stream. This is what your users experience.
- `hist_acc`: accuracy on past windows after each update. It drops if the model forgets.
- `fwt`: how much each update improved the window that triggered it. `bwt`: how much later updates changed earlier windows (negative = forgetting).
- `seconds`: what it cost.

## 1. Run the core comparison (≈ [[MEASURE: run_lab3.sh core time]])

```bash
bash tutorials/a6/scripts/run_lab3.sh
```

Three runs, all with the same ADWIN detector:

| Run | What it is |
|---|---|
| `lab3_never` | detection off, never adapt: the frozen model you deployed |
| `lab3_base_noreplay` | fine-tune on the new window only |
| `lab3_ewc_replay` | EWC plus replay of past windows (the Lab 1 setting) |

`lab3_never` uses `src.cl_only --schedule never`, which scores the frozen model on the same stream. Each run is a normal command you can copy from the script.

Reference result: [[MEASURE: lab3 core summary table from verify report]].

## 2. Read the results (6 min)

- How much stream accuracy does adapting buy over never adapting?
- Compare `hist_acc` and `bwt` between plain fine-tuning and EWC + replay. Which one forgets?
- Look at `seconds`. Is the better strategy also the more expensive one? By how much?

## 3. Is the detector worth it? (8 min, or in self-paced time)

A detector earns its keep only if adapting *when it says so* beats adapting on a clock. The full grid adds the other strategies plus two schedule-driven controls that use the same EWC + replay updater:

```bash
bash tutorials/a6/scripts/run_lab3.sh --all
```

| Run | Trigger |
|---|---|
| `lab3_every_window` | adapt after every window, regardless of drift (upper bound on effort) |
| `lab3_every_3rd` | adapt after every third window |
| `lab3_ewc_replay` | adapt only when ADWIN fires |

Compare `adaptations`, `stream_mean` and `seconds` across these three. The question to answer for your own work: *for the accuracy I need, which trigger gives the fewest, cheapest adaptations?*

Reference result: [[MEASURE: lab3 --all summary table from verify report]].

## Stretch

1. **Tune the stability knob.** Re-run EWC with `--set continual_learning.ewc_lambda=100` and `=10000`, and watch `fwt` versus `hist_acc` move in opposite directions.
2. **Budget.** Halve and double `--set train.max_iter=...`. Where does extra adaptation effort stop paying?
3. **Write your own update rule.** Updaters subclass `BaseUpdater` (`src/apeiron/training/updater/base.py`) and override hooks:
   - `cl_preprocessing` / `cl_postprocessing` run before and after each adaptation. EWC estimates weight importance here.
   - `update_post_fwd_bwd` runs after `backward()` and before the optimizer step. Modify the gradients here, and return the penalty value for logging.
   - `update_post_optimizer_call` runs after the optimizer step.

   `ewc.py` (about 140 lines) is a compact example to copy. Plug yours in without editing Apeiron:
   ```bash
   python tutorials/a6/scripts/a6_run.py --config tutorials/a6/configs/a6_mnist.toml \
       --updater path/to/my_updater.py:MyUpdater \
       --set logging.metrics_output_path=output/a6/lab3_mine.csv
   ```
