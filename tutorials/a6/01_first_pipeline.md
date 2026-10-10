# Lab 1: Your first adaptive pipeline

**15 minutes.** You run the full loop (monitor, detect drift, adapt, redeploy) on a stream that drifts, and learn to read what it did.

## The scenario

A digit classifier was trained once and deployed. Each new **stream window** delivers the MNIST test images through a camera that keeps degrading: every window adds another small rotation, shear and zoom on top of the previous ones. The model never sees labels for training while it is deployed. It is only scored on the stream, the way you would score a surrogate against measurements as they arrive. Your pipeline has to notice the degradation and fix the model on its own.

The starting model is deliberately under-trained, at about 50% accuracy. That leaves room to see adaptation help.

## 1. Read the config (3 min)

Open `tutorials/a6/configs/a6_mnist.toml`. Each section is one piece of the loop:

| Section | Role in the loop |
|---|---|
| `[model]`, `[data]` | Which model and which data stream. `data.name = "mnist"` selects the bundled harness in `examples/mnist/model.py`. |
| `[drift_detection]` | How the model is watched. It is scored on every stream batch; every `detection_interval` batches the scores are aggregated (`mean`) and the chosen `metric_index` (0 = accuracy) goes to the detector. |
| `[continual_learning]` | What happens when drift is flagged. `update_mode` is the strategy; `mix_historic_data = true` mixes in past windows (replay). |
| `[train]` | The adaptation budget: `max_iter` gradient steps per drift event. |
| `[logging]` | Where metrics go. The CSV at `metrics_output_path` is what you analyse. |

`max_stream_updates = 6` means 6 windows, so 6 levels of distortion.

## 2. Run it (about 3 min on a desktop Linux CPU; slower laptops can take 10–20 min)

```bash
python -m src.main --config tutorials/a6/configs/a6_mnist.toml
```

While it runs, watch the log for this sequence:

```
==== DRIFT DETECTED (Event #1)! ====
    Regime: ...            <- how severe the detector thinks it is
-> Dispatching continual learning module...
==== Continual Learning ====
    Initial test acc: ...  <- accuracy on the current window before adapting
CL Updates (drift_event_id=1): 100%|████| 100/100
    Test Accuracy: ...     <- ... and after adapting
    FWT: ...               <- the gain from this update (after minus before)
==== RESUMING MONITORING! ====
```

## 3. Look at what happened (5 min)

```bash
python tutorials/a6/scripts/plot_run.py output/a6/lab1.csv
python tutorials/a6/scripts/compare_runs.py output/a6/lab1.csv
```

Open `output/a6/lab1.png`. The blue line is what the detector saw: accuracy averaged over each group of 10 batches. Red dashed lines are detections, and each one was followed by an adaptation.

What you should see (reference run, seed 1337): **2 detections, each followed by an update**. The first comes in the 4th window, and the update lifts accuracy on that window from about 72% to 97%. The second, in the 5th window, takes it from about 92–94% to 95%. Accuracy on earlier windows stays around 95% (backward transfer near zero, so very little forgetting). Exact values vary a little between machines. Averaged over the whole stream the watched accuracy is about 78%, and about 94% over the last fifth.

You may also notice accuracy *rise* in the 3rd window, before any update. The deliberately under-trained starting model happens to score better on that window's distortion. Drift detectors watch for any change in the score, and a rise is a change too.

The CSV has one row per logged value (`step, metric, value`). The metrics you will use most:

| Metric | Meaning |
|---|---|
| `eval/accuracy` | accuracy on each stream batch, i.e. what the deployed model delivers |
| `drift/metric_0` | the aggregated value the detector received at each check |
| `drift/detected` | 1 when the detector fired. Rows with 0 are sampled at 10% to keep the file small. |
| `eval/test_pre_cl_acc`, `eval/test_curr_acc` | current-window accuracy before and after each adaptation |
| `eval/test_hist_acc` | accuracy on replayed past windows after each adaptation, i.e. did we forget? |
| `eval/fwt`, `eval/bwt` | transfer metrics. `fwt` is the gain on the new window; `bwt` is the change on earlier windows (negative = forgetting). |

## 4. Think about it (2 min)

- Did every detection correspond to a real change in the stream? The windows change at fixed points, but the detector only sees noisy averages.
- After an adaptation, did accuracy on the stream go up and stay up?
- What would the same plot look like for your model? What is your "stream window": a shift, a run, a simulation batch, a day of beam time?

## Stretch

1. **Detectors are two-sided.** Run again with the detector *not* reset after adapting:
   ```bash
   python -m src.main --config tutorials/a6/configs/a6_mnist.toml \
       --set drift_detection.reset_after_learning=false \
       --set logging.metrics_output_path=output/a6/lab1_noreset.csv
   python tutorials/a6/scripts/plot_run.py output/a6/lab1.csv output/a6/lab1_noreset.csv --out output/a6/lab1_reset.png
   ```
   Any change in the metric's mean looks like drift to a statistical detector, including the *improvement* right after an adaptation. Count the extra detections.
2. **Track it in a dashboard.** Add `--set logging.backend=mlflow`, then run `mlflow ui` and open http://localhost:5000.
