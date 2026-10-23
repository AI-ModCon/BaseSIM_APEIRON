# Adaptive Models: Building Drift-Aware Continual Learning Systems

Hands-on material for the Genesis Mission Platform session on keeping scientific AI models accurate when the data changes underneath them.

Models trained offline go stale once the instrument, the operating regime or the region of parameter space moves. In this session you build a pipeline that notices when that happens and adapts the model without retraining it from scratch. The tool is **Apeiron**, an open-source PyTorch framework that wraps an existing model and runs the loop for you:

```
stream of data ──► score the model ──► drift detector ──► no drift: keep going
                                              │
                                              └─ drift: adapt the model (continual learning) ──► redeploy ──► keep going
```

## What you will be able to do

- Run a complete drift-aware pipeline and read what it did: when drift was flagged, what adaptation cost, what it bought.
- Choose and tune a drift detector for your signal, and explain the trade-off between missed drift and false alarms.
- Choose a continual-learning strategy and check whether it adapts to new data without forgetting old regimes.
- Tell whether drift-triggered adaptation beats adapting on a fixed schedule for your problem.
- Wrap your own model and data stream so Apeiron can monitor and adapt it.

## Labs

| Lab | Time | What you do |
|---|---|---|
| [0. Setup](00_setup.md) | before the session | Install Apeiron, fetch the data, run a smoke test |
| [1. Your first adaptive pipeline](01_first_pipeline.md) | 15 min | Run monitor → detect → adapt on a drifting MNIST stream and plot it |
| [2. Choosing a drift detector](02_drift_detectors.md) | 18 min | Compare ADWIN, KSWIN, Page-Hinkley and ensembles on the same stream |
| [3. Choosing an adaptation strategy](03_continual_learning.md) | 22 min | Fine-tuning vs. EWC vs. replay; forgetting; is the detector worth it? |
| [4. Your own problem](04_your_own_problem.md) | self-paced | Wrap your model and data with the harness template or an agent skill |

The MNIST example is small on purpose, so every run finishes in minutes on a laptop CPU. The same configuration switches, detectors and strategies are what you would use on a facility data stream.

## What is in this folder

```
README.md                 this page
00_setup.md … 04_*.md     the labs
configs/a6_mnist.toml     base config for Labs 1-3 (each lab changes one thing with --set)
configs/a6_custom.toml    config for the Lab 4 harness template
scripts/plot_run.py       plot a run: monitored metric, detections, adaptations
scripts/compare_runs.py   summary table across runs
scripts/run_lab2.sh       detector sweep (Lab 2)
scripts/run_lab3.sh       strategy sweep (Lab 3); --all for the full grid
scripts/a6_run.py         run Apeiron with your own harness and/or detector class
scripts/custom_detector.py  a detector you can read in one minute and modify
scripts/harness_template.py a complete harness for a drifting regression stream
scripts/verify_all.sh     run every lab end to end and write a report (instructors)
```

All commands assume that this folder is at `tutorials/a6/` inside the Apeiron repository, and that you run them from the repository root.

## Getting help

- Apeiron documentation: https://basesim-apeiron.readthedocs.io
- Source and issue tracker: https://github.com/AI-ModCon/BaseSIM_APEIRON
- Agent skills for Apeiron (for agentic coding tools): the `basesim-skills` domain of https://github.com/AI-ModCon/genesis-skills. They also ship inside the Apeiron repo under `.claude/skills/` and `.codex/skills/`.
