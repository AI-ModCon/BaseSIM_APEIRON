# Lab 0: Setup (do this before the session)

About 15 minutes, most of it download time. At the end you will have Apeiron installed, the MNIST data cached, and a 1-minute smoke test passing.

## What you need

- **macOS on Apple Silicon, or Linux** (x86_64 or ARM). About 3 GB of free disk space.
- **Python 3.13.** Apeiron currently requires exactly 3.13. The steps below install it for you with `uv`, a fast Python package manager.
- **Windows:** use WSL2 with Ubuntu and follow the Linux steps inside it. Native Windows is not tested for this session.
- **Intel Macs:** current PyTorch releases no longer ship for Intel macOS, so a native install will fail. Use a Linux machine, a remote workstation, or the Docker image in the repo (`dockerfiles/Dockerfile`).

No GPU is needed. If you have one, everything also works on CUDA or Apple MPS (see the end of this page).

## 1. Install uv and get Apeiron

```bash
# uv (skip if `uv --version` already works)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/AI-ModCon/BaseSIM_APEIRON.git
cd BaseSIM_APEIRON
```

## 2. Create the environment

```bash
uv sync                        # creates .venv from uv.lock; downloads Python 3.13 if needed
source .venv/bin/activate      # every later command assumes this is active
```

On **Linux without an NVIDIA GPU**, `uv sync` installs the CUDA build of PyTorch. Replace it with the CPU build, which is much smaller:

```bash
uv pip install "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cpu
```

After that, run commands in the activated environment as shown in the labs (plain `python ...`). Avoid `uv run` and `uv sync`, which would put the CUDA build back; use `uv run --no-sync` if you prefer `uv run`.

Check the install:

```bash
python -c "import apeiron, torch; print('apeiron OK, torch', torch.__version__)"
```

## 3. Find the session material

The labs live in the repository at `tutorials/a6/` (this file is `tutorials/a6/00_setup.md`). All lab commands are run from the repository root and refer to `tutorials/a6/...`.

## 4. Download MNIST once

The labs stream the MNIST test set. Fetch it now rather than on shared conference Wi-Fi:

```bash
python -c "from torchvision import datasets; [datasets.MNIST('./data', train=t, download=True) for t in (True, False)]"
```

This creates `./data/MNIST/` (about 60 MB).

## 5. Smoke test

Run one stream window with a tiny adaptation budget:

```bash
python -m src.main --config tutorials/a6/configs/a6_mnist.toml \
    --set drift_detection.max_stream_updates=1 --set train.max_iter=5 \
    --set logging.metrics_output_path=output/a6/smoke.csv
```

You should see `==== Starting Continuous Monitoring ====`, a progress bar over the stream batches, and finally `==== Continuous Monitoring Complete ====`. Check that `output/a6/smoke.csv` exists. On a laptop CPU this takes [[MEASURE: smoke test time]].

If that works, you are ready. Delete `output/a6/smoke.csv` if you like.

## Optional: agent skills

Apeiron ships task-oriented agent skills: installing it into another project, running the examples, choosing a detector, scaffolding an experiment for your own data, and adding drift detection to an existing training loop. They follow the open `SKILL.md` format, which works in several agentic coding tools.

- They are already in the repo under `.claude/skills/` and `.codex/skills/`. Tools that read those folders pick them up when you open the repo.
- For other tools, or to install them globally, use the `basesim-skills` domain from the Genesis skills catalog: `./unpack.sh basesim-skills` in a clone of https://github.com/AI-ModCon/genesis-skills.

Lab 4 uses them. The session demonstrates one tool for time, but any tool that supports skills works.

## Using a GPU (optional)

The lab config pins `device = "cpu"` so everyone gets comparable timings. To use your accelerator, add `--set device=auto` to any command. `auto` picks CUDA, then Apple MPS, then CPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `requires-python` / `No solution found` during install | You are not on Python 3.13. Let `uv sync` provide it, or run `uv python install 3.13`, and use the `.venv` it creates. |
| `No module named 'examples'` or `'src'` | Run commands from the repository root, not from `tutorials/a6`. |
| A browser or login prompt for Weights & Biases | Your command is using a bundled example config. The A6 configs set `[logging] backend = "none"`. Or add `--set logging.backend=none`. |
| `TypeError: ... unexpected keyword argument` at start-up | A config key is misspelled. Apeiron rejects unknown keys inside a section. |
| No CSV appears | The CSV is written when the run finishes. Also check `[logging] metrics_output_path`; a `[visualization]` section in older docs is ignored. |
| Runs are very slow | Reduce `--set drift_detection.max_stream_updates=4` and `--set train.max_iter=50`. |
| Install fails on an Intel Mac | See "What you need" above. |
| CUDA libraries reappear after a CPU-only install | `uv run` or `uv sync` re-synced `.venv` to `uv.lock`. Re-run the CPU install line and use the activated environment. |
