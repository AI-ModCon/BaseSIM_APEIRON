---
name: custom-experiment
description: |
  Run an apeiron experiment on the user's OWN dataset and model end-to-end.
  Use when the user wants to bring their own data + architecture (beyond the
  shipped MNIST/CIFAR examples), scaffold a custom model harness, write a config
  for it, smoke-test it, and run the full experiment. Self-contained: it creates
  the harness, data utilities, and TOML, registers them in the example factory,
  and runs. For trying the bundled examples instead, use explore-examples; for
  adding apeiron to a separate project's training loop, use integrate-apeiron.
argument-hint: "<short_name> [config_output_path]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

Scaffold and run an apeiron experiment on the user's own data and model.

## Arguments
- `$1`: Short name for the dataset/harness (lowercase, e.g. `fashionmnist`, `mytabular`). Used for the `examples/$1/` dir and the `data.name` factory key.
- `$2`: Optional output path for the TOML config. Defaults to `examples/$1/$1.toml`.

## Procedure

### 1. Gather the specifics from the user
Ask only for what isn't already provided:
- Dataset source and how to load it (torchvision, HuggingFace, local files, custom `Dataset`).
- Model architecture (CNN, MLP, ViT, …), input shape, number of classes/outputs.
- Type of drift to simulate on the stream (e.g. affine transforms for images, feature noise for tabular). apeiron's examples simulate drift inside `update_data_stream()`.
- Pretrained weights? Path if so (optional — harness should tolerate their absence).
- Which drift detector and CL updater to start with (default `ADWINDetector` + `base`).

### 2. Read the current patterns (don't hardcode signatures — they rot)
Mirror the live source rather than assuming method names:
```bash
cat src/apeiron/model/torch_model_harness.py   # the ABC + abstract methods to implement
cat examples/mnist/model.py                      # canonical harness
cat examples/mnist/utils.py                       # data-loading + drift-sim pattern
cat examples/utils.py                             # get_example() factory to extend
grep -nA6 "class .*Cfg" src/apeiron/config/configuration.py  # config fields
```
Implement exactly the `@abstractmethod`s the ABC declares (currently includes `get_optmizer` — note that spelling — `update_data_stream`, `get_stream_dataloader`, `get_hist_dataloaders`, `get_train_dataloaders`, `get_criterion`). Set `self.eval_metrics` with at least an `accuracy` entry from `apeiron.evaluation.metrics`.

### 3. Scaffold the files
- `examples/$1/__init__.py` — empty.
- `examples/$1/model.py` — `BaseModelHarness` subclass calling `super().__init__(cfg=cfg, model=<nn.Module>)`, implementing every abstract method, applying cumulative drift in `update_data_stream()`, and returning `(None, None)` from `get_hist_dataloaders()` on the first task.
- `examples/$1/utils.py` — dataset loaders, a deterministic drift transform, a `TransformedView` wrapper, and a `make_loader(...)` factory (follow the MNIST utils structure).
- Config at `$2` (default `examples/$1/$1.toml`) with `[model]`, `[data]` (`name = "$1"`), `[train]`, `[drift_detection]`, optional `[continual_learning]`, `[visualization]`. Read an existing config for the exact key set.

### 4. Register in the factory (in-repo)
Add a branch to `get_example()` in `examples/utils.py`:
```python
elif cfg.data.name == "$1":
    from examples.$1.model import <HarnessClass>
    return <HarnessClass>(cfg=cfg)
```

### 5. Validate
```bash
python -c "import tomllib; tomllib.load(open('$2','rb')); print('TOML OK')"
poetry run python -c "from examples.utils import get_example; print('factory OK')"
```
If `pretrained_path` is set, confirm the file exists; warn if missing (run will train from scratch).

### 6. Smoke-test before the full run
Run a tiny, fast pass to catch wiring errors cheaply, then **confirm with the user** before the real run:
```bash
poetry run python -m src.main --config $2 \
  --set train.max_iter=2 \
  --set drift_detection.max_stream_updates=2 \
  --set drift_detection.detection_interval=1 \
  --set device=cpu \
  --set logging.backend=none
```
If it fails, read the traceback, fix the harness/config, and re-run the smoke test. Do not proceed until it completes cleanly.

### 7. Full run and report
```bash
poetry run python -m src.main --config $2
```
Report drift events, final accuracy, and the output CSV path (the config's `visualization.input`). Note the package emits this CSV for inspection; it does not ship a built-in dashboard renderer.

## Notes
- Registration here uses the in-repo factory pattern. To instead drive apeiron from your *own* project without editing this repo, use the integrate-apeiron skill.
- The repo also has older `new-harness` / `new-config` skills covering pieces of this; they are stale (pre-`src/apeiron/` layout) and slated for refresh — prefer this skill.
