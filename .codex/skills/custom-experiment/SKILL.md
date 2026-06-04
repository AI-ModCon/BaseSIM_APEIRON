---
name: custom-experiment
description: Scaffold and run an apeiron experiment for the user's own dataset and model. Use when the user wants to bring custom data or architecture beyond shipped examples, create a model harness, write a config, register it in the example factory, smoke-test it, and run the full experiment. For bundled demos, use explore-examples. For integrating apeiron into an existing external training loop, use integrate-apeiron.
metadata:
  short-description: Build a custom apeiron experiment
---

# Custom Experiment

Scaffold and run an apeiron experiment on the user's own data and model.

## Inputs

- Short name: lowercase identifier such as `fashionmnist` or `mytabular`. Use it for `examples/<name>/` and `data.name`.
- Optional config output path: default to `examples/<name>/<name>.toml`.

Ask only for missing details that cannot be inferred:

- dataset source and loading method
- model architecture, input shape, and number of classes or outputs
- drift simulation to apply to the stream
- optional pretrained weights path
- starting drift detector and continual-learning updater, defaulting to `ADWINDetector` and `base`

## Procedure

### 1. Read Current Patterns

Mirror live source instead of assuming signatures:

```bash
cat src/apeiron/model/torch_model_harness.py
cat examples/mnist/model.py
cat examples/mnist/utils.py
cat examples/utils.py
grep -nA6 "class .*Cfg" src/apeiron/config/configuration.py
```

Implement exactly the abstract methods declared by the current harness ABC. Preserve known current spelling such as `get_optmizer` if the source still declares it that way.

Set `self.eval_metrics` with at least an `accuracy` entry from `apeiron.evaluation.metrics` when the task is classification.

### 2. Scaffold Files

Create:

- `examples/<name>/__init__.py`
- `examples/<name>/model.py`
- `examples/<name>/utils.py`
- the config TOML at the requested output path or `examples/<name>/<name>.toml`

`model.py` should:

- define a `BaseModelHarness` subclass
- call `super().__init__(cfg=cfg, model=<nn.Module>)`
- implement every abstract method from the current ABC
- apply cumulative drift in `update_data_stream()`
- return `(None, None)` from `get_hist_dataloaders()` for the first task when no history exists

`utils.py` should:

- load the dataset
- include deterministic drift transforms
- provide a lightweight transformed-view wrapper
- expose a `make_loader(...)` helper following the MNIST example pattern

The TOML config should follow existing examples for the exact key set and include:

- `[model]`
- `[data]` with `name = "<name>"`
- `[train]`
- `[drift_detection]`
- `[continual_learning]` when needed
- `[visualization]`

### 3. Register In The Example Factory

Add a branch to `get_example()` in `examples/utils.py`:

```python
elif cfg.data.name == "<name>":
    from examples.<name>.model import <HarnessClass>
    return <HarnessClass>(cfg=cfg)
```

Match the surrounding factory style exactly.

### 4. Validate

Run:

```bash
python -c "import tomllib; tomllib.load(open('<config_path>', 'rb')); print('TOML OK')"
poetry run python -c "from examples.utils import get_example; print('factory OK')"
```

If `pretrained_path` is configured, confirm the file exists. Warn if it is missing and make the harness tolerate training from scratch when possible.

### 5. Smoke-Test

Run a small CPU-only smoke test before any full run:

```bash
poetry run python -m src.main --config <config_path> \
  --set train.max_iter=2 \
  --set drift_detection.max_stream_updates=2 \
  --set drift_detection.detection_interval=1 \
  --set device=cpu \
  --set logging.backend=none
```

If it fails, read the traceback, fix the harness or config, and re-run the smoke test until it completes.

Confirm with the user before starting a full experiment run.

### 6. Full Run And Report

Run:

```bash
poetry run python -m src.main --config <config_path>
```

Report drift events, final accuracy or metric, and the output CSV path from `visualization.input`.

## Notes

- This skill uses the in-repo example factory pattern.
- To wire apeiron into an existing project without adopting the example runner, use `integrate-apeiron`.
- Older piecewise Claude skills such as `new-harness` or `new-config` may be stale against the current `src/apeiron/` layout.
