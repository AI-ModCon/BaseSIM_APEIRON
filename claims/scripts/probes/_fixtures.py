"""Shared fixtures for the behaviour probes.

This module is imported by the probe drivers and NOT by the layers under
test, so both sides of a comparison get byte-identical fixtures and differ
only in the implementation being driven. It lives beside the drivers so the
script directory puts it on sys.path; `apeiron` resolves from PYTHONPATH,
which the harness points at one tree or the other.

Everything here is deterministic. Fixed weights rather than seeded random
ones, fixed data rather than sampled, and no clock reading -- a probe that
moves between runs cannot distinguish an implementation difference from its
own noise, which is the failure recorded four times over in A38.
"""
import io
import json
import pathlib
import sys
import tempfile

import torch
import torch.nn as nn

#: The probe's result goes to the real stdout; everything the layer PRINTS
#: goes nowhere. The monitor and trainer log through a console logger that
#: writes to stdout, which would interleave with the JSON this driver emits
#: and make the harness report "did not emit JSON" for both sides. Swapped
#: at import time, before any layer code builds a logging handler, because
#: `logging.StreamHandler(sys.stdout)` captures the stream object it is
#: given rather than looking it up later.
_REAL_STDOUT = sys.stdout


def silence() -> None:
    """Send anything the layer prints to a throwaway buffer."""
    sys.stdout = io.StringIO()

#: Written once per probe process; `scrub` replaces these paths in output so
#: two runs in two different temporary directories do not read as a
#: divergence (the first A38 false positive).
_SCRUB: list[str] = []

MINIMAL_TOML = """
seed = 1
device = "cpu"

[model]
name = "MNIST_CNN"
pretrained_path = ""

[data]
name = "mnist"
path = "/tmp/data"

[train]
batch_size = 4
num_workers = 0
init_lr = 0.01

[drift_detection]
detector_name = "ADWINDetector"
detection_interval = 10
"""


def tmpdir() -> pathlib.Path:
    """A temporary directory whose path is scrubbed from probe output."""
    path = pathlib.Path(tempfile.mkdtemp())
    _SCRUB.append(str(path))
    return path


def scrub(text: str) -> str:
    for path in _SCRUB:
        text = text.replace(path, "<tmp>")
    return text


def show(value) -> str:
    """One canonical string per observable, so the two sides are comparable."""
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, torch.Tensor):
        flat = value.detach().flatten().tolist()
        return json.dumps([round(float(x), 5) for x in flat])
    if isinstance(value, float):
        return repr(round(value, 5))
    if isinstance(value, dict):
        return json.dumps({str(k): show(v) for k, v in sorted(
            value.items(), key=lambda kv: str(kv[0]))})
    if isinstance(value, (list, tuple)):
        return json.dumps([show(v) for v in value])
    if isinstance(value, set):
        return json.dumps(sorted(show(v) for v in value))
    rendered = repr(value)
    if " at 0x" in rendered:
        # A default repr carrying a memory address. These vary per process,
        # so the harness would report the case as unstable and the real
        # observable would never be compared at all. This happened: a probe
        # rendered `runner.drift_windows` without calling it, and four cases
        # became four bound-method addresses. Loud beats silent.
        raise AssertionError(
            f"the probe is rendering an object's identity, not its behaviour: "
            f"{rendered[:120]}. Call the method, or pick an attribute that "
            f"has a stable value.")
    return rendered


def attempt(fn, *args, **kwargs) -> str:
    try:
        return show(fn(*args, **kwargs))
    except BaseException as exc:                     # noqa: BLE001 — probed
        return show(exc)


def emit(out: dict) -> None:
    """Write the probe's result to the real stdout, temp paths neutralised."""
    json.dump({k: scrub(v) for k, v in out.items()}, _REAL_STDOUT,
              sort_keys=True)
    _REAL_STDOUT.flush()


def make_cfg(extra: str = "", overrides=()):
    """A Config, built through the layer's own loader so nothing is faked."""
    from apeiron.config.configuration import build_config
    path = tmpdir() / "cfg.toml"
    path.write_text(MINIMAL_TOML + extra)
    return build_config(["--config", str(path)] + list(overrides))


class Tiny(nn.Module):
    """Four inputs, three classes, weights fixed by construction."""

    def __init__(self, in_dim: int = 4, out_dim: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            self.fc.weight.copy_(
                torch.linspace(-0.5, 0.5, in_dim * out_dim).view(out_dim, in_dim))
            self.fc.bias.copy_(torch.linspace(-0.1, 0.1, out_dim))

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))


def fixed_loader(n: int = 8, batch: int = 4, in_dim: int = 4,
                 classes: int = 3, offset: float = 0.0):
    """A DataLoader over deterministic tensors, never shuffled."""
    xs = (torch.linspace(-1, 1, max(n, 1) * in_dim)[:n * in_dim]
          .view(max(n, 0), in_dim) + offset) if n else torch.zeros(0, in_dim)
    ys = torch.arange(max(n, 0)) % classes
    data = torch.utils.data.TensorDataset(xs, ys)
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=False)


def params(model: nn.Module) -> str:
    """Every parameter, rounded -- the observable for "did training move it"."""
    return json.dumps([
        [name, [round(float(x), 5) for x in p.detach().flatten().tolist()]]
        for name, p in sorted(model.named_parameters())])


def make_harness(cfg=None, n: int = 8, batch: int = 4):
    """A concrete BaseModelHarness over fixed data.

    Defined here rather than in the layers so both sides drive the same
    subclass. Only the abstract surface is implemented; everything the base
    class provides is what the probe is actually measuring.
    """
    from apeiron.model.torch_model_harness import BaseModelHarness

    class FixedHarness(BaseModelHarness):
        def __init__(self, cfg, model):
            super().__init__(cfg, model)
            self._extended = 0
            # The base class assigns `self.eval_metrics = {}`, so it is an
            # instance attribute and a subclass property would shadow it.
            # Metrics are plain functions of (output, target) and return a
            # float, so a parameter-level comparison stays exact.
            self.eval_metrics = {
                "accuracy": lambda out, tgt: float(
                    (out.argmax(dim=1) == tgt).float().mean() * 100.0),
                "mean_logit": lambda out, tgt: float(out.mean()),
            }

        def get_stream_dataloader(self):
            return fixed_loader(n, batch, offset=0.1 * self._extended)

        def get_train_dataloaders(self):
            return fixed_loader(n, batch), fixed_loader(n, batch, offset=0.5)

        def get_hist_dataloaders(self):
            return fixed_loader(n, batch, offset=-0.5), \
                fixed_loader(n, batch, offset=-0.25)

        def update_data_stream(self):
            self._extended += 1

        def get_criterion(self):
            return nn.CrossEntropyLoss()

        def get_optimizer(self):
            # SGD without momentum: one step is a closed-form function of the
            # gradient, so a parameter comparison after training is exact
            # rather than an accumulation of optimiser state.
            return torch.optim.SGD(self.model.parameters(),
                                   lr=self.cfg.train.init_lr)

    cfg = cfg if cfg is not None else make_cfg()
    return FixedHarness(cfg, Tiny())
