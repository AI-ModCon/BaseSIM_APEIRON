from __future__ import annotations
# mypy: ignore-errors

import copy
import gc
import random
import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from config.configuration import Config
from examples.matey.src.matey_batches import (
    MateyInputBatch,
    MateyLoaderAdapter as _MateyLoaderAdapter,
    MateyTargetBatch,
)
from model.torch_model_harness import BaseModelHarness

DEFAULT_MATEY_YAML = Path("examples/matey/Demo_SOLPS_vit.yaml")
DEFAULT_MATEY_PROFILE = "basic_config"
DEFAULT_SOLPS2DWION_ROOT = Path("examples/matey/dump/SOLPS2DwION")
SUPPORTED_UPDATE_MODES = {"base", "none"}
DEFAULT_NOISE_STD = 0.02


class _InputL2MetricModel(nn.Module):
    """Placeholder outer-loop model: returns per-sample L2 norm of inputs."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: MateyInputBatch) -> Tensor:
        if batch.is_graph:
            raise NotImplementedError(
                "Outer-loop placeholder model currently supports tensor inputs only."
            )
        if batch.input is None:
            raise RuntimeError("Matey input batch is missing `input` tensor.")

        # Keep autograd graph valid for `base` updater even without model params.
        inp = batch.input.float().detach().requires_grad_(True)
        if inp.ndim == 6 and inp.shape[1] >= 1:
            # Monitor only the first temporal slice t=0: CxDxHxW.
            inp_t0 = inp[:, 0, ...]
        else:
            inp_t0 = inp

        l2 = torch.linalg.vector_norm(
            inp_t0.reshape(inp_t0.shape[0], -1), ord=2, dim=1
        )
        return l2.unsqueeze(-1)


class _NoOpOptimizer:
    """Optimizer interface shim for parameter-free models."""

    def zero_grad(self, set_to_none: bool = False) -> None:
        return None

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is None:
            return None
        return closure()


class _NoisyMateyLoader:
    """Wrap a Matey adapter and inject Gaussian noise into `input` tensors."""

    def __init__(self, base_loader: _MateyLoaderAdapter, scale: float, seed: int):
        self._base_loader = base_loader
        self._scale = float(max(0.0, scale))
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))

    def __len__(self) -> int:
        return len(self._base_loader)

    def __iter__(self):
        for input_batch, target_batch in self._base_loader:
            if (
                self._scale <= 0.0
                or input_batch.is_graph
                or input_batch.input is None
            ):
                yield input_batch, target_batch
                continue

            noisy_input = input_batch.input.clone()
            if noisy_input.ndim == 6 and noisy_input.shape[1] >= 1:
                # Inject white noise only on the T=1 slice (index 0): CxDxHxW.
                noise = torch.randn(
                    noisy_input[:, 0, ...].shape,
                    dtype=noisy_input.dtype,
                    device=noisy_input.device,
                    generator=self._rng,
                ) * self._scale
                noisy_input[:, 0, ...] = noisy_input[:, 0, ...] + noise
            else:
                noise = torch.randn(
                    noisy_input.shape,
                    dtype=noisy_input.dtype,
                    device=noisy_input.device,
                    generator=self._rng,
                ) * self._scale
                noisy_input = noisy_input + noise

            yield replace(input_batch, input=noisy_input), target_batch


class MATEYHarness(BaseModelHarness):
    """Outer-loop Matey harness with synthetic input-noise drift."""

    def __init__(self, cfg: Config):
        self._assert_supported_update_mode(cfg)

        self._data_root = self._resolve_data_root(cfg)
        self._validate_data_root(self._data_root)

        modules = self._load_matey_modules()
        params = self._build_matey_params(cfg, modules["YParams"])
        self._configure_data_paths(params)

        self._modules = modules
        self._params = params

        model = _InputL2MetricModel()
        super().__init__(cfg=cfg, model=model)

        self.task_counter = 0
        self._cur_train_loader: Any | None = None
        self._cur_val_loader: Any | None = None

        self.eval_metrics = {
            "input_l2": self._input_l2_metric,
            "loss": self.get_criterion(),
        }
        self.higher_is_better = {"input_l2": False, "loss": False}

        #self._debug_input_tensor_shapes()

    def get_optmizer(self) -> Optimizer:
        return _NoOpOptimizer()

    def update_data_stream(self) -> None:
        self._dispose_current_loaders()
        stream_seed = int(self.cfg.seed + self.task_counter)
        self._set_stream_seed(stream_seed)

        train_loader, train_dataset, _ = self._build_loader(self._params, split="train")
        val_loader, val_dataset, _ = self._build_loader(self._params, split="val")

        scale = self._noise_std_for_stream(self.task_counter)

        self._cur_train_loader = _NoisyMateyLoader(
            _MateyLoaderAdapter(train_loader, train_dataset),
            scale=scale,
            seed=stream_seed,
        )
        self._cur_val_loader = _NoisyMateyLoader(
            _MateyLoaderAdapter(val_loader, val_dataset),
            scale=scale,
            seed=stream_seed + 10_000,
        )

        self.task_counter += 1

    def get_cur_data_loaders(self) -> tuple[Any, Any]:
        if self._cur_train_loader is None or self._cur_val_loader is None:
            raise RuntimeError(
                "Matey stream has not been initialized. Call update_data_stream() first."
            )
        return self._cur_train_loader, self._cur_val_loader

    def get_hist_data_loaders(self) -> tuple[None, None]:
        return None, None

    def get_criterion(self):
        def criterion(y_hat: Tensor, y: MateyTargetBatch | Tensor) -> Tensor:
            target_l2 = self._target_l2(y)
            return torch.mean((y_hat - target_l2) ** 2)

        return criterion

    def _unpack(
        self, batch: tuple[MateyInputBatch, MateyTargetBatch | Tensor]
    ) -> tuple[MateyInputBatch, Tensor]:
        x, y = batch
        return x, self._as_target_tensor(y)

    @staticmethod
    def _assert_supported_update_mode(cfg: Config) -> None:
        if cfg.continual_learning.update_mode not in SUPPORTED_UPDATE_MODES:
            raise NotImplementedError(
                "Matey outer-loop harness supports only continual_learning.update_mode "
                "in {'base', 'none'}."
            )

    @staticmethod
    def _resolve_data_root(cfg: Config) -> Path:
        raw = cfg.data.path.strip()
        path = Path(raw) if raw else DEFAULT_SOLPS2DWION_ROOT
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @staticmethod
    def _validate_data_root(data_root: Path) -> None:
        if not data_root.exists():
            raise FileNotFoundError(
                f"Matey data root path does not exist: {data_root}. "
                "Set [data].path to a local SOLPS2DwION dataset root, or leave it empty "
                "to use examples/matey/dump/SOLPS2DwION."
            )
        if not data_root.is_dir():
            raise NotADirectoryError(
                f"Matey data root path is not a directory: {data_root}"
            )
        if not any(data_root.rglob("*.nc")):
            raise FileNotFoundError(
                f"No netCDF files were found under Matey data root: {data_root}"
            )

        if not DEFAULT_MATEY_YAML.exists():
            raise FileNotFoundError(
                f"Required Matey YAML config was not found: {DEFAULT_MATEY_YAML}."
            )

    def _load_matey_modules(self) -> dict[str, Any]:
        self._install_optional_import_shims()
        try:
            # Import netCDF4 before h5py to avoid HDF5 library collision.
            import netCDF4 as _netCDF4  # noqa: F401

            from matey.data_utils.datasets import get_data_loader
            from matey.utils.YParams import YParams
        except ModuleNotFoundError:
            try:
                from examples.matey.MATEY.matey.data_utils.datasets import get_data_loader
                from examples.matey.MATEY.matey.utils.YParams import YParams
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Matey dependency import failed. Install with "
                    "`poetry install --extras matey` or ensure local MATEY module is importable."
                ) from exc

        return {
            "YParams": YParams,
            "get_data_loader": get_data_loader,
        }

    @staticmethod
    def _install_optional_import_shims() -> None:
        """
        MATEY's dataset registry imports graph/XGC modules eagerly.
        For outer-loop SOLPS-only usage, provide a minimal xgc_reader shim so
        optional graph dependencies do not block import.
        """
        if "xgc_reader" in sys.modules:
            return

        try:
            import xgc_reader as _xgc_reader  # noqa: F401

            return
        except ModuleNotFoundError:
            pass

        shim_pkg = types.ModuleType("xgc_reader")
        shim_base = types.ModuleType("xgc_reader.base")

        def _missing_xgc1(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "xgc_reader is not installed/available. "
                "It is only required for graph/XGC datasets, not SOLPS outer-loop runs."
            )

        shim_base.xgc1 = _missing_xgc1
        shim_pkg.base = shim_base
        sys.modules["xgc_reader"] = shim_pkg
        sys.modules["xgc_reader.base"] = shim_base

    def _build_matey_params(self, cfg: Config, yparams_cls: type[Any]) -> Any:
        params = yparams_cls(str(DEFAULT_MATEY_YAML), DEFAULT_MATEY_PROFILE)

        params.use_ddp = False
        params.use_fsdp = False
        params.log_to_screen = False
        params.log_to_wandb = False
        params.enable_sync = False
        params.profiling = False

        params.batch_size = max(1, int(cfg.train.batch_size))
        params.num_data_workers = max(0, int(cfg.train.num_workers))
        params.learning_rate = float(cfg.train.init_lr)

        return params

    @staticmethod
    def _as_config_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd()))
        except ValueError:
            return str(path.resolve())

    def _configure_data_paths(self, params: Any) -> None:
        entry = [self._as_config_path(self._data_root), "SOLPS2DwION", "", "tk-2D"]
        params.train_data_paths = [copy.deepcopy(entry)]
        params.valid_data_paths = [copy.deepcopy(entry)]

    def _debug_input_tensor_shapes(self, max_batches: int = 3) -> None:
        print("==== Matey outer-loop input shape probe ====", flush=True)
        try:
            raw_loader, mixed_dataset, _ = self._build_loader(self._params, split="train")
            probe_loader = _MateyLoaderAdapter(raw_loader, mixed_dataset)
            saw_batch = False

            for batch_idx, (input_batch, _) in enumerate(probe_loader):
                if batch_idx >= max_batches:
                    break
                saw_batch = True
                if input_batch.input is None:
                    print(
                        f"\tProbe batch {batch_idx}: no tensor `input` present.",
                        flush=True,
                    )
                else:
                    print(
                        f"\tProbe batch {batch_idx}: inps shape={tuple(input_batch.input.shape)}",
                        flush=True,
                    )

            if not saw_batch:
                print("\tProbe loader returned zero batches.", flush=True)
        except Exception as exc:
            print(f"Input shape probe failed: {exc}", flush=True)
        finally:
            gc.collect()

    def _dispose_current_loaders(self) -> None:
        if self._cur_train_loader is not None:
            del self._cur_train_loader
            self._cur_train_loader = None
        if self._cur_val_loader is not None:
            del self._cur_val_loader
            self._cur_val_loader = None
        gc.collect()

    @staticmethod
    def _set_stream_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _noise_std_for_stream(stream_idx: int) -> float:
        return DEFAULT_NOISE_STD * float(stream_idx)

    @contextmanager
    def _matey_single_worker_loader_patch(self, get_data_loader: Callable[..., Any]):
        """Patch MATEY's DataLoader symbol to support num_workers=0 safely."""
        module = sys.modules.get(get_data_loader.__module__)
        if module is None:
            yield
            return

        original_loader = getattr(module, "DataLoader", None)
        if original_loader is None:
            yield
            return

        def _patched_loader(*args: Any, **kwargs: Any):
            if int(kwargs.get("num_workers", 0)) == 0:
                kwargs["prefetch_factor"] = None
                kwargs["persistent_workers"] = False
            return original_loader(*args, **kwargs)

        setattr(module, "DataLoader", _patched_loader)
        try:
            yield
        finally:
            setattr(module, "DataLoader", original_loader)

    def _build_loader(self, params: Any, split: str) -> tuple[Any, Any, Any]:
        get_data_loader = self._modules["get_data_loader"]
        with self._matey_single_worker_loader_patch(get_data_loader):
            return get_data_loader(
                params,
                params.train_data_paths if split == "train" else params.valid_data_paths,
                False,
                split=split,
                train_offset=getattr(params, "embedding_offset", 0),
                global_rank=0,
                num_sp_groups=1,
                group_size=1,
            )

    @staticmethod
    def _as_target_tensor(target: MateyTargetBatch | Tensor) -> Tensor:
        return target.target if isinstance(target, MateyTargetBatch) else target

    @staticmethod
    def _target_l2(target: MateyTargetBatch | Tensor) -> Tensor:
        tar = MATEYHarness._as_target_tensor(target).float()
        l2 = torch.linalg.vector_norm(tar.reshape(tar.shape[0], -1), ord=2, dim=1)
        return l2.unsqueeze(-1)

    @staticmethod
    def _input_l2_metric(y_hat: Tensor, _y: MateyTargetBatch | Tensor) -> Tensor:
        return y_hat.mean()
