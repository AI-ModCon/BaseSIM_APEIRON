from __future__ import annotations
# mypy: ignore-errors

import copy
import gc
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from config.configuration import Config
from examples.matey.src.matey_batches import (
    MateyInputBatch,
    MateyLoaderAdapter as _MateyLoaderAdapter,
    MateyModelAdapter as _MateyModelAdapter,
    MateyTargetBatch,
)
from examples.matey.src.solps_split import SolpsStagedSplit, stage_solps_split
from logger import get_logger
from model.torch_model_harness import BaseModelHarness

DEFAULT_MATEY_YAML = Path("examples/matey/Demo_SOLPS_vit.yaml")
DEFAULT_MATEY_PROFILE = "basic_config"
DEFAULT_MATEY_TRAIN_VAL_TEST = (0.7, 0.15, 0.15)
DEFAULT_SOLPS_CACHE_ROOT = Path("output/matey_split_cache")
MATEY_GIT_COMMIT = "4e615bb5c86024632e386153bfbed028b38a8262"
MATEY_GIT_URL = f"git+ssh://git@github.com/FusionFM/MATEY.git@{MATEY_GIT_COMMIT}"
SUPPORTED_UPDATE_MODES = {"base", "none"}


class MATEYHarness(BaseModelHarness):
    def __init__(self, cfg: Config):
        self._assert_supported_update_mode(cfg)
        self._solps_split: SolpsStagedSplit | None = None
        self._solps_split_logged = False
        self._split_seed = int(cfg.seed)

        self._data_root = self._resolve_data_root(cfg)
        self._validate_data_root(self._data_root)

        modules = self._load_matey_modules()
        params = self._build_matey_params(cfg, modules["YParams"])
        self._configure_data_split(params)
        matey_model = self._build_matey_model(cfg, params, modules)

        self._adapter_model = _MateyModelAdapter(
            matey_model=matey_model,
            params=params,
            forward_options_cls=modules["ForwardOptionsBase"],
            rearrange_fn=modules["rearrange"],
            autoregressive_rollout_fn=modules["autoregressive_rollout"],
            determine_turt_levels_fn=modules["determine_turt_levels"],
        )
        super().__init__(cfg=cfg, model=self._adapter_model)

        self._modules = modules
        self._params = params

        self.task_counter = 0
        self._cur_train_loader: _MateyLoaderAdapter | None = None
        self._cur_val_loader: _MateyLoaderAdapter | None = None

        self.eval_metrics = {
            "nrmse": self._nrmse_metric,
            "rmse": self._rmse_metric,
            "loss": self.get_criterion(),
        }
        self.higher_is_better = {"nrmse": False, "rmse": False, "loss": False}

    def get_optmizer(self) -> Optimizer:
        optimizer_name = str(getattr(self._params, "optimizer", "AdamW")).lower()
        lr = float(getattr(self._params, "learning_rate", self.cfg.train.init_lr))
        weight_decay = float(getattr(self._params, "weight_decay", 0.0))

        add_weight_decay = self._modules["add_weight_decay"]
        param_groups = add_weight_decay(self._adapter_model.matey_model, weight_decay)

        if optimizer_name == "dadaptadam":
            dadapt_cls = self._modules.get("DAdaptAdam")
            if dadapt_cls is None:
                raise RuntimeError(
                    "MATEY optimizer is configured as DAdaptAdam but "
                    "`dadaptation` is not installed in this environment."
                )
            return cast(
                Optimizer,
                dadapt_cls(
                    param_groups, lr=1.0, growth_rate=1.05, log_every=100, decouple=True
                ),
            )

        if optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    def update_data_stream(self) -> None:
        self._dispose_current_loaders()
        self._set_stream_seed(self.cfg.seed + self.task_counter)
        self._log_solps_split_details()

        train_params = self._params_for_loader_split("train")
        val_params = self._params_for_loader_split("val")
        train_loader, train_dataset, _ = self._build_loader(train_params, split="train")
        val_loader, val_dataset, _ = self._build_loader(val_params, split="val")

        self._cur_train_loader = _MateyLoaderAdapter(train_loader, train_dataset)
        self._cur_val_loader = _MateyLoaderAdapter(val_loader, val_dataset)

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
        def criterion(y_hat: Tensor, y: MateyTargetBatch) -> Tensor:
            target = self._select_target_tensor(
                y, self._adapter_model.last_rollout_steps
            )
            return self._compute_nrmse(y_hat, target)

        return criterion

    def _unpack(
        self, batch: tuple[MateyInputBatch, MateyTargetBatch]
    ) -> tuple[MateyInputBatch, MateyTargetBatch]:
        return batch

    @staticmethod
    def _assert_supported_update_mode(cfg: Config) -> None:
        if cfg.continual_learning.update_mode not in SUPPORTED_UPDATE_MODES:
            raise NotImplementedError(
                "Matey phase-1 harness supports only continual_learning.update_mode "
                "in {'base', 'none'}. Modes {'jvp_reg', 'ewc_online', 'kfac_online'} "
                "are not supported yet."
            )

    @staticmethod
    def _resolve_data_root(cfg: Config) -> Path:
        raw = cfg.data.path.strip()
        if not raw:
            raise ValueError(
                "Matey data path is empty. Set [data].path to your local SOLPS "
                "dataset root containing 'train/' and 'valid/' directories."
            )
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @staticmethod
    def _validate_data_root(data_root: Path) -> None:
        if not data_root.exists():
            raise FileNotFoundError(
                f"Matey data root path does not exist: {data_root}. "
                "Set [data].path to your local SOLPS dataset root path."
            )
        if not data_root.is_dir():
            raise NotADirectoryError(
                f"Matey data root path is not a directory: {data_root}"
            )

        if not DEFAULT_MATEY_YAML.exists():
            raise FileNotFoundError(
                f"Required Matey YAML config was not found: {DEFAULT_MATEY_YAML}."
            )

    def _load_matey_modules(self) -> dict[str, Any]:
        try:
            # Import netCDF4 before h5py to avoid HDF5 library collision.
            # Both ship their own libhdf5; whichever loads first wins.
            import netCDF4 as _netCDF4  # noqa: F401

            from einops import rearrange
            from matey.data_utils.datasets import get_data_loader
            from matey.models.avit import build_avit
            from matey.models.svit import build_svit
            from matey.models.turbt import build_turbt
            from matey.models.vit import build_vit
            from matey.utils.YParams import YParams
            from matey.utils.distributed_utils import add_weight_decay
            from matey.utils.distributed_utils import determine_turt_levels
            from matey.utils.forward_options import ForwardOptionsBase
            from matey.utils.training_utils import autoregressive_rollout
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Matey dependency import failed. Ensure MATEY requirements are "
                "installed in the active environment (for example: "
                f"`poetry install --extras matey` or `pip install \"matey @ {MATEY_GIT_URL}\"`) "
                "and that `[data].path` points to your local SOLPS dataset root."
            ) from exc

        dadapt = None
        try:
            from dadaptation import DAdaptAdam as _DAdaptAdam

            dadapt = _DAdaptAdam
        except ModuleNotFoundError:
            dadapt = None

        return {
            "YParams": YParams,
            "get_data_loader": get_data_loader,
            "build_avit": build_avit,
            "build_svit": build_svit,
            "build_vit": build_vit,
            "build_turbt": build_turbt,
            "add_weight_decay": add_weight_decay,
            "determine_turt_levels": determine_turt_levels,
            "ForwardOptionsBase": ForwardOptionsBase,
            "autoregressive_rollout": autoregressive_rollout,
            "rearrange": rearrange,
            "DAdaptAdam": dadapt,
        }

    def _build_matey_params(self, cfg: Config, yparams_cls: type[Any]) -> Any:
        yaml_path = DEFAULT_MATEY_YAML
        params = yparams_cls(str(yaml_path), DEFAULT_MATEY_PROFILE)

        params.use_ddp = False
        params.use_fsdp = False
        params.log_to_screen = False
        params.log_to_wandb = False
        params.enable_sync = False
        params.profiling = False

        params.batch_size = max(1, int(cfg.train.batch_size))
        params.num_data_workers = max(0, int(cfg.train.num_workers))
        params.learning_rate = float(cfg.train.init_lr)

        if not hasattr(params, "weight_decay"):
            params.weight_decay = 0.0
        if not hasattr(params, "optimizer"):
            params.optimizer = "AdamW"
        if not hasattr(params, "embedding_offset"):
            params.embedding_offset = 0

        return params

    @staticmethod
    def _normalize_path_entry(entry: Any) -> list[Any]:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(
                "MATEY data path entries must be list/tuple values with at least "
                "[path, dataset_type, ...]."
            )
        return list(entry)

    @staticmethod
    def _to_abs_path(path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @staticmethod
    def _as_config_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd()))
        except ValueError:
            return str(path.resolve())

    def _configure_solps_staged_pool(self, params: Any) -> None:
        train_paths = [
            self._normalize_path_entry(entry)
            for entry in getattr(params, "train_data_paths", [])
        ]
        val_paths = [
            self._normalize_path_entry(entry)
            for entry in getattr(params, "valid_data_paths", [])
        ]

        if not train_paths or not val_paths:
            raise ValueError(
                "MATEY SOLPS split requires non-empty train_data_paths and "
                "valid_data_paths."
            )

        train_solps = [entry for entry in train_paths if str(entry[1]) == "SOLPS2D"]
        val_solps = [entry for entry in val_paths if str(entry[1]) == "SOLPS2D"]

        if not train_solps and not val_solps:
            return

        if len(train_solps) != len(train_paths) or len(val_solps) != len(val_paths):
            raise ValueError(
                "MATEY SOLPS split mode does not support mixing SOLPS2D with "
                "non-SOLPS datasets in train/valid data paths."
            )

        signatures = {tuple(entry[1:]) for entry in (train_solps + val_solps)}
        if len(signatures) != 1:
            raise ValueError(
                "MATEY SOLPS split requires matching dataset metadata across "
                "train_data_paths and valid_data_paths (dataset/include/tk-head)."
            )

        src_paths = [
            self._to_abs_path(str(entry[0])) for entry in train_solps + val_solps
        ]
        split_view = stage_solps_split(
            source_roots=src_paths,
            ratios=DEFAULT_MATEY_TRAIN_VAL_TEST,
            seed=self._split_seed,
            cache_root=DEFAULT_SOLPS_CACHE_ROOT,
        )
        self._solps_split = split_view
        signature = list(next(iter(signatures)))
        train_entry = [self._as_config_path(split_view.train_dir), *signature]
        val_entry = [self._as_config_path(split_view.val_dir), *signature]
        params.train_data_paths = [copy.deepcopy(train_entry)]
        params.valid_data_paths = [copy.deepcopy(val_entry)]

    def _configure_user_data_paths(self, params: Any) -> None:
        train_dir = self._data_root / "train"
        val_dir = self._data_root / "valid"

        # Keep compatibility with non-SOLPS test fixtures that mock custom paths.
        if not train_dir.exists() and not val_dir.exists():
            return

        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(
                "Matey data root must contain both 'train/' and 'valid/' directories. "
                f"Missing paths: train={train_dir.exists()}, valid={val_dir.exists()}."
            )

        params.train_data_paths = [[self._as_config_path(train_dir), "SOLPS2D", "", "tk-2D"]]
        params.valid_data_paths = [[self._as_config_path(val_dir), "SOLPS2D", "", "tk-2D"]]

    def _configure_data_split(self, params: Any) -> None:
        self._configure_user_data_paths(params)
        params.train_val_test = list(DEFAULT_MATEY_TRAIN_VAL_TEST)
        self._configure_solps_staged_pool(params)

    def _log_solps_split_details(self) -> None:
        if self._solps_split is None or self._solps_split_logged:
            return

        logger = get_logger()
        counts = self._solps_split.counts
        logger.info("==== MATEY SOLPS staged split ready ====", level=0)
        logger.info(f"\tCache dir: {self._solps_split.cache_dir}", level=1)
        logger.info(
            f"\tSplit counts train/val/test: {counts['train']}/{counts['val']}/{counts['test']}",
            level=1,
        )
        logger.info(
            f"\tSplit cache reused: {self._solps_split.reused_cache}",
            level=1,
        )
        self._solps_split_logged = True

    def _params_for_loader_split(self, split: str) -> Any:
        loader_params = copy.deepcopy(self._params)
        if self._solps_split is None:
            return loader_params

        if split == "train":
            loader_params.train_val_test = [1.0, 0.0, 0.0]
        elif split == "val":
            loader_params.train_val_test = [0.0, 1.0, 0.0]
        else:
            loader_params.train_val_test = [0.0, 0.0, 1.0]
        return loader_params

    @staticmethod
    def _build_matey_model(
        cfg: Config, params: Any, modules: dict[str, Any]
    ) -> nn.Module:
        model_type = str(getattr(params, "model_type", "vit_all2all"))
        if model_type == "avit":
            model = modules["build_avit"](params)
        elif model_type == "svit":
            model = modules["build_svit"](params)
        elif model_type == "turbt":
            model = modules["build_turbt"](params)
        else:
            model = modules["build_vit"](params)

        MATEYHarness._load_pretrained_weights_if_available(
            model=model,
            pretrained_path=cfg.model.pretrained_path,
        )

        if bool(getattr(params, "compile", False)):
            model = torch.compile(model)

        return model

    @staticmethod
    def _load_pretrained_weights_if_available(
        model: nn.Module, pretrained_path: str
    ) -> None:
        raw_path = str(pretrained_path).strip()
        if not raw_path:
            return

        checkpoint_path = Path(raw_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"MATEY pretrained checkpoint not found: {checkpoint_path}"
            )
        if checkpoint_path.is_dir():
            raise ValueError(
                "MATEY pretrained checkpoint path must be a file, not a directory: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = MATEYHarness._extract_model_state_dict(checkpoint)

        attempts = [
            ("raw", state_dict),
            ("strip_module_prefix", MATEYHarness._strip_prefix(state_dict, "module.")),
            (
                "strip_orig_mod_prefix",
                MATEYHarness._strip_prefix(state_dict, "_orig_mod."),
            ),
            (
                "strip_module_then_orig_mod",
                MATEYHarness._strip_prefix(
                    MATEYHarness._strip_prefix(state_dict, "module."),
                    "_orig_mod.",
                ),
            ),
            (
                "strip_orig_mod_then_module",
                MATEYHarness._strip_prefix(
                    MATEYHarness._strip_prefix(state_dict, "_orig_mod."),
                    "module.",
                ),
            ),
        ]

        logger = get_logger()
        last_error: RuntimeError | None = None
        for transform_name, candidate in attempts:
            try:
                model.load_state_dict(candidate)
                logger.info(
                    f"Loaded MATEY pretrained weights: {checkpoint_path}",
                    level=0,
                )
                if transform_name != "raw":
                    logger.info(
                        f"\tApplied checkpoint key transform: {transform_name}",
                        level=1,
                    )
                return
            except RuntimeError as exc:
                last_error = exc

        raise RuntimeError(
            "Failed to load MATEY pretrained weights from "
            f"{checkpoint_path}. Last error: {last_error}"
        )

    @staticmethod
    def _extract_model_state_dict(checkpoint: Any) -> dict[str, Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("model_state", "state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return cast(dict[str, Tensor], value)

            # Raw state_dict case (all tensor-ish values)
            if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
                return cast(dict[str, Tensor], checkpoint)

        raise ValueError(
            "Unsupported MATEY checkpoint format. Expected a state_dict or a dict "
            "containing one of: model_state, state_dict, model_state_dict, model."
        )

    @staticmethod
    def _strip_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
        if not prefix:
            return state_dict
        plen = len(prefix)
        return {
            (key[plen:] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }

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
                params.train_data_paths
                if split == "train"
                else params.valid_data_paths,
                False,
                split=split,
                train_offset=getattr(params, "embedding_offset", 0),
                global_rank=0,
                num_sp_groups=1,
                group_size=1,
            )

    def _select_target_tensor(
        self, target: MateyTargetBatch | Tensor, rollout_steps: int | None
    ) -> Tensor:
        tar = target.target if isinstance(target, MateyTargetBatch) else target
        if tar.ndim == 6:
            step = rollout_steps
            if step is None and isinstance(target, MateyTargetBatch):
                if target.leadtime is not None and target.leadtime.numel() > 0:
                    step = int(target.leadtime.min().item())
            if step is None:
                step = 1
            step = max(1, min(int(step), tar.shape[1]))
            tar = tar[:, step - 1, ...]
        return tar

    @staticmethod
    def _compute_nrmse(pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-7
        if pred.ndim == 2:
            num = (pred - target).pow(2).mean(dim=0)
            den = target.pow(2).mean(dim=0) + eps
            return torch.sqrt((num / den).mean())

        spatial_dims = tuple(range(2, pred.ndim))
        num = (pred - target).pow(2).mean(spatial_dims)
        den = target.pow(2).mean(spatial_dims) + eps
        return torch.sqrt((num / den).mean())

    @staticmethod
    def _compute_rmse(pred: Tensor, target: Tensor) -> Tensor:
        if pred.ndim == 2:
            return (pred - target).pow(2).mean(dim=0).sqrt().mean()

        spatial_dims = tuple(range(2, pred.ndim))
        return (pred - target).pow(2).mean(spatial_dims).sqrt().mean()

    def _nrmse_metric(self, y_hat: Tensor, y: MateyTargetBatch) -> Tensor:
        target = self._select_target_tensor(y, self._adapter_model.last_rollout_steps)
        return self._compute_nrmse(y_hat, target)

    def _rmse_metric(self, y_hat: Tensor, y: MateyTargetBatch) -> Tensor:
        target = self._select_target_tensor(y, self._adapter_model.last_rollout_steps)
        return self._compute_rmse(y_hat, target)
