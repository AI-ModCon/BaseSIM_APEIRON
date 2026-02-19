from __future__ import annotations

import copy
import gc
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from config.configuration import Config
from model.torch_model_harness import BaseModelHarness

DEFAULT_MATEY_YAML = Path("examples/matey/Demo_SOLPS_vit.yaml")
DEFAULT_MATEY_PROFILE = "basic_config"
SUPPORTED_UPDATE_MODES = {"base", "none"}


def _move_to_device(x: Any, device: str | torch.device) -> Any:
    if x is None:
        return None
    if hasattr(x, "to"):
        return x.to(device)
    return x


@dataclass(frozen=True)
class MateyInputBatch:
    input: Tensor | None = None
    graph: Any = None
    field_labels: Tensor | None = None
    bcs: Tensor | None = None
    leadtime: Tensor | None = None
    cond_field_labels: Tensor | None = None
    cond_fields: Tensor | None = None
    cond_input: Tensor | None = None
    field_labels_out: Tensor | None = None
    tkhead_name: str | None = None
    blockdict: dict[str, Any] | None = None
    is_graph: bool = False

    def to(self, device: str | torch.device) -> MateyInputBatch:
        return MateyInputBatch(
            input=cast(Optional[Tensor], _move_to_device(self.input, device)),
            graph=_move_to_device(self.graph, device),
            field_labels=cast(
                Optional[Tensor], _move_to_device(self.field_labels, device)
            ),
            bcs=cast(Optional[Tensor], _move_to_device(self.bcs, device)),
            leadtime=cast(Optional[Tensor], _move_to_device(self.leadtime, device)),
            cond_field_labels=cast(
                Optional[Tensor], _move_to_device(self.cond_field_labels, device)
            ),
            cond_fields=cast(
                Optional[Tensor], _move_to_device(self.cond_fields, device)
            ),
            cond_input=cast(Optional[Tensor], _move_to_device(self.cond_input, device)),
            field_labels_out=cast(
                Optional[Tensor], _move_to_device(self.field_labels_out, device)
            ),
            tkhead_name=self.tkhead_name,
            blockdict=copy.deepcopy(self.blockdict),
            is_graph=self.is_graph,
        )


@dataclass(frozen=True)
class MateyTargetBatch:
    target: Tensor
    leadtime: Tensor | None = None
    is_graph: bool = False

    def to(self, device: str | torch.device) -> MateyTargetBatch:
        return MateyTargetBatch(
            target=cast(Tensor, _move_to_device(self.target, device)),
            leadtime=cast(Optional[Tensor], _move_to_device(self.leadtime, device)),
            is_graph=self.is_graph,
        )

    @property
    def shape(self) -> torch.Size:
        return self.target.shape


class _MateyLoaderAdapter:
    def __init__(self, raw_loader: Any, mixed_dataset: Any):
        self._raw_loader = raw_loader
        self._mixed_dataset = mixed_dataset

    def __len__(self) -> int:
        return len(self._raw_loader)

    def __iter__(self):
        for raw_batch in self._raw_loader:
            yield self._convert_batch(raw_batch)

    def _convert_batch(
        self, raw_batch: dict[str, Any]
    ) -> tuple[MateyInputBatch, MateyTargetBatch]:
        dset_idx_obj = raw_batch.get("dset_idx")
        if isinstance(dset_idx_obj, torch.Tensor):
            dset_idx = int(dset_idx_obj.flatten()[0].item())
        else:
            dset_idx = int(dset_idx_obj)

        sub_dset = self._mixed_dataset.sub_dsets[dset_idx]
        tkhead_name = cast(str | None, getattr(sub_dset, "tkhead_name", None))
        blockdict = copy.deepcopy(getattr(sub_dset, "blockdict", None))

        field_labels = cast(Tensor, raw_batch["field_labels"])
        bcs = cast(Tensor, raw_batch["bcs"])
        leadtime = cast(Optional[Tensor], raw_batch.get("leadtime"))
        cond_field_labels = cast(Optional[Tensor], raw_batch.get("cond_field_labels"))
        cond_fields = cast(Optional[Tensor], raw_batch.get("cond_fields"))
        cond_input = cast(Optional[Tensor], raw_batch.get("cond_input"))

        if "graph" in raw_batch:
            graph = raw_batch["graph"]
            graph_leadtime = getattr(graph, "leadtime", leadtime)
            input_batch = MateyInputBatch(
                graph=graph,
                field_labels=field_labels,
                field_labels_out=cast(
                    Optional[Tensor], raw_batch.get("field_labels_out")
                ),
                bcs=bcs,
                leadtime=graph_leadtime,
                cond_field_labels=cond_field_labels,
                cond_fields=cond_fields,
                cond_input=cond_input,
                tkhead_name=tkhead_name,
                blockdict=blockdict,
                is_graph=True,
            )
            target_batch = MateyTargetBatch(
                target=cast(Tensor, graph.y),
                leadtime=cast(Optional[Tensor], graph_leadtime),
                is_graph=True,
            )
            return input_batch, target_batch

        input_batch = MateyInputBatch(
            input=cast(Tensor, raw_batch["input"]),
            field_labels=field_labels,
            field_labels_out=field_labels,
            bcs=bcs,
            leadtime=leadtime,
            cond_field_labels=cond_field_labels,
            cond_fields=cond_fields,
            cond_input=cond_input,
            tkhead_name=tkhead_name,
            blockdict=blockdict,
            is_graph=False,
        )
        target_batch = MateyTargetBatch(
            target=cast(Tensor, raw_batch["label"]),
            leadtime=leadtime,
            is_graph=False,
        )
        return input_batch, target_batch


class _MateyModelAdapter(nn.Module):
    def __init__(
        self,
        matey_model: nn.Module,
        params: Any,
        forward_options_cls: type[Any],
        rearrange_fn: Callable[..., Any],
        autoregressive_rollout_fn: Callable[..., Any],
        determine_turt_levels_fn: Callable[..., Any] | None = None,
    ):
        super().__init__()
        self.matey_model = matey_model
        self.params = params
        self._forward_options_cls = forward_options_cls
        self._rearrange = rearrange_fn
        self._autoregressive_rollout = autoregressive_rollout_fn
        self._determine_turt_levels = determine_turt_levels_fn
        self.last_rollout_steps: int | None = None

    def forward(self, batch: MateyInputBatch) -> Tensor:
        if batch.field_labels is None or batch.bcs is None:
            raise RuntimeError("Matey input batch is missing required fields.")

        cond_dict: dict[str, Tensor] = {}
        if batch.cond_field_labels is not None and batch.cond_fields is not None:
            cond_dict["labels"] = batch.cond_field_labels
            cond_dict["fields"] = self._rearrange(
                batch.cond_fields, "b t c d h w -> t b c d h w"
            )

        leadtime = batch.leadtime
        if leadtime is None:
            leadtime = torch.ones(
                (1, 1), dtype=torch.long, device=batch.field_labels.device
            )

        imod = 0
        hierarchical = getattr(self.params, "hierarchical", None)
        if isinstance(hierarchical, dict):
            imod = int(hierarchical.get("nlevels", 1) - 1)

        imod_bottom = 0
        if (
            not batch.is_graph
            and imod > 0
            and self._determine_turt_levels is not None
            and batch.tkhead_name is not None
            and batch.input is not None
        ):
            tk_size = self.matey_model.tokenizer_heads_params[batch.tkhead_name][-1]
            imod_bottom = int(
                self._determine_turt_levels(tk_size, batch.input.shape[-3:], imod)
            )

        opts = self._forward_options_cls(
            imod=imod,
            imod_bottom=imod_bottom,
            tkhead_name=batch.tkhead_name,
            sequence_parallel_group=None,
            leadtime=leadtime,
            blockdict=copy.deepcopy(batch.blockdict),
            cond_dict=copy.deepcopy(cond_dict),
            cond_input=batch.cond_input,
            isgraph=batch.is_graph,
            field_labels_out=(
                batch.field_labels_out
                if batch.field_labels_out is not None
                else batch.field_labels
            ),
        )

        if batch.is_graph:
            inp = batch.graph
        else:
            if batch.input is None:
                raise RuntimeError("Matey tensor input is missing.")
            inp = self._rearrange(batch.input, "b t c d h w -> t b c d h w")

        if bool(getattr(self.params, "autoregressive", False)):
            output, rollout_steps = self._autoregressive_rollout(
                self.matey_model,
                inp,
                batch.field_labels,
                batch.bcs,
                opts,
                pushforward=True,
            )
            self.last_rollout_steps = int(rollout_steps)
            return output

        self.last_rollout_steps = None
        return self.matey_model(inp, batch.field_labels, batch.bcs, opts)


class MATEYHarness(BaseModelHarness):
    def __init__(self, cfg: Config):
        self._assert_supported_update_mode(cfg)

        self._matey_root = self._resolve_matey_root(cfg)
        self._validate_matey_root(self._matey_root)
        self._ensure_matey_on_pythonpath(self._matey_root)

        modules = self._load_matey_modules()
        params = self._build_matey_params(cfg, modules["YParams"])
        matey_model = self._build_matey_model(params, modules)

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

    @staticmethod
    def _assert_supported_update_mode(cfg: Config) -> None:
        if cfg.continual_learning.update_mode not in SUPPORTED_UPDATE_MODES:
            raise NotImplementedError(
                "Matey phase-1 harness supports only continual_learning.update_mode "
                "in {'base', 'none'}. Modes {'jvp_reg', 'ewc_online', 'kfac_online'} "
                "are not supported yet."
            )

    @staticmethod
    def _resolve_matey_root(cfg: Config) -> Path:
        raw = cfg.data.path.strip() if cfg.data.path.strip() else "MATEY"
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    @staticmethod
    def _ensure_matey_on_pythonpath(matey_root: Path) -> None:
        root_str = str(matey_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    @staticmethod
    def _validate_matey_root(matey_root: Path) -> None:
        if not matey_root.exists():
            raise FileNotFoundError(
                f"Matey root path does not exist: {matey_root}. "
                "Set [data].path to the local MATEY checkout path."
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
                "`pip install -r MATEY/requirements.txt`) and that "
                "`[data].path` points to the MATEY repo root."
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
        params.num_data_workers = max(1, int(cfg.train.num_workers))
        params.learning_rate = float(cfg.train.init_lr)

        if not hasattr(params, "weight_decay"):
            params.weight_decay = 0.0
        if not hasattr(params, "optimizer"):
            params.optimizer = "AdamW"
        if not hasattr(params, "embedding_offset"):
            params.embedding_offset = 0

        return params

    @staticmethod
    def _build_matey_model(params: Any, modules: dict[str, Any]) -> nn.Module:
        model_type = str(getattr(params, "model_type", "vit_all2all"))
        if model_type == "avit":
            model = modules["build_avit"](params)
        elif model_type == "svit":
            model = modules["build_svit"](params)
        elif model_type == "turbt":
            model = modules["build_turbt"](params)
        else:
            model = modules["build_vit"](params)

        if bool(getattr(params, "compile", False)):
            model = torch.compile(model)

        return model

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

        get_data_loader = self._modules["get_data_loader"]
        train_loader, train_dataset, _ = get_data_loader(
            self._params,
            self._params.train_data_paths,
            False,
            split="train",
            train_offset=getattr(self._params, "embedding_offset", 0),
            global_rank=0,
            num_sp_groups=1,
            group_size=1,
        )
        val_loader, val_dataset, _ = get_data_loader(
            self._params,
            self._params.valid_data_paths,
            False,
            split="val",
            train_offset=getattr(self._params, "embedding_offset", 0),
            global_rank=0,
            num_sp_groups=1,
            group_size=1,
        )

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
