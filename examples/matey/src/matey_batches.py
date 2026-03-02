from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

import torch
from torch import Tensor, nn


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


class MateyLoaderAdapter:
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


class MateyModelAdapter(nn.Module):
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
