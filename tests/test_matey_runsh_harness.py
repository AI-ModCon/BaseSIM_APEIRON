from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ModelCfg,
    TrainCfg,
)


def _import_matey_symbols():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from examples.matey.model import MATEYHarness, MateyInputBatch, MateyTargetBatch

    return MATEYHarness, MateyInputBatch, MateyTargetBatch


def _make_cfg(data_path: str, update_mode: str = "base") -> Config:
    return Config(
        model=ModelCfg(name="matey_vit", pretrained_path=""),
        data=DataCfg(name="matey", path=data_path),
        train=TrainCfg(batch_size=2, num_workers=1, init_lr=0.001, max_iter=2),
        continual_learning=ContinualLearningCfg(update_mode=update_mode),
        drift_detection=DriftDetectionCfg(detection_interval=1, max_stream_updates=1),
        seed=7,
        device="cpu",
        multi_gpu=False,
    )


def _make_fake_matey_root(tmp_path: Path) -> Path:
    matey_root = tmp_path / "MATEY"
    cfg_dir = matey_root / "examples" / "config"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "Demo_SOLPS_vit.yaml").write_text(
        "basic_config:\n  model_type: vit_all2all\n"
    )
    return matey_root


def test_unsupported_updater_mode_raises_before_loading() -> None:
    MATEYHarness, _, _ = _import_matey_symbols()
    cfg = _make_cfg(data_path="MATEY", update_mode="jvp_reg")
    with pytest.raises(NotImplementedError, match="supports only"):
        MATEYHarness(cfg)


def test_missing_matey_root_path_raises(tmp_path: Path) -> None:
    MATEYHarness, _, _ = _import_matey_symbols()
    cfg = _make_cfg(data_path=str(tmp_path / "missing_matey_root"))
    with pytest.raises(FileNotFoundError, match="Matey root path does not exist"):
        MATEYHarness(cfg)


def test_harness_builds_stream_and_loss_with_mocked_matey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MATEYHarness, MateyInputBatch, MateyTargetBatch = _import_matey_symbols()
    matey_root = _make_fake_matey_root(tmp_path)

    class DummyYParams:
        def __init__(self, yaml_filename: str, config_name: str):
            self.yaml_filename = yaml_filename
            self.config_name = config_name
            self.model_type = "vit_all2all"
            self.optimizer = "AdamW"
            self.weight_decay = 0.0
            self.learning_rate = 0.001
            self.train_data_paths = [["unused", "unused", "", "tk-2D"]]
            self.valid_data_paths = [["unused", "unused", "", "tk-2D"]]
            self.embedding_offset = 0
            self.autoregressive = False
            self.hierarchical = None
            self.compile = False

    class DummyForwardOptionsBase:
        def __init__(self, **kwargs: Any):
            self.__dict__.update(kwargs)

    class DummySubDataset:
        tkhead_name = "tk-2D"
        type = "dummy"
        blockdict = None

    class DummyMixedDataset:
        sub_dsets = [DummySubDataset()]

    class DummyRawLoader:
        def __init__(self, batch: dict[str, Any]):
            self._batch = batch

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            yield self._batch

    class DummyMateyCore(torch.nn.Module):
        def forward(self, inp, field_labels, bcs, opts):
            if isinstance(inp, torch.Tensor):
                return inp[-1].float()
            raise RuntimeError("Dummy core only supports tensor inputs for this test.")

    def fake_rearrange(x: torch.Tensor, pattern: str) -> torch.Tensor:
        assert pattern == "b t c d h w -> t b c d h w"
        return x.permute(1, 0, 2, 3, 4, 5).contiguous()

    def fake_get_data_loader(*args, **kwargs):
        batch = {
            "input": torch.randn(2, 3, 4, 2, 2, 2),
            "label": torch.randn(2, 3, 4, 2, 2, 2),
            "bcs": torch.zeros(2, 1),
            "leadtime": torch.ones(2, 1, dtype=torch.long),
            "field_labels": torch.tensor(
                [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
            ),
            "dset_idx": torch.tensor([0, 0], dtype=torch.long),
        }
        return DummyRawLoader(batch), DummyMixedDataset(), None

    fake_modules = {
        "YParams": DummyYParams,
        "get_data_loader": fake_get_data_loader,
        "build_avit": lambda p: DummyMateyCore(),
        "build_svit": lambda p: DummyMateyCore(),
        "build_vit": lambda p: DummyMateyCore(),
        "build_turbt": lambda p: DummyMateyCore(),
        "add_weight_decay": lambda model, wd: model.parameters(),
        "determine_turt_levels": lambda *_: 0,
        "ForwardOptionsBase": DummyForwardOptionsBase,
        "autoregressive_rollout": lambda model,
        inp,
        labels,
        bcs,
        opts,
        pushforward=True: (
            model(inp, labels, bcs, opts),
            1,
        ),
        "rearrange": fake_rearrange,
        "DAdaptAdam": None,
    }

    monkeypatch.setattr(MATEYHarness, "_load_matey_modules", lambda self: fake_modules)
    monkeypatch.setattr(
        MATEYHarness,
        "_build_matey_model",
        staticmethod(lambda params, modules: DummyMateyCore()),
    )

    cfg = _make_cfg(data_path=str(matey_root), update_mode="base")
    harness = MATEYHarness(cfg)

    harness.update_data_stream()
    train_loader, val_loader = harness.get_cur_data_loaders()

    train_batch = next(iter(train_loader))
    assert isinstance(train_batch[0], MateyInputBatch)
    assert isinstance(train_batch[1], MateyTargetBatch)
    assert train_batch[1].shape[0] == 2

    x, y = train_batch[0].to("cpu"), train_batch[1].to("cpu")
    y_hat = harness.model(x)
    loss = harness.get_criterion()(y_hat, y)
    assert torch.isfinite(loss)

    nrmse = harness.eval_metrics["nrmse"](y_hat, y)
    rmse = harness.eval_metrics["rmse"](y_hat, y)
    assert torch.isfinite(nrmse)
    assert torch.isfinite(rmse)

    _ = next(iter(val_loader))
    harness.update_data_stream()
    assert harness.task_counter == 2


def test_examples_factory_dispatch_for_matey(monkeypatch: pytest.MonkeyPatch) -> None:
    _import_matey_symbols()
    from examples.utils import get_example

    class DummyHarness:
        def __init__(self, cfg):
            self.cfg = cfg

    monkeypatch.setattr("examples.matey.model.MATEYHarness", DummyHarness)

    cfg = _make_cfg(data_path="MATEY")
    harness = get_example(cfg)
    assert isinstance(harness, DummyHarness)
