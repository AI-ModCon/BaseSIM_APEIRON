from __future__ import annotations

import copy
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

    from examples.matey.matey_batches import MateyInputBatch, MateyTargetBatch
    from examples.matey.matey_bridge import MateyBridge
    from examples.matey.model import MATEYHarness

    return MATEYHarness, MateyBridge, MateyInputBatch, MateyTargetBatch


def _make_cfg(
    data_path: str,
    update_mode: str = "base",
    num_workers: int = 1,
) -> Config:
    return Config(
        model=ModelCfg(name="matey_vit", pretrained_path=""),
        data=DataCfg(name="matey", path=data_path),
        train=TrainCfg(
            batch_size=2, num_workers=num_workers, init_lr=0.001, max_iter=2
        ),
        continual_learning=ContinualLearningCfg(update_mode=update_mode),
        drift_detection=DriftDetectionCfg(detection_interval=1, max_stream_updates=1),
        seed=7,
        device="cpu",
        multi_gpu=False,
    )


def _make_fake_matey_root(tmp_path: Path) -> Path:
    matey_root = tmp_path / "MATEY"
    matey_root.mkdir(parents=True)
    return matey_root


def _make_fake_bridge(
    bridge_cls: type[Any],
    *,
    solps_root: Path | None = None,
    train_data_paths: list[list[Any]] | None = None,
    valid_data_paths: list[list[Any]] | None = None,
    loader_calls: list[dict[str, Any]] | None = None,
    graph_batches: bool = False,
):
    class DummyYParams:
        def __init__(self, yaml_filename: str, config_name: str):
            self.yaml_filename = yaml_filename
            self.config_name = config_name
            self.model_type = "vit_all2all"
            self.weight_decay = 0.0
            self.learning_rate = 0.001
            self.embedding_offset = 0
            self.autoregressive = False
            self.compile = False

            if train_data_paths is not None and valid_data_paths is not None:
                self.train_data_paths = copy.deepcopy(train_data_paths)
                self.valid_data_paths = copy.deepcopy(valid_data_paths)
            elif solps_root is None:
                self.train_data_paths = [["unused", "SOLPS2D", "", "tk-2D"]]
                self.valid_data_paths = [["unused", "SOLPS2D", "", "tk-2D"]]
            else:
                self.train_data_paths = [
                    [str(solps_root / "train"), "SOLPS2D", "", "tk-2D"]
                ]
                self.valid_data_paths = [
                    [str(solps_root / "valid"), "SOLPS2D", "", "tk-2D"]
                ]

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
        if loader_calls is not None:
            params = args[0]
            paths = args[1]
            loader_calls.append(
                {
                    "split": kwargs.get("split"),
                    "paths": copy.deepcopy(paths),
                    "train_val_test": list(getattr(params, "train_val_test", [])),
                }
            )

        if graph_batches:
            batch = {
                "graph": object(),
                "bcs": torch.zeros(2, 1),
                "field_labels": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
                "dset_idx": torch.tensor([0, 0], dtype=torch.long),
            }
            return DummyRawLoader(batch), DummyMixedDataset(), None

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

    bridge = bridge_cls(
        YParams=DummyYParams,
        get_data_loader=fake_get_data_loader,
        build_vit=lambda p: DummyMateyCore(),
        add_weight_decay=lambda model, wd: model.parameters(),
        ForwardOptionsBase=DummyForwardOptionsBase,
        autoregressive_rollout=lambda model, inp, labels, bcs, opts, pushforward=True: (
            model(inp, labels, bcs, opts),
            1,
        ),
        rearrange=fake_rearrange,
    )

    return bridge


def _write_solps_samples(solps_root: Path) -> None:
    (solps_root / "train").mkdir(parents=True, exist_ok=True)
    (solps_root / "valid").mkdir(parents=True, exist_ok=True)
    for filename in (
        solps_root / "train" / "sample-a.nc",
        solps_root / "train" / "sample-b.nc",
        solps_root / "valid" / "sample-c.nc",
    ):
        filename.write_text("stub", encoding="utf-8")


def test_unsupported_updater_mode_raises_before_loading() -> None:
    MATEYHarness, _, _, _ = _import_matey_symbols()
    cfg = _make_cfg(data_path="MATEY", update_mode="jvp_reg")
    with pytest.raises(NotImplementedError, match="supports only"):
        MATEYHarness(cfg)


def test_missing_matey_root_path_raises(tmp_path: Path) -> None:
    MATEYHarness, _, _, _ = _import_matey_symbols()
    cfg = _make_cfg(data_path=str(tmp_path / "missing_matey_root"))
    with pytest.raises(FileNotFoundError, match="Matey root path does not exist"):
        MATEYHarness(cfg)


def test_zero_workers_raises_clear_error(tmp_path: Path) -> None:
    MATEYHarness, _, _, _ = _import_matey_symbols()
    matey_root = _make_fake_matey_root(tmp_path)
    cfg = _make_cfg(data_path=str(matey_root), num_workers=0)
    with pytest.raises(ValueError, match="num_workers >= 1"):
        MATEYHarness(cfg)


def test_harness_builds_stream_and_loss_with_mocked_matey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MATEYHarness, MateyBridge, MateyInputBatch, MateyTargetBatch = (
        _import_matey_symbols()
    )
    matey_root = _make_fake_matey_root(tmp_path)
    solps_root = tmp_path / "solps"
    _write_solps_samples(solps_root)
    fake_bridge = _make_fake_bridge(MateyBridge, solps_root=solps_root)

    monkeypatch.setattr(
        MATEYHarness,
        "_load_bridge",
        staticmethod(lambda matey_root: fake_bridge),
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


def test_default_split_applied_for_matey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MATEYHarness, MateyBridge, _, _ = _import_matey_symbols()
    monkeypatch.setattr(
        "examples.matey.model.DEFAULT_SOLPS_CACHE_ROOT", tmp_path / "cache"
    )
    matey_root = _make_fake_matey_root(tmp_path)
    solps_root = tmp_path / "solps"
    _write_solps_samples(solps_root)

    fake_bridge = _make_fake_bridge(MateyBridge, solps_root=solps_root)
    monkeypatch.setattr(
        MATEYHarness,
        "_load_bridge",
        staticmethod(lambda matey_root: fake_bridge),
    )

    cfg = _make_cfg(data_path=str(matey_root))
    harness = MATEYHarness(cfg)

    assert harness._params.train_val_test == [0.7, 0.15, 0.15]
    train_root = Path(harness._params.train_data_paths[0][0]).resolve()
    val_root = Path(harness._params.valid_data_paths[0][0]).resolve()
    assert train_root.name == "train"
    assert val_root.name == "val"
    assert train_root.parent == val_root.parent
    train_files = [path for path in train_root.rglob("*") if path.is_file()]
    val_files = [path for path in val_root.rglob("*") if path.is_file()]
    assert train_files
    assert val_files
    staged_names = {path.name for path in (train_files + val_files)}
    assert {"sample-a.nc", "sample-b.nc", "sample-c.nc"} <= staged_names


def test_train_and_val_loaders_use_staged_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MATEYHarness, MateyBridge, _, _ = _import_matey_symbols()
    monkeypatch.setattr(
        "examples.matey.model.DEFAULT_SOLPS_CACHE_ROOT", tmp_path / "cache"
    )
    matey_root = _make_fake_matey_root(tmp_path)
    solps_root = tmp_path / "solps"
    _write_solps_samples(solps_root)

    loader_calls: list[dict[str, Any]] = []
    fake_bridge = _make_fake_bridge(
        MateyBridge,
        solps_root=solps_root,
        loader_calls=loader_calls,
    )
    monkeypatch.setattr(
        MATEYHarness,
        "_load_bridge",
        staticmethod(lambda matey_root: fake_bridge),
    )

    cfg = _make_cfg(data_path=str(matey_root))
    harness = MATEYHarness(cfg)
    harness.update_data_stream()

    assert [call["split"] for call in loader_calls] == ["train", "val"]
    assert loader_calls[0]["train_val_test"] == [0.7, 0.15, 0.15]
    assert loader_calls[1]["train_val_test"] == [0.7, 0.15, 0.15]

    train_loader_path = Path(loader_calls[0]["paths"][0][0]).resolve()
    val_loader_path = Path(loader_calls[1]["paths"][0][0]).resolve()
    assert train_loader_path.name == "train"
    assert val_loader_path.name == "val"
    assert train_loader_path.parent == val_loader_path.parent


def test_graph_batch_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MATEYHarness, MateyBridge, _, _ = _import_matey_symbols()
    monkeypatch.setattr(
        "examples.matey.model.DEFAULT_SOLPS_CACHE_ROOT", tmp_path / "cache"
    )
    matey_root = _make_fake_matey_root(tmp_path)
    solps_root = tmp_path / "solps"
    _write_solps_samples(solps_root)

    fake_bridge = _make_fake_bridge(
        MateyBridge,
        solps_root=solps_root,
        graph_batches=True,
    )
    monkeypatch.setattr(
        MATEYHarness,
        "_load_bridge",
        staticmethod(lambda matey_root: fake_bridge),
    )

    cfg = _make_cfg(data_path=str(matey_root))
    harness = MATEYHarness(cfg)
    harness.update_data_stream()
    train_loader, _ = harness.get_cur_data_loaders()

    with pytest.raises(RuntimeError, match="graph batches are not supported"):
        _ = next(iter(train_loader))


def test_mixed_solps_and_non_solps_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MATEYHarness, MateyBridge, _, _ = _import_matey_symbols()
    monkeypatch.setattr(
        "examples.matey.model.DEFAULT_SOLPS_CACHE_ROOT", tmp_path / "cache"
    )
    matey_root = _make_fake_matey_root(tmp_path)
    solps_root = tmp_path / "solps"
    _write_solps_samples(solps_root)

    train_paths = [
        [str(solps_root / "train"), "SOLPS2D", "", "tk-2D"],
        [str(tmp_path / "other"), "incompNS", "", "tk-2D"],
    ]
    valid_paths = [[str(solps_root / "valid"), "SOLPS2D", "", "tk-2D"]]

    fake_bridge = _make_fake_bridge(
        MateyBridge,
        train_data_paths=train_paths,
        valid_data_paths=valid_paths,
    )
    monkeypatch.setattr(
        MATEYHarness,
        "_load_bridge",
        staticmethod(lambda matey_root: fake_bridge),
    )

    cfg = _make_cfg(data_path=str(matey_root))
    with pytest.raises(ValueError, match="supports SOLPS2D entries only"):
        MATEYHarness(cfg)


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
