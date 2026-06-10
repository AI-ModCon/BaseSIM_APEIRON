from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


def _import_harness():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from examples.matey.model import MATEYHarness

    return MATEYHarness


MATEYHarness = _import_harness()


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _assert_same_params(
    model: torch.nn.Module, expected: dict[str, torch.Tensor]
) -> None:
    current = model.state_dict()
    assert set(current.keys()) == set(expected.keys())
    for key in current:
        assert torch.allclose(current[key], expected[key])


def test_load_pretrained_from_raw_state_dict(tmp_path: Path) -> None:
    source = _TinyModel()
    expected = _clone_state_dict(source)
    ckpt = tmp_path / "raw_state.pt"
    torch.save(expected, ckpt)

    target = _TinyModel()
    MATEYHarness._load_pretrained_weights_if_available(target, str(ckpt))

    _assert_same_params(target, expected)


def test_load_pretrained_from_model_state_key(tmp_path: Path) -> None:
    source = _TinyModel()
    expected = _clone_state_dict(source)
    ckpt = tmp_path / "wrapped_state.pt"
    torch.save({"model_state": expected, "epoch": 3}, ckpt)

    target = _TinyModel()
    MATEYHarness._load_pretrained_weights_if_available(target, str(ckpt))

    _assert_same_params(target, expected)


def test_load_pretrained_strips_module_prefix(tmp_path: Path) -> None:
    source = _TinyModel()
    expected = _clone_state_dict(source)
    prefixed = {f"module.{k}": v for k, v in expected.items()}

    ckpt = tmp_path / "module_prefixed.pt"
    torch.save({"model_state": prefixed}, ckpt)

    target = _TinyModel()
    MATEYHarness._load_pretrained_weights_if_available(target, str(ckpt))

    _assert_same_params(target, expected)


def test_missing_pretrained_path_raises(tmp_path: Path) -> None:
    target = _TinyModel()
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="pretrained checkpoint not found"):
        MATEYHarness._load_pretrained_weights_if_available(target, str(missing))


def test_resolve_checkpoint_hyperparams_yaml(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo_nbatchsloc100"
    ckpt_dir = run_dir / "training_checkpoints"
    ckpt_dir.mkdir(parents=True)
    hyperparams = run_dir / "hyperparams.yaml"
    hyperparams.write_text("model_type: turbt\n", encoding="utf-8")
    ckpt = ckpt_dir / "best_ckpt.tar"
    ckpt.write_bytes(b"placeholder")

    resolved = MATEYHarness._resolve_checkpoint_hyperparams_yaml(str(ckpt))
    assert resolved == hyperparams


def test_resolve_checkpoint_hyperparams_yaml_missing(tmp_path: Path) -> None:
    ckpt = tmp_path / "missing.pt"
    ckpt.write_bytes(b"x")
    assert MATEYHarness._resolve_checkpoint_hyperparams_yaml(str(ckpt)) is None


def test_unsupported_checkpoint_format_raises(tmp_path: Path) -> None:
    target = _TinyModel()
    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"epoch": 1, "optimizer": {}}, bad_ckpt)

    with pytest.raises(ValueError, match="Unsupported MATEY checkpoint format"):
        MATEYHarness._load_pretrained_weights_if_available(target, str(bad_ckpt))
