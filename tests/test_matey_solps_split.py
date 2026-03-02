from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _load_stage_fn():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from examples.matey.src.solps_split import stage_solps_split

    return stage_solps_split


stage_solps_split = _load_stage_fn()


def _write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_staged_split_counts_are_deterministic(tmp_path: Path) -> None:
    source_root = tmp_path / "solps"
    train_root = source_root / "train"
    valid_root = source_root / "valid"

    # 10 valid files across train/valid roots.
    for idx in range(7):
        _write_file(train_root / f"train_{idx}.nc")
    for idx in range(2):
        _write_file(valid_root / f"valid_{idx}.nc")
    _write_file(valid_root / "holdout_sol.exo")

    # Hidden artifacts should be ignored.
    _write_file(train_root / ".cache" / "ignored.nc")

    result = stage_solps_split(
        [train_root, valid_root],
        ratios=(0.7, 0.15, 0.15),
        seed=7,
        cache_root=tmp_path / "cache",
    )

    assert result.reused_cache is False
    assert result.counts == {"train": 7, "val": 2, "test": 1}
    assert result.train_dir.exists()
    assert result.val_dir.exists()
    assert result.test_dir.exists()

    train_files = [path for path in result.train_dir.rglob("*") if path.is_file()]
    val_files = [path for path in result.val_dir.rglob("*") if path.is_file()]
    test_files = [path for path in result.test_dir.rglob("*") if path.is_file()]
    assert len(train_files) == 7
    assert len(val_files) == 2
    assert len(test_files) == 1
    assert all(
        ".cache" not in path.as_posix()
        for path in (train_files + val_files + test_files)
    )


def test_cache_reused_for_identical_inputs(tmp_path: Path) -> None:
    source_root = tmp_path / "solps"
    train_root = source_root / "train"
    valid_root = source_root / "valid"
    _write_file(train_root / "sample_1.nc")
    _write_file(valid_root / "sample_2.nc")

    first = stage_solps_split(
        [train_root, valid_root],
        seed=42,
        cache_root=tmp_path / "cache",
    )
    second = stage_solps_split(
        [train_root, valid_root],
        seed=42,
        cache_root=tmp_path / "cache",
    )

    assert first.cache_dir == second.cache_dir
    assert first.fingerprint == second.fingerprint
    assert first.reused_cache is False
    assert second.reused_cache is True


def test_missing_source_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        stage_solps_split([tmp_path / "missing"], cache_root=tmp_path / "cache")


def test_requires_enough_files_for_non_empty_train_val(tmp_path: Path) -> None:
    root = tmp_path / "solps"
    train_root = root / "train"
    valid_root = root / "valid"
    valid_root.mkdir(parents=True, exist_ok=True)
    _write_file(train_root / "only_file.nc")

    with pytest.raises(ValueError, match="at least two files"):
        stage_solps_split([train_root, valid_root], cache_root=tmp_path / "cache")
