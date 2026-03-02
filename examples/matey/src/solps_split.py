from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
_SUPPORTED_SUFFIXES = (".nc",)
_SUPPORTED_EXODUS_SUFFIX = "_sol.exo"
_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SolpsStagedSplit:
    cache_dir: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    common_root: Path
    fingerprint: str
    counts: dict[str, int]
    reused_cache: bool


def stage_solps_split(
    source_roots: Sequence[Path],
    *,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
    cache_root: Path = Path("output/matey_split_cache"),
) -> SolpsStagedSplit:
    roots = _normalize_source_roots(source_roots)
    if not roots:
        raise ValueError("SOLPS split staging requires at least one source directory.")

    common_root = _resolve_common_root(roots)
    pool = _build_pool(roots=roots, common_root=common_root)
    total_files = len(pool)
    if total_files == 0:
        raise ValueError("SOLPS split staging found no '*.nc' or '*_sol.exo' files.")
    if total_files < 2:
        raise ValueError(
            "SOLPS split staging requires at least two files so train and val are both non-empty."
        )

    ordered_rel_paths = sorted(pool.keys())
    rel_paths_by_seed = sorted(
        ordered_rel_paths,
        key=lambda rel: (_stable_seed_key(seed=seed, rel_path=rel), rel),
    )

    train_count, val_count, _test_count = _compute_split_counts(
        total=total_files,
        ratios=ratios,
    )

    train_rel = rel_paths_by_seed[:train_count]
    val_rel = rel_paths_by_seed[train_count : train_count + val_count]
    test_rel = rel_paths_by_seed[train_count + val_count :]
    split_counts: dict[str, int] = {
        "train": len(train_rel),
        "val": len(val_rel),
        "test": len(test_rel),
    }

    entries: list[dict[str, str | int]] = []
    for split_name, rel_paths in (
        ("train", train_rel),
        ("val", val_rel),
        ("test", test_rel),
    ):
        for rel_path in rel_paths:
            src = pool[rel_path]
            stat = src.stat()
            entries.append(
                {
                    "split": split_name,
                    "rel_path": rel_path,
                    "source": str(src),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )

    spec = {
        "version": _MANIFEST_VERSION,
        "seed": int(seed),
        "ratios": [float(r) for r in ratios],
        "common_root": str(common_root),
        "source_roots": [str(path) for path in roots],
        "entries": entries,
        "counts": split_counts,
    }

    fingerprint = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    cache_root_abs = (
        cache_root if cache_root.is_absolute() else (Path.cwd() / cache_root)
    )
    cache_dir = cache_root_abs.resolve() / fingerprint
    manifest_path = cache_dir / "manifest.json"

    if _can_reuse_cache(cache_dir=cache_dir, manifest_path=manifest_path, spec=spec):
        return SolpsStagedSplit(
            cache_dir=cache_dir,
            train_dir=cache_dir / "train",
            val_dir=cache_dir / "val",
            test_dir=cache_dir / "test",
            common_root=common_root,
            fingerprint=fingerprint,
            counts=split_counts,
            reused_cache=True,
        )

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    for split_name in ("train", "val", "test"):
        (cache_dir / split_name).mkdir(parents=True, exist_ok=True)

    for entry in entries:
        split_name = str(entry["split"])
        rel_file = Path(str(entry["rel_path"]))
        source = Path(str(entry["source"]))
        destination = cache_dir / split_name / rel_file
        _materialize_link(source=source, destination=destination)

    manifest_payload = {"fingerprint": fingerprint, "spec": spec}
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return SolpsStagedSplit(
        cache_dir=cache_dir,
        train_dir=cache_dir / "train",
        val_dir=cache_dir / "val",
        test_dir=cache_dir / "test",
        common_root=common_root,
        fingerprint=fingerprint,
        counts=split_counts,
        reused_cache=False,
    )


def _normalize_source_roots(source_roots: Sequence[Path]) -> list[Path]:
    roots: list[Path] = []
    for root in source_roots:
        resolved = root if root.is_absolute() else (Path.cwd() / root)
        resolved = resolved.resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(
                f"SOLPS source root does not exist or is not a directory: {resolved}"
            )
        roots.append(resolved)
    return roots


def _resolve_common_root(paths: Sequence[Path]) -> Path:
    try:
        from os.path import commonpath

        common = Path(commonpath([str(path) for path in paths])).resolve()
    except ValueError as exc:
        raise ValueError(
            f"Could not derive a shared SOLPS root from: {[str(path) for path in paths]}"
        ) from exc

    if common == common.parent:
        raise ValueError(
            "Shared SOLPS root resolved to filesystem root; please make source paths more specific."
        )

    return common


def _is_hidden(relative_path: Path) -> bool:
    return any(part.startswith(".") for part in relative_path.parts)


def _is_supported_file(path: Path) -> bool:
    if path.suffix in _SUPPORTED_SUFFIXES:
        return True
    return path.name.endswith(_SUPPORTED_EXODUS_SUFFIX)


def _build_pool(roots: Sequence[Path], common_root: Path) -> dict[str, Path]:
    pool: dict[str, Path] = {}
    for root in roots:
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue

            rel_to_root = candidate.relative_to(root)
            if _is_hidden(rel_to_root):
                continue
            if not _is_supported_file(candidate):
                continue

            rel_to_common = candidate.relative_to(common_root)
            rel_key = rel_to_common.as_posix()
            existing = pool.get(rel_key)
            resolved_candidate = candidate.resolve()

            if existing is not None and existing != resolved_candidate:
                raise ValueError(
                    "SOLPS split staging found duplicate relative file paths under the shared "
                    f"root: {rel_key}"
                )

            pool[rel_key] = resolved_candidate

    return pool


def _stable_seed_key(*, seed: int, rel_path: str) -> str:
    payload = f"{seed}:{rel_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compute_split_counts(
    *, total: int, ratios: tuple[float, float, float]
) -> tuple[int, int, int]:
    if len(ratios) != 3:
        raise ValueError("SOLPS split ratios must contain exactly three values.")

    fractions = [float(v) for v in ratios]
    if any(v < 0 for v in fractions):
        raise ValueError("SOLPS split ratios must be non-negative.")

    ratio_sum = sum(fractions)
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"SOLPS split ratios must sum to 1.0. Received sum={ratio_sum:.8f}."
        )

    raw_counts = [value * total for value in fractions]
    counts = [int(value) for value in raw_counts]

    remaining = total - sum(counts)
    if remaining > 0:
        remainders = [raw - int(raw) for raw in raw_counts]
        for idx in sorted(range(3), key=lambda i: remainders[i], reverse=True):
            if remaining == 0:
                break
            counts[idx] += 1
            remaining -= 1

    _ensure_non_empty_train_val(counts)
    return counts[0], counts[1], counts[2]


def _ensure_non_empty_train_val(counts: list[int]) -> None:
    train_idx = 0
    val_idx = 1

    if counts[train_idx] == 0:
        donor = _find_donor_split(counts=counts, exclude={train_idx})
        if donor is None:
            raise ValueError(
                "SOLPS split staging could not allocate at least one file to train."
            )
        counts[donor] -= 1
        counts[train_idx] += 1

    if counts[val_idx] == 0:
        donor = _find_donor_split(counts=counts, exclude={val_idx, train_idx})
        if donor is None:
            donor = _find_donor_split(counts=counts, exclude={val_idx})
        if donor is None:
            raise ValueError(
                "SOLPS split staging could not allocate at least one file to val."
            )
        counts[donor] -= 1
        counts[val_idx] += 1

    if counts[train_idx] <= 0 or counts[val_idx] <= 0:
        raise ValueError(
            "SOLPS split staging requires non-empty train and val partitions."
        )


def _find_donor_split(counts: Sequence[int], exclude: set[int]) -> int | None:
    candidates = [
        idx for idx, value in enumerate(counts) if idx not in exclude and value > 1
    ]
    if candidates:
        return max(candidates, key=lambda idx: counts[idx])

    fallback = [
        idx for idx, value in enumerate(counts) if idx not in exclude and value > 0
    ]
    if fallback:
        return max(fallback, key=lambda idx: counts[idx])

    return None


def _can_reuse_cache(
    cache_dir: Path, manifest_path: Path, spec: dict[str, object]
) -> bool:
    if not cache_dir.exists() or not manifest_path.exists():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    if manifest.get("spec") != spec:
        return False

    entries = cast(list[dict[str, str | int]], spec["entries"])
    for entry in entries:
        split_name = str(entry["split"])
        rel_path = Path(str(entry["rel_path"]))
        destination = cache_dir / split_name / rel_path
        if not destination.exists() or not destination.is_file():
            return False

    return True


def _materialize_link(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(source, destination)
        return
    except OSError:
        pass

    try:
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.link(source, destination)
        return
    except OSError:
        pass

    shutil.copy2(source, destination)
