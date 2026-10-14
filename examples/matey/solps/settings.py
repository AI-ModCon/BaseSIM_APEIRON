"""Run settings that belong to MATEY rather than to APEIRON.

APEIRON's ``Config`` is shared with every other user of the framework, so it
should not grow SOLPS vocabulary -- a ``[data] dset_type`` default of
``"SOLPS2D"`` or a ``solps_field_labels`` key means nothing to the MNIST or
CIFAR examples and would have to be maintained by people who do not run
plasma-edge simulations.

These three settings describe the *data* and the *checkpoint*, not the
continual-learning run, so they are read from a ``matey_settings.json`` in the
data root -- the same place, and the same reasoning, as the
``stream_manifest.json`` that ``model_stream.py`` reads its arrival order from.

Example ``matey_settings.json``::

    {
      "dset_type": "SOLPS2DwION",
      "field_labels": [533, 534, 535],
      "leadtime": 1,
      "use_step_inference": true
    }

Every field has a default, so the file is optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SETTINGS_NAME = "matey_settings.json"
SUPPORTED_SOLPS_DSET_TYPES = frozenset({"SOLPS2D", "SOLPS2DwION"})


def _parse_field_labels(raw: Any) -> tuple[int, ...]:
    """Accept a sequence or an ``"a,b,c"`` string."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = [
            p for p in raw.replace("[", "").replace("]", "").split(",") if p.strip()
        ]
        return tuple(int(p) for p in parts)
    return tuple(int(v) for v in raw)


@dataclass(frozen=True)
class MateySettings:
    """What the harness needs to know about a SOLPS root and its checkpoint."""

    # Which MATEY dataset registry key the root should be loaded as. SOLPS2D is
    # single-species KSTAR netCDF; SOLPS2DwION is the b2time.nc tree.
    dset_type: str = "SOLPS2D"

    # Field-embedding indices to force for the stream.
    #
    # MATEY gives each dataset a slice of a global field-embedding table by
    # walking DSET_NAME_TO_OBJECT in insertion order, so registering a dataset
    # class at runtime appends it and the slice depends on the *local* registry
    # rather than the one used during pre-training. For the leadtime_1
    # checkpoint the loader hands out [532, 533, 534] where training used
    # [533, 534, 535]. The off-by-one is silent -- the table has 536 columns --
    # and it raises SOLPS NRMSE from ~0.11 to ~0.63. Empty means trust the
    # loader. Re-derive for a new checkpoint before trusting any number.
    field_labels: tuple[int, ...] = ()

    # Rollout horizon the checkpoint was trained for. 0 leaves MATEY's own
    # leadtime_max alone.
    leadtime: int = 0

    # Score one autoregressive step at a time against the staged frames, the way
    # FusionBench's run_matey_inference_1step does, instead of rolling out over a
    # pooled split. The staged-pool split is skipped in this mode because each
    # arrival is already its own train/valid bundle.
    use_step_inference: bool = False

    # FusionBench's 1-step path hardcodes cond_input=None; set this to reproduce
    # its numbers exactly.
    drop_cond_input: bool = False

    # Extra per-device normalisation envelopes, keyed by the token that appears
    # in that device's file paths, e.g.
    #   {"MAST": {"ne": [lo, hi], "te": [...], "ti": [...], "tflux": [...]}}
    # They live here rather than in solps2dwion_dataset.py so that a device
    # under a distribution restriction can be registered by whoever stages its
    # data, without its name entering the repository.
    norm_envelopes: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=dict
    )

    @classmethod
    def resolve(cls, data_root: Path) -> "MateySettings":
        """Load ``matey_settings.json`` from ``data_root``, or fall back to defaults."""
        path = Path(data_root) / SETTINGS_NAME
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text())
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "MateySettings":
        dset_type = str(raw.get("dset_type", cls.dset_type)).strip()
        if dset_type not in SUPPORTED_SOLPS_DSET_TYPES:
            raise ValueError(
                f"Unsupported dset_type={dset_type!r} in {SETTINGS_NAME}. "
                f"Expected one of {sorted(SUPPORTED_SOLPS_DSET_TYPES)}."
            )
        leadtime = int(raw.get("leadtime", cls.leadtime))
        if leadtime < 0:
            raise ValueError(f"leadtime must be >= 0, got {leadtime}.")
        return cls(
            dset_type=dset_type,
            field_labels=_parse_field_labels(raw.get("field_labels")),
            leadtime=leadtime,
            use_step_inference=bool(
                raw.get("use_step_inference", cls.use_step_inference)
            ),
            drop_cond_input=bool(raw.get("drop_cond_input", cls.drop_cond_input)),
            norm_envelopes={
                str(token): {
                    str(f): (float(lo), float(hi)) for f, (lo, hi) in b.items()
                }
                for token, b in (raw.get("norm_envelopes") or {}).items()
            },
        )

    def describe(self) -> str:
        return (
            f"dset_type={self.dset_type}, "
            f"field_labels={list(self.field_labels) or 'from loader'}, "
            f"leadtime={self.leadtime or 'from checkpoint'}, "
            f"step_inference={self.use_step_inference}, "
            f"drop_cond_input={self.drop_cond_input}, "
            f"extra_envelopes={sorted(self.norm_envelopes) or 'none'}"
        )
