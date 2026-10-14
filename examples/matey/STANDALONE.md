# Can this example run outside ORNL?

**The code is no longer the obstacle; the data and the checkpoint are.** MATEY is
published: `github.com/ORNL/MATEY`, MIT, tagged `v1.0.0`. Every symbol this
example imports exists there, at the same module path, with an **identical
signature** to the commit the harness pins (checked by comparing ASTs; see
[Verifying the public release](#verifying-the-public-release)). What is left is
one packaging gap, one untested Python floor, and two permission questions that
belong to other people.

An earlier revision of this file called the MATEY internals private. That was
wrong, and the correction matters: it moves the example from "cannot be run by
anyone outside" to "can be run by anyone who is given the data".

## Blockers

| Blocker | Detail | Who can unblock it |
|---|---|---|
| `matey` is not pip-installable | `setup.py` at `v1.0.0` is still a docstring behind `#FIXME: WIP, not ready to used yet`, so `pip install .` yields an empty distribution. Obtainable regardless: clone the tag and put it on `PYTHONPATH`, which is what `MATEY_SRC` in `submit_stream_cl.sh` already does. The PyPI name `matey` is taken by an unrelated project, so a future release needs a different distribution name | **MATEY team** -- uncomment `setup.py`, pick a distribution name, make `flash-attn` / `mpi4py` / `exodusii` optional |
| Python version | apeiron declares `requires-python >=3.13,<3.14`; MATEY is exercised under 3.10. All 52 modules of `v1.0.0` **compile clean under 3.13.0** (one cosmetic `SyntaxWarning` for an escape sequence in a docstring), so the language is not the problem. What is untested is whether its dependency set -- ROCm torch, `flash-attn`, `mpi4py` -- resolves there | **ours to finish**: the syntax half is answered, the wheel half is an install attempt |
| Checkpoint redistribution | 76.1 MiB `best_ckpt.tar`, plus the 2.8 KB `hyperparams.yaml` beside it. The YAML is **required**, not optional: it carries `model_type`, `embed_dim` and `n_states`, and the architecture is rebuilt from it | **MATEY team** |
| SOLPS source data | The `b2time.nc` files the stream is sliced from: 326 MiB, 825 MiB and 4.19 GiB | **the `fus183` project** -- release permission for a derived, time-sliced subset |
| The third device | Arrivals 24-31 of the 32-arrival stream. `stream_manifest.json` carries its own note not to name it in write-ups | Assume no. Exclude from any bundle, and never name |

Anonymising the data does **not** substitute for that permission, and is not
worth building. Device names can be stripped from the manifest and the fields
shipped in normalised units, but the physics stays identifiable: the grid
geometry is a fingerprint (38x98 for one device against roughly 1.7x the cells
for the others), the profile magnitudes identify the machine to anyone who works
on edge plasma, and the per-device normalisation envelopes are themselves device
constants -- which is exactly why using the wrong one is a bug and not a
rescaling. The restriction is on the physics, not on the labels.

## Verifying the public release

`v1.0.0` does not contain the pinned commit -- that lives in a private fork -- so
name-compatibility is not enough on its own. What was checked, and reproduces:

| Symbol | Module | Signature at `v1.0.0` |
|---|---|---|
| `build_turbt` | `matey/models/turbt.py` | identical |
| `add_weight_decay` | `matey/utils/distributed_utils.py` | identical |
| `determine_turt_levels` | `matey/utils/distributed_utils.py` | identical |
| `ForwardOptionsBase` | `matey/utils/forward_options.py` | identical |
| `autoregressive_rollout` | `matey/utils/training_utils.py` | identical |
| `get_data_loader` | `matey/data_utils/datasets.py` | identical |
| `BasenetCDFDirectoryDataset` | `matey/data_utils/netcdf_datasets.py` | identical |
| `YParams` | `matey/utils/YParams.py` | identical |

`examples/config/Demo_SOLPS_vit.yaml`, which this example ships a copy of, is in
the public tree as well.

Identical signatures are necessary, not sufficient -- behaviour inside those
functions was not compared. **The remaining verification is one run**: clone
`v1.0.0`, point `MATEY_SRC` at it, and check the control arm reproduces. That is
the single experiment that would let this file be deleted.

A checkpoint without permission is **worse than no checkpoint**: an empty
`model.pretrained_path` starts the ViT from random weights, and every number the
run reports becomes meaningless. The harness warns, but it does not stop.

## What a distributable bundle would cost

Sizes measured, not estimated.

| Tier | Contents | Size | What it buys |
|---|---|---:|---|
| T0 | `stream_manifest.json`, `matey_settings.json`, `Demo_SOLPS_vit.yaml` | ~15 KB | the layout is self-documenting; tests can build fixtures |
| T1 | 2 arrivals, one per machine | ~91 MiB | wiring test, one monitoring window |
| T2 | 8 arrivals spanning one regime change | ~330 MiB | a real detection and one CL round |
| T3 | arrivals 0-23 | **972 MiB** | reproduces the adaptation-sequence figure |
| T4 | all 32 arrivals | 1.4 GiB | **not distributable** -- contains the restricted device |
| CKPT | `best_ckpt.tar` + `hyperparams.yaml` | **76.1 MiB** | required by every tier above T0 |

T1 and the checkpoint fit a GitHub release asset comfortably; T3 wants Zenodo,
where it would also get a DOI. `.nc` is already HDF5-backed, so compression buys
almost nothing -- **slicing frames is the only lever that works**, which is what
`stage_solps_stream.py --window` already does.

## What is not blocked

Worth saying, because "you cannot run it" is not the same as "you cannot review
it":

- the harness imports, and its test files collect and pass, with **no MATEY, no
  checkpoint and no data** -- the batch-slicing and settings tests need none of
  the three;
- the drift-detection and continual-learning logic is entirely reviewable, and is
  exercised by the framework's own suite against a dummy harness;
- the XGC study reads data only and never runs a MATEY forward pass.

## What was fixed here

The parts that did not need anyone's permission:

- **`stage_solps_stream.py` is shipped again.** Without it the README documented a
  `stream_manifest.json` layout and gave no way to produce one. Its site paths now
  come from `$MATEYDATA`, `--out` is required, and the restricted device is not in
  the default case order.
- **`sweep_field_labels.py` is shipped again.** The README says to re-derive
  `field_labels` whenever the checkpoint changes, having just explained that
  getting it wrong inflates NRMSE roughly six-fold. A warning with no remedy is
  worse than neither.
- **`download_data.sh`** fetches a tier, verifies it against a committed
  `checksums.txt`, and fails hard on a mismatch. Silently-wrong data is this
  example's established failure mode, so an unverified download is not worth
  having.
- **Site paths are out of the scripts.** `submit_joint_oracle.sh` and the
  `drift_showcase/` scripts take `${VAR:?}` / `${VAR:-default}` like
  `submit_stream_cl.sh` already did.

## Why `matey` is not declared in `pyproject.toml`

Still not, but for one reason now rather than three:

- A dependency declaration is a promise of installability, and `setup.py` at
  `v1.0.0` builds an empty distribution. Declaring it would make that promise
  falsely, and Poetry and uv resolve extras **at lock time even when they are not
  installed**, so the failure would land on every contributor and on CI.

The other two objections are gone: the source is public, so no SSH key into a
private organisation is needed, and the heavy requirements (`flash-attn`,
`mpi4py`, `adios2`, `exodusii`) are avoidable for a SOLPS-only run --
`install_matey_optional_import_shims()` already stubs the graph/XGC path.

MATEY is supplied on `PYTHONPATH`; the version of record is `MATEY_GIT_COMMIT` in
`examples/matey/model.py`, with `MATEY_PUBLIC_URL` beside it for the tag anyone
can clone.

**Delete this file the day `setup.py` is uncommented and the run above
reproduces.** At that point add `[project.optional-dependencies] matey = [...]`.

## What to ask for, and of whom

The open questions, smallest first, so a meeting can work down the list:

1. **Does the harness run against public `v1.0.0`?** One control-arm run. Nobody's
   permission required -- this is ours to answer.
2. **Does MATEY's dependency set install under Python 3.13?** Its own code already
   compiles there; what is left is ROCm torch, `flash-attn` and `mpi4py`. Also
   ours, and it decides whether apeiron's floor has to move.
3. **May the checkpoint be redistributed?** MATEY team. 76.1 MiB, and useless
   without its `hyperparams.yaml`.
4. **May a time-sliced SOLPS subset be released?** The `fus183` project.
   Arrivals 0-23 are the ask; the third device is not.

Only items 3 and 4 need anyone outside the team, and item 4 is the one that
decides whether this example ships with data or stays a code-only reference. For
what runs today with neither, see `examples/synthetic_drift/`.
