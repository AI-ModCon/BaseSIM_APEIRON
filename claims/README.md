# The claim ladder

Every Python file under `src/` that has criteria is pinned by one of ten
claims. Each claim names an acceptance boundary — its test files, their
fixtures, and a hash-pinned dependency lock — and treats the implementation
beneath that boundary as regrowable. See [LOG.md](LOG.md) for findings and
[IMPROVEMENTS.md](IMPROVEMENTS.md) for the ranked work list.

## The layers

Read bottom-up: a claim's components must hold before it does.

| claim | root | its own files | components | regrown + crosschecked |
|---|---|---|---|---|
| `apeiron-config` | `9cf772046e17…` | 2 of 2 | — | yes |
| `apeiron-evaluation` | `49d0b74e706d…` | 3 of 3 | — | yes |
| `apeiron-profilers` | `6fe145d6eb07…` | 2 of 3 + 1 pinned | — | yes |
| `apeiron-model-harness` | `d7eedc525fb2…` | 2 of 7 | 2 | yes |
| `apeiron-updaters` | `ed5c2c29373e…` | 7 of 14 | 3 | yes |
| `apeiron-logger` | `1f461066564e…` | 5 of 7 | 1 | yes |
| `apeiron-drift-detection` | `d0eec347dc29…` | 5 of 7 | 1 | yes |
| `apeiron-trainer` | `f4370ebed255…` | 2 of 24 | 6 | pending |
| `apeiron-monitor` | `d585d03dd4e9…` | 3 of 32 | 8 | pending |
| `apeiron-entrypoints` | `d48d0bec5f9c…` | 5 of 36 | 9 | pending |

"own files" is what this layer regrows itself; the rest arrive from its
components. The share matters when reading `ret assess`: a mutation score
for a deep claim is mostly sampling component code, which that claim's gate
rightly does not test (LOG R4).

`ret audit claims/entrypoints` re-earns the whole thing: the top gate plus
every component's own gates, each run against the bytes this claim ships, in
a sandbox built only from declared files. It reports **9/9 layers earned**
and takes about half a minute.

## What is not pinned

Five of the 37 Python files under `src/`:

- `src/__init__.py` and `src/apeiron/deployment/__init__.py` — empty
  namespace markers. `src/__init__.py` cannot sit at a room's root without
  making the room itself a package, which changes how the entry points
  resolve their imports.
- Nothing else. `apeiron/__init__.py`, `apeiron/model/__init__.py` and
  `apeiron/evaluation/evaluation.py` were the last three holdouts and are
  now covered.

## Working on a claim

Run reticuli from the sibling checkout, under Python 3.13 — river 0.23.0
ships no 3.14 wheels, so every lock here targets 3.13:

```
R=../reticuli/src
PYTHONPATH=$R python3.13 -m reticuli verify claims/<name>   # identity, instant
PYTHONPATH=$R python3.13 -m reticuli audit  claims/<name>   # re-earn the verdicts
PYTHONPATH=$R python3.13 -m reticuli assess claims/<name> --mutants 30
```

Two rules the tooling enforces and one it does not:

1. **Editing a pinned test moves the root.** That is the design — the name
   tracks the criteria — so expect `verify` to report `broken` until you
   reseal with `ret pack`.
2. **A dependent must declare every file its components supply.** Miss one
   and the warm pack still passes while `ret audit` fails, because the audit
   room is built from declarations alone.
3. **Resealing a leaf orphans the component copies inside its dependents.**
   Nothing warns you. Re-pull and reseal bottom-up; `scripts/` holds no
   helper for this yet, the order is the table above.

## Regrowing a layer

The point of the boundary is that an independent producer can rebuild
beneath it. The local Codex CLI costs nothing per run:

```
PYTHONPATH=$R python3.13 -m reticuli rebuild claims/<name> \
  --producer "env PYTHONPATH=$R python3 -m reticuli.producers.codex" \
  -o /tmp/m3-<name>
PYTHONPATH=$R python3.13 -m reticuli crosscheck claims/<name> /tmp/m3-<name> --record-proof
```

Add `--recursive` on the top claim to regrow every layer bottom-up.

Treat a single regrowth as one sample, not a verdict: three runs of the same
claim under identical conditions gave two successes and one failure
(LOG R7). Divergences that recur are findings about the boundary;
divergences that appear once are findings about the producer.

## Where reconstruction stands

Seven of the ten layers have been regrown from their criteria alone and
crosschecked — original, byte copy and rebuild each passing independently
and landing on the same root — all on the first attempt. Across those seven
layers, 2,293 logical lines of code (blanks, comments and docstrings
excluded) came back as 1,132: **roughly half of APEIRON's code is not
pinned down by its tests**, even after this round of test writing.

The top three layers are pending only because the Codex account hit its
usage limit mid-run; nothing about them failed. The recursive form
(`--recursive`) is not the way to finish them — nine chained layers
multiply the producer's per-layer failure rate into a coin flip (LOG
A24).
