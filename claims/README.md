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
| `apeiron-config` | `8cfb4c6c3ec8…` | 2 of 2 | — | yes |
| `apeiron-evaluation` | `da4cc107e9a8…` | 3 of 3 | — | yes |
| `apeiron-profilers` | `54a5267fd4a0…` | 2 of 2 | — | yes |
| `apeiron-model-harness` | `4adee990728c…` | 2 of 7 | 2 | yes |
| `apeiron-updaters` | `01494aa9c12e…` | 7 of 14 | 3 | yes |
| `apeiron-logger` | `1f461066564e…` | 5 of 7 | 1 | yes |
| `apeiron-drift-detection` | `713bdfd1bf0e…` | 5 of 7 | 1 | yes |
| `apeiron-trainer` | `59fd040ade1f…` | 2 of 24 | 6 | yes |
| `apeiron-monitor` | `1d8ba1409af4…` | 3 of 32 | 8 | yes |
| `apeiron-entrypoints` | `e456834bf11c…` | 5 of 36 | 9 | yes |

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
   reseal with `ret pack`. A reseal also invalidates any recorded
   reconstruction proof, and `ret status` will keep printing
   `proof  recorded` for one earned on the old bytes (LOG R12): re-earn it
   with `claims/scripts/regrow.sh <layer>`.
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

Or do every layer with retries and crosschecks in one go:

```
RETICULI_SRC=../reticuli/src RETICULI_PYTHON=python3.13 \
  claims/scripts/regrow.sh                  # all ten, bottom-up
claims/scripts/regrow.sh trainer monitor    # or just these
```

Do not reach for `ret rebuild --recursive` here; see below.

Then read what came back:

```
python3 claims/scripts/surface_diff.py /tmp/m3-<name> <name>
```

It reports how much of each file the criteria actually pin (logical lines,
docstrings excluded) and which public names the regrowth does without,
folding in inheritance and `__getattr__` delegation so only genuine
divergences show. Every line it prints is either a freedom you are granting
or a hole to pin — that judgement is the one part of this that cannot be
automated.

Treat a single regrowth as one sample, not a verdict: three runs of the same
claim under identical conditions gave two successes and one failure
(LOG R7). Divergences that recur are findings about the boundary;
divergences that appear once are findings about the producer.

## Where reconstruction stands

**All ten layers have been regrown from their criteria alone and
crosschecked** against current roots — original, byte copy and rebuild each
passing independently and landing on the same root, with the proof recorded
in each claim's store. Nine passed on the first attempt; `profilers` needed
the ATen table re-modelled as a pinned input first (LOG R11).

How much each layer's criteria actually pin down, in logical lines of code
(blanks, comments and docstrings excluded, since raw line counts flatter
the result):

| layer | src | regrown | pinned |
|---|---|---|---|
| `config` | 231 | 163 | 71% |
| `evaluation` | 40 | 27 | 68% |
| `profilers` | 192 | 109 | 57% |
| `model_harness` | 134 | 109 | 81% |
| `updaters` | 355 | 218 | 61% |
| `logger` | 549 | 256 | 47% |
| `drift_detection` | 621 | 293 | 47% |
| `trainer` | 223 | 72 | 32% |
| `monitor` | 243 | 97 | 40% |
| `entrypoints` | 624 | 172 | 28% |

Across the whole framework, **3212 lines of code came back as 1516** — so
roughly 47% of APEIRON is what its tests demand, and the rest is
latitude. Read the composed layers' low numbers with care: a deep claim
regrows only its own files, and a thin orchestrator over well-pinned
components is a small amount of genuinely free code, not a large gap.

The recursive form (`--recursive`) is not the way to do this — nine chained
layers multiply the producer's per-layer failure rate into a coin flip (LOG
A24). `claims/scripts/regrow.sh` does them one at a time.
