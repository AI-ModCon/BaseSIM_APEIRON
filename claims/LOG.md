# Reconstruction log

Findings from putting APEIRON under reticuli, layer by layer. Each claim
pins a layer's tests as its acceptance boundary and treats the
implementation as regrowable; what the regrowth machinery trips over is
recorded here. Entries say where the problem lives (APEIRON, reticuli, or
this workspace), why it matters, and what has been done about it.

Claims live under `claims/`; sealed roots are in each claim's
`.reticuli/manifest.json`. Source snapshots referenced below are from
`main` at 63a877a.

## 2026-10-21

### A1 — APEIRON: the package `__init__` welds every layer together
`src/apeiron/__init__.py` eagerly imports config, harness, monitor,
detectors, trainer, and logger, so `import apeiron.config.configuration`
needs river, evidently, wandb, and mlflow installed — the whole dependency
tree to parse a TOML. No layer is importable in isolation, which blocks
claiming any layer in place and is why `claims/config/` carries copies and
a namespace package instead. Candidate fix: a lazy `__init__` (PEP 562
module `__getattr__`) that keeps `from apeiron import Config` working.
Status: open; first candidate for an upstream branch.

### A2 — APEIRON: config tests kill 36% of injected faults
`ret assess claims/config --mutants 25`: mutation score 0.36. Branch
mutations 0/4 detected, constants 0/4 (of 88 constant sites — the
dataclass defaults are largely unasserted; flip `max_iter` or a detector
threshold and the suite passes), booleans 1/4. Returns and arithmetic
100%. The best-tested layer in the repo under-constrains its
implementation substantially.
Status: open; the survivors name the tests to write.

### A3 — APEIRON: `accuracy_topk` and all of `evaluation.py` are untested
`tests/test_evaluation.py` exercises `accuracy` only. `accuracy_topk`
(the general form, used by harnesses) has no criteria of its own, and
`src/apeiron/evaluation/evaluation.py` is entirely untested — in fact
nothing imports it (dead code), and its docstring says it returns a float
ratio while the code returns an `(accuracy_percent, test_loss)` tuple.
Status: open; delete or specify `evaluation.py`, add topk criteria.

### A4 — APEIRON: blanket `*.toml` / `*.json` in `.gitignore`
Lines 2 and 161 ignore every TOML and JSON in the repository. Tracked
files survive, but any new config artifact is silently dropped — here it
ate a claim's recipe and sealed manifest from a commit before negations
were added. Upstream, the same rule means a contributor adding any TOML
or JSON fixture will silently not commit it.
Status: fixed in this workspace with negations; open upstream.

### R1 — reticuli: strict seatbelt broke every furnished claim on macOS
The first furnished-environment strict audit ever run (`apeiron-config`)
failed with `Operation not permitted` on the venv's bin. CPython's
launcher `realpath(3)`s its own path, stat'ing each ancestor as a
separate seatbelt check; the `/Users` mask denied them and the cache
subpath allow does not cover the road to the cache. Fixed by allowing
`file-read-metadata` on the ancestor chains of the workspace and cache
(existence only), with a red-then-green regression test.
Status: fixed, reticuli commit 038d734.

### R2 — reticuli: `ret init` gitignore patterns miss nested claim stores
Root-anchored patterns (`.reticuli/tmp/`) do not cover a claim packed in
place in a subdirectory, so its scratch residue was committable (and got
committed here before being caught). Fixed with `**/`-prefixed patterns
plus identity negations (`!**/reticuli.toml`, `!**/.reticuli/manifest.json`)
so a workspace's own blunt ignore rules cannot drop the recipe or root.
Status: fixed, reticuli commit 765cf4b.
