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
Confirmed by regrowth (2026-10-21, later): codex rebuilt the claim in
30 s and crosscheck accepted — M1/M2/M3 all earn root `99b37045…` — but
the regrown `metrics.py` contains no `accuracy_topk` at all; a seven-line
argmax accuracy is the entire equivalence class. `accuracy_topk` also has
zero callers anywhere in the repository (CLAUDE.md documents it as part
of the layer), so both it and `evaluation.py` are unconsumed public
surface: specify them or delete them.

### A4 — APEIRON: blanket `*.toml` / `*.json` in `.gitignore`
Lines 2 and 161 ignore every TOML and JSON in the repository. Tracked
files survive, but any new config artifact is silently dropped — here it
ate a claim's recipe and sealed manifest from a commit before negations
were added. Upstream, the same rule means a contributor adding any TOML
or JSON fixture will silently not commit it.
Status: fixed in this workspace with negations; open upstream.

### A5 — APEIRON: a blind regrowth passes the config gate at 148 lines of 403
`ret rebuild claims/config --producer codex` (local Codex CLI, 57 s, one
call): the producer regrew `configuration.py` from the pinned tests alone
and re-earned the gate at the same root. Crosscheck accepted — M1
(original), M2 (byte copy), M3 (regrowth) all audit `earned` at
`adec2e2b…` — and the proof is recorded in the claim store. The regrown
file is 148 lines against the original's 403: the gate's equivalence
class admits an implementation with no `LoggingCfg`, no `verbosity`, no
`metrics_output_path`, and none of the detector or updater
hyperparameter fields (`adwin_delta`, `ph_threshold`,
`kswin_window_size`, `jvp_rho_*`, `kfac_*`…). Any of those gaps breaks
the next layer instantly — `load_drift_detector` reads
`cfg.drift_detection.adwin_delta` on its first branch — so "passes the
config gate" is measurably not "substitutable into APEIRON". The
missing fields are precisely the criteria the drift-detection claim adds
back by composing on apeiron-config.
Status: working as designed; the divergence list is the test backlog for
`tests/test_config.py`.

### A6 — APEIRON: `EnsembleDetector` is hostage to the evidently module
`EnsembleDetector` and `ModelEvalDetector` live in
`model_performance_detector.py`, whose module-level hard dependency is
pandas (evidently itself is imported lazily at the call site). Neither
class needs pandas or evidently, but they are unimportable — and
untestable — wherever that module fails to import. Splitting them into
their own module would free them.
Status: open.

### A7 — APEIRON: the evidently guard in the drift tests is leaky
`tests/test_drift_detection.py` guards some tests with
`requires_evidently`, but `test_ensemble` and
`test_ensemble_without_sub_detectors_raises` are unguarded while using
the guarded names — on a host without evidently they die with
`NameError`/`ModuleNotFoundError` instead of skipping, the exact
deployment the file's own header comment promises to support. The flag
is also misnamed: `HAS_EVIDENTLY` actually measures whether pandas is
installed (the module imports evidently lazily), so a pandas-but-no-
evidently host reports `HAS_EVIDENTLY = True` and then fails at
detector construction.
Status: open.

### A8 — APEIRON: `requires-python >=3.13` but river 0.23.0 ships no 3.14 wheels
Furnishing the drift claim's environment on Python 3.14 refused: river
0.23.0 (the version `uv.lock` pins) has binary wheels only through
cp313; cp314 wheels start at river 0.24.2. APEIRON's declared support
for 3.14 therefore requires building river from source. The drift claim
pins a py3.13 environment to stay faithful to `uv.lock`.
Status: open upstream — bump river or narrow `requires-python`.

### A9 — APEIRON: drift tests kill 23% of injected faults
`ret assess claims/drift_detection --mutants 30`: mutation 0.23.
Comparison mutations 0/4 (of 222 sites) and arithmetic 0/3 (of 60) —
the severity thresholds and regime mapping in the detectors are barely
asserted; a detector that never escalates past `stable` at the wrong
boundary would pass. 63 tests, zero skips, with evidently furnished.
Status: open; boundary tests on severity/regime mapping are the gap.

### R4 — reticuli (observation): composed-claim mutation mixes layers
Assess on a composed claim samples mutation sites across every generated
file, component-supplied ones included — the drift claim's reported
survivors all sit in `configuration.py`, which its own tests rightly do
not cover (the component's claim does). A per-step or per-component
breakdown would keep the layer's own signal clean.
Status: noted as an enhancement idea, not a bug.

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

### R3 — reticuli: export dropped the environment file and manifest entries
`ret export` enumerated the recipe's literal `[claim] inputs` list, but
the pinned set also contains the declared `environment` file and an
`inputs_manifest`'s entries — all hashed into the root. A furnished or
format-2 claim therefore exported as a tar that could not re-derive its
own root; crosscheck's M2 leg (materialized via export/import) failed on
`apeiron-config` with `requirements.lock` missing. Export now uses the
kernel's enumeration for the claim and its exported components, with a
red-then-green round-trip test for both cases.
Status: fixed, reticuli commit 6fe0f82.
