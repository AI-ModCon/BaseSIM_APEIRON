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

### A10 — APEIRON: harness tests kill 16% of injected faults
`ret assess claims/model_harness --mutants 25`: mutation 0.16, the
weakest layer yet. Arithmetic, boolean, constant, and string mutations
all 0-for-sampled. The per-task transfer-metrics machinery
(`register_task`, `eval_past_tasks`, `task_diagonals` — documented in
CLAUDE.md and docs/tracking.md) has no criteria at all.
Status: open.

### A11 — APEIRON: the drift layer regrows at a quarter of its size
After R5 was fixed, the codex rebuild of apeiron-drift-detection
succeeded in 106 s: all five modules regrown blind against the 63-test
suite (evidently tests included — nothing skipped), crosscheck accepted
with M1/M2/M3 at root `afd55636…`, proof recorded. The compression says
how loose the boundary is: base.py 172→57 lines, statistical_detectors
316→113, model_performance_detector 462→81, the factory 97→25 — 1047
lines pass as 276. The evidently analysis pipeline in particular is
barely constrained: an 81-line stand-in satisfies every criterion the
suite states.
Status: evidence; each dropped behavior that matters is a test to write.

### A12 — APEIRON: updater math is nearly criterion-free
`ret assess claims/updaters --mutants 25`: mutation 0.12, the weakest
layer measured. Arithmetic mutations 0 of 4 detected across 290
arithmetic sites — the EWC penalty, the KFAC factors, and the JVP/SAM
robust update could compute the wrong quantity and the suite would
pass. For a continual-learning framework this is the highest-value gap
in the whole repository: the scientific core has the least constraining
tests. Mutation by layer so far: evaluation 0.76, config 0.36, drift
0.23, harness 0.16, updaters 0.12 — criteria strength falls as the code
gets closer to the science.
Status: open; numerical assertions on the regularizers are the ask.

### A13 — APEIRON: the JVP/SAM updater is interchangeable with a different algorithm
The codex rebuild of apeiron-updaters reached the sealed root in 68 s
(crosscheck accepted, proof recorded; apeiron-model-harness likewise in
37 s). The regrown `jvp_reg.py` is not the documented algorithm: it
perturbs parameters along the COMBINED current+historical gradient
where the original's Bertsimas formulation requires the current-only
gradient as the perturbation direction, and it drops the `jvp_rho_x`
input perturbation and `jvp_data_sign` entirely — yet every test
passes. The updater package regrows at 233 lines against 625. Together
with A12 this is the sharpest statement the track has produced: under
the current suite, APEIRON's core scientific algorithms are
interchangeable with different algorithms. The criteria to add are
numerical: fixed tiny model, fixed seed, assert the actual update
direction/penalty values for EWC, KFAC, and JVP.
Status: open; highest-value test work in the repository.

### A14 — coverage round one: the measured effect of the new tests
Five test files written against the ranked backlog in
[IMPROVEMENTS.md](IMPROVEMENTS.md), then pinned into their claims:

| layer | mutation before | after | what was added |
|---|---|---|---|
| config | 0.36 | 0.68 | the complete field set and every default |
| model-harness | 0.16 | 0.56 | transfer metrics, per-sample averaging |
| apeiron-updaters | 0.12 | 0.32 | EWC/KFAC/JVP numerics |
| drift-detection | 0.23 | 0.30 | severity-to-regime boundaries |

Same mutant sample sizes before and after. The prior updater regrowth
(the combined-gradient impostor of A13) fails 8 of the 11 new numerical
tests, so the hole that admitted it is closed. New roots: config
9cf77204, harness 5eaa8860, drift 0e01b720, updaters 51a4b922.

Also newly covered, outside any claim: the three entry points
(`tests/test_entry_points.py`), which run `main`, `drift_only`, and
`cl_only` end to end on a tiny harness — about a thousand lines that
had no tests at all. 262 tests pass, ruff and mypy clean.

### A17 — the loop closes: regrowth now converges on the real algorithm
Re-running the blind regrowths against the strengthened criteria is the
confirmation the whole method rests on. Codex regrew
`configuration.py` (59 s) and the updater package (63 s), both reaching
their new roots and crosschecking accepted with proofs recorded — and
the regrown `jvp_reg.py` now implements the DOCUMENTED algorithm:
perturbation along the negative, current-only gradient, the half/half
mix, and the signed `rho_x` input shift. The hole that admitted a
different algorithm (A13) is closed, demonstrated rather than asserted.
Regrowth sizes also moved toward the originals: `configuration.py` 148
→ 194 lines (of 403), the updater package 233 → 235 but with the right
mathematics inside. Updaters mutation at the resealed root: 0.37.
What remains uncovered is now mostly commentary, logging, and
defensive branches rather than behavior.

### A18 — APEIRON: no `[logging]` section means a run talks to wandb
`configure_backend(cfg)` returns `"wandb"` when `cfg.logging` is None,
so a TOML with no `[logging]` section tries to reach Weights & Biases —
surprising for a framework whose other defaults are all offline, and a
hang risk on an HPC node or in CI. `"none"` would be the defensible
default. Pinned by a test, so changing it is a deliberate decision.
Status: open.

### A19 — APEIRON: `evaluation.test()` under-reports loss by the batch count
Bringing the dead module under test (A3) found a real defect rather than
just unused code. `apeiron.evaluation.evaluation.test()` accumulates
`criterion(...).item()` per batch and divides by the SAMPLE count. Its own
comment says "use sum reduction so we can average correctly over dataset
size" — but it never requests `reduction="sum"`, and every harness
supplies the default mean reduction. Measured: 12 samples in 2 batches
reports 0.169 where the true mean loss is 1.017, exactly 6x low
(samples/batches). Reported loss therefore scales with batch size, which
a loss must not. Second defect: an empty loader returns a bare `0.0`
instead of a pair, so any caller unpacking the result raises TypeError.

Both are recorded as `strict=True` xfails in
`tests/test_evaluation_module.py` rather than pinned as behavior — a
passing test asserting a 6x error would make the error official — so they
flip green the moment either is fixed. The docstring is corrected to
state the reduction requirement and the real return type; behavior is
untouched.

A third, smaller defect: the function returns early when `total == 0`,
then guards the same condition again with `if total > 0: ... else: ...`,
so the else arm is dead code. It is the only unreachable branch left in
`src/` that is not an import fallback, a `__main__` guard, or a CUDA
call.

Recommendation: delete the module. Nothing in the framework calls it,
`BaseModelHarness.eval()` is the supported path, and it is wrong for
every realistic caller. It is referenced from `docs/api/evaluation.md`,
making it nominally public, so the removal is the team's call rather than
a mechanical cleanup.
Status: open — decision needed.

### A20 — APEIRON: a misspelled `aggregation` silently becomes the mean
`ContinuousMonitor._check_drift` dispatches on
`drift_detection.aggregation` with an `else: # Default to mean` arm, so
`aggregation = "geometric-mean"` (or any typo) runs the mean instead of
refusing. Same failure class as the unknown top-level config key fixed in
the determinism branch: a configuration the user believes is in effect,
silently is not. `detector_name` and `ensemble_voting` both raise on an
unknown value; this one should too.
Status: open. Current behavior is pinned by a test, so a fix is a
deliberate change.

### A21 — APEIRON: `ModelEvalDetector` prints, asserts, and fakes its score
Covering the evidently module surfaced three problems in its neighbour.
`ModelEvalDetector.update()` calls bare `print()` once per metric per
drift check — stdout rather than the logger, so it cannot be silenced by
`verbosity` and will flood an HPC job log. It guards its metric/reference
alignment with a bare `assert`, which `python -O` strips, turning a
mismatch into a silent wrong answer rather than a refusal. And every
`DriftSignal` it returns carries `drift_score=1` and `confidence=0.95`
with `# dummy` beside both, so the score column for an `EvalDetector`
run is a constant — and the ensemble averages that constant into its own
`drift_score`.
Status: open — replace the print with `logger.debug`, the assert with a
raise, and either compute a real score or document the sentinel.

### A22 — APEIRON: the detector's regime can outvote its own verdict
`ModelPerformanceDetector` decides detection from the current window's
drift share but the regime from the ten-window rolling mean, and the two
disagree in both directions. A first strong spike reports
`drift_detected=True` with regime `CONTINUAL_LEARNING`, because the mean
is still near zero; a quiet window after a drifted stretch reports
`drift_detected=False` while still recommending `FINE_TUNING`. Both are
pinned by tests now. Whether that is the intended hysteresis or an
accident is a design question for the team — it is undocumented either
way.
Status: open — document or reconcile.

### A15 — APEIRON: the shared DummyHarness could not drive the monitor
`tests/conftest.py`'s `DummyHarness.get_stream_dataloader` returned a
one-element tuple, which `ContinuousMonitor` cannot iterate (it unpacks
batches directly), so the shared fixture was unusable for any
loop-level test — nothing had ever tried. Every example harness returns
the loader itself. Fixed here while writing the entry-point tests.
Status: fixed in this workspace; carry upstream with the entry-point
tests.

### A16 — APEIRON: `--trigger-at` without `--schedule fixed` dies obscurely
`cl_only.py` defaults to `--schedule periodic`, so passing only
`--trigger-at 7,19` (the documented way to name windows) raises
`ValueError: --schedule periodic needs --period or --budget` — an error
about flags the user never mentioned. Either infer `fixed` from
`--trigger-at` or name the real problem. The current behavior is pinned
by a test so a fix is a deliberate change.
Status: open.

### A23 — the strengthened criteria reject the stand-in that used to pass
The decisive measurement for the evidently work. Before the new tests,
a blind codex regrowth of the drift layer compressed 1047 lines to 276
and passed the whole gate (A11). Against the resealed claim it **fails**:
19 of 120 tests reject it, every one in an evidently-dependent path, and
the regrown `model_performance_detector.py` punts on the hard part
outright with `raise NotImplementedError('Report integration requires
Evidently')`. The boundary now demands the real pipeline — the regime
ladder and the four result-shape refusals cannot be satisfied by a
stub.

One confound, stated: the producer cannot run the gate in this room
(R6 — the claim's environment is py3.13/river while the host default is
3.14), so it wrote a stub and never learned it had failed. A second
regrowth with the furnished venv on the producer's PATH is the clean
test of whether a reconstruction *can* satisfy the boundary, as opposed
to whether this handicapped one did.
Resolved the same day. Rerunning the rebuild with the furnished venv on
the producer's PATH — so codex could actually run the gate and iterate —
succeeded in 148 s, crosscheck accepted with M1/M2/M3 at root
`d0eec347…`, proof recorded. So the criteria are satisfiable; the first
failure was R6, the producer flying without gate feedback.

And the regrowth is now a different program. `model_performance_detector.py`:

| criteria | regrown size | what it does |
|---|---|---|
| before | 81 lines | never imports evidently |
| after | 130 lines | real `DataDriftPreset`, real `DriftedColumnsCount` lookup, the result-shape refusals |

The other modules barely moved (base 57 to 49, statistical 113 to 110,
factory 25 to 24) — they were already constrained by the severity tests.
This is the clearest demonstration the track has produced of criteria
changing what an independent reconstruction is *forced* to build, rather
than merely what it is allowed to pass.
Status: closed.

### A27 — the logger's explicit delegation is a freedom, not a requirement
`apeiron/logger/logger.py` spends 124 logical lines forwarding eighteen
methods and properties to its two halves, one wrapper each. The regrowth
does it in 40 lines with a single `__getattr__` and passes all 108 logger
tests.

Not a defect — the tests never asked for explicit methods, and the proxy
is arguably the better code — but the consequence is worth a decision: a
dynamic proxy gives up static typing, IDE completion and `mypy` checking
on the whole facade, which for a framework other people write harnesses
against is a real cost. If explicit delegation is wanted, something has to
demand it (a typing test, or `mypy --strict` over a consumer). If not, the
84 extra lines could go.

Found with `claims/scripts/surface_diff.py`, which compares a regrown room
against `src/` on logical lines and public surface, resolving inheritance
and dynamic delegation so only genuine divergences are reported.
Status: open — a decision, not a bug.

### A26 — a test that exists is not a criterion until a claim pins it
The whole-stack regrowth surfaced this by dropping features that *were*
tested. Diffing the public surface of each reconstruction against the
original showed `BaseModelHarness.save_ckpt`, `ckpts_enabled` and
`build_checkpoint_payload` simply absent — yet `tests/test_checkpointing.py`
covers all three. The reason: that file was not listed in any claim's
inputs, so it never entered a gate, so the boundary never mentioned
checkpointing and the producer was right to leave it out.

Audited across the suite: 22 of 27 test files were pinned; five were not,
worth 25 tests. Two were mine and cross-cutting
(`test_checkpointing.py`, and `test_device_and_guards.py`, which covered
config, drift and updater behaviour in one file and so belonged to no
single layer). They are now split along layer boundaries and pinned —
device selection into apeiron-config, ensemble nesting into
apeiron-drift-detection, the out-of-order updater commits folded into the
numerics file, checkpointing into apeiron-model-harness.

Three stay deliberately unpinned, and should: `test_rocm.py` needs a GPU
and skips entirely here, so a gate over it would be vacuous;
`test_validation_tests.py` is the MNIST golden regression and needs the
dataset, which is out of bounds for a hermetic room;
`test_simple.py` asserts `True`.

The general rule worth stating for the team: coverage measures what the
suite executes, claims measure what the suite *demands*. A file can do the
first without the second, and only the second survives a reconstruction.
Status: fixed here; the rule belongs in CONTRIBUTING if claims are adopted.

### A25 — APEIRON: every reported FLOPs figure rests on an unverified table
The sharpest finding from regrowing the whole stack.
`apeiron/profilers/aten_flops_map.py` maps 169 ATen operation names to a
FLOPs-per-element cost, and `FLOPSProfiler` multiplies those costs by call
counts to produce every `cperf_*` figure APEIRON logs. A blind regrowth
reproduced **14 of the 169 entries** and passed the entire profiler gate:
155 operation costs are completely unconstrained. A typo scaling one
common operation's cost would silently rescale the FLOPs reported by every
experiment, and nothing would fail.

Fixed structurally rather than by writing 169 assertions: the table is
reference DATA, not logic, so the claim now declares it `class = "pinned"`
instead of `generated`. Its bytes are inside the root — appending one byte
makes `ret verify` report a different claim — and a rebuild receives the
table rather than inventing it, which is the correct division: regrow the
profiler, hand over the constants. Root moved 25bb0aa2 to 6fe145d6.

The same question is worth asking of the reference CSVs under
tests/references/: data that decides a verdict belongs inside an
identity, not beside it.
Status: addressed in the claim; upstream, consider whether the table wants
a sanity test (costs positive, a few known operations correct).

### A24 — a nine-layer recursive rebuild compounds the producer's variance
`ret rebuild claims/entrypoints --recursive` regrew six layers in about
eleven minutes — config, evaluation, model-harness, updaters, logger,
profilers — and then aborted when the model-harness gate failed on one
test. Rebuilding that same layer alone immediately afterwards succeeded in
37 s, so the failure was R7 variance, not a boundary problem.

The arithmetic matters for how this is used. If a layer regrows
successfully with probability p, nine chained layers all succeed with p^9:
at p = 0.8 that is 13%. An atomic recursive run therefore measures the
producer's luck, not the boundaries. Regrowing each layer independently
against its own criteria, with its sealed components supplied and a couple
of retries, measures what we actually want to know — that every boundary
is satisfiable by an independent reconstruction — and it is what
`claims/README.md` recommends.
Status: methodology; drives the per-layer regrowth script.

### R12 — reticuli: `status` reports a proof that was earned on other bytes
A recorded crosscheck proof is stored on the claim's manifest and names the
roots it compared (`m2: f4370ebed255…`). Reseal the claim — which happens
whenever criteria are strengthened — and the proof stays on the manifest
naming the superseded root, while `ret status` keeps printing
`proof  recorded`. One claim in this ladder lost its proof outright in a
reseal and correctly showed `proof  none`; three kept a `recorded` line
for a proof earned on bytes that no longer exist.

`status` already does this correctly for the other evidence — it trusts an
audit receipt and an assess measurement "only while the recorded root still
matches" — so the proof line looks like an oversight rather than a design
choice. The fix is the same comparison.

Practical consequence while it stands: after any reseal, treat every
recorded proof as stale until re-earned. `claims/scripts/regrow.sh <layer>`
re-earns one.
Status: open; worth reporting upstream.

### R11 — reticuli: a rebuild reports success when it mints a different claim
Pinning the ATen cost table as `class = "pinned"` (A25) exposed this.
`rebuild` snapshots what the producer must not touch — the recipe plus the
declared inputs — and checks it before sealing. Pinned *produce outputs*
are not in that snapshot, although they are equally inside the root. The
producer duly rewrote `aten_flops_map.py`, so the rebuild sealed to root
`3bf74dd0…` instead of the source's `6fe145d6…`, printed `rebuilt
3bf74dd0…`, and exited 0.

Nothing unsound happened — identity tracked the bytes exactly as it
should, and `crosscheck` caught it immediately ("M3 earned at a different
root") — but a successful-looking rebuild that is quietly a *different
claim* is easy to miss, and `rebuild` is the verb people read for success.
Either add pinned outputs to the watched set, or say so plainly:
`rebuilt <root> — NOT the source root <root>`.

APEIRON side: the table is better modelled as a pinned INPUT than a
pinned output. Inputs are in the watched set, so the producer cannot touch
them at all, and "constants the implementation reads" is what an input
*is*. Changing it is queued behind the regrowth run currently using these
claims.
Status: open on both sides.

### R10 — reticuli: `status --files` omits pinned produce outputs
A produce step declared `class = "pinned"` is inside the root — verified
directly: appending a byte to one makes `ret verify` report a different
claim — but it does not appear in `ret status --files`, which lists pinned
inputs, generated outputs, the environment and the gate verdict. The
per-file account is the view an auditor reads to see what a claim covers,
so a covered file missing from it understates the claim.
Status: enhancement idea; cosmetic, but it is the view that answers "what
does this claim pin?".

### R8 — reticuli: a dependent silently under-declaring a component's files
Adding a file to a component claim (`apeiron/model/__init__.py`) left every
dependent's recipe listing the component's *old* output set. The warm pack
still passed, because the file was present on disk from `ret pull`; the
deep audit failed, because its room is built from declarations alone and
the file was never materialized. The failure surfaced as one puzzling
test (`re_exports_the_harness`) inside an otherwise green layer.

Working as designed — declarations are the authority, and the audit is
what checks them — but the gap between "passes warm" and "passes
declared" is exactly where a composed claim rots, and nothing points at
it. `ret pack` could compare each `from = "<component>"` step set against
the component's own generated outputs and warn on a subset.
Status: enhancement idea.

### R9 — reticuli: resealing a leaf orphans its dependents' component copies
`ret pull` copies a component claim *into* the dependent's store, and the
dependent's manifest link names that copy's root. Reseal the leaf — which
happens every time its criteria are strengthened — and the link now names
a root that exists nowhere: not in the leaf (it moved) and not in the
registry. The symptom is a later `ret pull` of the dependent failing with
`unresolved components`, naming leaves that are plainly present, which
sends you looking in the wrong place.

The fix is procedural: re-pull and reseal bottom-up in dependency order
after any leaf changes. Worth automating, since the order is a property
of the graph and reticuli already computes the chain.
Status: enhancement idea; the workaround is documented in claims/README.md.

### R5 — reticuli: recipe-in-place packing dropped component links
A produce step's `from = "<component>"` is honored by incremental
rebuild, deep audit, and export only through the manifest's component
links — which only the flag path (`pack --component`) wrote. A composed
claim sealed from its reticuli.toml carried no links, so `ret rebuild`
took the plain path and built a blind room missing the supplied layer:
the first codex rebuild of apeiron-drift-detection failed on
`No module named 'apeiron.config'`. pack_declared now resolves each
`from` against the claim's drawer (refusing with a remedy if it is not
registered) and writes the links the flag path writes.
Status: fixed, reticuli commit 0e53cb2.

### R6 — WITHDRAWN: producers do get the furnished environment
Originally recorded as a gap: codex appeared unable to run a furnished
claim's gate, because the claim's environment is py3.13/river while the
host default interpreter is 3.14. That was wrong. `_produce` in
`_kernel/build.py` already prepends the furnished venv's `bin` to the
producer's scrubbed PATH, exactly as it does for gates, so `python3` in a
gate command resolves to the claim's own interpreter inside a rebuild.

Measured directly afterwards: the same rebuild invocation that had failed
was re-run unchanged and succeeded in 95 s, reaching the sealed root. The
original failure was producer variance, not a missing mechanism. Nothing
to fix.
Status: withdrawn — the mechanism works as designed.

### R7 — measurement property: regrowth is a sampling instrument
Three blind regrowths of the same claim (apeiron-drift-detection at root
`d0eec347`) gave two successes and one failure: 95 s and 148 s reaching
the root, and one run that stubbed the evidently integration with `raise
NotImplementedError` and failed 19 of 120 tests. Same criteria, same
producer, same host.

So a single failed regrowth is NOT evidence that criteria are
unsatisfiable, and a single success is not evidence they are tight. Read
regrowth as a sample: repeat it, and treat the *distribution* of what
comes back as the measurement. Divergences that recur across runs are
findings about the boundary; divergences that appear once are findings
about the producer.
Status: methodology note — applies to every regrowth result in this log.

### R4 — reticuli: a composed claim's mutation score is mostly not its own
Assess samples mutation sites across every generated file in a claim,
component-supplied ones included, so for a deep claim the reported figure
says almost nothing about the layer itself. Measured on this ladder, the
share of a claim's generated tree that is its own code:

| claim | own files | reported mutation | meaningful? |
|---|---|---|---|
| config, evaluation, profilers | all | 0.68, 0.76, 0.68 | yes |
| logger, drift-detection | 5 of 7 | 0.32, 0.60 | mostly |
| updaters | 7 of 14 | 0.37 | half |
| model-harness | 2 of 7 | 0.56 | diluted |
| trainer | 2 of 24 | 0.12 | no |
| monitor | 3 of 32 | 0.12 | no |
| entrypoints | 5 of 36 | 0.20 | no |

Checked directly rather than assumed: every surviving mutant in a 30-mutant
run against apeiron-trainer landed in `evaluation.py` or
`configuration.py` — component code its own gate rightly does not test,
because those components' own claims test it. So 0.12 for the trainer is
not "the trainer is barely specified"; it is "92% of the sample missed the
trainer".

Scoping mutation to a claim's own produce steps — or reporting per-step
rates — would make the number mean what a reader assumes it means. Until
then, read composed-claim mutation only alongside the own-files share.
Status: enhancement idea; the numbers above are the workaround.

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
