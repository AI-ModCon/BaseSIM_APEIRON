# Reconstruction log

Findings from putting APEIRON under reticuli, layer by layer. Each claim
pins a layer's tests as its acceptance boundary and treats the
implementation as regrowable; what the regrowth machinery trips over is
recorded here. Entries say where the problem lives (APEIRON, reticuli, or
this workspace), why it matters, and what has been done about it.

Claims live under `claims/`; sealed roots are in each claim's
`.reticuli/manifest.json`. Source snapshots referenced below are from
`main` at 63a877a.

## Index

`Kind` says what a finding *is*, so the list of things to act on is
derivable from this file rather than from memory: **defect** needs a code
change, **decision** needs the team's judgment, **criteria-gap** was the
suite failing to demand something and is closed by the tests written in
response, **measurement** and **methodology** are records with nothing to
act on. State is **open** (needs diagnosis or a decision), **fixed here**
(a patch exists in this workspace and wants a PR upstream), or **closed**.

Numbers never change — they are referenced from test docstrings and from
pinned claim files, where a renumber would move claim roots.

| # | kind | state | finding |
|---|---|---|---|
| A1 | defect | open | APEIRON: the package `__init__` welds every layer together |
| A2 | measurement | closed | APEIRON: config tests kill 36% of injected faults |
| A3 | decision | open | APEIRON: `accuracy_topk` and all of `evaluation.py` are untested |
| A4 | defect | fixed here | APEIRON: blanket `*.toml` / `*.json` in `.gitignore` |
| A5 | measurement | closed | APEIRON: a blind regrowth passes the config gate at 148 lines of 403 |
| A6 | defect | open | APEIRON: `EnsembleDetector` is hostage to the evidently module |
| A7 | defect | open | APEIRON: the evidently guard in the drift tests is leaky |
| A8 | defect | open | APEIRON: `requires-python >=3.13` but river 0.23.0 ships no 3.14 wheel |
| A9 | measurement | closed | APEIRON: drift tests kill 23% of injected faults |
| A10 | measurement | closed | APEIRON: harness tests kill 16% of injected faults |
| A11 | measurement | closed | APEIRON: the drift layer regrows at a quarter of its size |
| A12 | measurement | closed | APEIRON: updater math is nearly criterion-free |
| A13 | measurement | closed | APEIRON: the JVP/SAM updater is interchangeable with a different algor |
| A14 | measurement | closed | coverage round one: the measured effect of the new tests |
| A15 | defect | fixed here | APEIRON: the shared DummyHarness could not drive the monitor |
| A16 | defect | open | APEIRON: `--trigger-at` without `--schedule fixed` dies obscurely |
| A17 | measurement | closed | the loop closes: regrowth now converges on the real algorithm |
| A18 | defect | open | APEIRON: no `[logging]` section means a run talks to wandb |
| A19 | defect | open | APEIRON: `evaluation.test()` under-reports loss by the batch count |
| A20 | defect | open | APEIRON: a misspelled `aggregation` silently becomes the mean |
| A21 | defect | open | APEIRON: `ModelEvalDetector` prints, asserts, and fakes its score |
| A22 | decision | open | APEIRON: the detector's regime can outvote its own verdict |
| A23 | measurement | closed | the strengthened criteria reject the stand-in that used to pass |
| A24 | methodology | closed | a nine-layer recursive rebuild compounds the producer's variance |
| A25 | defect | fixed here | APEIRON: every reported FLOPs figure rests on an unverified table |
| A26 | methodology | fixed here | a test that exists is not a criterion until a claim pins it |
| A27 | decision | open | the logger's explicit delegation is a freedom, not a requirement |
| A28 | defect | closed | the two side runners duplicate a loop they could inherit |
| A29 | criteria-gap | closed | the framework's main loop is not pinned by the layer that owns it |
| A30 | criteria-gap | closed | the forgetting metric a CL paper reports was unverified |
| A31 | defect | open | the transfer row cannot be joined to the event that produced it |
| A32 | criteria-gap | closed | two more logged quantities nobody named |
| A33 | measurement | open | the kept-ratio's noise floor, per layer |
| A34 | methodology | open | the divergence kinds are a triage function, and recurrence is its evidence |
| A35 | measurement | open | the paired experiment: strengthening criteria did move what a reconstruction is forced to build |
| A36 | measurement | open | does the documentation close the gaps? negative on drift, strongly positive on config defaults -- with a worse failure mode |
| A37 | measurement | open | the three river detectors are domesticated: pinned behaviour, independent redos, river replaceable but not yet replaced |
| A38 | measurement | open | behaviour, not lines: four reconstructions match the source on 441 probe cases off the pinned set |
| A39 | defect | open | APEIRON: `EnsembleDetector` reports a nondeterministic regime on a tied vote |
| A40 | measurement | open | the methodology across all ten layers: 1,049 probe cases, zero divergences |
| A41 | defect | fixed here | APEIRON: a run does not reproduce itself -- `seed` never reaches the loaders' RNG |
| A42 | measurement | open | a training RESULT pinned as a claim: reduced MNIST run, 1,665 metrics exact, 9/9 layers earned |
| A43 | methodology | closed | pinning a result: six unconstrained degrees of freedom, found by adversarial search, then reproduced exactly |
| A44 | measurement | closed | cross-family: Claude Opus reproduces the same training result first try, from a different implementation |
| A45 | measurement | closed | the ten proofs re-earned: twenty blind rebuilds, two families, thirty roots agree |
| A46 | measurement | open | CORRECTS A40 and A45: the regrown implementations are NOT behaviourally equivalent -- 20 rooms probed, 2 identical |
| A47 | measurement | closed | QUALIFIES A46 for `monitor`: loop equivalent; pinning the contract took 124 divergences to 2, both error prose |
| A48 | APEIRON | open | the monitor's subclass seams are unpinned by the module's own tests, and both side runners read them |
| R19 | methodology | fixed here | a probe that dies on the first absent name, and a weights comparison that never ran the trainer |
| R20 | methodology | fixed here | the probe silently compared the source with itself again -- a namespace portion losing to the editable install |
| R21 | methodology | fixed here | divergence counts conflated granted latitude with contract, mis-ranking the worklist |
| A49 | measurement | closed | `profilers` 117/111 is 4/3 contract divergences; pinning the CSV contract took codex to 1 |
| A50 | APEIRON | open | a measured block that RAISES is still reported as a completed zero-FLOP measurement, into the CSV |
| R22 | methodology | fixed here | an exception's TYPE is contract and its wording is prose; classified automatically rather than listed |
| A51 | APEIRON | closed | nothing pinned how a `verbosity` string becomes a level; one rebuild CRASHED on `"info"`. 14/10 contract divergences to 0 |
| A52 | APEIRON | open | an unrecognised verbosity silently becomes INFO, so `verbosity = "WARN"` logs MORE than asked |
| A53 | measurement | open | all twenty rooms re-measured under the guard: 681 raw divergences are 335 on contract, and three layers now sit at 0 |
| R23 | methodology | fixed here | the probe measured a room whose gate never passed, and called it a result |
| A54 | APEIRON | closed | nothing pinned WHEN a detector fires; one rebuild replaced river with its own algorithms and passed. 23 contract divergences to 2 |
| A55 | APEIRON | open | a non-string `ensemble_voting` leaks `AttributeError: 'int' object has no attribute 'strip'` |
| R24 | methodology | fixed here | a "must raise" criterion without matching "must accept" controls licensed a STRICTER implementation |
| A56 | APEIRON | open | the last five layers pinned: device precedence, config merge, metric guards, harness seams, trainer loop, schedule controls |
| A57 | APEIRON | open | seven divergences across four layers where the RECONSTRUCTION is right and the source is wrong |
| A58 | APEIRON | open | **per-layer equivalence does not imply COMPOSABILITY**: nine layers each passed, and the assembled framework would not import |
| A59 | APEIRON | open | `grad_accumulation_steps = 0` divides the loss by zero: `inf` loss, NaN weights, no error |
| A60 | APEIRON | closed | the logged forgetting loss was unpinned; a rebuild reported the full loss as the penalty with bit-identical weights |
| A61 | APEIRON | open | the regime ladder's evaluation ORDER is unpinned: inverted thresholds give the least adaptation, a rebuild gave the most |
| A62 | APEIRON | open | third edge of A58: cross-layer method SIGNATURES unpinned; the assembled framework imported and died mid-training |
| A63 | APEIRON | open | fourth edge: WHICH method of an injected collaborator a layer calls. `print_performance` documents `logger.info()`; a rebuild duck-typed to `log` |
| A64 | APEIRON | open | the metrics file's SCHEMA is unpinned: the run trains, completes, and writes a file no analysis can read |
| R25 | methodology | fixed here | my own validation was vacuous: `pytest --rootdir=claims/<layer>` puts the claim's sealed `apeiron/` first on sys.path |
| R26 | methodology | WITHDRAWN | a forced-drift count table, measured once against an UNSEEDED fixture; the source varies run to run |
| A65 | measurement | closed | LOCATED: the warmup window is evaluated but not logged; the rebuild logged it, shifting every row by two |
| A66 | APEIRON | open | the metrics file's contents are coupled to the PROFILER's warmup: raising `warmup_iters` silently deletes rows |
| R1 | bug (fixed) | closed | reticuli: strict seatbelt broke every furnished claim on macOS |
| R2 | bug (fixed) | closed | reticuli: `ret init` gitignore patterns miss nested claim stores |
| R3 | bug (fixed) | closed | reticuli: export dropped the environment file and manifest entries |
| R4 | enhancement | closed | reticuli: a composed claim's mutation score is mostly not its own |
| R5 | bug (fixed) | closed | reticuli: recipe-in-place packing dropped component links |
| R6 | withdrawn | closed | WITHDRAWN: producers do get the furnished environment |
| R7 | methodology | closed | measurement property: regrowth is a sampling instrument |
| R8 | enhancement | closed | reticuli: a dependent silently under-declaring a component's files |
| R9 | enhancement | closed | reticuli: resealing a leaf orphans its dependents' component copies |
| R10 | enhancement | closed | reticuli: `status --files` omits pinned produce outputs |
| R11 | enhancement | closed | reticuli: a rebuild reports success when it mints a different claim |
| R12 | enhancement | closed | reticuli: `status` reports a proof that was earned on other bytes |
| R13 | bug | closed | reticuli: no codex rebuild has ever recorded a cost |
| R14 | bug | closed | reticuli: a proof recorded which oracle it survived nowhere |
| R15 | bug | closed | reticuli: `--producer codex` reached no adapter, and the adapter could not have run |
| R16 | bug | fixed here | reticuli: the producer contract tells a model to read pinned DATA in full |
| R17 | bug | fixed here | reticuli: the claude producer records no oracle, so its proofs name none |
| R18 | enhancement | open | reticuli: a proof holds one producer, and which one is positional |

## Findings

### A1 — APEIRON: the package `__init__` welds every layer together
`src/apeiron/__init__.py` eagerly imports config, harness, monitor,
detectors, trainer, and logger, so `import apeiron.config.configuration`
needs river, evidently, wandb, and mlflow installed — the whole dependency
tree to parse a TOML. No layer is importable in isolation, which blocks
claiming any layer in place and is why `claims/config/` carries copies and
a namespace package instead. Candidate fix: a lazy `__init__` (PEP 562
module `__getattr__`) that keeps `from apeiron import Config` working.
Kind: defect
Status: open; first candidate for an upstream branch.

### A2 — APEIRON: config tests kill 36% of injected faults
`ret assess claims/config --mutants 25`: mutation score 0.36. Branch
mutations 0/4 detected, constants 0/4 (of 88 constant sites — the
dataclass defaults are largely unasserted; flip `max_iter` or a detector
threshold and the suite passes), booleans 1/4. Returns and arithmetic
100%. The best-tested layer in the repo under-constrains its
implementation substantially.
Kind: measurement
Status: superseded — the config suite now kills 0.68 (was 0.36) after the
complete field-set and default-value tests. The survivors named here are
pinned.
### A3 — APEIRON: `accuracy_topk` and all of `evaluation.py` are untested
`tests/test_evaluation.py` exercises `accuracy` only. `accuracy_topk`
(the general form, used by harnesses) has no criteria of its own, and
`src/apeiron/evaluation/evaluation.py` is entirely untested — in fact
nothing imports it (dead code), and its docstring says it returns a float
ratio while the code returns an `(accuracy_percent, test_loss)` tuple.
Kind: decision
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
Kind: defect
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
Kind: measurement
Status: working as designed; the divergence list is the test backlog for
`tests/test_config.py`.

### A6 — APEIRON: `EnsembleDetector` is hostage to the evidently module
`EnsembleDetector` and `ModelEvalDetector` live in
`model_performance_detector.py`, whose module-level hard dependency is
pandas (evidently itself is imported lazily at the call site). Neither
class needs pandas or evidently, but they are unimportable — and
untestable — wherever that module fails to import. Splitting them into
their own module would free them.
Kind: defect
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
Kind: defect
Status: open.

### A8 — APEIRON: `requires-python >=3.13` but river 0.23.0 ships no 3.14 wheels
Furnishing the drift claim's environment on Python 3.14 refused: river
0.23.0 (the version `uv.lock` pins) has binary wheels only through
cp313; cp314 wheels start at river 0.24.2. APEIRON's declared support
for 3.14 therefore requires building river from source. The drift claim
pins a py3.13 environment to stay faithful to `uv.lock`.
Kind: defect
Status: open upstream — bump river or narrow `requires-python`.

### A9 — APEIRON: drift tests kill 23% of injected faults
`ret assess claims/drift_detection --mutants 30`: mutation 0.23.
Comparison mutations 0/4 (of 222 sites) and arithmetic 0/3 (of 60) —
the severity thresholds and regime mapping in the detectors are barely
asserted; a detector that never escalates past `stable` at the wrong
boundary would pass. 63 tests, zero skips, with evidently furnished.
Kind: measurement
Status: superseded — the drift suite now kills 0.60 (was 0.23) after the
severity-boundary and evidently-pipeline tests. Comparison mutations went
from 0 of 4 to 4 of 4 across 222 sites.
### A10 — APEIRON: harness tests kill 16% of injected faults
`ret assess claims/model_harness --mutants 25`: mutation 0.16, the
weakest layer yet. Arithmetic, boolean, constant, and string mutations
all 0-for-sampled. The per-task transfer-metrics machinery
(`register_task`, `eval_past_tasks`, `task_diagonals` — documented in
CLAUDE.md and docs/tracking.md) has no criteria at all.
Kind: measurement
Status: superseded — the harness suite now kills 0.56 (was 0.16) after the
transfer-metrics and checkpointing tests.
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
Kind: measurement
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
Kind: measurement
Status: addressed — `tests/test_updater_numerics.py` pins the EWC Fisher commit
and penalty, the KFAC factors and structured penalty, and the JVP/SAM
perturbation direction. Updaters mutation 0.12 to 0.37; arithmetic
mutations 0 of 4 to 2 of 2 in the re-measured run.
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
Kind: measurement
Status: addressed — the combined-gradient impostor described here fails 8 of the
11 numerical tests written in response. Every regrowth since builds the
current-only perturbation.
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
Kind: measurement
Status: recorded.

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
Kind: measurement
Status: recorded.

### A18 — APEIRON: no `[logging]` section means a run talks to wandb
`configure_backend(cfg)` returns `"wandb"` when `cfg.logging` is None,
so a TOML with no `[logging]` section tries to reach Weights & Biases —
surprising for a framework whose other defaults are all offline, and a
hang risk on an HPC node or in CI. `"none"` would be the defensible
default. Pinned by a test, so changing it is a deliberate decision.
Kind: defect
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
Kind: defect
Status: open — decision needed.

### A20 — APEIRON: a misspelled `aggregation` silently becomes the mean
`ContinuousMonitor._check_drift` dispatches on
`drift_detection.aggregation` with an `else: # Default to mean` arm, so
`aggregation = "geometric-mean"` (or any typo) runs the mean instead of
refusing. Same failure class as the unknown top-level config key fixed in
the determinism branch: a configuration the user believes is in effect,
silently is not. `detector_name` and `ensemble_voting` both raise on an
unknown value; this one should too.
Kind: defect
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
Kind: defect
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
Kind: decision
Status: open — document or reconcile.

### A15 — APEIRON: the shared DummyHarness could not drive the monitor
`tests/conftest.py`'s `DummyHarness.get_stream_dataloader` returned a
one-element tuple, which `ContinuousMonitor` cannot iterate (it unpacks
batches directly), so the shared fixture was unusable for any
loop-level test — nothing had ever tried. Every example harness returns
the loader itself. Fixed here while writing the entry-point tests.
Kind: defect
Status: fixed in this workspace; carry upstream with the entry-point
tests.

### A16 — APEIRON: `--trigger-at` without `--schedule fixed` dies obscurely
`cl_only.py` defaults to `--schedule periodic`, so passing only
`--trigger-at 7,19` (the documented way to name windows) raises
`ValueError: --schedule periodic needs --period or --budget` — an error
about flags the user never mentioned. Either infer `fixed` from
`--trigger-at` or name the real problem. The current behavior is pinned
by a test so a fix is a deliberate change.
Kind: defect
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
Kind: measurement
Status: closed.

### A30 — the forgetting metric a CL paper reports was unverified
`ContinuousTrainer.compute_bwt` computes backward transfer — the standard
continual-learning forgetting number — after every adaptation, and logs it.
Nothing in the suite mentioned it: zero hits for `compute_bwt` or `bwt`
across every test file. The trainer regrowth dropped the method outright
and passed, which is how it surfaced.

Its sign convention is the trap, and the docstring says so: with accuracy,
negative means forgetting; with a lower-is-better metric like SLAC-FEL's
MAE, *positive* means forgetting. A caller that assumes
negative-means-forgetting misreports the headline result of a run.

`fwt` (forward transfer, computed inline in the same place as
`cur - pre_cl`) was equally unverified, as was `_log_validation`, which
names every metric by its harness label and is how a run's validation
numbers get into the CSV at all.

Twenty-one tests now pin all three: BWT's mean over retained tasks, its
positional pairing of each task's current score with its own diagonal, the
`metric_index` selection, `None` before any task is registered, both sign
directions; FWT's gain, loss and zero cases; and the validation payload's
naming for both domains, its absent-domain behaviour and its
annotate-don't-advance logging. Trainer root 1b7fe903.
Kind: criteria-gap
Status: closed.

### A31 — the transfer row cannot be joined to the event that produced it
The row carrying `fwt` and `bwt` — the two headline transfer numbers —
contains no `drift_event_id`, while the validation rows and the
per-iteration loss rows both do. In the metrics CSV the forgetting numbers
therefore cannot be joined to the adaptation that produced them except by
step ordering. Recorded as a `strict=True` xfail so it flips green when
fixed, with its present behaviour pinned alongside.
Kind: defect
Status: open — a one-line fix, but it changes the CSV schema.

### A32 — two more logged quantities nobody named
Auditing every key the framework logs against the suite (16 of them) left
three unnamed. The `cperf_` prefix on profiler fields is load-bearing, not
cosmetic: `docs/profiler.md` documents it as the profiler namespace, and
the determinism work excludes wall-clock rows from a byte-for-byte CSV
comparison by matching it — rename it and that comparison silently starts
comparing timings. `decision_idx` and `trigger_count` on a scheduled run's
decision row were also unnamed, and `trigger_count` is subtle: the row
reports the count *including* the decision being logged.

Pinned at their own layers, per A29: the monitor's prefix in
`tests/test_logged_schema.py`, the runners' shared prefix and decision row
in `tests/test_runner_schema.py`. Monitor root f3515468, entrypoints
97785170.
Kind: criteria-gap
Status: closed.

### A29 — the framework's main loop is not pinned by the layer that owns it
`ContinuousMonitor.run()` — the monitoring loop the whole framework exists
to run — is demanded by no test in the apeiron-monitor claim: its three
test files call `_check_drift`, `_handle_drift` and the constructor, but
`.run()` appears zero times. A regrowth of that layer duly omitted it, and
passed. The loop is pinned one layer up instead, by apeiron-entrypoints,
whose tests drive `main()`.

Composition therefore covers it, but the component was not independently
meaningful: anyone auditing apeiron-monitor alone was auditing a monitor
that need not have a loop.

**Fixed.** Eleven tests now drive it at this layer: the loop running to its
stream budget, every batch evaluated, one drift check per interval with the
buffer left empty, a zero interval disabling checks, the stream extended
once to prime plus once per exhausted window, a detection dispatching
adaptation with the signal passed through, a quiet stream never adapting,
and the lazy construction of the detector and trainer (the property that
lets a drift-only run skip building an optimizer it never steps).

Re-regrown: `run` has left the dropped-names list — the reconstruction now
implements the loop — and the layer's kept ratio moved 33% to 38%. Monitor
root d585d03d -> d64a5d81.
Kind: criteria-gap
Status: closed.

### A28 — the two side runners duplicate a loop they could inherit
The entrypoints reconstruction regrew `drift_only.py` in **24 logical lines
against APEIRON's 242** — by subclassing `ContinuousMonitor` and overriding
`_handle_drift` to count the event instead of training. `cl_only.py` came
back at 137 against 355 the same way. Both originals open by saying they
"mirror ``ContinuousMonitor``'s loop — per-batch evaluation, interval drift
checks, the same aggregation, the same FLOPs profiling, the same logged
field names"; the reconstruction simply inherited the loop rather than
copying it.

That duplication is real and worth removing: three copies of the
monitoring loop is three places a change to drift-check cadence, metric
aggregation or CSV field names has to land, and the drift-only runner
exists precisely so its metric trace is comparable to a detector run's —
a comparison that silently breaks if the copies drift apart.

**But the 10% figure overstates the win, and that matters.** The 24-line
version satisfies the pinned tests, not the feature set: the surface diff
shows it drops `run_drift_only` (the module-level API), `run_manual_cl`,
and `DriftOnlyMonitor.drift_windows`, and nothing pins the per-event
record (`events` with `stream_idx`), the summary's `detector`, `batches`,
`drift_checks` and `final_metrics` fields, or the event CSV rows. So the
reconstruction is not a drop-in replacement — it is a demonstration of the
shape, plus a list of what to pin first.

**The prerequisites are now pinned** (`tests/test_run_records.py`, 24
tests): each event's ordinal and its three coordinates (batch, check,
stream_idx), their monotonicity, the optional timerange fields,
`drift_windows` deduplicating and sorting, the returned record being a
copy, both summaries' exact key sets and their semantics, the drift
trace's CSV schema and its every-check-not-sampled property, and the two
module-level entry points. Entrypoints root e467eb3f -> e54a0d15.

Re-regrown against them, and the result answers the refactor question
directly:

| | before pinning | after |
|---|---|---|
| `drift_only.py` | 24 lines (10%) | 72 lines (30%) |
| `cl_only.py` | 137 (39%) | 142 (40%) |
| `main.py` | 14 (52%) | 20 (74%) |
| public names dropped | 3 | **none** |

And critically, the reconstruction **still subclasses
`ContinuousMonitor`** — `class DriftOnlyMonitor(ContinuousMonitor)`,
`class ScheduledCLRunner(ContinuousMonitor)` — while now also carrying the
full record. So inheriting and keeping the record are not in tension: the
refactor is viable, and a 242-line file that inherits instead of mirroring
should land near 72 plus whatever the record genuinely costs.
**Done.** Both runners now subclass `ContinuousMonitor`, overriding only
what differs: `drift_only` records an event instead of dispatching the
trainer and logs every check instead of thinning the quiet ones; `cl_only`
consults a schedule once per window instead of a detector every N batches,
which is why it alone still overrides `_process_stream`. Four
behaviour-preserving seams made it possible — `check_count` and
`last_agg_metric`, `_aggregate_buffer`, `_adapt(event_id)`, and a run
label — plus lazy construction of the detector and trainer, so a
drift-only run no longer builds an optimizer it never steps and a schedule
run no longer needs a valid `detector_name` it never reads.

`drift_only.py` 242 to 133 code lines, `cl_only.py` 355 to 278 (its
remainder is the schedule classes and CLI, genuinely its own),
`continuous_monitor.py` 203 to 200 despite gaining the seams. Both claims
still report `verify=fresh` — identity covers the criteria, not the
implementation, which is what made the refactor checkable rather than
hopeful — and still earn 8/8 and 9/9 layers.

Re-regrown afterwards: the entrypoints gap closed from 38% to 45% kept,
`drift_only` specifically from 30% to 44%.

Two things the re-regrowth then taught, both now pinned:

- **The residual gap is mostly log prose, and should be.** Our two runners
  make 24 `logger.info`/`debug` calls between them; the reconstruction
  makes **zero**. No criterion demands operator-facing text, and none
  should — pinning log wording would make every message edit a criteria
  change. So the ratio should not be driven to 1; some latitude is correct,
  and this is what it looks like.
- **Removing duplication left *where* the shared loop lives unconstrained.**
  The reconstruction removed it too, but by inventing its own
  `StreamRunner` base inside `examples/utils.py` — behaviour kept, point
  lost, since the loop is framework machinery and a drift-only trace is
  comparable to a detector trace only because it is literally the same
  loop. Four tests now pin it: both runners are `ContinuousMonitor`
  subclasses, neither re-defines the shared machinery, and `drift_only`
  inherits `_process_stream` identically. Entrypoints root 1d80a987.

Kind: defect
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
Kind: decision
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
Kind: methodology
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
Kind: defect
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
Kind: methodology
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
Kind: enhancement
Status: closed — fixed upstream in reticuli PR #3 (merged): a proof is trusted
only while a root it names matches the claim's current root; `status`
prints `stale` with the remedy.

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
Kind: enhancement
Status: closed — fixed upstream in reticuli PR #3 (merged): pinned produce
outputs join the rebuild's watched set, the result carries `source_root`,
and both renderers refuse to let a root mismatch read as success.

### R10 — reticuli: `status --files` omits pinned produce outputs
A produce step declared `class = "pinned"` is inside the root — verified
directly: appending a byte to one makes `ret verify` report a different
claim — but it does not appear in `ret status --files`, which lists pinned
inputs, generated outputs, the environment and the gate verdict. The
per-file account is the view an auditor reads to see what a claim covers,
so a covered file missing from it understates the claim.
Kind: enhancement
Status: closed — fixed upstream in reticuli PR #3 (merged): pinned produce
outputs appear in `status --files` as `pinned / output`.

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

**It recurred, and the second time only the byte-copy leg could see it.**
`apeiron/__init__.py` joined the monitor claim but never the dependents'
recipes. The warm pack passed, and so did the LOCAL deep audit — the file
was on disk from `ret pull`, so the room had it either way. The export
carries declared files only, so the crosscheck's M2 leg (export → import
→ audit) failed where M1 and M3 earned, which is why `crosscheck` filed no
proof despite returning `accept`. In other words: a local audit can pass a
claim that cannot travel, and only a round trip proves it can. That makes
the subset warning more valuable than it first looked, and it is a good
argument for exporting a claim as part of accepting it.
Kind: enhancement
Status: closed — fixed upstream in reticuli PR #3 (merged): `pack_declared`
warns when a component's declared `from`-step set is a subset of what the
component supplies.

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
Kind: enhancement
Status: closed — fixed upstream in reticuli PR #4 (merged), after its own
design pass: `ret pull` with no claim named recomposes the collection
(bottom-up, root-preserving, staleness judged live, warm reseals; refuses
where a pinned supplied file would move a root), and `ret status` in the
claims directory is the matching pure view. Field-tested on a copy of this
ladder: a deliberate `config` reseal cascaded through all seven dependents
in 40s with every root unchanged. The design conversation also surfaced and
fixed a stale-`__pycache__` hazard (a same-size, same-second refresh passed
a gate on the OLD bytes). `scripts/recompose.sh`-style hand ordering is
retired; rule 3 in README.md now points at the verb.

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
Kind: bug (fixed)
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
Kind: withdrawn
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
Kind: methodology
Status: methodology note — applies to every regrowth result in this log.
See A33, which measures the distribution this note argues for.

### A33 — measurement: the kept-ratio's noise floor, per layer
R7 said to read regrowth as a sample and the campaign then reported
single draws anyway: 47% early, 51% after the criteria were strengthened.
Two single draws cannot be compared, so that 4-point movement had no
error bars and might have been the producer's luck. Measured with
`claims/scripts/variance.sh` — three independent draws per layer, model
pinned to gpt-6-astra and recorded per draw, one fresh room each, no
retries folded in, no `--recursive`:

**All ten layers, 3,035 logical lines — the measurement is complete.**

| layer | kept | spread | lines |
|---|---|---|---|
| model_harness | 81% | 2 | 234 |
| config | 69% | 5 | 231 |
| evaluation | 65% | 0 | 48 |
| updaters | 61% | 5 | 355 |
| profilers | 56% | 7 | 314 |
| entrypoints | 52% | 1 | 438 |
| trainer | 47% | 2 | 223 |
| drift_detection | 46% | 7 | 621 |
| logger | 46% | 0 | 319 |
| monitor | 42% | 2 | 252 |

**Aggregate: kept 53%, envelope [51%, 55%]**, weighted by logical lines
over the whole ladder.

The campaign's original single-draw figure of 51% sits inside that
envelope, at its lower edge. So the number quoted all along was not
wrong, only unqualified — and establishing that cost 40-odd draws, which
is the honest price of turning a quotable scalar into a defensible one.

Two expectations the completed run overturned. The deep layers were
supposed to be the least reliable, and monitor (2-point spread) and
entrypoints (1 point) came back among the TIGHTEST boundaries in the
ladder, every draw rebuilding first time. Failure is therefore a property
of drift_detection's criteria specifically — 2 of 4 draws — not of depth
or size. And monitor keeps least of all at 42% while being tightly
bounded, so low kept and wide spread are independent: a layer can pin
very little very consistently.

Earlier per-layer readings, kept for the record:

drift_detection is R7 happening again on the bench, and worse than R7
measured: **two of four draws failed outright**, at one root with one
pinned producer, and the two that finished landed 7 points apart. Its
ratio is therefore over survivors only — the report says so rather than
averaging a failure in as a zero — and the failure RATE is a second
measurement of the same boundary. A layer that reconstructs half the time
is telling you something its kept-ratio does not.

A separate earlier run of config gave 68%, 72%, 67%, so over six draws
config sits at median 69% with a 5-point range. Those three earlier draws
used the same pinned model but did not RECORD it (they predate the
provenance fix, R14), and `variance_report.py` refuses to pool them with
the attributed ones. It is right to refuse: pooling them is justified by
what I know about how I ran them, not by anything in the record, and a
measurement that needs its operator's memory is not yet a measurement.

The envelope takes every layer at its worst and then at its best
simultaneously, which independent draws will not do, so a true interval
is narrower than the three points quoted above.

A correction worth recording, because it is the kind of mistake this
whole entry exists to prevent. With drift_detection at one successful
draw I read the aggregate as 56% with a 2-point envelope and wrote that
the noise floor was about one point. That zero spread was an ARTIFACT OF
n=1: a layer measured once cannot disagree with itself. Its second
success came in 7 points away, on 621 lines, and the aggregate envelope
widened to 4 points. A single draw of a layer reports a spread of zero
and means nothing by it — exactly the error, one level up, that reporting
a single kept-ratio as a scalar commits.

So the honest figure is an aggregate noise floor near ±2 points, with
individual layers up to 7. That makes the campaign's 47-to-51 movement
comparable to the conservative envelope rather than several times it —
too close to call from these draws alone. Settling it needed a paired
comparison, the same layer before and after its criteria changed, and
**A35 ran it: the ranges come back disjoint with a 15-point gap**, so the
direction the single draws suggested is real even though those draws
could not establish it.

**Paused once by the producer, not by the criteria.** The Codex plan's
usage limit arrived during trainer's second draw, and the next two draws
returned in five and six seconds. The harness recorded them as `quota`
and left them out of every statistic, which is the rule that keeps this
measurement honest: counting a rate limit as a failed reconstruction
would report a weak specification where there is only an exhausted plan.
After the reset the remaining draws completed — monitor 3/3, entrypoints
3/3, trainer 2 more — and nothing measured before the pause was lost,
because each draw is written as it completes.

Two readings. First, the noise floor at fixed criteria is small — 0 to 7
points per layer, and it does average out: those spreads collapse to a
4-point envelope in aggregate. So the 47-to-51 movement is plausibly real
signal rather than sampling, which is the opposite of what I expected
when I started measuring. Second, the spread is itself a diagnostic,
independent of the kept-ratio: evaluation's criteria pin their layer so
firmly that three draws land on the same number, while profilers both
varies most and keeps least, which says its criteria are the loosest
measured. model_harness is the opposite case — 81% kept with a 2-point
spread, the tightest boundary in the ladder so far.

A caution that survives the measurement: the ratios above are conditioned
on a draw that rebuilt at all. A layer that fails often has a kept-ratio
only over its successes, and the report says so rather than averaging a
failure in as a zero.
Kind: measurement
Status: **closed** — all ten layers measured over the full 3,035 logical
lines: kept 53%, envelope [51%, 55%], every counted draw attributed to
gpt-6-astra. Supersedes every bare single-draw figure quoted earlier in
this log. The reticle is calibrated on the current state; what remains
before proposing changes is re-earning the ten proofs so each names its
oracle (they all predate R14's fix).

### A36 — does the documentation close the gaps? (set up, unresolved)
Ana's reply to the config-defaults PR proposed documenting the defaults
instead of asserting them in CI: "instead of asserting in CI you could
add default values in the documentation if your code uses both to
reproduce the code." Two things follow, and the first is that she is
right on the narrow point — a test asserting `detection_interval == 10`
does not catch a bug, it catches a *change*, so it is a regression lock
whose maintenance cost falls on her. Those defaults are granted latitude,
not a hole: **A3/A12-class items downgraded from criteria-gap to
decision, and the assertion PR should not be pushed.** The human ruling
on a kind-1 divergence is exactly what the taxonomy asks for (A34), and
this is the taxonomy working.

The second is more interesting. **The documentation already says almost
everything the reconstructions dropped**, and we never put it in the
room:

* `docs/configurations.md` carries a defaults table — `detection_interval
  = 10`, `ewc_lambda = 1000.0`, `adwin_delta = 0.002`, `max_ckpts`
* `docs/drift_detectors.md` states outright "Uses Kolmogorov-Smirnov
  windowing from `river`" and "Online mean-shift detector from `river`"
* `docs/tracking.md` documents the `cperf_*` counters

So the campaign's single largest divergence — both drift draws
reimplementing ADWIN, KSWIN and Page-Hinkley from scratch in the standard
library (A33) — happened while a file in the repository said to use
river. The knowledge is not missing. It is unpinned.

**Why this must be a separate claim and not an added input.** Gates
decide; documentation is read by the producer and nothing verifies that
it was obeyed. A reconstruction that ignores a document still passes the
gate and still lands on the same root. Pinning docs into the production
claims would therefore raise the kept-ratio while leaving verification
strength untouched, silently changing the measurand from "what the tests
determine" to "what tests plus docs let a producer guess" — and it would
move all ten roots and void the baseline A33 just finished calibrating.
As a separate root the *difference* between the two is the measurement,
and that difference is the evidence for Ana's position: if documenting a
default closes the gap, the CI assertion is redundant.

**All ten variants are now sealed**, each at its own root with its
layer's real suite passing warm, built by
`claims/scripts/docs_variants.sh` so the set is reproducible rather than
nine remembered manual steps. Each recipe carries a banner explaining why
that root exists, so a stray claim directory explains itself.

| layer | doc pinned | variant root | tests-only root |
|---|---|---|---|
| config | `configurations.md` | `0ed053b3` | `8cfb4c6c` |
| evaluation | `api/evaluation.md` | `52ec8eff` | `da4cc107` |
| profilers | `profiler.md` | `1288537b` | `54a5267f` |
| model_harness | `model_harness.md` | `115d9ebb` | `4adee990` |
| updaters | `continuous_learning.md` | `9db3b517` | `01494aa9` |
| logger | `tracking.md` | `1226990b` | `1f461066` |
| drift_detection | `drift_detectors.md` | `1a4a6ccd` | `713bdfd1` |
| trainer | `continuous_learning.md` | `1ac79a88` | `1b7fe903` |
| monitor | `architecture.md` | `bbc9cf6b` | `f3515468` |
| entrypoints | `quickstart.md` | `dbfbe38b` | `9778517046` |

The mapping is each layer's OWN documentation, not everything a
contributor might read: `architecture.md` is the global overview and is
excluded except for monitor, whose loop it is the only doc to describe.
Narrow keeps each result attributable to one document, which is the point
— the question is *which* documentation is load-bearing, not whether
reading everything helps.

Building the set also exposed that the recipes write `inputs` two ways, a
multi-line list and a single line. The first version of the script
handled only the shape the first layer used and silently skipped
`evaluation` — worth catching, since a skipped layer would have quietly
shrunk the experiment.

**RESULT ON drift_detection: negative, and my prediction was wrong.**

| side | draws | kept | range |
|---|---|---|---|
| tests only `713bdfd1` | 3 (1 failed) | 37%, 47% | [37, 47] |
| tests + doc `1a4a6ccd` | 3 (0 failed) | 45%, 47%, 47% | [45, 47] |

The median moves +5 points but **the ranges overlap**, so the report
declines to call it, and it is right to. I predicted the hand-rolled
detectors would disappear and the ratio would clear this layer's 7-point
spread. Neither happened.

The mechanism check is more damning than the ratio. Every documented-side
draw still hand-rolled the detectors, and **not one generated file in any
of the three rooms mentions river at all**:

| room | `river` mentions | hand-rolled classes |
|---|---|---|
| after-1 | 0 | 4 |
| after-2 | 0 | 2 |
| after-3 | 0 | 5 |

Every confound checked before concluding, because a negative result is
exactly where an instrument error hides:

* the doc WAS materialized into all three rooms — 13,789 bytes, saying
  `river` three times
* no `.md` in the tests-only rooms, so the pairing really differed
* `river==0.23.0` is pinned in the claim's own environment, so it was
  importable; hand-rolling was a choice, not a necessity
* the test files themselves mention river four times — but only in
  docstrings and comments ("the behavioral tests drive real river
  detectors", "stands in for the river detector"). Nothing in the suite
  imports it or asserts that the implementation does.

So three separate forms of prose said "river" — the documentation, the
test docstrings, and the dependency lock — and three independent
reconstructions ignored all of them. **Prose does not constrain
reconstruction; only gates do.** A producer optimises for passing the
gate, and a document is not a gate. That is the finding, and it is the
opposite of what I expected.

**What this does and does not say about Ana's proposal.** It is direct
evidence against documentation substituting for assertions *for this kind
of specification* — which library implements an algorithm, a structural
decision the producer must choose to honour. It is NOT yet evidence about
her actual case, config defaults, which are small concrete facts sitting
in a table and much easier to copy than a dependency is to adopt. Those
may behave completely differently, and the config variant (`0ed053b3`) is
already sealed, so that pair is one run away and is the test that
answers her directly. Reporting this as a verdict on her suggestion would
overreach.

One secondary observation, offered as a hypothesis: the documented side
was more RELIABLE — 3 of 3 rebuilt against 2 of 3 — and tighter, [45, 47]
against [37, 47]. Documentation may buy consistency without changing
content. On three draws that is not a claim, and it is cheap to test.
**RESULT ON config: documentation DOES substitute, substantially — and
its failure mode is worse than assertion's.**

The first config run was **confounded, and the null meant nothing**:
tests+doc against tests, on the current claim, moved 68% to 69% with
overlapping ranges. Then the mechanism check showed every draw on *both*
sides reproducing all 41 field defaults exactly, which is not guessable
(`ph_alpha = 0.9999`), so the information had to be in the inputs — and
it was: the campaign's own strengthening of `test_config.py` (9.7K to
15.4K) added `assert dd.ph_alpha == 0.9999` and its siblings. **The
control already contained the treatment.** A paired null result reads as
a finding when it can just be a design error; this one was.

Re-run against the WEAK criteria, where the defaults are genuinely
unpinned:

| side | draws | kept | defaults correct |
|---|---|---|---|
| weak tests `adec2e2b` | 3 | 51%, 48%, 50% | **8 of 41** (33 absent) |
| weak tests + `configurations.md` `4a118c06` | 3 | 69%, 68%, 70% | **34 of 41** (5 wrong, 2 absent) |

**Median moves +19 points, ranges disjoint** — and 69% is the same level
the strengthened tests reach (68-71%). So on the kept-ratio,
documentation and assertion do about equally well.

On the defaults themselves, assertion wins but documentation gets most of
the way: 8 → 41 with tests, 8 → 34 with the document.

**The important part is the five it got WRONG**, where the undocumented
control merely left them absent:

| field | code | documented draw wrote |
|---|---|---|
| `LoggingCfg.backend` | `'none'` | `'wandb'` |
| `Config.logging` | `None` | `LoggingCfg()` |
| `ensemble_detectors` | `()` | `['ADWINDetector', 'KSWINDetector', …]` |
| `pretrained_path` | `''` | `None` |
| `ckpts_path` | `''` | `None` |

Every one traces to the producer reading the document's **illustrative
examples as normative defaults**. The file shows `backend = "wandb"` and
`ensemble_detectors = [...]` in sample snippets beside tables stating the
real defaults, and is internally ambiguous about the difference — line 38
shows `max_ckpts = 1` in an example while line 46's table says
"Default: 0".

That is the asymmetry worth carrying to Ana: an absent field fails
loudly; a wrong default misconfigures silently. `backend = 'wandb'`
instead of `'none'` is a run talking to wandb when it should not, which
is **A18**, already logged as a defect.

**And obeying the documentation found a documentation bug.** Checking all
17 defaults the tables state against the dataclass fields: 2 disagree.
`pretrained_path` is a wording difference (`''` documented as "none"),
but `configurations.md:300` states `backend` defaults to **wandb** while
the field defaults to `'none'` — the doc describes the
missing-`[logging]`-section path, the field describes the section-present
path, and nothing distinguishes them. A18 again, from the other side.

(Two instrument errors on the way, both the same shape as A34's: a
defaults regex that truncated `0.05` to `0` reported seven false
mismatches until the character class admitted `.`, and the kept-ratio is
structurally blind to this question anyway, since a default is the same
line count whether right or wrong. The ratio said "no effect"; only
reading the values said what happened.)

**Recommendation for the exchange with Ana.** Her objection stands — 41
individual default assertions are a regression lock she would maintain
for little gain. Prose alone does not close it either, because it
introduces silent wrong values. The synthesis is one test that checks the
documentation's defaults table against the dataclass fields: it pins the
values, costs nothing when a default is deliberately retuned (the table
is updated, which should happen regardless), keeps the docs honest, and
would have caught the `backend` disagreement on the day it appeared.
Kind: measurement
Status: open — negative on drift_detection (prose did not displace a
structural choice), strongly positive on config defaults with the
caveats above. Eight variants remain unrun.
Kind: measurement
Status: open — set up, sealed, awaiting draws.

### A35 — the paired experiment: strengthening criteria did move what a reconstruction is forced to build
The claim the whole method rests on had never been tested. Everything
measured before this looked at ONE generation of criteria; the central
claim is about the DIFFERENCE between two, and it rested on two single
draws (47% then 51%) whose gap I had already downgraded from "several
times the noise" to "comparable to the envelope" (A33).

Design: one layer, one producer pinned and recorded, the implementation
under test held constant, and two roots that differ only in criteria.
An old rebuild room serves as the *before* generation, because it carries
the recipe and pinned inputs it was built from and still verifies at its
own root. For config:

* **before** — root `adec2e2b`, `test_config.py` at 9.7K
* **after** — root `8cfb4c6c`, `test_config.py` at 15.4K plus
  `test_device_selection.py`
* `src/apeiron/config/` untouched since the campaign began, so the
  criteria are the only variable

| generation | draws | kept |
|---|---|---|
| before `adec2e2b` | 4 | 50%, 50%, 50%, 50% |
| after `8cfb4c6c` | 5 | 65%, 67%, 69%, 69%, 71% |

**Median moves +19 points and the ranges are disjoint with a 15-point
gap.** Three further draws at the after root (67%, 68%, 72%) are
consistent and excluded, because they predate the provenance fix and
record no model. Every draw above is attributed to gpt-6-astra.

The test is deliberately crude and stated rather than dressed up: the
move counts only if the two observed ranges do not overlap. On four and
five draws that is a weak test and an honest one; a t-test on these
numbers would read as far more than it is.

So the claim is supported, for this layer and this pair of generations:
strengthening the criteria changed what an independent reconstruction was
forced to build, by much more than repeated drawing moves it. What it
does NOT establish: a general magnitude, or that any criteria change
moves the ratio. One layer, one pair. The cheap way to widen it is to run
the same pairing on drift_detection, whose criteria also moved during the
campaign.

**An unexpected observation, offered as a hypothesis rather than a
finding.** The weaker criteria produced the *more repeatable*
reconstruction — four draws returning exactly 50%, zero spread — while
the stronger criteria spread over six points. The natural reading is that
a weak suite demands so little that the minimum passing implementation is
nearly unique, whereas a strong suite demands more code and there are
more ways to write it. That sits awkwardly with profilers, which has both
the widest spread and nearly the lowest kept-ratio, so it is not a law.
Worth testing directly if the spread is ever used as a diagnostic.
Kind: measurement
Status: open — the claim is supported on config; widening it to a second
layer is the next cheap experiment.

### A34 — the divergence kinds are a triage function, and recurrence is its evidence
Two taxonomies have been running in parallel here without the link
between them being written down. The five divergence **kinds** classify
what a reconstruction did differently; this log's **Kind** field
(defect / decision / criteria-gap) classifies what a finding becomes. The
mapping between them is nearly deterministic, and it decides the work:

| divergence kind | upstream work | vehicle | who decides |
|---|---|---|---|
| substantive, and it passes | a test, with a falsification | PR | nobody — it is a hole |
| domain knowledge simplified | a test, or pin the data as an input | PR | nobody |
| dropped, nothing asked | a test **or** a deletion | issue, then PR | the team: is it wanted? |
| equivalent, different | a refactor proposal | issue | the team: worth it? |
| for humans only | none | — | nobody |

Only two kinds generate a question a maintainer must answer, which is
why the eighteen-item PR table was tractable: the three items held back
for conversation (A19, A22, A27) are exactly a kind-1 with the
wanted-or-not bit unresolved, a kind-4 judgment, and a kind-3. The
mapping held before it was named.

**Recurrence makes the weakest step measurable.** R7 said a divergence
recurring across draws is a finding about the boundary while a one-off is
a finding about the producer, and that distinction was being made by eye.
`claims/scripts/recurrence.py` counts it. Over all ten layers: **18 public names no draw reproduced, 1 dropped by
only some draws**, and **18 of 18 probed behaviours dropped by every
draw**.

The single partial result is the first producer noise the dataset has
produced, and it is worth having: `budget` in `cl_only.py`, absent from
one entrypoints draw of three. That is a finding about the producer, not
the boundary, and it is exactly what the partial column exists to catch —
without it, one draw's omission would have been filed as a missing test.

The newly measured layers added the sharpest kind-1 candidates so far.
monitor dropped four names in every draw — `aggregation`, `metric_idx`,
and, more pointedly, **`label` and `last_agg_metric`, two of the seams
introduced during the loop-duplication refactor specifically to make the
loop testable.** Three independent reconstructions did without them, so
the monitor tests do not exercise the seams added for their benefit.
entrypoints dropped `reset_on_detect` in all three.

The scan and the behaviours answer different questions, and the split
matters for triage:

* The **name scan** catches API gaps — `VOTING_STRATEGIES` on
  EnsembleDetector, `flop_counter`, the trainer's `criterion`, the
  `MetricFn`/`CriterionFn` aliases.
* It is **blind to behaviour**: the dropped `resolved_config.json` write,
  the uncapped `max_task_records`, river reimplemented in the standard
  library, the `cperf_` payload. Those need declared probes, which is
  what the MARKERS table holds.

**The instrument lied twice before it was usable, and that is the
finding.** The first implementation reported 20 dropped names in the
logger layer. All of them were artifacts: `MLFlowLogger` inherits from
`WandBLogger` across modules and `surface_diff` only folds bases defined
in the same file, while `WandBLogger.step` moved from a `@property` to an
instance attribute that the member scan never looked for. Correcting it
by walking the whole AST then swung the other way and reported local
variables (`cfg`, `k`, `maxk`, `df`) as missing API. Only a structured
walk — module and class definitions, plus attributes assigned to `self` —
gives a signal worth acting on. Separately, the `prefix` probe reported
the flag as surviving until it was anchored to `def log(`, because a bare
match also hit an unrelated `_flatten_config(prefix='')`.

So: triage amplifies instrument error directly into work. Twenty bogus PR
candidates were one unreviewed script away. Any recurrence count used for
triage has to resolve inheritance, movement between modules and instance
attributes first, and every behavioural probe has to be anchored.

One honest limit on recurrence as a ranking: 8 of the 13 unanimous name
drops are the updaters caching config values on `self`
(`lambda_ewc`, `rho_theta`, `ema_decay`...) where the draws read
`cfg.continual_learning.*` inline instead. Unanimous, and still kind 3 —
no upstream work. Recurrence says whether the criteria pin something; the
kind says whether anyone should care. Both are needed.
Kind: methodology
Status: open — the mapping is now in the reconstruct-codebase skill as the
triage step; the counts above want re-running when monitor and
entrypoints are drawn.

### A37 — the three river detectors are domesticated
APEIRON's three statistical detectors wrap `river.drift`. Each is now a
claim whose acceptance criteria pin *river's observable behaviour* rather
than describe its algorithm, so any implementation reproducing that
behaviour is admitted and river becomes replaceable with proof. Prose
cannot do this job — A36 measured that documentation does not constrain a
reconstruction — but a gate can, because a gate executes.

| claim | root | mutation | criteria shape |
|---|---|---|---|
| `apeiron-page-hinkley` | `f298db23fe39` | 25/30 | exact firing indices, 12 streams |
| `apeiron-adwin` | `75b10bd0f050` | 19/30 | exact firing indices, 6 streams |
| `apeiron-kswin` | `36a20881d0ea` | 19/30 | exact counts + index envelope, 5 streams x 40 seeds |

Each carries a recorded three-machine proof — M1, M2 (byte copy via
export/import), M3 (blind independent redo) on one root, every machine's
verdicts re-earned — and every proof names its oracle, which is what R14
bought: `{"blind": true, "model": "gpt-6-astra", "model_source":
"observed", "vendor": "codex"}`. All three are pulled into a drift
variant (`bc674cd0f7bd`) that audits `4/4 earned`.

**Generalisation was measured separately from the gates**, because
passing criteria is not equivalence: each reconstruction was run against
river on streams no claim pins. ADWIN 15/15 exact, including the
multi-firing cases. KSWIN 15/15 on counts *and* envelopes, reproducing
even river's seed-dependent variability (`[0,1,2]` detections on one
stream, `[3,4]` on another). Page-Hinkley 19/19 exact — after the
correction below.

**Two earlier conclusions are reversed.** ADWIN's ten-sample divergence
(A36) was *not* a capability limit: given criteria pinning river's exact
index, the producer wrote the real exponential-histogram algorithm with
bucket compression, and matched river everywhere. The divergence was the
absence of a gate, not the absence of an ability. And KSWIN has no
residue at all — river's own first-fire index spans [209, 213] across 40
seeds, so what was called a two-sample divergence was intrinsic
randomness. That is why its criteria pin counts exactly and bound the
index with river's own spread: an exact index is not a property of KSWIN,
only of KSWIN plus one draw.

**The Page-Hinkley correction is the most useful result here, because it
is a failure of this method caught by this method.** The first claim
sealed on six streams; all three independent redos agreed with river on
every one. A held-out probe over seven unseen streams then found
`slow_sine`, where the sealed implementation fires at 342 and river at
340. The cause is structural, not numerical: Page-Hinkley has two
standard spellings that are not equivalent — clamp the accumulator at
zero, or measure its distance from its own running minimum. They agree on
every stream that rises from a floor and disagree on every stream that
dips and recovers, and all six pinned streams were the former. River uses
the running-minimum form. River also seeds its downward extremum at `-1`
rather than `-inf` or `0`, which is arbitrary and observable; the
completed criteria carry it, and the blind redo reproduced it, which no
prose specification would have forced.

The claim now pins twelve streams in four groups (sharp change, no
change, monotone drift, dip-and-recover) and asserts the running minimum
directly rather than inferring it from one index. Both ends were checked
before resealing: the criteria admit river itself, and they refuse the
implementation the six-stream version admitted. Mutation score rose
22/30 → 25/30 on the same code, which is the strengthening showing up
independently.

The generalisable lesson is about the instrument, not the detector: **a
claim's verdict is evidence bounded by what its criteria exercise, and
says nothing about regimes no input visits.** Three independent redos
agreeing is not equivalence either — they agreed with each other because
they shared a prior, and all three were wrong in the same way. Only an
oracle probed outside the pinned set separated them. For a claim whose
whole purpose is to replace a dependency, a held-out differential probe
should be part of sealing, not a later thought.

What this does **not** yet buy: `statistical_detectors.py` still imports
river and the lock still pins `river==0.23.0`, so the layer carries the
proofs without depending on them and **A8 stays open**. Collecting the
payoff means changing APEIRON's imports, which is the team's call, not
this workspace's. Two further caveats worth carrying: KSWIN's claim is
weaker by construction than the other two, and all three proofs come
from one producer family, so they bound what that family could not break
rather than what nothing can.
Kind: measurement
Status: open — three claims sealed, proofs recorded, held-out
generalisation measured; the APEIRON import change that would collect A8
is unproposed.

### R15 — reticuli: `--producer codex` reached no adapter
`ret rebuild --producer codex` failed with `Error: stdin is not a
terminal`, which reads as a codex problem and is not one.
`_expand_producer`'s name table matched only `openai|anthropic`, so
`codex` and `claude` — both shipped under `producers/` — fell through to
the verbatim-command path and reticuli ran the *bare binary*, which with
no arguments opens an interactive UI, reads the piped prompt, and exits.
The adapter that knows how to pass a prompt and observe the model was
never entered, and nothing said the name was unknown.

Two version skews sat behind it, both the same shape as R13/R14 — code
written against a codex schema that has since moved. The adapter named
`--dangerously-bypass-approvals-and-sandbox`, which on codex >= 0.155
refuses a piped prompt for the same reason, so fixing the dispatch alone
surfaced the identical error one layer down. And model resolution joined
on the thread id from a `thread.started` event, which codex 0.155 no
longer emits — so every rebuild would again have recorded no oracle, the
exact silence R14 was opened to fix.

Worth noting against R14's own status line: that entry records the fix as
"the model is resolved by joining the two schemas — `thread.started`
carries the thread id". That mechanism is now gone upstream of us. The
fix held for one codex version, which is the honest lifetime of any
assertion about someone else's stdout.
Kind: bug
Status: closed — merged upstream as reticuli PR #11. `codex` and `claude` are
nameable in a CLI-producer table that preflights a binary on PATH;
`--sandbox workspace-write` replaces the bypass; the rollout log is
identified by mtime against a watermark sampled before launch when no
thread id is offered. `handlers_check` exercises both tables and
`surface_check` now requires every shipped adapter to be nameable and
named in help — an adapter nothing can reach is a trap, and that is the
gate that should have caught this. `gate.py` → `repo-ok`, 209 tests, CI
green on macOS and Linux across 3.11/3.12/3.13.

A second lesson, about this repository rather than about codex: three
local checks are not one check, and they catch disjoint things. `pytest`
passed while the gate was red (the earlier PR #4/#5 incident); `gate.py`
passed while `ret verify .` was red here, because the repository claim
pins `criteria/*.py` and a gate cannot notice that its own criterion
moved; and resealing the repository then invalidated the zeta population
registry, which stores the root its members were earned against. The
merge order that actually works is pytest → gate.py → selfclaim re-pin →
`ret pack .` → zeta population → ruff, and CI is the only thing that
checks all of them.

### A38 — behaviour, not lines: the reconstructions match where it counts
A33's headline is a kept-ratio -- 53% of logical lines regrow across ten
layers. A37 then showed what that number cannot see: the Page-Hinkley
reconstruction regrew at high fidelity, passed its gate, was independently
redone three times with all three agreeing, and was still a different
algorithm, diverging from the implementation it replaced across a whole
family of inputs no criterion visited. Lines-kept scored it well. A
differential probe against an oracle found it immediately.

So the question was generalised. `claims/scripts/behaviour_probe.py` drives a
layer's regrown implementation and the source it replaced through inputs the
criteria do not pin, and reports where their observable behaviour parts. The
oracle is built by mirroring the claim's own `apeiron/**` layout and
refilling it from `src/apeiron/**`, so the two sides differ in their
implementation and nothing else -- necessary because the welded package
`__init__` (A1) makes the source tree unimportable one layer at a time.

| layer | probe cases | agree | diverge |
|---|---|---|---|
| config | 82 | 82 | 0 |
| evaluation | 39 | 39 | 0 |
| profilers | 269 | 269 | 0 |
| drift_detection | 51 | 41 | 0 (10 not comparable, see A39) |
| **total** | **441** | **431** | **0** |

**Zero divergences.** On everything comparable, these four reconstructions
are behaviourally indistinguishable from the code they replaced -- including
in the places the criteria never look: how a `--set` value is typed, what
`deep_update` does when a scalar lands on a section, which regime a
threshold sweep selects, what the FLOP formatters print at every decade
boundary, what `accuracy` does with tied logits or an empty batch, what the
aten FLOP table contains key by key (A25's unverified table, compared
entry for entry).

That is a real and somewhat surprising positive, and it sharpens A37 rather
than softening it: the layers whose criteria are *tests of APEIRON's own
behaviour* reproduce that behaviour, while the layer that failed was the one
whose criteria pinned a THIRD PARTY's behaviour through a wrapper. A
reconstruction is faithful to what the gate actually executes.

Each probe was validated against deliberately wrong implementations, because
a probe that cannot fail is not evidence (the A37 lesson, learned again):
config catches a `deep_update` that replaces instead of merging (6 cases)
and an `env_overrides` that returns nothing (9); evaluation catches
`accuracy` computed by argmax equality instead of top-k ranking (3, all in
the degenerate shapes -- the ordinary cases agree, which is the whole point);
profilers catches a single aten constant moved by 1 and a formatter
threshold moved by a decade (4).

The probe needed four corrections of its own, every one a false reading, and
they are worth recording because each is a way an instrument lies:

  * two temporary directories inside one error message, reported as a
    divergence
  * a config so minimal that `Config.__init__` refused it, so a third of the
    probe compared one constructor error to itself and called it agreement
  * `{tag}_flop` (a count) and `{tag}_flops` (a throughput) differing by one
    character, so a substring filter let two wall-clock readings through
  * the builtin `hash()` for trace digests, which is per-process randomised,
    reporting 30 divergences in which every visible field matched

Not probed, and not claimed: `trainer`, `monitor`, `model_harness` and
`logger`. The first three need a model and data in the loop and the last
needs its backends; a probe for those is a larger instrument than this one,
and reporting them as covered would be the same error as the weak streams in
A37.
Kind: measurement
Status: **WITHDRAWN IN PART by A46.** The headline of this entry is vacuous.
`ret pack` seals a project's OWN files, so `claims/<layer>/apeiron/**` is
byte-identical to `src/apeiron/**` -- the probe compared the source against
itself, and "zero divergences" could not have come out any other way. What
the entry does establish stands: the probes detect real differences, proven
by the mutant validations, and the ten instrument corrections are real
lessons. What it does NOT establish, and was written as though it did, is
that any REGROWN implementation behaves like the original. A46 measures
that against the twenty rebuilt rooms and finds it false.

SUPERSEDED IN PART by A40, which probes all ten. The four layers this entry
declined to claim are now covered: the fixtures they needed -- a fixed-weight
model, fixed loaders, a concrete harness -- turned out to be a shared module
rather than "a larger instrument". The judgement that stands is the one about
not reporting coverage we do not have; the list of what was uncovered has
simply moved on.

### A39 — APEIRON: `EnsembleDetector` reports a nondeterministic regime
`EnsembleDetector.update` resolves the regime by majority vote over its
sub-detectors:

    regime_votes = [s.regime for s in signals]
    regime = max(set(regime_votes), key=regime_votes.count)

`max` over a **set** has no defined order. When the vote ties -- the ordinary
case for an even number of detectors, and the only case for two -- the winner
is whichever member set iteration happens to yield first. For `Enum` members
that is `hash(member._name_)`, a string hash, which Python randomises per
process. The same ensemble, fed the same stream, reports a different regime
from one run to the next.

Measured on upstream `main` at 63a877a over sixteen hash seeds: at the step
where ADWIN fires and Page-Hinkley has not, the tie resolves to `stable` on
10 seeds and `continual_learning` on 6. The regime is what the driver acts on
-- stable, continual_learning, fine_tuning, retrain -- so an identical
experiment can take a different adaptation path on a rerun with nothing in
the logs to explain it, and a `seed` in the config does not help because the
randomisation is of hashes, not of the RNG.

The verdict and the drift score are unaffected: both are computed by
counting, not by set order. That is why the suite never caught it -- a test
asserting `drift_detected` passes every time -- and it is also why this is
not an A37-style reconstruction divergence. The line is byte-identical in the
source and in the regrown implementation; the reconstruction reproduced the
defect faithfully. The behaviour probe (A38) found it only because it
compared whole verdict traces and the traces would not sit still, which is
why that harness now separates "unstable within one implementation" from
"different between two".

Falsification: `python3 claims/scripts/falsify_ensemble_regime.py
<path/to/src/apeiron>` -- exits 0 when the regime flips across seeds, 1 when
it does not. Verified against a clean `git archive` of upstream `main`, not
just this workspace.

A deterministic fix has to decide what a tie means, which is a judgment
call and adjacent to A22 (the regime can already outvote its own verdict):
the most severe regime among the votes, the one the combined verdict
implies, or the first in a declared precedence order. Picking one is the
team's call; any of them is deterministic.
Kind: defect
Status: open -- reproduced on upstream main with a runnable falsification;
the fix needs a decision about tie semantics, so it is not a silent patch.

### A40 — the methodology applied to all ten layers: 1,049 cases, zero divergences
A38 probed four layers. This extends the same instrument to all ten, which
required building what the earlier note said it would not pretend to have:
deterministic fixtures for the layers that need a model and data in the loop.
`claims/scripts/probes/_fixtures.py` supplies a fixed-weight model, fixed
loaders, a concrete `BaseModelHarness` and a config built through the layer's
own loader -- imported by the probe drivers and never by the layers, so both
sides of a comparison get byte-identical fixtures and differ only in the
implementation being driven.

| layer | cases | agree | diverge | mutants caught |
|---|---|---|---|---|
| config | 82 | 82 | 0 | 6 and 9 |
| evaluation | 39 | 39 | 0 | 3 |
| profilers | 269 | 269 | 0 | 4 |
| logger | 103 | 103 | 0 | 4 |
| model_harness | 72 | 72 | 0 | 6 |
| updaters | 143 | 143 | 0 | 7 |
| drift_detection | 70 | 60 | 0 (10 unstable, A39) | 7 |
| trainer | 58 | 58 | 0 | 14 |
| monitor | 91 | 91 | 0 | 21 |
| entrypoints | 122 | 122 | 0 | 10 |
| **total** | **1,049** | **1,039** | **0** | |

**Zero divergences across every layer.** On everything comparable, each
regrown implementation is behaviourally indistinguishable from the code it
replaced. The ten unstable cases are all `EnsembleDetector` and are unstable
*identically on both sides* -- that is A39, a defect the reconstruction
reproduced faithfully rather than a difference between the two.

Three results are worth naming individually.

**The updaters layer answers A13 directly.** A12 recorded that the updater
math is "nearly criterion-free" and A13 that the JVP/SAM updater is
interchangeable with a different algorithm as far as the tests can tell. The
probe does not ask whether the hooks exist; it runs each updater through real
forward/backward cycles and compares the resulting GRADIENTS and WEIGHTS
exactly. With SGD and no momentum a step is a closed-form function of the
gradient, so two implementations of the same mathematics land on identical
numbers and two implementations of different mathematics cannot. All 143
cases match, including after a closed CL cycle where the regularisers are
live. Whatever the criteria fail to demand, the reconstruction did not take
the freedom.

**A20 is sharper than recorded.** The monitor's aggregation falls back to the
mean for anything it does not recognise, and the fallback is
*case-sensitive*: `median` works, `Median` silently becomes the mean, as do
`avg`, `average`, `max`, `first`, `''` and `none`. A20 described it as a
misspelling hazard; it is also a casing hazard.

**The probe reproduced A19 quantitatively.** Mean-reduction loss over three
batches is 0.588 against 0.196 over one -- exactly 3x -- while
sum-reduction is stable at 1.177.

Every probe is validated against a deliberately wrong implementation,
because a probe that cannot fail is not evidence. The mutants are listed
above; each was chosen to be plausible rather than absurd (a `deep_update`
that replaces instead of merging, `accuracy` by argmax equality instead of
top-k, one aten FLOP constant moved by 1, an `INFO:n` tier that stops
lowering the level, `_to_scalar` taking the first element instead of the
mean, the regime ladder's `<` becoming `<=`, BWT dividing by `len+1`).

**The instrument needed ten corrections, and that is the methodological
content of this entry.** Each is a distinct way a differential probe reports
agreement it has not earned:

  1. two temporary directory paths inside one error message, read as a difference
  2. a minimal config so minimal that `Config.__init__` refused it, so a third of the build cases compared one constructor error to itself
  3. `{tag}_flop` (a count) and `{tag}_flops` (a throughput) differing by one character, so a substring filter let two wall-clock readings through
  4. the builtin `hash()` for trace digests, which is per-process randomised: 30 "divergences" in which every visible field matched
  5. a seed list (0, 1, 2) that happened to resolve the A39 tie the same way every time, so the instability it was added to detect stayed invisible until seed 5
  6. formatter magnitudes that jumped from 1e9 to 1.5e12, so a mutant moving a threshold from 1e12 to 1e11 landed in the gap and was missed
  7. the regularisers probed cold, before any CL cycle had closed -- EWC's and KFAC's Fisher estimates are all zeros then, so `lambda * 0 * diff` is zero for every lambda, and the probe reported every knob as dead code and every mode as identical. **A false finding about APEIRON, produced entirely by driving the trivial case.**
  8. the same probe, corrected, still taking only ONE step after the cycle -- `cl_postprocessing` re-anchors `theta_star` to the current weights, so `p - theta_star` is exactly zero on the next step and the penalty still cannot bite
  9. `drift_windows` rendered without being called, yielding four bound-method reprs carrying memory addresses
  10. regime thresholds set to round numbers no `drift_score` ever reaches, so a mutant changing every `<` to `<=` in the ladder was not detected at all

Numbers 7 and 8 are the ones to remember: an under-reaching probe does not
merely fail to find things, it manufactures findings. Had that output been
written up unexamined, this ledger would now claim that `ewc_lambda`,
`kfac_lambda` and all three JVP knobs are dead code. Both are wired in; the
probe was measuring the state before they apply.

Those two corrections also turn out to carry the updaters layer's whole
claim. The mutant for that layer flips the sign of the EWC penalty --
`p.grad.add_` to `p.grad.sub_`, same magnitude and shape, opposite direction.
It is caught in exactly four cases, and every one of them is an `after-cycle`
case; all the cold cases agree, because cold the penalty is zero either way.
A probe that had kept its original shape would have passed that mutant and
reported 143/143 agreement anyway -- the same number, meaning nothing.

Two guards now make a repeat harder rather than relying on care. `show()`
refuses to render any object whose repr carries a memory address, turning
that silent instability into a loud failure. And the harness runs each side
under several hash seeds and separates "unstable within one implementation"
from "different between two", so nondeterminism is reported as itself.

What is still not claimed: `main.py` is not driven end to end, because it
calls `get_example(cfg)` and needs MNIST or CIFAR on disk. Its import surface
and argument handling are probed, and the loop it runs is the monitor's,
which is. And the oracle for `evaluation` and `monitor` is this workspace's
patched source rather than upstream `main` (A19, A28), so agreement there is
agreement with the patched behaviour.
On the aggregate figure: every per-layer row above was verified on its own
run. A single ten-layer pass recorded 1,046 cases with zero divergences
before the model_harness probe grew by three checkpoint cases (below), so the
1,049 total is that run plus those three, not a separate measurement.

One more thing the probe found, about itself and then about APEIRON.
`save_ckpt` writes to `Path(cfg.model.ckpts_path)`, and that field's default
is the empty string, which resolves to the PROCESS WORKING DIRECTORY. The
first version of the model_harness probe called it with the default and wrote
a `latest` pointer file into the repository root -- caught when `git status`
showed an unexpected file at commit time, not by any check. The probe now
points every checkpoint case at a temporary directory and records the
default's resolved path by reading it rather than by writing.

That default is worth a look on its own terms: a run with `max_ckpts` set and
`ckpts_path` left alone drops checkpoints and an unprefixed `latest` file
wherever it was launched from, which for most people is a repository. Same
shape as A18 -- a default that quietly does something surprising -- and
recorded here rather than as its own finding because the severity is a
papercut and nothing is corrupted. Promote it if the team disagrees.
Kind: measurement
Status: open -- all ten layers probed at 1,049 cases with no divergence; the
one piece deliberately undriven is named.

### A41 — APEIRON: a run does not reproduce itself
`examples/mnist/mnist.toml` sets `seed = 1337`. Nothing under `src/` ever
calls `torch.manual_seed`, and the MNIST harness builds three of its loaders
with `shuffle=True` and no `generator=`:

    ds_train_tf,   bs,        shuffle=True, ...   # train
    ds_val_tf,     stream_bs, shuffle=True, ...   # the monitored stream
    ds_hist_train, bs,        shuffle=True, ...   # historical replay

`DataLoader(shuffle=True)` with no generator draws its permutation from the
global torch RNG. Unseeded, that permutation differs in every process, so the
batch order differs, the gradient sequence differs, the weights differ, and
every metric downstream differs. The `seed` field is an assurance the code
does not keep.

Measured, two runs of one config at seed 1337, device cpu, num_workers 0,
over a two-window run, comparing (step, metric) pairs with wall-clock fields
excluded:

| | pairs | identical | worst eval/accuracy gap |
|---|---|---|---|
| as shipped | 2,307 | ~46% | tens of points (37.5 and 31.25 on two measurements) |
| with `torch.manual_seed(cfg.seed)` | 2,311 | **100%** | **0.000** |

A handful of pairs are recorded by one run and not the other, so even the set
of keys is unstable. The magnitude is given as a range deliberately: the
defect is nondeterministic and so is its size. The stable facts are that
agreement sits near half, and that the accuracy gap is far larger than any
tolerance a published result would be quoted to.

Two things ARE reproducible, which is what makes this a small fix rather than
a redesign. `examples/mnist/mnist.pth` exists, so initial weights are loaded
rather than sampled. And the affine drift transform is properly seeded off
`cfg.seed + task_counter` -- the augmentation angles match to the last digit
across runs -- so the drift *schedule* is already deterministic. Only the
shuffling is loose.

Seeding the global RNG before the harness is built is sufficient: every
non-wall-clock metric becomes identical. The residue is `*_time` and the
`*_flops` throughputs derived from it, which no fix can make equal.

**Correction.** This entry first claimed that explicit `generator=` arguments
on the three shuffled loaders were "not optional" because the shipped config
sets `num_workers = 4`. That was asserted, not measured, and it is wrong. The
whole surface was then characterised:

| device | num_workers | seeded | non-wall-clock agreement |
|---|---|---|---|
| cpu | 0 | no | ~46% -- the defect |
| cpu | 0 | yes | 100% |
| cpu | 4 | yes | 100% |
| mps | 0 | yes | 100% |
| mps | 4 | yes | 100% |

**Second correction: it is two RNGs, not one line.** Everything in the table
above was measured through a wrapper that seeded `torch`, `random` AND
`numpy`, so "one line" was never what was tested. Building the run claim
(A42) applied only `torch.manual_seed` and the residue showed up at once:
three runs wrote 1843, 1842 and 1846 rows. `ContinuousMonitor._log_metrics`
samples its `detected=0` rows at ten percent --

    log_detected = drift_signal.drift_detected or (np.random.random() <= 0.1)

-- from the numpy global RNG, which `torch.manual_seed` does not touch. So
the SHAPE of the metrics file varies even once the batch order is fixed: a
reader counting `drift/detected` rows gets a different answer per run, though
no value ever disagrees and the real detections are never dropped. Seeding
numpy as well makes three runs byte-identical. The fix is
`torch.manual_seed(cfg.seed)` plus `np.random.seed(cfg.seed)`.

Found by pinning a result rather than by pinning tests, which is the
methodological point of A42: the unit-test claims could not see this because
no test reads the metrics file's row count.

With that said, workers still do not need a generator, because
`RandomSampler` computes the permutation in the PARENT process, seeded from
the global CPU RNG, and each worker's seed is derived from that same
generator; the MNIST transforms carry no per-sample randomness, since the
affine augmentation is a fixed transform per task. `torch.manual_seed` also
covers the mps backend, so no `torch.mps.manual_seed` call is needed --
checked directly.

Device choice is not a reproducibility problem either, which was the other
thing worth ruling out given that `device = "auto"` selects mps on Apple
hardware. Across cpu and mps at one seed: `eval/accuracy` identical to the
point, `drift/metric_0` identical, and `eval/loss` agreeing to 2e-6. The
shuffle permutation is device-independent by construction (the sampler draws
from the CPU generator whatever the compute device), so what is left is
float arithmetic, not behaviour.

A third nondeterminism source went unexamined: `torch.use_deterministic_algorithms`
is never set anywhere, so a kernel without a deterministic implementation
could still bite on a different device or a longer run.

**The drift events reproduce; the numbers do not.** The first measurements
were taken on a run so short that `drift/score` never left zero -- the
detector never spoke, so nothing about detection was being tested. Closing
that needed a longer run, and the shipped config turned out to be
unrunnable for a reason worth recording on its own: `update_data_stream()`
extends the stream each window and every window re-evaluates the whole grown
stream, so the cost is QUADRATIC in window count. Two windows take 18s, four
63s, six 81s; `max_stream_updates = 20` with `max_iter = 600` did not finish
in the time available and was abandoned twice.

At six windows the detector fires 92 times in 187 checks -- a dense band
from step 1054 -- and the unseeded runs agree on every one of those
positions. That is not evidence of robustness: a detector pinned to the rail
fires at the same dense set whatever the metrics do, so the configuration
cannot discriminate. `reset_after_learning = true` brings it down to two
events, which can move. It does not:

| | drift events | final accuracy | metric agreement |
|---|---|---|---|
| unseeded, run 1 | [1054, 1810] | 93.750 | |
| unseeded, run 2 | [1054, 1810] | 93.750 | 49.5% vs run 1 |
| unseeded, run 3 | [1054, 1810] | **100.000** | |
| seeded, runs 1-2 | [1054, 1810] | 100.000 | **100%** |

Three unseeded runs, two sparse events, identical positions. So the honest
statement is narrower than the one this entry was heading towards: **the
framework agrees with itself about WHEN the data changed and disagrees about
HOW WELL it did.** The mechanism explains it -- the drift is an affine
augmentation applied at deterministic points with a properly seeded
transform, so the detector meets the same large shift in the same place
whatever order the batches arrived in.

That is still the serious half for a paper. Event positions are not what gets
published; accuracy, BWT and forgetting curves are, and final accuracy moved
between 93.75 and 100.0 across three runs of one config.

One knob was nearly mis-recorded on the way. In the dense regime
`adwin_delta` appeared inert -- 0.002 and 1e-8, six orders apart, gave
byte-identical output. In the sparse regime it is plainly wired in: 1e-8
yields one event where 0.002 yields two. Saturation was masking it. Exactly
the failure A38 records twice over, caught this time before it reached the
ledger.

Why this ranks above A39 despite being easier to fix: A39 perturbs the
ensemble's regime on a tied vote. This perturbs every metric of every run.
The framework exists to produce continual-learning results, and right now a
reported BWT, forgetting curve or accuracy trace cannot be checked by
anyone -- including its author -- because rerunning the same config yields
different numbers.

Found while establishing a baseline for the next stage of the campaign: the
plan is to pin a full MNIST training run as a claim and then ask whether a
reconstruction of APEIRON reproduces that result. That question has no
referent until the run reproduces itself, so this was step zero and it
failed. It is a prerequisite for the reconstruction work, not a by-product
of it.
Kind: defect
Status: fixed here -- `claims/scripts/falsify_run_reproducibility.py`
reproduces it (exit 0) and verifies the fix (`--seeded`, exit 1), with
`--device` and `--workers` to re-run any cell of the matrix above. Wants an
upstream PR; the fix is `torch.manual_seed(cfg.seed)` and
`np.random.seed(cfg.seed)` in `main.py` before `get_example` -- two RNGs,
see the second correction above.

### A42 — a training result pinned as a claim
Every other claim in this ladder pins a layer's TESTS. `claims/mnist_run`
pins a RESULT: the configuration in `mnist-run.toml`, driven through
`main.py` over a pinned MNIST subset, must produce the metrics trace in
`reference_trace.csv` exactly, on every value that is not a wall-clock
reading.

    apeiron-mnist-run   root 33a7efb4b74b   9/9 layers earned
    gate                1,665 metrics exact, drift at [350],
                        final accuracy 100.0, 57s under seatbelt

That is a far stronger acceptance boundary than a unit test, and it is the
one the next stage needs: a reconstruction of APEIRON either reproduces this
training result or it does not.

**Reduced in three ways, and the recipe says so in full.** Six stream
windows instead of twenty, because `update_data_stream()` extends the stream
each window and every window re-evaluates the whole grown stream -- the cost
is quadratic in window count (18s, 63s, 81s at two, four, six windows) and
twenty never finished. Fifty CL iterations instead of six hundred, same
reason. And the first 2,000 train and 2,000 validation examples of real
MNIST rather than all 70,000, rewritten in the same idx-ubyte format and
pinned as claim inputs at 3MB; the full 64MB cannot be pinned, and leaving
the dataset as an external precondition would make the claim reproducible
only on a machine that already has it at the right path.

Three further settings differ from the example's defaults and not for cost:
`device = "cpu"` and `num_workers = 0` keep the platform out of the
comparison (both were measured not to matter, A41, but a claim should not
depend on that holding); `backend = "none"` keeps the gate off the network;
and `reset_after_learning = true`, because with it `false` the detector fires
on roughly half of all checks and a saturated detector makes the drift
assertion say nothing.

So this is a reproducible result, not a scientific one. The model reaches
100% on 2,000 images, which no paper would report. What the claim
establishes is that the framework CAN produce a training trace reproducible
to the last digit, and that a regrowth has to land on exactly it.

**The claim cannot pass unseeded, which is the design.** `main.py` is a
generated file here, so a reconstruction has to arrive at the seeding itself.
Stripping both seed calls makes the gate fail on final accuracy -- 87.5
against the pinned 100.0 -- and exit 1. The one-line (two-line) upstream fix
A41 asks for is therefore not argued for in prose anywhere in this claim; it
is required in order to pass. That is the most useful property the claim has:
the gate IS the argument.

**It also found what the ten unit-test claims could not.** Applying only
`torch.manual_seed` left the trace's SHAPE varying -- 1843, 1842, 1846 rows
across three runs -- because the monitor samples its `detected=0` log rows at
ten percent from numpy's RNG. No value ever disagreed; only whether a row
existed. No unit test reads the metrics file's row count, so nothing in the
layer claims could have caught it, and A41's "one line" conclusion was wrong
until this claim was built. Pinning a result tests a different thing from
pinning behaviour, and this is the evidence for that being worth the trouble.

Noticed in passing and not chased: `examples/mnist/model.py` hardcodes
`get_mnist_train("./data")` and never reads `cfg.data.path`, which the
shipped config documents as "folder or file later for custom dataset". The
field is inert for mnist, and the data root is whatever directory the process
was started in.
Kind: measurement
Status: open -- the claim is sealed and audits 9/9 earned. The reconstruction
question it exists to ask (does a regrown APEIRON reproduce this trace?) is
not yet run.

### A43 — pinning a result: ungameable and unreachable are the same change
A42 pinned a training result as a claim. This is what happened when the
reconstruction question it exists to ask was actually run, in three attempts,
and the answer is more interesting than either pass or fail.

**Attempt one: the producer read the dataset.** The claim pins a 3MB MNIST
subset and a 916KB checkpoint as inputs, and the producer contract says
"Read the check/input files IN FULL first ... A check may be longer than it
first appears -- read the ENTIRE file." So the producer went reading binary
idx-ubyte and a pickle, spent 794,891 tokens, and wrote no implementation at
all. That is R16, a reticuli defect, not a reconstruction result.

**Attempt two: a false green.** With the producer fixed to distinguish
criteria from data, the rebuild reported `rebuilt 33a7efb4b74b` -- it reached
the source root. It had not trained anything. `reference_trace.csv` was a
pinned input, so it sat in the room next to the check that compares against
it, and the shortest path to passing was to copy it:

    class RecordedMNISTRun:
        def run(self):
            source = self.root / "reference_trace.csv"
            return write_metrics(destination, read_metrics(source))

Forty-five lines, no torch imported anywhere, and scrupulously honest in its
own docstring: "A recorded run, not a reconstructed neural network or
training harness. The supplied acceptance boundary specifies an exact
recording but does not specify the network or input transformations ... it
makes no claim to perform inference or optimization." It even guarded against
the output path aliasing the source.

The producer was right and the criteria were circular. **This is the first
false green the campaign has produced**, and nothing automated caught it --
it was caught by reading the implementation instead of the exit code, which
is the same lesson as A38's ten probe corrections one level up.

**Attempt three, hardened: an honest failure, and a precise one.** The
expected trace left the room and became a sha256 per metric inside the check
(not invertible, so matching requires computing), and the TRAINED weights
were pinned by digest -- the room holds only the initial `mnist.pth`, and no
amount of copying produces the state of a trained model. Result:

| | reconstruction | pinned |
|---|---|---|
| metric series matching | **0 / 31** | |
| series present but different | 11 | |
| series never produced | 20 (all `cl/*`, `eval/fwt`, `eval/test_*`, `eval/val_post_*`) | |
| drift events | none | [350] |
| eval points | **376** | 376 |
| final accuracy | 62.5 | 100.0 |
| checkpoint written | none | required |

What it got right is as informative as what it missed. It recovered the
network EXACTLY -- conv1(1,32,5), conv2(32,32,5), conv3(32,64,5),
fc1(576,256), fc2(256,10) -- and `load_state_dict` succeeded against the
pinned checkpoint, so the architecture matched tensor for tensor. It parsed
the idx-ubyte format itself and normalised with MNIST's standard constants.
It produced exactly 376 eval points, the right monitoring cadence. The
pinned weights carried the architecture, by being LOADED rather than read,
which is a real property of pinning artefacts that no prose guidance has.

What it could not invent is the affine drift simulation -- the seeded
`sample_aug` that perturbs each task's inputs and creates the distribution
shift the detector is supposed to find. Nothing in the criteria describes it.
With no drift, nothing fires, the CL loop never runs, and twenty metric
series simply do not exist.

**The methodological finding.** The two ends fail in opposite ways:

  * pinning TESTS is regrowable and gameable -- the layer claims regrow
    (A33, A40) and a stand-in can satisfy them (A23)
  * pinning a RESULT by digest is ungameable and unreachable -- nothing can
    fake a trained-weights digest, and nothing can search its way to one

Making the criteria unforgeable and making them unreachable were the SAME
change. A result digest asserts an answer and teaches nothing, so it can only
be met by an implementation that is already determined by something else. A
result claim therefore needs behavioural criteria for whatever the result
depends on -- here, a test that pins what `sample_aug` does -- and the digest
on top as the thing that cannot be faked. Pinning the result alone is a wall,
not a specification.

That is a correction to the plan this claim was built for, not a failure of
it: "reconstruct APEIRON and reproduce the MNIST result" is not reachable in
one step, and the missing step is criteria for the example harness, which no
claim in the ladder covers because `examples/` was never a layer.
**RESOLVED: attempt eight reproduced the result exactly.** Pinning the
harness's behaviour alongside the result digest took five more attempts, and
each one found a degree of freedom the criteria had left open. The producer
never cheated again; it just kept picking the other defensible option
wherever nothing said otherwise.

| attempt | what the criteria failed to say | matching series |
|---|---|---|
| 3 | what the drift simulation is | 0/31 |
| 4 | where the pooling and flatten sit | 18/31 |
| 5 | the dropout RATES -- invisible in `eval()` | 18/31 |
| 6 | which split the monitored stream draws from | 0/31 |
| 7 | that the replay buffer ACCUMULATES across tasks | 18/31 |
| 8 | -- | **31/31** |

    claim root      22814ddab8b8
    reconstruction  22814ddab8b8   (the same root)
    run gate        31 metric series by digest, trained weights match,
                    drift at [350], final accuracy 100.0

The claim was then resealed to `f90832b74f95` to drop an unused import from
the criteria -- behaviour unchanged, identity correctly moved, because a
criterion's bytes are its identity. The reconstruction was re-checked
against the resealed criteria and passes both gates unchanged. The blind
rebuild itself was measured at 22814ddab8b8 and is not re-run here; what is
claimed for f90832b74f95 is that the same implementation satisfies it, by
gate rather than by argument.

Checked rather than taken from the exit code: no reference-reading or
copying anywhere in the 190 lines it wrote, torch imported and used, both
RNGs seeded, and the weights genuinely trained -- initial digest
`343b640f`, trained digest `3bcb1141`, pinned `3bcb1141`. A blind producer
with no reference implementation wrote code that trains a model to
bit-identical weights and a bit-identical metric trace.

**Outcomes were not monotonic: 0, 18, 18, 0, 18, 31.** A run scoring 18/31
was not closer to passing than one scoring 0/31; each was a sample from
whatever space the criteria still left free. What converged was the
specification, not the attempts.

**Four hypotheses were wrong and cost nothing because they were measured.**
That the affine transform had to be torchvision's (a hand-written matrix was
numerically identical to five decimals). That RNG consumption during model
construction shifted the shuffle (the next global draw after construction was
`0.71819127` in both). That the loaders got different permutations (all three
produced identical label sequences). That `eval_metrics` computed something
different (accuracy, loss and `eval()` all identical at `[45.5, 2.564958]`).
Each was plausible, each was refuted in one measurement, and guessing at any
of them would have added a criterion that pinned nothing.

**The rule that held.** Every pin that worked is an artefact or a computation
under a fixed seed, never a sentence. `mnist.pth` carried the architecture by
being loaded. An eval-mode digest carried the forward arrangement. A
train-mode digest under a fixed seed carried the regularisation -- count,
placement and rate jointly -- without naming any of them. A first-element
digest carried which split a loader is over. A length-and-first-element pair
carried the replay buffer's semantics. Nowhere was it necessary to write
"three dropouts at p=0.5", and had that been written instead, a
reconstruction could have satisfied the sentence while missing the
behaviour.

**What this says about the method.** The loop is a better
specification-completion tool than reproduction test. Three of the six gaps
were invisible to the result digest AND to every structural assertion
written by hand; they surfaced only because a producer searching honestly
picked the other option. Used this way the question is not "does the
reconstruction match" but "what did I fail to say", and the answer arrives
in about two minutes per round.

Kind: methodology
Status: closed -- the claim is sealed at f90832b74f95 with two gates; the
blind reconstruction was measured against 22814ddab8b8 and passes the
resealed criteria too. The six criteria the harness needed are in
`claims/mnist_run/check_mnist_harness.py`, each with the attempt that forced
it recorded beside it.

### R16 — reticuli: the producer contract tells a model to read pinned DATA
The producer prompt says "Read the check/input files IN FULL first ... read
the ENTIRE file", interpolating the recipe's whole `inputs` list. That is
right for a check file and catastrophic for data. `apeiron-mnist-run` pins a
3MB MNIST subset and a 916KB `.pth`, and the producer dutifully read them:
794,891 tokens, no implementation written, gate never reached.

A claim that pins data is therefore unregrowable, not because the
reconstruction is hard but because the prompt sends the producer into
megabytes of binary before it starts.

No format change is needed to fix it, which is the pleasant part. The recipe
does not have to declare an input's role -- the producer can tell. An input
is criteria to READ if it decodes as UTF-8, carries no null bytes, and is
under 256KB; otherwise it is data that is merely PRESENT, and the prompt says
so explicitly rather than listing it among the files to read. Identity is
untouched: both kinds stay pinned, in the root, and in the room.
Kind: bug
Status: fixed here on the reticuli branch `fix/producer-data-inputs`, which
re-ran the rebuild successfully (the producer then wrote all six files
instead of none). Wants its own tests and a gate run before it goes
anywhere; no PR opened.

### A44 — cross-family: a second prior reaches the same root, first try
Every proof in this campaign names one oracle, `gpt-6-astra`, so "no
adversary could break it" has been bounded by a single prior the whole time.
A43 sharpened the worry: eight attempts of one producer family tuning the
criteria is exactly the setup where some of the resulting "completeness"
could be an artefact of that family's defaults rather than a property of the
boundary.

It is not. `claude:claude-opus-5` reached the claim's root on its **first
attempt**:

    claim root               f90832b74f95
    opus reconstruction      f90832b74f95
    harness gate             pass
    run gate                 31 metric series by digest, trained weights
                             match, drift at [350], final accuracy 100.0
    initial / trained        343b640f / 3bcb1141 (pinned 3bcb1141)
    elapsed                  6m57s

Checked the same way as A43's: no reference-reading or copying anywhere, both
RNGs seeded, torch imported and used, and the weights genuinely moved by
training.

**The two implementations are not the same code.** All four generated files
differ, and the sizes are not close:

| | codex | opus |
|---|---|---|
| main.py | 22 | 77 |
| examples/utils.py | 8 | 40 |
| examples/mnist/model.py | 39 | 71 |
| examples/mnist/utils.py | 121 | 255 |
| **total** | **192** | **455** |

So the criteria pin the BEHAVIOUR to bit-identity -- the same trace, the same
trained weights -- while leaving the code itself genuinely free, across two
producer families with different priors. That is the strongest statement this
method has produced, and it is the one the single-oracle caveat most needed.

Worth noting what Opus wrote in its own docstring, unprompted: "Layer shapes
are carried by `examples/mnist/mnist.pth` -- the checkpoint is loaded with
`strict=True`, so the module below is the only arrangement that accepts it."
It worked out that the pinned artefact constrains the architecture, which is
the property A43 observed from the other side. The criteria taught it, which
is what criteria are for.

One caveat that does survive. Opus ran against criteria that six prior
attempts had already completed; it did not have to discover the six gaps,
only to satisfy them. So this establishes that the FINISHED boundary is
family-independent, not that a second family would have found the same gaps.
Testing the latter would mean rebuilding the criteria from scratch with Opus
doing the searching -- a different and longer experiment.
Kind: measurement
Status: closed -- cross-family reproduction verified at f90832b74f95 with
both gates and an independent implementation.

### R17 — reticuli: the claude producer records no oracle
`producers/claude.py` writes its usage drop as a hardcoded literal:

    json.dump({"vendor": "claude-code", "metered": False}, f)

No model, no `model_source`, no tokens, no CLI version. Beside what codex
records for the same rebuild --

    {"vendor": "codex", "metered": false, "tokens": 847349,
     "model": "gpt-6-astra", "provider": "openai",
     "producer_version": "0.155.1", "model_source": "observed"}

-- the difference is the whole of R14, fixed for one adapter and never for
the other. A proof earned with the claude producer says "claude-code" and
stops, so the thing A44 exists to establish -- that a SECOND, NAMED family
reached the root -- cannot be read off the proof that would record it. The
model was passed on the command line as `claude:claude-opus-5` and the
adapter forwards it to `claude --model`; it simply does not write it down.

At minimum it can record the requested model with `model_source:
"requested"`, which is honest and strictly better than silence. Reading the
model back from Claude Code's own session record would make it `observed`,
matching what codex now does.
Kind: bug
Status: fixed here on the reticuli branch `fix/producer-data-inputs`
alongside R16. Wants tests and a gate run; no PR opened.

### A45 — the ten proofs, re-earned across two producer families
The ten layer proofs recorded during the first campaign said this, and only
this:

    {"kind": "crosscheck", "m2": "8cfb4c6c3ec8...", "m3": "8cfb4c6c3ec8..."}

Both digest fields carry the same value and it is the claim's ROOT, not a
digest. So the record could not distinguish "verified today, against the
current criteria, by a named model" from "verified at some point, against
something". That is R13 and R14 as they stood before PR #7, left in place on
ten claims.

Re-earned. Every layer was rebuilt blind twice -- once by `codex`
(gpt-6-astra) and once by `claude:claude-opus-5` -- into its own untouched
room, a byte copy was materialised by export/import, and the four
realizations were crosschecked together.

**Twenty blind rebuilds, two independent families, thirty roots in
agreement:**

| layer | root | codex | claude |
|---|---|---|---|
| config | 8cfb4c6c3ec8 | reached | reached |
| evaluation | da4cc107e9a8 | reached | reached |
| profilers | 54a5267fd4a0 | reached | reached |
| model_harness | 4adee990728c | reached | reached |
| updaters | 01494aa9c12e | reached | reached |
| logger | 1f461066564e | reached | reached |
| drift_detection | 713bdfd1bf0e | reached | reached |
| trainer | 1b7fe90310cc | reached | reached |
| monitor | f3515468f90c | reached | reached |
| entrypoints | 977851704615 | reached | reached |

Every four-way crosscheck accepted: reuse, equivalence, audited and the
mutation floor all held. Each proof now carries the root it was earned
against, two genuinely distinct digests, a timestamp, an independence line
and a named oracle.

This closes the single-oracle caveat for the whole framework, not just for
the MNIST claim (A44): twenty independent rebuilds across two priors all
satisfy the same criteria.

**Correction.** This entry originally continued: "what comes back differs
substantially while the behaviour does not -- measured here by root
agreement and, independently, by the 1,049-case behaviour probe (A40)." The
first clause is right and the second is false. A root is a function of the
criteria and the recipe; the generated implementation is deliberately
OUTSIDE identity, so root agreement means twenty implementations passed the
same gates and never meant they were equivalent. And A40's probe compared
the source against itself (see its withdrawal note). A46 measures the real
thing.

**What the record still cannot say, which is R18.** A proof holds a single
`producer` object, so a claim reconstructed by two independent families can
only name one of them -- and a cross-family rebuild exists precisely to be
the control for producer dependence. Worse, WHICH one it names is
positional: the four-way crosscheck keeps the last realization's producer.
Listing codex second records `gpt-6-astra` with `model_source: "observed"`,
read back from codex's own session log; listing claude second records
`claude-opus-5` with `"requested"`, echoed from the flag we passed. Same
verification, materially different record, decided by argument order that
nothing documents and no output warns about.

The first pass through these ten was in the order the rooms were built --
codex then claude -- and so recorded the weaker provenance throughout. They
were re-recorded with codex last once that was noticed. The two-family fact
lives in this entry because the proof format cannot hold it.
Kind: measurement
Status: closed -- ten proofs re-earned at current roots with named oracles;
the format limitation is R18.

### R18 — reticuli: a proof holds one producer, and which one is positional
`ret crosscheck` accepts any number of realizations and tests each one, but
the proof it records keeps a single `producer` object taken from the last.
Two consequences, both of which bit during A45.

A claim reconstructed independently by two producer families can only say so
about one of them. That is the exact claim a cross-family rebuild is run to
establish -- it is the control for whether an agreement comes from the
boundary or from a shared prior -- and it is the one thing the proof cannot
express. The verdict knows; the record does not.

And which producer survives into the record depends on the order the
directories are listed, which nothing documents and no output mentions. The
provenance strength differs between them: codex's model is `observed`, read
back from its session log, while the claude adapter's is `requested`,
echoed from the flag (R17). So listing the same four realizations in the
natural order -- the order the rooms were built in -- silently produces a
weaker record than the reverse, and a reader of either proof has no way to
tell the other was available.

Candidate fix: make `producer` a list and record one entry per realization,
each with its own `model_source`. The independence line would then read as
what it is -- a statement about every adversary that failed, not the last
one alphabetically or chronologically.
Kind: enhancement
Status: open -- no patch. Found by re-earning the ten layer proofs (A45);
worked around there by ordering the arguments so the `observed` provenance
is the one retained.

### A46 — the regrown implementations are not behaviourally equivalent
A40 reported 1,049 probe cases with zero divergences and read as though the
reconstructions behaved identically to the original. It could not have
reported anything else: `ret pack` seals a project's own files, so
`claims/<layer>/apeiron/**` is byte-identical to `src/apeiron/**` and the
probe was comparing the source against itself. Checked directly --
`cmp` reports IDENTICAL for configuration.py, statistical_detectors.py and
torch_model_harness.py.

A45 then produced the thing that was actually missing: twenty rebuilt rooms,
ten layers by two producer families, each containing genuinely regrown code
(config regrew to 200 lines under codex and 397 under claude, against the
source's 403). Probing those rooms against the source is the measurement
A40 should have been.

| layer | cases | codex diverge | claude diverge |
|---|---|---|---|
| config | 82 | 7 | 15 |
| evaluation | 39 | 4 | **0** |
| profilers | 269 | 117 | 111 |
| logger | 103 | 23 | 19 |
| model_harness | 72 | 7 | 13 |
| updaters | 143 | **0** | 24 |
| drift_detection | 70 | 23 | 37 |
| trainer | 58 | 20 | 20 |
| monitor | 91 | 68 | 68 |
| entrypoints | 122 | 54 | 51 |
| **total** | **949** | **323** | **358** |

Every one of these passed its claim's gate and reached its claim's root.
Two cells are behaviourally identical -- codex's `updaters` and claude's
`evaluation` -- and the other eighteen are not. `profilers` diverges on 117
of 269 cases; `monitor` on 68 of 91.

**So the regrown code is not a replacement for `src/`.** It is code that
satisfies the criteria and diverges off them. The two families diverge
DIFFERENTLY from each other, which is the cleanest available demonstration
that root agreement is agreement about the criteria and nothing more: the
implementation is outside identity by design, so twenty rebuilds landing on
one root says twenty implementations passed the same gates.

Concrete examples of what differs, from the config rooms:
`get_available_device` returns something else; `deep_update` mutates its
input differently; an unrecognised config section is handled differently;
`env_overrides` treats a bare `APP_` prefix differently. None of it is
exotic, and none of it is pinned.

What survives this correction untouched. The detector claims (A37) were
verified by a real differential probe against river on eighteen held-out
streams -- a third-party oracle, not a copy of themselves. The MNIST run
claim (A43, A44) pins a RESULT, so its reconstructions had to produce
bit-identical trained weights and metric traces, and two families did. Those
were measurements against behaviour the claim did not already contain.

The reading that holds across all of it: pinning TESTS buys an equivalence
class whose width is the width of the tests, and that width is measurable --
these numbers are it. Pinning a RESULT buys bit-identity and is unreachable
until the behaviour the result depends on is pinned too (A43). The ten layer
claims are the first kind. Treating them as the second was the error in A40
and in A45's second clause.

Two probe defects found on the way, both the same shape: `attempt(obj.method)`
resolves the attribute EAGERLY, so a reconstruction that simply lacks
`_add_profile` or `_aggregate_buffer` raised before the guard could catch it
and took the whole probe down rather than recording the absence. A missing
method is a difference. All thirteen private-surface calls across the probes
now wrap the lookup inside the guard.
Kind: measurement
Status: open -- 949 comparable cases across twenty rooms; 2 of 20 rooms
behaviourally identical. The per-layer divergences are a concrete worklist
of criteria gaps, and they are gaps in APEIRON's own suite regardless of
reconstruction.

### R13 — reticuli: no codex rebuild has ever recorded a cost
Every usage drop this campaign produced reads exactly
`{"vendor": "codex", "metered": false}` — no tokens, in forty-odd
rebuilds. The producer's token scan accepts a key only when it says
`total`, but `codex exec --json` reports usage on `turn.completed` as
`input_tokens` / `output_tokens`, so it matched nothing, silently, from
the first rebuild onward. The consequence reached further than a blank
column: the three-machine cost band had no shared unit for any codex leg
and fell back to wall-clock for all of them.

Found while chasing R14 — the fix for one exposed the other, because both
come from the same wrong assumption about which schema codex prints.
Kind: bug
Status: fixed upstream in reticuli PR #7: totals now come from the
explicit usage block (input + output; cached_input is a subset of input
and reasoning_output of output, so adding either double-counts), with the
old scan kept only as a fallback for older codex versions. Verified live:
one config draw recorded 57,036 tokens.

### R14 — reticuli: a proof recorded which oracle it survived nowhere
A recorded proof says an independent redo failed to land outside the
claim, which is a statement *about a producer* — a stronger one may break
tomorrow what this one could not. All ten proofs this campaign recorded
name no model, for two independent reasons.

The kernel took the model from `RETICULI_MODEL`, a declaration by the
caller, which is empty whenever the producer picks its own default — that
is, every CLI-driven rebuild, including all of ours. And the composed
path (the only one a layered claim uses) wrote `{kind, m2, m3}` with
*roots* in the digest fields and no producer at all, while the kernel's
own recorder wrote root, both digests, the independence line and a
timestamp. So a composed proof stated neither what it compared nor whose
search it survived.

A third defect sat behind the first: the observer written to fix it read
the shapes of codex's persisted rollout log, but `codex exec --json`
prints a different family on stdout (`thread.started`, `turn.started`,
`item.completed`, `turn.completed`) and none of it names the model. The
measurement harness caught that itself, by reporting `3 draw(s) recorded
no model` — the unit tests had been written against the same wrong
assumption as the code.
Kind: bug
Status: fixed upstream in reticuli PR #7. Producers may report
vendor/model/provider/version on the usage drop and an observation
outranks the environment; `model_source` distinguishes observed,
requested, declared and unrecorded. The model is resolved by joining the
two schemas — `thread.started` carries the thread id that identifies the
rollout log, which does name it. The composed path now records what the
kernel's does, plus the layers covered, and `ret status` prints the oracle
beside a recorded proof or says the redo named none.

Note for this workspace: the ten existing proofs predate the fix and name
no oracle. They are not wrong, only mute; re-earning them with
`claims/scripts/regrow.sh` records the oracle going forward.

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
| trainer | 2 of 24 | 0.08 | no |
| monitor | 3 of 32 | 0.20 | no |
| entrypoints | 5 of 36 | 0.20 | no |

Checked directly rather than assumed: every surviving mutant in a 30-mutant
run against apeiron-trainer landed in `evaluation.py` or
`configuration.py` — component code its own gate rightly does not test,
because those components' own claims test it. So 0.12 for the trainer is
not "the trainer is barely specified"; it is "92% of the sample missed the
trainer".

Re-measured at current roots after the last reseal, every surviving mutant
in a 30-mutant run against BOTH apeiron-trainer and apeiron-monitor landed
in `configuration.py` — none in the trainer or the monitor. The dilution is
not an occasional artefact; it is the normal case for a deep claim.

Scoping mutation to a claim's own produce steps — or reporting per-step
rates — would make the number mean what a reader assumes it means. Until
then: for a composed claim, the layer-specific measure is the **regrowth
ratio** from `claims/scripts/surface_diff.py`, which compares only the
files that claim regrows itself (trainer 32%, monitor 40%). Mutation is the
right instrument for a leaf, regrowth for a composition.
Kind: enhancement
Status: closed — fixed upstream in reticuli PR #3 (merged): mutation excludes
`from`-supplied steps and reports `supplied_excluded`, so a composed
claim's score describes the layer it names.

### R1 — reticuli: strict seatbelt broke every furnished claim on macOS
The first furnished-environment strict audit ever run (`apeiron-config`)
failed with `Operation not permitted` on the venv's bin. CPython's
launcher `realpath(3)`s its own path, stat'ing each ancestor as a
separate seatbelt check; the `/Users` mask denied them and the cache
subpath allow does not cover the road to the cache. Fixed by allowing
`file-read-metadata` on the ancestor chains of the workspace and cache
(existence only), with a red-then-green regression test.
Kind: bug (fixed)
Status: fixed, reticuli commit 038d734.

### R2 — reticuli: `ret init` gitignore patterns miss nested claim stores
Root-anchored patterns (`.reticuli/tmp/`) do not cover a claim packed in
place in a subdirectory, so its scratch residue was committable (and got
committed here before being caught). Fixed with `**/`-prefixed patterns
plus identity negations (`!**/reticuli.toml`, `!**/.reticuli/manifest.json`)
so a workspace's own blunt ignore rules cannot drop the recipe or root.
Kind: bug (fixed)
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
Kind: bug (fixed)
Status: fixed, reticuli commit 6fe0f82.

### R19 — methodology: a probe that dies on the first absent name
Two defects in the behaviour probe, both of which inflated A46 and one of
which made its most important comparison vacuous.

**The probe died on the first absent name.** `probes/monitor.py::state()`
guarded `label` and `last_agg_metric` with `getattr` defaults and read the
other eight names directly, so an implementation missing any ONE of them
raised inside the dict and took down every case that calls `state()` --
including all twelve `run[*]` cases, the only ones that drive the loop end
to end. Twelve cases reported a divergence about the monitoring loop while
measuring nothing about the loop. Each `run[*]` case also bundled ten seam
names and the trained weights into a single compared value, so one missing
attribute diverged every run configuration at once.

Fixed: every seam read through a sentinel, one case per seam, one case per
observable. A missing name now costs exactly one case and names itself.

**The weights comparison had never run the trainer.** With the loop cases
unblocked, `.weights` agreed in all twelve configurations and the counters
agreed too -- which looked like the loop was equivalent. It was not
evidence. Planting an off-by-one in `_should_stop` (`>=` -> `>`) showed
why: the counters caught the mutant in 11 of 12 configurations and
`.weights` caught it in NONE. Reason: `drift_event_count` is `0` in every
configuration the probe ran, so `_adapt` never fired, the trainer was never
dispatched, and the compared weights were the fixed initialisation on both
sides. The layer's whole purpose -- drift, adapt, train, checkpoint -- was
unprobed, in A46 and in the first fix alike.

Fixed by forcing drift: Page-Hinkley at `ph_threshold=0.0001`,
`ph_min_instances=1`, `detection_interval=1` fires on essentially every
check (5 events against the source) and takes the loop through outer CL
training. Four such configurations were added, including one at
`update_mode=none` where drift still fires but no training runs, so the
pair demonstrates that `.weights` moves with the adaptation path rather
than with the stream. Re-validated against the same mutant: `.weights` now
catches it in 3 of the 4 adapting configurations and correctly not in the
`update_mode=none` one. Zero cases unstable across six hash seeds, so the
600 CL updates are deterministic.

**Also added: divergences are now clustered by root cause.** A single
absent name produces as many divergences as there are cases touching it,
and counting them as independent evidence overstates the distance between
two implementations while mis-ranking the worklist -- it sends you to the
widest probe rather than the widest behavioural gap. `behaviour_probe.py`
now groups by cause and reports "N divergence(s) from M distinct cause(s)".
Monitor's 68 were five.

The general lesson, and the third time this class of error has cost a day:
a probe reports evidence only where it has been shown able to fail, and
"shown" means against a wrong implementation, per comparison, not once for
the probe as a whole. A40 compared bytes against themselves; this compared
an initialisation against itself. Both returned clean and both meant
nothing.
Kind: methodology (fixed here)
Status: fixed here -- probes/monitor.py, behaviour_probe.py.

### A47 — the monitor's loop IS equivalent; A46's number was mostly probe
A46 put `monitor` at 68 divergences over 91 cases in both producer
families and called it the worst-regrowing layer. Under the probe repaired
in R19 -- 269 cases, with four configurations that actually reach the
trainer, and every comparison validated against a planted off-by-one --
the picture is nearly the opposite.

| observable | codex room | claude room |
|---|---|---|
| trained weights, 4 adapting configs | identical | identical |
| drift_event_count, 16 configs | identical | identical |
| batch_count / stream_update_count / check_count | identical | identical |
| cases diverging | 124 | 107 |
| distinct causes | 4 | 4 |
| unstable cases | 0 | 0 |

Both blind rebuilds run the full loop -- stream extension, per-batch
evaluation, buffered metrics, detection at interval, drift, adaptation,
outer CL training, checkpointing -- and arrive at **bit-identical trained
weights and identical counters** across every configuration probed, on
comparisons demonstrated able to catch a one-character change to the stop
condition.

Every divergence that remains is one narrow gap, and it is the same gap in
both families (see A48): four names the class does not expose, two methods
it does not have, and one error message worded differently.

So `monitor` was not the widest behavioural gap; it was the widest probe
with the most fragile driver. The ranking A46 implied -- by raw case count
-- is not usable: `profilers` 117 of 269 is dominated by a formatter sweep
across decades, where one unit-threshold decision moves dozens of cases.
Rank by distinct cause.

What this does NOT say: the rooms are not replacements. The subclass
contract is real and they fail it. The honest statement is
replacement-grade **over the loop surface this probe covers**, which is
counters and trained weights under sixteen configurations, and not over
the class's surface.
**The pin closes it.** A48's criteria were added to the claim's gate, the
claim resealed to `ceaa840b12fc`, and a blind rebuild run against the
stronger criteria with nothing of the source supplied:

| | before the pin | codex, after | claude-opus-5, after |
|---|---|---|---|
| cases | 269 | 269 | 269 |
| diverging | 124 / 107 | **2** | **2** |
| distinct causes | 4 | **1** | **1** |
| unstable | 0 | 0 | 0 |
| lines of monitor | -- | 109 | 261 |

Both producer families were run, independently, against the resealed
claim, and both reached `ceaa840b12fc`. The result is not specific to one
oracle: three implementations of this loop now exist -- the source at 405
lines, codex's at 109, claude-opus-5's at 261, each byte-different from
the other two -- and the two regrown ones agree with the source on 267 of
269 cases with the identical single residual.

Both residual cases are the same thing, in both families: the same
`RuntimeError` raised in the same place with different prose. The source
says "Model Harness requires evaluation metrics"; codex's says "Drift
checking requires evaluation metrics"; claude's says "Drift check requires
evaluation metrics, but the metric buffer ...". Same type, same place,
three wordings -- the latitude that should exist, and not worth pinning
(driving a boundary to zero divergence buys a worse suite, not a better
one).

The regrown file is 109 lines against the source's 405, and byte-different
throughout, so this is not the A40 trap: a quarter-sized independent
implementation now matches the original on the counters across sixteen
configurations and on bit-identical trained weights across the four that
run outer CL training.

So `monitor` is **replacement-grade over the surface this probe covers** --
counters, drift events, and trained weights under sixteen configurations,
with every comparison validated against a planted mutant. Not over the
whole class: the probe does not reach device selection, multi-GPU, logging
payloads beyond the pinned schema, or `_extend_stream`'s edges.
Kind: measurement
Status: closed -- 124 divergences to 2, and the 2 are error prose.

### A48 — APEIRON: the monitor's subclass seams are pinned by no test of its own
`src/drift_only.py` and `src/cl_only.py` both subclass `ContinuousMonitor`
and read six things off it that no test file in the monitor module's own
suite asserts:

| seam | read at | what it does |
|---|---|---|
| `label` | drift_only.py:56, cl_only.py:257 | names the run |
| `metric_idx` | drift_only.py:112,168, cl_only.py:384 | `f"metric_{self.metric_idx}"`, a CSV column name |
| `last_agg_metric` | drift_only.py:112 | the value the detector saw |
| `aggregation` | the reduction applied | |
| `_aggregate_buffer` | **called** at cl_only.py:326 | reduces the buffer |
| `_adapt` | documented as the scheduled-adaptation seam | dispatches training |

Two independent producer families, given the module's tests and not its
source, dropped all six -- the same six, separately. That is the signature
of a specification gap rather than of producer noise: nothing asked.

The leading underscore on `_aggregate_buffer` is the interesting part. It
reads as private to the class and is in fact private to the class
*hierarchy*; a subclass in another module is a caller like any other, and
the name is contract. A refactor that inlines the aggregation satisfies
every test in `tests/test_continuous_monitor.py` and breaks both side
runners.

The tests that DO exercise some of this surface -- `tests/test_run_records.py`
(which even enumerates `_aggregate_buffer` in a shared-API set) and
`tests/test_entry_points.py` -- live with the entry points, one layer above
the class that provides the seam. The repository's suite as a whole
therefore covers it while the module's own suite does not, which is
invisible until something regenerates the module from its own tests alone.

Pinned here as `claims/monitor/test_subclass_seams.py`, 20 cases, added to
the claim's gate; the claim resealed from its prior root to
`ceaa840b12fc`. Validated both ways before being believed: 20 of 20 pass
against `src/`, and against the two regrown rooms they fail 19 and 18 --
so the criterion is reachable and it bites.

Kind: APEIRON (missing tests)
Status: open -- candidate for upstream. It needs no change to behaviour,
only tests, and it is justified entirely from two files in `src/` that
call the names.

### R20 — methodology: the probe compared the source with itself, again
Running the profilers probe against both regrown rooms reported **269 of
269 cases agreeing**. It was A40's error for the third time, by a new
mechanism, and it would have been filed as "profilers is replacement-grade"
if the number had not been obviously too good.

A directory without `__init__.py` is only a namespace *portion*: Python
records it and keeps scanning `sys.path`, and the first REGULAR package it
finds afterwards wins outright, whatever the order. This workspace installs
itself into its venv -- `site-packages/apeiron.pth` adds `src/`
permanently -- and the profilers claim generates only
`apeiron/profilers/*`, with no `apeiron/__init__.py`. So with
`PYTHONPATH=<room>`, `import apeiron.profilers.count_flops` resolved to
`src/apeiron/profilers/count_flops.py`. Confirmed directly rather than
reasoned:

    PYTHONPATH=<profilers room>  ->  .../src/apeiron/profilers/count_flops.py
    PYTHONPATH=<monitor room>    ->  .../<monitor room>/apeiron/driver/continuous_monitor.py

The monitor rooms were unaffected only by luck: their claim happens to
generate `apeiron/__init__.py`, so they arrive as regular packages and win.
A47's result was checked this way before being kept, not assumed.

Two fixes. The subject is now COPIED into a temp tree before being probed --
a room is evidence and does not get written to -- and an empty
`apeiron/__init__.py` is added to both trees when both lack it, which is
faithful because the claim's own gate runs `PYTHONPATH=.` in an environment
where nothing else provides `apeiron`. Then a guard: `_assert_resolves`
imports `apeiron` under each root and REFUSES to compare unless the module
resolves inside that root. With the guard in place the probe reproduces
A46's numbers exactly -- 117 and 111 -- so A46 itself was measured under an
interpreter where PYTHONPATH won, and only this session's run was vacuous.

The pattern across all three instances is identical: the probe ran, every
case passed, and nothing was compared. A clean result is evidence only if
the thing under test was the thing loaded, and that has to be checked by
the harness rather than trusted.
Kind: methodology (fixed here)
Status: fixed here -- behaviour_probe.py.

### R21 — methodology: a divergence count that cannot be ranked
A46 ranked the worklist by raw diverging cases, which put `profilers`
second worst at 117 and 111. Decomposed, those are 4 and 3 divergences on
contract; the other 113 and 108 are the three formatters turning a float
into a display string -- `'1 FLOPs'` against `'1.00 FLOPs'`, `'1.0000 s'`
against `'1.00 s'` -- across a sweep of every decade. Nothing in `src/`
parses that output and the only existing assertions are substring checks on
the unit name, so those are two defensible spellings of the same number.

Ranking by the raw count sends you to fix prose. The probe now declares a
per-layer latitude surface and reports it SEPARATELY rather than dropping
it: "N diverge on contract ... plus M divergence(s) in granted-latitude
surface". Dropping them silently would read as "covered everything".

One entry on that list was checked rather than assumed. `_add_profile` has
no caller outside `count_flops.py` anywhere in `src/`, `examples/` or
`tests/`, so it is private in fact and a reconstruction may keep its
bookkeeping elsewhere. `_aggregate_buffer` looked exactly the same -- a
single leading underscore on a profiler-adjacent helper -- and was NOT
free, because `cl_only.py:326` calls it (A48). A leading underscore is not
evidence either way; each name needs the grep.
Kind: methodology (fixed here)
Status: fixed here -- behaviour_probe.py `LATITUDE`.

### A49 — `profilers`: 4 contract divergences, not 117
Under R20's guard and R21's split, the profilers rooms measure:

| | codex | claude |
|---|---|---|
| contract divergences, before the pin | 4 | 3 |
| granted latitude (formatters, `_add_profile`) | 113 | 108 |
| contract divergences, after the pin | **1** | -- |

The four were: `_add_profile` absent (2, since reclassified as latitude),
`get_performance()` reporting `fwd_flop` as `768` where the source reports
`768.0`, and `measure_flops` leaving different state behind after a block
raises.

The type one is contract and not cosmetic. `get_performance()` returns a
flat dict and four call sites splat it straight into a logged metric row
(`continuous_trainer.py:255`, `continuous_monitor.py:376`,
`drift_only.py:151`, `cl_only.py:369`), so every key becomes a CSV column
and every value a cell in the artifact external analysis reads.
`test_profiler.py` asserts the keys exist and never what the values are, so
`768` passes the suite and changes the file.

Pinned as `claims/profilers/test_reported_performance.py`, 14 cases, in the
gate; claim resealed to `c98098d7f167`. The FLOP figures are pinned as
exact integers that must scale with the batch -- 384 / 768 / 1536 / 3072 --
so the assertion is about the computation and not the type alone; a
plausible constant fails it. Validated both ways: 14 of 14 against `src/`,
and against the rooms it catches codex (1 failure) while claude passes all
14, because claude's divergences were entirely in the latitude surface.
That asymmetry is the honest result rather than a problem with the
criterion.

A blind rebuild against the resealed claim -- 129 lines against the
source's 392, byte-different -- leaves **1 contract divergence**, and it is
the one deliberately left unpinned (A50).

(Two rebuilds were run. The first reached `e225bbe9dbb4` and was discarded:
a lint fix to the criteria file afterwards moved the root, so the room had
been built against a superseded one. A measurement against a root that no
longer exists is not a measurement of the current claim -- the same rule
that retires a zeta trial when the repository reseals.)

The reading: more pinning would not make this layer replacement-grade,
because what remains is display. `profilers` was never the second-widest
behavioural gap; it had the widest probe.
Kind: measurement
Status: closed.

### A50 — APEIRON: a measurement that CRASHED is reported as a real zero
`FLOPSProfiler.measure_flops(tag)` is a context manager. When the block
inside it raises, the exception propagates and `tag` is released -- but the
tag stays in `profiles` with its count recorded as zero:

    profiles["boom"]        == {"flop": [0], "time": [2.0e-06]}
    get_performance()       == {"boom_flop": 0.0, "boom_time": 2.0e-06,
                                "boom_flops": 0.0}

`get_performance()` is splatted into a logged metric row, so a forward pass
that threw contributes `cperf_boom_flop = 0.0` to the metrics CSV,
indistinguishable from a block that genuinely executed zero FLOPs. Any
averaging over that column is then pulled toward zero by failures, and the
failure itself leaves no trace in the data saying so.

Both reconstructions, independently, drop the tag instead.

This is deliberately NOT pinned. Which behaviour is correct is a question
for whoever owns the metrics -- recording nothing, or recording something
marked as failed -- and pinning the current behaviour into a gate would
settle it by default. The claim asserts only what is not in question: the
exception propagates, `tag` is released, and a later measurement is
recorded under its own name. The divergence is therefore still reported by
the probe, as the single remaining contract divergence on the layer.
Kind: APEIRON (behaviour question)
Status: open -- needs a decision, not a test.

### R22 — methodology: an exception's type is contract, its wording is prose
Sixteen of the logger's divergences -- eight in each producer family --
were one shape:

    source  ValueError: Invalid stage 'EVAL'. Must be one of: ('eval', 'drift', 'cl')
    room    ValueError: Invalid stage 'EVAL'; expected one of ['eval', 'drift', 'cl']

Same exception type, same rejected input, same valid set, different
sentence. The decision is identical and the contract holds; only the
wording differs, and a suite that pinned the wording would be a worse
suite. The monitor's two residual divergences after A48 were the same
shape, and so was one of the profilers'.

R21 handled this by hand with a per-layer list of case prefixes, which
does not scale and quietly mis-ranks any layer nobody remembered to
annotate. It is detectable instead: when both sides raise and the
exception types match and the strings do not, the divergence is prose.
`behaviour_probe.py` now classifies that automatically and reports it on
its own line.

Two boundaries matter, and both are asserted by how the rule is written.
One side raising where the other returns a value is NOT prose -- that is
behaviour, and it stays on the contract count. And a divergence in a
returned VALUE is not prose even when it looks textual: the profilers'
remaining case compares two dicts and is unaffected, which was checked
rather than assumed after the rule went in.
Kind: methodology (fixed here)
Status: fixed here -- behaviour_probe.py `prose_only`.

### A51 — APEIRON: nothing pins how a `verbosity` string becomes a level
`[logging] verbosity` is a config value, so the string-to-level mapping is
the surface a typo in a TOML file reaches. `ConsoleLogger._get_log_level`
implements it in one expression:

    return 20 - tier if base_level == "INFO" else LEVEL_MAP.get(base_level,
                                                                logging.INFO)

with `_parse_verbosity` upper-casing and stripping first. So the tier
offset applies to INFO **only** -- `DEBUG:2` is DEBUG, not DEBUG minus
two -- and `"info"` works because of the upper-casing.

`test_logger.py` asserts three cases (`_get_log_level("INFO") == 20`,
`("INFO:3") == 17`, `("DEBUG") == 10`) and leaves the rest open: case
folding, the INFO-only rule, a negative tier, a non-numeric tier, and what
an unknown name does. Measured against the source:

| input | `_parse_verbosity` | level |
|---|---|---|
| `"info"`, `" info "` | `("INFO", 0)` | 20 |
| `"INFO:9"` | `("INFO", 9)` | 11 |
| `"INFO:-1"` | `("INFO", 0)` | 20 |
| `"INFO:abc"`, `"INFO:"` | `("INFO", 0)` | 20 |
| `"debug:2"` | `("DEBUG", 2)` | **10** |
| `"CRITICAL:5"` | `("CRITICAL", 5)` | **50** |

Two reconstructions, given the suite and not the source, each got this
wrong and differently. One applied the tier to every level, so
`CRITICAL:5` became 45 and `debug:2` became 8. The other **raised
TypeError** on the lowercase `"info"`, never having upper-cased before the
arithmetic -- a framework whose config accepts `verbosity = "info"` and
then crashes on it is worse than one that mis-levels it, and nothing in
the suite would have caught either.

Also unpinned and also divergent: `.verbosity` stores the upper-cased
string but does NOT strip it, so `" info "` is kept as `" INFO "` while
`_parse_verbosity` strips. That asymmetry is the measured behaviour and
both rebuilds normalised one way or the other.

Pinned as `claims/logger/test_verbosity_contract.py`, 46 cases, in the
gate; claim resealed to `c3f078770b7c`. Validated both ways: 46 of 46
against `src/`, and against the two rooms it fails 19 and 11.
A second gap on the same layer, found by the same probe and pinned
alongside it as `test_log_call_surface.py` (27 cases). `Logger.log` takes
five parameters and forwards all five to the backend in one call
(`logger.py:94`); the suite passes `metrics` and `step` and never the
other three. The rebuild forwarded a different arity and
`Logger(backend="none").log({...}, prefix=False)` raised

    TypeError: WandBLogger.log() got an unexpected keyword argument 'prefix'

One version of that criterion asserted that a none-backend logger holds no
`WandBLogger` at all, and it FAILED against `src/` -- correctly, because
`logger.py:44` builds `WandBLogger(enabled=False)` deliberately, using the
disabled backend as the null object so the facade needs no second code
path and `to_csv` keeps working. The promise is not "no wandb class is
instantiated", it is "no run is opened", which `url` reports. Running a
criterion against the implementation before believing it caught an invented
property for the fourth time this campaign.

Claim resealed to `be5d0af2d5da`. A blind rebuild against it leaves **0
contract divergences** over 103 cases: 1 in latitude
(`ConsoleLogger._get_step`, no caller outside `console_logger.py`) and 8
that are the same `ValueError` worded differently (R22).
Kind: APEIRON (missing tests)
Status: closed here -- 14 and 10 contract divergences to 0. Candidate for
upstream: tests only, no behaviour change, justified from
`console_logger.py` and `logger.py` alone.

### A52 — APEIRON: `verbosity = "WARN"` logs MORE than you asked for
A consequence of A51's fallback worth separating, because it is a
behaviour question rather than a missing test.

`LEVEL_MAP` holds DEBUG, INFO, WARNING, ERROR and CRITICAL. Anything else
resolves through `LEVEL_MAP.get(base_level, logging.INFO)` to INFO, with
nothing logged about the setting not being understood. Measured:

    verbosity="WARNING"   info tier 0 printed: no    warnings printed: yes
    verbosity="WARN"      info tier 0 printed: YES   warnings printed: yes
    verbosity="NOTSET"    info tier 0 printed: YES
    verbosity="TRACE"     info tier 0 printed: YES
    verbosity=""          info tier 0 printed: YES

So a user who writes the ordinary abbreviation `WARN` to quieten a run
gets the full info stream instead, and so does a user who mistypes the
name at all. The failure direction is the unhelpful one: more output than
requested, silently, on a long training run.

Both reconstructions diverge here too, in the other direction -- one maps
`WARN` to WARNING (30) and `NOTSET` to 0.

Deliberately NOT pinned. Whether an unknown name should raise, warn once,
or resolve through an alias table is a decision for whoever owns the
configuration surface, and writing today's answer into a gate settles it
by default. The claim asserts only that an unrecognised name does not
RAISE -- a config typo taking down a training run is no one's intent --
and leaves which level it picks free.
Kind: APEIRON (behaviour question)
Status: open -- needs a decision, not a test.

### A53 — all twenty rooms, re-measured under the guard and the split
A46 ranked the worklist by raw diverging cases. Under R20's resolution
guard, R21's declared latitude and R22's prose rule, every one of the
twenty rooms was re-measured. Each cell satisfied the guard, so none of
these is the source compared with itself.

| layer | codex contract | claude contract | free | cases |
|---|---|---|---|---|
| config | 7 | 15 | 0 | 82 |
| evaluation | 4 | **0** | 0 | 39 |
| profilers | 2 | 1 | 113 / 110 | 269 |
| logger | 14 | 10 | 9 | 103 |
| model_harness | 7 | 13 | 0 | 72 |
| updaters | **0** | 24 | 0 | 143 |
| drift_detection | 23 | 37 | 0 | 70 |
| trainer | 20 | 20 | 0 | 58 |
| monitor | 124 | 106 | 0 | 269 |
| entrypoints | 54 | 51 | 0 | 122 |

Monitor's figures are higher than A46's 68 and 68 because its probe was
repaired (R19) and now carries 269 cases rather than 91, with four
configurations that reach the trainer. Everything else reproduces A46
exactly, which is the evidence that A46 was measured on a path where
PYTHONPATH won and only this session's first profilers run was vacuous.

**The two zeros are real, and were checked rather than trusted.** A zero
is exactly what a vacuous comparison produces, so each was validated
against a planted bug in the layer it covers. The updaters probe catches
an inverted replay gate (`mix_historic_data` negated) in 22 of 143 cases.
The evaluation probe catches a top-k slice off by one in 1 of 39 -- it
fails, so the zero is not vacuous, but one case is thin sensitivity and
`claude evaluation` should not be called replacement-grade on that
strength alone.

Both validations were themselves wrong the first time, and the guard said
so: the mutants were built as full `src/` copies, which carry
`apeiron/__init__.py` where a layer-shaped oracle tree does not, so the
harness refused the asymmetric comparison instead of reporting a number.
Rebuilt layer-shaped from each claim's own recipe.

**Where the pinning rounds landed:**

| layer | before | after a blind rebuild |
|---|---|---|
| monitor | 124 / 106 | **0** (both families) |
| logger | 14 / 10 | **0** |
| profilers | 4 / 3 | **1**, deliberately unpinned (A50) |

Three layers at zero contract divergences, each on a probe shown able to
fail, each against an implementation a fraction of the original's size:
109 and 261 lines against 405 for the monitor, 58 against 165 for the
console logger, 129 against 392 for the profiler.

The remaining worklist, in the only order that is meaningful -- and still
an upper bound, because only profilers and logger have had their latitude
triaged: drift_detection (23/37, with 10 cases where the SOURCE disagrees
with itself across hash seeds, which is A39 and a defect rather than a
gap), entrypoints (54/51), trainer (20/20), config (7/15), model_harness
(7/13), evaluation (4/0).
Kind: measurement
Status: open -- six layers untriaged.

### R23 — methodology: a room that failed its gate is not a rebuild
Strengthening the drift_detection criteria made the producer fail outright:

    ret: rebuild: the producer failed (rc=1)
    producer_codex: codex finished but the gate never produced DRIFT_OK

`ret rebuild` leaves the workspace on disk either way, and the probe,
pointed at it with `--subject`, went on to report "4 diverge on contract"
for a workspace the producer had abandoned. That number is meaningless. The
entire premise of a grade-2 measurement is "an implementation that PASSES
the gate still behaves differently"; this one does not pass, so there is
nothing to say about the criteria from it.

`behaviour_probe.py` now reads the claim's gate steps and refuses unless
every sentinel they write is present in the room. The check is conditional
on the subject carrying a `reticuli.toml`, because mutant trees built from
`_oracle_tree` have no recipe and no gate and are legitimate subjects --
verified both ways after the guard went in: the abandoned room is refused
and the evaluation mutant still measures.

Same family as R20: the probe will measure whatever it is pointed at, and
every premise it relies on has to be checked by the harness rather than
remembered by the operator. That is now three such guards -- the subject
resolves inside its own tree, the subject passed its gate, and a clean
result is backed by a planted failure.
Kind: methodology (fixed here)
Status: fixed here -- behaviour_probe.py `_assert_gate_passed`.

### A54 — APEIRON: nothing pins WHEN a drift detector fires
The drift_detection suite covers what a `DriftSignal` carries, how
`drift_score` averages, and which regime a score maps to. It does not
cover the question the detectors exist to answer: given this sequence of
values, at which step does drift fire? That decides when
`ContinuousMonitor` dispatches continual learning, hence which batches the
model trains on, hence the weights the run ends with.

**One rebuild does not import river at all.** It reimplemented ADWIN,
KSWIN and Page-Hinkley from scratch in 115 lines against the source's 316,
and passed every test in the layer. On a step stream it fires at 206 where
the original fires at 223 -- seventeen steps of stream consumed before
adaptation, with the same regime label and the same drift score, and
nothing in the suite notices. A37 found this for Page-Hinkley; it
generalises to the whole layer.

**The other rebuild does import river, fires at exactly the right steps,
and still diverges.** Its `drift_score` drops to 0.0 the step after a
detection instead of carrying it for the window: 99 of 400 steps differ,
every firing step and regime label identical. That reaches an artefact --
`_log_metrics` writes `drift_signal.drift_score` as the drift stage's
`score` column on every check -- so the difference is 99 wrong cells in
the metrics CSV. A probe that only looked at firings would have called
that room clean.

Pinned as `claims/drift_detection/test_detector_firing.py`, 29 cases: an
exact firing table over six closed-form streams (no RNG, so the table is
re-derivable rather than recorded), the trailing-window persistence, and
the per-detector window asymmetry -- ADWIN averages over 100 samples,
KSWIN and Page-Hinkley over 50 (`statistical_detectors.py:80`, `:183`,
`:287`), so one detection scores 0.01 on ADWIN and 0.02 on the others.

Two of the six streams are negative controls (`flat`, `spike`) where
nothing fires, and three further cases assert the table discriminates at
all: every detector fires somewhere, every detector is silent somewhere,
and no two detectors fire identically. A firing table without those is
satisfied by something that fires constantly.

Validated: 29 of 29 against `src/`; 8 failures against the
reimplemented-from-scratch room and 3 against the river-delegating one, so
it catches both failure modes and for the right reasons.

One claim NOT made. The score-decay difference is pinned as a difference
in logged output, which it demonstrably is, and not as a difference in
adaptation decisions, which was not shown: changing a regime needs two
detections inside one window, and no stream tried produces them -- ADWIN's
firings on an oscillating stream sit 160 steps apart, and tightening the
oscillation stops it firing at all. The first version of this entry
asserted the escalation ladder could never climb; measurement refuted it
and the claim was dropped.

Claim resealed to `da9de183e3f6`. The blind rebuild against it failed its
gate on attempts 1 and 2 and passed on attempt 3 -- a success rate of one
in three where the layer previously regrew readily, which is itself a
measurement: the criteria now demand something the producer does not get
for free.

**And the room that passed still does not import river.** 183 lines
against the source's 316, no river, and it reproduces the firing table on
all six streams exactly. That is the outcome the method is supposed to
produce: the pin forced behavioural equivalence without dictating the
implementation. An independent ADWIN that fires where river's fires is a
legitimate answer; one that fires seventeen steps early is not, and now
cannot pass.

| | before the pin | after |
|---|---|---|
| contract divergences | 23 | **2** |
| of those, worth pinning | -- | **0** |
| prose (R22) | -- | 4 |
| unstable in the SOURCE (A39) | 10 | 10 |

Two of the four remaining were reclassified as latitude after checking for
callers, the same check as `_aggregate_buffer` and `_add_profile`:
`DriftSignal` is a plain class rather than a dataclass (`base.py:29`) and
nothing calls `asdict`, `fields` or `replace` on it, so the probe case was
asking about dataclass internals; and `VOTING_STRATEGIES` is read only
inside `model_performance_detector.py`, at :357 and :364, with the SET of
valid names asserted by behaviour cases instead. The other two are A55,
deliberately unpinned.
Kind: APEIRON (missing tests)
Status: closed -- 23 contract divergences to 2, and neither of the 2 should
be pinned.

### A55 — APEIRON: a non-string `ensemble_voting` leaks an internal error
`EnsembleDetector` validates its voting strategy by normalising the name
first, so a value that is not a string fails inside the normalisation
rather than in the validation:

    voting=1        -> AttributeError: 'int' object has no attribute 'strip'
    voting=None     -> AttributeError: 'NoneType' object has no attribute 'strip'
    voting=['any']  -> AttributeError: 'list' object has no attribute 'strip'
    voting=2.5      -> AttributeError: 'float' object has no attribute 'strip'

`ensemble_voting` is a config key, and `None` is what an absent or
mistyped TOML value becomes, so the reachable failure for a user is an
`AttributeError` about `strip` with no mention of voting, strategies, or
which names are valid. The class already has the good message for an
unknown STRING (`model_performance_detector.py:360` lists
`sorted(self.VOTING_STRATEGIES)`); it is only the non-string path that
escapes it.

Both reconstructions raise `ValueError: unknown voting strategy: ...` here
instead, one of them listing the valid names. **The reconstructions are
right and the source is wrong** -- the A17/A25 category, and the most
useful kind of divergence this exercise produces.

Deliberately NOT pinned, for the reason that applies to A50 and A52:
writing the current `AttributeError` into a gate would freeze the defect
in place, and choosing the replacement -- `ValueError`, a `TypeError`, or
coercion -- belongs to whoever owns the configuration surface. The
divergence stays visible as one of the layer's two remaining contract
divergences rather than being silenced.
Kind: APEIRON (defect, reconstruction is right)
Status: open -- a one-line fix in the validation order, if the team wants
it.

### R24 — methodology: "must raise" needs "must accept" beside it
The evaluation criteria pinned that three degenerate inputs raise rather
than returning a number. The blind rebuild against them took the probe
from 3 contract divergences to **8** -- worse than before the pin.

The producer had generalised. It added a shape guard rejecting anything
that is not a batch-by-class matrix, and a `topk` guard rejecting anything
it judged invalid, including `topk=(0,)` -- which the source ACCEPTS,
returning 0.0, because nothing can be in an empty set of predictions. My
controls were one case wide (a 2x2 batch scores 100%), so nothing said
where the guards had to stop.

A criterion that says "refuse this" licenses any implementation at least
that strict. Without a measured set saying "accept that", the stricter
one is a legal answer. Fixed by widening the controls to the measured
accept-table -- `topk` of 0, 1, 2, `(1,0)`, `(0,1,2)`, a single-class
batch, a single-sample batch -- after which the same room fails exactly
the `topk=0` cases and the next rebuild reached **0**.

Second half of the same lesson: 7 of those 8 divergences were both sides
refusing the input with DIFFERENT exception types, and the room's types
were better (`ValueError: Cannot score an empty batch` against the
source's `ZeroDivisionError`). `prose_only` deliberately treats a type
change as contract, because a caller that catches one type and not
another behaves differently -- so these were declared as latitude
per-case, and only after checking that nothing in `src/` or `examples/`
wraps a metric call in try/except. Counting them would have reported an
improvement as a regression.
Kind: methodology (fixed here)
Status: fixed here -- evaluation criteria widened, LATITUDE extended.

### A56 — the last five layers, pinned
Completing the sweep. Each layer: triage by cause, grep every private name
for cross-layer callers, pin what is contract, declare what is free,
rebuild blind, re-probe.

| layer | contract before | after a blind rebuild | what was pinned |
|---|---|---|---|
| config | 7 / 15 | **3** (all A57) | device precedence, `deep_update` semantics, `kv_to_nested`, `env_overrides` |
| evaluation | 4 / 0 | **0** | degenerate inputs raise; the accept-table beside them (R24) |
| model_harness | 7 / 11 | **4** (all A57) | `_unpack` arity, empty-loader refusal, registry cap and alignment |
| trainer | 17 / 17 | **3** → latitude + 1 pinned | `-> int` signature, short-window guard, the shape guard |
| entrypoints | 41 / 40 | pending | schedule firing tables, CLI defaults, budget matching |

**config.** `get_available_device` falls through CUDA, then MPS, then CPU;
one rebuild dropped the MPS branch and returned `cpu` on a host where MPS
is available. That is not only slower -- MPS and CPU kernels are not
bit-identical, so the same config runs a numerically different job. Pinned
against what the installed torch reports rather than a hard-coded name, so
the assertion holds on a CUDA box, an Apple laptop and a CPU-only runner
and still catches a missing branch. Also pinned: `deep_update` mutates its
first argument and returns it while REPLACING nested dicts rather than
mutating them (both halves observable, and the rebuilds picked different
halves); `kv_to_nested` does not strip keys, accepts an empty key, and
raises when asked to index through a scalar; `env_overrides` nests on `.`
and NOT on `__`, which is the convention a reader is likely to assume.

**model_harness.** `_unpack` and `_to_scalar` are called from
`continuous_monitor.py` and `register_task` from `continuous_trainer.py`,
so all of it is cross-layer contract despite the underscores. The registry
cap matters most: it defaults to 50, evicts the OLDEST, and keeps
`task_diagonals` aligned with `eval_past_tasks()` -- the alignment is the
point, per `register_task`'s own docstring, because the transfer metrics
index both by position and a length mismatch silently scores task i
against task j. Rebuilds defaulted the cap to 20 and to None.

**trainer.** Both rebuilds produce BIT-IDENTICAL weights on every
configuration probed, so the training computation was already equivalent
and nearly all the divergence was internal naming -- left free, with the
identical weights as the evidence. Two things were not: the declared
`-> int` return (the project runs `mypy .`, so a `None` return under that
signature fails its own type gate), and `_safe_next`'s refusal of a window
smaller than one batch. The latter is reachable from configuration --
`outer_cl_training_loop` passes `min_batch=cfg.train.batch_size` -- so a
`batch_size` larger than the drift window hits it, and one rebuild trained
on the short batch instead and reported losses. Silently training on a
fraction of the intended batch after every drift event is exactly the
difference that moves a published number and leaves no trace.

**entrypoints.** The largest and the most consequential, because
`cl_only.py`'s schedules ARE the experimental controls: `never` as the
frozen lower bound, `periodic` and `random` as rate-matched nulls. Both
rebuilds broke them. `RandomSchedule(prob=0.5)` fired on no window at all,
so the rate-matched null silently became the frozen baseline and a
"drift-triggered beats random" result would have been measured against
nothing happening. The same schedule produced an identical pattern for
every seed, so the control's variance would have read as zero. And
`make_schedule` refused `--budget` without `--period`, which is the
budget-matching path the module documents as preferred -- the period is
derived as `max(1, round(horizon / budget))` -- and refused a
non-positive `--horizon`, where the source falls back to
`cfg.drift_detection.max_stream_updates` and the CLI default is 0, making
the fallback the normal path.

Pinned as exact firing tables over fixed seeds: `np.random.default_rng` is
reproducible and the claim pins numpy in its lock, so the patterns are
computations rather than recordings. The CLI defaults are pinned with them
because the two interact -- `period`, `prob`, `budget` and `horizon`
default to numeric zeros and every branch tests them with `>`, so a rebuild
defaulting them to `None` turned each test into a TypeError.

One correction during the work: the budget-derived period table was first
written from arithmetic in my head (`round(24/4) = 5`) and failed against
the source. Measured instead: budget 4 -> period 6, budget 3 -> 8,
budget 6 -> 4, budget 24 -> 1.
Kind: APEIRON (missing tests)
Status: open -- entrypoints' rebuild still being earned.

### A57 — seven places where the reconstruction is right and the source is not
Collected because they share a disposition: all are left UNPINNED, because
pinning the current behaviour would freeze a defect, and choosing the
replacement belongs to the team.

| layer | surface | source | reconstruction |
|---|---|---|---|
| model_harness | `_unpack({"x":…, "y":…})` | returns the dict's KEYS, two strings | raises |
| model_harness | `_to_scalar(long/bool tensor)` | RuntimeError out of `mean()` | coerces |
| model_harness | `register_task(1.0)` | `TypeError: 'float' object is not iterable` | accepts as `[1.0]` |
| config | unknown TOML section | silently ignored | rejected |
| config | unknown TOP-LEVEL key | `TypeError: TrainCfg.__init__() got an unexpected keyword 'x'` | names the right place |
| drift_detection | non-string `ensemble_voting` | `AttributeError: 'NoneType' has no attribute 'strip'` | clean ValueError |
| evaluation | degenerate input | leaks `ZeroDivisionError` / torch's `t()` | ValueError naming the problem |

The `_unpack` one is the sharpest: `x, y = batch` on a dict unpacks its
KEYS, so a harness whose loader yields a dict gets two strings where the
monitor expects two tensors, and the failure surfaces later as
`'str' object has no attribute 'to'` -- an error about strings, from a
line about devices.

The config pair is held back for a second reason as well: an earlier
config-assertion PR was declined in favour of documenting the behaviour,
so asserting either answer here would re-open a settled question rather
than ask it.
Kind: APEIRON (defects, reconstruction is right)
Status: open -- seven one-line decisions, none of them ours to make.

### A58 — nine layers passed, and the framework they compose does not import
The point of the exercise is a replacement, so the test is a replacement:
take the blind rebuild of every layer, assemble one `apeiron/` tree out of
the files each layer generates ITSELF (`produce` steps with no `from`), and
run the pinned MNIST training claim against it. 31 files, 1,787 lines, not
one of them written by something that had seen the original.

It would not import.

    File "apeiron/model/torch_model_harness.py", line 8
      from apeiron.config.configuration import Config, torch
    ImportError: cannot import name 'torch' from 'apeiron.config.configuration'

**Why every gate missed it.** A layer's suite runs it against the REAL
implementations of the layers beneath. So a reconstruction may lean on any
name those happen to expose and still pass. The regrown harness reached
`torch` through the config module's namespace; `src/config/configuration.py`
does `import torch` at its top for `get_available_device`, so the name was
there and the harness's gate was satisfied. The regrown CONFIG has no need
for torch at module scope and does not import it. Neither layer is wrong by
its own criteria, and the pair does not compose. The source harness has no
such dependency -- it does `import torch` directly.

Fixing that exposed a second edge of the same problem:

    File "apeiron/__init__.py", line 7
      from apeiron.training.updater import BaseUpdater
    ImportError: cannot import name 'BaseUpdater' from
                 'apeiron.training.updater'

A package `__init__` is a contract BETWEEN layers, and only the top-level
one was ever pinned (`test_public_api.py` asserts `apeiron.__all__`).
The sub-packages that top level imports FROM were not, so three of them
came back short:

| package | names the regrowth omitted |
|---|---|
| `apeiron.training.updater` | `BaseUpdater` |
| `apeiron.logger` | `StageType`, `VALID_STAGES` |
| `apeiron.drift_detection` | all six detector classes |

Each omission is invisible to the layer that made it and fatal to the
composition. This is the sharpest form of the campaign's central point:
twenty rooms at agreeing roots, nine of them probed to zero or near-zero
contract divergence, and the thing they add up to does not start.

**Pinned, in two kinds.** `test_import_surface.py` (model_harness,
updaters) forbids a cross-layer import from pulling a MODULE out of a
sibling -- resolving each imported name and rejecting the ones that are
modules, which needs no list of third-party packages and names the
offending line. `test_export_surface.py` (seven layers) pins each
package's re-export set exactly, in both directions: every expected name
importable, and no unexpected ones, because a name exported here is one a
sibling may come to depend on.

Validated both ways throughout. Against `src/`: 11, 12, 3, 3, 3, 3, 3 pass.
Against the rooms that were short: drift_detection 8 failures, logger 3,
updaters 2, and profilers passes because it was never short -- which is the
control.

One of my own criteria was wrong first and the source said so: the
import-surface check asserted that every module imports torch directly, and
`create_updater.py` is a factory that never touches torch. Made conditional
on actual use. Third time this session that running a criterion against the
implementation before believing it caught something I had reasoned my way
into.
**A note on how the rebuild batch read, because it nearly misled me.**
Seven layers were resealed and rebuilt; three passed and four "FAILED after
3 attempts". Twelve failed attempts in a row looks exactly like an
unreachable criterion -- the trap of pinning something no reconstruction
can satisfy, which is as useless as pinning nothing. It was not. The
producer had hit its account quota:

    'You've hit your usage limit ... try again at 3:57 PM.'
    producer_codex: codex exited 1 and the gate has not passed

A producer refusal says nothing about the criteria and belongs in its own
bucket, not in the failure count. The four were re-run on the other
producer family, which has a separate quota. The criteria's reachability is
established by the three that passed and by the seven that pass against
`src/`, not by the absence of refusals.
Kind: APEIRON (missing tests)
Status: open -- the assembled run is the next measurement.

### A59 — APEIRON: `grad_accumulation_steps = 0` silently produces NaN weights
`BaseUpdater.fwd_bwd` scales the loss by the accumulation count
(`updater/base.py:55`):

    loss = loss / self.cfg.train.grad_accumulation_steps

`grad_accumulation_steps` is a config key defaulting to 1
(`configuration.py:104`), and nothing validates it. Set it to 0 and the
division is by zero -- a tensor division, so no exception:

    grad_accumulation_steps=0: loss=inf
    grad_accumulation_steps=1: loss=1.07294
    grad_accumulation_steps=2: loss=0.53647

`inf` then flows through `backward()` into the optimizer step, and the
probe's full step records the result: every weight becomes NaN. The run
continues, the metrics file fills with NaN, and nothing anywhere says the
configuration was impossible.

A reconstruction returned a finite loss here instead, which is how this
surfaced: it was one of sixteen divergences in a room whose weights were
otherwise bit-identical.

Not pinned, for the usual reason -- the fix is a validation decision
(reject 0, clamp to 1, or document) and belongs to whoever owns the config
surface. Worth noting it is cheap: the same `>= 1` check the schedule's
`--period` already has.
Kind: APEIRON (defect)
Status: open -- a one-line guard, if the team wants it.

### A60 — the logged forgetting loss was pinned by nothing
`update_post_fwd_bwd()`'s return value looks internal and is not.
`ContinuousTrainer` takes it as `reg_loss` (`:311`), returns it from
`inner_cl_training_loop` as `forgetting_loss`, and logs it (`:191`-`:196`):

    "jvp_reg_total_loss": generation_loss + forgetting_loss,
    "jvp_reg_forgetting_loss": forgetting_loss,

so the hook's number becomes two columns of the metrics CSV. Measured
against `src/`, with no historical batch supplied, **every one of the five
update modes returns exactly 0.0** under both `mix_historic_data`
settings -- nothing has been forgotten yet, so no regularisation term
applies.

A rebuild returned the full training loss instead, on every `jvp_reg`
configuration, **with weights bit-identical to the original's**. That is
the worst shape a divergence can take: the training is correct, the run
looks healthy, and the published forgetting curve rises from zero to the
generation loss while the total double-counts the same number. A probe
comparing only weights would have called the layer clean -- and for the
codex family it did, at 0 divergences, which is why this went unexamined
until the second producer family was run against the same criteria.

That is the argument for two families stated as a measurement rather than
a principle: the same criteria, a different oracle, and a logged metric
nobody had pinned.

Pinned as `claims/updaters/test_reported_penalty.py`, 20 cases across all
five modes and both mix settings, including that the value is not merely
falsy (`None` would pass a truthiness check and then crash the trainer's
`generation_loss + forgetting_loss`). Validated: 20 of 20 against `src/`,
4 failures against the room that reported the loss.
Kind: APEIRON (missing tests)
Status: closed here -- claim resealed to `23ad1babff15`.

### A61 — the regime ladder is unspecified where its thresholds cross
The regime is the detector's instruction to everything above it: `STABLE`
means carry on, and `CONTINUAL_LEARNING`, `FINE_TUNING` and `RETRAIN` ask
for progressively more adaptation. So a different regime for the same
drift is a different training run.

`test_drift_severity.py` covers the ordinary case -- thresholds ascending,
score between them. Nothing covers equal or inverted thresholds. The ladder
is a plain chain (`statistical_detectors.py:84`-`:91`) and `minor` is
tested FIRST whatever its relation to `moderate`:

| minor | moderate | score | regime |
|---|---|---|---|
| 0.3 | 0.6 | 0.01 | continual_learning |
| 0.005 | 0.015 | 0.01 | fine_tuning |
| 0.0 | 0.0 | 0.01 | retrain |
| 0.01 | 0.01 | 0.01 | retrain (`<` is strict) |
| **0.02** | **0.01** | 0.01 | **continual_learning** |
| -1.0 | 2.0 | 0.01 | fine_tuning |

The inverted row is the one that matters. `minor=0.02, moderate=0.01`
reads as a configuration asking for a low bar, and because `minor` is
tested first the answer is the LIGHTEST response. A reconstruction that
normalised the thresholds, or tested `moderate` first, answered `RETRAIN`
for the same input -- the heaviest response, on a stream the original would
barely have touched. Same step, same drift score, opposite ends of the
ladder.

Both are defensible readings of an inverted configuration. What is not
defensible is leaving it unspecified when the two differ by the whole width
of the ladder. Pinned as the measured table, with the strict `<` boundary
stated separately and a flat-stream control that no threshold pair can
change.

Found only because the second producer family was run against the hardened
criteria: the codex room agreed with the source here, and the claude room
did not. Second finding in a row that one oracle would have missed
entirely (see A60).
Kind: APEIRON (missing tests)
Status: open -- claim resealed to `300f5458a8cd`, rebuild in flight.

### A62 — the assembled framework imported, then died mid-training
With A58's two composition holes closed, the nine blind rebuilds assembled
into a framework that imports, and `check_mnist_harness.py` -- the pinned
harness gate, 2,000 MNIST images, affine drift schedule, stochastic
forward -- **passes against it**. Then the run gate started training and
stopped on this:

    File "apeiron/training/continuous_trainer.py", line 371
      self.profiler.print_performance(logger=self.logger, level=1)
    TypeError: FLOPSProfiler.print_performance() got an unexpected keyword
               argument 'logger'

The third edge of the same problem. A58 pinned which NAMES a package
exports; it said nothing about the signatures of those names. The trainer's
gate ran it against the SOURCE profiler, whose
`print_performance(logger=None, level=0)` accepts both keywords. The
profilers' own gate had no caller that passed either, so its reconstruction
dropped them and passed. Two layers, both correct by their own criteria,
and the pair breaks at the first drift event.

Pinned as `test_call_signatures.py` across five layers -- 21 methods that
are called from another layer -- asserting PARAMETER NAMES and which ones
keep defaults, rather than the signature string, because annotations are
free to change and a keyword is what a caller depends on.

**It found two more that had not surfaced yet.** Validated against the
rebuilt rooms: profilers fails 2 (the crash), logger fails 8, and
model_harness fails 1. The logger and harness divergences were latent --
they sit on call paths the run had not reached when the profiler killed it,
so fixing only the visible one would have bought one more crash, further
in. That is the argument for pinning a surface rather than chasing a
traceback.

The pattern across all three edges is identical and worth stating plainly:
**a layer's criteria are checked against real neighbours, so anything the
neighbours happen to provide is untested.** Borrowed module names, missing
re-exports, and dropped keyword parameters are three faces of it. Each was
invisible to every per-layer measurement -- twenty rooms, agreeing roots,
nine layers probed to zero or deliberate residuals -- and each was found
only by assembling the thing and running it.
Kind: APEIRON (missing tests)
Status: open -- five claims resealed, rebuilds in flight; the run gate is
the next measurement.

### A63 — the fourth composition edge: which collaborator method gets called
With the signatures pinned, the assembled framework imports, passes the
pinned MNIST harness gate, and gets further into training before stopping:

    File "apeiron/profilers/count_flops.py", line 260, in _emit
      candidate(message)
    File "apeiron/logger/logger.py", line 158, in log
      self._metrics.log(metrics, ...)
    File "apeiron/logger/wandb_logger.py", line 163, in log
      for key, value in metrics.items():
    AttributeError: 'str' object has no attribute 'items'

`print_performance` **documents its own contract** and the source honours
it exactly:

    logger: Optional logger instance. If provided, uses logger.info().
    output_fn = (lambda msg: logger.info(msg, level=level)) if logger else print

The reconstruction duck-typed instead -- `for method in ("log", "info",
"print")`, falling back on `TypeError`. Against the real `Logger` that
fails in a way no type check catches. `Logger.log(metrics, step=...,
commit=..., prefix=..., increment=...)` accepts a string POSITIONALLY as
`metrics`, so the call succeeds; it fails two frames deeper inside the
backend, with an `AttributeError`, which is not what the fallback chain
catches -- so `info` is never tried either.

A documented behaviour, unasserted, and invisible to every gate on both
sides of the boundary: the profiler's own tests pass a double or no logger,
and the logger's tests never see the profiler.

Pinned with a recording double rather than by reading the source, because
what matters is the call that actually happens. The double's `log` is
deliberately as strict as the real one about its first argument being a
mapping, so a profiler reaching for `log` fails in the test the same way
it fails in a run instead of being tolerated by a permissive stub.
Validated: 10 of 10 against `src/`, 8 failures against the duck-typing
room.

**Four edges now, all the same shape.** A module borrowed through a
sibling's namespace; a package that re-exported too little; a method whose
keyword parameters were dropped; and a collaborator method chosen by
guesswork. Every one passed both layers' gates, and every one was found
only by assembling the framework and running it. The generalisation is
worth stating as the campaign's main result: **criteria that test a layer
against its real neighbours cannot specify the boundary between them.**
What the neighbour happens to offer is free, and a reconstruction will
spend that freedom.
Kind: APEIRON (missing tests)
Status: open -- profilers resealed and rebuilding; the run gate continues.

### R25 — methodology: my validation harness hit the resolution trap
Validating the new monitor criterion, I ran it as

    PYTHONPATH=<room> pytest --rootdir=claims/monitor claims/monitor/test_*.py

and it passed against `src/` AND against the room. I read that as "the
criterion does not catch the divergence" and started looking for another
layer to blame.

It was the resolution trap, in my own check. `pytest` puts the rootdir on
`sys.path` ahead of everything, and `claims/monitor/` contains the claim's
own sealed `apeiron/` tree -- so both runs loaded the SOURCE and the room
was never imported. The earlier layers were validated by copying the test
file to a neutral directory, which is why they were sound; I shortened the
loop for brevity and reintroduced the bug R20 exists to prevent.

Re-run from a neutral cwd with only the room on PYTHONPATH: `src/` passes
6 of 6, the room fails 5 of 6. The criterion was right the whole time.

The lesson is narrower than R20's and worth separating: the probe harness
has `_assert_resolves` and refuses a vacuous comparison, and **my ad-hoc
validation commands do not**. Any check run outside the harness needs the
same guard applied by hand -- confirm what the interpreter actually
imported before believing either outcome. Validate from a neutral
directory, with one tree on the path.
Kind: methodology (fixed here)
Status: fixed here -- validations re-run from a neutral cwd.

### A64 — the metrics file's schema is unpinned, and a run can write gibberish
The assembled framework -- nine blind rebuilds, every layer at zero or
deliberate residual contract divergences -- imports, passes the pinned
MNIST harness gate, and **trains the entire run without error**. The run
claim then fails:

    AssertionError: drift fires at [], expected [350]

Drift had fired. The check selects `drift/score` from the produced CSV,
exactly as an analysis would, and found nothing. Measured as a unit, one
monitor loop under each implementation:

| | names a run logs |
|---|---|
| original | `eval/accuracy`, `eval/mean_logit`, `drift/score`, `drift/detected`, `drift/regime`, `drift/confidence`, `drift/metric_0`, six `drift/cperf_*` |
| rebuild | `drift/metric_0`, `drift/drift_score`, `drift/drift_detected`, `drift/batch_count`, `drift/stream_update_count`, `drift/drift_event_count` |

Twelve names against six, overlapping in one. The rebuild logs none of the
per-batch evaluation metrics, none of the profiler measurements, and no
regime or confidence; it writes monitor counters the original never
records; and it re-prefixes the two columns it shares, so the stage
namespace is applied twice (`drift/drift_score`).

`test_logged_schema.py` pins the `cperf_` prefix and nothing else -- and
with no profiler fields logged at all, that test had nothing to check.

**This is the quietest failure the campaign has produced.** Every layer
passed its gate. The composition imported. Training completed. The weights
were computed. And the artefact the whole framework exists to produce is
unreadable by the tooling that consumes it -- a file in which no analysis
can find the accuracy or the drift events. It is also the sharpest argument
for grading a rebuild by the surface a probe covers: the behaviour probe
drove this layer to **zero** contract divergences on counters and trained
weights, and never looked at what the run writes down.

Pinned as the stage-qualified name set a loop logs, with the eval names
DERIVED from the harness's own `eval_metrics` rather than listed -- the
first version hard-coded two names measured from a different fixture and
failed against `src/`, which is the third time this session that a
criterion had to be measured with the fixture it would run under.
Validated from a neutral directory (see R25): 6 of 6 against `src/`, 5 of
6 failing against the room.

NOT pinned, filed: `detected` is also emitted on a 10% sample of
non-events via the unseeded global numpy RNG, so whether that column
appears on a given row varies between runs of identical code. The schema
is nondeterministic by design -- A41's class, and a decision for whoever
owns the metrics.
Kind: APEIRON (missing tests)
Status: open -- monitor resealed and rebuilding.

### R26 — WITHDRAWN: a count table pinned against an unseeded fixture
Chasing the MNIST run's two-step offset, I compared drift-event and
evaluation counts between the source and a rebuilt monitor with drift
forced, and reported a divergence:

    cap=1   src 11 events / 52 evals     room  7 events / 44 evals
    cap=2   src 12 / 86                  room 15 / 92
    cap=3   src 28 / 150                 room 29 / 152
    cap=5   src 40 / 238                 room 44 / 246

and wrote a criterion pinning those numbers. **It failed against `src/`
on three of four rows**, which is what sent me to check the obvious thing
I should have checked first: running the source three times gives

    cap=1   7, 7, 10        cap=2   19, 14, 18
    cap=3   24, 18, 35      cap=5   36, 43, 52

The fixture harness in `claims/monitor/conftest.py` builds its tensors with
unseeded `torch.randn`, so every instantiation streams different data and
the detector fires a different number of times. The "divergence" was two
samples from that distribution, one per tree. There is no finding here, the
criterion is deleted, and it was never wired into the claim.

Two things went wrong and only one of them is about the fixture. The
fixture is unseeded, which is a real hazard. But I also compared one run
against one run and called the difference a divergence, with no check that
either side repeats -- and the probe harness has done that check since
A38, across six hash seeds, precisely because a single draw cannot tell a
difference from noise. I stepped outside the harness and lost the
guarantee it provides, the same way R25 lost the resolution guarantee.

**The rule, stated so it is checkable: before comparing two
implementations on any quantity, run ONE of them twice.** If it does not
repeat, the comparison is measuring the fixture.
Kind: methodology (withdrawn finding)
Status: withdrawn -- criterion deleted, no claim resealed.

### A65 — the assembled run performs two extra evaluations, deterministically
What survives R26's retraction, because it is measured against a SEEDED
harness and reproduces: the full-stack run of the assembled framework logs
**378** `eval/accuracy` rows where the original logs **376**, and the
original's accuracy sequence is the rebuild's **shifted by exactly two**:

    original  46.875, 34.375, 50.0,  40.625, 50.0, 59.375
    rebuild    50.0,  53.125, 46.875, 34.375, 50.0, 40.625

`drift/metric_0` is identical in both -- 37 values, no shift -- so the
aggregation the detector sees is unaffected, and the first two stream
batches and the RNG state after them are byte-identical between the trees.
The two extra evaluations are therefore additional, not different: the
rebuild evaluates twice more than the original over the same stream, and
every step label after that is offset by two. That is why the run claim
reads `drift fires at [352]` where the pinned answer is `[350]`, and it
would also fail `EXPECTED_EVAL_POINTS = 376`.

Both implementations call `model.eval()` before evaluating and both honour
`reset_after_learning` identically, so neither is the cause. Not located
further: the conftest fixture cannot be used to chase it (R26), and a
deterministic fixture for this layer -- a seeded stream harness -- is the
prerequisite for the next attempt rather than something to work around.

Kind: measurement
Status: open -- reproducible, cause unlocated, and the instrument needed to
locate it does not exist yet.

### A65 — located: the warmup window is evaluated and not logged
The two-row offset in the assembled run is `_evaluate_batch`'s gate:

    if self.batch_count > self.flops_profiler.warmup_iters:
        with self.flops_profiler.measure_flops(tag="infer"):
            ...  evaluate, then logger.stage("eval"); logger.log(...)
    else:
        ...  evaluate, and log nothing

For the first `warmup_iters + 1` batches the model is evaluated and the
result is discarded. With the default `warmup_iters = 1` that is exactly
two rows missing from the head of the metrics file, which is the offset:
the original writes 376 rows, the rebuild 378, and the original's accuracy
sequence is the rebuild's shifted by two. Every step label after the head
moves, so the run claim read `drift fires at [352]` against the pinned
`[350]`.

How it was narrowed, after R26's retraction taught me not to trust a
single draw:

1. A **seeded** monitor fixture, verified to repeat across three runs
   before any comparison, showed the regrown monitor agreeing with the
   source on every counter -- so the accounting was not the cause.
2. Swapping one regrown layer at a time into an otherwise-source tree
   isolated it to `monitor` in nine runs of a cheap reproducer (the MNIST
   harness at `max_stream_updates=2`: 124 rows against 126, about a
   minute per run).
3. Hashing the batch each implementation unpacked settled what it was
   NOT: `_unpack` is called 126 times in both, with byte-identical
   batches in identical order. Both evaluate the same data; only the
   logging differs.

Pinned at the unit level rather than through a loop, deliberately: a
seeded `DummyHarness` loop agreed on the row count in both
implementations while the MNIST harness did not, so a count assertion over
a fixture would have passed and taught nothing. The criterion asserts the
gate -- no row while `batch_count <= warmup_iters`, a row after, at four
warmup settings, with the `>` boundary stated separately and a case
confirming the evaluation still HAPPENS during warmup (silent is not
skipped: those metrics still feed the drift buffer). Validated: 12 of 12
against `src/`, 5 failures against the room.

One incidental find along the way, entirely my own: a diagnostic script
named `bisect.py` in the scratch directory shadowed the standard library,
so `import random` -- which imports `bisect` internally -- loaded my file
and tried to read a claim. Same family as every other resolution failure
in this log, and a reminder that `sys.path[0]` is the script's own
directory.
Kind: measurement
Status: closed -- cause located and pinned; monitor resealed.

### A66 — the metrics file's contents are coupled to the profiler's warmup
Noted while pinning A65, and worth raising on its own because it is a
design wart rather than a reconstruction gap.

`warmup_iters` exists so FLOP measurements skip cold iterations. Because
the logging call sits INSIDE the post-warmup branch of
`_evaluate_batch`, it silently also decides how many evaluations reach the
metrics file. Two consequences:

  * the file's first `warmup_iters + 1` evaluations are missing, with
    nothing in the file saying so -- an analysis that counts rows or
    aligns them to batch indices is quietly off by that many;
  * raising `warmup_iters` to profile more carefully DELETES data points
    from the analysis, which is not a trade anyone would choose
    deliberately.

Moving the `stage("eval")` / `log(...)` pair outside the branch, leaving
the profiling inside it, would decouple them: every evaluation recorded,
cold iterations still unprofiled. That is a behaviour change -- it alters
the row count of every existing run -- so it belongs to whoever owns the
metrics rather than to a criterion, and A65 pins the current gate without
assuming the fix.
Kind: APEIRON (improvement)
Status: open -- a two-line move, if the team wants it.
