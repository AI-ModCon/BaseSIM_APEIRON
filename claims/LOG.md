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
| R15 | bug | fixed here | reticuli: `--producer codex` reached no adapter, and the adapter could not have run |

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
Status: fixed here, upstream in reticuli PR #11. `codex` and `claude` are
nameable in a CLI-producer table that preflights a binary on PATH;
`--sandbox workspace-write` replaces the bypass; the rollout log is
identified by mtime against a watermark sampled before launch when no
thread id is offered. `handlers_check` exercises both tables and
`surface_check` now requires every shipped adapter to be nameable and
named in help — an adapter nothing can reach is a trap, and that is the
gate that should have caught this. `gate.py` → `repo-ok`, 209 tests.

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
