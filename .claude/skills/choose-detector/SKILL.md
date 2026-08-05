---
name: choose-detector
description: |
  Help the user pick a drift detector and tune its settings for an apeiron run.
  Use when the user asks which detector to use, how to configure drift detection,
  what ADWIN/KSWIN/PageHinkley/threshold values to set, whether to combine
  detectors into an ensemble and which voting rule to use, or wants a ready-to-use
  [drift_detection] config block. Asks a few questions about the monitored metric
  and drift shape, recommends a detector, writes a filled-in TOML block, and
  validates that it loads. Does NOT run a full experiment — for that use
  explore-examples or custom-experiment.
argument-hint: "[config_path]"
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - AskUserQuestion
---

Help the user choose a drift detector and produce a validated `[drift_detection]`
config block. The authoritative reference for detector behavior and every option
is `docs/drift_detectors.md` — read it first and keep this skill consistent with
it rather than restating numbers that may drift.

## Arguments
- `$1`: Optional path to an existing config TOML. If given, patch its
  `[drift_detection]` section in place. If omitted, emit a standalone block the
  user can paste into their config.

## Ground truth to respect (do not recommend around it)
`ContinuousMonitor._check_drift()` calls `detector.update(agg_metric)` with a
single aggregated scalar and no kwargs. Anything that needs more than that scalar
is not drop-in.

Plug-and-play:
- `ADWINDetector`, `KSWINDetector`, `PageHinkleyDetector` — take the scalar directly.
- `EnsembleDetector` — its `update(value, **kwargs)` forwards the scalar to each
  sub-detector, so it is drop-in **provided every name in `ensemble_detectors` is
  one of the three scalar detectors above**. Listing a non-drop-in detector as a
  sub-detector pushes the failure into the ensemble.

**Not** drop-in; do not recommend as the default:
- `ModelPerformanceDetector` needs reference data + batch DataFrames (not passed by
  the monitor); `update()` raises `ValueError` unless `set_reference()` ran first.
- `EvalDetector` (`ModelEvalDetector`) needs extra `update(...)` kwargs
  (`modelHarness`, `reference_validation_metrics`, `higher_is_better`) the monitor
  doesn't send.

Confirm this is still true before relying on it:
```bash
sed -n '1,100p' src/apeiron/drift_detection/load_drift_detector.py
```
If a user specifically wants one of the non-wired detectors, be honest that it
requires extra wiring and point them at the integration notes in the doc.

## Procedure

### 1. Understand the monitored signal
Ask the user (batch these with AskUserQuestion):
- **What metric** feeds the detector, and its rough scale? Bounded like accuracy
  or error rate in `[0, 1]`, or unbounded like a loss? (This drives `metric_index`
  and threshold scaling.)
- **What drift shape** do they expect?
  - abrupt jumps in the mean
  - gradual/slow drift
  - distribution or variance change with little mean movement
- **Sensitivity vs. false alarms**: react early and tolerate some false alarms,
  or only fire on clear, sustained drift? (A strong preference at either extreme,
  or "I expect more than one kind of drift", is the cue to consider an ensemble.)
- **Cadence**: roughly how many `update()` calls (i.e. `detection_interval`-sized
  checks) happen before they'd want a first detection, and how many batches per
  check. This sets warm-up expectations.

Note: the scalar detectors fire on **change in either direction** — they do not
know "good" from "bad". If the user only cares about degradation, say so plainly
(that is what the non-wired `EvalDetector` is for).

### 2. Recommend a detector
Map answers to a detector:
- distribution / variance / shape change without mean movement → **KSWIN**
- abrupt mean shift, want fast + cheap detection → **PageHinkley**
- gradual, mixed, or "not sure / general default" → **ADWIN**
- more than one drift shape expected, or an explicit sensitivity/false-alarm
  preference a single detector can't express → **Ensemble** over two or three of
  the above (see the voting rules in step 3)

Prefer a single detector when one clearly fits — the ensemble costs an update on
every sub-detector per check and makes tuning harder to reason about, since each
sub-detector is still driven by its own hyperparameters in the same config block.
Reach for it when the shapes genuinely differ (e.g. PageHinkley for abrupt jumps
plus KSWIN for variance changes) or when the user wants a deliberate
sensitivity/conservatism bias they can state as a voting rule.

State the recommendation and the one-line reason. If it's a close call, name the
runner-up and the tradeoff.

### 3. Recommend settings (scale to the metric)
Pull defaults and semantics from `docs/drift_detectors.md`. Adjust for the
metric scale and the sensitivity preference:
- **ADWIN** — `adwin_delta` is the main knob: lower (`~0.001`) = fewer false
  alarms/later, higher (`~0.01`) = more sensitive. Keep `adwin_minor_threshold`
  / `adwin_moderate_threshold` at `0.3` / `0.6` unless they want to steer the
  CL→fine-tune→retrain regime split.
- **KSWIN** — `kswin_alpha` for sensitivity; size `kswin_window_size` /
  `kswin_stat_size` to how many samples they can retain (`stat_size < window_size`).
- **PageHinkley** — `ph_threshold` scale **depends on the metric**: for a bounded
  metric in `[0,1]` (accuracy/error) the default `50` is very large and will
  rarely fire, so start much smaller (order `1–10`) and tune; for larger-magnitude
  losses, larger thresholds are appropriate. `ph_delta` is the slack (min change
  treated as real); `ph_min_instances` is warm-up.
- **Ensemble** — `ensemble_detectors` is the list of sub-detector names; each is
  built from this same `[drift_detection]` block, so a detector type can appear at
  most once and still needs its own hyperparameters set here. An empty list, a
  nested `"EnsembleDetector"`, or an unknown voting name raises `ValueError` at
  load. `ensemble_voting` sets the bias:
  - `any` (alias `or`) — fires when any sub-detector fires. Most sensitive; use
    when a missed drift costs more than a needless CL dispatch.
  - `majority` (default) — strictly more than half. Balanced; needs 3+ detectors
    to mean anything (with 2 it behaves like `unanimous`).
  - `unanimous` (aliases `all`, `and`) — every detector must fire. Most
    conservative; suppresses small/noisy changes at the cost of latency.
  Note `drift_score` is the mean of sub-detector scores and the regime is a
  plurality vote, both independent of the voting rule — so the
  `adwin_minor_threshold` / `adwin_moderate_threshold` regime split gets diluted
  by sub-detectors that report a score of 0.

Set `detection_interval`, `aggregation` (`mean`/`median`/`last`), `metric_index`,
and `max_stream_updates` from the cadence answers. Explain any value that
deviates from the doc default.

### 4. Produce the config block
Emit a complete, paste-ready section, e.g.:
```toml
[drift_detection]
detector_name = "ADWINDetector"
detection_interval = 10
aggregation = "mean"
metric_index = 0
reset_after_learning = false
max_stream_updates = 20

adwin_delta = 0.002
adwin_minor_threshold = 0.3
adwin_moderate_threshold = 0.6
```
For an ensemble, list the sub-detectors and keep each one's hyperparameters in the
same block:
```toml
[drift_detection]
detector_name = "EnsembleDetector"
ensemble_detectors = ["ADWINDetector", "PageHinkleyDetector"]
ensemble_voting = "unanimous"
detection_interval = 10
aggregation = "mean"
metric_index = 0
reset_after_learning = false
max_stream_updates = 20

# Sub-detectors read their usual hyperparameters from this same block
adwin_delta = 0.002
ph_threshold = 30
ph_delta = 0.5
```
If `$1` was given, patch that file's `[drift_detection]` section (Edit); keep the
keys that don't belong to the chosen detector untouched or drop the unused
detector-specific keys, matching the existing file style.

### 5. Validate that it loads (no full run)
Build the config and instantiate the detector — this confirms the TOML parses,
the dataclass validates, and the detector name + params are accepted, without a
training run:
```bash
PYTHONPATH=src poetry run python -c "
from apeiron.config.configuration import build_config
from apeiron.drift_detection.load_drift_detector import load_drift_detector
cfg = build_config(['--config', '<config_path>'])
d = load_drift_detector(cfg)
print('OK:', type(d).__name__)
print(getattr(d, 'voting', ''), [type(s).__name__ for s in getattr(d, 'detectors', [])])
print(cfg.drift_detection)
"
```
For an ensemble this is worth more than a syntax check: it is where an empty
`ensemble_detectors`, a nested `EnsembleDetector`, an unknown voting name, or an
unknown sub-detector name surfaces as a `ValueError` instead of at run time. The
second print confirms the resolved voting rule and that every sub-detector built.
(`PYTHONPATH=src` is required so `import apeiron` resolves — the package lives
under `src/apeiron` and `import apeiron` fails without it.)
For a standalone block (no `$1`), write it to a temp file first and validate that
(it still needs the other required sections `[model]`/`[data]`/`[train]` to build
a full `Config`; if the user has no config yet, validate against an example TOML
with `--set drift_detection.detector_name=...` overrides instead, or just confirm
the detector loads via `load_drift_detector` with an example config).

Report the outcome plainly: recommended detector, why, the settings that differ
from defaults, and that the config loaded successfully. To actually observe
detection behavior, hand off to `explore-examples` or `custom-experiment`.

## Notes
- Quick way to A/B a detector on a shipped example without editing files:
  `poetry run python -m src.main --config examples/mnist/mnist.toml --set drift_detection.detector_name=PageHinkleyDetector --set drift_detection.ph_threshold=5`
- `--set` values go through `json.loads`, so a list needs JSON syntax and shell
  quoting: `--set 'drift_detection.ensemble_detectors=["ADWINDetector","KSWINDetector"]'`
- Precedence when sources disagree: the code in
  `src/apeiron/drift_detection/` wins, then `docs/drift_detectors.md`, then this
  skill. Fix whichever is stale rather than working around it — this file has
  been wrong about detector wiring before.
