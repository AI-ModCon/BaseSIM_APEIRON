#!/usr/bin/env python3
"""Behaviour probe: apeiron.drift_detection, off the pinned set.

The drift layer is where this whole line of work started and where its
weakest numbers live: the lowest kept-ratio with the widest spread (A33: 46%
with a spread of 7, and two of four draws failing outright), the thinnest
mutation score (A9: 23%), and the most open findings (A6, A7, A21, A22, A31).
A37 then found that even a detector reconstructed at high fidelity, passing
its gate and independently redone three times, was a different algorithm.

This probe does not re-ask A37's question -- the three statistical detectors
are now domesticated claims with their own differential probe
(claims/scripts/heldout_detectors.py, 18 held-out streams). It asks about
the APEIRON WRAPPER around them, which nothing has probed: how a raw
detector verdict becomes a DriftSignal, where the regime thresholds cut, how
the ensemble combines votes, and what the loader does with a name it does
not know.

That wrapper is exactly where the open findings sit. A22 is the regime/
verdict disagreement; A6 and A7 are the evidently coupling; A21 is
ModelEvalDetector printing, asserting and fabricating a score. A probe that
only exercised the detectors would miss all of it.

Determinism: the stream is fixed, KSWIN is always seeded, and no model or
data is involved. The evidently-backed ModelPerformanceDetector is probed
only for whether it can be constructed and what it raises if not -- running
it needs a dataframe pipeline, and A6/A7 already record that it is coupled.
"""
import hashlib
import json
import math
import sys

from apeiron.drift_detection.detectors.base import (  # noqa: E402
    DriftSignal, LearningRegime,
)
from apeiron.drift_detection.detectors.statistical_detectors import (  # noqa: E402
    ADWINDetector, KSWINDetector, PageHinkleyDetector,
)

try:
    from apeiron.drift_detection.detectors.model_performance_detector import (
        EnsembleDetector, ModelEvalDetector,
    )
except Exception as exc:                             # noqa: BLE001 — probed
    EnsembleDetector = ModelEvalDetector = None
    _MP_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _MP_ERROR = None


def show(value) -> str:
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, float):
        return repr(round(value, 6))
    if isinstance(value, LearningRegime):
        return value.value if hasattr(value, "value") else str(value)
    if isinstance(value, DriftSignal):
        return json.dumps({
            "detected": bool(value.drift_detected),
            "score": round(float(value.drift_score), 6),
            "regime": show(value.regime),
        }, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps({k: show(v) for k, v in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return json.dumps([show(v) for v in value])
    return repr(value)


def attempt(fn, *args, **kwargs) -> str:
    try:
        return show(fn(*args, **kwargs))
    except BaseException as exc:                     # noqa: BLE001 — probed
        return show(exc)


def stream(n: int = 400):
    """One upward regime change, with a repeatable wobble."""
    return [(0.20 if i < 200 else 0.80)
            + 0.03 * math.sin(i * 0.7) + 0.01 * math.cos(i * 2.3)
            for i in range(n)]


def trace(detector, values) -> str:
    """The whole verdict sequence, not just whether it ever fired."""
    rows = []
    for i, value in enumerate(values):
        signal = detector.update(value)
        rows.append(f"{i}:{int(bool(signal.drift_detected))}"
                    f":{round(float(signal.drift_score), 4)}"
                    f":{show(signal.regime)}")
    fired = [r for r in rows if r.split(":")[1] == "1"]
    # sha256, not the builtin hash(): PYTHONHASHSEED is randomised per
    # process, so a builtin hash of the same trace differs between the two
    # subprocesses this probe runs. The first version used it and reported 30
    # divergences in which n, fired and first_fired all matched and only the
    # digest moved -- noise wearing a precise-looking number.
    digest = hashlib.sha256("|".join(rows).encode()).hexdigest()[:16]
    return json.dumps({"n": len(rows), "fired": len(fired),
                       "first_fired": fired[0] if fired else None,
                       "digest": digest})


def main() -> int:
    out = {}
    values = stream()

    # --- 1. the full verdict trace of each detector -----------------------
    # Not "did it fire" but the whole sequence, reduced to a digest plus the
    # first firing. A wrapper that computes drift_score differently shows up
    # here even when the underlying detector agrees.
    for name, make in (
        ("ADWIN", lambda: ADWINDetector(delta=0.002, minor_threshold=0.3,
                                        moderate_threshold=0.6)),
        ("PageHinkley", lambda: PageHinkleyDetector(
            min_instances=30, delta=0.005, threshold=50.0, alpha=0.9999)),
        ("KSWIN", lambda: KSWINDetector(alpha=0.005, window_size=100,
                                        stat_size=30, seed=7)),
    ):
        out[f"trace[{name}]"] = attempt(lambda m=make: trace(m(), values))

    # --- 2. where the regime thresholds cut -------------------------------
    # A22: the regime can outvote the verdict.
    #
    # The thresholds below include the EXACT values drift_score takes on this
    # stream -- 0.0, 0.01 and 0.02 -- because a comparison's boundary is only
    # observable when a score lands on it. The first version of this sweep
    # used round numbers (0.3/0.6, 1.0, ...) that no score ever reaches, and
    # a mutant that changed every `<` to `<=` in the regime ladder was not
    # detected at all. Same failure as the formatter decades in A38: a probe
    # that straddles no boundary cannot see one move.
    SCORES = (0.0, 0.01, 0.02)
    boundaries = [(s, s) for s in SCORES]
    boundaries += [(s, s + 0.01) for s in SCORES]
    boundaries += [(0.005, 0.015), (0.01, 0.02), (0.02, 0.01)]
    for minor, moderate in (((0.3, 0.6), (0.0, 0.0), (1.0, 1.0),
                             (0.6, 0.3), (-1.0, 2.0)) + tuple(boundaries)):
        out[f"regime[ADWIN,{minor},{moderate}]"] = attempt(
            lambda a=minor, b=moderate: trace(
                ADWINDetector(delta=0.002, minor_threshold=a,
                              moderate_threshold=b), values))
        out[f"regime[PageHinkley,{minor},{moderate}]"] = attempt(
            lambda a=minor, b=moderate: trace(
                PageHinkleyDetector(minor_threshold=a,
                                    moderate_threshold=b), values))

    # --- 3. KSWIN's seeding, which is the one stochastic surface ----------
    out["KSWIN[same-seed-twice]"] = attempt(
        lambda: trace(KSWINDetector(alpha=0.005, window_size=100,
                                    stat_size=30, seed=11), values)
        == trace(KSWINDetector(alpha=0.005, window_size=100,
                               stat_size=30, seed=11), values))
    out["KSWIN[different-seeds-differ]"] = attempt(
        lambda: trace(KSWINDetector(alpha=0.005, window_size=100,
                                    stat_size=30, seed=1), values)
        != trace(KSWINDetector(alpha=0.005, window_size=100,
                               stat_size=30, seed=2), values))
    out["KSWIN[no-seed-constructs]"] = attempt(
        lambda: bool(KSWINDetector(alpha=0.005, window_size=100,
                                   stat_size=30)))

    # --- 4. degenerate streams --------------------------------------------
    for name, data in (("empty", []), ("single", [0.5]),
                       ("constant", [0.5] * 100),
                       ("nan", [0.5] * 50 + [float("nan")] + [0.5] * 50),
                       ("inf", [0.5] * 50 + [float("inf")] + [0.5] * 50),
                       ("negative", [-x for x in stream(120)])):
        out[f"ADWIN[{name}]"] = attempt(
            lambda d=data: trace(ADWINDetector(delta=0.002), d))
        out[f"PageHinkley[{name}]"] = attempt(
            lambda d=data: trace(PageHinkleyDetector(), d))

    # --- 5. the ensemble's voting table -----------------------------------
    if EnsembleDetector is None:
        out["ensemble[import]"] = _MP_ERROR
    else:
        out["ensemble[VOTING_STRATEGIES]"] = attempt(
            lambda: sorted(EnsembleDetector.VOTING_STRATEGIES))

        def built(detectors, voting):
            subs = []
            for which in detectors:
                if which == "adwin":
                    subs.append(ADWINDetector(delta=0.002))
                elif which == "ph":
                    subs.append(PageHinkleyDetector())
                else:
                    subs.append(KSWINDetector(alpha=0.005, window_size=100,
                                              stat_size=30, seed=3))
            return EnsembleDetector(detectors=subs, voting=voting)

        # every documented name and alias, plus the ones that must refuse
        for voting in ("majority", "any", "or", "unanimous", "all", "and",
                       "MAJORITY", "Any", " majority ", "", "plurality",
                       "veto", None, 1):
            out[f"ensemble[voting={voting!r}]"] = attempt(
                lambda v=voting: trace(built(("adwin", "ph"), v), values))
        out["ensemble[empty-detectors]"] = attempt(
            lambda: built((), "majority"))
        out["ensemble[single-detector]"] = attempt(
            lambda: trace(built(("adwin",), "majority"), values))
        out["ensemble[three-detectors]"] = attempt(
            lambda: trace(built(("adwin", "ph", "kswin"), "majority"), values))
        # majority of two: does a 1-of-2 split count as a majority?
        out["ensemble[two-detectors-majority]"] = attempt(
            lambda: trace(built(("adwin", "ph"), "majority"), values))

    # --- 6. ModelEvalDetector, which A21 says prints and fabricates -------
    if ModelEvalDetector is None:
        out["eval_detector[import]"] = _MP_ERROR
    else:
        out["eval_detector[construct]"] = attempt(
            lambda: bool(ModelEvalDetector(metric_index=0)))
        for name, data in (("rising", [0.1, 0.2, 0.3, 0.4]),
                           ("falling", [0.9, 0.7, 0.4, 0.1]),
                           ("flat", [0.5, 0.5, 0.5, 0.5]),
                           ("single", [0.5])):
            out[f"eval_detector[{name}]"] = attempt(
                lambda d=data: trace(ModelEvalDetector(metric_index=0), d))
        out["eval_detector[bad-metric-index]"] = attempt(
            lambda: trace(ModelEvalDetector(metric_index=99), [0.1, 0.2]))

    # --- 7. the DriftSignal contract itself -------------------------------
    out["DriftSignal[fields]"] = attempt(
        lambda: sorted(DriftSignal.__dataclass_fields__))
    out["LearningRegime[members]"] = attempt(
        lambda: [m.name for m in LearningRegime])
    out["LearningRegime[values]"] = attempt(
        lambda: [show(m) for m in LearningRegime])

    json.dump(out, sys.stdout, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
