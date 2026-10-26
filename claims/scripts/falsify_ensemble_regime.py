#!/usr/bin/env python3
"""Minimal falsification: EnsembleDetector returns a nondeterministic regime.

`EnsembleDetector.update` resolves the regime by majority vote over its
sub-detectors:

    regime_votes = [s.regime for s in signals]
    regime = max(set(regime_votes), key=regime_votes.count)

`max` over a SET has no defined order, so when the vote ties -- the ordinary
case for an even number of detectors, and the only case for the default pair
-- the winner is whichever member `set` iteration yields first. For Enum
members that is `hash(member._name_)`, a string hash, which Python randomises
per process unless PYTHONHASHSEED is fixed.

So the same detector fed the same stream reports a different REGIME from one
run to the next. The regime is what the framework acts on -- stable,
continual_learning, fine_tuning, retrain -- so an identical experiment can
take a different adaptation path on a rerun, with nothing in the logs
explaining why.

    python3 claims/scripts/falsify_ensemble_regime.py [path/to/src]

Exits 1 if the regime is stable across seeds (the defect is absent), 0 if it
flips (the defect reproduces). Measured on this checkout: over seeds 0-15 the
tie resolves 10 times to STABLE and 6 times to CONTINUAL_LEARNING.

The verdict and the drift score are NOT affected -- both are computed by
counting, not by set order -- which is why the test suite does not catch it:
a test asserting `drift_detected` passes every time.
"""
import os
import pathlib
import subprocess
import sys
import tempfile

SEEDS = [str(i) for i in range(16)]

CHILD = '''
import math, sys
sys.path.insert(0, sys.argv[1])
from apeiron.drift_detection.detectors.statistical_detectors import (
    ADWINDetector, PageHinkleyDetector)
from apeiron.drift_detection.detectors.model_performance_detector import (
    EnsembleDetector)

values = [(0.20 if i < 200 else 0.80) + 0.03 * math.sin(i * 0.7)
          + 0.01 * math.cos(i * 2.3) for i in range(400)]
ensemble = EnsembleDetector(
    detectors=[ADWINDetector(delta=0.002), PageHinkleyDetector()],
    voting="any")
regimes = []
for value in values:
    signal = ensemble.update(value)
    regimes.append(getattr(signal.regime, "value", str(signal.regime)))
# step 223 is where ADWIN fires and Page-Hinkley has not: a 1-1 regime tie
print(regimes[223])
'''


def _layer_tree(src: pathlib.Path, into: pathlib.Path) -> None:
    """The two subpackages needed, without the welded top-level __init__ (A1)."""
    for part in ("config", "drift_detection"):
        for path in (src / part).rglob("*.py"):
            target = into / "apeiron" / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def main(argv: list[str]) -> int:
    src = pathlib.Path(argv[0] if argv else "src/apeiron").resolve()
    if not (src / "drift_detection").is_dir():
        sys.exit(f"no apeiron source tree at {src}")

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _layer_tree(src, root / "tree")
        child = root / "child.py"
        child.write_text(CHILD)

        seen = {}
        for seed in SEEDS:
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run(
                [sys.executable, str(child), str(root / "tree")],
                capture_output=True, text=True, env=env)
            if proc.returncode != 0:
                sys.exit(f"the child failed (seed {seed}):\n"
                         f"{proc.stderr.strip()[-2000:]}")
            seen.setdefault(proc.stdout.strip(), []).append(seed)

    print("EnsembleDetector(voting='any'), two detectors, identical stream.")
    print(f"regime reported at step 223, over {len(SEEDS)} hash seeds:\n")
    for regime, seeds in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        print(f"  {regime:<22} {len(seeds):>2}/{len(SEEDS)} seeds  "
              f"(PYTHONHASHSEED={','.join(seeds)})")

    if len(seen) == 1:
        print("\nstable across every seed tried — the defect does not "
              "reproduce here")
        return 1
    print(f"\nREPRODUCED: {len(seen)} different regimes for one input. "
          f"`max(set(...))` has no tie-break, and Enum hashes are "
          f"per-process.\nA deterministic fix has to say which regime wins a "
          f"tie — most severe, or the one the verdict implies (see A22).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
