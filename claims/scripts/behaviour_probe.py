#!/usr/bin/env python3
"""Does a reconstruction BEHAVE like the source it replaced, off the pinned set?

The campaign's headline measurement is a kept-ratio: how much of a layer's
code comes back when a producer regrows it from the criteria alone (A33, 53%
over ten layers). That number cannot answer the question anyone actually
asks, and A37 is the proof: the Page-Hinkley reconstruction regrew at high
fidelity, passed its gate, and was independently redone three times with all
three agreeing -- and it was still a different algorithm, diverging from the
implementation it replaced on a whole family of inputs none of its criteria
visited. Lines-kept did not see it. A differential probe against an oracle
did, immediately.

This harness generalises that probe to the APEIRON layers. The oracle is the
source the claim was built from; the subject is the claim's regrown
implementation. Both are driven through inputs the criteria do not pin, and
the harness reports where their observable behaviour parts.

    python3 claims/scripts/behaviour_probe.py <layer> [--python PATH]
    python3 claims/scripts/behaviour_probe.py --all

A divergence is not automatically a defect. It is one of:

  criteria gap    the source's behaviour is the intended one and nothing
                  demanded it -- the finding is a missing test
  free choice     both behaviours are defensible and the specification
                  genuinely does not care -- the finding is that the claim
                  should say so
  source defect   the reconstruction is RIGHT and the source is wrong, which
                  has happened (A17, A25) and is the most valuable outcome

The harness classifies nothing. It reports, and the triage function in A34
decides.

HOW THE ORACLE IS BUILT, and why it is not just `src/`. `src/apeiron/
__init__.py` eagerly imports every layer (A1), so importing one layer from
the source tree drags in river, evidently, wandb and mlflow. The claims work
around this by carrying an implicit namespace package with no welded
`__init__`. The oracle tree here is built the same way -- the claim's own
`apeiron/**` layout, refilled with bytes from `src/apeiron/**` -- so the two
sides differ in their implementation and in nothing else. A file the claim
carries that the source does not have is reported rather than skipped.

CAVEAT on this workspace: `src/apeiron/` carries the campaign's own
"fixed here" patches for `evaluation.py` and `continuous_monitor.py`
(A19, A28), so for those two layers the oracle is the patched source, not
upstream `main`. That is the right oracle -- it is what the criteria were
written against -- but it means a divergence there is not a statement about
upstream.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBES = pathlib.Path(__file__).resolve().parent / "probes"

#: Layers with a probe. The ones left out need a model and data in the loop
#: (trainer, monitor, model_harness) or an external service (logger's
#: backends); a probe for those is a bigger instrument than this one and
#: pretending otherwise would report coverage we do not have.
LAYERS = ("config", "evaluation", "profilers", "drift_detection")


def _claim_dir(layer: str) -> pathlib.Path:
    d = ROOT / "claims" / layer
    if not (d / "reticuli.toml").is_file():
        sys.exit(f"no claim at {d}")
    return d


def _oracle_tree(layer: str, into: pathlib.Path) -> list[str]:
    """Mirror the claim's apeiron/** layout, filled from src/apeiron/**."""
    claim = _claim_dir(layer)
    subject = claim / "apeiron"
    source = ROOT / "src" / "apeiron"
    missing = []
    for path in sorted(subject.rglob("*.py")):
        rel = path.relative_to(subject)
        origin = source / rel
        target = into / "apeiron" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_file():
            shutil.copy2(origin, target)
        else:
            # The claim carries a file the source does not. Usually a
            # producer's own helper split; record it, and leave the oracle
            # without it so an import error is visible rather than papered
            # over with the subject's own bytes.
            missing.append(str(rel))
    return missing


#: Each side is run once per seed. Varying PYTHONHASHSEED varies the
#: iteration order of sets and dicts keyed by anything string-hashed --
#: including Enum members, whose `__hash__` hashes the member NAME. Code that
#: resolves a tie with `max(set(...), key=...)` therefore answers differently
#: from one process to the next, and comparing one run of each side reports
#: that as a divergence between the two implementations.
#:
#: This is not hypothetical: it is how `EnsembleDetector.update` was found to
#: return a nondeterministic `regime` on a tied vote, in code byte-identical
#: on both sides. A harness that cannot tell "unstable" from "different"
#: would have filed that as a reconstruction defect.
#:
#: The seed list is a DETECTOR, not a proof. Each seed fixes one hash
#: ordering, so a case that is unstable in reality looks stable if every seed
#: here happens to pick the same ordering -- which is exactly what the first
#: version of this list did: seeds 0, 1 and 2 all resolve the
#: EnsembleDetector tie to STABLE, and the instability only appears from seed
#: 5 onward. Measured across seeds 0-15 the split is 10 STABLE to 6
#: CONTINUAL_LEARNING, so these six seeds cover both orderings. Stability
#: reported here means "not caught", never "cannot happen".
SEEDS = ("0", "1", "2", "5", "11", "13")


def _run_once(driver: pathlib.Path, root: pathlib.Path, python: str,
              seed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTHONHASHSEED"] = seed
    env.pop("APP_SEED", None)                     # config reads APP_* overrides
    proc = subprocess.run([python, str(driver)], capture_output=True,
                          text=True, env=env, cwd=str(ROOT))
    if proc.returncode != 0:
        sys.exit(f"the probe failed under {root} (seed {seed}):\n"
                 f"{proc.stderr.strip()[-3000:]}")
    try:
        return json.loads(proc.stdout)
    except ValueError:
        sys.exit(f"the probe under {root} did not emit JSON:\n"
                 f"{proc.stdout.strip()[-2000:]}")


def _run(driver: pathlib.Path, root: pathlib.Path, python: str) -> tuple:
    """Every seed's answers, plus the cases that were not stable across them."""
    runs = [_run_once(driver, root, python, seed) for seed in SEEDS]
    unstable = {}
    for case in sorted(set().union(*(set(r) for r in runs))):
        seen = {r.get(case, "<absent>") for r in runs}
        if len(seen) > 1:
            unstable[case] = sorted(seen)
    return runs[0], unstable


def probe(layer: str, python: str, subject_dir: str | None = None) -> dict:
    driver = PROBES / f"{layer}.py"
    if not driver.is_file():
        sys.exit(f"no probe driver at {driver}")
    subject_root = pathlib.Path(subject_dir) if subject_dir else _claim_dir(layer)

    with tempfile.TemporaryDirectory() as tmp:
        oracle_root = pathlib.Path(tmp)
        absent = _oracle_tree(layer, oracle_root)
        oracle, oracle_unstable = _run(driver, oracle_root, python)
        subject, subject_unstable = _run(driver, subject_root, python)

    # A case that is not stable within a side cannot be compared across
    # sides; it is reported as nondeterminism in that implementation, which
    # is a finding of its own and usually a defect.
    unstable = {}
    for side, found in (("source", oracle_unstable),
                        ("reconstruction", subject_unstable)):
        for case, values in found.items():
            unstable.setdefault(case, {})[side] = values

    cases = sorted(set(oracle) | set(subject))
    agree, diverge = [], []
    for case in cases:
        if case in unstable:
            continue
        want, got = oracle.get(case, "<absent>"), subject.get(case, "<absent>")
        (agree if want == got else diverge).append((case, want, got))
    return {"layer": layer, "cases": len(cases), "agree": len(agree),
            "diverge": diverge, "absent_from_source": absent,
            "unstable": unstable}


def report(result: dict) -> None:
    layer, n = result["layer"], result["cases"]
    bad = result["diverge"]
    print(f"\n=== {layer}: {n} probe cases, "
          f"{result['agree']} agree, {len(bad)} diverge ===")
    if result.get("unstable"):
        print(f"  {len(result['unstable'])} case(s) are NOT STABLE across "
              f"PYTHONHASHSEED and so cannot be compared:")
        for case, sides in sorted(result["unstable"].items()):
            print(f"    UNSTABLE {case}  ({', '.join(sorted(sides))})")
            for side, values in sorted(sides.items()):
                for value in values:
                    print(f"      {side:<15} {str(value)[:120]}")
    if result["absent_from_source"]:
        print(f"  note: the claim carries files the source does not: "
              f"{result['absent_from_source']}")
    for case, want, got in bad:
        print(f"  DIVERGE  {case}")
        print(f"    source       {str(want)[:160]}")
        print(f"    reconstruction {str(got)[:160]}")
    if not bad:
        print("  the reconstruction matches the source on every probed case")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter with the layer's dependencies installed")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--subject", default=None,
                    help="import root to probe instead of the claim's — for "
                         "validating the probe against a known-wrong "
                         "implementation (a probe that cannot fail is not "
                         "evidence; see A37)")
    a = ap.parse_args(argv)

    layers = LAYERS if a.all else ([a.layer] if a.layer else None)
    if not layers:
        ap.error("name a layer or pass --all")

    if a.subject and len(layers) != 1:
        ap.error("--subject probes one layer")
    results = [probe(layer, a.python, a.subject) for layer in layers]
    if a.json:
        print(json.dumps(results, indent=1, default=str))
    else:
        for result in results:
            report(result)
        total = sum(len(r["diverge"]) for r in results)
        shaky = sum(len(r.get("unstable", {})) for r in results)
        cases = sum(r["cases"] for r in results)
        print(f"\n{total} divergence(s) and {shaky} unstable case(s) over "
              f"{cases} probe cases in {len(results)} layer(s)")
    return 1 if any(r["diverge"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
