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
`__init__`. The oracle tree is assembled the same way: every file the claim's
recipe declares as a `produce` step is refilled from the repository (from
`src/` for package modules and the runnable entrypoints, from the repository
root for `examples/`), and every file the recipe PINS as an input is copied
from the claim itself, because a pinned input is identical on both sides by
definition. So the two trees differ in their implementation and in nothing
else. A generated file the source does not have is reported rather than
papered over with the subject's own bytes.

The file list comes from the recipe rather than from globbing because the
recipe is the authority on what the claim treats as regrowable. Globbing
`apeiron/**` silently missed the entrypoints layer's three runnable scripts,
which sit at the claim root and are that layer's entire content.

ALL TEN LAYERS are probed. A layer is driven through the module it owns; the
lower layers it carries are probed in their own right rather than twice.
`main.py` is the one piece deliberately not driven end to end -- it calls
`get_example(cfg)`, which needs MNIST or CIFAR on disk -- so its import
surface and argument handling are probed and the loop it runs is probed in
the monitor layer.

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
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBES = pathlib.Path(__file__).resolve().parent / "probes"

#: Layers with a probe, in dependency order. All ten of the campaign's
#: layers are covered; a layer named here is driven through its own module,
#: with the lower layers it carries already probed in their own right.
LAYERS = ("config", "evaluation", "profilers", "logger", "model_harness",
          "updaters", "drift_detection", "trainer", "monitor",
          "entrypoints")


def _claim_dir(layer: str) -> pathlib.Path:
    d = ROOT / "claims" / layer
    if not (d / "reticuli.toml").is_file():
        sys.exit(f"no claim at {d}")
    return d


def _source_of(rel: str) -> pathlib.Path:
    """Where a claim's generated file came from in the repository.

    Package modules live under `src/`; so do the runnable entrypoints
    (`main.py`, `drift_only.py`, `cl_only.py`). The example harnesses live at
    the repository root instead, outside `src/`, which is why this is a
    mapping rather than one prefix.
    """
    if rel.startswith("examples/"):
        return ROOT / rel
    return ROOT / "src" / rel


def _oracle_tree(layer: str, into: pathlib.Path) -> list[str]:
    """Refill the claim's GENERATED files from the repository source.

    The file list comes from the recipe's `produce` steps rather than from
    globbing, because that is precisely the set the claim treats as
    regrowable -- everything else in the claim directory is pinned criteria,
    which must be identical on both sides and is simply not copied. Globbing
    `apeiron/**` missed the entrypoints layer's three runnable scripts, which
    sit at the claim root and are the whole content of that layer.
    """
    claim = _claim_dir(layer)
    recipe = tomllib.loads((claim / "reticuli.toml").read_text())
    generated = [step["output"] for step in recipe.get("step", [])
                 if step.get("kind") == "produce"]

    missing = []
    for rel in sorted(generated):
        if not rel.endswith(".py"):
            continue
        origin = _source_of(rel)
        target = into / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_file():
            shutil.copy2(origin, target)
        else:
            # The claim declares a generated file the source does not have --
            # usually a producer's own helper split. Recorded, and left out
            # of the oracle so an import error is visible rather than papered
            # over with the subject's own bytes.
            missing.append(rel)

    # The claim's PINNED INPUTS come from the claim itself, not the source.
    # They are identical on both sides by definition -- that is what pinned
    # means -- so copying them keeps the comparison honest while making the
    # oracle importable. The profilers layer pins `aten_flops_map.py` as data
    # rather than generating it, and without this the oracle tree cannot
    # import its own profiler at all.
    for rel in sorted(recipe.get("claim", {}).get("inputs", [])):
        origin = claim / rel
        if not origin.is_file():
            continue
        target = into / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(origin, target)
    return missing


def _ensure_importable(root: pathlib.Path) -> bool:
    """Make `apeiron` a REGULAR package at `root`, and say whether we had to.

    A directory without `__init__.py` is only a namespace *portion*: Python
    records it and keeps scanning `sys.path`, and the first regular package
    it finds afterwards wins outright. This repository installs itself into
    the venv -- `site-packages/apeiron.pth` adds `src/` permanently -- so a
    room whose tree is `apeiron/profilers/` and nothing else loses to
    `src/apeiron/__init__.py` no matter what PYTHONPATH says.

    That is not a hypothetical either. It silently turned a profilers
    comparison into source-against-source and reported 269 of 269 cases
    agreeing, which is A40's error for the third time: the probe ran, every
    case passed, and nothing was compared. The monitor rooms were unaffected
    only because their claim happens to generate `apeiron/__init__.py`, so
    they arrive as regular packages.

    An empty `__init__.py` is written into BOTH trees when absent from
    both, which is faithful: the claim's own gate runs `PYTHONPATH=.` inside
    a fresh environment where nothing else provides `apeiron`, so a
    namespace portion is all it ever needs there.
    """
    pkg = root / "apeiron"
    init = pkg / "__init__.py"
    if not pkg.is_dir() or init.exists():
        return False
    init.write_text("")
    return True


def _assert_gate_passed(layer: str, subject: pathlib.Path) -> None:
    """A room that failed its gate is not a rebuild, and must not be probed.

    `ret rebuild` leaves the workspace on disk even when the producer never
    satisfies the criteria, so `--subject <room>` will happily measure code
    that does not reach the claim's root. That measurement means nothing:
    the whole premise is "an implementation that passes the gate still
    behaves differently", and this one does not pass.

    It is not hypothetical. Strengthening the drift_detection criteria made
    the producer fail outright -- `codex finished but the gate never
    produced DRIFT_OK` -- and the probe went on to report 4 contract
    divergences for the abandoned workspace as though it were a result.

    Only applied when the subject actually looks like a claim room (it
    carries a `reticuli.toml`). Mutant trees built from `_oracle_tree` have
    no recipe and no gate, and are legitimate subjects.
    """
    if not (subject / "reticuli.toml").is_file():
        return
    recipe = tomllib.loads((_claim_dir(layer) / "reticuli.toml").read_text())
    sentinels = [step["output"] for step in recipe.get("step", [])
                 if step.get("kind") == "gate"]
    missing = [s for s in sentinels if not (subject / s).exists()]
    if missing:
        sys.exit(
            f"REFUSING to probe {subject}: it is a claim room whose gate "
            f"never passed -- {', '.join(missing)} absent. A room that does "
            f"not reach the claim's root is not a rebuild, and comparing it "
            f"would describe a workspace the producer abandoned. Re-run the "
            f"rebuild (regrowth is a sampling instrument; retry before "
            f"concluding) or probe a room that passed.")


def _assert_resolves(root: pathlib.Path, python: str) -> None:
    """Refuse to compare a tree the interpreter will not actually import.

    The guard the profilers false green needed. A clean result is evidence
    only if the thing under test was the thing loaded.
    """
    snippet = ("import json, apeiron; "
               "print(json.dumps({'file': getattr(apeiron, '__file__', None), "
               "'path': [str(p) for p in list(apeiron.__path__)]}))")
    env = dict(os.environ, PYTHONPATH=str(root))
    proc = subprocess.run([python, "-c", snippet], capture_output=True,
                          text=True, env=env, check=False)
    if proc.returncode != 0:
        sys.exit(f"the tree at {root} does not import `apeiron` at all:\n"
                 f"{proc.stderr.strip()[-1500:]}")
    info = json.loads(proc.stdout)
    where = info["file"] or (info["path"][0] if info["path"] else None)
    if not where:
        sys.exit(f"`apeiron` imported from {root} reports no location")
    resolved = pathlib.Path(where).resolve()
    if not str(resolved).startswith(str(root.resolve())):
        sys.exit(
            f"REFUSING to compare: with PYTHONPATH={root}, `apeiron` "
            f"resolves to {resolved}, which is outside that tree. Any "
            f"result would describe whatever was actually imported -- the "
            f"same vacuity as comparing a thing against itself. Usually an "
            f"editable install (site-packages/apeiron.pth) winning with a "
            f"regular package against a namespace portion.")


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
    if subject_dir:
        _assert_gate_passed(layer, subject_root)

    with tempfile.TemporaryDirectory() as tmp:
        oracle_root = pathlib.Path(tmp) / "oracle"
        oracle_root.mkdir()
        absent = _oracle_tree(layer, oracle_root)

        # The subject is copied rather than imported in place: the guard
        # below may need to add a package marker, and a room is evidence --
        # it does not get written to.
        staged = pathlib.Path(tmp) / "subject"
        shutil.copytree(subject_root, staged,
                        ignore=shutil.ignore_patterns("__pycache__"))

        # Scaffolding only when BOTH sides need it, so the two trees stay
        # comparable. One side needing it and the other not is a real
        # difference in what the claim produces, and is left to surface.
        scaffolded = False
        if not (oracle_root / "apeiron" / "__init__.py").exists() \
                and not (staged / "apeiron" / "__init__.py").exists():
            scaffolded = _ensure_importable(oracle_root)
            scaffolded = _ensure_importable(staged) or scaffolded

        _assert_resolves(oracle_root, python)
        _assert_resolves(staged, python)

        oracle, oracle_unstable = _run(driver, oracle_root, python)
        subject, subject_unstable = _run(driver, staged, python)

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
    granted = LATITUDE.get(layer, ())
    declared = [d for d in diverge if granted and d[0].startswith(granted)]
    prose = [d for d in diverge if d not in declared
             and prose_only(d[1], d[2])]
    free = declared + prose
    contract = [d for d in diverge if d not in free]
    return {"layer": layer, "cases": len(cases), "agree": len(agree),
            "diverge": diverge, "causes": causes(contract),
            "contract_diverge": len(contract),
            "latitude_diverge": len(declared),
            "prose_diverge": len(prose),
            "latitude_surface": list(granted),
            "absent_from_source": absent, "unstable": unstable,
            "scaffolded_package_marker": scaffolded}


#: Case prefixes whose divergences are GRANTED LATITUDE, per layer: surface
#: where two implementations are both correct and the specification does not
#: care. Reported separately rather than dropped -- a harness that silently
#: excluded them would read as "covered everything" when it did not.
#:
#: `profilers` is the whole reason this exists. Its probe reports 117
#: divergences for one rebuild and 111 for another, and about 95% of both
#: are the three formatters turning a float into a display string:
#: `'1 FLOPs'` against `'1.00 FLOPs'`, `'1.0000 s'` against `'1.00 s'`,
#: across a sweep of every decade. Nothing in `src/` parses that output, so
#: those are two defensible spellings of the same number and pinning them
#: would buy criteria mass and no protection. Left in the probe because the
#: divergence is worth seeing; kept out of the contract count because
#: ranking work by it sends you to fix prose.
#: `_add_profile` is here on a different ground, and the distinction is the
#: one that mattered for `monitor`: a leading underscore is not by itself
#: evidence either way. `grep` across `src/`, `examples/` and `tests/` finds
#: no caller of `_add_profile` outside `count_flops.py`, so it is private in
#: fact and a reconstruction that keeps its bookkeeping elsewhere is free to.
#: `_aggregate_buffer` looked identical and was NOT free -- `cl_only.py:326`
#: calls it -- so each private name has to be checked rather than assumed.
#: `ConsoleLogger._get_step` is the same case as `_add_profile`, checked the
#: same way: no caller outside `console_logger.py` anywhere in `src/` or
#: `examples/`. `Logger.__init__` hands the console logger a `get_step`
#: CALLABLE (`logger.py:48`), and how the console logger then stores and
#: reads it is its own business -- the regrown one keeps no method by that
#: name and reads the callable directly.
#: drift_detection, both checked the same way:
#:
#: `DriftSignal[fields]` asks for `__dataclass_fields__`. The source's
#: `DriftSignal` is a plain class with an `__init__` (`base.py:29`), not a
#: dataclass, so the case reads the ABSENCE of dataclass machinery as the
#: expected answer. Nothing in `src/` or `examples/` calls `asdict`,
#: `fields` or `replace` on it, so whether a reconstruction reaches the
#: same five attributes via `@dataclass` is its own business.
#:
#: `ensemble[VOTING_STRATEGIES]` asks for a class attribute used only
#: inside `model_performance_detector.py` itself -- for validation at
#: :357 and lookup at :364, with no reader anywhere else. The SET of valid
#: strategy names is contract and is asserted by the behaviour cases; the
#: table's existence as a class attribute is not.
LATITUDE = {
    "profilers": ("_format_flops", "_format_time", "_format_throughput",
                  "_add_profile"),
    "logger": ("ConsoleLogger[step-callback-default]",),
    "drift_detection": ("DriftSignal[fields]",
                        "ensemble[VOTING_STRATEGIES]"),
}


#: An exception's TYPE is contract; its wording is prose.
_EXC = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning)): ")


def prose_only(want: str, got: str) -> bool:
    """True when both sides raise the same exception and differ in wording.

    This generalises what a per-layer prefix list was doing by hand. Both
    implementations reject the same input with the same exception type, and
    the strings differ: `ValueError: Invalid stage 'EVAL'. Must be one
    of: ('eval', 'drift', 'cl')` against `ValueError: Invalid stage 'EVAL';
    expected one of ['eval', 'drift', 'cl']`. The decision, the type and
    the valid set all agree; only the sentence differs, and a suite that
    pinned the sentence would be a worse suite.

    Sixteen of the logger's divergences and both of the monitor's residual
    two are this and nothing else. Detected rather than listed, so a new
    layer gets the classification without anyone remembering to add it.

    One side raising where the other does not is NOT prose -- that is a
    behavioural difference and stays on the contract count.
    """
    mine, theirs = _EXC.match(str(want)), _EXC.match(str(got))
    return bool(mine and theirs and mine.group(1) == theirs.group(1)
                and str(want) != str(got))


def _cause(got: str) -> str:
    """The shared root of a divergence, when one is legible from the value.

    A single absent name produces as many divergences as there are cases
    that touch it, and counting those as independent evidence badly
    overstates how far apart two implementations are: `monitor` reported 68
    divergences over 91 cases in both producer families, and 67 of them
    were four names the class did not have. Ranking work by raw case count
    therefore sends you to the widest probe rather than the widest
    behavioural gap. Cluster first, then rank by cause.
    """
    missing = re.search(r"has no attribute '([^']+)'", got)
    if missing:
        return f"no attribute {missing.group(1)!r}"
    exc = re.match(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning)): ",
                   got)
    if exc:
        return f"raises {exc.group(1)}"
    return "value differs"


def causes(diverge: list) -> dict:
    """Divergences grouped by shared root cause, widest group first."""
    grouped: dict = {}
    for case, _want, got in diverge:
        grouped.setdefault(_cause(str(got)), []).append(case)
    return dict(sorted(grouped.items(), key=lambda kv: -len(kv[1])))


def report(result: dict) -> None:
    layer, n = result["layer"], result["cases"]
    bad = result["diverge"]
    # `or` would be wrong here: an empty dict is falsy, so a room with
    # ZERO contract divergences fell back to grouping every divergence
    # and printed "0 diverge on contract from 1 distinct cause(s)",
    # listing a cause it had just excluded.
    grouped = result["causes"] if "causes" in result else causes(bad)
    free = result.get("latitude_diverge", 0)
    held = result.get("contract_diverge", len(bad))
    print(f"\n=== {layer}: {n} probe cases, "
          f"{result['agree']} agree, {held} diverge on contract "
          f"from {len(grouped)} distinct cause(s) ===")
    if free:
        print(f"  plus {free} divergence(s) in granted-latitude surface "
              f"({', '.join(result.get('latitude_surface', []))}), "
              f"reported and not counted as contract")
    if result.get("prose_diverge"):
        print(f"  plus {result['prose_diverge']} divergence(s) where both "
              f"sides raise the same exception and only the wording "
              f"differs, likewise not counted as contract")
    for cause, cases_ in grouped.items():
        print(f"  CAUSE  {cause}  ({len(cases_)} case(s), "
              f"e.g. {cases_[0]})")
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
        distinct = sum(len(r.get("causes", {})) for r in results)
        print(f"\n{total} divergence(s) from {distinct} distinct cause(s), "
              f"and {shaky} unstable case(s), over {cases} probe cases "
              f"in {len(results)} layer(s)")
    return 1 if any(r["diverge"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
