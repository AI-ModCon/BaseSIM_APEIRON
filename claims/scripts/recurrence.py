#!/usr/bin/env python3
"""Count how many independent draws dropped the same thing.

R7 says a divergence that recurs across draws is a finding about the
BOUNDARY, while one that appears in a single draw is a finding about the
PRODUCER. That distinction was being made by eye. With several draws per
layer on disk it can be counted instead, which is the difference between
"the tests probably do not pin this" and "3 of 3 producers dropped it".

The unit is a public name, not a line: reformatting changes every line and
would drown the signal, while a name that is absent from the reconstruction
is absent from what the criteria demanded. Inheritance and __getattr__
delegation are folded in by surface_diff, so a method merely moved to a base
class does not count as dropped.

    python3 claims/scripts/recurrence.py                 # every layer
    python3 claims/scripts/recurrence.py logger updaters  # just these
    python3 claims/scripts/recurrence.py --json

Triage reading, per the five-kind classification:

    dropped in every draw   the criteria do not pin it -- a test (or a
                            deliberate deletion), and the strongest case
    dropped in some draws   partly pinned, or the name is reachable by more
                            than one route; read the diff before filing
    dropped in one draw      producer noise, not a boundary finding
"""

from __future__ import annotations

import ast

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import surface_diff as sd  # noqa: E402

ROOMS_ENV = "RECURRENCE_ROOMS"

#: Where the measured draws live, relative to the rooms directory. Taken from
#: the variance runs: one fresh room per draw, model pinned and recorded.
DRAWS = {
    "config": ["var2/config-1", "var2/config-2", "var2/config-3"],
    "evaluation": ["var2/evaluation-1", "var2/evaluation-2", "var2/evaluation-3"],
    "profilers": ["var2/profilers-1", "var2/profilers-2", "var2/profilers-3"],
    "model_harness": ["var3/model_harness-1", "var3/model_harness-2",
                      "var3/model_harness-3"],
    "updaters": ["var4/updaters-1", "var4/updaters-2", "var4/updaters-3"],
    "logger": ["var4/logger-1", "var4/logger-2"],
    "drift_detection": ["var4/drift_detection-2", "var5/drift_detection-1"],
    "trainer": ["var6/trainer-1"],
}


#: Behaviours worth counting that are NOT public names, so the name scan
#: above is blind to them: a dropped side effect, a vanished dependency, an
#: algorithm swapped out. Each probe is a regex over the layer's own source
#: in one draw; the report says in how many draws the behaviour survived.
#: These are the written-up observations, restated as something countable.
MARKERS = {
    "config": [
        ("writes resolved_config.json", r"resolved_config\.json"),
        ("imports torch at module scope", r"(?m)^import torch\b"),
    ],
    "evaluation": [
        ("@torch.no_grad on the metrics", r"no_grad"),
    ],
    "profilers": [
        ("uses FlopCounterMode", r"FlopCounterMode"),
        ("uses pandas", r"\bpandas\b|DataFrame"),
    ],
    "model_harness": [
        ("caps retained tasks at 50", r"max_task_records[^\n]*50"),
        ("raises on an empty loader", r"raise\s+RuntimeError"),
    ],
    "updaters": [
        ("cites the source papers", r"doi\.org|proceedings\.mlr\.press"),
        ("EWC allocates a per-step Fisher", r"_batch_fisher"),
    ],
    "logger": [
        ("uses pandas", r"\bpandas\b|pd\.DataFrame"),
        # anchored on the signature: a bare /prefix/ also matches
        # _flatten_config(prefix=''), which has nothing to do with it
        ("log() takes a prefix flag", r"def log\([^)]*prefix"),
        ("emits a per-stage step key", r"""['"]/step|\{self\._current_stage\}/step"""),
    ],
    "drift_detection": [
        ("wraps the river library", r"\briver\b"),
        ("votes on the regime", r"regime_votes"),
        ("uses numpy", r"\bnumpy\b|\bnp\."),
    ],
    "trainer": [
        ("shows a tqdm progress bar", r"\btqdm\b"),
        ("logs the cperf_ FLOPS payload", r"cperf_"),
        ("frees the CUDA cache", r"empty_cache"),
    ],
}


def marker_hits(paths: list[str], pattern: str) -> bool:
    import re
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                if re.search(pattern, f.read()):
                    return True
        except OSError:
            continue
    return False


def _defined(tree: ast.AST) -> set[str]:
    """Every name this module defines that a caller could reach.

    Deliberately flat and generous, because the question is "does the
    reconstruction have this at all", not "is it in the same place". It
    therefore counts module-level defs, classes and assignments, class
    members, AND attributes assigned to self inside any method -- the last
    because a property upstream is routinely a plain instance attribute in a
    reconstruction, which is not a dropped name.
    """
    out: set[str] = set()

    def self_attrs(fn):
        """Attributes the method hangs on self -- instance API."""
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for t in targets:
                flat = list(t.elts) if isinstance(t, (ast.Tuple, ast.List)) else [t]
                for e in flat:
                    if (isinstance(e, ast.Attribute)
                            and isinstance(e.value, ast.Name)
                            and e.value.id == "self"):
                        out.add(e.attr)

    def named(stmts, in_class):
        """Walk a statement list, counting definitions but NOT locals.

        A function body is entered only to collect `self.x` assignments; its
        local variables are not API and would bury the signal under names
        like `tmp` and `i`.
        """
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add(node.name)
                if in_class:
                    self_attrs(node)
            elif isinstance(node, ast.ClassDef):
                out.add(node.name)
                named(node.body, True)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    flat = list(t.elts) if isinstance(t, (ast.Tuple, ast.List)) else [t]
                    for e in flat:
                        if isinstance(e, ast.Name):
                            out.add(e.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    out.add(node.target.id)
            elif isinstance(node, (ast.If, ast.Try)):
                # definitions behind a version or import guard still count
                named(getattr(node, "body", []), in_class)
                named(getattr(node, "orelse", []), in_class)
                named(getattr(node, "finalbody", []), in_class)
                for handler in getattr(node, "handlers", []):
                    named(handler.body, in_class)

    named(getattr(tree, "body", []), False)
    return out


def names_of(paths: list[str]) -> set[str] | None:
    """Every name defined anywhere across one version of a layer.

    Scoped to the whole layer rather than one file, so a class or helper the
    reconstruction moved to a different module is not reported as missing --
    only something it never wrote at all.
    """
    out: set[str] = set()
    seen_any = False
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError, ValueError):
            continue
        seen_any = True
        out |= _defined(tree)
    return out if seen_any else None


def dynamic_anywhere(paths: list[str]) -> bool:
    """Does this version delegate unknown attributes at runtime?"""
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        if any(d in src for d in sd.DYNAMIC):
            return True
    return False


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    wanted = [a for a in argv[1:] if not a.startswith("--")] or list(DRAWS)
    rooms_dir = os.environ.get(ROOMS_ENV)
    if not rooms_dir:
        print(f"set {ROOMS_ENV} to the directory holding the draw rooms",
              file=sys.stderr)
        return 2

    report = []
    for layer in wanted:
        draws = DRAWS.get(layer)
        if not draws:
            print(f"no draws recorded for {layer!r}", file=sys.stderr)
            continue
        files = sd.own_outputs(layer)
        src_paths = [os.path.join(REPO, "src", rel) for rel in files]
        upstream = names_of(src_paths) or set()
        # which file each upstream name came from, for the report only
        origin = {}
        for rel in files:
            got = names_of([os.path.join(REPO, "src", rel)]) or set()
            for name in got:
                origin.setdefault(name, rel)

        counts: dict[str, int] = {}
        kept_markers: dict[str, int] = {}
        present = 0
        for draw in draws:
            room = os.path.join(rooms_dir, draw)
            if not os.path.isdir(room):
                continue
            paths = [os.path.join(room, rel) for rel in files
                     if os.path.isfile(os.path.join(room, rel))]
            theirs = names_of(paths)
            if theirs is None:
                continue
            present += 1
            for label, pattern in MARKERS.get(layer, []):
                if marker_hits(paths, pattern):
                    kept_markers[label] = kept_markers.get(label, 0) + 1
            if dynamic_anywhere(paths):
                # answers to every name at runtime; absence proves nothing
                continue
            for name in upstream:
                if name.startswith("_") or name in theirs:
                    continue
                counts[name] = counts.get(name, 0) + 1

        rows = [{"file": origin.get(name, "?").replace("apeiron/", ""),
                 "name": name, "draws": n, "of": present}
                for name, n in sorted(counts.items(),
                                      key=lambda kv: (-kv[1], kv[0]))]
        markers = [{"label": label, "kept": kept_markers.get(label, 0),
                    "of": present,
                    "upstream": marker_hits(src_paths, pattern)}
                   for label, pattern in MARKERS.get(layer, [])]
        report.append({"layer": layer, "draws": present, "dropped": rows,
                       "markers": markers})

    if as_json:
        json.dump(report, sys.stdout, indent=2)
        print()
        return 0

    for entry in report:
        unanimous = [r for r in entry["dropped"] if r["draws"] == entry["draws"]]
        partial = [r for r in entry["dropped"] if r["draws"] < entry["draws"]]
        print(f"\n{entry['layer']}  ({entry['draws']} draw(s))")
        print("-" * 66)
        if not entry["dropped"]:
            print("  every public name survived every draw")
        for r in unanimous:
            print(f"  {r['draws']}/{r['of']}  {r['file']:<42} {r['name']}")
        for r in partial:
            print(f"  {r['draws']}/{r['of']}  {r['file']:<42} {r['name']}"
                  f"   (producer noise?)")
        if unanimous:
            print(f"\n  {len(unanimous)} name(s) no draw reproduced -- the "
                  f"criteria do not pin them")
        live = [m for m in entry["markers"] if m["upstream"]]
        if live:
            print("\n  behaviours (not names, so the scan above cannot see them):")
            for m in live:
                verdict = ("every draw kept it" if m["kept"] == m["of"] and m["of"]
                           else "NO draw kept it" if m["kept"] == 0
                           else f"{m['kept']} of {m['of']} draws kept it")
                print(f"    {m['kept']}/{m['of']}  {m['label']:<38} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
