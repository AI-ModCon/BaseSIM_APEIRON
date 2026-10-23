#!/usr/bin/env python3
"""Compare a regrown claim room against src/, and report what the tests let go.

A reconstruction that passes a claim's gate is, by construction, inside the
equivalence class the tests define. Everything it leaves out is therefore
unconstrained — so the diff between it and the original is a worklist of
things the suite does not demand.

Two measurements, because they answer different questions:

  size      how much code the criteria actually pin down, counting logical
            lines only (blanks, comments and docstrings excluded, since
            APEIRON is heavily docstringed and raw line counts flatter the
            result)
  surface   which public names exist upstream and not in the regrowth

Surface comparison resolves inheritance before reporting, so a method the
regrowth moved to a base class is not reported as missing. What survives is
capability the regrowth genuinely does without.

Usage:
    python3 claims/scripts/surface_diff.py [--json] <room> [<claim-name>]
    python3 claims/scripts/surface_diff.py /tmp/regrown/updaters updaters

--json emits one object instead of the table: the totals and per-file counts,
for a harness taking repeated draws of the same layer.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "src")


def own_outputs(claim: str) -> list[str]:
    """The files a claim regrows itself, not the ones its components supply."""
    recipe = os.path.join(REPO, "claims", claim, "reticuli.toml")
    text = open(recipe, encoding="utf-8").read()
    return re.findall(
        r'\[\[step\]\]\nkind = "produce"\noutput = "([^"]+)"\nclass = "generated"\n(?!from)',
        text,
    )


def code_lines(path: str) -> int | None:
    """Physical lines that are neither blank, comment, nor docstring."""
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return None
    docs: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docs.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return sum(
        1
        for i, line in enumerate(src.splitlines(), 1)
        if line.strip() and not line.strip().startswith("#") and i not in docs
    )


DYNAMIC = {"__getattr__", "__getattribute__"}


def _members(node: ast.ClassDef) -> set[str]:
    out = set()
    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(sub.name)
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            out.add(sub.target.id)
        elif isinstance(sub, ast.Assign):
            out.update(t.id for t in sub.targets if isinstance(t, ast.Name))
    return out


def surface(path: str) -> dict[str, set[str]] | None:
    """{"": module-level names, "ClassName": its members}, with bases folded in."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError):
        return None
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    out: dict[str, set[str]] = {"": set()}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[""].add(node.name)
        elif isinstance(node, ast.ClassDef):
            out[""].add(node.name)
            members = _members(node)
            # fold in locally-defined bases: a method the regrowth moved up
            # into a base class is still present on the subclass
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in classes:
                    members |= _members(classes[base.id])
            out[node.name] = members
        elif isinstance(node, ast.Assign):
            out[""].update(t.id for t in node.targets if isinstance(t, ast.Name))
    return out


def main(argv: list[str]) -> int:
    argv = list(argv)
    as_json = "--json" in argv
    if as_json:
        argv.remove("--json")
    if not 2 <= len(argv) <= 3:
        print(__doc__.strip().split("Usage:")[-1].strip(), file=sys.stderr)
        return 2
    room = argv[1].rstrip("/")
    claim = argv[2] if len(argv) == 3 else os.path.basename(room)

    try:
        files = own_outputs(claim)
    except OSError:
        print(f"no claim named {claim!r} under claims/", file=sys.stderr)
        return 1
    if not files:
        print(f"claim {claim!r} regrows nothing of its own", file=sys.stderr)
        return 1

    rows, dropped, total_a, total_b = [], [], 0, 0
    for rel in files:
        a_path, b_path = os.path.join(SRC, rel), os.path.join(room, rel)
        a, b = code_lines(a_path), code_lines(b_path)
        if a is None or b is None or a == 0:
            continue
        rows.append((rel, a, b))
        total_a += a
        total_b += b

        sa, sb = surface(a_path), surface(b_path)
        if not sa or not sb:
            continue
        for scope, names in sa.items():
            missing = sorted(
                n for n in names - sb.get(scope, set()) if not n.startswith("_")
            )
            if not missing:
                continue
            where = f"{rel}::{scope}" if scope else rel
            # a class that delegates through __getattr__ still answers to
            # every name at runtime; report the choice, not 18 phantom gaps
            if scope and DYNAMIC & sb.get(scope, set()):
                dropped.append(
                    (
                        where,
                        [
                            f"(delegates {len(missing)} names dynamically "
                            f"through __getattr__ instead of defining them)"
                        ],
                    )
                )
            else:
                dropped.append((where, missing))

    if not rows:
        print(f"nothing comparable in {room}", file=sys.stderr)
        return 1

    if as_json:
        # one object per room, for harnesses that take many draws of the same
        # layer and need the numbers rather than the table
        json.dump({"claim": claim, "room": room,
                   "src_lines": total_a, "regrown_lines": total_b,
                   "kept": round(total_b / total_a, 4),
                   "files": [{"path": rel, "src": a, "regrown": b}
                             for rel, a, b in sorted(rows, key=lambda r: -r[1])],
                   "dropped_public_names": sum(len(m) for _, m in dropped)},
                  sys.stdout)
        print()
        return 0

    width = max(len(r[0]) for r in rows)
    print(f"claim {claim}  —  logical code lines\n")
    print(f"{'file':<{width}}  {'src':>5} {'regrown':>8} {'kept':>6}")
    print("-" * (width + 23))
    for rel, a, b in sorted(rows, key=lambda r: -r[1]):
        print(f"{rel:<{width}}  {a:>5} {b:>8} {b / a:>5.0%}")
    print("-" * (width + 23))
    print(f"{'TOTAL':<{width}}  {total_a:>5} {total_b:>8} {total_b / total_a:>5.0%}")

    print("\npublic names the regrowth does without")
    if not dropped:
        print("  (none — the criteria pin the whole public surface)")
    for where, missing in dropped:
        print(f"  {where}")
        for name in missing:
            print(f"      {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
