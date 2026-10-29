#!/usr/bin/env python3
"""List pinned criteria that constrain a module their claim does not own.

R33: a claim may be handed a component as `class = "generated"` with
`from = "<component>"`. The producer then writes its own copy of that
file, and the gate validates THAT copy -- which the assembly discards in
favour of the owning room's. A criterion asserting a property of a
supplied module therefore constrains a file that never ships.

Run from the repository root. Reports, per claim, every pinned criterion
whose imports reach a module the claim is supplied rather than owns, and
names the claim that does own it. Most hits are benign (importing
`Config` to build a fixture constrains nothing); the ones to act on are
criteria that ASSERT properties of the supplied module.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

IMPORT = re.compile(r"from (apeiron[\w.]*) import|import (apeiron[\w.]*)")


def main(root: pathlib.Path) -> int:
    claims = sorted(p for p in (root / "claims").iterdir()
                    if (p / "reticuli.toml").exists())
    owner: dict[str, list[str]] = {}
    supplied: dict[str, set[str]] = {}
    for claim in claims:
        spec = tomllib.loads((claim / "reticuli.toml").read_text())
        handed = set()
        for step in spec.get("step", []):
            out = step.get("output", "")
            if not out.endswith(".py"):
                continue
            if step.get("from"):
                handed.add(out)
            else:
                owner.setdefault(out, []).append(claim.name)
        supplied[claim.name] = handed

    findings = 0
    for claim in claims:
        spec = tomllib.loads((claim / "reticuli.toml").read_text())
        for name in spec["claim"].get("inputs", []):
            path = claim / name
            if not name.startswith("test_") or not path.exists():
                continue
            modules = {
                m.replace(".", "/") + ".py"
                for pair in IMPORT.findall(path.read_text())
                for m in pair if m
            }
            crossing = sorted(m for m in modules if m in supplied[claim.name])
            if not crossing:
                continue
            findings += 1
            print(f"{claim.name}/{name}")
            for module in crossing:
                owners = ", ".join(owner.get(module, ["(owned by no claim)"]))
                print(f"    constrains {module}  -> owned by {owners}")

    print(f"\n{findings} pinned criterion file(s) reach a module their claim "
          f"does not own.")
    print("A criterion that ASSERTS a property of one of those belongs in "
          "the owning claim (R33).")
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
