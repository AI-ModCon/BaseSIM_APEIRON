"""Acceptance criteria for ADWIN: river's observable behaviour, pinned.

Recorded from `river.drift.ADWIN` 0.23.0 at delta=0.002. ADWIN is
deterministic, so firing indices are a stable property and are asserted
exactly -- which matters here more than for any other detector, because
detection latency is the one place an independent reconstruction of ADWIN
was measured to diverge from river while agreeing on everything else:
three draws fired ten samples early and agreed with each other more
closely than with river (claims/LOG.md A36).

These criteria therefore deliberately EXCLUDE the implementation language
models converge on. Passing them requires river's actual windowing
behaviour, not an approximation that happens to notice the same change.
"""
import math
import sys

sys.path.insert(0, ".")
from adwin import ADWIN  # noqa: E402

REFERENCE = {"step": [223], "constant": [], "ramp": [159],
             "spike": [], "down_step": [223], "down_ramp": [159]}


def streams():
    w = lambda i: 0.03 * math.sin(i * 0.7) + 0.01 * math.cos(i * 2.3)
    return {
        "step": [(0.20 if i < 200 else 0.80) + w(i) for i in range(400)],
        "constant": [0.5] * 200,
        "ramp": [0.1 + 0.004 * i for i in range(250)],
        "spike": [0.3] * 100 + [9.0] + [0.3] * 99,
        "down_step": [(0.80 if i < 200 else 0.20) + w(i) for i in range(400)],
        "down_ramp": [0.9 - 0.004 * i for i in range(250)],
    }


def firings(values, **kw):
    d = ADWIN(delta=0.002, **kw)
    out = []
    for i, v in enumerate(values):
        d.update(v)
        if d.drift_detected:
            out.append(i)
    return out


def main():
    data = streams()
    for name, expected in REFERENCE.items():
        got = firings(data[name])
        assert got == expected, (
            f"{name}: river fires at {expected}, this implementation at {got}")
    bare = ADWIN()
    assert abs(bare.delta - 0.002) < 1e-12, "river's default delta is 0.002"
    assert bare.drift_detected is False
    d = ADWIN(delta=0.002)
    for v in data["constant"][:40]:
        d.update(v)
    assert d.width == 40, f"width {d.width}, expected 40 samples retained"
    with open("ADWIN_OK", "w", encoding="utf-8") as f:
        f.write("ok\n")
    print("adwin: river's behaviour reproduced on 6 streams")
    return 0


if __name__ == "__main__":
    sys.exit(main())
