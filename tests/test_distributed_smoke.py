"""Opt-in real multi-process test of the distributed path (CPU / gloo).

Marked ``slow`` so the default ``-m "not slow"`` suite skips it (it spawns two
subprocesses). Run explicitly with::

    poetry run pytest -m slow tests/test_distributed_smoke.py

Launches tests/dist_smoke.py under ``torch.distributed.run`` with 2 processes and
asserts the in-script checks (shards cover the window; params stay in sync across
ranks after data-parallel adaptation) all passed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_two_process_gloo(tmp_path):
    script = Path(__file__).parent / "dist_smoke.py"
    env = dict(
        os.environ,
        DIST_SMOKE_STORE=str(tmp_path / "store"),
        OMP_NUM_THREADS="1",
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--master_port=29527",
        str(script),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "DIST SMOKE OK" in proc.stdout, proc.stdout
