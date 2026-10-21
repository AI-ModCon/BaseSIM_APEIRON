"""Making a run a function of its config.

``Config.seed`` is only a promise; this module is where it is kept. Every
entry point applies :func:`seed_everything` before anything draws randomness,
so two runs of one config produce the same stream of decisions and the same
metrics file. Components that need their own stream derive it with
:func:`stable_seed` instead of seed arithmetic.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def stable_seed(*parts: object) -> int:
    """A seed derived by hashing, not by arithmetic.

    ``seed + window`` collides: window 2 of seed 40 shares a stream with
    window 1 of seed 41. Hashing the parts does not, and unlike ``hash()``
    the result is the same in every process.
    """
    text = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def seed_everything(seed: int) -> None:
    """Seed every generator the run draws from."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
