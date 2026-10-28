"""What `apeiron.driver` re-exports, which is this layer's glue.

A package `__init__` is a contract BETWEEN layers, and this directory's
tests never asserted it. `test_public_api.py` pins the top-level
`apeiron.__all__`; nothing pinned the sub-packages that top level imports
FROM, so a reconstruction is free to leave a name out and still pass every
gate -- because its own gate runs against the real siblings, which have it.

Nine layers were regrown blind, every one passed, and assembling them
failed on exactly this:

    File "apeiron/__init__.py", line 7
      from apeiron.training.updater import BaseUpdater
    ImportError: cannot import name 'BaseUpdater' from
                 'apeiron.training.updater'

Three packages were short between them -- `training.updater` missing
`BaseUpdater`, `logger` missing `StageType` and `VALID_STAGES`, and
`drift_detection` missing all six detector classes. Each omission is
invisible to the layer that made it and fatal to the composition.

Pinned as the exact set, because "at least these" would let the surface
grow silently and a re-export is cheap to add and awkward to remove.
"""

from __future__ import annotations

import importlib
import types

import pytest

#: Measured from `src/`. Every name a sibling layer may import from here.
EXPECTED = {'ContinuousMonitor'}


@pytest.fixture(scope="module")
def package() -> types.ModuleType:
    return importlib.import_module("apeiron.driver")


def _public_names(module: types.ModuleType) -> set[str]:
    """Public, non-module attributes -- submodules are not the API."""
    return {
        name for name in dir(module)
        if not name.startswith("_")
        and not isinstance(getattr(module, name), types.ModuleType)
    }


def test_every_expected_name_is_importable(package):
    missing = sorted(EXPECTED - _public_names(package))
    assert not missing, (
        f"`apeiron.driver` does not re-export {missing}; a sibling layer importing "
        f"one of those cannot compose with this one"
    )


def test_the_surface_has_not_grown(package):
    """Not a style rule: a name exported here is one a sibling may come to
    depend on, and the next reconstruction then has to guess it too."""
    extra = sorted(_public_names(package) - EXPECTED)
    assert not extra, f"`apeiron.driver` re-exports unexpected names: {extra}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_name_is_the_right_kind(package, name):
    """A re-export has to be the object, not a placeholder."""
    value = getattr(package, name)
    assert value is not None
    assert not isinstance(value, types.ModuleType)
