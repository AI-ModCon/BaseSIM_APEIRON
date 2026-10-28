"""This layer must not borrow third-party modules through a sibling.

A layer's test suite runs it against the REAL implementations of the layers
beneath it, so a reconstruction may lean on any name those happen to expose
and still pass every gate. Compose two independently regrown layers and the
borrowed name is not there.

That is not hypothetical. Nine layers were regrown blind, every one passed
its own gate, and the assembled framework would not import:

    File "apeiron/model/torch_model_harness.py", line 8
      from apeiron.config.configuration import Config, torch
    ImportError: cannot import name 'torch' from 'apeiron.config.configuration'

The regrown harness reached `torch` through the config module's namespace.
The config module in `src/` does `import torch` at its top (for
`get_available_device`), so the name happens to be there and the harness's
own gate passed. The regrown CONFIG does not need torch at module scope and
does not import it, so the pair does not compose. The source harness has no
such dependency -- it does `import torch` directly
(`torch_model_harness.py:7`).

The rule this file pins: **a cross-layer import may take only names the
component is for.** `Config` from the config module is the contract;
`torch` through it is a coincidence of that file's imports. Checked
statically, over the layer's own source, so it holds without composing
anything and names the offending line when it fails.

Why static rather than "try importing it with a stub": a stub would have to
enumerate every component's surface and would then be a second, drifting
copy of it. Asking instead "does this import pull a MODULE out of a
sibling?" needs no such list, and a module is exactly what cannot be part
of a component's API contract -- `apeiron.config` is not the thing that
provides torch.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest

import apeiron.model.torch_model_harness as harness_module

#: The files this layer generates. Resolved from the imported module rather
#: than hard-coded, so the check follows the implementation wherever it puts
#: itself.
OWN_MODULES = (harness_module,)


def _cross_layer_imports(module: types.ModuleType):
    """Every `from apeiron.… import a, b` in the module, as (line, mod, names)."""
    source = pathlib.Path(module.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("apeiron."):
            yield node.lineno, node.module, [a.name for a in node.names]


@pytest.mark.parametrize("module", OWN_MODULES, ids=lambda m: m.__name__)
def test_no_cross_layer_import_pulls_a_module(module):
    """`from apeiron.config.configuration import Config, torch` is the shape
    this forbids. Resolving each imported name and rejecting the ones that
    are modules catches it without needing to know which third-party
    packages exist."""
    offences = []
    for line, source_module, names in _cross_layer_imports(module):
        imported = __import__(source_module, fromlist=names)
        for name in names:
            value = getattr(imported, name, None)
            if isinstance(value, types.ModuleType):
                offences.append(
                    f"{pathlib.Path(module.__file__).name}:{line} takes the "
                    f"MODULE `{name}` out of `{source_module}` -- import it "
                    f"directly instead; a sibling layer is not what provides it"
                )
    assert not offences, "\n".join(offences)


@pytest.mark.parametrize("module", OWN_MODULES, ids=lambda m: m.__name__)
def test_a_module_that_uses_torch_imports_it_itself(module):
    """The positive half, conditional on actually needing it.

    The first version of this asserted that EVERY module imports torch, and
    it failed against `src/` -- `create_updater.py` is a factory and never
    touches torch. A criterion has to be measured against the
    implementation it was written from, not from what sounds right.
    """
    source = pathlib.Path(module.__file__).read_text()
    tree = ast.parse(source)
    uses_torch = any(
        isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        for node in ast.walk(tree)
    ) or "torch." in source
    if not uses_torch:
        pytest.skip(f"{pathlib.Path(module.__file__).name} does not use torch")
    direct = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torch" in direct, (
        f"{pathlib.Path(module.__file__).name} uses torch and must import it "
        "itself, not receive it from a sibling module's namespace"
    )
