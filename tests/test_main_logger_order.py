"""``main()`` must configure the logger before it builds the model harness.

``get_logger()`` returns any existing instance without applying its arguments, so
whichever caller reaches it first fixes the run's configuration. A harness that
logs from ``__init__`` -- one announcing its resolved settings, say -- would
otherwise pin the default backend and a null CSV path, and ``visualization.input``
would be silently dropped for the entire run: no metrics file, and no error.

Asserted on the source rather than by running ``main()``: importing it pulls in
``examples.utils``, which is not on the path for the test run, and the property
under test is purely the statement order inside the function.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MAIN_PY = Path(__file__).resolve().parents[1] / "src" / "main.py"


def _call_names(node: ast.AST) -> set[str]:
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


@pytest.fixture(scope="module")
def main_body() -> list[ast.stmt]:
    tree = ast.parse(MAIN_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node.body
    pytest.fail("no main() found in src/main.py")


def _index_of_call(body: list[ast.stmt], name: str) -> int:
    for i, stmt in enumerate(body):
        if name in _call_names(stmt):
            return i
    pytest.fail(f"main() never calls {name}()")


def test_logger_is_configured_before_the_harness_is_built(main_body):
    assert _index_of_call(main_body, "get_logger") < _index_of_call(
        main_body, "get_example"
    ), "the harness is built before the logger is configured"


def test_backend_is_resolved_before_the_logger(main_body):
    assert _index_of_call(main_body, "configure_backend") < _index_of_call(
        main_body, "get_logger"
    )


def test_main_does_not_need_to_reset_the_singleton(main_body):
    """Ordering makes the reset unnecessary; reintroducing one is a smell."""
    calls: set[str] = set()
    for stmt in main_body:
        calls |= _call_names(stmt)
    assert "reset_logger" not in calls
