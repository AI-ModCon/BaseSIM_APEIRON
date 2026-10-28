"""The step columns the logger contributes to every row it records.

`log()` does not only pass metrics through. It adds two of its own
(`wandb_logger.py:122`):

    prefixed_metrics["step"] = current_step
    if self._current_stage:
        prefixed_metrics[f"{self._current_stage}/step"] = \\
            self._stage_steps[self._current_stage]

so every recorded row carries a global `step` and a per-stage
`{stage}/step`. Both reach the metrics file, and the MLflow backend reads
`step` back out of the history when it replays rows
(`mlflow_logger.py:240`) -- so this is a column other code consumes, not
decoration.

A rebuilt logger omitted both. The consequence only showed in a full run:
the assembled framework's MNIST file carried 32 metric series where the
original carries 31, and three of the differences were the missing
`cl/step`, `drift/step` and `eval/step`. Nothing in this directory's tests
looks at what `log()` ADDS -- they check that the metrics they passed come
back -- so the omission passed every gate on both sides.

Pinned through `to_dataframe()`, which is what the CSV is written from, so
the assertion is about the record rather than about the internals that
build it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from apeiron.logger.logger import Logger


def _logger() -> Logger:
    """A logger that RECORDS.

    `metrics_history` is only appended when a `csv_path` is configured
    (`wandb_logger.py:129`), so a logger built without one keeps nothing
    and `to_dataframe()` returns None. The first version of this file
    omitted the path and failed all ten cases against `src/` -- the
    fixture's fault, not the implementation's.
    """
    path = Path(tempfile.mkdtemp()) / "metrics.csv"
    return Logger(backend="none", csv_path=path)


def _logged_names(stage: str | None, rows: list[dict]) -> set[str]:
    """The metric NAMES a record holds.

    `to_dataframe()` is long-format -- columns `metric`, `step`, `value` --
    so a metric's name is a value in the `metric` column, not a column of
    its own. The first version of this file asserted against
    `frame.columns` and failed six cases against `src/` for that reason.
    This is also exactly how `check_mnist_run.py` reads the file.
    """
    logger = _logger()
    if stage is not None:
        logger.stage(stage)
    for row in rows:
        logger.log(row)
    frame = logger.to_dataframe()
    assert frame is not None, "nothing was recorded"
    return {str(v) for v in frame["metric"]}


class TestTheStepColumns:
    @pytest.mark.parametrize("stage", ["eval", "drift", "cl"])
    def test_a_bare_step_column_is_added(self, stage):
        """The global step. `MLFlowLogger` reads it back by this exact
        name when it replays the history."""
        logger = _logger()
        logger.stage(stage)
        logger.log({"loss": 1.0})
        frame = logger.to_dataframe()
        assert "step" in frame.columns

    @pytest.mark.parametrize("stage", ["eval", "drift", "cl"])
    def test_a_per_stage_step_column_is_added(self, stage):
        """`{stage}/step`, which is how a reader tells the third eval from
        the third drift check."""
        assert f"{stage}/step" in _logged_names(stage, [{"loss": 1.0}])

    def test_both_appear_without_being_passed_in(self):
        """The point: the caller logs one metric and the row carries three
        columns. A logger that passes metrics through unchanged omits two
        columns the file is expected to have."""
        names = _logged_names("eval", [{"loss": 1.0}])
        assert {"eval/step", "eval/loss"} <= names

    def test_the_metric_itself_is_still_prefixed_by_the_stage(self):
        """Guard: adding step columns must not disturb the namespacing the
        rest of the schema relies on."""
        names = _logged_names("drift", [{"score": 0.5}])
        assert "drift/score" in names
        assert "score" not in names

    def test_several_rows_each_carry_the_columns(self):
        logger = _logger()
        logger.stage("eval")
        for value in (1.0, 2.0, 3.0):
            logger.log({"loss": value})
        frame = logger.to_dataframe()
        names = {str(v) for v in frame["metric"]}
        assert "eval/step" in names
        assert "eval/loss" in names
        assert "step" in frame.columns

    def test_a_row_logged_with_no_stage_still_has_the_bare_step(self):
        """Without a stage there is no `{stage}/step` to add, but the
        global one is unconditional."""
        logger = _logger()
        logger.log({"loss": 1.0})
        frame = logger.to_dataframe()
        assert "step" in frame.columns
