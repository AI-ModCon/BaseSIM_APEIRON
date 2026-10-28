"""`Logger.log`'s keyword surface, and what `backend = "none"` promises.

`Logger.log` is the method every layer above calls, and its signature is
five parameters wide:

    def log(self, metrics, step=None, commit=True, prefix=True,
            increment=True) -> None

Each one is a caller-facing choice -- `prefix` decides whether the stage
name is prepended to a metric key and so decides a CSV column name;
`increment` decides whether two calls land on one step or two; `step`
overrides the counter outright. The existing suite calls `log(metrics)`
and `log(metrics, step=...)`; the other three keywords are never passed,
so a reconstruction is free to drop them or to forward them somewhere
that cannot take them.

A rebuild did exactly that. `Logger(backend="none").log({...},
prefix=False)` raised

    TypeError: WandBLogger.log() got an unexpected keyword argument 'prefix'

The facade forwards all five to the backend in one call
(`logger.py:94`), so a reconstruction that forwards a different arity --
or forwards to a backend built with a different signature -- breaks on the
first caller that passes the keyword nothing tested.

`backend = "none"` is the other half. It is NOT "no backend object": the
source deliberately builds `WandBLogger(enabled=False)` so the facade
needs no second code path and CSV output keeps working. What it promises
is that no run is opened and nothing is sent, which is what `url`
reports. So this file asserts the facade accepts its own documented
keywords in every combination a caller can reach, that `"none"` stays
`"none"` while they are exercised, and that the CSV a none-backend run
exists for is still written.
"""

from __future__ import annotations

import itertools

import pytest

from apeiron.logger.logger import Logger


def _none_logger(stage: str | None = "eval") -> Logger:
    logger = Logger(backend="none")
    if stage is not None:
        logger.stage(stage)
    return logger


class TestDocumentedKeywords:
    """Every keyword in the signature, accepted by the facade."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"step": 42},
            {"step": 0},
            {"commit": False},
            {"commit": True},
            {"prefix": False},
            {"prefix": True},
            {"increment": False},
            {"increment": True},
        ],
    )
    def test_each_keyword_is_accepted(self, kwargs):
        _none_logger().log({"loss": 1.0}, **kwargs)

    @pytest.mark.parametrize(
        ("prefix", "increment", "commit"),
        list(itertools.product([True, False], repeat=3)),
    )
    def test_the_combinations_are_accepted(self, prefix, increment, commit):
        """A caller can set these independently, so a reconstruction that
        handles one and forwards another is caught here rather than in
        whichever run first passes the pair."""
        _none_logger().log(
            {"loss": 1.0}, prefix=prefix, increment=increment, commit=commit
        )

    def test_positional_step_still_works(self):
        """`step` is the second positional parameter, and callers use it
        that way."""
        _none_logger().log({"loss": 1.0}, 7)

    def test_logging_before_a_stage_is_named_is_accepted(self):
        _none_logger(stage=None).log({"loss": 1.0})


class TestNoneBackendStaysNone:
    """`backend = "none"` is a promise about the network, not a default."""

    def test_the_backend_is_reported_as_none(self):
        assert _none_logger().backend == "none"

    @pytest.mark.parametrize(
        "kwargs",
        [{"prefix": False}, {"increment": False}, {"commit": False},
         {"step": 3}],
    )
    def test_it_is_still_none_after_logging_with_any_keyword(self, kwargs):
        logger = _none_logger()
        logger.log({"loss": 1.0}, **kwargs)
        assert logger.backend == "none"

    def test_finish_is_accepted_on_a_none_backend(self):
        logger = _none_logger()
        logger.log({"loss": 1.0})
        logger.finish(save_csv=False)

    def test_it_opens_no_run(self):
        """What `"none"` actually promises.

        The first version of this test asserted that a none-backend logger
        holds no `WandBLogger` at all, and it failed against `src/` --
        correctly, because `logger.py:44` builds
        `WandBLogger(enabled=False)` on purpose, using the disabled backend
        as the null object so the facade needs no second code path and
        `to_csv` keeps working. The promise is not "no wandb class is
        instantiated"; it is "no run is opened and nothing is sent". That
        is what `url` reports, and it is assertable without reaching into
        the facade's internals.
        """
        logger = _none_logger()
        logger.log({"loss": 1.0})
        assert logger.url is None

    def test_the_csv_path_still_works_on_a_none_backend(self, tmp_path):
        """Which is the reason the disabled backend exists rather than a
        null object: `backend = "none"` is how you run with CSV output and
        no service, so the CSV must still be produced."""
        logger = Logger(backend="none", csv_path=tmp_path / "m.csv")
        logger.stage("eval")
        logger.log({"loss": 1.0})
        written = logger.to_csv()
        assert written is not None
        assert written.exists()
