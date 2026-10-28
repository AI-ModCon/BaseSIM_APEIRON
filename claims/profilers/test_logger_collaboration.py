"""Which logger method `print_performance` calls, and how.

`print_performance(logger=None, level=0)` documents its own contract:

    logger: Optional logger instance. If provided, uses logger.info().
            If None, falls back to print() for backward compatibility.
    level:  Verbosity level for logging (default: 0). Only used when
            logger is provided.

and the implementation matches it exactly (`count_flops.py`):

    output_fn = (lambda msg: logger.info(msg, level=level)) if logger else print

Nothing asserted it, and the omission survived every per-layer gate on
both sides of the boundary: the profiler's own tests pass a double or no
logger at all, and the logger's tests never see the profiler.

A reconstruction duck-typed instead, trying `("log", "info", "print")` in
that order and falling back on `TypeError`. Against the real `Logger` that
is fatal in a way no type check catches: `Logger.log(metrics, step=...,
commit=..., prefix=..., increment=...)` accepts a string POSITIONALLY as
`metrics`, so the call succeeds and then fails inside the backend --

    File "apeiron/logger/wandb_logger.py", line 163, in log
      for key, value in metrics.items():
    AttributeError: 'str' object has no attribute 'items'

-- and `AttributeError` is not what the fallback chain catches, so it does
not try `info` either. The assembled framework imported, passed the pinned
MNIST harness gate, and died here at the first drift event.

Pinned with a recording double rather than by inspecting the source,
because what matters is the call that actually happens: `info` is used,
`log` is not, the message is a string, and `level` is forwarded.
"""

from __future__ import annotations

import pytest

from apeiron.profilers.count_flops import FLOPSProfiler


class RecordingLogger:
    """Accepts every method the real `Logger` has, and records the calls.

    `log` is deliberately present and deliberately strict about its first
    argument being a mapping, exactly as the real one is -- so a profiler
    that reaches for `log` with a string fails here the same way it fails
    in a run, rather than being quietly tolerated by a permissive double.
    """

    def __init__(self) -> None:
        self.info_calls: list[tuple[str, int]] = []
        self.log_calls: list[object] = []

    def info(self, msg, level=0, *args, **kwargs):
        self.info_calls.append((msg, level))

    def log(self, metrics, step=None, commit=True, prefix=True,
            increment=True):
        self.log_calls.append(metrics)
        metrics.items()          # the real one does this; strings fail

    def stage(self, name):
        pass


@pytest.fixture()
def measured() -> FLOPSProfiler:
    """A profiler with something to report."""
    import torch
    from torch import nn

    model = nn.Linear(8, 4)
    profiler = FLOPSProfiler(warmup_iters=0)
    with profiler.measure_flops("fwd"):
        model(torch.ones(2, 8))
    return profiler


class TestItUsesInfo:
    """The documented collaboration."""

    def test_the_report_goes_to_info(self, measured):
        logger = RecordingLogger()
        measured.print_performance(logger=logger, level=1)
        assert logger.info_calls, "nothing was sent to logger.info"

    def test_it_does_not_use_log(self, measured):
        """`Logger.log` is for metric dictionaries. Sending a report line
        there is accepted positionally and fails inside the backend."""
        logger = RecordingLogger()
        measured.print_performance(logger=logger, level=1)
        assert not logger.log_calls, (
            f"print_performance sent {logger.log_calls!r} to logger.log; "
            f"that parameter is a metrics mapping, not a message"
        )

    def test_every_message_is_a_string(self, measured):
        logger = RecordingLogger()
        measured.print_performance(logger=logger, level=0)
        assert all(isinstance(msg, str) for msg, _level in logger.info_calls)

    @pytest.mark.parametrize("level", [0, 1, 2, 5])
    def test_the_level_is_forwarded(self, measured, level):
        """Documented as "only used when logger is provided", and it is how
        an operator turns the table off without turning the run off."""
        logger = RecordingLogger()
        measured.print_performance(logger=logger, level=level)
        assert {lvl for _msg, lvl in logger.info_calls} == {level}

    def test_an_empty_profiler_still_reports_through_info(self):
        """The no-data branch takes the same route."""
        logger = RecordingLogger()
        FLOPSProfiler().print_performance(logger=logger, level=0)
        assert logger.info_calls
        assert not logger.log_calls


class TestItFallsBackToPrint:
    """Documented: no logger means `print`, for backward compatibility."""

    def test_no_logger_does_not_raise(self, measured, capsys):
        measured.print_performance()
        assert capsys.readouterr().out.strip()

    def test_no_logger_with_a_level_does_not_raise(self, measured, capsys):
        """`level` is documented as ignored without a logger, so passing it
        must not become an error."""
        measured.print_performance(logger=None, level=3)
        assert capsys.readouterr().out.strip()
