"""The three seams a subclass uses, exercised by an actual subclass.

`test_subclass_seams.py` pins the surface `src/drift_only.py` and
`src/cl_only.py` *read* off `ContinuousMonitor`. This file pins the three
places they *substitute themselves into* it -- which is a different
thing, because reading a name only requires the name to exist, while
overriding a method requires the base class to call it a particular way,
and calling an inherited one requires it to do its whole job.

  `_log_metrics(drift_signal, metric_value)`
      `drift_only.py:142` overrides it with two parameters and logs the
      value the detector was given. The base class calls it as
      `self._log_metrics(drift_signal, agg_metric)`
      (`continuous_monitor.py:282`), so the second argument is part of
      the seam: a base class that passes only the signal cannot call
      either override at all.

  `_process_stream()` raising `StopIteration`
      `cl_only.py:277` runs a whole window, consults its schedule, and
      raises `StopIteration` to hand control back. `run()` catches it
      and extends the stream:

          while not self._should_stop():
              try:
                  self._process_stream()
              except StopIteration:
                  self._extend_stream()

      The base class's own `_process_stream` ends the same way, so a
      `run()` that instead calls `_process_stream()` and then
      `_extend_stream()` unconditionally behaves identically **for the
      base class** and lets the exception out of every override.

HOW THIS WAS FOUND, because it says something about the criteria. A
regrowth of this layer passed every gate here, then reproduced the pinned
MNIST run exactly -- 31 metric series by digest, the trained weights, the
drift step, the final accuracy. It then failed 36 of the upstream suite's
entry-point tests, all with one of two errors:

    TypeError: DriftOnlyMonitor._log_metrics() missing 1 required
               positional argument: 'metric_value'
    StopIteration

Both seams are invisible to this layer's criteria for the same reason:
the only code that overrides them lives in another layer. A test that
drives `ContinuousMonitor` itself cannot tell a one-argument
`_log_metrics` from a two-argument one, and cannot tell a `run()` that
catches `StopIteration` from one that never needs to. The fix is to put
the subclass in the test -- below, `_Recording` does what `drift_only.py`
does and nothing else.

  `_adapt(event_id)` checkpointing
      The third, found the same way and documented on the class in
      `TestAdaptingCheckpoints` below: `cl_only.py:353` calls
      `self._adapt(self.trigger_count)` and does nothing else, so every
      promise in that method's docstring -- "run the CL loop, checkpoint,
      and resume" -- is a promise to the scheduled runner. A regrowth
      moved the save into `_adapt`'s caller.

Pinned as the protocol rather than as the source's exact control flow: a
`run()` may be written either way as long as an override that raises
`StopIteration` is caught, `_log_metrics` may name its parameters
anything as long as it takes the aggregate alongside the signal, and the
checkpoint may be written however so long as `_adapt` is what writes it.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from apeiron.driver.continuous_monitor import ContinuousMonitor


@pytest.fixture(autouse=True)
def _patch_logger():
    mock_logger = MagicMock()
    mock_logger.step = 0
    with patch("apeiron.driver.continuous_monitor.get_logger",
               return_value=mock_logger):
        with patch("apeiron.training.continuous_trainer.get_logger",
                   return_value=mock_logger):
            yield mock_logger


def _quiet_cfg(default_cfg, **drift):
    """A config that checks drift often and extends the stream twice."""
    return replace(default_cfg,
                   drift_detection=replace(default_cfg.drift_detection,
                                           detection_interval=1,
                                           max_stream_updates=2, **drift))


def _ckpt_cfg(default_cfg, tmp_path, *, keep: int):
    """A config with checkpointing set, for BOTH the monitor and the harness.

    `ckpts_enabled` reads the HARNESS's own config, not the monitor's, so
    the harness has to be built from this same object. The first version
    of these cases built the harness from the unmodified fixture config
    and the guard below caught it -- which is the only reason the cases
    are not quietly vacuous.
    """
    return replace(default_cfg,
                   model=replace(default_cfg.model, max_ckpts=keep,
                                 ckpts_path=str(tmp_path / "ckpts")))


class _Recording(ContinuousMonitor):
    """What `drift_only.py` is, reduced to the two seams.

    Deliberately NOT a MagicMock: a mock accepts any arity and would make
    the first assertion vacuous. A real `def` with two parameters is the
    only thing that fails when the base class passes one argument.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged: list[tuple[object, float]] = []

    def _log_metrics(self, drift_signal, metric_value) -> None:
        self.logged.append((drift_signal, metric_value))


class _WindowRaises(ContinuousMonitor):
    """What `cl_only.py` is: a window that ends by raising."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.windows = 0

    def _process_stream(self) -> None:
        self.windows += 1
        raise StopIteration


class TestTheLogMetricsSeamTakesTheAggregate:
    def test_a_two_parameter_override_is_callable(self, default_cfg,
                                                  dummy_harness):
        """The whole finding. `drift_only.py` and `cl_only.py` both define
        `_log_metrics(self, drift_signal, metric_value)`; a base class that
        calls it with one argument raises TypeError on every drift check."""
        monitor = _Recording(cfg=_quiet_cfg(default_cfg),
                             modelHarness=dummy_harness)
        monitor.run()
        assert monitor.logged, (
            "the override was never called, so this case proves nothing -- "
            "the fixture must reach at least one drift check"
        )

    def test_the_value_passed_is_the_aggregate_the_detector_saw(
        self, default_cfg, dummy_harness
    ):
        """Not any float: the same number `_aggregate_buffer` produced and
        the detector was updated with. `drift_only.py` logs it as
        `metric_{self.metric_idx}`, so a different value would mislabel
        the column rather than fail."""
        monitor = _Recording(cfg=_quiet_cfg(default_cfg),
                             modelHarness=dummy_harness)
        monitor.run()
        assert monitor.logged
        for _signal, value in monitor.logged:
            assert isinstance(value, float), (
                f"the aggregate reached the override as {type(value)!r}; "
                f"it is logged as a metric value"
            )
        last_signal, last_value = monitor.logged[-1]
        assert last_value == monitor.last_agg_metric, (
            f"the override was passed {last_value} but the monitor's own "
            f"`last_agg_metric` is {monitor.last_agg_metric}; the two "
            f"subclasses log one and read the other, and they must agree"
        )

    def test_the_signal_is_passed_too(self, default_cfg, dummy_harness):
        monitor = _Recording(cfg=_quiet_cfg(default_cfg),
                             modelHarness=dummy_harness)
        monitor.run()
        signal, _value = monitor.logged[-1]
        assert hasattr(signal, "drift_detected"), (
            "the first argument must be the DriftSignal; both overrides "
            "read `drift_signal.drift_detected` off it"
        )


class TestAWindowMayEndByRaising:
    def test_stop_iteration_from_the_override_is_caught(self, default_cfg,
                                                        dummy_harness):
        """`cl_only.py:277` raises `StopIteration` once its window is done,
        documented in its own docstring as "Always, once the window is
        exhausted, to hand control back to run()". If `run()` does not
        catch it, every scheduled-CL run dies on its first window."""
        monitor = _WindowRaises(cfg=_quiet_cfg(default_cfg),
                                modelHarness=dummy_harness)
        monitor.run()          # must not propagate
        assert monitor.windows > 0, "the override never ran"

    def test_each_raise_still_extends_the_stream(self, default_cfg,
                                                 dummy_harness):
        """Caught, not swallowed: the loop has to make progress or it
        spins forever on a window that always raises."""
        monitor = _WindowRaises(cfg=_quiet_cfg(default_cfg),
                                modelHarness=dummy_harness)
        monitor.run()
        assert monitor.stream_update_count >= monitor.max_stream_updates, (
            f"the run stopped after {monitor.stream_update_count} stream "
            f"update(s) of {monitor.max_stream_updates}; a window that ends "
            f"by raising must still advance the stream"
        )
        assert monitor.windows == monitor.stream_update_count, (
            f"{monitor.windows} window(s) ran against "
            f"{monitor.stream_update_count} stream update(s); one each"
        )


class TestAdaptingCheckpoints:
    """The third seam: what `_adapt` has to DO, not just take.

    Its docstring says "Pause monitoring, run the CL loop, checkpoint, and
    resume", and the body does all four:

        self.trainer.outer_cl_training_loop(drift_event_id=event_id)
        if self.modelHarness.ckpts_enabled:
            ckptpath = self.modelHarness.save_ckpt(event=event_id)

    `cl_only.py:353` calls `self._adapt(self.trigger_count)` and nothing
    else -- no training call of its own, no save. So every promise in that
    docstring is a promise to the scheduled runner.

    A regrowth put the save in `_adapt`'s CALLER instead, keyed on
    `self.drift_event_count`. For the detection path that is identical,
    which is why the pinned MNIST run still reproduced to the digit,
    trained weights included. For the schedule path it means a run that
    adapts on a timetable trains the model and never persists it -- and
    `drift_event_count` is zero throughout, so even a save placed there
    would overwrite one file.

    Pinned on the seam: call `_adapt` directly, as `cl_only.py` does,
    and require the checkpoint. Keyed on the ARGUMENT, because that is
    the detail the two paths disagree about.
    """

    def test_adapt_writes_a_checkpoint(self, default_cfg, make_harness,
                                       tmp_path):
        """The whole finding, reached the way `cl_only.py` reaches it."""
        cfg = _ckpt_cfg(default_cfg, tmp_path, keep=2)
        monitor = ContinuousMonitor(cfg=cfg, modelHarness=make_harness(cfg))
        assert monitor.modelHarness.ckpts_enabled, (
            "checkpointing is off in this fixture, so the case below would "
            "pass on a monitor that never saves"
        )
        monitor._adapt(7)
        written = sorted(p.name for p in
                         (tmp_path / "ckpts").glob("drift_adaptation_*.pt"))
        assert written, (
            "`_adapt` trained but wrote no checkpoint; a subclass that "
            "adapts on a schedule calls only this and would never persist "
            "the model it just trained"
        )

    def test_the_checkpoint_is_named_for_the_argument(self, default_cfg,
                                                      make_harness, tmp_path):
        """Not for the monitor's own drift counter. `cl_only.py` passes its
        `trigger_count`, and a save keyed on `drift_event_count` would name
        every scheduled checkpoint `_0` and evict the one before it."""
        cfg = _ckpt_cfg(default_cfg, tmp_path, keep=5)
        monitor = ContinuousMonitor(cfg=cfg, modelHarness=make_harness(cfg))
        for event in (3, 4):
            monitor._adapt(event)
        written = sorted(p.name for p in
                         (tmp_path / "ckpts").glob("drift_adaptation_*.pt"))
        assert written == ["drift_adaptation_3.pt", "drift_adaptation_4.pt"], (
            f"two adaptations at events 3 and 4 wrote {written}; the file "
            f"name has to follow the event id `_adapt` was given, since "
            f"the monitor's own counter does not move on the schedule path"
        )

    def test_no_checkpoint_when_checkpointing_is_off(self, default_cfg,
                                                     make_harness, tmp_path):
        """Guard, and the other direction: `ckpts_enabled` is consulted,
        so a monitor that always saves fails here."""
        cfg = _ckpt_cfg(default_cfg, tmp_path, keep=0)
        monitor = ContinuousMonitor(cfg=cfg, modelHarness=make_harness(cfg))
        assert not monitor.modelHarness.ckpts_enabled
        monitor._adapt(1)
        assert not list(tmp_path.glob("ckpts/*.pt")), (
            "a checkpoint was written with max_ckpts=0"
        )


class TestTheFixtureCouldHaveFailed:
    """Guards, because both cases above pass trivially on a monitor that
    never reaches a drift check or never loops."""

    def test_the_base_class_reaches_a_drift_check(self, default_cfg,
                                                  dummy_harness):
        monitor = ContinuousMonitor(cfg=_quiet_cfg(default_cfg),
                                    modelHarness=dummy_harness)
        monitor.run()
        assert monitor.check_count > 0, (
            "no drift check happened, so `_log_metrics` would never be "
            "called whatever its signature"
        )

    def test_the_base_class_loops_more_than_once(self, default_cfg,
                                                 dummy_harness):
        monitor = ContinuousMonitor(cfg=_quiet_cfg(default_cfg),
                                    modelHarness=dummy_harness)
        monitor.run()
        assert monitor.stream_update_count >= 2, (
            "the loop ran once, so a `run()` that mishandles the second "
            "iteration would pass"
        )
