"""Backward and forward transfer: the forgetting numbers a CL paper reports.

`compute_bwt` is called after every adaptation and its value is logged, and
nothing in the suite mentioned it — a reconstruction of the trainer dropped
the method outright and passed (claims/LOG.md, A30). `fwt` is computed inline
in the same place and was equally unverified.

Both are differences of evaluation scores, so their *sign* is the whole
reading, and the sign convention depends on the metric's direction: with
accuracy, negative BWT means forgetting; with a lower-is-better metric like
MAE, positive BWT means forgetting. That trap is what these tests pin.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apeiron.training.continuous_trainer import ContinuousTrainer


@pytest.fixture()
def trainer(default_cfg, dummy_harness):
    return ContinuousTrainer(
        cfg=default_cfg,
        modelHarness=dummy_harness,
        logger=MagicMock(),
        profiler=None,
    )


def _tasks(monkeypatch, trainer, rows, diagonals):
    """Stub the harness's task registry: `rows` is R[T][i], `diagonals` R[i][i].

    `task_diagonals` is a property, so it has to be replaced on the CLASS --
    through monkeypatch, or the stub outlives the test and every later test
    reading the real registry sees this one's fixture.
    """
    monkeypatch.setattr(trainer.modelHarness, "eval_past_tasks", lambda: rows)
    monkeypatch.setattr(
        type(trainer.modelHarness), "task_diagonals", property(lambda self: diagonals)
    )


class TestBackwardTransfer:
    def test_none_before_any_task_is_registered(self, monkeypatch, trainer):
        # the first drift event: the sum is empty, and 0.0 would be a lie —
        # it would read as "no forgetting" rather than "not yet measurable"
        _tasks(monkeypatch, trainer, [], [])
        assert trainer.compute_bwt() is None

    def test_one_task_is_the_single_delta(self, monkeypatch, trainer):
        # scored 90 right after adapting to it, 80 now: 10 points lost
        _tasks(monkeypatch, trainer, [[80.0]], [[90.0]])
        assert trainer.compute_bwt() == pytest.approx(-10.0)

    def test_the_mean_is_over_tasks_not_over_metrics(self, monkeypatch, trainer):
        # deltas -10, +4, -6 -> mean -4
        _tasks(monkeypatch, trainer, [[80.0], [94.0], [64.0]], [[90.0], [90.0], [70.0]])
        assert trainer.compute_bwt() == pytest.approx(-4.0)

    def test_retained_tasks_only(self, monkeypatch, trainer):
        # the harness caps retained tasks; BWT averages over the survivors, so
        # the divisor is the number of rows, never the number of events so far
        _tasks(monkeypatch, trainer, [[80.0], [80.0]], [[90.0], [90.0]])
        assert trainer.compute_bwt() == pytest.approx(-10.0)

    def test_improvement_on_past_tasks_is_positive(self, monkeypatch, trainer):
        # backward transfer is not only forgetting: adapting can help the past
        _tasks(monkeypatch, trainer, [[95.0]], [[90.0]])
        assert trainer.compute_bwt() == pytest.approx(5.0)

    def test_no_change_is_zero(self, monkeypatch, trainer):
        _tasks(monkeypatch, trainer, [[90.0], [70.0]], [[90.0], [70.0]])
        assert trainer.compute_bwt() == pytest.approx(0.0)

    def test_metric_index_selects_the_metric(self, monkeypatch, trainer):
        # rows are [accuracy, loss]; index 0 forgets, index 1 improves
        _tasks(monkeypatch, trainer, [[80.0, 0.5]], [[90.0, 0.9]])
        assert trainer.compute_bwt(metric_index=0) == pytest.approx(-10.0)
        assert trainer.compute_bwt(metric_index=1) == pytest.approx(-0.4)

    def test_the_sign_follows_the_raw_metric(self, monkeypatch, trainer):
        # the documented trap: with a higher-is-better metric a drop is
        # negative, but with a lower-is-better one (MAE) forgetting shows up
        # POSITIVE, because the error grew
        _tasks(monkeypatch, trainer, [[0.8]], [[0.5]])  # MAE rose from 0.5 to 0.8
        assert trainer.compute_bwt() == pytest.approx(0.3)
        assert trainer.compute_bwt() > 0, (
            "a lower-is-better metric degrading must read positive; a caller "
            "that assumes negative-means-forgetting will misreport it"
        )

    def test_rows_and_diagonals_are_paired_positionally(self, monkeypatch, trainer):
        # zip pairs task i's current score with task i's own diagonal; a
        # misalignment would compare different tasks and silently invent
        # forgetting that never happened
        _tasks(monkeypatch, trainer, [[10.0], [20.0]], [[10.0], [20.0]])
        assert trainer.compute_bwt() == pytest.approx(0.0)
        _tasks(monkeypatch, trainer, [[10.0], [20.0]], [[20.0], [10.0]])
        assert trainer.compute_bwt() == pytest.approx(0.0)  # deltas -10, +10
        _tasks(monkeypatch, trainer, [[10.0], [10.0]], [[20.0], [10.0]])
        assert trainer.compute_bwt() == pytest.approx(-5.0)


class TestForwardTransfer:
    """FWT = R[i][i] - R[i-1][i]: what adapting bought on the window that fired.

    Computed inline in `outer_cl_training_loop`, so it is pinned by driving
    that loop with evaluation scores we control.
    """

    def _run(
        self, monkeypatch, trainer, before, after, past=(), diagonals=(), max_iter=0
    ):
        """Drive one adaptation whose eval() returns `before` then `after`.

        `past`/`diagonals` stand in for the harness's task registry, so a run
        can be given a history and produce a BWT.
        """
        scores = [before, after]
        harness = trainer.modelHarness
        monkeypatch.setattr(harness, "eval", lambda: scores.pop(0) if scores else after)
        monkeypatch.setattr(harness, "history_eval", lambda: None)
        monkeypatch.setattr(harness, "register_task", lambda metrics: None)
        monkeypatch.setattr(harness, "eval_past_tasks", lambda: list(past))
        monkeypatch.setattr(
            type(harness), "task_diagonals", property(lambda self: list(diagonals))
        )

        logged: list[dict] = []
        logger = MagicMock()
        logger.log = lambda payload, *a, **kw: logged.append(dict(payload))
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=logger
        ):
            trainer.cfg = trainer.cfg.__class__(
                **{
                    **vars(trainer.cfg),
                    "train": trainer.cfg.train.__class__(
                        **{**vars(trainer.cfg.train), "max_iter": max_iter}
                    ),
                }
            )
            trainer.outer_cl_training_loop(drift_event_id=7)
        return logged

    def _eval_row(self, logged):
        return next(row for row in logged if "fwt" in row)

    def test_fwt_is_the_gain_on_the_window_that_triggered(self, monkeypatch, trainer):
        row = self._eval_row(
            self._run(monkeypatch, trainer, before=[60.0], after=[75.0])
        )
        assert row["fwt"] == pytest.approx(15.0)
        assert row["test_pre_cl_acc"] == pytest.approx(60.0)
        assert row["test_curr_acc"] == pytest.approx(75.0)

    def test_fwt_is_negative_when_adapting_hurt_the_window(self, monkeypatch, trainer):
        row = self._eval_row(
            self._run(monkeypatch, trainer, before=[75.0], after=[60.0])
        )
        assert row["fwt"] == pytest.approx(-15.0)

    def test_fwt_is_zero_when_nothing_moved(self, monkeypatch, trainer):
        row = self._eval_row(
            self._run(monkeypatch, trainer, before=[70.0], after=[70.0])
        )
        assert row["fwt"] == pytest.approx(0.0)

    def test_no_historical_split_omits_its_score(self, monkeypatch, trainer):
        row = self._eval_row(
            self._run(monkeypatch, trainer, before=[60.0], after=[75.0])
        )
        assert "test_hist_acc" not in row

    def test_bwt_is_absent_on_the_first_event(self, monkeypatch, trainer):
        # no past tasks yet, so the column is sparse rather than zero-filled
        row = self._eval_row(
            self._run(monkeypatch, trainer, before=[60.0], after=[75.0])
        )
        assert "bwt" not in row

    def test_bwt_joins_the_row_once_a_history_exists(self, monkeypatch, trainer):
        row = self._eval_row(
            self._run(
                monkeypatch,
                trainer,
                before=[60.0],
                after=[75.0],
                past=[[80.0]],
                diagonals=[[90.0]],
            )
        )
        # forgetting on the one retained task, reported alongside the gain on
        # the window that triggered
        assert row["bwt"] == pytest.approx(-10.0)
        assert row["fwt"] == pytest.approx(15.0)

    @pytest.mark.xfail(
        reason="A31: the row carrying fwt and bwt has no drift_event_id, so it "
        "cannot be joined to the adaptation that produced it — unlike the "
        "validation rows and the per-iteration loss rows, which both carry it",
        strict=True,
    )
    def test_the_transfer_row_can_be_joined_to_its_event(self, monkeypatch, trainer):
        row = self._eval_row(
            self._run(monkeypatch, trainer, before=[60.0], after=[75.0])
        )
        assert row["drift_event_id"] == 7

    def test_the_transfer_row_currently_omits_the_event_id(self, monkeypatch, trainer):
        # present behaviour, recorded so the xfail above has a counterpart
        row = self._eval_row(
            self._run(monkeypatch, trainer, before=[60.0], after=[75.0])
        )
        assert "drift_event_id" not in row


class TestValidationPayload:
    """`_log_validation` names every metric by its harness label.

    `eval()` returns a positional list and `eval_metrics` holds the labels in
    the same order; if that pairing slips, a run's recorded numbers are
    mislabelled and no test would notice.
    """

    def test_every_metric_is_named_for_both_domains(self, monkeypatch, trainer):
        trainer.modelHarness.eval_metrics = {"accuracy": None, "loss": None}
        logged: list[dict] = []
        logger = MagicMock()
        logger.log = lambda payload, *a, **kw: logged.append(dict(payload))
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=logger
        ):
            trainer._log_validation("post", [91.0, 0.4], [88.0, 0.6], 3)

        assert logged == [
            {
                "drift_event_id": 3,
                "val_post_cur_accuracy": 91.0,
                "val_post_cur_loss": 0.4,
                "val_post_hist_accuracy": 88.0,
                "val_post_hist_loss": 0.6,
            }
        ]

    def test_an_absent_domain_contributes_nothing(self, monkeypatch, trainer):
        trainer.modelHarness.eval_metrics = {"accuracy": None}
        logged: list[dict] = []
        logger = MagicMock()
        logger.log = lambda payload, *a, **kw: logged.append(dict(payload))
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=logger
        ):
            trainer._log_validation("pre", [91.0], None, 1)

        assert logged[0] == {"drift_event_id": 1, "val_pre_cur_accuracy": 91.0}

    def test_the_tag_separates_the_rounds(self, monkeypatch, trainer):
        trainer.modelHarness.eval_metrics = {"accuracy": None}
        logged: list[dict] = []
        logger = MagicMock()
        logger.log = lambda payload, *a, **kw: logged.append(dict(payload))
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=logger
        ):
            trainer._log_validation("pre", [60.0], None, 1)
            trainer._log_validation("post", [75.0], None, 1)

        assert "val_pre_cur_accuracy" in logged[0]
        assert "val_post_cur_accuracy" in logged[1]

    def test_the_round_s_step_is_annotated_not_advanced(self, monkeypatch, trainer):
        # increment=False and commit=False: these rows annotate the CL round
        # rather than advancing the metric step, so a later row lands on the
        # same step as the losses it describes
        trainer.modelHarness.eval_metrics = {"accuracy": None}
        logger = MagicMock()
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=logger
        ):
            trainer._log_validation("post", [91.0], None, 1)
        assert logger.log.call_args.kwargs == {"commit": False, "increment": False}
