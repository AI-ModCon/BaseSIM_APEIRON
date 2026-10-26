"""Tests for logging: ConsoleLogger, WandBLogger, and unified Logger."""

from __future__ import annotations

import logging

import pytest

from apeiron.logger.console_logger import ConsoleLogger, ColoredFormatter, StepFilter
from apeiron.logger.wandb_logger import WandBLogger, VALID_STAGES


# ---------------------------------------------------------------------------
# ConsoleLogger
# ---------------------------------------------------------------------------
class TestConsoleLogger:
    def test_parse_verbosity_plain(self):
        c = ConsoleLogger(verbosity="WARNING")
        base, tier = c._parse_verbosity("WARNING")
        assert base == "WARNING"
        assert tier == 0

    def test_parse_verbosity_with_tier(self):
        c = ConsoleLogger(verbosity="INFO:3")
        base, tier = c._parse_verbosity("INFO:3")
        assert base == "INFO"
        assert tier == 3

    def test_parse_verbosity_invalid_tier(self):
        c = ConsoleLogger(verbosity="INFO")
        base, tier = c._parse_verbosity("INFO:abc")
        assert base == "INFO"
        assert tier == 0

    def test_get_log_level_info(self):
        c = ConsoleLogger(verbosity="INFO")
        assert c._get_log_level("INFO") == 20

    def test_get_log_level_info_tier(self):
        c = ConsoleLogger(verbosity="INFO:3")
        assert c._get_log_level("INFO:3") == 17  # 20 - 3

    def test_get_log_level_debug(self):
        c = ConsoleLogger(verbosity="DEBUG")
        assert c._get_log_level("DEBUG") == logging.DEBUG

    def test_verbosity_property(self):
        c = ConsoleLogger(verbosity="INFO")
        assert c.verbosity == "INFO"
        c.verbosity = "DEBUG"
        assert c.verbosity == "DEBUG"

    def test_info_level_above_max_uses_debug(self):
        c = ConsoleLogger(verbosity="DEBUG")
        # level >= 10 should fall through to debug
        c.info("test message", level=10)  # should not raise


# ---------------------------------------------------------------------------
# StepFilter
# ---------------------------------------------------------------------------
class TestStepFilter:
    def test_adds_step_to_record(self):
        f = StepFilter(get_step=lambda: 42)
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        assert record.step == 42  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ColoredFormatter
# ---------------------------------------------------------------------------
class TestColoredFormatter:
    def test_formats_without_error(self):
        fmt = ColoredFormatter("%(levelname)s | %(message)s")
        record = logging.LogRecord("test", logging.WARNING, "", 0, "hello", (), None)
        result = fmt.format(record)
        assert "hello" in result


# ---------------------------------------------------------------------------
# WandBLogger (with wandb disabled)
# ---------------------------------------------------------------------------
class TestWandBLogger:
    def test_stage_valid(self):
        w = WandBLogger(enabled=False)
        w.stage("eval")
        assert w.current_stage == "eval"
        w.stage("drift")
        assert w.current_stage == "drift"
        w.stage("cl")
        assert w.current_stage == "cl"

    def test_stage_invalid(self):
        w = WandBLogger(enabled=False)
        with pytest.raises(ValueError, match="Invalid stage"):
            w.stage("invalid")  # type: ignore[arg-type]

    def test_step_increments_on_log(self):
        w = WandBLogger(enabled=False, csv_path="/tmp/test.csv")
        assert w.step == 0
        w.stage("eval")
        w.log({"acc": 95.0})
        assert w.step == 1
        w.log({"acc": 96.0})
        assert w.step == 2

    def test_log_no_increment(self):
        w = WandBLogger(enabled=False, csv_path="/tmp/test.csv")
        w.stage("eval")
        w.log({"acc": 95.0}, increment=False)
        assert w.step == 0

    def test_stage_step_tracking(self):
        w = WandBLogger(enabled=False, csv_path="/tmp/test.csv")
        w.stage("eval")
        w.log({"acc": 95.0})
        w.log({"acc": 96.0})
        w.stage("cl")
        w.log({"loss": 0.5})
        assert w.get_stage_step("eval") == 2
        assert w.get_stage_step("cl") == 1
        assert w.get_stage_step("drift") == 0

    def test_prefix_applied(self):
        w = WandBLogger(enabled=False, csv_path="/tmp/test.csv")
        w.stage("eval")
        w.log({"acc": 95.0})
        entry = w.metrics_history[-1]
        assert "eval/acc" in entry
        assert "step" in entry

    def test_prefix_skipped_for_slashed_keys(self):
        w = WandBLogger(enabled=False, csv_path="/tmp/test.csv")
        w.stage("eval")
        w.log({"custom/metric": 1.0})
        entry = w.metrics_history[-1]
        assert "custom/metric" in entry
        assert "eval/custom/metric" not in entry

    def test_to_dataframe_empty(self):
        w = WandBLogger(enabled=False)
        assert w.to_dataframe() is None

    def test_to_dataframe_with_data(self):
        w = WandBLogger(enabled=False, csv_path="/tmp/test.csv")
        w.stage("eval")
        w.log({"acc": 95.0})
        df = w.to_dataframe()
        assert df is not None
        assert len(df) > 0
        assert "step" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_to_csv(self, tmp_path):
        csv_path = tmp_path / "metrics.csv"
        w = WandBLogger(enabled=False, csv_path=str(csv_path))
        w.stage("eval")
        w.log({"acc": 95.0})
        result = w.to_csv()
        assert result == csv_path
        assert csv_path.exists()

    def test_to_csv_returns_none_when_empty(self):
        w = WandBLogger(enabled=False, csv_path="/tmp/test.csv")
        assert w.to_csv() is None

    def test_finish_disabled(self):
        w = WandBLogger(enabled=False)
        result = w.finish()
        assert result is None

    def test_url_none_when_no_run(self):
        w = WandBLogger(enabled=False)
        assert w.url is None


# ---------------------------------------------------------------------------
# VALID_STAGES constant
# ---------------------------------------------------------------------------
class TestValidStages:
    def test_contains_expected(self):
        assert "eval" in VALID_STAGES
        assert "drift" in VALID_STAGES
        assert "cl" in VALID_STAGES
        assert len(VALID_STAGES) == 3
