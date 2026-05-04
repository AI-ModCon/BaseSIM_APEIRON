"""Console logging interface with verbosity control."""

from __future__ import annotations

import logging
import sys
from typing import Callable


# Constants
MAX_INFO_TIER = 9  # Support INFO:0 through INFO:9
LOG_FORMAT = "%(levelname)s | %(asctime)s | step=%(step)s | %(module)s | %(message)s"
TIME_FORMAT = "%H:%M:%S"


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    WHITE = "\033[37m"


class StepFilter(logging.Filter):
    """Filter that adds step information to log records."""

    def __init__(self, get_step: Callable[[], int]):
        super().__init__()
        self.get_step = get_step

    def filter(self, record: logging.LogRecord) -> bool:
        record.step = self.get_step()
        return True


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""

    LEVEL_COLORS = {
        "CRITICAL": Colors.RED,
        "ERROR": Colors.RED,
        "WARNING": Colors.YELLOW,
        "DEBUG": Colors.WHITE,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add INFO:n colors dynamically
        self.LEVEL_COLORS["INFO:0"] = Colors.GREEN
        for n in range(1, MAX_INFO_TIER + 1):
            self.LEVEL_COLORS[f"INFO:{n}"] = Colors.BLUE

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colored level name."""
        color = self.LEVEL_COLORS.get(record.levelname, "")

        # Create a copy with colored level name
        record_copy = logging.makeLogRecord(record.__dict__)
        if color:
            record_copy.levelname = f"{color}{record.levelname}{Colors.RESET}"

        return super().format(record_copy)


class ConsoleLogger:
    """Console logging with tiered INFO verbosity (INFO:0 to INFO:9)."""

    LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def _parse_verbosity(self, level_str: str) -> tuple[str, int]:
        """Parse 'INFO:2' → ('INFO', 2) or 'DEBUG' → ('DEBUG', 0)."""
        level_str = level_str.upper().strip()

        if ":" in level_str:
            base_level, tier_str = level_str.split(":", 1)
            try:
                tier = max(0, int(tier_str))
            except ValueError:
                tier = 0
            return (base_level, tier)

        return (level_str, 0)

    def _get_log_level(self, level_str: str) -> int:
        """Calculate numeric level: INFO:n → 20-n, DEBUG → 10, etc."""
        base_level, tier = self._parse_verbosity(level_str)
        return (
            20 - tier
            if base_level == "INFO"
            else self.LEVEL_MAP.get(base_level, logging.INFO)
        )

    def __init__(
        self, verbosity: str = "INFO", get_step: Callable[[], int] | None = None
    ):
        """Initialize console logger with tiered INFO verbosity support.

        Args:
            verbosity: Console level (DEBUG, INFO, INFO:n, WARNING, ERROR, CRITICAL)
            get_step: Callback to get current step for display (defaults to returning 0)
        """
        self._verbosity = verbosity.upper()
        self._get_step = get_step or (lambda: 0)

        # Register custom INFO:n levels (INFO:0=20, INFO:1=19, ..., INFO:9=11)
        for n in range(MAX_INFO_TIER + 1):
            logging.addLevelName(20 - n, f"INFO:{n}")

        # Configure logger
        self._logger = logging.getLogger(__name__)
        log_level = self._get_log_level(self._verbosity)
        self._logger.setLevel(log_level)

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt=TIME_FORMAT))
            handler.addFilter(StepFilter(self._get_step))
            self._logger.addHandler(handler)

        self._logger.propagate = False

    @property
    def verbosity(self) -> str:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: str) -> None:
        self._verbosity = level.upper()
        log_level = self._get_log_level(self._verbosity)
        self._logger.setLevel(log_level)
        for handler in self._logger.handlers:
            handler.setLevel(log_level)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, stacklevel=3, **kwargs)

    def info(self, msg: str, level: int = 0, *args, **kwargs) -> None:
        """Log at INFO:level tier (0=standard, 1+=more verbose)."""
        if level >= 10:
            self._logger.debug(msg, *args, stacklevel=3, **kwargs)
        else:
            self._logger.log(20 - level, msg, *args, stacklevel=3, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, stacklevel=3, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, stacklevel=3, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, stacklevel=3, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, stacklevel=3, **kwargs)
