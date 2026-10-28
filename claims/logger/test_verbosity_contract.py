"""How a `verbosity` string becomes a log level.

`[logging] verbosity` is a config value, so this is the surface a typo in a
TOML file reaches, and the mapping from string to level is user-facing
behaviour however private the methods implementing it are. Nothing in the
suite asserts it: `test_logger.py` constructs loggers and checks that they
log, never that a particular string yields a particular level.

The rules, read off `console_logger.LEVEL_MAP` and `_get_log_level`:

    return 20 - tier if base_level == "INFO" else LEVEL_MAP.get(base_level,
                                                                logging.INFO)

so the tier offset applies to INFO **only** -- `DEBUG:2` is DEBUG, not
DEBUG minus two -- and a name `LEVEL_MAP` does not hold falls back to INFO.
Names are upper-cased and surrounding whitespace is stripped, so `"info"`
and `" info "` work.

These two methods are private by name and are already the surface
`test_logger.py` tests directly -- `test_get_log_level_info_tier` asserts
`_get_log_level("INFO:3") == 17` -- so extending their coverage follows the
suite rather than reaching past it. What is added is the part the existing
cases leave open: case folding, the INFO-only rule, a negative or
non-numeric tier, and what an unknown name does not do.

Two reconstructions of this layer, given the suite and not the source, each
got this wrong and differently: one applied the tier to every level
(`CRITICAL:5` -> 45, `debug:2` -> 8), and one **raised TypeError** on the
lowercase `"info"` because it never upper-cased before arithmetic. A
framework whose config accepts `verbosity = "info"` and crashes on it is
worse than one that mis-levels it, and nothing here would have caught
either.

WHAT THIS DELIBERATELY DOES NOT PIN, and why -- the same reasoning as the
profiler's failed-measurement state:

  * which level an unrecognised name produces. The current answer is INFO
    for everything `LEVEL_MAP` lacks, which means `verbosity = "WARN"` --
    the ordinary abbreviation -- asks for warnings and gets the full info
    stream, with nothing said about the setting not being understood. The
    same holds for "NOTSET", "TRACE" and "". Whether that should raise,
    warn, or map through aliases is a decision for whoever owns the
    configuration surface, and pinning today's answer into a gate settles
    it by default. Asserted here is only the part that is not in question:
    an unrecognised name must not RAISE, because a config typo taking down
    a training run is no one's intent.

  * `Logger`'s default backend. It is `"wandb"`, so a config with no
    `[logging]` section contacts a metrics service (claims/LOG.md, A18);
    one reconstruction defaulted to `"none"` instead. That is the same kind
    of open question and is left to the same judgement.
"""

from __future__ import annotations

import logging

import pytest

from apeiron.logger.console_logger import ConsoleLogger


@pytest.fixture()
def console() -> ConsoleLogger:
    return ConsoleLogger(verbosity="INFO")


class TestNameNormalisation:
    """`verbosity = "info"` is a thing people write."""

    @pytest.mark.parametrize("raw", ["INFO", "info", "Info", " info ", "\tINFO "])
    def test_case_and_whitespace_are_ignored(self, console, raw):
        assert console._parse_verbosity(raw) == ("INFO", 0)
        assert console._get_log_level(raw) == logging.INFO

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("DEBUG", logging.DEBUG),
            ("debug", logging.DEBUG),
            ("WARNING", logging.WARNING),
            ("warning", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
            ("critical", logging.CRITICAL),
        ],
    )
    def test_every_mapped_name_resolves_in_any_case(self, console, raw, expected):
        assert console._get_log_level(raw) == expected


class TestTierOffset:
    """`INFO:n` is the documented tier scheme, and it is INFO-only."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("INFO:0", 20), ("INFO:1", 19), ("INFO:2", 18), ("INFO:9", 11)],
    )
    def test_the_tier_subtracts_from_info(self, console, raw, expected):
        assert console._get_log_level(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # The offset is NOT applied to a named level other than INFO.
            # Both reconstructions subtracted here and produced 8 and 45.
            ("DEBUG:2", logging.DEBUG),
            ("debug:2", logging.DEBUG),
            ("WARNING:5", logging.WARNING),
            ("CRITICAL:5", logging.CRITICAL),
            ("ERROR:3", logging.ERROR),
        ],
    )
    def test_the_tier_does_not_move_other_levels(self, console, raw, expected):
        assert console._get_log_level(raw) == expected

    @pytest.mark.parametrize("raw", ["INFO:-1", "INFO:-9"])
    def test_a_negative_tier_is_floored_at_zero(self, console, raw):
        """Otherwise a negative tier would make the logger QUIETER than the
        level named, by raising the threshold above it."""
        assert console._parse_verbosity(raw) == ("INFO", 0)
        assert console._get_log_level(raw) == logging.INFO

    @pytest.mark.parametrize("raw", ["INFO:", "INFO:abc", "INFO:1:2", "INFO: "])
    def test_a_tier_that_is_not_a_number_is_zero(self, console, raw):
        assert console._parse_verbosity(raw) == ("INFO", 0)
        assert console._get_log_level(raw) == logging.INFO


class TestUnrecognisedNames:
    """Only that they are survivable -- see the module docstring."""

    @pytest.mark.parametrize(
        "raw", ["WARN", "NOTSET", "TRACE", "", ":", ":3", "nope", "INF0"]
    )
    def test_an_unknown_name_does_not_raise(self, console, raw):
        """A config typo must not take the run down. Which level it maps to
        is deliberately not asserted."""
        level = console._get_log_level(raw)
        assert isinstance(level, int)

    @pytest.mark.parametrize("raw", ["WARN", "", "nope", "info"])
    def test_constructing_with_one_does_not_raise(self, raw):
        """The constructor path, which is what a config actually drives."""
        assert ConsoleLogger(verbosity=raw) is not None


class TestVerbosityRoundTrip:
    """What the property gives back, since callers read it."""

    def test_the_property_returns_what_was_set(self):
        assert ConsoleLogger(verbosity="INFO:2").verbosity == "INFO:2"

    @pytest.mark.parametrize(
        ("raw", "stored"),
        [
            ("info:2", "INFO:2"),
            ("debug", "DEBUG"),
            ("warn", "WARN"),
            # Upper-cased but NOT stripped, unlike `_parse_verbosity`, which
            # strips. The asymmetry is the measured behaviour; a
            # reconstruction that normalised both the same way, either way,
            # would differ here.
            (" info ", " INFO "),
        ],
    )
    def test_the_stored_string_is_upper_cased_and_not_stripped(self, raw, stored):
        assert ConsoleLogger(verbosity=raw).verbosity == stored

    def test_the_setter_takes_effect_on_the_level(self):
        console = ConsoleLogger(verbosity="INFO")
        console.verbosity = "ERROR"
        assert console._get_log_level(console.verbosity) == logging.ERROR

    def test_the_setter_survives_a_name_it_does_not_know(self):
        console = ConsoleLogger(verbosity="INFO")
        console.verbosity = "NOPE:x"
        assert isinstance(console._get_log_level(console.verbosity), int)
