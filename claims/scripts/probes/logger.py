#!/usr/bin/env python3
"""Behaviour probe: apeiron.logger, off the pinned set.

The logger regrew at 46% with zero variance across draws (A33) and is the
layer where A27 sits -- the question of whether its explicit delegation to a
backend is a requirement or a freedom. It also holds A18: `Logger()` defaults
its backend to `"wandb"`, so a run with no `[logging]` section in its config
talks to a metrics service.

Where this aims:

  verbosity parsing   `_parse_verbosity` and `_get_log_level` turn a string
                      like "INFO:3" into a level number, with a documented
                      tier scheme (INFO:n -> 20-n). Pure functions over
                      strings, and the layer's most reachable surface: a
                      typo in a config lands here. Probed across malformed,
                      out-of-range, cased and padded input.
  stage bookkeeping   stage / step / get_stage_step decide how a metric gets
                      named and counted. Nothing pins what happens when you
                      log before naming a stage, name an unknown stage, or
                      name the same stage twice.
  the metrics table   what actually lands in the dataframe and the CSV --
                      column names, prefixing, step numbering -- which is
                      the artifact every downstream analysis reads.

The backend is `"none"` throughout except where the default is the thing
being observed, so no probe case contacts a service. Timestamps are dropped
from compared output: the CSV carries wall-clock columns, and comparing
those would make the probe flap (the A38 lesson).
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from _fixtures import attempt, emit, make_cfg, show, tmpdir  # noqa: E402

from apeiron.logger.console_logger import ConsoleLogger  # noqa: E402
from apeiron.logger.logger import Logger  # noqa: E402

#: Columns whose values are clocks or paths, not behaviour.
VOLATILE = ("time", "timestamp", "date", "_runtime", "wall", "elapsed")


def table(logger) -> str:
    """The metrics dataframe reduced to columns and rows, clocks removed."""
    frame = logger.to_dataframe()
    if frame is None:
        return "<no dataframe>"
    keep = [c for c in frame.columns
            if not any(v in str(c).lower() for v in VOLATILE)]
    rows = frame[keep].to_dict(orient="records") if keep else []
    return json.dumps({"columns": sorted(str(c) for c in keep),
                       "rows": [{k: show(v) for k, v in sorted(r.items())}
                                for r in rows]}, sort_keys=True)


def main() -> int:
    out = {}

    # --- 1. verbosity parsing, the surface a config typo reaches ----------
    console = ConsoleLogger(verbosity="INFO")
    for raw in ("INFO", "info", " info ", "INFO:0", "INFO:1", "INFO:9",
                "INFO:10", "INFO:99", "INFO:-1", "INFO:", "INFO:abc",
                "INFO:1:2", "DEBUG", "WARNING", "ERROR", "CRITICAL",
                "TRACE", "", ":", ":3", "debug:2", "CRITICAL:5",
                "WARN", "NOTSET"):
        out[f"_parse_verbosity[{raw!r}]"] = attempt(
            console._parse_verbosity, raw)
        out[f"_get_log_level[{raw!r}]"] = attempt(console._get_log_level, raw)

    out["ConsoleLogger[verbosity-roundtrip]"] = attempt(
        lambda: ConsoleLogger(verbosity="info:2").verbosity)
    out["ConsoleLogger[default-verbosity]"] = attempt(
        lambda: ConsoleLogger().verbosity)
    out["ConsoleLogger[step-callback-default]"] = attempt(
        lambda: ConsoleLogger()._get_step())

    # --- 2. the default backend, which is A18 -----------------------------
    # Observed, not exercised: constructing the default would enable a real
    # wandb logger. What is asserted is which backend a bare Logger reports.
    out["Logger[default-backend]"] = attempt(
        lambda: Logger(backend="none").backend)
    out["Logger[default-backend-signature]"] = attempt(
        lambda: __import__("inspect").signature(Logger.__init__)
        .parameters["backend"].default)
    for backend in ("none", "nope", "", None, "NONE", "Wandb"):
        out[f"Logger[backend={backend!r}]"] = attempt(
            lambda b=backend: Logger(backend=b).backend)

    # --- 3. stage and step bookkeeping ------------------------------------
    def fresh(**kwargs):
        return Logger(backend="none", **kwargs)

    out["stage[initial-current]"] = attempt(lambda: fresh().current_stage())
    out["stage[initial-step]"] = attempt(lambda: fresh().step)
    for stage in ("eval", "drift", "cl", "EVAL", "unknown", "", None):
        out[f"stage[set={stage!r}]"] = attempt(
            lambda s=stage: (lambda lg: (lg.stage(s), show(lg.current_stage()))
                             [1])(fresh()))
        out[f"stage[step-after={stage!r}]"] = attempt(
            lambda s=stage: (lambda lg: (lg.stage(s), lg.step)[1])(fresh()))
    out["stage[same-twice]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.stage("eval"),
                             show(lg.get_stage_step("eval")))[2])(fresh()))
    out["stage[get-unset]"] = attempt(
        lambda: fresh().get_stage_step("eval"))
    out["stage[get-unknown]"] = attempt(
        lambda: fresh().get_stage_step("nope"))

    # --- 4. what lands in the metrics table -------------------------------
    out["log[before-any-stage]"] = attempt(
        lambda: (lambda lg: (lg.log({"loss": 1.0}), table(lg))[1])(fresh()))
    out["log[with-stage]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({"loss": 1.0}),
                             table(lg))[2])(fresh()))
    out["log[prefix-false]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"),
                             lg.log({"loss": 1.0}, prefix=False),
                             table(lg))[2])(fresh()))
    out["log[explicit-step]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({"loss": 1.0}, step=42),
                             table(lg))[2])(fresh()))
    out["log[increment-false]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"),
                             lg.log({"a": 1.0}, increment=False),
                             lg.log({"b": 2.0}, increment=False),
                             table(lg))[3])(fresh()))
    out["log[two-stages]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({"loss": 1.0}),
                             lg.stage("cl"), lg.log({"loss": 2.0}),
                             table(lg))[4])(fresh()))
    out["log[empty-metrics]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({}), table(lg))[2])(fresh()))
    out["log[none-value]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({"loss": None}),
                             table(lg))[2])(fresh()))
    out["log[nested-value]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({"d": {"x": 1}}),
                             table(lg))[2])(fresh()))
    out["log[nan-value]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"),
                             lg.log({"loss": float("nan")}), table(lg))[2])(fresh()))
    out["log[duplicate-key-two-calls]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({"loss": 1.0}),
                             lg.log({"loss": 2.0}), table(lg))[3])(fresh()))
    out["to_dataframe[nothing-logged]"] = attempt(
        lambda: show(fresh().to_dataframe()))

    # --- 5. the CSV, which is the artifact analyses read ------------------
    def csv_rows(logger, path):
        written = logger.to_csv(path)
        if written is None or not path.exists():
            return show(written)
        lines = path.read_text().strip().splitlines()
        header = lines[0].split(",") if lines else []
        keep = [i for i, c in enumerate(header)
                if not any(v in c.lower() for v in VOLATILE)]
        return json.dumps({
            "header": [header[i] for i in keep],
            "nrows": len(lines) - 1,
            "first": ([lines[1].split(",")[i] for i in keep]
                      if len(lines) > 1 else None)})

    d = tmpdir()
    out["to_csv[after-logging]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({"loss": 1.0}),
                             csv_rows(lg, d / "a.csv"))[2])(fresh()))
    out["to_csv[nothing-logged]"] = attempt(
        lambda: csv_rows(fresh(), d / "b.csv"))
    out["to_csv[no-path-configured]"] = attempt(lambda: fresh().to_csv())
    out["to_csv[nested-dir]"] = attempt(
        lambda: (lambda lg: (lg.stage("eval"), lg.log({"loss": 1.0}),
                             csv_rows(lg, d / "deep" / "c.csv"))[2])(fresh()))
    out["finish[no-backend]"] = attempt(
        lambda: show(fresh(csv_path=d / "f.csv").finish()))
    out["finish[save-csv-false]"] = attempt(
        lambda: show(fresh(csv_path=d / "g.csv").finish(save_csv=False)))
    out["finish[twice]"] = attempt(
        lambda: (lambda lg: (lg.finish(), show(lg.finish()))[1])(
            fresh(csv_path=d / "h.csv")))

    # --- 6. odds and ends callers touch ----------------------------------
    out["url[backend-none]"] = attempt(lambda: show(fresh().url))
    out["verbosity[property]"] = attempt(lambda: fresh().verbosity)
    out["verbosity[setter]"] = attempt(
        lambda: (lambda lg: (setattr(lg, "verbosity", "DEBUG"),
                             lg.verbosity)[1])(fresh()))
    out["verbosity[setter-garbage]"] = attempt(
        lambda: (lambda lg: (setattr(lg, "verbosity", "NOPE:x"),
                             lg.verbosity)[1])(fresh()))
    out["init[with-cfg]"] = attempt(
        lambda: (lambda lg: (lg.init(make_cfg()), show(lg.backend))[1])(fresh()))
    out["save[missing-file]"] = attempt(
        lambda: show(fresh().save(d / "does-not-exist.txt")))

    emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
