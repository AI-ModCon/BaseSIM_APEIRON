#!/usr/bin/env python3
"""Behaviour probe: apeiron.config.configuration, off the pinned set.

Emits {case: observable} as JSON. Run by behaviour_probe.py once against the
source tree and once against the claim's regrown implementation; the harness
diffs the two.

Where this aims. The config layer's criteria are its strongest (A33 puts its
kept-ratio at 69%, the second highest) and its tests assert the dataclass
defaults directly. What they exercise far less is the machinery BETWEEN a
TOML file and a Config: how `--set` values are typed, how an override merges
onto a nested section, what `APP_*` does and who wins when both are present,
and what happens to a key nobody recognises. A20 already lives in that gap
(a misspelled `aggregation` silently becomes the mean) and A18 next door (no
`[logging]` section means a run talks to wandb), so it is the right place to
look for more.

Every observable is reduced to a string, exceptions included -- the type and
message, not the traceback, since a reconstruction that raises a different
error for the same bad input has diverged in a way a caller can see, while
line numbers are noise.
"""
import json
import os
import pathlib
import sys
import tempfile

from apeiron.config.configuration import (  # noqa: E402
    Config, build_config, deep_update, env_overrides, get_available_device,
    kv_to_nested, load_toml,
)

#: A config that actually builds. The first version of this probe omitted
#: `seed` and `device`, which `Config` requires with no default, so EVERY
#: build case raised the same constructor TypeError on both sides and
#: "agreed" -- a third of the probe was comparing one error message to
#: itself. Same failure shape as the weak held-out streams in A37: an
#: instrument can report agreement because it never reached the behaviour.
#: `[continual_learning]` is deliberately left out, because whether an
#: omitted optional section is filled with defaults is one of the questions.
MINIMAL = """
seed = 1
device = "cpu"

[model]
name = "MNIST_CNN"
pretrained_path = ""

[data]
name = "mnist"
path = "/tmp/data"

[train]
batch_size = 4
num_workers = 0
init_lr = 0.001

[drift_detection]
detector_name = "ADWINDetector"
detection_interval = 10
"""


#: Paths that differ between the two runs by construction -- each side gets
#: its own temporary directory -- and would otherwise read as a divergence.
#: The first version of this probe reported exactly one "difference",
#: `load_toml[missing]`, which was two different tmpdir names inside the same
#: FileNotFoundError. A probe that cries wolf once teaches you to skim.
_SCRUB: list[str] = []


def scrub(text: str) -> str:
    for path in _SCRUB:
        text = text.replace(path, "<tmp>")
    return text


def show(value) -> str:
    """One canonical string for a value, so the two sides are comparable."""
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=repr)
    if isinstance(value, (list, tuple)):
        return json.dumps([repr(v) for v in value])
    return repr(value)


def attempt(fn, *args, **kwargs) -> str:
    try:
        return show(fn(*args, **kwargs))
    except BaseException as exc:                     # noqa: BLE001 — probed
        return show(exc)


def config_shape(cfg: Config) -> str:
    """A Config reduced to every scalar it carries, section by section."""
    out = {}
    for field in sorted(cfg.__dataclass_fields__):
        section = getattr(cfg, field)
        if hasattr(section, "__dataclass_fields__"):
            for sub in sorted(section.__dataclass_fields__):
                out[f"{field}.{sub}"] = repr(getattr(section, sub))
        else:
            out[field] = repr(section)
    return json.dumps(out, sort_keys=True)


def main() -> int:
    out = {}

    # --- 1. how a --set value is typed -------------------------------------
    # The tests pin a handful of overrides. These are the forms a user will
    # actually type, and each one is a question: is "true" the string or the
    # boolean, is "1e-3" a float, does "none" clear a value, is "a,b" a list?
    for raw in ("train.batch_size=8", "train.init_lr=1e-3",
                "train.init_lr=.5", "seed=0", "seed=-1", "seed=007",
                "multi_gpu=true", "multi_gpu=True", "multi_gpu=TRUE",
                "multi_gpu=false", "multi_gpu=1", "multi_gpu=0",
                "multi_gpu=yes", "logging.backend=none",
                "logging.experiment_name=none", "logging.experiment_name=null",
                "logging.experiment_name=", "model.name=a=b",
                "drift_detection.ensemble_detectors=a,b",
                "drift_detection.ensemble_detectors=[a,b]",
                "drift_detection.ph_threshold=50", "verbosity=DEBUG",
                "a.b.c.d=1", "nodots=1", "=1", "train.batch_size",
                "train.batch_size=0x10", "train.batch_size=1_000",
                "train.init_lr=nan", "train.init_lr=inf",
                "device=cpu", "  spaced.key  =  2  "):
        out[f"kv_to_nested[{raw!r}]"] = attempt(kv_to_nested, [raw])

    # repeated and interacting keys: last-wins, or merge?
    out["kv_to_nested[repeat]"] = attempt(
        kv_to_nested, ["train.batch_size=1", "train.batch_size=2"])
    out["kv_to_nested[sibling]"] = attempt(
        kv_to_nested, ["train.batch_size=1", "train.num_workers=2"])
    out["kv_to_nested[parent-then-child]"] = attempt(
        kv_to_nested, ["train=1", "train.batch_size=2"])
    out["kv_to_nested[child-then-parent]"] = attempt(
        kv_to_nested, ["train.batch_size=2", "train=1"])
    out["kv_to_nested[empty]"] = attempt(kv_to_nested, [])

    # --- 2. how an override merges ----------------------------------------
    # deep_update decides whether an override replaces a section or merges
    # into it, and what a None or a type change does. None of this is pinned.
    base = {"train": {"batch_size": 1, "num_workers": 2},
            "logging": {"backend": "none"}, "seed": 1}
    for name, patch in (
            ("nested-partial", {"train": {"batch_size": 9}}),
            ("nested-new-key", {"train": {"unknown": 9}}),
            ("scalar-over-dict", {"train": 9}),
            ("dict-over-scalar", {"seed": {"a": 1}}),
            ("none-over-scalar", {"seed": None}),
            ("none-over-dict", {"train": None}),
            ("empty-dict-over-dict", {"train": {}}),
            ("new-section", {"brand_new": {"k": 1}}),
            ("list-replace", {"train": {"batch_size": [1, 2]}}),
    ):
        out[f"deep_update[{name}]"] = attempt(deep_update, dict(base), patch)
    # does it mutate its first argument, and is the result the same object?
    probe_base = {"train": {"batch_size": 1}}
    result = attempt(deep_update, probe_base, {"train": {"batch_size": 5}})
    out["deep_update[mutates-input]"] = show(probe_base)
    out["deep_update[returns]"] = result

    # --- 3. APP_* overrides ------------------------------------------------
    # Nesting convention, case handling, typing, and whether an empty value
    # is an override or absent.
    saved = {k: v for k, v in os.environ.items() if k.startswith("APP_")}
    for key in list(saved):
        del os.environ[key]
    try:
        for name, pairs in (
                ("flat", {"APP_SEED": "5"}),
                ("lower", {"APP_seed": "5"}),
                ("nested-double", {"APP_TRAIN__BATCH_SIZE": "7"}),
                ("nested-dot", {"APP_TRAIN.BATCH_SIZE": "7"}),
                ("nested-single", {"APP_TRAIN_BATCH_SIZE": "7"}),
                ("bool", {"APP_MULTI_GPU": "true"}),
                ("empty", {"APP_SEED": ""}),
                ("not-prefixed", {"SEED": "5"}),
                ("prefix-only", {"APP_": "5"}),
        ):
            for k, v in pairs.items():
                os.environ[k] = v
            out[f"env_overrides[{name}]"] = attempt(env_overrides)
            for k in pairs:
                os.environ.pop(k, None)
    finally:
        for k, v in saved.items():
            os.environ[k] = v

    # --- 4. load_toml edges ------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        _SCRUB.append(str(d))
        cases = {
            "missing": d / "nope.toml",
            "empty": d / "empty.toml",
            "malformed": d / "bad.toml",
            "not-toml": d / "bin.toml",
            "duplicate-key": d / "dup.toml",
        }
        cases["empty"].write_text("")
        cases["malformed"].write_text("[model\nname = 1")
        cases["not-toml"].write_text("\x00\x01binary")
        cases["duplicate-key"].write_text("[model]\nname='a'\nname='b'\n")
        for name, path in cases.items():
            out[f"load_toml[{name}]"] = attempt(load_toml, path)

    # --- 5. build_config end to end ---------------------------------------
    # The optional sections are where A18 lives: what a run does when
    # [logging] is simply absent. And an unrecognised key is where A20 lives.
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        _SCRUB.append(str(d))

        def built(name: str, toml: str, extra=()):
            path = d / f"{name}.toml"
            path.write_text(toml)
            argv = ["--config", str(path)] + list(extra)
            try:
                return config_shape(build_config(argv))
            except BaseException as exc:             # noqa: BLE001 — probed
                return show(exc)

        out["build[minimal]"] = built("minimal", MINIMAL)
        out["build[no-logging-section]"] = built("nolog", MINIMAL)
        out["build[empty-logging-section]"] = built(
            "emptylog", MINIMAL + "\n[logging]\n")
        out["build[no-cl-section]"] = built("nocl", MINIMAL)
        out["build[empty-cl-section]"] = built(
            "emptycl", MINIMAL + "\n[continual_learning]\n")
        out["build[unknown-top-key]"] = built(
            "unktop", MINIMAL + "\nnot_a_real_key = 1\n")
        out["build[unknown-section]"] = built(
            "unksec", MINIMAL + "\n[not_a_real_section]\nk = 1\n")
        out["build[unknown-key-in-section]"] = built(
            "unkkey", MINIMAL + "\n[logging]\nbackend='none'\nnope=1\n")
        # A20's neighbourhood: a misspelling that should not be silent
        out["build[misspelled-aggregation]"] = built(
            "misagg", MINIMAL.replace("detection_interval = 10",
                                      "detection_interval = 10\naggregation = 'mien'"))
        out["build[misspelled-detector]"] = built(
            "misdet", MINIMAL.replace("ADWINDetector", "ADWINDetecter"))
        out["build[wrong-type-scalar]"] = built(
            "wrongtype", MINIMAL.replace("batch_size = 4", "batch_size = 'four'"))
        out["build[missing-required-section]"] = built(
            "noreq", MINIMAL.replace("[drift_detection]", "[dd_typo]"))
        out["build[set-overrides-file]"] = built(
            "ovr", MINIMAL, ["--set", "train.batch_size=99"])
        out["build[set-unknown-key]"] = built(
            "ovrunk", MINIMAL, ["--set", "train.not_real=1"])
        out["build[set-nested-new-section]"] = built(
            "ovrnew", MINIMAL, ["--set", "brand.new=1"])
        out["build[no-config-arg]"] = attempt(build_config, [])

        # precedence: file < env < --set, or some other order?
        path = d / "prec.toml"
        path.write_text(MINIMAL)
        os.environ["APP_SEED"] = "222"
        try:
            out["build[env-vs-file]"] = attempt(
                lambda: config_shape(build_config(["--config", str(path)])))
            out["build[set-vs-env]"] = attempt(
                lambda: config_shape(build_config(
                    ["--config", str(path), "--set", "seed=333"])))
        finally:
            os.environ.pop("APP_SEED", None)

    # --- 6. device selection ----------------------------------------------
    for multi in (False, True):
        out[f"get_available_device[multi_gpu={multi}]"] = attempt(
            lambda m=multi: str(get_available_device(m)))

    json.dump({k: scrub(v) for k, v in out.items()}, sys.stdout,
              sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
