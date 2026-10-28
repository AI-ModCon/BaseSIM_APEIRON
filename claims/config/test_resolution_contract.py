"""How a configuration resolves: the device, the merge, and the overrides.

This is the layer every other one composes on, so a divergence here moves
numbers in every run. Three surfaces are unpinned and all three diverged in
reconstruction.

**`get_available_device`.** The fallback order is CUDA, then Apple MPS,
then CPU (`configuration.py:48-68`). One rebuild dropped the MPS branch and
returned `cpu` on a host where `torch.backends.mps.is_available()` is True.
That is not only slower: MPS and CPU kernels do not produce bit-identical
floating point, so the same config runs a numerically different training
job. Nothing in the suite asked which device a host should get.

The assertions below compare against what the installed torch reports
rather than against a hard-coded name, so they hold on a CUDA box, an
Apple laptop and a CPU-only runner, and they still catch a reconstruction
that omits a branch -- which is the portable way to pin a decision whose
correct answer depends on the host.

**`deep_update`.** It mutates its first argument and returns it, but
recurses through `dict(x[k])` -- a copy -- so the top-level dict is updated
in place while nested dicts are REPLACED rather than mutated. Both halves
of that are observable to a caller holding a reference, and the
reconstructions picked different halves.

**`kv_to_nested` and `env_overrides`.** These parse `--set k=v` and `APP_*`,
so they are the surface a typo or a shell quirk reaches. The measured
behaviour is more surprising than the docstrings suggest: keys are not
stripped, an empty key is accepted, and a dotted path nests while a double
underscore does not.
"""

from __future__ import annotations

import pytest
import torch

from apeiron.config.configuration import (
    deep_update,
    env_overrides,
    get_available_device,
    kv_to_nested,
)


def _mps_available() -> bool:
    try:
        return bool(hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available())
    except Exception:            # noqa: BLE001 — same guard the source uses
        return False


def _expected_device_type() -> str:
    """The branch the source documents, re-derived from the live backends.

    `multi_gpu` is not a parameter here because it does not change the
    device TYPE: the source returns `cuda` for multi_gpu and `cuda:0`
    otherwise, and both report `.type == "cuda"`. The index distinction is
    asserted separately, where a CUDA host exists to assert it on.
    """
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


class TestDeviceSelection:
    """The fallback order, asserted without hard-coding a host."""

    @pytest.mark.parametrize("multi_gpu", [False, True])
    def test_it_returns_a_torch_device(self, multi_gpu):
        assert isinstance(get_available_device(multi_gpu=multi_gpu),
                          torch.device)

    @pytest.mark.parametrize("multi_gpu", [False, True])
    def test_it_picks_the_documented_branch(self, multi_gpu):
        assert get_available_device(multi_gpu=multi_gpu).type == \
            _expected_device_type()

    @pytest.mark.skipif(torch.cuda.is_available(),
                        reason="the CPU/MPS precedence only applies without CUDA")
    def test_mps_is_preferred_over_cpu_when_present(self):
        """The branch one rebuild dropped. On a host with MPS the answer
        must be MPS, and on a host without it must be CPU -- so this case
        is meaningful either way rather than skipped on half the fleet."""
        expected = "mps" if _mps_available() else "cpu"
        assert get_available_device().type == expected

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="needs a CUDA host to observe the index")
    def test_multi_gpu_decides_whether_an_index_is_pinned(self):
        """`multi_gpu=True` leaves the device unindexed so the caller can
        spread over all of them; `False` restricts to the chosen one."""
        assert get_available_device(multi_gpu=True).index is None
        assert get_available_device(multi_gpu=False).index == 0

    def test_it_never_raises_without_a_gpu_toolchain(self):
        """Documented: "Never raises if nvidia-smi is missing"."""
        for multi_gpu in (False, True):
            get_available_device(multi_gpu=multi_gpu)


class TestDeepUpdateSemantics:
    """In-place at the top level, replacing underneath."""

    def test_it_returns_the_dict_it_was_given(self):
        x = {"seed": 1}
        assert deep_update(x, {"seed": 2}) is x

    def test_it_mutates_the_caller_s_dict(self):
        x = {"train": {"batch_size": 1, "num_workers": 2}, "seed": 1}
        deep_update(x, {"train": {"batch_size": 5}})
        assert x == {"train": {"batch_size": 5, "num_workers": 2}, "seed": 1}

    def test_a_nested_dict_is_replaced_rather_than_mutated(self):
        """`deep_update(dict(x[k]), v)` copies, so a caller holding the old
        inner dict does not see the change. A reconstruction that recursed
        into `x[k]` directly would mutate it and pass every other case."""
        x = {"train": {"batch_size": 1, "num_workers": 2}}
        inner = x["train"]
        deep_update(x, {"train": {"batch_size": 5}})
        assert x["train"] is not inner
        assert inner == {"batch_size": 1, "num_workers": 2}

    def test_a_scalar_replaces_a_dict_and_a_dict_replaces_a_scalar(self):
        x = {"a": {"b": 1}, "c": 2}
        deep_update(x, {"a": 9, "c": {"d": 3}})
        assert x == {"a": 9, "c": {"d": 3}}

    def test_an_empty_mapping_changes_nothing(self):
        x = {"train": {"batch_size": 1}}
        deep_update(x, {"train": {}})
        assert x == {"train": {"batch_size": 1}}

    def test_a_list_replaces_rather_than_merges(self):
        x = {"a": [1, 2]}
        deep_update(x, {"a": [3]})
        assert x == {"a": [3]}

    def test_none_replaces_a_scalar(self):
        x = {"seed": 1}
        deep_update(x, {"seed": None})
        assert x == {"seed": None}


class TestKvToNested:
    """`--set k=v`, exactly as parsed."""

    @pytest.mark.parametrize(
        ("item", "expected"),
        [
            ("train.batch_size=2", {"train": {"batch_size": 2}}),
            ("a.b.c=3", {"a": {"b": {"c": 3}}}),
            ("seed=1", {"seed": 1}),
            # Values are coerced; `none` is NOT a null.
            ("x=true", {"x": True}),
            ("y=1.5", {"y": 1.5}),
            ("z=none", {"z": "none"}),
            ("q=[1,2]", {"q": [1, 2]}),
            ("s=hello", {"s": "hello"}),
        ],
    )
    def test_the_parse_table(self, item, expected):
        assert kv_to_nested([item]) == expected

    def test_keys_keep_their_whitespace_while_values_do_not(self):
        """Surprising, unasserted, and both rebuilds normalised it away.
        A shell that leaves a space after `--set` therefore produces a
        key that no dataclass field matches, and the failure surfaces
        later as an unexpected-keyword TypeError."""
        assert kv_to_nested(["  spaced.key  =  2  "]) == {
            "  spaced": {"key  ": 2}
        }

    def test_an_empty_key_is_accepted(self):
        assert kv_to_nested(["=1"]) == {"": 1}

    def test_a_scalar_then_a_path_through_it_raises(self):
        """`--set train=1 --set train.batch_size=2` asks to index into an
        int. One rebuild silently accepted it and dropped the first
        assignment."""
        with pytest.raises(TypeError):
            kv_to_nested(["train=1", "train.batch_size=2"])

    def test_later_items_win(self):
        assert kv_to_nested(["seed=1", "seed=2"]) == {"seed": 2}


class TestEnvOverrides:
    """`APP_*`, exactly as parsed."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for key in list(__import__("os").environ):
            if key.startswith("APP_"):
                monkeypatch.delenv(key, raising=False)

    def test_a_name_is_lowercased(self, monkeypatch):
        monkeypatch.setenv("APP_SEED", "9")
        assert env_overrides() == {"seed": 9}

    def test_a_dotted_name_nests(self, monkeypatch):
        monkeypatch.setenv("APP_train.batch_size", "4")
        assert env_overrides() == {"train": {"batch_size": 4}}

    def test_a_double_underscore_does_NOT_nest(self, monkeypatch):
        """The convention a reader is likely to assume, and not the one
        implemented: the key stays literal."""
        monkeypatch.setenv("APP_TRAIN__BATCH_SIZE", "3")
        assert env_overrides() == {"train__batch_size": 3}

    def test_a_single_underscore_is_kept_in_the_key(self, monkeypatch):
        monkeypatch.setenv("APP_A_B", "2")
        assert env_overrides() == {"a_b": 2}

    def test_a_bare_prefix_yields_an_empty_key(self, monkeypatch):
        """`APP_=5` is accepted and produces `{"": 5}` rather than being
        ignored. One rebuild returned `{}`."""
        monkeypatch.setenv("APP_", "5")
        assert env_overrides() == {"": 5}

    def test_names_without_the_prefix_are_ignored(self, monkeypatch):
        monkeypatch.setenv("NOPE", "1")
        assert env_overrides() == {}
