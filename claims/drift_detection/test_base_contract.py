"""What `BaseDriftDetector` promises before any detector implements it.

Two things, both of which a regrowth of this layer left out, and both of
which the upstream suite asserts from another room (see R33 in
claims/LOG.md and the note at the bottom).

**An unimplemented hook refuses, and names itself.**

    raise NotImplementedError(f"Method not implemented for {self.name}")

on both `update` (base.py:92) and `reset` (base.py:97). The decorator
makes the class unconstructible without them; the body is what a
subclass reaches when it delegates upward -- which happens here for real:
`EnsembleDetector` wraps sub-detectors, and `ModelPerformanceDetector`
implements `update` but has nothing to reset. The message carries
`self.name`, which matters because the ensemble holds several detectors
and a bare `NotImplementedError` from inside it would not say which one.

**A detector prints as its class and its name.**

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

`base.py:171`. The regrowth has a `__repr__` on `DriftSignal` and none on
`BaseDriftDetector`, so a detector falls back to
`<...Bare object at 0x14b999be0>`. That reaches people: the monitor logs
its detector, and an address is not a fact about a run. It also reaches
machines -- an address differs between processes, so any log line or
record carrying a detector repr stops being comparable across runs, which
is the one property the pinned-run claim in this repository depends on.

Pinned as the shape rather than the exact string where that is enough:
the repr must contain the class name and the detector's name and must not
contain an address. The refusal is pinned as type plus the name appearing
in the message, not as the sentence.

WHY THIS FILE IS HERE RATHER THAN UPSTREAM'S BEING ENOUGH.
`tests/test_contracts_and_branches.py` asserts all of it and is pinned
into `claims/monitor`, where `detectors/base.py` is *supplied* by this
component. So the monitor's producer wrote a base detector that passes,
the monitor's gate went green, and the assembly then shipped THIS room's
base detector, which was never asked. Same information, moved to the room
that owns the module.
"""

from __future__ import annotations

import re

import pytest

from apeiron.drift_detection.detectors.base import BaseDriftDetector

#: The hooks whose `src/` bodies raise.
REFUSING_HOOKS = ("update", "reset")

#: An object address, which must not appear in a detector's repr.
ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")


class Bare(BaseDriftDetector):
    """A detector that implements the hooks by delegating upward.

    The same shape `EnsembleDetector` has: concrete enough to construct,
    so the base class's bodies are reachable.

    Declared at module level deliberately. Nested in a fixture its
    `__qualname__` becomes `bare.<locals>.Bare`, which contains the
    string "bare" -- so a default `object.__repr__` would satisfy "the
    repr carries the detector's name" by accident. At module level the
    qualname is just `Bare` and that case discriminates.
    """

    def update(self, value, **kwargs):
        return super().update(value, **kwargs)

    def reset(self):
        return super().reset()


@pytest.fixture()
def bare():
    return Bare(name="watcher")


class TestTheHooksAreAbstract:
    @pytest.mark.parametrize("hook", REFUSING_HOOKS)
    def test_the_hook_is_declared_abstract(self, hook):
        assert hook in BaseDriftDetector.__abstractmethods__, (
            f"`{hook}` is not abstract; a detector that forgets it would "
            f"construct and fail at the first metric value instead"
        )


class TestAnUnimplementedHookRefuses:
    @pytest.mark.parametrize("hook", REFUSING_HOOKS)
    def test_calling_up_raises(self, bare, hook):
        """An empty body returns `None`. `update`'s caller does
        `drift_signal.drift_detected` on the result, so `None` becomes an
        AttributeError inside the monitor's check loop rather than a
        refusal at the point of the mistake."""
        with pytest.raises(NotImplementedError):
            if hook == "update":
                bare.update(1.0)
            else:
                bare.reset()

    @pytest.mark.parametrize("hook", REFUSING_HOOKS)
    def test_the_refusal_names_the_detector(self, bare, hook):
        """`f"Method not implemented for {self.name}"`. An ensemble holds
        several detectors and the message is how an operator learns which
        one refused."""
        with pytest.raises(NotImplementedError) as caught:
            if hook == "update":
                bare.update(1.0)
            else:
                bare.reset()
        assert "watcher" in str(caught.value), (
            f"the refusal from `{hook}` reads {str(caught.value)!r} and "
            f"does not name the detector; inside an ensemble that leaves "
            f"the operator with no way to tell which member refused"
        )


class TestADetectorPrintsAsItself:
    def test_the_repr_carries_the_class_name(self, bare):
        assert type(bare).__name__ in repr(bare), (
            f"repr is {repr(bare)!r} and does not name the class"
        )

    def test_the_repr_carries_the_detector_name(self, bare):
        assert "watcher" in repr(bare), (
            f"repr is {repr(bare)!r} and does not carry `name`"
        )

    def test_the_repr_is_not_an_address(self, bare):
        """The whole finding. A default repr is
        `<module.Bare object at 0x14b999be0>`: it identifies nothing
        about the run and differs between processes, so a record that
        carries one is not comparable across runs."""
        text = repr(bare)
        assert not ADDRESS.search(text), (
            f"repr is {text!r}; the address makes every log line and "
            f"record holding it differ between runs"
        )

    def test_the_repr_is_stable_across_instances(self, bare):
        """Two detectors with the same name print identically, which is
        what "comparable across runs" means in practice."""
        other = type(bare)(name="watcher")
        assert repr(other) == repr(bare)

    def test_two_names_print_differently(self, bare):
        """Guard: a repr that ignored `name` would satisfy everything
        above except this."""
        other = type(bare)(name="other")
        assert repr(other) != repr(bare)


class TestTheFixtureReachesTheBodies:
    """Guard: the refusals must come from the base class, not from the
    fixture failing to construct."""

    def test_the_fixture_is_concrete(self, bare):
        assert not type(bare).__abstractmethods__

    def test_the_name_round_trips(self, bare):
        assert bare.name == "watcher", (
            "the base class did not keep `name`, so every assertion above "
            "about naming is testing the fixture"
        )
