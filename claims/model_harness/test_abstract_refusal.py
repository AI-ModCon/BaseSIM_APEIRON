"""An unimplemented hook refuses; it does not quietly return nothing.

`BaseModelHarness` gives six methods a body that raises, and marks five
of them `@abstractmethod`:

    line  72  get_optimizer           abstract
    line  97  update_data_stream      abstract
    line 105  get_stream_dataloader   abstract
    line 115  get_hist_dataloaders    abstract
    line 128  get_criterion           abstract
    line 123  get_train_dataloaders   NOT abstract -- refuses if inherited

The decorator and the body do different jobs, and `get_train_dataloaders`
is the proof: it has one without the other. `@abstractmethod` stops
`BaseModelHarness()` and any subclass that overrides none of them from
being instantiated at all. The body is for the subclass that *does*
override -- which is every real harness, since the decorator forces it --
and then calls up:

    def get_hist_dataloaders(self):
        if self._hist is None:
            return super().get_hist_dataloaders()   # None, or a refusal?

`src/` refuses. A harness that only sometimes has historical data, or
that delegates an optional hook upward, gets an exception naming the
problem instead of a `None` that travels until something unrelated fails
on it -- and `None` travels a long way here: `get_hist_dataloaders`'s
result is unpacked into two loaders and passed to the trainer, so the
failure would surface inside a CL round.

A regrowth of this layer left every body empty -- decorator present,
`...` beneath it -- which is the conventional way to write an ABC and is
what the type checker asks for. Nothing in this directory's criteria
distinguished the two, so the choice fell to convention.

WHY THE OTHER ROOMS DID NOT CATCH IT, which is the part worth recording.
The upstream suite does assert this, in
`tests/test_contracts_and_branches.py`, and that file IS pinned -- into
`claims/monitor`. In the monitor's room `torch_model_harness.py` is
supplied by this component, so the monitor's producer wrote a harness that
passes it, and the monitor's gate went green. The assembly then takes the
harness from THIS room, where the constraint was never stated, and ships
a harness that fails it. A constraint satisfied in a room that does not
ship the file is not a constraint on the file that ships (claims/LOG.md,
R33).

So this file is not new information about APEIRON. It is the same
information, moved to the room that owns the module.
"""

from __future__ import annotations

import inspect

import pytest

from apeiron.model.torch_model_harness import BaseModelHarness

#: Every hook whose `src/` body raises.
REFUSING_HOOKS = (
    "get_optimizer",
    "update_data_stream",
    "get_stream_dataloader",
    "get_hist_dataloaders",
    "get_train_dataloaders",
    "get_criterion",
)

#: Of those, the ones `src/` also marks `@abstractmethod`. MEASURED, not
#: assumed: `get_train_dataloaders` refuses in its body and is NOT
#: abstract, so a harness may omit it and inherit the refusal. The first
#: version of this file asserted all six were abstract and failed against
#: `src/` on exactly that one -- the two properties are independent and
#: this hook has one without the other.
ABSTRACT_HOOKS = (
    "get_optimizer",
    "update_data_stream",
    "get_stream_dataloader",
    "get_hist_dataloaders",
    "get_criterion",
)


@pytest.fixture()
def delegating(default_cfg, tiny_model):
    """A concrete harness that overrides every hook by calling up.

    This is the shape that reaches the bodies: `@abstractmethod` is
    satisfied by the override existing, and `super()` runs the base
    class's body. A harness that did not override could not be
    instantiated, so the bodies would be unreachable and unassertable.
    """
    namespace = {
        name: (lambda self, *a, _n=name, **k:
               getattr(super(type(self), self), _n)(*a, **k))
        for name in REFUSING_HOOKS
    }
    Delegating = type("Delegating", (BaseModelHarness,), namespace)
    return Delegating(default_cfg, tiny_model)


class TestTheHooksAreAbstract:
    """The decorator half. Pinned too, because the two are independent and
    a body that raises is not a substitute for refusing instantiation."""

    @pytest.mark.parametrize("hook", ABSTRACT_HOOKS)
    def test_the_hook_is_declared_abstract(self, hook):
        assert hook in BaseModelHarness.__abstractmethods__, (
            f"`{hook}` is not abstract, so a harness that forgets it is "
            f"constructible and fails later, at use"
        )

    def test_the_abstract_set_is_exactly_this(self):
        """Said as a set, so a regrowth cannot pass by marking everything
        abstract. `get_train_dataloaders` must stay optional: it refuses
        in its body and a harness may inherit that refusal."""
        assert set(BaseModelHarness.__abstractmethods__) == set(ABSTRACT_HOOKS), (
            f"abstract set is "
            f"{sorted(BaseModelHarness.__abstractmethods__)}, expected "
            f"{sorted(ABSTRACT_HOOKS)}"
        )

    def test_the_base_class_cannot_be_instantiated(self, default_cfg,
                                                   tiny_model):
        with pytest.raises(TypeError):
            BaseModelHarness(default_cfg, tiny_model)   # type: ignore[abstract]


class TestTheBodiesRefuse:
    """The body half, which is what a delegating subclass reaches."""

    @pytest.mark.parametrize("hook", REFUSING_HOOKS)
    def test_calling_up_raises_rather_than_returning_none(self, delegating,
                                                          hook):
        """The finding. An empty body returns `None`, and `None` here is
        unpacked into loaders and handed to the trainer."""
        with pytest.raises(NotImplementedError):
            getattr(delegating, hook)()

    @pytest.mark.parametrize("hook", REFUSING_HOOKS)
    def test_the_body_is_not_empty(self, hook):
        """Said structurally as well as behaviourally: the source of each
        hook has to do something. A body of `...` or a bare docstring is
        the exact shape the regrowth produced."""
        source = inspect.getsource(getattr(BaseModelHarness, hook))
        stripped = [
            line.strip() for line in source.splitlines()
            if line.strip() and not line.strip().startswith(("#", "@", "def "))
        ]
        meaningful = [
            line for line in stripped
            if line not in ("...", "pass") and not line.startswith(('"', "'"))
        ]
        assert meaningful, (
            f"`{hook}` has no statement in its body; a subclass that "
            f"delegates to it receives None"
        )


class TestTheFixtureReachesTheBodies:
    """Guard: if the delegating subclass were not actually calling up,
    every case above would be asserting something about itself."""

    def test_the_subclass_is_concrete(self, delegating):
        assert not getattr(type(delegating), "__abstractmethods__", None), (
            "the fixture is still abstract, so it could not have been "
            "instantiated and the bodies were never reached"
        )

    def test_a_non_refusing_hook_still_works_through_it(self, delegating):
        """The subclass delegates, so something the base class DOES
        implement must still come back normally -- otherwise the
        exceptions above might be the delegation failing, not the body."""
        assert delegating.model is not None
        assert delegating.cfg is not None
