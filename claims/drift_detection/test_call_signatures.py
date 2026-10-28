"""The signatures other layers call this one by.

Pinning WHICH names a package exports is not enough; a caller in another
layer invokes them with particular keywords, and nothing asserted those.
The full-stack assembly of nine blind rebuilds imported cleanly and then
died mid-training:

    File "apeiron/training/continuous_trainer.py", line 371, in
          outer_cl_training_loop
      self.profiler.print_performance(logger=self.logger, level=1)
    TypeError: FLOPSProfiler.print_performance() got an unexpected keyword
               argument 'logger'

Both layers passed every gate. The trainer's gate ran it against the
SOURCE profiler, which takes `(logger=None, level=0)`; the profilers' gate
had no caller that passed either keyword, so its reconstruction was free to
drop them. This is the third edge of the same problem -- after a module
borrowed through a sibling's namespace, and a package that re-exported too
little -- and the pattern is always the same: a layer's criteria are
checked against real neighbours, so whatever the neighbours happen to
offer is untested.

Asserted as parameter NAMES and their defaults rather than as the full
signature string, because annotations are free to change and a keyword
argument is what a caller actually depends on.
"""

from __future__ import annotations

import inspect

import pytest

from apeiron.drift_detection.detectors.base import BaseDriftDetector

#: method -> (owner, ordered parameter names after self, names with defaults)
CALLED_ACROSS_LAYERS = {
    'BaseDriftDetector.update': (BaseDriftDetector, ['value'], []),
    'BaseDriftDetector.reset': (BaseDriftDetector, [], []),
}


@pytest.mark.parametrize("label", sorted(CALLED_ACROSS_LAYERS))
def test_the_parameters_a_caller_passes_are_there(label):
    owner, expected, _defaults = CALLED_ACROSS_LAYERS[label]
    method = getattr(owner, label.split(".", 1)[1])
    params = [
        name for name, p in inspect.signature(method).parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert params == list(expected), (
        f"{label} takes {params}; callers in other layers pass "
        f"{list(expected)}"
    )


@pytest.mark.parametrize("label", sorted(CALLED_ACROSS_LAYERS))
def test_the_optional_parameters_stay_optional(label):
    """A caller that omits one must keep working, so a default cannot
    become required."""
    owner, _expected, defaults = CALLED_ACROSS_LAYERS[label]
    method = getattr(owner, label.split(".", 1)[1])
    params = inspect.signature(method).parameters
    missing = [
        name for name in defaults
        if name not in params or params[name].default is inspect.Parameter.empty
    ]
    assert not missing, f"{label} made {missing} required"
