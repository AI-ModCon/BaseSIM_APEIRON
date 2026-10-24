#!/usr/bin/env python3
"""Behaviour probe: apeiron.evaluation, off the pinned set.

The evaluation layer is the campaign's clearest criteria story and its
clearest warning. A3 recorded that `accuracy_topk` and all of
`evaluation.py` were untested; tests were written in response; A33 then put
the layer's kept-ratio at 65% with ZERO variance across draws -- the most
determinate layer measured. A determinate reconstruction is not the same as
a correct one, and this probe asks the second question.

Where it aims: the degenerate and boundary shapes a metric function meets in
practice but a test suite rarely writes down. Ties in the top-k ranking, k
larger than the number of classes, a single-class output, a one-sample
batch, an empty batch, targets outside the label range, a topk tuple that is
empty or unsorted or has duplicates. Each is a question about behaviour that
the docstring does not answer and the tests do not ask.

`test()` carries a documented arithmetic caveat (A19: it divides summed
batch losses by the sample count, so a mean-reduction criterion
under-reports by the batch count) and a documented oddity (an empty loader
returns a bare `0.0` rather than a pair). Both are probed, because a
reconstruction is free to "fix" either one and a fix is a divergence.

Everything is deterministic: fixed tensors, seeded model weights, no data
loading. Values are rounded before comparison so the two sides are not
separated by float formatting, but only to a precision far finer than any
behavioural difference would be.
"""
import json
import sys

import torch

from apeiron.evaluation.metrics import accuracy, accuracy_topk  # noqa: E402

try:
    from apeiron.evaluation.evaluation import test as eval_test
except Exception as exc:                             # noqa: BLE001 — probed
    eval_test = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def show(value) -> str:
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, torch.Tensor):
        return json.dumps([round(float(x), 6) for x in value.flatten()])
    if isinstance(value, (list, tuple)):
        return json.dumps([show(v) for v in value])
    if isinstance(value, float):
        return repr(round(value, 6))
    return repr(value)


def attempt(fn, *args, **kwargs) -> str:
    try:
        return show(fn(*args, **kwargs))
    except BaseException as exc:                     # noqa: BLE001 — probed
        return show(exc)


def main() -> int:
    out = {}
    torch.manual_seed(0)

    # --- 1. ordinary shapes, to anchor the comparison ---------------------
    logits = torch.tensor([[0.1, 0.9, 0.0],
                           [0.8, 0.1, 0.1],
                           [0.2, 0.2, 0.6],
                           [0.4, 0.5, 0.1]])
    target = torch.tensor([1, 0, 2, 2])
    out["accuracy[ordinary]"] = attempt(accuracy, logits, target)
    for topk in ((1,), (2,), (3,), (1, 2), (1, 2, 3), (3, 1), (1, 1), (),
                 (4,), (0,), (-1,)):
        out[f"accuracy_topk[{topk}]"] = attempt(accuracy_topk, logits, target,
                                                topk=topk)

    # --- 2. ties in the ranking -------------------------------------------
    # Every logit equal: which class does topk pick, and is the accuracy the
    # chance rate or something else? Nothing pins this, and an implementation
    # using argmax vs topk vs sorting can differ.
    flat = torch.zeros(4, 3)
    out["accuracy[all-tied]"] = attempt(accuracy, flat, target)
    out["accuracy_topk[all-tied,k=2]"] = attempt(accuracy_topk, flat, target,
                                                 topk=(2,))
    two_way = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    out["accuracy[two-way-tie]"] = attempt(accuracy, two_way,
                                           torch.tensor([0, 1]))

    # --- 3. degenerate shapes ---------------------------------------------
    out["accuracy[single-sample]"] = attempt(
        accuracy, torch.tensor([[0.2, 0.8]]), torch.tensor([1]))
    out["accuracy[single-class]"] = attempt(
        accuracy, torch.tensor([[1.0], [1.0]]), torch.tensor([0, 0]))
    out["accuracy[empty-batch]"] = attempt(
        accuracy, torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
    out["accuracy[k-exceeds-classes]"] = attempt(
        accuracy_topk, torch.tensor([[0.1, 0.9]]), torch.tensor([1]), topk=(5,))
    out["accuracy[all-correct]"] = attempt(
        accuracy, torch.eye(3), torch.tensor([0, 1, 2]))
    out["accuracy[none-correct]"] = attempt(
        accuracy, torch.eye(3), torch.tensor([1, 2, 0]))

    # --- 4. ill-typed and out-of-range input ------------------------------
    out["accuracy[target-out-of-range]"] = attempt(
        accuracy, logits, torch.tensor([9, 0, 2, 2]))
    out["accuracy[target-negative]"] = attempt(
        accuracy, logits, torch.tensor([-1, 0, 2, 2]))
    out["accuracy[float-target]"] = attempt(
        accuracy, logits, torch.tensor([1.0, 0.0, 2.0, 2.0]))
    out["accuracy[mismatched-batch]"] = attempt(
        accuracy, logits, torch.tensor([1, 0]))
    out["accuracy[1d-output]"] = attempt(
        accuracy, torch.tensor([0.1, 0.9]), torch.tensor([1]))
    out["accuracy[3d-output]"] = attempt(
        accuracy, torch.zeros(2, 3, 4), torch.tensor([1, 2]))
    out["accuracy[nan-logits]"] = attempt(
        accuracy, torch.full((2, 3), float("nan")), torch.tensor([0, 1]))
    out["accuracy[inf-logits]"] = attempt(
        accuracy, torch.tensor([[float("inf"), 0.0, 0.0],
                                [0.0, float("-inf"), 0.0]]),
        torch.tensor([0, 2]))
    # does it leave its inputs alone? `correct_k.mul_` is in-place on a
    # derived tensor, but an implementation could mutate `output`
    keep = logits.clone()
    attempt(accuracy, logits, target)
    out["accuracy[mutates-output]"] = show(bool(torch.equal(keep, logits)))
    out["accuracy[requires-grad-input]"] = attempt(
        accuracy, logits.clone().requires_grad_(True), target)

    # --- 5. test(), including its documented caveats ----------------------
    if eval_test is None:
        out["test[import]"] = _IMPORT_ERROR
    else:
        class Cfg:
            device = "cpu"

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 3)
                with torch.no_grad():
                    self.fc.weight.copy_(torch.linspace(-1, 1, 12).view(3, 4))
                    self.fc.bias.copy_(torch.tensor([0.1, 0.0, -0.1]))

            def forward(self, x):
                return self.fc(x)

        def loader(n: int, batch: int):
            xs = torch.linspace(-1, 1, max(n, 1) * 4)[:n * 4].view(n, 4) \
                if n else torch.zeros(0, 4)
            ys = torch.arange(n) % 3
            data = torch.utils.data.TensorDataset(xs, ys)
            return torch.utils.data.DataLoader(data, batch_size=batch)

        model = Tiny()
        for name, (n, batch, reduction) in {
            "one-batch-sum": (6, 6, "sum"),
            "one-batch-mean": (6, 6, "mean"),
            # A19 lives here: three batches instead of one changes the
            # reported loss under a mean-reduction criterion
            "three-batches-mean": (6, 2, "mean"),
            "three-batches-sum": (6, 2, "sum"),
            "ragged-last-batch": (7, 3, "sum"),
            "single-sample": (1, 1, "sum"),
            # the documented oddity: a bare 0.0 rather than a pair
            "empty-loader": (0, 1, "sum"),
        }.items():
            criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
            out[f"test[{name}]"] = attempt(eval_test, model, loader(n, batch),
                                           criterion, Cfg())
        # does it restore training mode, or leave the model in eval?
        model.train()
        attempt(eval_test, model, loader(4, 2),
                torch.nn.CrossEntropyLoss(reduction="sum"), Cfg())
        out["test[leaves-model-in-eval]"] = show(not model.training)

    json.dump(out, sys.stdout, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
