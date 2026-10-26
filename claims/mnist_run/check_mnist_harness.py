"""Acceptance criteria for the MNIST example harness.

This exists because of A43. The run claim beside it pins a training RESULT
by digest, which cannot be faked and also cannot be reached: a blind
reconstruction recovered the network exactly from the pinned weights, parsed
the dataset itself, and hit the right evaluation cadence -- and still matched
0 of 31 metric series, because nothing told it what the simulated drift is.
A digest asserts an answer and teaches nothing.

So this file teaches. It pins the harness's observable behaviour the way the
ten layer claims pin theirs, and the result digest sits on top as the part
that cannot be forged. Between them: reachable, and unfakeable.

What determines the trace, and is pinned here:

  the drift schedule   `sample_aug(seed)` is the whole drift simulation. One
                       affine transform per task, drawn from a seeded
                       generator, chained cumulatively over history so the
                       distribution walks away from the original. Exact
                       values are pinned for several seeds -- this is the
                       thing a reconstruction cannot invent.
  the chaining         each task's transform is APPENDED and all of them are
                       re-applied, so task three sees three compositions, not
                       one. Pinned by applying a known history to a known
                       tensor.
  the sample shape     a stream sample is [H, W], not [1, H, W] -- the view
                       squeezes the channel and the network adds it back.
                       Pinned because it changes what a batch looks like.
  the optimiser        Adam at `train.init_lr`, not SGD. Different update
                       rule, different trace.
  stochastic forward   the network's output differs between train() and
                       eval() for one input, which is dropout being present.
                       Pinned as behaviour rather than as an architecture
                       detail, so any equivalent spelling passes.
  the loaders          three of them per task, with specific batch sizes and
                       shuffle flags: train at `train.batch_size` shuffled,
                       validation at the same size unshuffled, and the
                       monitored stream at `data.batch_size` shuffled.

Deliberately NOT pinned: the layer sizes and kernel shapes. Those are
already carried by `examples/mnist/mnist.pth`, which the reconstruction
recovered exactly by loading it -- a pinned artefact constrains better than a
written-down number, and repeating it here would only let an implementation
satisfy the criteria while mismatching the checkpoint.
"""

import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from apeiron.config.configuration import build_config  # noqa: E402
from examples.mnist.utils import FixedAffine, sample_aug  # noqa: E402

#: Recorded from the harness. angle and scale come from two successive draws
#: of a generator seeded with the task's seed; shear follows angle; translate
#: is the scale truncated to int, which is 1 for every scale in range.
AUG = {
    1337: {"angle": 0.7831710577011108, "scale": 0.9978083670139313,
           "shear": 0.7831710577011108, "translate": (0, 0)},
    1338: {"angle": 7.27046012878418, "scale": 0.9618054032325745,
           "shear": 7.27046012878418, "translate": (0, 0)},
    1339: {"angle": 9.169278740882874, "scale": 1.138285905122757,
           "shear": 9.169278740882874, "translate": (1, 1)},
    0: {"angle": 4.962565898895264, "scale": 1.1341108977794647,
        "shear": 4.962565898895264, "translate": (1, 1)},
}

#: The network's output for a fixed input, in eval() mode, with the pinned
#: weights loaded. This pins the FORWARD ARRANGEMENT without describing it.
#:
#: `mnist.pth` constrains the layer shapes -- a reconstruction recovered them
#: exactly by loading it -- but says nothing about where the pooling and the
#: regularisation sit. One attempt pooled once instead of twice, used two
#: dropouts instead of three, and reached adaptive_avg_pool2d for the
#: flatten; it satisfied every other criterion here and still missed 13 of 31
#: metric series, because a different arrangement computes different
#: activations (A43). In eval() there is no dropout, so this digest is
#: deterministic, and matching it means the computation is the same one.
FORWARD_DIGEST = '48b9f52ca7e877708a92cc377e63daa2945968c6503e5811fbbbc08ab2da420a'

#: The same fixed input through the network in TRAINING mode, under a fixed
#: RNG seed. This pins the regularisation itself.
#:
#: FORWARD_DIGEST above is taken in eval(), where dropout is a no-op, so it
#: constrains the convolutions, the pooling and the flatten and says nothing
#: about how strongly the network regularises. One reconstruction matched it
#: exactly -- same three dropout points, same positions, derived generically
#: from the state dict -- while using rates of 0.25, 0.25, 0.5 against the
#: real 0.5, 0.5, 0.5, and still missed 13 of 31 metric series because a
#: differently-regularised network trains to a different place (A43).
#:
#: In train() the dropout draws come from the seeded global RNG, so this
#: digest is a function of how many dropout points there are, where they sit,
#: and what each one's rate is -- jointly, without naming any of them.
TRAIN_FORWARD_SEED = 20261026
TRAIN_FORWARD_DIGEST = '8d95a4338bc421ce4d47396375dd4c78ad46ab1d96fc0b89af7b3fe69e6a874c'

#: What the chained transform actually COMPUTES, for one-, two- and
#: three-task histories applied to a fixed picture.
#:
#: The structural assertions below -- that the three differ, that an empty
#: history is a no-op, that order matters -- are necessary and nowhere near
#: sufficient. One reconstruction satisfied all of them with a hand-written
#: affine matrix of its own: it rotated, sheared and scaled by the right
#: parameters and produced different pixels, because interpolation and
#: padding are part of the transform and its matrix was not torchvision's.
#: Every structural property held and the training trace was unrecognisable
#: (A43). Pinning the output closes it without naming an implementation.
AFFINE_DIGESTS = {1: "6ae36ba8642ede07", 2: "e2b9bf7f34ae86ee", 3: "803a7097a7162ef5"}

#: Which split each loader draws from, pinned by what its first element is.
#:
#: The batch sizes and shuffle flags below are not enough. One reconstruction
#: matched all of them and streamed the TRAIN split where the harness streams
#: the VALIDATION split -- the monitored metric was therefore measured on
#: data the model was training on, and the whole trace was unrecognisable
#: while every other criterion here passed (A43). MNIST's two splits start
#: with different labels (test: 7,2,1,0,...; train: 5,0,4,1,...), so one
#: element identifies the split.
SPLIT_FIRST = {"stream": 7, "train": 5, "val": 7}

#: The replay buffer, task by task: its size and which regime comes first.
#:
#: This is what the CL loop trains against, and it is the last thing the
#: criteria left free. The buffer spans EVERY prior cumulative regime, not
#: just the most recent one, so it grows: nothing at task one, one split's
#: worth at task two, two splits' worth at task three. One reconstruction
#: kept only the latest regime -- constant 2,000 instead of growing to 4,000
#: -- and matched 18 of 31 metric series, missing exactly the thirteen
#: accuracy and loss series, because replaying different data trains to a
#: different place while leaving every count, timing and FLOP figure intact
#: (A43).
#:
#: The first element is pinned too: the buffer starts at the EARLIEST regime,
#: which is what keeps adaptation from specialising to the latest drift.
REPLAY = {1: None, 2: (2000, "756a92aa07c9bd1b"), 3: (4000, "756a92aa07c9bd1b")}

#: The bounds the sampler draws within, asserted separately so an
#: implementation that happens to match the seeds above but uses a different
#: range is still refused.
ANGLE_MAX = 10.0
SCALE_MIN, SCALE_MAX = 0.75, 1.25


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    # --- 1. the drift schedule, value for value ---------------------------
    for seed, want in sorted(AUG.items()):
        got = sample_aug(seed=seed)
        assert set(got) >= {"angle", "scale", "shear", "translate"}, (
            f"sample_aug must return angle, scale, shear and translate; "
            f"got {sorted(got)}")
        for key in ("angle", "scale", "shear"):
            assert approx(float(got[key]), want[key]), (
                f"sample_aug({seed})[{key!r}] = {got[key]!r}, expected "
                f"{want[key]!r}. This is the drift simulation: the whole "
                f"point of the run is that the distribution moves in exactly "
                f"this way, so a different draw is a different experiment.")
        assert tuple(int(v) for v in got["translate"]) == want["translate"], (
            f"sample_aug({seed})['translate'] = {got['translate']!r}, "
            f"expected {want['translate']!r}")

    # the same seed twice is the same transform, and different seeds differ
    assert sample_aug(seed=5) == sample_aug(seed=5), (
        "the drift must be reproducible from its seed")
    assert sample_aug(seed=5) != sample_aug(seed=6), (
        "different tasks must get different drift")

    # shear tracks angle, which is a property of the schedule and not a
    # coincidence of the seeds above
    for seed in (11, 22, 4242):
        aug = sample_aug(seed=seed)
        assert approx(float(aug["shear"]), float(aug["angle"])), (
            f"shear and angle must be equal; seed {seed} gave "
            f"{aug['shear']} and {aug['angle']}")

    # and the ranges
    for seed in range(40):
        aug = sample_aug(seed=seed)
        assert 0.0 <= float(aug["angle"]) <= ANGLE_MAX, (
            f"angle {aug['angle']} outside [0, {ANGLE_MAX}] at seed {seed}")
        assert SCALE_MIN <= float(aug["scale"]) <= SCALE_MAX, (
            f"scale {aug['scale']} outside [{SCALE_MIN}, {SCALE_MAX}] at "
            f"seed {seed}")

    # --- 2. the chaining, which is what makes the drift cumulative --------
    picture = torch.linspace(0, 1, 28 * 28).view(1, 28, 28)

    def digest(tensor: torch.Tensor) -> str:
        flat = tensor.detach().flatten().tolist()
        return hashlib.sha256(
            ",".join(f"{v:.5f}" for v in flat).encode()).hexdigest()[:16]

    one = FixedAffine(aug_history=[sample_aug(1337)])
    two = FixedAffine(aug_history=[sample_aug(1337), sample_aug(1338)])
    three = FixedAffine(
        aug_history=[sample_aug(1337), sample_aug(1338), sample_aug(1339)])

    d1, d2, d3 = digest(one(picture)), digest(two(picture)), digest(three(picture))
    assert d1 != d2 != d3 and d1 != d3, (
        "each task must add a transform: applying a one-, two- and "
        "three-entry history to the same picture gave the same result, so "
        "the drift is not accumulating")
    assert digest(FixedAffine(aug_history=[])(picture)) == digest(picture), (
        "an empty history must leave the picture alone")

    # and what each chain computes, which the three-way difference above does
    # not pin: a transform with the right parameters but different
    # interpolation or padding passes everything else here and trains
    # somewhere else entirely
    if "--emit-affine" in sys.argv:
        print(f'AFFINE_DIGESTS = {{1: "{d1}", 2: "{d2}", 3: "{d3}"}}')
        return 0
    for n, got_digest in ((1, d1), (2, d2), (3, d3)):
        assert got_digest == AFFINE_DIGESTS[n], (
            f"a {n}-task history transforms the fixed picture to "
            f"{got_digest}, expected {AFFINE_DIGESTS[n]}. The angle, scale "
            f"and shear are already pinned above, so this is about HOW the "
            f"transform is applied -- interpolation, padding and the matrix "
            f"convention are part of it.")
    # The chain is ordered -- but only observably so for some pairs, which is
    # worth stating rather than glossing. Rotation and uniform scaling
    # commute, so two transforms whose translate is the same compose to the
    # same picture either way: (1337, 1338) and (1337, 1339) are both
    # order-insensitive here, and an earlier version of this check asserted
    # order on (1337, 1338) and failed against the real implementation.
    # (1338, 1339) differ in translate, and translation does not commute with
    # rotation, so that pair does see the order.
    forward = FixedAffine(aug_history=[sample_aug(1338), sample_aug(1339)])
    reverse = FixedAffine(aug_history=[sample_aug(1339), sample_aug(1338)])
    assert digest(forward(picture)) != digest(reverse(picture)), (
        "the transforms must be applied in history order: reversing a pair "
        "that differs in translate gave the same picture")
    # and shape survives
    assert one(picture).shape == picture.shape, (
        f"the transform changed the shape to {tuple(one(picture).shape)}")

    # --- 3. the harness itself --------------------------------------------
    from examples.utils import get_example

    cfg = build_config(["--config", str(HERE / "mnist-run.toml"),
                        "--set", "device=cpu"])
    harness = get_example(cfg=cfg)

    assert harness.get_stream_dataloader() is None, (
        "before the first update_data_stream() there is no stream yet")

    harness.update_data_stream()
    first = sample_aug(seed=cfg.seed + 0)
    assert approx(float(harness.cur_aug["angle"]), first["angle"]), (
        f"the first task's drift must be sample_aug(cfg.seed + 0); got angle "
        f"{harness.cur_aug['angle']}, expected {first['angle']}")
    assert len(harness.aug_history) == 1, (
        f"aug_history holds {len(harness.aug_history)} after one task")

    harness.update_data_stream()
    second = sample_aug(seed=cfg.seed + 1)
    assert approx(float(harness.cur_aug["angle"]), second["angle"]), (
        "the second task's drift must be sample_aug(cfg.seed + 1) -- the "
        "task counter advances the seed")
    assert len(harness.aug_history) == 2, (
        "history must accumulate, not be replaced")

    # the three loaders, their batch sizes and their shuffling
    stream = harness.get_stream_dataloader()
    train, val = harness.get_train_dataloaders()
    assert stream is not None and train is not None and val is not None
    assert stream.batch_size == cfg.data.batch_size, (
        f"the monitored stream batches at data.batch_size "
        f"({cfg.data.batch_size}); got {stream.batch_size}")
    assert train.batch_size == cfg.train.batch_size, (
        f"training batches at train.batch_size ({cfg.train.batch_size}); "
        f"got {train.batch_size}")
    assert val.batch_size == cfg.train.batch_size

    def shuffles(loader) -> bool:
        from torch.utils.data import RandomSampler
        return isinstance(loader.sampler, RandomSampler)

    # which split each loader is over -- the monitored stream is the
    # VALIDATION data, not the training data
    for name, loader in (("stream", stream), ("train", train), ("val", val)):
        _x, first_label = loader.dataset[0]
        assert int(first_label) == SPLIT_FIRST[name], (
            f"the {name} loader's dataset starts with label "
            f"{int(first_label)}, expected {SPLIT_FIRST[name]}. MNIST's "
            f"splits start differently, so this says which split it is: the "
            f"monitored stream and the validation loader are over the "
            f"VALIDATION split, and only the train loader is over the "
            f"training split. Streaming the training data measures the "
            f"metric on data the model is fitting.")
    assert stream.dataset is not train.dataset, (
        "the stream and the training loader must not be the same data")

    assert shuffles(stream), "the stream loader is shuffled"
    assert shuffles(train), "the train loader is shuffled"
    assert not shuffles(val), "the validation loader is NOT shuffled"

    # a stream sample is [H, W]: the view drops the channel, the net re-adds it
    x, y = next(iter(stream))
    assert x.dim() == 3, (
        f"a stream batch has shape {tuple(x.shape)}; expected [B, H, W] -- "
        f"the dataset view squeezes the channel away")
    assert x.shape[1:] == (28, 28), f"sample shape {tuple(x.shape[1:])}"
    assert y.dim() == 1 and y.dtype == torch.int64

    # --- 3b. the replay buffer the CL loop trains against ----------------
    replay_harness = get_example(cfg=cfg)
    for task in sorted(REPLAY):
        replay_harness.update_data_stream()
        hist = replay_harness.get_hist_dataloaders()
        want_replay = REPLAY[task]
        if want_replay is None:
            assert hist[0] is None, (
                f"after {task} task there is no history yet, so the replay "
                f"buffer must be None; got a loader")
            continue
        want_len, want_first = want_replay
        assert hist[0] is not None, (
            f"after {task} tasks there must be a replay buffer")
        got_len = len(hist[0].dataset)
        assert got_len == want_len, (
            f"after {task} tasks the replay buffer holds {got_len} samples, "
            f"expected {want_len}. It spans EVERY prior cumulative regime, "
            f"not just the most recent -- a buffer that stays one split long "
            f"replays different data and trains somewhere else, while leaving "
            f"every count and timing in the trace intact.")
        first_x, _first_y = hist[0].dataset[0]
        if "--emit-replay" in sys.argv:
            print(f"  task {task}: ({got_len}, {digest(first_x)!r})")
            continue
        assert digest(first_x) == want_first, (
            f"after {task} tasks the replay buffer starts at {digest(first_x)}, "
            f"expected {want_first} -- the earliest regime comes first, which "
            f"is what anchors adaptation against specialising to the latest "
            f"drift.")

    # --- 4. optimiser, metrics, and the stochastic forward ----------------
    optimizer = harness.get_optimizer()
    assert type(optimizer).__name__ == "Adam", (
        f"the harness optimises with Adam; got {type(optimizer).__name__}. A "
        f"different update rule produces a different trace.")
    assert approx(optimizer.param_groups[0]["lr"], cfg.train.init_lr), (
        "the optimiser takes its learning rate from train.init_lr")

    assert sorted(harness.eval_metrics) == ["accuracy", "loss"], (
        f"eval_metrics are accuracy and loss, in that order of index; got "
        f"{sorted(harness.eval_metrics)}")
    assert harness.higher_is_better == {"accuracy": True, "loss": False}, (
        f"higher_is_better must say which direction each metric improves; "
        f"got {getattr(harness, 'higher_is_better', None)}")

    # the forward pass is stochastic in training mode and not in eval mode.
    # Stated as behaviour so any equivalent regularisation spelling passes.
    sample = x[:4]
    harness.model.eval()
    with torch.no_grad():
        a = harness.model(sample)
        b = harness.model(sample)
    assert torch.equal(a, b), "in eval() the forward pass must be deterministic"
    harness.model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        c = harness.model(sample)
    torch.manual_seed(1)
    with torch.no_grad():
        d = harness.model(sample)
    assert not torch.equal(c, d), (
        "in train() the forward pass must be stochastic -- the network "
        "regularises with dropout, and a network without it trains to a "
        "different place")

    # log_softmax output: exponentiates to a distribution
    with torch.no_grad():
        harness.model.eval()
        probs = harness.model(sample).exp().sum(dim=1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5), (
        f"the network returns log-probabilities; exp().sum() gave "
        f"{probs.tolist()}")
    assert a.shape == (4, 10), f"ten classes expected; got {tuple(a.shape)}"

    # --- 5. the forward arrangement, pinned by what it computes -----------
    fixed = (torch.linspace(-1.0, 1.0, 4 * 28 * 28)
             .view(4, 28, 28))
    harness.model.eval()
    with torch.no_grad():
        logits = harness.model(fixed)
    forward = hashlib.sha256(
        ",".join(f"{v:.5f}" for v in logits.flatten().tolist()).encode()
    ).hexdigest()
    harness.model.train()
    torch.manual_seed(TRAIN_FORWARD_SEED)
    with torch.no_grad():
        trained_mode = harness.model(fixed)
    train_forward = hashlib.sha256(
        ",".join(f"{v:.5f}" for v in trained_mode.flatten().tolist()).encode()
    ).hexdigest()
    harness.model.eval()

    if "--emit" in sys.argv:
        print(f"FORWARD_DIGEST = {forward!r}")
        print(f"TRAIN_FORWARD_DIGEST = {train_forward!r}")
        return 0
    assert forward == FORWARD_DIGEST, (
        f"the network computes {forward[:16]}... on the fixed input, expected "
        f"{FORWARD_DIGEST[:16]}...\n"
        f"The layer shapes are already fixed by examples/mnist/mnist.pth, so "
        f"this is about ARRANGEMENT: where the pooling happens, how many "
        f"regularisation points there are, and how the convolutional output "
        f"is flattened. A different arrangement trains to a different place.")
    assert train_forward == TRAIN_FORWARD_DIGEST, (
        f"in train() under seed {TRAIN_FORWARD_SEED} the network computes "
        f"{train_forward[:16]}..., expected {TRAIN_FORWARD_DIGEST[:16]}...\n"
        f"The eval() digest already matched, so the convolutions, pooling and "
        f"flatten are right and this is about REGULARISATION: how many "
        f"dropout points there are, where they sit, and the rate of each. "
        f"Those are invisible in eval() and decide where training lands.")

    (HERE / "HARNESS_OK").write_text("ok\n")
    print(f"mnist harness: drift schedule pinned at {len(AUG)} seeds, "
          f"chaining ordered and cumulative, Adam at {cfg.train.init_lr}, "
          f"stochastic forward confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
