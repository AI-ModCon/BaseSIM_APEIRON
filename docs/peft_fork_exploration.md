# PEFT Fork Exploration: Recasting BaseSim as a HuggingFace PEFT Fork

## Executive Summary

Yes — BaseSim can be recast as a PEFT fork, but it requires adding a **new axis
of abstraction** rather than shoehorning CL updaters into the existing
`BaseTuner` mold.  PEFT tuners modify *model architecture* (inject adapter
layers that change the forward pass).  BaseSim updaters modify the *training
loop* (inject hooks around forward/backward/optimizer).  A fork would keep
PEFT's tuner infrastructure, add a parallel `BaseCLMethod` hierarchy for
continual-learning strategies, and compose both under a unified `CLPeftModel`.

The drift monitor would ship as a standalone module that orchestrates the whole
cycle: monitor → detect → dispatch CL training with the registered method.

---

## 1. Architectural Comparison

### 1.1 Where the two systems operate

```
                         PEFT                          BaseSim
                    ─────────────                ──────────────────
 Model wrapping     PeftModel wraps base model   BaseModelHarness wraps model
 What gets          Target modules replaced       Nothing replaced — hooks on
 modified           with adapter-wrapped layers   the training loop instead
 Forward pass       Modified (adapter layers)     Unmodified (standard forward)
 Backward pass      Standard                      May be modified (JVP uses
                                                  torch.func.jvp for dual backward)
 Optimizer step     Standard                      Pre/post hooks (EWC injects
                                                  gradient penalty, KFAC adds
                                                  curvature-weighted gradients)
 Training loop      Delegated to HF Trainer       Owned by ContinuousTrainer
 Drift detection    N/A                           ContinuousMonitor + detectors
```

### 1.2 Concept mapping

| BaseSim                | PEFT Analog               | Compatibility |
|------------------------|---------------------------|---------------|
| `BaseUpdater`          | No direct analog           | **New concept** — PEFT has no training-loop hooks |
| `BaseUpdater.fwd_bwd()`| Closest to a Tuner's modified `forward()` | Partial — BaseSim also controls backward |
| `OnlineEWCUpdater`     | Conceptually a "tuner" that adds regularization | Medium fit — operates on all params, not per-module |
| `OnlineKFACUpdater`    | **Best fit** — hooks on `nn.Linear`/`nn.Conv2d` | High — already identifies target modules + registers hooks |
| `JVPRegUpdater`        | No analog — uses `torch.func` functional transforms | Low — fundamentally a training strategy, not a model mod |
| `ContinuousMonitor`    | No analog                  | **New module** |
| `BaseDriftDetector`    | No analog                  | **New module** |
| `BaseModelHarness`     | Somewhat like `PeftModel`  | Both wrap a base model |
| `ContinualLearningCfg` | Like `PeftConfig` subclasses | Configuration per method |
| `create_updater()`     | `get_peft_model()` / `PEFT_TYPE_TO_TUNER_MAPPING` | Factory dispatch pattern |

---

## 2. PEFT's Extension Points

PEFT's `BaseTuner` (in `src/peft/tuners/tuners_utils.py`) provides:

```python
class BaseTuner(nn.Module):
    def __init__(self, model, peft_config, adapter_name):
        self.model = model
        self.inject_adapter(model, adapter_name)

    def inject_adapter(self, model, adapter_name):
        """Walk named_modules, call _create_and_replace on targets."""
        for key, module in model.named_modules():
            if self._check_target_module_exists(config, key):
                self._create_and_replace(config, adapter_name, module, key)
        self._mark_only_adapters_as_trainable(model)

    # --- Subclasses must implement ---
    def _prepare_adapter_config(self, config, model_config): ...
    def _create_and_replace(self, config, adapter_name, target, target_name): ...
    def _check_target_module_exists(self, config, key): ...
```

Each tuner also has a companion `BaseTunerLayer` that wraps individual modules
and manages adapter weights (enable/disable/merge/unmerge).

**Critical observation:** PEFT tuners operate at the *module-replacement* level.
They never touch the training loop.  The training loop is entirely owned by
the HF `Trainer` (or the user's custom loop).

---

## 3. Proposed Fork Architecture

### 3.1 High-level design

```
peft-cl/                            (the fork)
├── src/peft/
│   ├── peft_model.py               KEEP — PeftModel, PeftModelForCausalLM, etc.
│   ├── config.py                   EXTEND — add CLMethodConfig base
│   ├── mapping.py                  EXTEND — add CL_TYPE_TO_METHOD_MAPPING
│   │
│   ├── tuners/                     KEEP ALL — LoRA, AdaLoRA, IA3, etc.
│   │   ├── lora/
│   │   ├── ia3/
│   │   └── ...
│   │
│   ├── cl_methods/                 NEW — continual learning methods
│   │   ├── __init__.py
│   │   ├── base.py                 BaseCLMethod (the BaseUpdater analog)
│   │   ├── ewc_online/
│   │   │   ├── config.py           EWCConfig(CLMethodConfig)
│   │   │   └── method.py           OnlineEWCMethod(BaseCLMethod)
│   │   ├── kfac_online/
│   │   │   ├── config.py           KFACConfig(CLMethodConfig)
│   │   │   └── method.py           OnlineKFACMethod(BaseCLMethod)
│   │   ├── jvp_reg/
│   │   │   ├── config.py           JVPConfig(CLMethodConfig)
│   │   │   └── method.py           JVPRegMethod(BaseCLMethod)
│   │   └── none/
│   │       └── method.py           NoOpMethod(BaseCLMethod)
│   │
│   ├── monitoring/                 NEW — drift detection (from BaseSim)
│   │   ├── __init__.py
│   │   ├── base.py                 BaseDriftDetector, DriftSignal, LearningRegime
│   │   ├── adwin.py
│   │   ├── kswin.py
│   │   ├── page_hinkley.py
│   │   ├── model_eval.py
│   │   ├── ensemble.py
│   │   └── continuous_monitor.py   ContinuousMonitor
│   │
│   └── cl_trainer.py               NEW — CL-aware Trainer
│       CLTrainer(transformers.Trainer)
│         ├── overrides training_step() with CL hooks
│         ├── wires up ContinuousMonitor
│         └── supports both PEFT adapters + CL methods simultaneously
```

### 3.2 The `BaseCLMethod` — overloading the inner training loop

This is the core of the idea.  Rather than replacing modules (like `BaseTuner`),
`BaseCLMethod` provides hooks that the training loop calls at defined points:

```python
class BaseCLMethod:
    """
    Base class for continual-learning update strategies.
    Follows PEFT conventions (config-driven, registerable) but hooks
    into the training loop rather than the model architecture.
    """

    def __init__(self, model: nn.Module, cl_config: CLMethodConfig):
        self.model = model
        self.config = cl_config

    # ── lifecycle hooks (called by CLTrainer) ──────────────────────

    def cl_preprocessing(self) -> None:
        """Called once before a CL training loop starts (post-drift)."""
        pass

    def cl_postprocessing(self) -> None:
        """Called once after a CL training loop ends."""
        pass

    # ── per-step hooks ─────────────────────────────────────────────

    def pre_forward_backward(self) -> None:
        """Called after optimizer.zero_grad(), before forward pass."""
        pass

    def forward_backward(
        self,
        model: nn.Module,
        batch: tuple[Tensor, Tensor],
        hist_batch: tuple[Tensor, Tensor] | None,
        criterion: Callable,
        grad_accumulation_steps: int,
    ) -> float:
        """Default: standard forward + loss.backward().
        Override for custom forward/backward (e.g., JVP)."""
        x, y = batch
        loss = criterion(model(x), y) / grad_accumulation_steps
        loss.backward()
        return loss.item()

    def post_forward_backward(self) -> float:
        """Called after backward, before optimizer.step().
        Returns regularization loss (for logging).
        Use this to inject gradient penalties (EWC, KFAC)."""
        return 0.0

    def post_optimizer_step(self) -> None:
        """Called after optimizer.step().
        Use this to update internal state (Fisher accumulators, etc.)."""
        pass
```

### 3.3 The `CLTrainer` — overloaded inner loop

The `CLTrainer` extends HuggingFace's `Trainer` and wires the hooks in:

```python
class CLTrainer(transformers.Trainer):
    """
    Extends HF Trainer with continual-learning hooks.
    Supports both PEFT adapters (via PeftModel) and CL methods
    (via BaseCLMethod) simultaneously.
    """

    def __init__(self, *args, cl_method: BaseCLMethod | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cl_method = cl_method

    def training_step(self, model, inputs, num_items_in_batch=None):
        # ── pre-hook ──
        if self.cl_method:
            self.cl_method.pre_forward_backward()

        # ── forward + backward ──
        if self.cl_method:
            loss = self.cl_method.forward_backward(
                model=model,
                batch=(inputs["input_ids"], inputs["labels"]),
                hist_batch=self._get_hist_batch(),   # from replay buffer
                criterion=self.compute_loss,
                grad_accumulation_steps=self.args.gradient_accumulation_steps,
            )
        else:
            loss = super().training_step(model, inputs, num_items_in_batch)

        # ── post-backward hook (gradient penalties) ──
        if self.cl_method:
            reg_loss = self.cl_method.post_forward_backward()

        return loss

    # Override optimizer step to add post-optimizer hook
    def _inner_training_loop(self, ...):
        # ... after optimizer.step() ...
        if self.cl_method:
            self.cl_method.post_optimizer_step()
```

### 3.4 Composing PEFT adapters + CL methods

The real power: use LoRA for parameter efficiency **and** EWC for forgetting prevention:

```python
from peft import get_peft_model, LoraConfig
from peft.cl_methods import OnlineEWCMethod, EWCConfig
from peft.monitoring import ContinuousMonitor, ADWINDetector

# 1. Base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)

# 2. CL method (EWC regularization on LoRA params only)
ewc_config = EWCConfig(ewc_lambda=1000.0, ewc_ema_decay=0.95)
cl_method = OnlineEWCMethod(model, ewc_config)

# 3. Drift-aware trainer
trainer = CLTrainer(
    model=model,
    cl_method=cl_method,
    train_dataset=stream_dataset,
    # ... standard HF Trainer args ...
)

# 4. Monitor wraps the trainer
monitor = ContinuousMonitor(
    trainer=trainer,
    detector=ADWINDetector(delta=0.002),
    detection_interval=100,
)
monitor.run()  # monitor → detect drift → dispatch CL → resume
```

---

## 4. How Each Updater Maps to a PEFT CL Method

### 4.1 OnlineEWCMethod (from `OnlineEWCUpdater`)

**Fit: Medium-High** — EWC maintains per-parameter state (anchor θ*, Fisher F*).
In the fork, it would target only PEFT adapter parameters (e.g., only LoRA
weights), making it naturally parameter-efficient.

```
PEFT hooks used:
  cl_preprocessing()   → allocate Fisher accumulators
  post_forward_backward() → inject EWC gradient penalty: ∇ += λ·F·(θ-θ*)
  post_optimizer_step()   → accumulate batch Fisher from grad²
  cl_postprocessing()     → EMA update: F* ← decay·F* + F_cl_avg; θ* ← θ_final
```

### 4.2 OnlineKFACMethod (from `OnlineKFACUpdater`)

**Fit: High** — KFAC already identifies target modules (`nn.Linear`, `nn.Conv2d`)
and registers forward/backward hooks — this is almost exactly how PEFT tuners
work.  In the fork, KFAC could additionally use `_check_target_module_exists`
from `BaseTuner` to target the same modules PEFT adapters are applied to.

```
PEFT hooks used:
  __init__()              → register forward/backward hooks on target modules
                            (reuse PEFT's target_modules matching)
  cl_preprocessing()      → allocate Kronecker factor accumulators
  post_forward_backward() → accumulate A=a^T·a, G=g^T·g; inject KFAC penalty
  post_optimizer_step()   → increment step counter
  cl_postprocessing()     → EMA update Kronecker factors; update anchor
```

### 4.3 JVPRegMethod (from `JVPRegUpdater`)

**Fit: Low-Medium** — JVP fundamentally replaces the forward/backward pass using
`torch.func.jvp` and `functional_call`.  It doesn't inject into the model
architecture at all.  In the fork, this becomes a CL method that overrides
`forward_backward()` entirely.

```
PEFT hooks used:
  pre_forward_backward()  → cache params dict for functional API
  forward_backward()      → FULLY OVERRIDDEN: dual forward with JVP
                            grad_combined = grad_curr + grad_mem + λ·grad_jvp
  post_forward_backward() → apply accumulated gradients to params
```

This is the method that benefits *least* from the PEFT tuner pattern but
benefits *most* from the training-loop hooks.

---

## 5. What You Gain from the Fork

### 5.1 Inherited from PEFT for free

| Capability | Value |
|---|---|
| Adapter save/load (`save_pretrained`/`from_pretrained`) | Save CL state alongside adapter weights |
| Multi-adapter management | Switch between CL strategies at runtime |
| Merge/unmerge | Merge adapted weights back into base model |
| Quantization support (QLoRA, GPTQ, AWQ) | CL on quantized models |
| HF Hub integration | Push/pull CL-adapted models |
| `PeftModel` wrapping | Consistent API for any base model |
| 20+ existing PEFT methods | Combine CL with LoRA, IA3, prefix tuning, etc. |

### 5.2 New capabilities from the CL layer

| Capability | Value |
|---|---|
| Drift detection | Statistical monitoring (ADWIN, KSWIN, Page-Hinkley) |
| Automatic CL dispatch | Monitor detects drift → pauses → trains → resumes |
| Forgetting prevention | EWC/KFAC regularization during adaptation |
| Replay integration | Historical batch support in training loop |
| CL + PEFT composition | e.g., LoRA + EWC: parameter-efficient AND forgetting-resistant |

### 5.3 What doesn't map cleanly

| BaseSim Concept | Issue |
|---|---|
| `BaseModelHarness` | PEFT models are HF `PreTrainedModel`s. The harness's data-stream management and `update_data_stream()` would need to be refactored into a dataset wrapper or callback. |
| `update_data_stream()` / affine drift simulation | Specific to BaseSim's experiment design. Would become an example/benchmark, not part of the core library. |
| TOML config system | PEFT uses `PeftConfig` dataclasses. BaseSim's TOML→frozen-dataclass pattern would be replaced by PEFT's config system. |
| FLOPSProfiler | Orthogonal to PEFT. Could be a separate utility or dropped. |

---

## 6. Implementation Strategy

### Phase 1: Fork + Scaffolding
1. Fork `huggingface/peft`
2. Add `src/peft/cl_methods/` with `BaseCLMethod` and config base class
3. Add `src/peft/monitoring/` with `BaseDriftDetector`, `DriftSignal`
4. Register CL method types in the mapping system

### Phase 2: Port CL Methods
1. Port `OnlineKFACUpdater` → `OnlineKFACMethod` (best fit, start here)
2. Port `OnlineEWCUpdater` → `OnlineEWCMethod`
3. Port `JVPRegUpdater` → `JVPRegMethod`
4. Port `BaseUpdater` → base `BaseCLMethod` (vanilla SGD after drift)

### Phase 3: Port Monitoring
1. Port `BaseDriftDetector` + all detector implementations
2. Port `ContinuousMonitor` → adapted to work with HF Trainer
3. Wire up `CLTrainer` with monitor integration

### Phase 4: Integration Tests
1. LoRA + EWC on a streaming classification task
2. Adapter + KFAC on a vision transformer with simulated drift
3. Benchmark against standalone BaseSim to verify equivalence

---

## 7. Risks and Trade-offs

| Risk | Mitigation |
|---|---|
| PEFT upstream moves fast — fork maintenance burden | Keep CL additions in clearly separated directories; minimize changes to existing PEFT code |
| `BaseTuner` and `BaseCLMethod` are fundamentally different abstractions | Don't force CL methods into `BaseTuner`. Keep them as a parallel hierarchy that follows PEFT *conventions* without inheriting from `BaseTuner`. |
| JVP method uses `torch.func` which conflicts with some PEFT internals | Isolate JVP's functional transforms; test with `PeftModel.disable_adapter()` context manager |
| HF Trainer coupling | Provide both a `CLTrainer(Trainer)` subclass and a standalone `CLTrainingLoop` for non-HF use |
| Loss of BaseSim's simplicity | The standalone `BaseCLMethod` + hooks pattern remains simple; PEFT complexity is opt-in |

---

## 8. Conclusion

The fork is viable and valuable.  The key insight is:

> **Don't map updaters to tuners.  Add a parallel CL-method axis that hooks
> into the training loop, while keeping PEFT's tuner axis for model-architecture
> modifications.  The two compose naturally: LoRA for parameter efficiency,
> EWC/KFAC for forgetting prevention, and the monitor for autonomous drift
> response.**

The biggest win is the **composition**: no existing library combines
parameter-efficient adaptation (PEFT) with continual-learning regularization
(EWC/KFAC) under a unified API with drift detection.
