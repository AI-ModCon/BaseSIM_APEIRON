## Summary

- **Add ImageNet example**: new `examples/imagenet/` harness with model loading (ViT + CNN families), ImageNet data pipelines, and affine-drift transforms for continual learning experiments
- **Refactor GPU selection**: set `CUDA_VISIBLE_DEVICES` *before* CUDA init using `nvidia-smi`, replacing the previous post-init `torch.cuda.mem_get_info` approach — avoids silent multi-GPU memory allocation bugs
- **Handle missing historical data gracefully**: `history_eval()` now returns `None` when no prior tasks exist; `ContinuousTrainer` logs and metrics adapt accordingly instead of crashing
- **Switch monitoring to validation loader**: `ContinuousMonitor` now iterates `val_loader` (not `train_loader`) for drift detection, with a `tqdm` progress bar
- **Begin strict typing + ruff compliance**: type annotations on `main()`, `history_eval()`; all code passes `ruff check`

## Changed Files

| Area | Files | What |
|------|-------|------|
| New example | `examples/imagenet/model.py`, `examples/imagenet/src/utils.py`, `examples/imagenet/src/__init__.py` | ImageNet harness, data utils, model loader |
| Core | `src/config/configuration.py` | GPU selection refactor (`_select_best_gpu`) |
| Core | `src/driver/continuous_monitor.py` | Use val loader + tqdm, remove dead code |
| Core | `src/training/continuous_trainer.py` | Handle `None` historical metrics |
| Core | `src/model/torch_model_harness.py` | `history_eval()` → `Optional[List[float]]` |
| Core | `src/main.py` | Type annotation on `main()` |
| Updaters | `src/training/updater/ewc.py`, `kfac.py` | Add paper references to docstrings |
| Config | `examples/mnist/mnist.toml` | `num_workers` 1 → 4 |
| Plumbing | `examples/utils.py`, `.gitignore`, `README.md` | Register imagenet example, ignore `*.toml`, add run commands |

## Test Plan

- [ ] Verify MNIST example still runs end-to-end (`poetry run python -m src.main --config examples/mnist/mnist.toml`)
- [ ] Verify ImageNet example loads correctly (requires ImageNet data at configured path)
- [ ] Confirm single-GPU selection works via `CUDA_VISIBLE_DEVICES` on a multi-GPU node
- [ ] Check that first-task scenario (no historical data) logs cleanly without errors

N/A: breaking changes, migrations, rollback plan, dependencies added/removed, security considerations.
