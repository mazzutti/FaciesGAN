## Plan: Integrate NVIDIA Apex Library Across FaciesGAN

**TL;DR** — Replace native PyTorch optimizers, normalization layers, mixed-precision (AMP), and gradient-reduction utilities with their NVIDIA Apex equivalents. Apex is always-on (import-time failure if not installed). Five areas are touched: **FusedAdam** (optimizer), **FusedLayerNorm** (interpolator), **FusedInstanceNorm** (conv blocks), **apex.amp O1** (mixed precision), and **Apex parallel flatten/unflatten** (all-reduce). The largest refactor is replacing `torch.amp.autocast` + `GradScaler` with `apex.amp.initialize` + `amp.scale_loss`.

**Steps**

1. **Add `apex` to [requirements.txt](requirements.txt)** — append `apex` (or `nvidia-apex`) to the dependency list.

2. **Create an Apex compatibility shim** — add a new file `apex_utils.py` at the project root that centralises all Apex imports. This file eagerly imports Apex at module level and exposes:
   - `from apex.optimizers import FusedAdam`
   - `from apex.normalization import FusedLayerNorm`
   - `from apex import amp` (for O1/O2 mixed precision)
   - `from apex.parallel import flatten as apex_flatten, unflatten as apex_unflatten` (coalesced gradient utilities)
   - A try-import of `from apex.contrib.instance_norm import InstanceNorm2dNVFuser` (may not exist in all Apex builds — fall back to a custom `nn.InstanceNorm2d` wrapper if unavailable).

3. **Replace `torch.optim.Adam` → `apex.optimizers.FusedAdam`** in [trainning/torch/trainer.py](trainning/torch/trainer.py#L519-L538) (`setup_optimizers`):
   - Remove the `fused=` and `foreach=` kwargs (FusedAdam doesn't support those — it's always fused).
   - Two call sites: discriminator optimizer (L519) and generator optimizer (L532).
   - `FusedAdam` is a drop-in replacement: same constructor args (`params, lr, betas, eps, weight_decay`).
   - LR schedulers (`MultiStepLR`) remain unchanged — they wrap any optimizer.

4. **Replace `nn.LayerNorm` → `apex.normalization.FusedLayerNorm`** in [interpolators/neural.py](interpolators/neural.py#L369-L388) (`ResidualMLP.__init__`):
   - Add import: `from apex.normalization import FusedLayerNorm`.
   - Four substitutions in `layer1`, `layer2`, `skip_layer`, `layer3` — same constructor: `FusedLayerNorm(hidden_dim)`.

5. **Replace `nn.InstanceNorm2d` → `InstanceNorm2dNVFuser`** in [models/torch/custom_layer.py](models/torch/custom_layer.py):
   - Three sites: `TorchConvBlock` (L62), `TorchDiscConvBlock` (L107), `TorchSPADE` (L149).
   - Import from `apex.contrib.instance_norm` (gated behind availability check in `apex_utils.py`).
   - Same constructor: `InstanceNorm2dNVFuser(channels, affine=True)`.
   - Update weight-init in [models/torch/utils.py](models/torch/utils.py#L138) `weights_init()` to also recognise `InstanceNorm2dNVFuser` in the `isinstance` check.

6. **Replace native `torch.amp` with `apex.amp`** in [models/torch/facies_gan.py](models/torch/facies_gan.py):

   This is the largest change. The current code uses `torch.amp.autocast` + two `GradScaler` objects. Apex's `amp.initialize` + `amp.scale_loss` replaces both.

   **6a. Remove `GradScaler` imports and creation** (L17-L18, L101-L102):
   - Remove `from torch.amp.autocast_mode import autocast` and `from torch.amp.grad_scaler import GradScaler`.
   - Remove `self._grad_scaler_g` and `self._grad_scaler_d`.
   - Add `from apex import amp`.

   **6b. Call `amp.initialize` when models are ready** — after `self.setup_framework()` (around L130):
   - Call `amp.initialize(self, opt_level="O1", verbosity=0)` on the top-level `TorchFaciesGAN` module. This patches all sub-modules (generator, discriminator, color_quantizer) for O1 mixed precision.
   - O1 ("mixed precision") keeps model weights in fp32 but casts inputs/outputs of allowlisted ops (matmul, conv) to fp16 automatically — no explicit `autocast` needed.
   - Since per-scale optimizers don't exist yet at init time, pass `optimizers=None`. Optimizers are registered later via step 6d.

   **6c. Replace `autocast` regions**:
   - [L335](models/torch/facies_gan.py#L335) (D non-GP step): remove the `with autocast(...)` wrapper. O1 handles casting automatically. For GP steps where AMP must be **disabled**, use `with amp.disable_casts():` around the GP computation.
   - [L562](models/torch/facies_gan.py#L562) (G forward): remove `with autocast(...)` wrapper.
   - [L727](models/torch/facies_gan.py#L727) (GP fp32): replace `with autocast("cuda", enabled=False):` with `with amp.disable_casts():`.

   **6d. Replace `GradScaler.scale(loss).backward()` → `amp.scale_loss`**:
   - [D step L354-L358](models/torch/facies_gan.py#L354-L358): replace with
     `with amp.scale_loss(total, optimizers[scale]) as scaled: scaled.backward()`
   - [G step L471](models/torch/facies_gan.py#L471): replace `self._grad_scaler_g.scale(metrics.total).backward()` with `with amp.scale_loss(metrics.total, [optimizers[s] for s in sorted_scales]) as scaled: scaled.backward()` — Apex handles multi-optimizer scaling.
   - [Legacy G L1289](models/torch/facies_gan.py#L1289): same pattern with `amp.scale_loss`.

   **6e. Remove all `unscale_()` / `step()` / `update()` calls** — `amp.scale_loss` context manager handles unscaling automatically before `.backward()` returns. The optimizer `.step()` calls remain but the explicit scaler orchestration is removed:
   - Remove L379-L385 (D scaler unscale/step/update).
   - Remove L496-L509 (G scaler unscale/step/update) — keep only the `optimizer.step()` calls.
   - Remove L1292-L1295 (legacy G scaler).

   **6f. Handle per-scale optimizer registration**:
   - In [trainer.py `setup_optimizers`](trainning/torch/trainer.py#L516) after creating each `FusedAdam`, call `amp.initialize(model_block, optimizer, opt_level="O1", cast_model_type=None)` per-scale so Apex can track the optimizer's master-weight state. Use `cast_model_type=None` to avoid re-patching model layers (already done in 6b).

   **6g. AMP state save/restore**:
   - Add `amp.state_dict()` alongside model checkpoints in save routines.
   - Add `amp.load_state_dict(...)` in resume routines.
   - Update the checkpoint loading in the trainer and [resume.py](resume.py).

7. **Replace `torch._utils._flatten_dense_tensors` → `apex.parallel.flatten`** in [facies_gan.py `_allreduce_grads_coalesced`](models/torch/facies_gan.py#L1222-L1244):
   - `from apex.parallel import flatten as apex_flatten, unflatten as apex_unflatten`
   - Replace `torch._utils._flatten_dense_tensors(grads)` → `apex_flatten(grads)`
   - Replace `torch._utils._unflatten_dense_tensors(flat, grads)` → `apex_unflatten(flat, grads)`
   - These are the canonical Apex utility functions (the `torch._utils` variants were copied from Apex into PyTorch core).

8. **Suppress `TORCH_LOGS` override** — in [main.py L16-L18](main.py#L16-L18), the current code sets `TORCH_LOGS=-dynamo` to suppress `torch.compile` warnings. With Apex AMP (O1), `torch.compile` still works but Apex issues its own diagnostics. Add `amp.register_half_function` / `amp.register_float_function` calls in `apex_utils.py` if any custom layers need special handling.

9. **Verify weight-init compatibility** — in [models/torch/utils.py `weights_init`](models/torch/utils.py#L136-L141), extend the `isinstance` check to include the Apex norm types:
   ```
   isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, FusedLayerNorm, InstanceNorm2dNVFuser))
   ```

**Verification**

- `pip install nvidia-apex` (or build from source with `--cuda_ext --cpp_ext`).
- Run the DDP training task: `torchrun --nproc_per_node=2 main.py --input-path data ...` — verify no import errors and training progresses.
- Compare first-epoch loss values against a pre-Apex baseline to confirm numerical parity (O1 should be nearly identical).
- Check GPU utilization with `nvidia-smi` — FusedAdam should show higher GPU kernel occupancy.
- Run `TORCH_LOGS=recompiles` to confirm `torch.compile` still works with Apex-patched modules.

**Decisions**

- **apex.amp O1 over O2**: O1 is chosen because the codebase explicitly needs fp32 for gradient penalty (`create_graph=True`). O2 casts model weights to fp16 which would break GP. O1 keeps master weights in fp32 and only casts activations.
- **Per-scale `amp.initialize`**: Since optimizers are created dynamically per-scale, each (sub-model, optimizer) pair is registered with Apex separately. `cast_model_type=None` on per-scale calls avoids double-patching layers.
- **`InstanceNorm2dNVFuser` gated import**: This Apex contrib module may not be available in all Apex builds. The `apex_utils.py` shim falls back to `nn.InstanceNorm2d` if import fails, with a runtime warning.
- **Always-on**: No `--use-apex` CLI flag. Apex is a hard dependency — import failure crashes immediately with a clear error message.
