# Sprint Log

## @model-expansion-preflight-2025-08-21
**Goal:** Reinforce training script contracts and registry hook without behavior changes.
**Changes:**
- Reproducibility seeds (python/numpy/torch/cuda).
- Optional cuDNN deterministic settings.
-Typo fix: "Reseample" -> "Resample".
- Diagnostics fix: per-fold accuracy logs use correct variable.
- Explicit dtypes in TensorDataset (float32/long).
**Tests:**
- Preprocess: ✅
- Train (figure2, 1 epoch): ✅
- Inference smoke: ✅
**Notes:** Baseline intact; high CV variance due to class imbalance recorded for later migration.

## @model-expansion-registry-2025-08-21
**Goal:** Make model lookup a single source of truth and expose dynamic choices for CLI/infra.
**Changes:** 
- Added `models/registry.py` with `choices()` and `build()` helpers.
- `scripts/train_model.py` imports registry, uses `choices()` for argparse and `build()` for contruction.
- Removed direct model selection logic from training script. 
**Tests:**
- Train (figure2) via registry: ✅
- Inference unchanged paths: ✅
**Notes:** Artifacts remain `outputs/{model}_model.pth` to avoid breaking validator; inference arch flag to be added next.
## @model-expansion-resnet18vision-2025-08-21
**Goal:** Introduce a second architecture and prove multi-model training/inference via shared registry.
**Changes:** `models/resnet18_vision.py` (1D), registry entry, `run_inference.py --arch`.
**Tests:** Train (1 epoch) -> `outputs/resnet18vision_model.pth`; Inference JSON ✅
**Notes:** Backward compatibility preserved (`--arch` defaults to figure2).
