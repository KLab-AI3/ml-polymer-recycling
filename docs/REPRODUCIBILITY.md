# 📚 REPRODUCIBILITY.md

*AI-Driven Polymer Aging Prediction & Classification System*
*(Canonical Raman-only Pipeline)*

> **Purpose**
> A single document that lets any new user clone the repo, arquire the dataset, recreate the conda environment, and generate the validated Raman pipeline artifacts.

---

## 1. System Requirements

| Component | Minimum Version | Notes |
|-----------|-----------------|-------|
| Python | 3.10+  | Conda recommended |
| Git | 2.30+ | Any modern version |
| Conda | 23.1+ | Mamba also fine |
| OS  | Linux / MacOS / Windows | CPU run (no GPU needed) |
| Disk | ~1 GB | Dataset + artifacts |

---

## 2. Clone Repository

```bash
git clone https://github.com/dev-jaser/ai-ml-polymer-aging-prediction.git
cd ai-ml-polymer-aging-prediction
git checkout main
```

---

## 3. Create & Activate Conda Environment

```bash
conda env create -f environment.yml
conda activate polymer_env
```

> **Tip:** If you already created `polymer_env` just run `conda activate polymer_env`

---

## 4. Download RDWP Raman Dataset

1. Visit https://data.mendeley.com/datasets/kpygrf9fg6/1
2. Download the archive (**RDWP.zip or similar**) by clicking `Download Add 10.3 MB`
3. Extract all `*.txt` Raman files into:

```bash
ai-ml-polymer-aging-prediction/datasets/rdwp
```

4. Quick sanity check:

```bash
ls datasets/rdwp | grep ".txt" | wc -l # -> 170 + files expected
```

---

## 5. Validate the Entire Pipeline

Run the canonical smoke-test harness:

```bash
./validate_pipeline.sh
```

Successful run prints:

```bash
[PASS] Preprocessing
[PASS] Training & artificats
[PASS] Inference
[PASS] Plotting
All validation checks passed!
```

Artifacts created:

```bash
outputs/figure2_model.pth
outputs/logs/raman_figure2_diagnostics.json
outputs/inference/test_prediction.json
outputs/plots/validation_plot.png
```

---

## 6. Optional: Train ResNet Variant

```python
python scripts/train_model.py --model resnet --target-len 4000 --baseline --smooth --normalize
```

Check that these exist now:

```bash
outputs/resnet_model.pth
outputs/logs/raman_resnet_diagnostics.json
```

---

## 7. Clean-up & Re-Run

To re-run from a clean state:

```bash
rm -rf outputs/*
./validate_pipeline.sh
```

All artifacts will be regenerated.

---

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError` during scripts| `conda activate polymer_env` not done | Activate env|
| `CUDA not available` warning | Running on CPU | Safe to ignore |
| Fewer than 170 files in `datasets/rdwp` | Incomplete extract | Re-download archive |
| `validate_pipeline.sh: Permission denied` | Missing executable bit | `chmod +x validated_pipeline.sh` |

---

## 9. Contact

For issues or questions, open an Issue in the GitHub repo or contact @dev-jaser
