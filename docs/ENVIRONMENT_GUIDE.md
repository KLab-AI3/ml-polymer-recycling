# 🔧 Environment Management Guide

## AI-Driven Polymer Aging Prediction and Classification System

**Maintainer:** Jaser Hasan
**Snapshot:** `@artifact-isolation-complete`
**Last Updated:** 2025-06-26
**Environments:** Conda (local) + venv on `/scratch` (HPC)

---

## 🧠 Overview

This guide describes how to set up and activate the Python environments required to run the Raman pipeline on both:

- **Local Systems:** (Mac/Windows/Linux)
- **CWRU Pioneer HPC:** (GPU nodes, venv based)

This guide documents the environment structure and the divergence between the **local Conda environment (`polymer_env`)** and the **HPC Python virtual environment (`polymer_venv`)**.

---

## 📁 Environment Overview

| Platform | Environment | Manager | Path | Notes |
|----------|-------------|---------|------|-------|
| Local (dev) | `polymer_env` | **Conda** | `~/miniconda3/envs/polymer_env` | Primary for day-to-day development |
| HPC (Pioneer) | `polymer_venv` | **venv** (Python stdlib) | `/scratch/users/<case_id>/polymer_project/polymer_venv` | Created under `/scratch` to avoid `/home` quota limits |

---

## 💻 Local Installation (Conda)

```bash

git clone https://github.com/dev-jaser/ai-ml-polymer-aging-prediction.git
cd polymer_project
conda env create -f environment.yml
conda activate polymer_env
python -c "import torch, sys; print('PyTorch:', torch.__version__, 'Python', sys.version")
```

> **Tip:** Keep Conda updated ('conda update conda') to reduce solver errors issues.

---

## 🚀 CWRU Pioneer HPC Setup (venv + pip)

> Conda is intentionally **not** used on Pioneer due to prior codec and disk-quota

### 1. Load Python Module

```bash

module purge
module load Python/3.12.3-GCCcore-13.2.0
```

### 2. Create Working Directory in `/scratch`

```bash

mkdir -p /scratch/users/<case_id>/polymer_project_runtime
cd /scratch/users/<case_id>/polymer_project_runtime
git clone https://github.com/dev-jaser/ai-ml-polymer-aging-prediction.git
```

### 3. Create & Activate Virtual Environment

```bash

python3 -m venv polymer_env
source polymer_env/bin/activate
```

### 4. Install Dependencies

```bash

pip install --upgrade pip
pip install -r environment_hpc.yml      # Optimized dependencies list for Pioneer
```

(Optional) Save a reproducible freeze:

```bash

pip freeze > requirements_hpc.txt
```

---

## ✅ Supported CLI Workflows (Raman-only)

| Script | Purpose |
|--------|---------|
| `scripts/train_model.py` | 10-fold CV training ('--model figure2' or 'resnet') |
| `scripts/run_inference.py` | Predict single Raman spectrum |
| `scripts/preprocess_dataset.py` | Apply full preprocessing chain |
| `scripts/plot_spectrum.py` | Quick spectrum visualization (.png) |

> FTIR-related scripts are archived and *not installed* into the active environments.

---

## 🔁 Cross-Environment Parity

- Package sets in environment.yml and environment_hpc.yml are aligned.
- Diagnostics JSON structure and checkpoint filenames are identical on both systems.
- Training commands are copy-paste compatible between local shell and HPC login shell.

---

## 📦 Best Practices

- **Local:** use Conda for rapid iteration, notebook work, small CPU inference.
- **HPC:** use venv in  `/scratch` for GPU training, never install large packages into `/home` (`'~/'`)
- Keep environments lightweight; remove unused libraries to minimize rebuild time.
- Update this guide if either environment definition changes.
