#!/usr/bin/env bash
# ===========================================
# validate_pipeline.sh — Canonical Smoke Test
# AI-Driven Polymer Aging Prediction System
# Requires: conda (or venv) already installed
# ===========================================

set -euo pipefail
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
NC='\033[0m'

die() {
    echo -e "{RED}[FAIL] $1${NC}"
    exit 1
}
pass() { echo -e "{GRN}[PASS] $1${NC}"; }

echo -e "${YLW}>>> Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polymer_env || die "conda env 'polymer_env' not found"

root_dir="$(dirname "$(readlink -f "$0")")"
cd "$root_dir" || fir "repo root not found"

# ---------- Step 1: Preprocessing ----------
echo -e "${YLW}>>> Step 1: Preprocessing${NC}"
python scripts/preprocess_dataset.py datasets/rdwp \
    --target-len 500 --baseline --smooth --normalize |
    grep -q "X shape:" || die "preprocess_dataset.py failed"
pass "Preprocessing"

# ---------- Step 2: CV Training (Figure2) ----------
echo -e "${YLW}>>> Step 2: 10-Fold CV Training${NC}"
python scripts/train_model.py \
    --target-len 500 --baseline --smooth --normalize \
    --model figure2
[[ -f outputs/figure2_model.pth ]] || die "model .pth not found"
[[ -f outputs/logs/raman_figure2_diagnostics.json ]] || die "diagnostics JSON not found"
pass "Training & artifacts"

# ---------- Step 3: Inference ----------
echo -e "${YLW}>>> Step 3: Inference${NC}"
python scripts/run_inference.py \
    --target-len 500 \
    --input datasets/rdwp/wea-100.txt \
    --model outputs/figure2_model.pth \
    --output outputs/inference/test_prediction.json
[[ -f outputs/inference/test_prediction.json ]] || die "inference output missing"
pass "Inference"

# ---------- Step 4: Spectrum Plot ----------
echo -e "${YLW}>>> Step 4: Plot Spectrum${NC}"
python scripts/plot_spectrum.py --input datasets/rdwp/sta-10.txt
[[ $? -eq 0 ]] || die "plot_spectrum.py failed"
pass "Plotting"


echo -e "${GRN}All validation checks passed!${NC}"
