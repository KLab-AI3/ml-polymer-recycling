#!/usr/bin/env python3
"""
audit.py - quick audit tool for preprocessing baseline

Searches for relevant keywords in the ml-polymer-recycling repo
to confirm what preprocessing steps (resample, baseline, smooth,
normalize, etc.) are actually implemented in code/docs.
"""

import re
from pathlib import Path

# ||== KEYWORDS TO TRACE ==||
KEYWORDS = [
    "resample", "baseline", "smooth", "Savitz",
    "normalize", "minmax" "TARGET_LENGTH", "WINDOW_LENGTH",
    "POLYORDER", "DEGREE", "input_length", "target_len", "Figure2CNN", "ResNet"
]

# ||==== DIRECTORIES/FILES TO SCAN ====||
TARGETS = [
    "scripts/preprocess_dataset.py",
    "scripts/run_inferece.py",
    "models/",
    "utils/",
    "README.md",
    "GROUND_TRUTH_PIPELINE.md",
    "docs/"
]

# ||==== COMPILE REGEX FOR KEYWORDS  ====||
pattern = re.compile("|".join(KEYWORDS), re.IGNORECASE)

def scan_file(path: Path):
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if pattern.search(line):
                    print(f"{path}:{i}: {line.strip()}")
    except Exception as e:
        print(f"[ERR] Could not read {path}: {e}")

def main():
    root = Path(".").resolve()
    for target in TARGETS:
        p = root / target
        if p.is_file():
            scan_file(p)
        elif p.is_dir():
            for sub in p.rglob("*.py"):
                scan_file(sub)
            for sub in p.rglob("*.md"):
                scan_file(sub)

if __name__ == "__main__":
    main()