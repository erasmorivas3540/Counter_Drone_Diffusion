#!/usr/bin/env bash
# Zip the project (code, results, checkpoints; no figures cache or pycache) for off-machine backup.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/mnt/user-data/outputs/counter-drone-diffusion-backup.zip}
rm -f "$OUT"
zip -q -r "$OUT" src tests experiments results figures notebooks pyproject.toml README.md \
    -x "*/__pycache__/*" "*.egg-info/*" 2>/dev/null || zip -q -r "$OUT" src experiments results figures pyproject.toml README.md -x "*/__pycache__/*"
echo "$OUT $(du -h "$OUT" | cut -f1)  checkpoints: $(cat results/checkpoints/*.jsonl 2>/dev/null | wc -l) records"
