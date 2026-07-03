#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml}"
CHECKPOINT="${2:-checkpoints/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02/best.ckpt}"

python -m src.training.evaluate_classifier \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT"
