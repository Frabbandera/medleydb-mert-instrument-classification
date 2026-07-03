#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml}"

python -m src.features.extract_mert_embeddings \
  --experiment-config "$CONFIG" \
  --batch-size "${BATCH_SIZE:-1}" \
  --device "${DEVICE:-auto}"
