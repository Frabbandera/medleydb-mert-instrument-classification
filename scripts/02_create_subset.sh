#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/datasets/subset_largest_balanced_medleydb_instrument.yaml}"

python -m src.data.create_balanced_subset \
  --config "$CONFIG"
