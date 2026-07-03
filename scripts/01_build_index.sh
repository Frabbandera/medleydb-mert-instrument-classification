#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-.}"
MEDLEYDB_ROOT="${MEDLEYDB_ROOT:-MedleyDB}"

python -m src.data.build_stem_index \
  --medleydb-root "$MEDLEYDB_ROOT" \
  --out "$RUN_ROOT/data/metadata/stem_index.csv" \
  --report-dir "$RUN_ROOT/data/reports"
