# Local CPU/GPU commands and troubleshooting

Run every command from the repository root. On Colab, set `RUN_ROOT` to the
shared Drive artifact folder and `MEDLEYDB_ROOT` to the MedleyDB audio folder.

## Environment

```bash
python -m pip install -r requirements.txt
python -m compileall -q src tests
python -m pytest -q
```

## 1. Build the stem index

```bash
export RUN_ROOT="${RUN_ROOT:-$PWD}"
export MEDLEYDB_ROOT="${MEDLEYDB_ROOT:-MedleyDB}"
python -m src.data.build_stem_index --medleydb-root "$MEDLEYDB_ROOT" --out "$RUN_ROOT/data/metadata/stem_index.csv" --report-dir "$RUN_ROOT/data/reports"
```

## 2. Create the clean isolated-stem subset

```bash
python -m src.data.create_balanced_subset --config configs/datasets/subset_largest_balanced_medleydb_instrument.yaml
```

Protocol ablation datasets are built the same way from `configs/datasets/`, for
example:

```bash
python -m src.data.create_balanced_subset --config configs/datasets/subset_largest_balanced_medleydb_instrument_2s.yaml
python -m src.data.create_balanced_subset --config configs/datasets/subset_capped_natural_medleydb_instrument.yaml
```

## 3. Extract matching MERT embeddings

```bash
python -m src.features.extract_mert_embeddings --experiment-config configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml --batch-size 1 --device auto
```

Every layer, pooling mode, model size, segment duration, and data protocol has a
separate cache directory. Matching caches are skipped; pass `--overwrite` only
after intentionally replacing a cache.

## 4. Run one registered experiment

```bash
python -m src.experiments.run_experiment --config configs/experiments/classical_largest_balanced_medleydb_mfcc_svm.yaml
python -m src.experiments.run_experiment --config configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml
```

The runner saves metrics, reports, confusion matrices for single-label runs,
predictions, resolved config, checkpoints or serialized classical model, and one
registry row after successful evaluation.

## 5. Run polyphonic protocols

```bash
python -m src.data.create_mixture_manifest --config configs/mixtures/largest_balanced_synthetic_k.yaml
python -m src.features.extract_mert_mixture_embeddings --experiment-config configs/experiments/polyphonic_largest_balanced_synthetic_k_mert95_last_mean_mlp.yaml --batch-size 1 --device auto
python -m src.experiments.run_experiment --config configs/experiments/polyphonic_largest_balanced_synthetic_k_mert95_last_mean_mlp.yaml
```

Swap the config path for `polyphonic_largest_balanced_same_song_same_time...` or
`polyphonic_largest_balanced_full_mix...` after building the matching manifest.

## Troubleshooting

- Missing cache: run the extractor with the exact experiment config.
- Existing result: use a new `experiment_id` or deliberately pass
  `--replace-existing`.
- CUDA out of memory: use extraction `--batch-size 1`; avoid the 330M configs
  until the 95M grid is complete.
- Hugging Face model-code issues: MERT uses `trust_remote_code=True`; pin a
  known working Transformers 4.x release if needed.
- Windows paths: quote paths containing spaces and keep `num_workers: 0` unless
  multiprocessing has been tested.
