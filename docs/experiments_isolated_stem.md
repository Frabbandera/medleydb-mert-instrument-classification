# Isolated-stem experiment plan

The final isolated-stem protocol is `largest_balanced + medleydb_instrument`.
Each notebook changes one report-motivated factor while keeping the split,
label space, and artifact layout auditable through YAML configs.

## Architecture

- Notebooks launch experiments and define short `SELECTED_EXPERIMENTS` lists.
- YAML files in `configs/experiments/` and `configs/ablations/` define the
  actual experiment settings.
- Cached embeddings are reusable only when data, split, MERT model, model
  revision, layer, pooling, label granularity, subset profile, and segment
  duration match. Cache metadata is checked before reuse.
- Results and checkpoints are stored per `experiment_id`; polyphonic configs use
  a run layout with `seed_<seed>/<run_id>`.
- `results/experiment_registry.csv` is the comparison table for completed runs.
- New experiments should normally require only a new YAML config and a notebook
  list entry, not source-code modification.

## 1. Classical baseline and frozen classifier heads

The classical baseline uses MFCC and spectral descriptor statistics with an SVM
on the same largest-balanced MedleyDB instrument split. The frozen MERT head
ablation reuses the MERT-v1-95M last-layer mean-pooling cache and varies only
head capacity, dropout, and one learning rate.

Configs:

- `configs/experiments/classical_largest_balanced_medleydb_mfcc_svm.yaml`
- `configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_linear.yaml`
- `configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d00.yaml`
- `configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml`
- `configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d05.yaml`
- `configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h512_d02.yaml`
- `configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02_lr3e4.yaml`

## 2. Representation, pooling, and model size

The head is fixed to MLP hidden 256 dropout 0.2. Each representation or pooling
variant has a separate cache, and model-size runs use separate `mert_v1_330m`
cache folders.

Configs include layer 6, layer 9, last, last3avg, last6avg, all-layer learned
softmax weighting, max pooling, meanmax pooling, MERT-v1-330M last mean, and an
optional MERT-v1-330M all-layer weighted run.

## 3. Direct MERT / fine-tuning

These runs use the same subset but load audio on the fly. Batch size 1, mixed
precision, gradient checkpointing, and accumulation are set for Colab GPUs.
Full fine-tuning is disabled in the final configs.

Configs:

- `configs/experiments/direct_largest_balanced_medleydb_mert95_frozen.yaml`
- `configs/experiments/finetune_largest_balanced_medleydb_mert95_last1.yaml`
- `configs/experiments/finetune_largest_balanced_medleydb_mert95_last2.yaml`

## 4. Protocol and seed ablations

`configs/datasets/` contains dataset builders for 2 s, 5 s, 10 s,
`subset_40_per_class`, and `capped_natural`. Matching experiment configs live in
`configs/ablations/`. Segment-length and protocol changes require rebuilding the
subset CSV and extracting a matching cache. Seed repeats reuse the default 5 s
cache.

## 5. Polyphonic follow-up

The polyphonic notebooks should normally use the best isolated-stem
representation selected above. The default configs keep MERT-v1-95M last-layer
mean pooling so the baseline is fixed and easy to swap.

Configs cover synthetic random mixtures with controlled overlap k, same-song
same-time reconstructed mixtures, and original MedleyDB full mixes.

## Comparison rule

Use validation macro-F1 for model selection. Test macro-F1, weighted-F1,
accuracy, per-class metrics, confusion matrices, and predictions support the
report after selection. Do not rank different data protocols as if they were the
same task; group by `subset_profile`, `segment_seconds`, and `mixture_mode`.
