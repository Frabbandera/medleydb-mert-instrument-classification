# Experiment notebooks guide

The notebooks are readable interfaces to reusable code. They do not reimplement
models or training loops and are committed without outputs. Each experiment
notebook exposes dictionaries of config groups plus `SELECTED_EXPERIMENTS` so
one config can be run now and another later without editing Python source.

## Recommended order

1. `pipeline_medleydb.ipynb`: Colab authentication, Drive paths, clean subset
   creation, default MERT cache extraction, and one smoke/full experiment.
2. `00_dataset_exploration_medleydb.ipynb`: dataset health, label mapping,
   subset summaries, leakage checks, and report-ready dataset plots.
3. `01_frozen_mert_classifier_head.ipynb`: classical MFCC/spectral SVM baseline
   and frozen MERT classifier-head ablation.
4. `02_mert_representation_ablation.ipynb`: layer, aggregation, pooling, and
   MERT model-size ablations.
5. `03_mert_finetuning_experiments.ipynb`: direct frozen MERT and partial
   fine-tuning; GPU-heavy and intentionally one-at-a-time.
6. `04_segment_and_training_protocol_ablation.ipynb`: segment duration,
   subset_40_per_class, capped_natural imbalance handling, and seed repeats.
7. `05_polyphonic_mixture_dataset.ipynb`: build synthetic random, same-song
   same-time, and original full-mix manifests.
8. `06_polyphonic_mert_multilabel.ipynb`: extract mixture embeddings and train
   the multi-label classifier.
9. `07_overlap_robustness_analysis.ipynb`: compute metrics grouped by number of
   active instruments.
10. `08_genre_instrumentation_cooccurrence.ipynb`: analyze instrumentation and
    co-occurrence context.
11. `09_error_analysis_audio_examples.ipynb`: inspect selected correct and
    incorrect predictions with audio examples.
12. `99_isolated_stem_final_summary.ipynb`: registry-only final comparison; no
    training or extraction.

## Paths and execution

Launch locally from the repository root or set `PROJECT_ROOT`. Set `RUN_ROOT`
for generated metadata, caches, results, checkpoints, and registry rows. Set
`MEDLEYDB_ROOT` to the local or mounted dataset location.

On Colab, the intended shared Drive layout is:

```text
MyDrive/medleydb_mert_project/
  isolated_stem_v1/
    data/
      metadata/
      cache/
      reports/
    checkpoints/
    results/
```

Repository YAML files use relative generated paths such as `data/...` and
`results/...`. When `RUN_ROOT` is set, the project resolver maps those paths
under Drive, preserving nested cache and ablation folders.

If an experiment ID already exists, the runner stops. Use `--replace-existing`
or `REPLACE_EXISTING = True` only when intentionally replacing its checkpoints,
results, and registry row.
