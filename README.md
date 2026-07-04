# MedleyDB MERT Instrument Classification

## Overview

This repository documents a team-developed machine-learning workflow for automatic musical-instrument recognition on MedleyDB. The project compares classical audio descriptors with self-supervised MERT representations, first on isolated stems and then on controlled polyphonic mixtures.

The repository is intended as a technical portfolio and research-code case study. It emphasises dataset construction, leakage-aware splitting, validation-based model selection, ablation studies, error analysis and multi-label robustness, rather than deployment of an end-user application.

## Context

Academic project for the Selected Topics in Music and Acoustic Engineering course in the MSc programme in Music and Acoustic Engineering at Politecnico di Milano.

The project was developed by a four-person team and addresses the course task of music instrument classification. The public repository includes source code, configuration files, notebooks, tests and the final report, but it does not include MedleyDB audio, generated mixtures, cached embeddings, model checkpoints or large result folders.

## Objective

The project aimed to evaluate how different audio representations and classification protocols affect instrument-recognition performance on real multitrack music data.

The main objectives were to:

- construct a controlled MedleyDB v1 subset for instrument classification;
- prevent data leakage through track-disjoint train/validation/test splits;
- compare handcrafted MFCC/spectral descriptors with self-supervised MERT embeddings;
- evaluate frozen classifier heads, representation layers, pooling strategies and MERT model sizes;
- test limited MERT fine-tuning under computational constraints;
- extend the isolated-stem setting to controlled synthetic polyphonic mixtures;
- analyse multi-label robustness as the number of active instruments increases.

## Methods

### Dataset construction

The isolated-stem experiments use MedleyDB v1 processed stems. Stems marked with bleed were excluded for the single-label setting, and the final task uses a balanced 12-class subset with 160 five-second segments per class. Splits are performed at track level, so segments from the same song do not appear in multiple partitions.

### Classical baseline

A classical baseline extracts MFCC and spectral descriptors from five-second segments and trains an RBF-kernel Support Vector Machine. This baseline provides a computationally efficient reference for comparison with learned audio representations.

### MERT representation learning workflow

The neural experiments use MERT as a self-supervised music-audio representation model. The pipeline evaluates frozen MERT embeddings with lightweight classifier heads, different Transformer layers, learned layer aggregation, temporal pooling strategies and the MERT-v1-95M and MERT-v1-330M checkpoints.

### Fine-tuning and protocol ablations

The project compares cached frozen embeddings with direct-audio MERT training and limited final-layer fine-tuning. Additional protocol ablations examine segment duration, reduced training data, natural-frequency class distributions and alternative training settings.

### Polyphonic multi-label extension

The polyphonic stage generates controlled synthetic mixtures with one to four active instrument classes. The classifier is adapted from single-label softmax classification to twelve independent sigmoid outputs trained with weighted binary cross-entropy.

## Tools and Technologies

- Python
- PyTorch and Lightning
- Hugging Face Transformers
- MERT self-supervised audio representations
- scikit-learn
- librosa and soundfile
- NumPy and pandas
- Matplotlib
- YAML-based experiment configuration
- Jupyter / Google Colab workflow
- pytest-based regression tests

## Repository Structure

```text
medleydb-mert-instrument-classification/
  README.md
  AUTHORS.md
  NOTICE.md
  requirements.txt
  pytest.ini
  configs/
    datasets/
    experiments/
    ablations/
    mixtures/
  src/
    data/
    features/
    models/
    training/
    experiments/
    utils/
  notebooks/
  docs/
    reports/
      automatic_musical_instrument_recognition_with_mert.pdf
    commands.md
    data_subset_protocol.md
    experiments_isolated_stem.md
    google_colab.md
    notebooks_guide.md
    theory_mert_stem_classification.md
  tests/
  data/
    cache/.gitkeep
    metadata/.gitkeep
    reports/.gitkeep
  results/.gitkeep
```

## Key Results

### Isolated-stem classification

The MFCC/spectral SVM baseline reached a held-out test macro-F1 of 0.6752 on the balanced 12-class isolated-stem task. The initial frozen MERT-v1-95M final-layer reference reached 0.6431 test macro-F1.

Representation choice was the most influential factor. Intermediate-layer and learned-layer MERT embeddings improved performance, and MERT-v1-330M with learned all-layer aggregation obtained the highest observed isolated-stem test macro-F1 of 0.7290.

### Fine-tuning

Limited final-layer fine-tuning of MERT-v1-95M produced only a small test improvement over the direct frozen configuration and did not improve the validation macro-F1 used for model selection. Under the available data and computational constraints, frozen representations were therefore more reliable than the tested fine-tuning strategy.

### Polyphonic mixtures

In the controlled synthetic-mixture setting, the frozen MERT-v1-95M multi-label classifier obtained macro-F1 around 0.535 and macro ROC-AUC around 0.787. Exact-match accuracy remained low and deteriorated as the number of active instruments increased.

The main failure mode was not complete absence of recognition. The model often detected part of the active instrument set, but tended to over-predict additional labels or miss acoustically heterogeneous classes.

## How to Run / Reproduce

This repository does not include MedleyDB audio files, cached embeddings, trained checkpoints or generated result folders.

A local run requires:

1. Python 3.10 or later;
2. a CUDA-compatible PyTorch installation for efficient MERT inference;
3. local access to MedleyDB v1;
4. installation of the Python requirements;
5. configuration of `MEDLEYDB_ROOT` and `RUN_ROOT`.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Typical command sequence:

```bash
export RUN_ROOT="$PWD"
export MEDLEYDB_ROOT="/path/to/MedleyDB"

python -m src.data.build_stem_index \
  --medleydb-root "$MEDLEYDB_ROOT" \
  --out "$RUN_ROOT/data/metadata/stem_index.csv" \
  --report-dir "$RUN_ROOT/data/reports"

python -m src.data.create_balanced_subset \
  --config configs/datasets/subset_largest_balanced_medleydb_instrument.yaml

python -m src.features.extract_mert_embeddings \
  --experiment-config configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml \
  --batch-size 1 \
  --device auto

python -m src.experiments.run_experiment \
  --config configs/experiments/isolated_largest_balanced_medleydb_mert95_last_mean_mlp_h256_d02.yaml
```

Run the classical baseline without MERT embedding extraction:

```bash
python -m src.experiments.run_experiment \
  --config configs/experiments/classical_largest_balanced_medleydb_mfcc_svm.yaml
```

## Report and Documentation

The complete technical report is available in:

- [`docs/reports/automatic_musical_instrument_recognition_with_mert.pdf`](docs/reports/automatic_musical_instrument_recognition_with_mert.pdf)

Additional documentation:

- [`docs/commands.md`](docs/commands.md)
- [`docs/data_subset_protocol.md`](docs/data_subset_protocol.md)
- [`docs/experiments_isolated_stem.md`](docs/experiments_isolated_stem.md)
- [`docs/google_colab.md`](docs/google_colab.md)
- [`docs/notebooks_guide.md`](docs/notebooks_guide.md)
- [`docs/theory_mert_stem_classification.md`](docs/theory_mert_stem_classification.md)
