"""Train and evaluate classical audio-descriptor baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.audio_io import load_audio_segment
from src.experiments.run_utils import save_evaluation_artifacts
from src.utils.paths import ensure_directory, resolve_data_path


def _feature_vector(waveform: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
    """Return compact MFCC and spectral summary statistics for one segment."""

    n_mfcc = int(config.get("n_mfcc", 20))
    hop_length = int(config.get("hop_length", 512))
    n_fft = int(config.get("n_fft", 2048))
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    descriptors = [
        mfcc,
        librosa.feature.delta(mfcc),
        librosa.feature.spectral_centroid(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length),
        librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length),
        librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length),
        librosa.feature.spectral_contrast(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length),
        librosa.feature.zero_crossing_rate(y=waveform, frame_length=n_fft, hop_length=hop_length),
        librosa.feature.rms(y=waveform, frame_length=n_fft, hop_length=hop_length),
    ]
    parts = []
    for descriptor in descriptors:
        parts.append(np.mean(descriptor, axis=1))
        parts.append(np.std(descriptor, axis=1))
    return np.concatenate(parts).astype(np.float32)


def _load_features(
    frame: pd.DataFrame,
    *,
    medleydb_root: Path,
    sample_rate: int,
    feature_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row in frame.itertuples(index=False):
        waveform, _ = load_audio_segment(
            resolve_data_path(medleydb_root, row.audio_path),
            float(row.start_seconds),
            float(row.duration_seconds),
            target_sample_rate=sample_rate,
            normalize=True,
            pad=True,
        )
        features.append(_feature_vector(waveform.cpu().numpy(), sample_rate, feature_config))
        labels.append(int(row.label_id))
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def _classifier(config: dict[str, Any], seed: int):
    classifier_type = str(config.get("classifier_type", "svm")).lower()
    if classifier_type == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=float(config.get("C", 10.0)),
                kernel=str(config.get("kernel", "rbf")),
                gamma=str(config.get("gamma", "scale")),
                probability=True,
                class_weight=config.get("class_weight"),
                random_state=seed,
            ),
        )
    if classifier_type == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(config.get("C", 1.0)),
                max_iter=int(config.get("max_iter", 2000)),
                class_weight=config.get("class_weight"),
                random_state=seed,
            ),
        )
    if classifier_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 500)),
            max_depth=config.get("max_depth"),
            class_weight=config.get("class_weight"),
            random_state=seed,
            n_jobs=int(config.get("n_jobs", -1)),
        )
    raise ValueError("model.classifier_type must be svm, logistic_regression, or random_forest")


def _label_names(path: Path) -> list[str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("label_to_id must contain a JSON object")
    return [label for label, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def train_and_evaluate_classical(
    config_path: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Train an sklearn baseline and save report-compatible artifacts."""

    del config_path
    data = config["data"]
    subset = pd.read_csv(data["subset_csv"])
    required = {
        "segment_id",
        "track_id",
        "audio_path",
        "start_seconds",
        "duration_seconds",
        "label_id",
        "split",
    }
    missing = sorted(required - set(subset.columns))
    if missing:
        raise ValueError(f"Subset is missing columns: {', '.join(missing)}")
    frames = {
        split: subset[subset["split"] == split].copy().reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("Train, validation, and test splits must all be non-empty")

    feature_config = config.get("features", {})
    if not isinstance(feature_config, dict):
        raise ValueError("features must be a mapping")
    sample_rate = int(feature_config.get("sample_rate", 22050))
    medleydb_root = Path(data.get("medleydb_root", "MedleyDB"))
    train_x, train_y = _load_features(
        frames["train"],
        medleydb_root=medleydb_root,
        sample_rate=sample_rate,
        feature_config=feature_config,
    )
    val_x, val_y = _load_features(
        frames["val"],
        medleydb_root=medleydb_root,
        sample_rate=sample_rate,
        feature_config=feature_config,
    )
    test_x, test_y = _load_features(
        frames["test"],
        medleydb_root=medleydb_root,
        sample_rate=sample_rate,
        feature_config=feature_config,
    )

    model = _classifier(config["model"], int(config.get("seed", 42)))
    model.fit(np.concatenate([train_x, val_x]), np.concatenate([train_y, val_y]))
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_x)
    else:
        scores = model.decision_function(test_x)
        scores = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

    checkpoint_dir = ensure_directory(Path(config["output"]["checkpoint_dir"]))
    model_path = checkpoint_dir / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_config": feature_config,
            "sample_rate": sample_rate,
            "label_names": _label_names(Path(data["label_to_id"])),
        },
        model_path,
    )
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "best_val_macro_f1": "NA",
                "best_val_accuracy": "NA",
                "best_epoch": "NA",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = save_evaluation_artifacts(
        results_dir=Path(config["output"]["results_dir"]),
        resolved_config=config,
        metadata=frames["test"],
        targets=test_y,
        probabilities=probabilities,
        label_names=_label_names(Path(data["label_to_id"])),
    )
    return model_path, metrics
