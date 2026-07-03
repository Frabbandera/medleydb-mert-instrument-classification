from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.run_utils import (
    build_predictions_frame,
    normalize_confusion_matrix,
    save_evaluation_artifacts,
)


def test_normalized_confusion_matrix_handles_empty_rows() -> None:
    matrix = np.array([[2, 1], [0, 0]])
    normalized = normalize_confusion_matrix(matrix)
    np.testing.assert_allclose(normalized[0], [2 / 3, 1 / 3])
    np.testing.assert_allclose(normalized[1], [0, 0])


def test_predictions_format_and_complete_artifacts(tmp_path: Path) -> None:
    metadata = pd.DataFrame({
        "segment_id": ["s1", "s2"], "track_id": ["t1", "t2"],
        "audio_path": ["a.wav", "b.wav"],
    })
    probabilities = np.array([[0.8, 0.2], [0.7, 0.3]])
    targets = np.array([0, 1])
    predictions = build_predictions_frame(metadata, targets, probabilities, ["bass", "guitar"])
    assert predictions["correct"].tolist() == [True, False]
    assert predictions["probability_true_class"].tolist() == [0.8, 0.3]
    assert {"prob_bass", "prob_guitar"}.issubset(predictions.columns)
    results = tmp_path / "results"
    save_evaluation_artifacts(results_dir=results, resolved_config={"experiment_id": "x"},
                              metadata=metadata, targets=targets,
                              probabilities=probabilities, label_names=["bass", "guitar"])
    expected = {
        "config_resolved.yaml", "test_metrics.json", "classification_report.csv",
        "confusion_matrix_raw.csv", "confusion_matrix_normalized.csv",
        "confusion_matrix_raw.png", "confusion_matrix_normalized.png", "predictions.csv",
    }
    assert {path.name for path in results.iterdir()} == expected
