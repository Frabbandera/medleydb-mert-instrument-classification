import json
from pathlib import Path


EXPERIMENT_NOTEBOOKS = [
    "01_frozen_mert_classifier_head.ipynb",
    "02_mert_representation_ablation.ipynb",
    "03_mert_finetuning_experiments.ipynb",
    "04_segment_and_training_protocol_ablation.ipynb",
    "99_isolated_stem_final_summary.ipynb",
]

EXPECTED_NOTEBOOKS = [
    "00_dataset_exploration_medleydb.ipynb",
    "pipeline_medleydb.ipynb",
    *EXPERIMENT_NOTEBOOKS,
    "05_polyphonic_mixture_dataset.ipynb",
    "06_polyphonic_mert_multilabel.ipynb",
    "07_overlap_robustness_analysis.ipynb",
    "08_genre_instrumentation_cooccurrence.ipynb",
    "09_error_analysis_audio_examples.ipynb",
]


def test_experiment_notebooks_are_clean_and_explanatory() -> None:
    root = Path(__file__).parents[1] / "notebooks"
    for name in EXPERIMENT_NOTEBOOKS:
        notebook = json.loads((root / name).read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for heading in ("Research question", "Approach", "What is fixed",
                        "What is varied", "Expected interpretation"):
            assert heading in text


def test_expected_notebooks_exist_and_are_clean() -> None:
    root = Path(__file__).parents[1] / "notebooks"
    for name in EXPECTED_NOTEBOOKS:
        notebook = json.loads((root / name).read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []


def test_notebooks_have_colab_badges_and_colab_safe_setup() -> None:
    root = Path(__file__).parents[1] / "notebooks"
    for name in EXPECTED_NOTEBOOKS:
        notebook = json.loads((root / name).read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert "Open in Colab" in markdown
        assert "What you can change" in markdown
        first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        source = "".join(first_code.get("source", []))
        for token in ("PROJECT_ROOT =", "RUN_ROOT =", "MEDLEYDB_ROOT =", "os.chdir(PROJECT_ROOT)", "sys.path.insert"):
            assert token in source
        src_import = source.find("from src")
        sys_path = source.find("sys.path.insert")
        assert src_import < 0 or sys_path < src_import


def test_no_notebook_embeds_token_in_git_url() -> None:
    root = Path(__file__).parents[1] / "notebooks"
    for path in root.glob("*.ipynb"):
        text = path.read_text(encoding="utf-8")
        notebook = json.loads(text)
        assert "https://{token}@github.com" not in text
        assert "x-access-token:{token}@github.com" not in text
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []


def test_report_analysis_notebooks_use_existing_artifacts_only() -> None:
    root = Path(__file__).parents[1] / "notebooks"
    expected_tokens = {
        "08_genre_instrumentation_cooccurrence.ipynb": [
            "ANALYSIS_SOURCE",
            "instrument_counts",
            "cooccurrence_matrix",
            "FIGURE_DIR = RESULTS_DIR",
            "TABLE_DIR = RESULTS_DIR",
            "Genre metadata is missing",
        ],
        "09_error_analysis_audio_examples.ipynb": [
            "EXPERIMENT_ID",
            "TASK_TYPE",
            "N_EXAMPLES_PER_TYPE",
            "EXPORT_AUDIO",
            "ERROR_DIR = RESULTS_DIR",
            "detect_task_type",
        ],
    }
    forbidden = (
        "extract_mert_embeddings",
        "extract_mert_mixture_embeddings",
        "run_experiment",
        "train_classifier",
        "train_multilabel_classifier",
    )
    for name, tokens in expected_tokens.items():
        text = (root / name).read_text(encoding="utf-8")
        for token in tokens:
            assert token in text
        for token in forbidden:
            assert token not in text
