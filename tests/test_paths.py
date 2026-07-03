from pathlib import Path

from src.utils.paths import (
    experiment_run_dir,
    isolated_embedding_cache_dir,
    mixture_embedding_cache_dir,
    model_cache_name,
    paths_from_env,
    representation_cache_name,
    resolve_run_path,
)


def test_paths_from_env_and_cache_layout(monkeypatch) -> None:
    monkeypatch.setenv("PROJECT_ROOT", "/project")
    monkeypatch.setenv("RUN_ROOT", "/drive/run")
    monkeypatch.setenv("MEDLEYDB_ROOT", "/drive/MedleyDB")
    paths = paths_from_env()
    assert paths.project_root == Path("/project")
    assert paths.metadata_dir == Path("/drive/run/data/metadata")
    assert paths.cache_dir == Path("/drive/run/data/cache")
    assert paths.medleydb_root == Path("/drive/MedleyDB")


def test_canonical_cache_and_run_paths() -> None:
    assert model_cache_name("m-a-p/MERT-v1-95M") == "mert_v1_95m"
    assert representation_cache_name("last4avg", "mean") == "layer_last4avg_pool_mean"
    assert representation_cache_name(6, "mean") == "layer6_pool_mean"
    assert isolated_embedding_cache_dir(
        Path("data/cache"),
        model_name_or_cache="m-a-p/MERT-v1-95M",
        subset_profile="largest_balanced",
        label_granularity="medleydb_instrument",
        layer="last",
        pooling="mean",
    ) == Path("data/cache/mert_v1_95m/largest_balanced/medleydb_instrument/layer_last_pool_mean")
    assert mixture_embedding_cache_dir(
        Path("data/cache"),
        model_name_or_cache="mert_v1_95m",
        mixture_dataset_id="debug_synthetic_k",
        layer="all",
        pooling="mean",
    ) == Path("data/cache/mert_v1_95m/mixtures/debug_synthetic_k/layer_all_pool_mean")
    assert experiment_run_dir(Path("results"), "exp", 42, "latest") == Path("results/exp/seed_42/latest")


def test_resolve_run_path_uses_run_root_for_generated_artifacts(monkeypatch) -> None:
    monkeypatch.setenv("RUN_ROOT", "/drive/run")
    assert resolve_run_path("data/metadata/stem_index.csv") == Path("/drive/run/data/metadata/stem_index.csv")
    assert resolve_run_path("results/experiment_registry.csv") == Path("/drive/run/results/experiment_registry.csv")
    assert resolve_run_path("configs/experiments/run.yaml") == Path("/drive/run/configs/experiments/run.yaml")
    assert resolve_run_path("MedleyDB") == Path("MedleyDB")
