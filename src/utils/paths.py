"""Path helpers shared by command-line modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROJECT_ROOT = Path("/content/medleydb-mert-instrument-classification")
DEFAULT_RUN_ROOT = Path("/content/drive/MyDrive/medleydb_mert_project/isolated_stem_v1")
DEFAULT_MEDLEYDB_ROOT = Path("/content/drive/MyDrive/medleydb_mert_project/MedleyDB")
GENERATED_ROOT_NAMES = {"data", "results", "checkpoints", "reports", "configs"}


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project and persistent-artifact paths for local or Colab runs."""

    project_root: Path
    run_root: Path
    medleydb_root: Path
    data_dir: Path
    metadata_dir: Path
    cache_dir: Path
    results_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    configs_dir: Path

    def ensure(self) -> "ProjectPaths":
        """Create generated-artifact directories and return ``self``."""

        for path in (
            self.data_dir,
            self.metadata_dir,
            self.cache_dir,
            self.results_dir,
            self.checkpoints_dir,
            self.reports_dir,
            self.configs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def paths_from_env(
    *,
    project_root: str | Path | None = None,
    run_root: str | Path | None = None,
    medleydb_root: str | Path | None = None,
    create: bool = False,
) -> ProjectPaths:
    """Resolve standard local/Colab/Drive paths from arguments or environment."""

    project = Path(project_root or os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))
    run = Path(run_root or os.environ.get("RUN_ROOT", DEFAULT_RUN_ROOT))
    medleydb = Path(medleydb_root or os.environ.get("MEDLEYDB_ROOT", DEFAULT_MEDLEYDB_ROOT))
    paths = ProjectPaths(
        project_root=project,
        run_root=run,
        medleydb_root=medleydb,
        data_dir=run / "data",
        metadata_dir=run / "data" / "metadata",
        cache_dir=run / "data" / "cache",
        results_dir=run / "results",
        checkpoints_dir=run / "checkpoints",
        reports_dir=run / "data" / "reports",
        configs_dir=run / "configs",
    )
    return paths.ensure() if create else paths


def model_cache_name(model_name: str) -> str:
    """Return a stable filesystem-safe cache name for a Hugging Face model id."""

    tail = str(model_name).split("/")[-1].strip().lower()
    return tail.replace("-", "_").replace(".", "_")


def representation_cache_name(layer: str | int, pooling: str) -> str:
    """Return the standard cache leaf for one layer/pooling representation."""

    layer_text = str(layer).strip().lower().replace("/", "_").replace("-", "_")
    prefix = f"layer{layer_text}" if layer_text.lstrip("-").isdigit() else f"layer_{layer_text}"
    return f"{prefix}_pool_{str(pooling).strip().lower()}"


def isolated_embedding_cache_dir(
    cache_root: Path,
    *,
    model_name_or_cache: str,
    subset_profile: str,
    label_granularity: str,
    layer: str | int,
    pooling: str,
) -> Path:
    """Return the canonical isolated-stem MERT embedding cache directory."""

    model_part = (
        model_name_or_cache
        if "/" not in str(model_name_or_cache)
        else model_cache_name(model_name_or_cache)
    )
    return (
        Path(cache_root)
        / model_part
        / str(subset_profile)
        / str(label_granularity)
        / representation_cache_name(layer, pooling)
    )


def mixture_embedding_cache_dir(
    cache_root: Path,
    *,
    model_name_or_cache: str,
    mixture_dataset_id: str,
    layer: str | int,
    pooling: str,
) -> Path:
    """Return the canonical polyphonic-mixture MERT embedding cache directory."""

    model_part = (
        model_name_or_cache
        if "/" not in str(model_name_or_cache)
        else model_cache_name(model_name_or_cache)
    )
    return (
        Path(cache_root)
        / model_part
        / "mixtures"
        / str(mixture_dataset_id)
        / representation_cache_name(layer, pooling)
    )


def make_run_id(value: str | None = None) -> str:
    """Return an explicit run id or a UTC timestamp safe for path names."""

    if value:
        return str(value)
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def experiment_run_dir(root: Path, experiment_id: str, seed: int, run_id: str | None = None) -> Path:
    """Return ``root/experiment_id/seed_<seed>/<run_id>`` for organized outputs."""

    return Path(root) / str(experiment_id) / f"seed_{int(seed)}" / make_run_id(run_id)


def resolve_run_path(path: str | Path, run_root: str | Path | None = None) -> Path:
    """Resolve relative generated-artifact paths below ``RUN_ROOT`` when set."""

    value = Path(path)
    if value.is_absolute():
        return value
    root_text = run_root or os.environ.get("RUN_ROOT")
    if not root_text or not value.parts or value.parts[0] not in GENERATED_ROOT_NAMES:
        return value
    return Path(root_text) / value


def ensure_parent(path: Path) -> Path:
    """Create a file's parent directory and return the input path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def portable_relative_path(path: Path, root: Path) -> str:
    """Return a POSIX-style path relative to ``root``.

    Generated metadata stays portable between Windows and Linux by never
    storing machine-specific absolute paths.
    """

    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_data_path(root: Path, relative_path: str | Path) -> Path:
    """Resolve a portable metadata path below a dataset root safely."""

    root = root.resolve()
    path = (root / Path(str(relative_path))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Audio path escapes the MedleyDB root: {relative_path}") from exc
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with a useful error when the shape is invalid."""

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(value).__name__}")
    return value


def atomic_replace(temp_path: Path, final_path: Path) -> None:
    """Atomically replace ``final_path`` with a completed temporary file."""

    ensure_parent(final_path)
    os.replace(temp_path, final_path)
