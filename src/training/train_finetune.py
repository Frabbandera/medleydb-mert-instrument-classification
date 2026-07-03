"""Train and evaluate direct-audio frozen or partially fine-tuned MERT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.experiments.run_utils import save_evaluation_artifacts
from src.models.mert_finetune_classifier import MertFinetuneClassifier
from src.training.audio_datamodule import AudioSegmentDataModule
from src.utils.paths import ensure_directory


def train_and_evaluate_finetune(
    config_path: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Run the heavy direct-audio experiment selected by a resolved config."""

    del config_path
    L.seed_everything(int(config["seed"]), workers=True)
    data, model_cfg, training = config["data"], config["model"], config["training"]
    model_revision = model_cfg.get("model_revision")
    if model_revision in {"", "NA", "null"}:
        model_revision = None
    try:
        from transformers import AutoFeatureExtractor

        processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if model_revision:
            processor_kwargs["revision"] = model_revision
        processor = AutoFeatureExtractor.from_pretrained(
            str(model_cfg["model_name"]), **processor_kwargs
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Hugging Face feature extractor for fine-tuning. "
            "This path intentionally uses the same processor-compatible audio "
            f"preprocessing as frozen extraction. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    sample_rate = int(getattr(processor, "sampling_rate", 24000))
    data_module = AudioSegmentDataModule(
        Path(data["subset_csv"]), Path(data["medleydb_root"]), sample_rate=sample_rate,
        batch_size=int(data["batch_size"]), num_workers=int(data["num_workers"]),
        sampler=str(data["sampler"]), processor=processor,
    )
    data_module.setup("fit")
    class_weights = data_module.class_weights if training["loss_weighting"] == "inverse_frequency" else None
    model = MertFinetuneClassifier(
        model_name=str(model_cfg["model_name"]), num_classes=data_module.num_classes,
        model_revision=model_revision,
        classifier_type=str(model_cfg["classifier_type"]), hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]), pooling=str(config["representation"]["pooling"]),
        unfreeze_mode=str(model_cfg["unfreeze_mode"]),
        allow_full_finetune=bool(model_cfg["allow_full_finetune"]),
        head_learning_rate=float(training["head_learning_rate"]),
        backbone_learning_rate=float(training["backbone_learning_rate"]),
        weight_decay=float(training["weight_decay"]), class_weights=class_weights,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
    )
    checkpoint_dir = ensure_directory(Path(config["output"]["checkpoint_dir"]))
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename="best",
                                          monitor="val_macro_f1", mode="max", save_top_k=1,
                                          auto_insert_metric_name=False, enable_version_counter=False)
    logger: CSVLogger | bool
    if bool(training.get("logger", True)):
        logger = CSVLogger(
            save_dir=str(checkpoint_dir.parent / "logs"),
            name=str(config.get("experiment_name", config.get("experiment_id", "mert_finetune"))),
        )
    else:
        logger = False
    trainer = L.Trainer(
        max_epochs=int(training["max_epochs"]), accelerator=training["accelerator"],
        devices=training["devices"], precision=training["precision"],
        deterministic=bool(training["deterministic"]),
        accumulate_grad_batches=int(training["accumulate_grad_batches"]),
        limit_train_batches=training.get("limit_train_batches", 1.0),
        limit_val_batches=training.get("limit_val_batches", 1.0),
        enable_progress_bar=bool(training.get("enable_progress_bar", True)),
        enable_model_summary=bool(training.get("enable_model_summary", True)),
        log_every_n_steps=int(training.get("log_every_n_steps", 50)),
        num_sanity_val_steps=int(training.get("num_sanity_val_steps", 2)),
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_macro_f1", mode="max",
                                                      patience=int(training["early_stopping_patience"]))],
        logger=logger,
        default_root_dir=str(checkpoint_dir),
    )
    trainer.fit(model, datamodule=data_module)
    checkpoint = Path(checkpoint_callback.best_model_path)
    if not checkpoint.is_file():
        raise RuntimeError("Fine-tuning finished without a best checkpoint")
    summary = {
        "best_val_macro_f1": (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else "NA"
        ),
        "best_val_accuracy": "NA",
        "best_epoch": "NA",
    }
    metrics_path = Path(logger.log_dir) / "metrics.csv" if logger is not False else None
    if metrics_path is not None and metrics_path.is_file():
        metrics_frame = pd.read_csv(metrics_path)
        if "val_macro_f1" in metrics_frame:
            rows = metrics_frame.dropna(subset=["val_macro_f1"]).copy()
            if not rows.empty:
                best_row = rows.loc[rows["val_macro_f1"].astype(float).idxmax()]
                summary["best_val_macro_f1"] = float(best_row["val_macro_f1"])
                if "val_accuracy" in best_row and not pd.isna(best_row["val_accuracy"]):
                    summary["best_val_accuracy"] = float(best_row["val_accuracy"])
                if "epoch" in best_row and not pd.isna(best_row["epoch"]):
                    summary["best_epoch"] = int(best_row["epoch"])
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    best = MertFinetuneClassifier.load_from_checkpoint(checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and str(training["accelerator"]) != "cpu" else "cpu")
    best.to(device).eval()
    probabilities: list[torch.Tensor] = []
    targets: list[int] = []
    indices: list[int] = []
    with torch.inference_mode():
        for waveforms, labels, batch_indices in data_module.test_dataloader():
            probabilities.append(torch.softmax(best(waveforms.to(device)), dim=1).cpu())
            targets.extend(labels.tolist())
            indices.extend(batch_indices.tolist())
    mapping = json.loads(Path(data["label_to_id"]).read_text(encoding="utf-8"))
    label_names = [label for label, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    metadata = data_module.frames["test"].iloc[indices][
        ["segment_id", "track_id", "audio_path", "start_seconds", "duration_seconds"]
    ].reset_index(drop=True)
    metrics = save_evaluation_artifacts(
        results_dir=Path(config["output"]["results_dir"]), resolved_config=config,
        metadata=metadata, targets=targets,
        probabilities=torch.cat(probabilities).numpy(), label_names=label_names,
    )
    return checkpoint, metrics
