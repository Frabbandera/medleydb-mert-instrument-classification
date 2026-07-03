"""Train a lightweight Lightning classifier on cached MERT embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from src.experiments.config import resolve_experiment_config
from src.models.embedding_classifier import EmbeddingClassifier
from src.training.datamodule import CachedEmbeddingDataModule
from src.utils.paths import ensure_directory, load_yaml


def _best_validation_summary(logger: CSVLogger | bool, best_score: float | None) -> dict[str, Any]:
    """Read validation metrics for the epoch selected by ModelCheckpoint."""

    summary: dict[str, Any] = {
        "best_val_macro_f1": float(best_score) if best_score is not None else "NA",
        "best_val_accuracy": "NA",
        "best_epoch": "NA",
    }
    if logger is False:
        return summary
    metrics_path = Path(logger.log_dir) / "metrics.csv"
    if not metrics_path.is_file():
        return summary
    import pandas as pd

    metrics = pd.read_csv(metrics_path)
    if "val_macro_f1" not in metrics:
        return summary
    validation_rows = metrics.dropna(subset=["val_macro_f1"]).copy()
    if validation_rows.empty:
        return summary
    best_index = validation_rows["val_macro_f1"].astype(float).idxmax()
    row = validation_rows.loc[best_index]
    summary["best_val_macro_f1"] = float(row["val_macro_f1"])
    if "val_accuracy" in row and not pd.isna(row["val_accuracy"]):
        summary["best_val_accuracy"] = float(row["val_accuracy"])
    if "epoch" in row and not pd.isna(row["epoch"]):
        summary["best_epoch"] = int(row["epoch"])
    return summary


class MertEmbeddingClassifier(L.LightningModule):
    """Lightning wrapper around the trainable embedding classifier head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        classifier_type: str = "mlp",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_layers: int | None = None,
        layer_aggregation: str = "none",
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.classifier = EmbeddingClassifier(
            input_dim,
            num_classes,
            classifier_type=classifier_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            layer_aggregation=layer_aggregation,
        )
        if class_weights is None:
            class_weights = torch.ones(num_classes)
        self.register_buffer("class_weights", class_weights.float(), persistent=False)
        self.metrics = torch.nn.ModuleDict()
        for stage in ("train", "val", "test"):
            self.metrics[f"{stage}_accuracy"] = MulticlassAccuracy(num_classes=num_classes)
            self.metrics[f"{stage}_macro_f1"] = MulticlassF1Score(
                num_classes=num_classes, average="macro"
            )
            self.metrics[f"{stage}_weighted_f1"] = MulticlassF1Score(
                num_classes=num_classes, average="weighted"
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        embeddings, labels = batch
        logits = self(embeddings)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        predictions = logits.argmax(dim=1)
        batch_size = labels.shape[0]
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        for metric_name in ("accuracy", "macro_f1", "weighted_f1"):
            metric = self.metrics[f"{stage}_{metric_name}"]
            metric.update(predictions, labels)
            self.log(
                f"{stage}_{metric_name}",
                metric,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )


def _config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def train_from_config(config_path: Path, *, resolved_config: dict[str, Any] | None = None) -> Path:
    """Train from YAML and return the best checkpoint path."""

    config = resolved_config or load_yaml(config_path)
    data_config = _config_section(config, "data")
    model_config = _config_section(config, "model")
    train_config = _config_section(config, "training")
    seed = int(config.get("seed", 42))
    L.seed_everything(seed, workers=True)

    data_module = CachedEmbeddingDataModule(
        cache_dir=Path(data_config["cache_dir"]),
        batch_size=int(data_config.get("batch_size", 32)),
        num_workers=int(data_config.get("num_workers", 0)),
        sampler=str(data_config.get("sampler", "shuffle")),
    )
    data_module.setup("fit")
    layer_aggregation = str(model_config.get("layer_aggregation", "none"))
    if data_module.num_layers is not None and layer_aggregation != "learned_softmax":
        raise ValueError(
            "A multi-layer cache requires model.layer_aggregation: learned_softmax"
        )
    model = MertEmbeddingClassifier(
        input_dim=data_module.input_dim,
        num_classes=data_module.num_classes,
        classifier_type=str(model_config.get("classifier_type", "mlp")),
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        dropout=float(model_config.get("dropout", 0.3)),
        learning_rate=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
        num_layers=data_module.num_layers,
        layer_aggregation=layer_aggregation,
        class_weights=(
            data_module.class_weights
            if str(train_config.get("loss_weighting", "none")) == "inverse_frequency"
            else None
        ),
    )
    checkpoint_dir = ensure_directory(Path(train_config["checkpoint_dir"]))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=int(train_config.get("early_stopping_patience", 8)),
    )
    logger: CSVLogger | bool
    if bool(train_config.get("logger", True)):
        logger = CSVLogger(
            save_dir=str(checkpoint_dir.parent / "logs"),
            name=str(config.get("experiment_name", "mert_frozen_subset")),
        )
    else:
        logger = False
    trainer = L.Trainer(
        max_epochs=int(train_config.get("max_epochs", 50)),
        accelerator=train_config.get("accelerator", "auto"),
        devices=train_config.get("devices", 1),
        precision=train_config.get("precision", "32-true"),
        deterministic=bool(train_config.get("deterministic", True)),
        accumulate_grad_batches=int(train_config.get("accumulate_grad_batches", 1)),
        limit_train_batches=train_config.get("limit_train_batches", 1.0),
        limit_val_batches=train_config.get("limit_val_batches", 1.0),
        enable_progress_bar=bool(train_config.get("enable_progress_bar", True)),
        enable_model_summary=bool(train_config.get("enable_model_summary", True)),
        log_every_n_steps=int(train_config.get("log_every_n_steps", 50)),
        num_sanity_val_steps=int(train_config.get("num_sanity_val_steps", 2)),
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        default_root_dir=str(checkpoint_dir),
    )
    trainer.fit(model, datamodule=data_module)
    best_path = Path(checkpoint_callback.best_model_path)
    if not best_path.is_file():
        raise RuntimeError("Training finished without producing a best checkpoint")
    best_score = (
        float(checkpoint_callback.best_model_score)
        if checkpoint_callback.best_model_score is not None
        else None
    )
    summary = _best_validation_summary(logger, best_score)
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if summary["best_val_macro_f1"] == "NA":
        print("Best validation macro-F1: NA")
    else:
        print(f"Best validation macro-F1: {float(summary['best_val_macro_f1']):.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_experiment_config(args.config, experiment_id=args.experiment_id)
    train_from_config(args.config, resolved_config=config)


if __name__ == "__main__":
    main()
