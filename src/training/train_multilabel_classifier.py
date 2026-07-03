"""Train a multi-label classifier on cached MERT mixture embeddings."""

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
from torchmetrics.classification import MultilabelF1Score

from src.experiments.config import resolve_experiment_config
from src.models.multilabel_embedding_classifier import MultilabelEmbeddingClassifier
from src.training.multilabel_datamodule import MultilabelCachedEmbeddingDataModule
from src.training.train_classifier import _best_validation_summary
from src.utils.paths import ensure_directory, load_yaml


class MertMultilabelClassifier(L.LightningModule):
    """Lightning wrapper for sigmoid/BCE multi-label classification."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        *,
        classifier_type: str = "mlp",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        threshold: float = 0.5,
        num_layers: int | None = None,
        layer_aggregation: str = "none",
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.classifier = MultilabelEmbeddingClassifier(
            input_dim,
            num_labels,
            classifier_type=classifier_type,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            layer_aggregation=layer_aggregation,
        )
        if pos_weight is None:
            pos_weight = torch.ones(num_labels)
        self.register_buffer("pos_weight", pos_weight.float(), persistent=False)
        self.val_macro_f1 = MultilabelF1Score(num_labels=num_labels, average="macro", threshold=threshold)
        self.val_micro_f1 = MultilabelF1Score(num_labels=num_labels, average="micro", threshold=threshold)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        embeddings, labels = batch
        logits = self(embeddings)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=labels.shape[0])
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        embeddings, labels = batch
        logits = self(embeddings)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        self.val_macro_f1.update(probs, labels.int())
        self.val_micro_f1.update(probs, labels.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=labels.shape[0])
        self.log("val_macro_f1", self.val_macro_f1, on_step=False, on_epoch=True, batch_size=labels.shape[0])
        self.log("val_micro_f1", self.val_micro_f1, on_step=False, on_epoch=True, batch_size=labels.shape[0])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def train_from_config(config_path: Path, *, resolved_config: dict[str, Any] | None = None) -> Path:
    config = resolved_config or load_yaml(config_path)
    data_config = _section(config, "data")
    model_config = _section(config, "model")
    train_config = _section(config, "training")
    eval_config = config.get("evaluation", {}) if isinstance(config.get("evaluation", {}), dict) else {}
    seed = int(config.get("seed", 42))
    L.seed_everything(seed, workers=True)
    data_module = MultilabelCachedEmbeddingDataModule(
        cache_dir=Path(data_config["cache_dir"]),
        batch_size=int(data_config.get("batch_size", 32)),
        num_workers=int(data_config.get("num_workers", 0)),
    )
    data_module.setup("fit")
    layer_aggregation = str(model_config.get("layer_aggregation", "none"))
    if data_module.num_layers is not None and layer_aggregation != "learned_softmax":
        raise ValueError("A multi-layer cache requires model.layer_aggregation: learned_softmax")
    use_pos_weight = str(train_config.get("loss_weighting", "pos_weight")) == "pos_weight"
    model = MertMultilabelClassifier(
        input_dim=data_module.input_dim,
        num_labels=data_module.num_labels,
        classifier_type=str(model_config.get("classifier_type", "mlp")),
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        dropout=float(model_config.get("dropout", 0.3)),
        learning_rate=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
        threshold=float(eval_config.get("threshold", 0.5)),
        num_layers=data_module.num_layers,
        layer_aggregation=layer_aggregation,
        pos_weight=data_module.pos_weight if use_pos_weight else None,
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
            name=str(config.get("experiment_name", "polyphonic_multilabel")),
        )
    else:
        logger = False
    trainer = L.Trainer(
        max_epochs=int(train_config.get("max_epochs", 50)),
        accelerator=train_config.get("accelerator", "auto"),
        devices=train_config.get("devices", 1),
        precision=train_config.get("precision", "32-true"),
        deterministic=bool(train_config.get("deterministic", True)),
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
