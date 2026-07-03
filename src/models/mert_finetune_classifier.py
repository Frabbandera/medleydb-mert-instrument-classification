"""Lightning model for frozen or partially fine-tuned MERT classification."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from src.models.embedding_classifier import EmbeddingClassifier


def configure_mert_trainability(backbone: nn.Module, mode: str, *, allow_full: bool = False) -> None:
    """Freeze MERT, unfreeze its last one/two layers, or explicitly unfreeze all."""

    if mode not in {"frozen", "last_1", "last_2", "full"}:
        raise ValueError("unfreeze_mode must be frozen, last_1, last_2, or full")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if mode == "full":
        if not allow_full:
            raise ValueError("Full MERT fine-tuning requires allow_full_finetune: true")
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        return
    if mode == "frozen":
        return
    layers = getattr(getattr(backbone, "encoder", None), "layers", None)
    if layers is None:
        raise ValueError("Loaded MERT model does not expose encoder.layers")
    count = 1 if mode == "last_1" else 2
    if len(layers) < count:
        raise ValueError(f"MERT exposes only {len(layers)} Transformer layers")
    for layer in layers[-count:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)


class MertFinetuneClassifier(L.LightningModule):
    """Direct-audio MERT classifier with controlled backbone unfreezing."""

    def __init__(self, *, model_name: str, num_classes: int, model_revision: str | None = None,
                 classifier_type: str = "mlp",
                 hidden_dim: int = 256, dropout: float = 0.3, pooling: str = "mean",
                 unfreeze_mode: str = "last_1", allow_full_finetune: bool = False,
                 head_learning_rate: float = 1e-3, backbone_learning_rate: float = 1e-5,
                 weight_decay: float = 1e-4, class_weights: torch.Tensor | None = None,
                 gradient_checkpointing: bool = True, backbone: nn.Module | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "class_weights"])
        if backbone is None:
            from transformers import AutoModel
            kwargs: dict[str, Any] = {"trust_remote_code": True}
            if model_revision:
                kwargs["revision"] = model_revision
            backbone = AutoModel.from_pretrained(model_name, **kwargs)
        self.mert = backbone
        configure_mert_trainability(self.mert, unfreeze_mode, allow_full=allow_full_finetune)
        if (
            gradient_checkpointing
            and unfreeze_mode != "frozen"
            and hasattr(self.mert, "gradient_checkpointing_enable")
        ):
            self.mert.gradient_checkpointing_enable()
        hidden_size = int(getattr(self.mert.config, "hidden_size", 768))
        input_dim = hidden_size * (2 if pooling == "meanmax" else 1)
        self.classifier = EmbeddingClassifier(input_dim, num_classes,
                                              classifier_type=classifier_type,
                                              hidden_dim=hidden_dim, dropout=dropout)
        if pooling not in {"mean", "max", "meanmax"}:
            raise ValueError("pooling must be mean, max, or meanmax")
        self.pooling = pooling
        self.register_buffer(
            "class_weights",
            torch.ones(num_classes) if class_weights is None else class_weights.float(),
            persistent=False,
        )
        self.metrics = nn.ModuleDict()
        for stage in ("train", "val", "test"):
            self.metrics[f"{stage}_accuracy"] = MulticlassAccuracy(num_classes=num_classes)
            self.metrics[f"{stage}_macro_f1"] = MulticlassF1Score(num_classes=num_classes, average="macro")
            self.metrics[f"{stage}_weighted_f1"] = MulticlassF1Score(num_classes=num_classes, average="weighted")

    def train(self, mode: bool = True) -> "MertFinetuneClassifier":
        """Keep a fully frozen backbone deterministic while training its head."""

        super().train(mode)
        if str(self.hparams.unfreeze_mode) == "frozen":
            self.mert.eval()
        return self

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        hidden = self.mert(input_values=waveforms).last_hidden_state
        if self.pooling == "mean":
            pooled = hidden.mean(dim=1)
        elif self.pooling == "max":
            pooled = hidden.amax(dim=1)
        else:
            pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1)], dim=-1)
        return self.classifier(pooled)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        waveforms, labels, _ = batch
        logits = self(waveforms)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        predictions = logits.argmax(dim=1)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, batch_size=len(labels))
        for name in ("accuracy", "macro_f1", "weighted_f1"):
            metric = self.metrics[f"{stage}_{name}"]
            metric.update(predictions, labels)
            self.log(f"{stage}_{name}", metric, on_step=False, on_epoch=True, batch_size=len(labels))
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        self._step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        del batch_idx
        self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        backbone_parameters = [p for p in self.mert.parameters() if p.requires_grad]
        head_parameters = list(self.classifier.parameters())
        groups = [{"params": head_parameters, "lr": float(self.hparams.head_learning_rate)}]
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": float(self.hparams.backbone_learning_rate)})
        return torch.optim.AdamW(groups, weight_decay=float(self.hparams.weight_decay))
