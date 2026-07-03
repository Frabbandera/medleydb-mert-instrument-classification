"""Multi-label classifier heads for cached MERT mixture embeddings."""

from __future__ import annotations

import torch
from torch import nn

from src.models.embedding_classifier import WeightedLayerAggregation


class MultilabelEmbeddingClassifier(nn.Module):
    """Linear or MLP multi-label head returning raw logits."""

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        *,
        classifier_type: str = "mlp",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int | None = None,
        layer_aggregation: str = "none",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_labels <= 0:
            raise ValueError("input_dim and num_labels must be positive")
        if layer_aggregation == "learned_softmax":
            if num_layers is None:
                raise ValueError("num_layers is required for learned layer aggregation")
            self.layer_aggregation: nn.Module = WeightedLayerAggregation(num_layers)
        elif layer_aggregation == "none":
            self.layer_aggregation = nn.Identity()
        else:
            raise ValueError("layer_aggregation must be 'none' or 'learned_softmax'")
        if classifier_type == "linear":
            self.network = nn.Linear(input_dim, num_labels)
        elif classifier_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        else:
            raise ValueError("classifier_type must be 'mlp' or 'linear'")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(self.layer_aggregation(embeddings))
