"""Small trainable heads for cached MERT embeddings."""

from __future__ import annotations

import torch
from torch import nn


class WeightedLayerAggregation(nn.Module):
    """Learn a softmax-normalized mixture of cached MERT hidden layers."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("Weighted layer aggregation requires at least two layers")
        self.logits = nn.Parameter(torch.zeros(num_layers))

    @property
    def weights(self) -> torch.Tensor:
        """Return normalized layer weights."""

        return torch.softmax(self.logits, dim=0)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reduce `[batch, layers, dimension]` to `[batch, dimension]`."""

        if embeddings.ndim != 3 or embeddings.shape[1] != len(self.logits):
            raise ValueError("Expected cached embeddings with shape [batch, layers, dim]")
        return torch.sum(embeddings * self.weights.view(1, -1, 1), dim=1)


class EmbeddingClassifier(nn.Module):
    """Linear or one-hidden-layer classifier for fixed-size embeddings."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        classifier_type: str = "mlp",
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int | None = None,
        layer_aggregation: str = "none",
    ) -> None:
        super().__init__()
        if layer_aggregation == "learned_softmax":
            if num_layers is None:
                raise ValueError("num_layers is required for learned layer aggregation")
            self.layer_aggregation: nn.Module = WeightedLayerAggregation(num_layers)
        elif layer_aggregation == "none":
            self.layer_aggregation = nn.Identity()
        else:
            raise ValueError("layer_aggregation must be 'none' or 'learned_softmax'")
        if input_dim <= 0 or num_classes < 2:
            raise ValueError("input_dim must be positive and num_classes must be at least 2")
        if classifier_type == "linear":
            self.network = nn.Linear(input_dim, num_classes)
        elif classifier_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError("classifier_type must be 'mlp' or 'linear'")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""

        return self.network(self.layer_aggregation(embeddings))
