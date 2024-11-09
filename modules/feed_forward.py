import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """Feed-forward network module.

    Args:
        d_model (int): Embedding dimension (default 512).
        d_ff (int): Feed forward network dimension (default 2048).
        dropout (float): dropout value (default 0.1)."""

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn_block = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass for feed-forward network layer.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, max_len, d_model).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model)."""

        x = self.ffn_block(x) + x # Feed forward network + residual input
        return x
