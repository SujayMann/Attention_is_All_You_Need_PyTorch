import torch
from torch import nn
import math
from typing import Tuple


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention module.

    Args:
        d_model (int): Embedding dimension.
        num_heads (int): Number of attention heads."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)

    def scaled_dot_product_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, max_len, d_k).
            key (torch.Tensor): Key tensor of shape (batch_size, num_heads, max_len, d_k).
            value (torch.Tensor): Value tensor of shape (batch_size, num_heads, max_len, d_k).
            mask (torch.Tensor, optional): Optional mask tensor.

        Returns:
            Tuple:
                attn_output (torch.Tensor): Output of the attention layer.
                attn_weights (torch.Tensor): Attention weights."""

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  # Scaled dot product

        if mask is not None:
            # Reshape to (batch_size, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)
            # Check for expected dimensions before repeat
            if len(mask.shape) == 4:
                mask = mask.repeat(1, self.num_heads, 1, 1)
                scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)  # Softmax
        attn_output = torch.matmul(attn_weights, value)  # Weighted sum
        return attn_output, attn_weights

    def forward(self, query: torch.Tensor, key: torch.Tensor = None, value: torch.Tensor = None, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Multi-head Attention.

        Args:
            query (torch.Tensor): Input tensor of shape (batch_size, max_len, d_model).
            key (torch.Tensor, optional): Key tensor (defaults to query).
            value (torch.Tensor, optional): Value tensor (defaults to query).
            mask (torch.Tensor, optional): Mask of input tensor (default None).

        Returns:
            Tuple:
                output (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model).
                attn_weights (torch.Tensor): Attention weights tensor of shape (batch_size, num_heads, max_len, max_len)."""

        batch_size = query.size(0)

        if key is None:
            key = query
        if value is None:
            value = query

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Multihead outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_o(attn_output)
        return output, attn_weights
