import torch
from torch import nn
from position_encoding import PositionEncoding
from multihead_attention import MultiHeadSelfAttention
from feed_forward import FeedForwardNetwork
import math


class EncoderLayer(nn.Module):
    """A single Encoder layer

    Args:
        d_model (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Feed foward network dimension.
        dropout (float): Dropout value (default 0.1)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout=0.1) -> None:
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads) # Multi-head attention
        self.layernorm1 = nn.LayerNorm(d_model) # Layer normalization
        self.dropout1 = nn.Dropout(p=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout) # Feed forward network
        self.layernorm2 = nn.LayerNorm(d_model) # Layer normlization
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the encoder layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_len, d_model).
            mask (torch.Tensor, optional): Optional mask of input tensor (default None).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model)."""

        attn_output, _ = self.mha(x, key=x, value=x, mask=mask) # Self attention
        x = self.layernorm1(x + self.dropout1(attn_output))  # Add and norm
        ffn_output = self.ffn(x)  # Feed forward network
        x = self.layernorm2(x + self.dropout2(ffn_output))  # Add and norm
        return x


class Encoder(nn.Module):
    """Encoder block.

    Args:
        d_model (int): Embedding dimension.
        d_ff (int): Feed forward network dimension.
        num_heads (int): Number of attention heads.
        max_len (int): Max length of input sequence. (default 128).
        dropout (float): Dropout value (default 0.1).
        num_layers (int): Number of encoder layers (default 6)."""

    def __init__(self, d_model: int, d_ff: int, num_heads: int, max_len: int = 128, dropout: float = 0.1, num_layers: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        self.position_encoding = PositionEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (d_model, max_len).
            mask (torch.Tensor, optional): Optional mask for input tensor (default None).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, d_model, max_len)."""

        x = self.position_encoding(x * math.sqrt(self.d_model)) # Position encoding
        x = self.dropout(x) 

        # Pass through each Encoder layer
        for layer in self.layers:
            x = layer(x, mask)

        return x
