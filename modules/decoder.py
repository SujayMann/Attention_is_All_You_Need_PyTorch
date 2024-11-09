import torch
from torch import nn
from position_encoding import PositionEncoding
from multihead_attention import MultiHeadSelfAttention
from feed_forward import FeedForwardNetwork
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DecoderLayer(nn.Module):
    """A single Decoder layer.

    Args:
        d_model (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Feed forward network dimension.
        dropout (float): Dropout value (default 0.1)"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)  # Masked MHA
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads)  # Cross-attention
        self.layernorm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_len, d_model).
            encoder_output (torch.Tensor): Output tensor from the encoder, shape (batch_size, max_len, d_model).
            tgt_mask (torch.Tensor, optional): Mask tensor for masked multi-head attention (default None).
            src_mask (torch.Tensor, optional): Mask tensor for padding (default None).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model)."""

        attn_output, _ = self.self_attn(x, key=x, value=x, mask=tgt_mask)  # Masked self-attention
        x = self.layernorm1(x + self.dropout1(attn_output))  # add and norm
        # Cross-attention to encoder's output
        attn_output, _ = self.cross_attn(x, key=encoder_output, value=encoder_output, mask=src_mask)
        x = self.layernorm2(x + self.dropout2(attn_output))  # add and norm
        ffn_output = self.ffn(x)  # feed forward network
        x = self.layernorm3(x + self.dropout3(ffn_output))  # add and norm
        return x


class Decoder(nn.Module):
    """Decoder block.

    Args:
        d_model (int): Embedding dimension.
        d_ff (int): Feed forward network dimension.
        num_heads (int): Number of attention heads.
        max_len (int): Max length of input sequence (default 128).
        dropout (float): Dropout value (default 0.1).
        num_layers (int): Number of decoder layers (default 6)."""

    def __init__(self, d_model: int, d_ff: int, num_heads: int, max_len: int = 128, dropout: float = 0.1, num_layers: int = 6, vocab_size=128) -> None:
        super().__init__()
        self.d_model = d_model
        self.position_encoding = PositionEncoding(d_model, max_len)
        self.embedding = nn.Embedding(vocab_size, d_model)  # Add embedding layer
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the decoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (d_model, max_len).
            encoder_output (torch.Tensor): Output tensor from the encoder block, shape (batch_size, d_model, max_len).
            mask (torch.Tensor, optional): Optional mask of input tensor.
            padding_mask (torch.Tensor, optional): Optional padding mask.

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model)."""

        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        x = x.to(device)
        x = self.embedding(x)
        x = self.position_encoding(x * math.sqrt(self.d_model))
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
            
        return x
