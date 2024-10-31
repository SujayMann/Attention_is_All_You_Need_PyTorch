import torch
from torch import nn
from typing import Tuple

# Position Encoding
class PositionEncoding(torch.nn.Module):
  """Positional Encoding for Transformer model.

  Args:
      d_model (int): Embedding dimension.
      max_len (int): Maximum length of input sequences (default 10000)."""

  def __init__(self, d_model: int, max_len: int=10000) -> None:
    super().__init__()
    self.encoding = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)

    self.encoding[:, 0::2] = torch.sin(position / div_term) # Even terms (2i)
    self.encoding[:, 1::2] = torch.cos(position / div_term) # Odd terms (2i+1)
    self.encoding = self.encoding.unsqueeze(0) # batch dimension

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass to add positional encoding to input embeddings.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, max_len, d_model).

    Returns:
        x (torch.Tensor): Output tensor with position encoding added to it of shape (batch_size, max_len, d_model)."""

    x = x + self.encoding[:, :x.size(1), :]
    # print(f"Position Encoding output shape: {x.shape}")
    return x

# Multi-head Self Attention
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

  def scaled_dot_product_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
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
    # print(f"SDPA input shapes: {query.shape, key.shape, value.shape}")

    dot_product = torch.matmul(query, key.transpose(-2, -1))
    dot_product /= torch.sqrt(torch.tensor(self.d_k, dtype=query.dtype))

    if mask is not None:
      # print(f"Dot product shape: {dot_product.shape}")
      # print(f"Mask dimension: {mask.shape}")'
      while len(mask.shape) < 4:
        mask = mask.unsqueeze(1)

      mask = mask.repeat(1, self.num_heads, 1, 1)

      if mask.shape != dot_product.shape:
        mask = mask[:, :, :, :dot_product.shape[-1]]

      dot_product = dot_product.masked_fill(mask == 0, 1e-9)

    attn_weights = torch.softmax(dot_product, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights

  def forward(self, x: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for Multi-head Attention.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, max-len, d_model).
        mask (torch.Tensor, optional): Mask of input tensor (default None).

    Returns:
        Tuple:
            output (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model).
            attn_weights (torch.Tensor): Attention weights tensor of shape (batch_size, num_heads, max_len, max_len)."""
    # print(f"Multihead Attention input shape: {x.shape}")

    batch_size = x.size(0)

    Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    output = self.W_o(attn_output)
    # print(f"Multihead Attention output shapes: {output.shape, attn_weights.shape}")
    return output, attn_weights

# Feed Forward Network
class FeedForwardNetwork(nn.Module):
  """Feed-forward network module.

  Args:
      d_model (int): Embedding dimension (default 512).
      d_ff (int): Feed forward network dimension (default 2048).
      dropout (float): dropout value (default 0.1)."""

  def __init__(self, d_model: int=512, d_ff: int=2048, dropout: float=0.1) -> None:
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
    # print(f"Feedforward network input shape: {x.shape}")

    x = self.ffn_block(x)
    # print(f"Feedforward network output shape: {x.shape}")
    return x

# Single Encoder Layer
class EncoderLayer(nn.Module):
  """A single Encoder layer

  Args:
      d_model (int): Embedding dimension.
      num_heads (int): Number of attention heads.
      d_ff (int): Feed foward network dimension.
      dropout (float): Dropout value (default 0.1)."""

  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout=0.1) -> None:
    super().__init__()
    self.mha = MultiHeadSelfAttention(d_model, num_heads)
    self.layernorm1 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(p=dropout)
    self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.dropout2 = nn.Dropout(p=dropout)

  def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """Forward pass for the encoder layer.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, max_len, d_model).
        mask (torch.Tensor, optional): Optional mask of input tensor (default None).

    Returns:
        x (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model)."""
    # print(f"Encoder layer input shape: {x.shape}")

    attn_output, _ = self.mha(x, mask) # Multi-head attention
    x = self.layernorm1(x + self.dropout1(attn_output)) # Add and norm

    ffn_output = self.ffn(x) # Feed forward network
    x = self.layernorm2(x + self.dropout2(ffn_output)) # Add and norm

    # print(f"Encoder layer output shape: {x.shape}")
    return x

# Single Decoder Layer
class DecoderLayer(nn.Module):
  """A single Decoder layer.

  Args:
      d_model (int): Embedding dimension.
      num_heads (int): Number of attention heads.
      d_ff (int): Feed forward network dimension.
      dropout (float): Dropout value (default 0.1)"""

  def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.1) -> None:
    super().__init__()
    self.masked_mha = MultiHeadSelfAttention(d_model, num_heads) # Masked MHA
    self.layernorm1 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(p=dropout)
    self.mha = MultiHeadSelfAttention(d_model, num_heads)
    self.layernorm2 = nn.LayerNorm(d_model)
    self.dropout2 = nn.Dropout(p=dropout)
    self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
    self.layernorm3 = nn.LayerNorm(d_model)
    self.dropout3 = nn.Dropout(p=dropout)

  def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor=None, padding_mask: torch.Tensor=None) -> torch.Tensor:
    """Forward pass for decoder layer.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, max_len, d_model).
        encoder_output (torch.Tensor): Output tensor from the encoder, shape (batch_size, max_len, d_model).
        mask (torch.Tensor, optional): Mask tensor for masked multi-head attention (default None).
        padding_mask (torch.Tensor, optional): Mask tensor for padding (default None).

    Returns:
        x (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model)."""
    # print(f"Decoder Layer input shapes: {x.shape, encoder_output.shape}")

    attn_output, _ = self.masked_mha(x, mask) # masked multi-head attention
    x = self.layernorm1(x + self.dropout1(attn_output)) # add and norm

    attn_output, _ = self.mha(x, encoder_output) # multi-head attention
    x = self.layernorm2(x + self.dropout2(attn_output)) # add and norm

    ffn_output = self.ffn(x) # feed forward network
    x = self.layernorm3(x + self.dropout3(ffn_output)) # add and norm

    # print(f"Decoder layer output shape: {x.shape}")
    return x

# Encoder block
class Encoder(nn.Module):
  """Encoder block.

  Args:
      d_model (int): Embedding dimension.
      d_ff (int): Feed forward network dimension.
      num_heads (int): Number of attention heads.
      max_len (int): Max length of input sequence. (default 10000).
      dropout (float): Dropout value (default 0.1).
      num_layers (int): Number of encoder layers (default 6)."""

  def __init__(self, d_model: int, d_ff: int, num_heads: int, max_len: int=10000, dropout: float=0.1, num_layers: int=6) -> None:
    super().__init__()
    self.d_model = d_model
    self.position_encoding = PositionEncoding(d_model, max_len)
    self.layers = nn.Sequential(*[
        EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
    ])

  def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """Forward pass for the encoder block.

    Args:
        x (torch.Tensor): Input tensor of shape (d_model, max_len).
        mask (torch.Tensor, optional): Optional mask for input tensor.

    Returns:
        x (torch.Tensor): Output tensor of shape (batch_size, d_model, max_len)."""
    # print(f"Encoder input shape: {x.shape}")

    x = self.position_encoding(x)

    for layer in self.layers:
      x = layer(x, mask)

    # print(f"Encoder output shape: {x.shape}")
    return x

# Decoder block
class Decoder(nn.Module):
  """Decoder block.

  Args:
      d_model (int): Embedding dimension.
      d_ff (int): Feed forward network dimension.
      num_heads (int): Number of attention heads.
      max_len (int): Max length of input sequence (default 10000).
      dropout (float): Dropout value (default 0.1).
      num_layers (int): Number of decoder layers (default 6)."""

  def __init__(self, d_model: int, d_ff: int, num_heads: int, max_len: int=10000, dropout: float=0.1, num_layers: int=6) -> None:
    super().__init__()
    self.d_model = d_model
    self.position_encoding = PositionEncoding(d_model, max_len)
    self.layers = nn.Sequential(*[
        DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
    ])
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor=None, padding_mask: torch.Tensor=None) -> torch.Tensor:
    """Forward pass for the decoder block.

    Args:
        x (torch.Tensor): Input tensor of shape (d_model, max_len).
        encoder_output (torch.Tensor): Output tensor from the encoder block, shape (batch_size, d_model, max_len).
        mask (torch.Tensor, optional): Optional mask of input tensor.
        padding_mask (torch.Tensor, optional): Optional padding mask.

    Returns:
        x (torch.Tensor): Output tensor of shape (batch_size, max_len, d_model)."""
    # print(f"Decoder input shapes: {x.shape, encoder_output.shape}")

    x = self.position_encoding(x)
    x = self.dropout(x)

    for layer in self.layers:
      x = layer(x, encoder_output, mask)

    # print(f"Decoder output shape: {x.shape}")
    return x

# Transformer
class Transformer(nn.Module):
  """Transformer block.

  Args:
      d_model (int): Embedding dimension.
      d_ff (int): Feed forward network dimension.
      target_size (int): Target size for the output sequence.
      max_len (int): Max length of the input sequence (default 10000).
      dropout (float): Dropout value (default 0.1).
      num_layers (int): Number of encoder and decoder layers (default 6)."""

  def __init__(self, d_model: int, d_ff: int, num_heads: int, target_size: int, max_len: int=10000, dropout: float=0.1, num_layers: int=6) -> None:
    super().__init__()
    self.encoder = Encoder(d_model, d_ff, num_heads, max_len, dropout, num_layers)
    self.decoder = Decoder(d_model, d_ff, num_heads, max_len, dropout, num_layers)
    self.linear = nn.Linear(in_features=d_model, out_features=target_size)

  def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, encoder_mask: torch.Tensor=None, decoder_mask: torch.Tensor=None) -> torch.Tensor:
    """Forward pass for the transformer block.

    Args:
        encoder_input (torch.Tensor): Input tensor for the encoder, shape (batch_size, max_len, d_model).
        decoder_input (torch.Tensor): Input tensor for the decoder, shape (batch_size, max_len, d_model).
        encoder_mask (torch.Tensor, optional): Optional mask for the encoder input.
        decoder_mask (torch.Tensor, optional): Optional mask for the decoder input.

    Returns:
        x (torch.Tensor): Output from the transformer, shape (batch_size, d_model, max_len)"""
    # print(f"Transformer input shapes: {encoder_input.shape, decoder_input.shape}")

    encoder_output = self.encoder(encoder_input, encoder_mask)
    decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask)
    output = self.linear(decoder_output)

    # print(f"Transformer output shape: {output.shape}")
    return output
