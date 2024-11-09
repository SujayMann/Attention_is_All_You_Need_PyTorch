import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    """Transformer block.

    Args:
        d_model (int): Embedding dimension.
        d_ff (int): Feed forward network dimension.
        num_heads (int): Number of attention heads.
        target_size (int): Target size for the output sequence.
        max_len (int): Max length of the input sequence (default 128).
        dropout (float): Dropout value (default 0.1).
        num_layers (int): Number of encoder and decoder layers (default 6)."""

    def __init__(self, d_model: int, d_ff: int, num_heads: int, target_size: int, max_len: int = 128, dropout: float = 0.1, num_layers: int = 6) -> None:
        super().__init__()
        self.encoder = Encoder(d_model, d_ff, num_heads, max_len, dropout, num_layers) 
        self.decoder = Decoder(d_model, d_ff, num_heads, max_len, dropout, num_layers, target_size)
        self.linear = nn.Linear(in_features=d_model, out_features=target_size)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, encoder_mask: torch.Tensor = None, decoder_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the transformer block

        Args:
            encoder_input (torch.Tensor): Input tensor for the encoder, shape (batch_size, max_len, d_model).
            decoder_input (torch.Tensor): Input tensor for the decoder, shape (batch_size, max_len, d_model).
            encoder_mask (torch.Tensor, optional): Optional mask for the encoder input.
            decoder_mask (torch.Tensor, optional): Optional mask for the decoder input.

        Returns:
            x (torch.Tensor): Output from the transformer, shape (batch_size, d_model, max_len)"""

        # Pass through encoder
        encoder_output = self.encoder(encoder_input, encoder_mask)

        # Pass through decoder
        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask, encoder_mask)

        # Linear transformation to get the output
        output = self.linear(decoder_output)
        return output
