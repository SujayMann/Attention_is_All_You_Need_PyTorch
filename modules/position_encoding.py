import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PositionEncoding(torch.nn.Module):
    """Positional Encoding for Transformer model.

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum length of input sequences (default 128).
        device (str): 'cpu' or 'cuda'"""

    def __init__(self, d_model: int, max_len: int = 128, device=device) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)

        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model).to(device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to add positional encoding to input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_len, d_model).

        Returns:
            x (torch.Tensor): Output tensor with position encoding added to it of shape (batch_size, max_len, d_model)."""

        x = x.to(device)
        x = x + self.pe[:x.size(1), :]  # Apply positional encoding
        return x
