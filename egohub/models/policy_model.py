import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input sequence.
    From the original 'Attention Is All You Need' paper.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LatentPolicyModel(nn.Module):
    """
    A Transformer-based model to learn from sequences of latent vectors.

    This model takes a sequence of latent vectors and can be used for various
    pre-training tasks, such as next-step prediction or masked modeling.

    Args:
        latent_dim: The dimensionality of the input latent vectors.
        nhead: The number of heads in the multiheadattention models.
        d_hid: The dimension of the feedforward network model.
        nlayers: The number of sub-encoder-layers in the encoder.
        dropout: The dropout value.
    """

    def __init__(
        self,
        latent_dim: int,
        nhead: int = 4,
        d_hid: int = 256,
        nlayers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(latent_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            latent_dim, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.latent_dim = latent_dim

        # Output layer to predict the next latent vector
        self.output_predictor = nn.Linear(latent_dim, latent_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.output_predictor.bias.data.zero_()
        self.output_predictor.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: The sequence to the encoder. Shape: [batch_size, seq_len, latent_dim].
            src_mask: The additive mask for the src sequence.

        Returns:
            output: The predicted next latent vector. Shape:
                [batch_size, seq_len, latent_dim].
        """
        src = src * math.sqrt(self.latent_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.output_predictor(output)
        return output
