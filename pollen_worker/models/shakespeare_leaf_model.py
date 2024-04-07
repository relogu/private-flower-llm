"""An LSTM model for Shakespeare dataset.

Originally from: LEAF: A Benchmark for Federated Settings. CoRR abs/1812.01097 (2018).
"""

import torch
from torch import nn

LEAF_CHARACTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


class ShakespeareLeafNet(nn.Module):
    """Create Shakespeare model for LEAF baselines.

    Args:
        chars (str, optional): String of possible characters (letters+digits).
            Defaults to LEAF_CHARACTERS.
        seq_len (int, optional): Length of each sequence. Defaults to 80.
        hidden_size (int, optional): Size of hidden layer. Defaults to 256.
        embedding_dim (int, optional): Dimension of embedding. Defaults to 8.
    """

    def __init__(
        self,
        chars: str = LEAF_CHARACTERS,
        seq_len: int = 80,
        hidden_size: int = 256,
        embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.dict_size = len(chars)
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(self.dict_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,  # Notice batch is first dim now
        )
        self.decoder = nn.Linear(self.hidden_size, self.dict_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forward sentence to obtain next character.

        Args:
            sentence (torch.Tensor): Tensor containing indices of characters

        Returns
        -------
            torch.Tensor: Vector encoding position of predicted character
        """
        encoded_seq = self.encoder(sentence.long())  # (batch, seq_len, embedding_dim)
        _, (h_n, _) = self.lstm(encoded_seq)  # (batch, seq_len, hidden_size)
        pred = self.decoder(h_n[-1])
        return pred
