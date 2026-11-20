"""Mean pooling for IMDb and the random-initialisation control from Task 3."""

import numpy as np
from torch import nn


class EmbedAndPoolEncoder(nn.Module):
    """Represent a review by the mean of its word vectors, or optionally their sum."""

    def __init__(self, embedding_matrix, average_pool=True, padding_idx=0, freeze_embeddings=True):
        super().__init__()
        self.average_pool = average_pool
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx
        )

    @property
    def enc_dim(self):
        return self.embedding.embedding_dim

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        # The IMDb embedding tables use a zero vector for padding.
        total = self.embedding(x).sum(dim=-2)
        divisor = 1
        if self.average_pool:
            divisor = x.ne(self.pad_idx).sum(dim=-1, keepdim=True).clamp_min(1)
        return total / divisor


def random_embedding_table(shape, padding_idx):
    """Replace pretrained values while preserving the vocabulary and vector size."""
    table = np.random.default_rng(0).standard_normal(shape, dtype=np.float32)
    table *= 1 / np.sqrt(shape[1])
    table[padding_idx] = 0
    return table
