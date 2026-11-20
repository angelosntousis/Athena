"""LSTM encoding and the half-data comparison from Task 4."""

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMEncoder(nn.Module):
    """Represent a review by the hidden state after its last token."""

    def __init__(self, hidden_size, embedding_matrix, padding_idx=0, freeze_embeddings=True, word_dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size, batch_first=True)

    @property
    def enc_dim(self):
        return self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        embedded = self.embedding(x)
        valid = x.ne(self.pad_idx)
        if self.training and self.word_dropout > 0:
            drop = (torch.rand(x.shape, device=embedded.device) < self.word_dropout) & valid
            embedded = embedded.masked_fill(drop.unsqueeze(-1), 0.0)

        # Packing stops the LSTM at the last real token in each review.
        lengths = valid.sum(dim=-1).cpu()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]


def half_training_indices(fit_indices, labels):
    """Select half the fitting reviews with the original sentiment proportions."""
    subset, _ = train_test_split(
        fit_indices, train_size=0.5, stratify=labels[fit_indices], random_state=0
    )
    return subset
