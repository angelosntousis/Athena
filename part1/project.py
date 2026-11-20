"""Athena Project 1: IMDb sentiment classification in one file.

Implemented from tokenisation.py, features.py, loglin.py, smlp.py,
train_loglin.py, train_mlp.py, analysis.py and rating_rule.py.

Examples (activate ~/envs/nlp1-2026 first):
    python part1/project.py data
    python part1/project.py run --model lr --representation tf-idf-bigrams-nb
    python part1/project.py run --model mlp --hidden-layers 2 --hidden-dim 256
    python part1/project.py sweep part1/configs/final/mlp_1x128.yaml --local --count 1
    python part1/project.py compare results/final_predictions
    python part1/project.py rule results/final_predictions/lr_bow_seed-0.npz

Training uses stanfordnlp/imdb's training split. Tokenizers learn from its
separate unlabelled split. Cross-validation uses the tutorial's shuffled folds;
MLPs use an inner validation split to choose an epoch. This file needs the
packages in requirements.txt; W&B sweeps additionally need wandb and PyYAML.
Results go to results/final/<run>/; model and fitted-feature saving is off by
default. Use --save-models true to keep them. No existing project module is imported.

AI assistance: Codex consolidated and checked the team's existing implementation.
"""

import argparse
import csv
import itertools
import json
import os
import pickle
import random
import re
import tempfile
import time
import uuid
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final"
DATASET = "stanfordnlp/imdb"
KINDS = ("word", "bpe")
MIN_FREQUENCY = 2
UNK = "[UNK]"


# Tokenisation and text cleaning.

def train_tokenizer(kind, vocab_size, corpus=None, min_frequency=MIN_FREQUENCY):
    """
    Constructs and trains a tokeniser.

    kind: "word" or "bpe"
    vocab_size: maximum number of distinct tokens, [UNK] included
    corpus: list of texts to learn from; defaults to the 50k unlabelled IMDb reviews
    min_frequency: words (word-level) or pairs (BPE) seen fewer times are not added to the vocabulary
    """
    if kind == "word":
        tokenizer = Tokenizer(WordLevel(unk_token=UNK))
        trainer = WordLevelTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[UNK])
    elif kind == "bpe":
        # the ## prefix marks pieces inside a word (##able), so they differ from whole words (able)
        tokenizer = Tokenizer(BPE(unk_token=UNK, continuing_subword_prefix="##"))
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[UNK],
                             continuing_subword_prefix="##")
    else:
        raise ValueError(f"Unknown kind={kind}, expected one of {KINDS}")

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace(Regex(r"<br\s*/?>"), " "),  # HTML line breaks
        # the replacements come before NFKC, which would turn ´ into a space plus an accent
        normalizers.Replace("\x96", "-"),  # dashes and ellipses mis-decoded from Windows-1252
        normalizers.Replace("\x97", "-"),
        normalizers.Replace(Regex(r"-{2,}"), " - "),  # -- and --- used as dashes, so word--word splits like word - word
        normalizers.Replace("\x85", "..."),
        normalizers.Replace("\x91", "'"),  # ‘ mis-decoded from Windows-1252
        normalizers.Replace("’", "'"),  # ’ and ´ used as apostrophes, so didn’t/didn´t split like didn't
        normalizers.Replace("´", "'"),
        normalizers.NFKC(),
        # cap repeated punctuation (after NFKC, which turns … into ...): !!!!! -> !!, ????? -> ??, ...... -> ...
        normalizers.Replace(Regex(r"!{3,}"), "!!"),
        normalizers.Replace(Regex(r"\?{3,}"), "??"),
        normalizers.Replace(Regex(r"\.{4,}"), "..."),
        normalizers.Lowercase(),
    ])
    # words, with clitics split off (didn't -> did n't, kubrick's -> kubrick 's), punctuation as its own token
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"\w+(?=n't)|n't|'(?:s|re|ve|ll|d|m)\b|\w+|[^\w\s]"), behavior="removed", invert=True)

    if corpus is None:
        corpus = load_dataset(DATASET, split="unsupervised")["text"]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def get_tokenizer(kind, vocab_size, folder=ROOT / "results" / "tokenizers", min_frequency=MIN_FREQUENCY):
    """
    Returns the tokeniser saved as {folder}/{kind}_vs_{vocab_size}_mf_{min_frequency}.json,
    training and saving it first if it does not exist yet.
    """
    path = Path(folder) / f"{kind}_vs_{vocab_size}_mf_{min_frequency}.json"
    if path.exists():
        return Tokenizer.from_file(str(path))
    tokenizer = train_tokenizer(kind, vocab_size, min_frequency=min_frequency)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


# Sparse feature representations. Fit IDF, pair vocabularies and NB weights on training data.

class BagOfWords:
    """Raw token counts. min_df filters bigrams only; unigram columns follow token IDs."""

    def __init__(self, tokenizer: Tokenizer, ngram_range=(1, 1), min_df=2):
        if ngram_range not in ((1, 1), (1, 2), (2, 2)):
            raise ValueError("ngram_range must be (1, 1), (1, 2), or (2, 2).")
        self.tokenizer = tokenizer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self._size = tokenizer.get_vocab_size()
        self._include_unigrams = ngram_range[0] == 1
        self._bigrams = BigramCounts(tokenizer, min_df=min_df) if ngram_range[1] == 2 else None

    def __len__(self) -> int:
        """Number of feature columns; bigram encoders must be fitted first."""
        return ((self._size if self._include_unigrams else 0)
                + (len(self._bigrams) if self._bigrams is not None else 0))

    def _unigram_counts(self, texts) -> csr_matrix:
        rows, columns, values = [], [], []
        n_documents = 0
        for row, text in enumerate(texts):
            counts = Counter(self.tokenizer.encode(text).ids)
            for token_id, count in counts.items():
                rows.append(row)
                columns.append(token_id)
                values.append(count)
            n_documents += 1
        return csr_matrix(
            (values, (rows, columns)),
            shape=(n_documents, self._size),
            dtype=np.float64,
        )

    def fit(self, training_texts):
        """Learn eligible bigram columns from training documents only."""
        if self._bigrams is not None:
            self._bigrams.fit(training_texts)
        return self

    def fit_transform(self, training_texts) -> csr_matrix:
        """Fit then encode, accepting a one-use iterable of training strings."""
        texts = list(training_texts)
        self.fit(texts)
        return self.transform(texts)

    def transform(self, texts) -> csr_matrix:
        """Encode a batch using fixed columns, with no vocabulary/IDF update."""
        texts = list(texts)
        # len also checks that any learned bigram vocabulary is fitted.
        dimension = len(self)
        if not texts:
            return csr_matrix((0, dimension), dtype=np.float64)
        parts: list[csr_matrix] = []
        if self._include_unigrams:
            parts.append(self._unigram_counts(texts))
        if self._bigrams is not None:
            parts.append(self._bigrams.transform(texts))
        return parts[0] if len(parts) == 1 else csr_matrix(hstack(parts, format="csr"))

    def get_feature_names_out(self) -> np.ndarray:
        """Human-readable labels in exactly the matrix-column order."""
        names = []
        if self._include_unigrams:
            names.extend(f"unigram:{self.tokenizer.id_to_token(i)}" for i in range(self._size))
        if self._bigrams is not None:
            for pair in self._bigrams.vectorizer.get_feature_names_out():
                left, right = map(int, pair.split(":"))
                names.append(f"bigram:{self.tokenizer.id_to_token(left)} → "
                             f"{self.tokenizer.id_to_token(right)}")
        return np.asarray(names, dtype=object)

    def __call__(self, text: str) -> np.ndarray:
        """Return one dense vector, as expected by the tutorial classifiers."""
        return self.transform([text]).toarray()[0]


class TokenIndicator(BagOfWords):
    """Binary presence of each retained token or adjacent pair."""

    def transform(self, texts) -> csr_matrix:
        counts = super().transform(texts)
        assert counts.data is not None
        counts.data[:] = 1.0
        return counts


class NormalisedBagOfWords(BagOfWords):
    """Counts divided by each document's L1 or L2 norm."""

    def __init__(self, tokenizer: Tokenizer, ngram_range=(1, 1), min_df=2, norm="l2"):
        if norm not in ("l1", "l2"):
            raise ValueError("norm must be 'l1' or 'l2'.")
        super().__init__(tokenizer, ngram_range=ngram_range, min_df=min_df)
        self.norm = norm

    def transform(self, texts) -> csr_matrix:
        counts = super().transform(texts)
        if counts.shape[0] == 0:
            return counts
        return csr_matrix(normalize(counts, norm=self.norm))


class TfidfEncoder(BagOfWords):
    """Smoothed IDF with optional log-TF, followed by L2 normalisation."""

    def __init__(self, tokenizer: Tokenizer, ngram_range=(1, 1), min_df=2, sublinear_tf=False):
        super().__init__(tokenizer, ngram_range=ngram_range, min_df=min_df)
        self.transformer = TfidfTransformer(
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=sublinear_tf
        )

    def fit(self, training_texts):
        texts = list(training_texts)
        if not texts:
            raise ValueError("TF-IDF requires at least one training document.")
        super().fit(texts)
        counts = super().transform(texts)
        self.transformer.fit(counts)
        return self

    def transform(self, texts) -> csr_matrix:
        check_is_fitted(self.transformer, "idf_")
        counts = super().transform(texts)
        shape = counts.shape
        assert shape is not None
        if shape[0] == 0:
            return counts
        return csr_matrix(self.transformer.transform(counts))


class NaiveBayesWeighting:
    """Multiply features by their smoothed class log-ratio (Wang & Manning, 2012)."""

    def __init__(self, encoder: BagOfWords, alpha=1.0, renormalise=True):
        self.encoder = encoder
        self.alpha = alpha
        self.renormalise = renormalise
        self.r = None

    def __len__(self) -> int:
        return len(self.encoder)

    def fit(self, training_texts, labels):
        texts = list(training_texts)
        labels = np.asarray(labels)
        if set(np.unique(labels)) - {0, 1}:
            raise ValueError("Naive Bayes weighting expects binary labels 0 and 1.")
        present = self.encoder.fit_transform(texts)
        present.data[:] = 1.0  # documents containing each feature, as in the paper's binarised features
        p = self.alpha + np.asarray(present[labels == 1].sum(axis=0)).ravel()
        q = self.alpha + np.asarray(present[labels == 0].sum(axis=0)).ravel()
        self.r = np.log(p / p.sum()) - np.log(q / q.sum())
        return self

    def fit_transform(self, training_texts, labels) -> csr_matrix:
        texts = list(training_texts)
        return self.fit(texts, labels).transform(texts)

    def transform(self, texts) -> csr_matrix:
        if self.r is None:
            raise ValueError("NaiveBayesWeighting must be fitted first.")
        weighted = csr_matrix(self.encoder.transform(texts).multiply(self.r[np.newaxis, :]))
        if weighted.shape[0] == 0:
            return weighted
        return csr_matrix(normalize(weighted, norm="l2")) if self.renormalise else weighted

    def __call__(self, text: str) -> np.ndarray:
        return self.transform([text]).toarray()[0]


class BigramCounts:
    """Adjacent token-ID pairs; fit a vocabulary using training documents only."""

    def __init__(self, tokenizer: Tokenizer, min_df=2):
        self.tokenizer = tokenizer
        self.vectorizer = CountVectorizer(
            analyzer=self._pairs, min_df=min_df, dtype=np.float64
        )

    def _pairs(self, text):
        ids = self.tokenizer.encode(text).ids
        return [f"{left}:{right}" for left, right in zip(ids, ids[1:])]

    def fit_transform(self, training_texts) -> csr_matrix:
        return csr_matrix(self.vectorizer.fit_transform(training_texts))

    def fit(self, training_texts):
        self.fit_transform(training_texts)
        return self

    def __len__(self):
        check_is_fitted(self.vectorizer, "vocabulary_")
        return len(self.vectorizer.vocabulary_)

    def transform(self, texts) -> csr_matrix:
        return csr_matrix(self.vectorizer.transform(texts))

    def __call__(self, text):
        return self.transform([text]).toarray()[0]


# Models: logistic regression and the tutorial's MLP with configurable depth.

class SparseMLP(nn.Module):
    """Sparse input, ReLU/dropout hidden layers, then class log-probabilities."""

    def __init__(self, input_dim, hidden_dim, num_classes=2, dropout=0.1, hidden_layers=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        layers = []
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Sequential(nn.Linear(hidden_dim, num_classes), nn.LogSoftmax(dim=-1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        hidden = torch.sparse.mm(x, self.weight) + self.bias
        return self.output(self.hidden(self.dropout(self.activation(hidden))))


def scipy_to_torch_sparse(matrix):
    """Convert a batch without making its vocabulary-sized input dense."""
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    values = torch.from_numpy(matrix.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, matrix.shape, check_invariants=False).coalesce()


class SparseDataset(Dataset):
    def __init__(self, matrix, labels):
        self.matrix = matrix
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, index):
        return self.matrix[index], self.labels[index]


def sparse_collate(batch):
    rows, labels = zip(*batch)
    return scipy_to_torch_sparse(vstack(rows)), torch.stack(labels)


def epoch_metrics(model, loader, device, optimizer=None):
    """An optimizer enables training; otherwise evaluate with dropout disabled."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, size = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            log_probs = model(features)
            loss = nn.functional.nll_loss(log_probs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (log_probs.argmax(dim=1) == labels).sum().item()
            size += len(labels)
    return {"loss": total_loss / size, "accuracy": correct / size}


def train_mlp(features, labels, validation_features, validation_labels, config):
    """Restore the epoch with lowest inner-validation loss before outer-fold scoring."""
    train_loader = DataLoader(
        SparseDataset(features, labels), batch_size=config["batch_size"],
        shuffle=True, collate_fn=sparse_collate,
    )
    validation_loader = DataLoader(
        SparseDataset(validation_features, validation_labels), batch_size=config["batch_size"],
        shuffle=False, collate_fn=sparse_collate,
    )
    model = SparseMLP(
        features.shape[1], config["hidden_dim"], dropout=config["dropout_rate"],
        hidden_layers=config["hidden_layers"],
    ).to(config["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    history, best_loss, best_epoch = [], float("inf"), 0

    # A single temporary checkpoint avoids keeping another large model in RAM.
    with tempfile.TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "best.pt"
        for epoch in range(config["max_epochs"]):
            training = epoch_metrics(model, train_loader, config["device"], optimizer)
            validation = epoch_metrics(model, validation_loader, config["device"])
            history.append({"training": training, "validation": validation})
            print(
                f"Epoch {epoch + 1}: train loss {training['loss']:.4f}, "
                f"accuracy {training['accuracy']:.4f}; validation loss "
                f"{validation['loss']:.4f}, accuracy {validation['accuracy']:.4f}", flush=True,
            )
            if validation["loss"] <= best_loss:
                best_loss, best_epoch = validation["loss"], epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= config["patience"]:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=config["device"], weights_only=True))

    best = history[best_epoch]
    metrics = dict(
        best_epoch=best_epoch + 1, epochs_run=len(history),
        train_accuracy=best["training"]["accuracy"],
        validation_accuracy=best["validation"]["accuracy"], validation_loss=best_loss,
        num_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )
    return model, metrics, history


def predict_mlp(model, features, batch_size=256):
    model.eval()
    probabilities = []
    device = model.weight.device
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = scipy_to_torch_sparse(features[start:start + batch_size]).to(device)
            probabilities.append(model(batch).exp().cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, 2))


# Shared experiment settings. Existing YAML sweep files can override these.

COMMON_DEFAULTS = dict(
    model="lr", tokenizer="bpe32k", representation="tf-idf", ngram_range=[1, 1],
    min_df=2, sublinear_tf=False, nb_alpha=1.0, folds=5, split_seeds=[0], seed=0,
    save_models=False,
)
MODEL_DEFAULTS = {
    "lr": dict(regularisation="l2", regularisation_coefficient=1.0, epochs=100, tol=1e-4),
    "mlp": dict(
        hidden_layers=1, hidden_dim=128, dropout_rate=0.1, learning_rate=1e-4,
        batch_size=64, weight_decay=1e-4, max_epochs=10, patience=3,
        validation_fraction=0.1, device="cpu",
    ),
}


def numbers(value):
    """PyYAML may read scientific notation such as 1e-4 as a string."""
    if isinstance(value, list):
        return [numbers(item) for item in value]
    if isinstance(value, str) and re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", value):
        return float(value)
    return value


def prepare_config(values):
    model = values.get("model", "lr")
    if model not in MODEL_DEFAULTS:
        raise ValueError("model must be 'lr' or 'mlp'.")
    config = {**COMMON_DEFAULTS, **MODEL_DEFAULTS[model]}
    unknown = set(values) - set(config)
    if unknown:
        raise ValueError(f"Unknown settings for {model}: {sorted(unknown)}")
    config.update({key: numbers(value) for key, value in values.items()})
    if isinstance(config["split_seeds"], (int, float)):
        config["split_seeds"] = [config["split_seeds"]]
    integer_keys = ["folds", "seed", "min_df", "epochs", "max_epochs", "patience",
                    "hidden_dim", "hidden_layers", "batch_size"]
    for key in integer_keys:
        if key in config:
            value = config[key]
            if int(value) != value or value < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be an integer in its valid range.")
            config[key] = int(value)
    config["split_seeds"] = [int(seed) for seed in config["split_seeds"]]
    if not config["split_seeds"] or len(set(config["split_seeds"])) != len(config["split_seeds"]):
        raise ValueError("Use at least one split seed, without duplicates.")
    if config["folds"] < 2 or config["nb_alpha"] <= 0:
        raise ValueError("Use at least two folds and positive NB smoothing.")
    if not isinstance(config["save_models"], bool) or not isinstance(config["sublinear_tf"], bool):
        raise ValueError("save_models and sublinear_tf must be booleans.")
    if model == "lr":
        penalty = config["regularisation"]
        if isinstance(penalty, (int, float)):
            config.update(regularisation="l2", regularisation_coefficient=float(penalty))
        elif "_" in penalty:
            penalty, _, strength = penalty.partition("_")
            config.update(regularisation=penalty, regularisation_coefficient=float(strength))
        if config["regularisation"] not in ("l1", "l2", "no", "none"):
            raise ValueError("Use l1, l2, none, or a nonnegative L2 coefficient.")
        if config["regularisation_coefficient"] < 0:
            raise ValueError("The regularisation coefficient cannot be negative.")
    elif not (0 <= config["dropout_rate"] < 1 and 0 < config["validation_fraction"] < 1):
        raise ValueError("Check dropout_rate and validation_fraction.")
    elif config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Use a positive learning rate and nonnegative weight decay.")
    return config


def parse_tokenizer(name):
    match = re.fullmatch(r"(word|bpe)_?(\d+(?:\.\d+)?k?)?", name)
    if match is None:
        raise ValueError("Use word or bpe, optionally with a size, such as bpe32k.")
    kind, size = match.groups()
    if size is None:
        return kind, 32000
    size = float(size[:-1]) * 1000 if size.endswith("k") else float(size)
    if size < 2 or not size.is_integer():
        raise ValueError("Tokenizer vocabulary size must be an integer greater than 1.")
    return kind, int(size)


def make_encoder(config, tokenizer):
    name = config["representation"].replace("token_indicators", "token_indicator")
    naive_bayes = name.endswith("-nb")
    name = name.removesuffix("-nb")
    bigrams = name.endswith("-bigrams")
    name = name.removesuffix("-bigrams")
    ngrams = (1, 2) if bigrams else tuple(config["ngram_range"])
    min_df = config["min_df"]
    if name == "token_indicator":
        encoder = TokenIndicator(tokenizer, ngrams, min_df)
    elif name == "bow":
        encoder = BagOfWords(tokenizer, ngrams, min_df)
    elif name in ("bow_l1", "bow_l2"):
        encoder = NormalisedBagOfWords(tokenizer, ngrams, min_df, norm=name[-2:])
    elif name == "tf-idf":
        encoder = TfidfEncoder(tokenizer, ngrams, min_df, sublinear_tf=config["sublinear_tf"])
    else:
        raise ValueError(f"Unknown representation: {config['representation']}")
    return NaiveBayesWeighting(encoder, alpha=config["nb_alpha"]) if naive_bayes else encoder


def cross_validation_split(size, folds, seed):
    """The tutorial's Python-random shuffle, including its training-row order."""
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    parts = np.array_split(np.asarray(indices), folds)
    for fold, evaluation_ids in enumerate(parts):
        training_ids = np.concatenate([part for index, part in enumerate(parts) if index != fold])
        yield fold, training_ids, evaluation_ids


def fit_logistic_regression(features, labels, config):
    strength = config["regularisation_coefficient"]
    no_penalty = config["regularisation"] in ("no", "none") or strength == 0
    model = LogisticRegression(
        solver="saga", C=float("inf") if no_penalty else 1 / strength,
        l1_ratio=1.0 if config["regularisation"] == "l1" else 0.0,
        max_iter=config["epochs"], tol=config["tol"], random_state=config["seed"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, labels)
    metrics = dict(
        converged=not any(issubclass(warning.category, ConvergenceWarning) for warning in caught),
        epochs_run=int(model.n_iter_.max()), nonzero_weights=int(np.count_nonzero(model.coef_)),
    )
    return model, metrics


def save_model(folder, model, encoder, config):
    """Keep the fitted encoder with the weights so prediction can be reproduced."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    base = encoder.encoder if isinstance(encoder, NaiveBayesWeighting) else encoder
    base.tokenizer.save(str(folder / "tokenizer.json"))
    state = {}
    if isinstance(encoder, NaiveBayesWeighting):
        state["naive_bayes_r"] = encoder.r
    if isinstance(base, TfidfEncoder):
        state["idf"] = base.transformer.idf_
    if base._bigrams is not None:
        state["bigram_vocabulary"] = base._bigrams.vectorizer.vocabulary_
    # Store learned values, rather than pickling classes tied to a script's module name.
    with (folder / "features.pickle").open("wb") as file:
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
    if config["model"] == "lr":
        with (folder / "model.pickle").open("wb") as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        architecture = dict(
            input_dim=len(encoder), hidden_dim=config["hidden_dim"], num_classes=2,
            dropout=config["dropout_rate"], hidden_layers=config["hidden_layers"],
        )
        (folder / "model.json").write_text(json.dumps(architecture, indent=2))
        torch.save(model.state_dict(), folder / "model.pt")


def load_model(folder, device="cpu"):
    """Load a fold saved by this file. Only load your own trusted pickle files."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    encoder = make_encoder(config, tokenizer)
    with (folder / "features.pickle").open("rb") as file:
        state = pickle.load(file)
    base = encoder.encoder if isinstance(encoder, NaiveBayesWeighting) else encoder
    if base._bigrams is not None:
        base._bigrams.vectorizer.vocabulary_ = state["bigram_vocabulary"]
        base._bigrams.vectorizer.fixed_vocabulary_ = True
    if isinstance(base, TfidfEncoder):
        base.transformer.idf_ = state["idf"]
        base.transformer.n_features_in_ = len(base)
    if isinstance(encoder, NaiveBayesWeighting):
        encoder.r = state["naive_bayes_r"]
    if (folder / "model.pickle").exists():
        with (folder / "model.pickle").open("rb") as file:
            model = pickle.load(file)
    else:
        architecture = json.loads((folder / "model.json").read_text())
        model = SparseMLP(**architecture).to(device)
        model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
    return model, encoder


def predict_probs(texts, model, encoder):
    features = encoder.transform(texts)
    if isinstance(model, SparseMLP):
        return predict_mlp(model, features)
    return model.predict_proba(features)


def cross_validate(values, texts, labels, tokenizer=None, save_to=None, log=print):
    """Return resolved settings, fold scores, and held-out probabilities for one configuration."""
    config = prepare_config(values)
    labels = np.asarray(labels)
    if len(texts) != len(labels) or len(texts) < config["folds"]:
        raise ValueError("Need matching texts/labels and at least one review per fold.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("IMDb experiments require both binary labels: 0 and 1.")
    if tokenizer is None:
        tokenizer = get_tokenizer(*parse_tokenizer(config["tokenizer"]))
    records, predictions = [], []
    for split_seed in config["split_seeds"]:
        for fold, training_ids, evaluation_ids in cross_validation_split(len(texts), config["folds"], split_seed):
            print(f"Seed {split_seed}, fold {fold + 1}/{config['folds']}", flush=True)
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            fit_ids = training_ids
            if config["model"] == "mlp":
                shuffled = np.random.default_rng(config["seed"]).permutation(training_ids)
                size = round(config["validation_fraction"] * len(training_ids))
                if not 0 < size < len(training_ids):
                    raise ValueError("Not enough reviews for the inner validation split.")
                validation_ids, fit_ids = shuffled[:size], shuffled[size:]

            # The outer evaluation fold never fits IDF, pair vocabulary, NB weights or epochs.
            start = time.perf_counter()
            encoder = make_encoder(config, tokenizer)
            fit_texts = [texts[index] for index in fit_ids]
            if isinstance(encoder, NaiveBayesWeighting):
                features = encoder.fit_transform(fit_texts, labels[fit_ids])
            else:
                features = encoder.fit_transform(fit_texts)
            if config["model"] == "lr":
                model, metrics = fit_logistic_regression(features, labels[fit_ids], config)
                history = None
            else:
                validation_features = encoder.transform([texts[index] for index in validation_ids])
                model, metrics, history = train_mlp(
                    features, labels[fit_ids], validation_features, labels[validation_ids], config,
                )
            train_time = time.perf_counter() - start
            start = time.perf_counter()
            probabilities = predict_probs([texts[index] for index in evaluation_ids], model, encoder)
            predicted = probabilities.argmax(axis=1)
            record = dict(
                split_seed=split_seed, fold=fold, num_features=len(encoder),
                accuracy=accuracy_score(labels[evaluation_ids], predicted),
                macro_f1=f1_score(labels[evaluation_ids], predicted, average="macro"),
                train_time=train_time, predict_time=time.perf_counter() - start, **metrics,
            )
            records.append(record)
            predictions.append(dict(
                split_seed=np.full(len(evaluation_ids), split_seed), fold=np.full(len(evaluation_ids), fold),
                row_id=evaluation_ids, label=labels[evaluation_ids], prediction=predicted,
                prob_positive=probabilities[:, 1],
            ))
            if save_to is not None:
                fold_folder = Path(save_to) / f"seed-{split_seed}" / f"fold-{fold}"
                if config["save_models"]:
                    save_model(fold_folder, model, encoder, config)
                if history is not None:
                    fold_folder.mkdir(parents=True, exist_ok=True)
                    (fold_folder / "history.json").write_text(json.dumps(history, indent=2))
            log(record)
    combined = {key: np.concatenate([part[key] for part in predictions]) for key in predictions[0]}
    return config, records, combined


def run_experiment(config, texts, labels, output=RESULTS, wandb_run=None):
    run_id = wandb_run.id if wandb_run else f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    folder = Path(output) / run_id
    folder.mkdir(parents=True, exist_ok=False)

    def log(record):
        print(json.dumps(record), flush=True)
        with (folder / "folds.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        if wandb_run is not None:
            wandb_run.log({f"fold/{key}": value for key, value in record.items()})

    config = prepare_config(config)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    config, records, predictions = cross_validate(config, texts, labels, save_to=folder, log=log)
    summary = dict(
        accuracy=float(np.mean([record["accuracy"] for record in records])),
        macro_f1=float(np.mean([record["macro_f1"] for record in records])),
        train_time=float(np.mean([record["train_time"] for record in records])),
    )
    np.savez_compressed(
        folder / "predictions.npz", split_seed=predictions["split_seed"],
        fold=predictions["fold"], row_id=predictions["row_id"], label=predictions["label"],
        prediction=predictions["prediction"], prob_positive=predictions["prob_positive"],
    )
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    if wandb_run is not None:
        wandb_run.summary.update(summary)
    print(f"Saved {folder}\n{json.dumps(summary)}", flush=True)
    return summary


# Evaluation: paired outer-fold scores and the tutorial's Bayesian ROPE analysis.

def bayesian_comparison(scores_a, scores_b, runs=1, rope=0.005):
    """Posterior probabilities for A better, equivalent, and B better (difference B - A)."""
    import baycomp

    a, b = np.asarray(scores_a), np.asarray(scores_b)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite([a, b]).all():
        raise ValueError("Supply paired finite fold scores of equal length.")
    if runs < 1 or len(a) % runs or len(a) // runs < 2 or rope <= 0:
        raise ValueError("Need complete repeated CV scores and a positive ROPE.")
    # An exactly constant difference has a point-mass posterior, not a Student t density.
    difference = b - a
    if np.var(difference) == 0:
        delta = float(difference[0])
        probabilities = (float(delta < -rope), float(abs(delta) <= rope), float(delta > rope))
    else:
        probabilities = baycomp.CorrelatedTTest.probs(a, b, rope=rope, runs=runs)
    return dict(zip(("p_a_better", "p_equivalent", "p_b_better"), map(float, probabilities)))


def load_prediction_groups(paths):
    """Read this file's runs or the team's exported final_predictions directory."""
    groups = {}
    for source in map(Path, paths):
        index = source / "index.csv"
        if source.is_file():
            entries = [(source, source.stem)]
        elif index.exists():
            with index.open() as file:
                entries = [(source / row["file"], f"{row['model']} / {row['representation']}")
                           for row in csv.DictReader(file)]
        else:
            files = sorted(source.rglob("predictions.npz"))
            if not files:
                raise ValueError(f"No predictions found in {source}")
            entries = [(file, file.parent.name) for file in files]
        for file, name in entries:
            with np.load(file, allow_pickle=False) as saved:
                values = {key: saved[key].copy() for key in
                          ("split_seed", "fold", "row_id", "label", "prediction", "prob_positive")}
            groups.setdefault(name, []).append(values)
    combined = {}
    for name, parts in groups.items():
        values = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        keys = list(zip(values["split_seed"], values["row_id"]))
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate predictions for the same seed/review in {name}")
        combined[name] = values
    return combined


def paired_fold_scores(a, b):
    """Pair only shared seeds; verify the actual held-out reviews, labels and fold IDs."""
    seeds = sorted(set(a["split_seed"]) & set(b["split_seed"]))
    if not seeds:
        raise ValueError("The two systems have no shared split seeds.")
    scores_a, scores_b, fold_counts = [], [], []
    for seed in seeds:
        selected_a = np.flatnonzero(a["split_seed"] == seed)
        selected_b = np.flatnonzero(b["split_seed"] == seed)
        selected_a = selected_a[np.argsort(a["row_id"][selected_a])]
        selected_b = selected_b[np.argsort(b["row_id"][selected_b])]
        for key in ("row_id", "label", "fold"):
            if not np.array_equal(a[key][selected_a], b[key][selected_b]):
                raise ValueError(f"Unpaired {key} for split seed {seed}")
        folds = sorted(set(a["fold"][selected_a]))
        fold_counts.append(len(folds))
        for fold in folds:
            ia = selected_a[a["fold"][selected_a] == fold]
            ib = selected_b[b["fold"][selected_b] == fold]
            scores_a.append(accuracy_score(a["label"][ia], a["prediction"][ia]))
            scores_b.append(accuracy_score(b["label"][ib], b["prediction"][ib]))
    if len(set(fold_counts)) != 1:
        raise ValueError("Each split seed must have the same number of folds.")
    return scores_a, scores_b, seeds


def compare_results(paths, rope, output):
    groups = load_prediction_groups(paths)
    if len(groups) < 2:
        raise ValueError("At least two prediction groups are needed for a comparison.")
    for name, values in groups.items():
        accuracy = accuracy_score(values["label"], values["prediction"])
        print(f"{name}: {accuracy:.2%} pooled accuracy, {len(values['row_id'])} held-out predictions")
    rows = []
    for (name_a, a), (name_b, b) in itertools.combinations(groups.items(), 2):
        scores_a, scores_b, seeds = paired_fold_scores(a, b)
        row = dict(
            system_a=name_a, system_b=name_b, split_seeds=" ".join(map(str, seeds)),
            folds=len(scores_a), rope=rope,
            **bayesian_comparison(scores_a, scores_b, runs=len(seeds), rope=rope),
        )
        rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}. ROPE ±{100 * rope:g} percentage points; see shared seeds for each pair.")


# Optional rating rule, as implemented in rating_rule.py.

RATING_PATTERN = re.compile(
    r"(?<![\d/.])(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10(?![\d/])", re.IGNORECASE,
)


def parse_rating(text):
    """Return a single unambiguous rating out of ten, otherwise None."""
    ratings = {float(value) for value in RATING_PATTERN.findall(text) if float(value) <= 10}
    return ratings.pop() if len(ratings) == 1 else None


def rule_prediction(rating):
    if rating is None:
        return None
    if rating >= 7:
        return 1
    if rating <= 4:
        return 0
    return None


def apply_rule(texts, predictions):
    """Replace decisions where an explicit rating decides. Probabilities stay model outputs."""
    rule = [rule_prediction(parse_rating(text)) for text in texts]
    used = np.array([value is not None for value in rule])
    combined = np.array([value if value is not None else prediction
                         for value, prediction in zip(rule, predictions)])
    return combined, used


def score_rating_rule(paths, texts, labels, rope):
    for name, values in load_prediction_groups(paths).items():
        ids = values["row_id"]
        if not np.array_equal(np.asarray(labels)[ids], values["label"]):
            raise ValueError("The saved labels do not match this IMDb training split.")
        predicted, used = apply_rule([texts[index] for index in ids], values["prediction"])
        updated = {**values, "prediction": predicted}
        a, b, seeds = paired_fold_scores(values, updated)
        changes = predicted != values["prediction"]
        report = dict(
            system=name, rule_coverage=float(used.mean()),
            original_accuracy=accuracy_score(values["label"], values["prediction"]),
            combined_accuracy=accuracy_score(values["label"], predicted),
            corrected=int(np.sum(changes & (predicted == values["label"]))),
            lost=int(np.sum(changes & (values["prediction"] == values["label"]))),
            **bayesian_comparison(a, b, runs=len(seeds), rope=rope),
        )
        print(json.dumps(report, indent=2))


def load_train_split():
    data = load_dataset(DATASET, split="train")
    return list(data["text"]), np.asarray(data["label"])


def describe_data(split):
    """Label counts and character-length quartiles, as in the dataset exploration."""
    data = load_dataset(DATASET, split=split)
    texts, labels = list(data["text"]), np.asarray(data["label"])
    lengths = np.array([len(text) for text in texts])
    report = dict(
        dataset=DATASET, split=split, reviews=len(texts),
        labels={str(label): int(np.sum(labels == label)) for label in (0, 1)},
        character_length_quartiles=np.quantile(lengths, [0.25, 0.5, 0.75]).tolist(),
        reviews_with_single_rating=sum(parse_rating(text) is not None for text in texts),
    )
    print(json.dumps(report, indent=2))


# Command-line interface and existing W&B/local sweep support.

def grid(parameters):
    names = list(parameters)
    choices = [spec["values"] if "values" in spec else [spec["value"]]
               for spec in parameters.values()]
    for values in itertools.product(*choices):
        yield dict(zip(names, values))


def parse_value(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return numbers(value)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Use true or false.")


def add_training_arguments(parser):
    parser.add_argument("--model", choices=("lr", "mlp"), default=argparse.SUPPRESS)
    string_options = ("tokenizer", "representation", "device")
    int_options = ("folds", "seed", "min_df", "epochs", "max_epochs", "patience",
                   "hidden_dim", "hidden_layers", "batch_size")
    float_options = ("nb_alpha", "regularisation_coefficient", "tol", "dropout_rate",
                     "learning_rate", "weight_decay", "validation_fraction")
    for kind, names in ((str, string_options), (int, int_options), (float, float_options),
                        (parse_value, ("ngram_range", "split_seeds", "regularisation"))):
        for name in names:
            flags = list(dict.fromkeys(["--" + name.replace("_", "-"), "--" + name]))
            parser.add_argument(*flags, dest=name, type=kind, default=argparse.SUPPRESS)
    for name in ("save_models", "sublinear_tf"):
        parser.add_argument(
            "--" + name.replace("_", "-"), "--" + name, dest=name,
            type=parse_bool, nargs="?", const=True, default=argparse.SUPPRESS,
        )


def run_sweep(args):
    import yaml

    with args.config.open() as file:
        sweep = yaml.safe_load(file)
    program = Path(sweep.get("program", "")).name
    model = args.model
    if model is None:
        model = {"train_mlp.py": "mlp", "train_loglin.py": "lr"}.get(program)
    if model is None:
        raise ValueError("Specify --model lr or mlp for this sweep.")
    if args.local:
        configurations = list(grid(sweep["parameters"]))
        if sweep.get("method", "grid") == "random":
            random.Random(0).shuffle(configurations)
        texts, labels = load_train_split()
        for config in configurations[:args.count]:
            run_experiment({**config, "model": model}, texts, labels, output=args.output)
        return

    import sys
    import wandb

    # Existing YAMLs name the old scripts. New agents run this file with the active interpreter.
    sweep["program"] = str(Path(__file__).resolve())
    sweep["command"] = [
        "${env}", sys.executable, "${program}", "run", "--model", model,
        "--output", str(args.output), "${args}",
    ]
    sweep.setdefault("method", "grid")
    sweep_id = wandb.sweep(sweep, entity=args.entity, project=args.project)
    wandb.agent(sweep_id, entity=args.entity, project=args.project, count=args.count)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="cross-validate one configuration")
    add_training_arguments(run)
    run.add_argument("--output", type=Path, default=RESULTS)

    sweep = commands.add_parser("sweep", help="use an existing grid/random sweep YAML")
    sweep.add_argument("config", type=Path)
    sweep.add_argument("--model", choices=("lr", "mlp"))
    sweep.add_argument("--local", action="store_true", help="run here without W&B")
    sweep.add_argument("--count", type=int)
    sweep.add_argument("--project", default="project-1")
    sweep.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    sweep.add_argument("--output", type=Path, default=RESULTS)

    data = commands.add_parser("data", help="describe the IMDb split")
    data.add_argument("--split", choices=("train", "test"), default="train")
    compare = commands.add_parser("compare", help="compare saved out-of-fold predictions")
    compare.add_argument("paths", nargs="+", type=Path)
    compare.add_argument("--rope", type=float, default=0.005)
    compare.add_argument("--output", type=Path, default=RESULTS / "comparisons.csv")
    rule = commands.add_parser("rule", help="score the rating rule on saved predictions")
    rule.add_argument("paths", nargs="+", type=Path)
    rule.add_argument("--rope", type=float, default=0.005)
    args = parser.parse_args()

    if args.command == "data":
        describe_data(args.split)
    elif args.command == "compare":
        compare_results(args.paths, args.rope, args.output)
    elif args.command == "rule":
        texts, labels = load_train_split()
        score_rating_rule(args.paths, texts, labels, args.rope)
    elif args.command == "sweep":
        if args.count is not None and args.count < 1:
            parser.error("--count must be positive.")
        run_sweep(args)
    else:
        config = {key: value for key, value in vars(args).items() if key not in ("command", "output")}
        if "WANDB_SWEEP_ID" in os.environ:
            import wandb

            with wandb.init() as active_run:
                config.update(dict(active_run.config))
                texts, labels = load_train_split()
                run_experiment(config, texts, labels, output=args.output, wandb_run=active_run)
        else:
            config = prepare_config(config)
            texts, labels = load_train_split()
            run_experiment(config, texts, labels, output=args.output)


if __name__ == "__main__":
    main()
