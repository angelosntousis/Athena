"""Convert reviews into numerical features using a supplied trained tokenizer.

Representations:
- TokenIndicator: binary presence (0 or 1).
- BagOfWords: raw occurrence counts.
- TfidfEncoder: TF-IDF weighting with L2 normalisation.

ngram_range: (1, 1) = unigrams, (2, 2) = bigrams, (1, 2) = both.
With BPE, tokens may be word pieces rather than complete words.
min_df filters bigrams by training-document frequency; it does not filter unigrams.
sublinear_tf=True enables logarithmic counts for TF-IDF.

Example:
    from part1.features import TfidfEncoder

    encoder = TfidfEncoder(
        tokenizer, ngram_range=(2, 2), min_df=2, sublinear_tf=True
    )
    X_train = encoder.fit_transform(train_texts)
    X_test = encoder.transform(test_texts)

Train the tokenizer and fit the encoder on training text only.
Keep the tokenizer unchanged; use transform() for validation/test.
Repeat this process separately within each cross-validation fold.

Batch output is a sparse CSR matrix: rows are reviews, columns are features.
encoder(text) returns one dense vector for tutorial compatibility.
Bigram fitting fails if no pairs survive min_df.
"""

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils.validation import check_is_fitted
from tokenizers import Tokenizer

from .clslib import Encoder


class BagOfWords(Encoder):
    """Raw occurrence counts, without length normalisation.

    tokenizer must provide encode(text).ids, get_vocab_size(), and id_to_token().
    min_df applies only to learned bigrams and counts training documents, not
    occurrences. Unigram coordinates always follow the supplied token IDs.
    fit is required for bigrams; for unigrams it is a no-op because the supplied
    tokeniser already defines the vocabulary. Call fit consistently for either.
    """

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
    """For each selected unigram/bigram: 1 if present, otherwise 0."""

    def transform(self, texts) -> csr_matrix:
        counts = super().transform(texts)
        assert counts.data is not None
        counts.data[:] = 1.0
        return counts


class TfidfEncoder(BagOfWords):
    """Counts times smoothed IDF, followed by per-document L2 normalisation.

    IDF(t) = log((1 + number_of_training_documents) / (1 + document_frequency(t))) + 1

    In combined mode, weight and normalise the whole joint unigram/bigram vector.
    No labels are required. fit must receive only training text. sublinear_tf=True
    optionally replaces nonzero counts with 1 + log(count), before IDF weighting.
    """

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


class BigramCounts(Encoder):
    """
    min_df=2 discards pairs occurring in fewer than two training documents.
    Reversed pairs are distinct. Pairs never cross document boundaries, and
    unseen evaluation pairs are ignored without changing the vocabulary.
    """

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

