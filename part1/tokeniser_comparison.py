"""
Task 2: word-level (A) vs BPE (B) tokenisers at four vocabulary sizes.

For each of the 8 tokenisers (part1/tokenisation.py, trained on the `unsupervised` split):
    - text statistics on the `train` split: [UNK] rate, share of reviews with an [UNK],
      tokens per word, share of words split into pieces
    - log-linear (liblinear) on unigram TF-IDF, 5-fold CV x 3 split seeds (the tutorial's folds,
      part1/experiment.py: cross_validation_split), C in {1, 4}; IDF is fitted on the training folds only
Then a Bayesian correlated t-test (Benavoli et al., 2017) of B - A at every size, with a ROPE.

Usage, from the repository root or from part1/:

    python part1/tokeniser_comparison.py              # train/load tokenisers, cross-validate, analyse
    python part1/tokeniser_comparison.py --analyse    # only analyse results/tokeniser_comparison.json

Tokenisers are saved in results/tokenizers (shared with train_loglin.py and train_mlp.py).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so that `part1` can be imported when this file is run as a script

from part1.tokenisation import VOCAB_SIZES, get_tokenizer  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "tokeniser_comparison.json"
SPLIT_SEEDS, K, CS = [0, 3, 5], 5, [1.0, 4.0]
ROPE = 0.5  # percentage points


def folds(n, seed):
    """The same folds as part1/experiment.py: cross_validation_split(pairs, k=K, seed=seed)."""
    random.seed(seed)
    ids = list(range(n))
    random.shuffle(ids)
    parts = np.array_split(np.array(ids), K)
    for i in range(K):
        yield np.concatenate([p for j, p in enumerate(parts) if j != i]), parts[i]


def encode(tokenizer, texts):
    """Token-count matrix (reviews x vocabulary) and text statistics."""
    vocab = tokenizer.get_vocab()
    unk = vocab["[UNK]"]
    continuation = np.zeros(tokenizer.get_vocab_size(), bool)  # pieces inside a word (##...)
    for token, i in vocab.items():
        continuation[i] = token.startswith("##")
    rows, cols = [], []
    n_tokens = n_unk = n_words = n_split = reviews_with_unk = 0
    for r, encoding in enumerate(tokenizer.encode_batch(texts)):
        ids = np.array(encoding.ids, dtype=np.int64)
        rows.append(np.full(len(ids), r))
        cols.append(ids)
        inside = continuation[ids]
        n_tokens += len(ids)
        n_words += int((~inside).sum())
        n_split += int((~inside[:-1] & inside[1:]).sum())  # a word start followed by a piece
        unk_here = int((ids == unk).sum())
        n_unk += unk_here
        reviews_with_unk += unk_here > 0
    X = sp.csr_matrix((np.ones(n_tokens), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(texts), tokenizer.get_vocab_size()))
    X.sum_duplicates()
    stats = dict(unk_rate=n_unk / n_tokens, reviews_with_unk=reviews_with_unk / len(texts),
                 tokens_per_word=n_tokens / n_words, words_split=n_split / n_words)
    return X, stats


def run():
    data = load_dataset("stanfordnlp/imdb", split="train")
    texts, labels = list(data["text"]), np.array(data["label"])
    results = {}
    for kind in ("word", "bpe"):
        for size in VOCAB_SIZES:
            start = time.time()
            X, stats = encode(get_tokenizer(kind, size, folder=RESULTS / "tokenizers"), texts)
            accuracies = {str(c): [] for c in CS}
            for seed in SPLIT_SEEDS:
                for train_ids, eval_ids in folds(len(texts), seed):
                    tfidf = TfidfTransformer().fit(X[train_ids])
                    X_train, X_eval = tfidf.transform(X[train_ids]), tfidf.transform(X[eval_ids])
                    for c in CS:
                        model = LogisticRegression(C=c, solver="liblinear", max_iter=1000).fit(X_train, labels[train_ids])
                        accuracies[str(c)].append(float((model.predict(X_eval) == labels[eval_ids]).mean()))
            name = f"{kind}_{size // 1000}k"
            results[name] = dict(kind=kind, size=size, **stats, accuracy=accuracies)
            print(name, {k: round(v, 4) for k, v in stats.items()},
                  {c: round(float(np.mean(a)), 4) for c, a in accuracies.items()}, f"{time.time() - start:.0f}s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    return results


def correlated_t(a, b, runs=len(SPLIT_SEEDS)):
    """Posterior of the mean difference b - a in percentage points (Benavoli et al., 2017)."""
    d = 100 * (np.array(b) - np.array(a))
    n, rho = len(d), 1 / (len(d) / runs)
    return st.t(n - 1, loc=d.mean(), scale=np.sqrt(d.var(ddof=1) * (1 / n + rho / (1 - rho))))


def analyse(results):
    sizes = sorted({r["size"] for r in results.values()})
    for c in map(str, CS):
        print(f"\nC = {c}: mean CV accuracy, and B - A with P(A better), P(equal), P(B better), ROPE ±{ROPE} pp")
        for size in sizes:
            a = results[f"word_{size // 1000}k"]["accuracy"][c]
            b = results[f"bpe_{size // 1000}k"]["accuracy"][c]
            t = correlated_t(a, b)
            p_a, p_b = t.cdf(-ROPE), 1 - t.cdf(ROPE)
            low, high = t.interval(0.95)
            print(f"  {size // 1000:>2}k  A {100 * np.mean(a):.2f}  B {100 * np.mean(b):.2f}  B-A {t.mean():+.2f} pp"
                  f"  [{low:+.2f}, {high:+.2f}]  {p_a:.3f} / {1 - p_a - p_b:.3f} / {p_b:.3f}")
    print("\nText statistics on `train`:")
    for name, r in results.items():
        print(f"  {name:>8}  [UNK] {100 * r['unk_rate']:.1f} %  reviews with [UNK] {100 * r['reviews_with_unk']:.0f} %"
              f"  tokens/word {r['tokens_per_word']:.3f}  words split {100 * r['words_split']:.1f} %")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyse", action="store_true", help=f"only analyse {OUT.relative_to(ROOT)}")
    args = parser.parse_args()
    results = json.loads(OUT.read_text()) if args.analyse else run()
    analyse(results)


if __name__ == "__main__":
    main()
