"""
Task 6: describe every IMDb training review with the features the BDA tests.

Two steps, run from the repository root (Athena/):

    python -m part2.task6_error_analysis.features text        # needs only the reviews
    python -m part2.task6_error_analysis.features embedding   # needs embeddings/imdb_<model>.txt.gz

`text` writes text_features.csv.gz and `embedding` writes embedding_features.csv.gz, both next to
this file, with one row per review (row_id = position in the IMDb train split).

Text features
    label              1 = positive, 0 = negative
    log_length         log of the number of words (tokens with a letter)
    negation_rate      negation words per 100 words
    contrast_rate      contrast words per 100 words
    hedge_rate         hedging words per 100 words
    rare_word_rate     share of content words seen in fewer than RARE_DOCS of the reviews the
                       model trained on (the 18,000 fitting reviews of the review's own fold)

Embedding features (one column per embedding)
    sentiment_strength_<model>   |mean over content words of cos(word, sentiment direction)|, where
                                 the direction is mean(positive lexicon words) - mean(negative ones)
                                 and word scores are centred on the IMDb average
    mismatch_<model>             mean over content words of 1 - (shared nearest neighbours / K),
                                 comparing <model> with fastText (glove and word2vec only)

Word lists are rates, not counts, so they are not just another measure of length.
Reviews are split exactly as in part 1 and tasks 3-4 (part1 cleanup and word splitting).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from part2.task1_choose_embeddings.embedding_check import (  # noqa: E402
    DIMS, EMB_DIR, adapt_tokens, imdb_splitter, token_kind,
)

HERE = Path(__file__).resolve().parent
TEXT_FEATURES = HERE / "text_features.csv.gz"
EMBEDDING_FEATURES = HERE / "embedding_features.csv.gz"

NEGATIONS = {"not", "no", "never", "n't", "cannot", "nothing", "nobody", "none", "neither", "nor", "without"}
CONTRASTS = {"but", "however", "although", "though", "yet", "despite", "nevertheless"}
HEDGES = {"somewhat", "bit", "almost", "rather", "fairly", "slightly", "kinda"}

RARE_DOCS = 20          # a word is rare if it occurs in fewer fitting reviews than this
COMMON_COUNT = 20       # words seen this often in IMDb train get nearest neighbours
NEIGHBOURS = 10         # K, the neighbours compared between two embeddings
EMBEDDINGS = ("glove", "word2vec", "fasttext")
REFERENCE = "fasttext"  # meaning mismatch is measured against this embedding


def load_reviews():
    """IMDb train texts and labels, in dataset order, and each review split as in part 1."""
    from datasets import load_dataset
    data = load_dataset("stanfordnlp/imdb", split="train")
    split = imdb_splitter()
    return [split(text) for text in data["text"]], np.array(data["label"])


def stop_words():
    from nltk.corpus import stopwords
    return set(stopwords.words("english")) | NEGATIONS | CONTRASTS | HEDGES


def content_words(tokens, stop):
    """Tokens with a letter that are not stop words or contraction pieces (n't, 's, ...)."""
    return [t for t in tokens if token_kind(t) == "word" and t not in stop and "'" not in t]


# ---------------------------------------------------------------- text features

def text_features(reviews, labels):
    from part2.project import prepare_folds
    stop = stop_words()
    content = [content_words(r, stop) for r in reviews]

    rare_rate = np.full(len(reviews), np.nan)
    for fold in prepare_folds(len(reviews)).values():
        docs = Counter(w for i in fold.fit for w in set(content[i]))
        for i in fold.evaluation:
            if content[i]:
                rare_rate[i] = np.mean([docs[w] < RARE_DOCS for w in content[i]])

    rows = []
    for i, tokens in enumerate(reviews):
        words = [t for t in tokens if token_kind(t) == "word"]
        n = max(len(words), 1)
        rows.append({
            "row_id": i,
            "label": int(labels[i]),
            "log_length": np.log(n),
            "negation_rate": 100 * sum(t in NEGATIONS for t in words) / n,
            "contrast_rate": 100 * sum(t in CONTRASTS for t in words) / n,
            "hedge_rate": 100 * sum(t in HEDGES for t in words) / n,
            "rare_word_rate": rare_rate[i],
        })
    table = pd.DataFrame(rows)
    table["rare_word_rate"] = table["rare_word_rate"].fillna(0.0)
    return table


# ---------------------------------------------------------------- embedding features

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_embeddings(folder):
    from part2.clslib import PretrainedEmbeddings
    return {m: PretrainedEmbeddings(str(folder / f"imdb_{m}.txt.gz"), DIMS) for m in EMBEDDINGS}


def vectors(model, words):
    return np.stack([model.vectors[model.vocab.index(w)] for w in words]).astype(np.float32)


def sentiment_scores(models, reviews, stop):
    """Signed and absolute mean sentiment of each review, as read by each embedding."""
    from nltk.corpus import opinion_lexicon
    shared = lambda words: sorted(w for w in words if all(w in m.vocab for m in models.values()))
    positive, negative = shared(opinion_lexicon.positive()), shared(opinion_lexicon.negative())
    print(f"Sentiment direction from {len(positive):,} positive and {len(negative):,} negative lexicon words")

    columns = {}
    for name, model in models.items():
        direction = unit(unit(vectors(model, positive)).mean(0) - unit(vectors(model, negative)).mean(0))
        words = [[w for w in content_words(adapt_tokens(r, name), stop) if w in model.vocab] for r in reviews]
        vocab = sorted({w for ws in words for w in ws})
        score = dict(zip(vocab, unit(vectors(model, vocab)) @ direction))
        centre = np.mean([score[w] for ws in words for w in ws])  # the average IMDb word is neutral
        signed = np.array([np.mean([score[w] for w in ws]) - centre if ws else 0.0 for ws in words])
        columns[f"sentiment_score_{name}"] = signed
        columns[f"sentiment_strength_{name}"] = np.abs(signed)
    return columns


def nearest_neighbours(model, words, k, batch=2048):
    """For each word, the set of indices (into words) of its k closest other words by cosine."""
    x = unit(vectors(model, words))
    result = []
    for start in range(0, len(words), batch):
        sim = x[start:start + batch] @ x.T
        sim[np.arange(sim.shape[0]), np.arange(start, start + sim.shape[0])] = -np.inf  # not itself
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        result.extend(set(row) for row in top)
    return result


def mismatch_scores(models, reviews, stop):
    """Mean share of each review's common content words whose neighbours differ from fastText's."""
    counts = Counter(w for r in reviews for w in content_words(r, stop))
    common = sorted(w for w, c in counts.items() if c >= COMMON_COUNT and all(w in m.vocab for m in models.values()))
    print(f"Nearest neighbours among {len(common):,} content words seen >= {COMMON_COUNT} times and known to all models")
    index = {w: i for i, w in enumerate(common)}
    reference = nearest_neighbours(models[REFERENCE], common, NEIGHBOURS)

    columns = {}
    for name in EMBEDDINGS:
        if name == REFERENCE:
            continue
        own = nearest_neighbours(models[name], common, NEIGHBOURS)
        word_mismatch = np.array([1 - len(a & b) / NEIGHBOURS for a, b in zip(own, reference)])
        print(f"  {name} vs {REFERENCE}: mean word mismatch {word_mismatch.mean():.2f}")
        columns[f"mismatch_{name}"] = np.array([
            np.mean([word_mismatch[index[w]] for w in ws]) if ws else 0.0
            for ws in ([w for w in content_words(r, stop) if w in index] for r in reviews)
        ])
    return columns


def embedding_features(reviews, folder):
    models = load_embeddings(folder)
    stop = stop_words()
    columns = {"row_id": np.arange(len(reviews))}
    columns.update(sentiment_scores(models, reviews, stop))
    columns.update(mismatch_scores(models, reviews, stop))
    return pd.DataFrame(columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["text", "embedding"])
    parser.add_argument("--embeddings-dir", type=Path, default=EMB_DIR,
                        help="Folder with imdb_<model>.txt.gz (default: embeddings/ in the repository)")
    args = parser.parse_args()

    reviews, labels = load_reviews()
    if args.step == "text":
        table, out = text_features(reviews, labels), TEXT_FEATURES
    else:
        table, out = embedding_features(reviews, args.embeddings_dir), EMBEDDING_FEATURES
    table.to_csv(out, index=False, float_format="%.6g")
    print(f"Wrote {len(table):,} reviews x {table.shape[1] - 1} features to {out}")
    print(table.drop(columns="row_id").describe().T[["mean", "std", "min", "max"]].round(3).to_string())
