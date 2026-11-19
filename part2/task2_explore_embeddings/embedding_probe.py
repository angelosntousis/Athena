"""
Task 2: what do the three pretrained embedding models capture that matters for IMDb sentiment?

Uses the IMDb-vocabulary vectors written by `python part2/task1_choose_embeddings/embedding_check.py export`
(embeddings/imdb_<model>.txt.gz), loaded with the tutorial's part2.clslib.PretrainedEmbeddings.

The two tests on the Task 2 slide (run from the repository root, Athena/):

    python part2/task2_explore_embeddings/embedding_probe.py neighbours   # 1. closest words to review words
    python part2/task2_explore_embeddings/embedding_probe.py blindspots   # 2. opposites and negation

Both compare the models on the same words: those seen >= 20 times in IMDb train and known to all
three models.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # the repository root (Athena/)
sys.path.insert(0, str(ROOT))

from part2.clslib import PretrainedEmbeddings  # noqa: E402
from part2.task1_choose_embeddings.embedding_check import DIMS, EMB_DIR, MODELS, imdb_splitter, token_kind  # noqa: E402

SEED = 0


# ---------------------------------------------------------------- data

def load_models(names=tuple(MODELS)):
    return {n: PretrainedEmbeddings(str(EMB_DIR / f"imdb_{n}.txt.gz"), DIMS) for n in names}


def common_words(models, min_count=20):
    """Words (with a letter) seen >= min_count times in IMDb train (split as in part 1) and known to every model."""
    from datasets import load_dataset
    split = imdb_splitter()
    counts = Counter(token for text in load_dataset("stanfordnlp/imdb", split="train")["text"] for token in split(text))
    return sorted(w for w, c in counts.items()
                  if c >= min_count and token_kind(w) == "word" and all(w in m.vocab for m in models.values()))


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cos(u, v):
    return float(unit(u) @ unit(v))


# ---------------------------------------------------------------- 1. closest words to review words

REVIEW_WORDS = ["plot", "acting", "waste", "lame", "sick", "cheesy", "flick", "overrated", "underrated"]


def neighbours(models, words, k=5):
    """
    The k closest words (by cosine) to common review words, among the words all models know.
    Shows which meaning of a review word each model has learned.
    """
    print(f"Test 1: closest words to review words (among {len(words):,} common IMDb words)")
    for n, m in models.items():
        x = unit(np.stack([m.embed_word(w) for w in words]))
        print(f"  [{n}]")
        for w in REVIEW_WORDS:
            sim = x @ unit(m.embed_word(w))
            top = [words[i] for i in np.argsort(-sim) if words[i] != w][:k]
            print(f"    {w:11} {', '.join(top)}")


# ---------------------------------------------------------------- 2. blind spots: opposites and negation

OPPOSITES = [("good", "bad"), ("great", "terrible"), ("best", "worst"), ("love", "hate"),
             ("beautiful", "ugly"), ("interesting", "boring"), ("brilliant", "awful"),
             ("excellent", "poor"), ("entertaining", "dull"), ("overrated", "underrated")]
SAME_MEANING = [("good", "great"), ("bad", "terrible"), ("best", "finest"), ("love", "adore"),
                ("beautiful", "gorgeous"), ("boring", "dull"), ("brilliant", "superb"),
                ("awful", "horrible"), ("funny", "hilarious"), ("excellent", "outstanding")]
NEGATED = [("good", "bad"), ("great", "terrible"), ("funny", "boring"), ("interesting", "boring"),
           ("brilliant", "awful"), ("excellent", "poor"), ("beautiful", "ugly"), ("entertaining", "dull")]


def blind_spots(models, words, samples=2000):
    """
    Opposites: mean cosine of opposite pairs vs same-meaning pairs vs random pairs of common words.
    Negation:  is the average of "not" and w closer to w or to its opposite?
               (what an embed-and-pool encoder sees for "not w")
    """
    rng = np.random.default_rng(SEED)
    pairs = [(i, j) for i, j in rng.choice(len(words), size=(samples, 2)) if i != j]
    print("\nTest 2a: opposites look alike (mean cosine; -1 opposite direction, 1 identical)")
    print(f"  {'':10}{'opposites':>11}{'same meaning':>14}{'random words':>14}   cos(overrated, underrated)")
    for n, m in models.items():
        opp = np.mean([cos(m.embed_word(a), m.embed_word(b)) for a, b in OPPOSITES])
        same = np.mean([cos(m.embed_word(a), m.embed_word(b)) for a, b in SAME_MEANING])
        rand = np.mean([cos(m.embed_word(words[i]), m.embed_word(words[j])) for i, j in pairs])
        print(f"  {n:10}{opp:>11.2f}{same:>14.2f}{rand:>14.2f}   "
              f"{cos(m.embed_word('overrated'), m.embed_word('underrated')):.2f}")
    print("  opposite pairs:", ", ".join(f"{a}/{b}" for a, b in OPPOSITES))

    print("\nTest 2b: averaging cannot express negation: cos(mean(not, w), w) vs cos(mean(not, w), opposite of w)")
    for n, m in models.items():
        closer_to_w, cells = 0, []
        for w, opposite in NEGATED:
            v = (m.embed_word("not") + m.embed_word(w)) / 2
            to_w, to_opposite = cos(v, m.embed_word(w)), cos(v, m.embed_word(opposite))
            closer_to_w += to_w > to_opposite
            cells.append(f"not {w}: {to_w:.2f} vs {to_opposite:.2f}")
        print(f"  {n:10}closer to w in {closer_to_w}/{len(NEGATED)} pairs   ({'; '.join(cells[:2])})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", nargs="?", choices=["neighbours", "blindspots", "all"], default="all")
    args = parser.parse_args()

    models = load_models()
    words = common_words(models)
    if args.test in ("neighbours", "all"):
        neighbours(models, words)
    if args.test in ("blindspots", "all"):
        blind_spots(models, words)
