"""
Q1: statistics of the three splits of stanfordnlp/imdb, as on the overview slide.

    sizes and label balance, review length in whitespace words (by class too),
    noise (<br /> tags, mis-decoded Windows-1252 characters, other HTML, links),
    ratings quoted in the text ("7/10", "3 out of 10") and how often they match the label,
    exact duplicate reviews within and across train and test

Usage, from the repository root or from part1/:

    python part1/dataset_stats.py
"""
import re

import numpy as np
from datasets import load_dataset

# a number, then "/10" or "out of 10"; not part of a date or fraction (same pattern as part1/rating_rule.py)
RATING = re.compile(r"(?<![\d/.])(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10(?![\d/])", re.IGNORECASE)


def quoted_rating(text):
    """The one rating out of 10 quoted in the text, or None if there is none or several different ones."""
    ratings = {float(value) for value in RATING.findall(text) if float(value) <= 10}
    return ratings.pop() if len(ratings) == 1 else None


def main():
    splits = {name: load_dataset("stanfordnlp/imdb", split=name) for name in ("train", "test", "unsupervised")}
    for name, data in splits.items():
        texts = data["text"]
        words = np.array([len(text.split()) for text in texts])
        print(f"\n{name}: {len(texts)} reviews, {words.sum():,} words")
        print(f"  words per review: median {np.median(words):.0f}, mean {words.mean():.1f}, "
              f"10th-90th percentile {np.percentile(words, 10):.0f}-{np.percentile(words, 90):.0f}, "
              f"range {words.min()}-{words.max()}")
        print(f"  <br /> {np.mean(['<br' in t for t in texts]):.1%} · "
              f"Windows-1252 characters {np.mean([any(c in t for c in chr(0x85) + chr(0x91) + chr(0x96) + chr(0x97)) for t in texts]):.1%} · "
              f"other HTML tags {np.mean([bool(re.search(r'<(?!br)[a-z]+', t, re.I)) for t in texts]):.2%} · "
              f"links {np.mean([('http' in t or 'www.' in t) for t in texts]):.2%}")
        if name == "unsupervised":
            continue
        labels = np.array(data["label"])
        print(f"  positive {labels.mean():.1%}; median words positive {np.median(words[labels == 1]):.0f}, "
              f"negative {np.median(words[labels == 0]):.0f}")
        ratings = [quoted_rating(t) for t in texts]
        decides = np.array([r is not None and (r >= 7 or r <= 4) for r in ratings])  # IMDb's label thresholds
        predicted = np.array([1 if r is not None and r >= 7 else 0 for r in ratings])
        print(f"  quotes one rating: {np.mean([r is not None for r in ratings]):.1%}; "
              f"a rating <= 4 or >= 7 matches the label for {np.mean(predicted[decides] == labels[decides]):.1%} "
              f"of the {decides.mean():.1%} it decides")

    train, test = splits["train"]["text"], splits["test"]["text"]
    print(f"\nexact duplicates: {len(train) - len(set(train))} within train, {len(test) - len(set(test))} within test, "
          f"{len(set(train) & set(test))} reviews in both")


if __name__ == "__main__":
    main()
