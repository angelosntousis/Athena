"""Prints 10 misclassified test reviews of the best model (MLP 2x256, tf-idf + bigrams + NB),
one negative and one positive review per confidence band. Run from the repo root: python part1/show_errors.py
"""
from datasets import load_dataset

# (row_id, true label, P(positive), confidence band); picked at random within each band from test_errors.csv
EXAMPLES = [
    (9115, 0, 0.9909, "> 0.99"), (20096, 1, 0.0067, "> 0.99"),
    (2924, 0, 0.9487, "0.90-0.99"), (18942, 1, 0.0273, "0.90-0.99"),
    (9207, 0, 0.8509, "0.75-0.90"), (18212, 1, 0.1047, "0.75-0.90"),
    (6975, 0, 0.7395, "0.60-0.75"), (22741, 1, 0.2551, "0.60-0.75"),
    (691, 0, 0.5807, "0.50-0.60"), (23144, 1, 0.4829, "0.50-0.60"),
]
NAMES = {0: "negative", 1: "positive"}

test = load_dataset("stanfordnlp/imdb", split="test")
for row_id, label, prob, band in EXAMPLES:
    assert test[row_id]["label"] == label, f"row {row_id}: label mismatch"
    predicted = 1 - label
    confidence = prob if predicted == 1 else 1 - prob
    print(f"\n=== row {row_id} | true {NAMES[label]}, predicted {NAMES[predicted]} "
          f"| confidence {confidence:.3f} (band {band})")
    print(test[row_id]["text"].replace("<br />", " "))
