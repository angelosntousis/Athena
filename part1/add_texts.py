"""Adds the review text to results/test/<run>/test_errors.csv (the best model's misclassified test reviews).

Run from the repo root:  python part1/add_texts.py
Writes results/test/<run>/test_errors_with_text.csv, sorted from most to least confident mistake.
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

RUN = Path("results/test/mlp-2x256_tf-idf-bigrams-nb_bpe32k_seed-0")

errors = pd.read_csv(RUN / "test_errors.csv")
test = load_dataset("stanfordnlp/imdb", split="test")
errors["text"] = [test[int(i)]["text"] for i in errors["row_id"]]
errors.to_csv(RUN / "test_errors_with_text.csv", index=False)
print(f"{len(errors)} errors -> {RUN / 'test_errors_with_text.csv'}")
