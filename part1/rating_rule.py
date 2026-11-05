"""
Ratings out of 10 written in a review ("7/10", "3 out of 10"), used directly as the prediction.

IMDb labels come from the reviewers' ratings: 4 or less out of 10 is negative, 7 or more positive
(reviews rated 5 or 6 are not in the dataset). A rating written in the text is therefore nearly a copy
of the label. The rule uses exactly these thresholds, fixed by the dataset's definition, so nothing is
learned from the data and it can be applied to held-out predictions without any leakage.

    rating = parse_rating(text)            # 7.0, or None if no single rating is found
    prediction = rule_prediction(rating)   # 1, 0, or None (rating 5-6, or no rating)

Usage, to score saved runs with and without the rule:

    python part1/rating_rule.py results/loglin/runs/<run> [results/loglin/runs/<other run> ...]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# a number, then "/10" or "out of 10"; not part of a date or fraction such as 10/10/2005 or 3/100
RATING_PATTERN = re.compile(r"(?<![\d/.])(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10(?![\d/])", re.IGNORECASE)
POSITIVE_FROM = 7  # IMDb: ratings of 7 or more are positive
NEGATIVE_TO = 4    # IMDb: ratings of 4 or less are negative


def parse_rating(text):
    """The rating out of 10 stated in the text, or None if there is none or several different ones."""
    ratings = {float(value) for value in RATING_PATTERN.findall(text) if float(value) <= 10}
    return ratings.pop() if len(ratings) == 1 else None


def rule_prediction(rating):
    """1 (positive), 0 (negative), or None when the rating does not decide (5 to 6, or no rating)."""
    if rating is None:
        return None
    if rating >= POSITIVE_FROM:
        return 1
    if rating <= NEGATIVE_TO:
        return 0
    return None


def apply_rule(texts, predictions):
    """
    The model's predictions, replaced by the rating rule wherever it decides.
    Returns the new predictions and a mask of where the rule was used.
    """
    rule = [rule_prediction(parse_rating(text)) for text in texts]
    used = np.array([r is not None for r in rule])
    combined = np.array([r if r is not None else p for r, p in zip(rule, predictions)])
    return combined, used


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run folders containing predictions.npz")
    parser.add_argument("--rope", type=float, default=0.005, help="region of practical equivalence, in accuracy")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from part1.analysis import BenavoliCorrelatedT
    from part1.train_loglin import load_train_split

    texts, labels = load_train_split()
    rule = np.array([-1 if (r := rule_prediction(parse_rating(t))) is None else r for t in texts])
    print(f"The rule decides {np.sum(rule >= 0)} of {len(texts)} reviews ({np.mean(rule >= 0):.1%}); "
          f"on those, it alone is right for {np.mean(rule[rule >= 0] == labels[rule >= 0]):.2%}")

    for folder in args.runs:
        p = np.load(Path(folder) / "predictions.npz")
        config = json.loads((Path(folder) / "config.json").read_text())
        rule_here = rule[p["row_id"]]
        combined = np.where(rule_here >= 0, rule_here, p["prediction"])
        folds = sorted({(s, f) for s, f in zip(p["split_seed"], p["fold"])})  # seeds, then folds, as in training
        model_acc, combined_acc = [], []
        for s, f in folds:
            sel = (p["split_seed"] == s) & (p["fold"] == f)
            model_acc.append(np.mean(p["prediction"][sel] == p["label"][sel]))
            combined_acc.append(np.mean(combined[sel] == p["label"][sel]))
        model_acc, combined_acc = np.array(model_acc), np.array(combined_acc)
        diff = combined_acc - model_acc
        runs = len(set(p["split_seed"]))
        test = BenavoliCorrelatedT(model_acc, combined_acc, runs=runs, names=["model", "model + rule"])
        p_model, p_tie, p_rule = test.probs(args.rope)
        print(f"\n{folder}: {config['representation']}, ngram_range {config.get('ngram_range')}, "
              f"{len(folds)} folds ({runs} split seed{'s' if runs > 1 else ''})")
        print(f"  model          {model_acc.mean():.4f}")
        print(f"  model + rule   {combined_acc.mean():.4f}")
        print(f"  difference     {100 * diff.mean():+.2f} pp (per fold {100 * diff.min():+.2f} to {100 * diff.max():+.2f}; "
              f"rule better on {np.sum(diff > 0)}/{len(diff)} folds, worse on {np.sum(diff < 0)})")
        print(f"  changed predictions: {np.sum(combined != p['prediction'])} "
              f"({np.sum((combined != p['prediction']) & (combined == p['label']))} fixed, "
              f"{np.sum((combined != p['prediction']) & (p['prediction'] == p['label']))} broken)")
        print(f"  Bayesian correlated t-test, ROPE ±{100 * args.rope:.1f} pp: "
              f"P(rule better) {p_rule:.3f}, P(practically equivalent) {p_tie:.3f}, P(model better) {p_model:.3f}")


if __name__ == "__main__":
    main()
