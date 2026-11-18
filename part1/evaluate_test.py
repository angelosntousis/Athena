"""
Final evaluation of one MLP configuration on the IMDb `test` split.

The model is trained as in every cross-validation fold of train_mlp.py, but on the whole `train` split:
10% of it (validation_fraction) is held out to choose the epoch, the representation is fitted on the rest,
and the best epoch's weights are scored once on the 25,000 test reviews.

Usage (config keys and defaults as in train_mlp.py):

    python part1/evaluate_test.py --representation=tf-idf-bigrams-nb --tokenizer=bpe32k \
        --hidden_layers=2 --hidden_dim=256 --dropout_rate=0.1 --learning_rate=1e-4 --weight_decay=1e-4

Writes results/test/{name}/: config.json, metrics.json, predictions.npz (row_id, label, prediction,
prob_positive), the model (model.pt, model.json) and the learned features (features.pickle).
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so that `part1` can be imported when this file is run as a script

from part1 import smlp  # noqa: E402
from part1.features import NaiveBayesWeighting  # noqa: E402
from part1.tokenisation import get_tokenizer  # noqa: E402
from part1.train_loglin import load_train_split, numbers, parse_tokenizer, split_overrides  # noqa: E402
from part1.train_mlp import DEFAULTS, RESULTS, feature_state, make_encoder, set_seed  # noqa: E402


def main():
    _, overrides = split_overrides(sys.argv[1:], set())
    config = {**DEFAULTS, **{key: numbers(value) for key, value in overrides.items()}}
    name = (f"mlp-{config['hidden_layers']}x{config['hidden_dim']}_{config['representation']}_{config['tokenizer']}"
            f"_seed-{config['seed']}")
    folder = RESULTS / "test" / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {config}")

    texts, labels = load_train_split()
    test = load_dataset("stanfordnlp/imdb", split="test")
    test_texts, test_labels = list(test["text"]), np.array(test["label"])
    kind, vocab_size = parse_tokenizer(config["tokenizer"])
    tokenizer = get_tokenizer(kind, vocab_size, folder=RESULTS / "tokenizers")

    # as in a cross-validation fold: a validation part, taken from the training data, for choosing the epoch
    shuffled = np.random.default_rng(config["seed"]).permutation(len(texts))
    num_validation = round(config["validation_fraction"] * len(texts))
    validation_ids, fit_ids = shuffled[:num_validation], shuffled[num_validation:]

    set_seed(config["seed"])
    start = time.perf_counter()
    encoder = make_encoder(config, tokenizer)
    fit_texts = [texts[i] for i in fit_ids]
    if isinstance(encoder, NaiveBayesWeighting):
        encoder.fit(fit_texts, labels[fit_ids])
    else:
        encoder.fit(fit_texts)
    metrics, best_epoch, _ = smlp.train_neural_network_model(
        [(texts[i], labels[i]) for i in fit_ids],
        [(texts[i], labels[i]) for i in validation_ids],
        num_classes=len(np.unique(labels)),
        encoder=encoder,
        save_to=str(folder / "model"),
        hidden_dim=config["hidden_dim"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout_rate"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        num_epochs=config["max_epochs"],
        patience=config["patience"],
        device=config["device"],
    )
    train_time = time.perf_counter() - start
    model = smlp.load(str(folder / "model"), device=config["device"])  # the best epoch's weights

    probs = smlp.predict_probs(test_texts, model, encoder, batch_size=256)
    predictions = probs.argmax(axis=1)
    accuracy = accuracy_score(test_labels, predictions)
    # 95% interval for the accuracy, from the binomial standard error over the test reviews
    half_width = 1.96 * np.sqrt(accuracy * (1 - accuracy) / len(test_labels))
    result = dict(
        test_accuracy=float(accuracy),
        test_accuracy_95ci=[float(accuracy - half_width), float(accuracy + half_width)],
        test_macro_f1=float(f1_score(test_labels, predictions, average="macro")),
        test_errors=int(np.sum(predictions != test_labels)),
        false_positives=int(np.sum((predictions == 1) & (test_labels == 0))),
        false_negatives=int(np.sum((predictions == 0) & (test_labels == 1))),
        best_epoch=best_epoch + 1,  # counted from 1
        validation_accuracy=metrics[best_epoch]["val_metrics"]["accuracy"],
        train_accuracy=metrics[best_epoch]["train_metrics"]["accuracy"],
        train_time=train_time,
    )
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    (folder / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(folder / "predictions.npz", row_id=np.arange(len(test_labels)), label=test_labels,
                        prediction=predictions, prob_positive=probs[:, 1])
    with open(folder / "features.pickle", "wb") as f:
        pickle.dump(feature_state(encoder), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
