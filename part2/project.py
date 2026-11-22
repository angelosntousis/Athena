"""Train embed-and-pool and LSTM classifiers on the Part 1 IMDb folds.

Run from the repository root with python -m part2.project.
"""

import argparse
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from part1.project import cross_validation_split
from part1.train_loglin import load_train_split
from part2 import experiment
from part2.clslib import PretrainedEmbeddings
from part2.task1_choose_embeddings.embedding_check import ImdbTokenizer
from part2.task3_embed_and_pool.embed_and_pool import random_embedding_table
from part2.task4_recurrent_encoder.lstm import half_training_indices


TRAINING_DEFAULTS = dict(classifier_dropout=0.1, learning_rate=0.001, batch_size=64)
LSTM_DEFAULTS = dict(hidden_size=20, num_layers=1, lstm_dropout=0.0)


class Fold(NamedTuple):
    fit: np.ndarray
    validation: np.ndarray
    evaluation: np.ndarray


def prepare_folds(n_reviews, split_seed=3):
    """Return Part 1 row indices for fitting, choosing an epoch and scoring."""
    folds = {}
    for fold, training, evaluation in cross_validation_split(n_reviews, 5, split_seed):
        # Part 1 resets the inner seed to 0 for each fold.
        shuffled = np.random.default_rng(0).permutation(training)
        n_validation = round(0.1 * len(training))
        folds[fold] = Fold(shuffled[n_validation:], shuffled[:n_validation], evaluation)
    return folds


def check_baselines(labels, folds, split_seed):
    """Match each saved prediction to its original row, fold and sentiment label."""
    expected_fold = np.full(len(labels), -1)
    all_rows = np.arange(len(labels))
    for number, fold in folds.items():
        np.testing.assert_array_equal(np.sort(np.concatenate(fold)), all_rows)
        expected_fold[fold.evaluation] = number
    if (expected_fold < 0).any() or sum(len(f.evaluation) for f in folds.values()) != len(labels):
        raise ValueError("Every review must appear in exactly one evaluation fold.")

    folder = Path(__file__).resolve().parents[1] / "results" / "final_predictions"
    with (folder / "index.csv").open() as index:
        files = [row["file"] for row in csv.DictReader(index) if int(row["split_seed"]) == split_seed]
    if not files:
        raise ValueError(f"No saved Part 1 predictions for seed {split_seed}.")
    for filename in files:
        with np.load(folder / filename) as saved:
            ids = saved["row_id"]
            np.testing.assert_array_equal(np.sort(ids), all_rows, err_msg=filename)
            np.testing.assert_array_equal(saved["split_seed"], np.full_like(ids, split_seed), err_msg=filename)
            np.testing.assert_array_equal(saved["fold"], expected_fold[ids], err_msg=filename)
            np.testing.assert_array_equal(saved["label"], labels[ids], err_msg=filename)
    return len(files)


def train_fold(texts, labels, fold, *, encoder, embeddings, tokenizer,
               freeze_embeddings=True, epochs=20, device="cpu", history_path=None, settings=None):
    """Fit one tutorial encoder and return outer-fold scores and review predictions."""
    if encoder not in ("embed-and-pool", "lstm") or epochs < 1:
        raise ValueError("Choose embed-and-pool or lstm and at least one epoch.")
    torch.manual_seed(0)
    if settings is None:
        settings = {**TRAINING_DEFAULTS, **LSTM_DEFAULTS}
    enc_kwargs = {"freeze_embeddings": freeze_embeddings}
    if encoder == "lstm":
        enc_kwargs.update(hidden_size=settings["hidden_size"], num_layers=settings["num_layers"],
                          lstm_dropout=settings.get("lstm_dropout", 0.0))

    with TemporaryDirectory() as tmp:
        checkpoint = str(Path(tmp) / "model")
        start = perf_counter()
        history, best_epoch = experiment.train(
            [(texts[i], labels[i]) for i in fold.fit],
            [(texts[i], labels[i]) for i in fold.validation],
            tokenizer, embeddings, num_classes=2, enc_type=encoder, device=device,
            enc_kwargs=enc_kwargs, save_to=checkpoint,
            regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50,
                              "dropout": settings["classifier_dropout"]},
            opt_kwargs={"learning_rate": settings["learning_rate"], "weight_decay": 1e-4,
                        "batch_size": settings["batch_size"], "num_epochs": epochs},
        )
        train_seconds = perf_counter() - start
        if history_path is not None:
            with history_path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"])
                for epoch, metrics in enumerate(history, start=1):
                    train, val = metrics["train_metrics"], metrics["val_metrics"]
                    writer.writerow([epoch, train["loss"], train["accuracy"], val["loss"], val["accuracy"]])
        model = experiment.load(checkpoint, embeddings, device=device)
        log_probs = experiment.predict_logprobs(
            model, tokenizer, [texts[i] for i in fold.evaluation], batch_size=64, device=device,
        )

    gold = labels[fold.evaluation]
    predicted = log_probs.argmax(axis=1)
    scores = {
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": f1_score(gold, predicted, average="macro"),
        "best_epoch": best_epoch + 1,
        "train_seconds": train_seconds,
    }
    predictions = {"row_id": fold.evaluation, "label": gold, "prediction": predicted,
                   "prob_positive": np.exp(log_probs[:, 1])}
    return scores, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    train = commands.add_parser("train", help="Train and log one five-fold IMDb configuration.")
    train.add_argument("--encoder", choices=("embed-and-pool", "lstm"), required=True)
    train.add_argument("--embedding", choices=("glove", "fasttext", "word2vec"), required=True)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train.add_argument("--weights", choices=("frozen", "fine-tuned", "random-trainable"), default="frozen")
    train.add_argument("--half-data", action="store_true", help="Fit on a stratified half of each fitting set.")
    args = parser.parse_args()
    if args.command == "train" and args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Check the GPU allocation and the PyTorch installation.")
    split_seed = 3
    texts, labels = load_train_split()
    folds = prepare_folds(len(texts), split_seed)
    checked = check_baselines(labels, folds, split_seed)
    if args.command == "train" and args.half_data:
        for number, fold in folds.items():
            folds[number] = fold._replace(fit=half_training_indices(fold.fit, labels))
    print("fold   fit    validation   evaluation")
    for number, fold in folds.items():
        print(f"{number:>4} {len(fold.fit):>6} {len(fold.validation):>13} {len(fold.evaluation):>12}")
    print(f"Verified {len(texts):,} IMDb reviews against {checked} Part 1 baselines (seed {split_seed}).")
    if args.command == "train":
        run_training(texts, labels, folds, split_seed, args.encoder, args.embedding, args.device,
                     weights=args.weights, half_data=args.half_data)


def run_training(texts, labels, folds, split_seed, encoder, embedding, device, *, weights="frozen", half_data=False):
    import wandb

    root = Path(__file__).resolve().parents[1]
    vectors = PretrainedEmbeddings(str(root / "embeddings" / f"imdb_{embedding}.txt.gz"), 300)
    if weights == "random-trainable":
        vectors.vectors = random_embedding_table(vectors.vectors.shape, vectors.vocab.index(vectors.pad_token))
    tokenizer = ImdbTokenizer(vectors, embedding)
    config = dict(encoder=encoder, embedding=embedding, split_seed=split_seed, folds=len(folds),
                  weights=weights, freeze_embeddings=(weights == "frozen"), fit_fraction=0.5 if half_data else 1.0,
                  epochs=20, device=device, dataset="stanfordnlp/imdb", **TRAINING_DEFAULTS)
    if encoder == "lstm":
        config.update(LSTM_DEFAULTS)
    name = f"{encoder}-{embedding}"
    if weights != "frozen":
        name += f"-{weights}"
    if half_data:
        name += "-half-data"
    run_name = None if os.environ.get("WANDB_SWEEP_ID") else name
    with wandb.init(entity="nlp1-athena", project="Part 2", name=run_name, config=config) as run:
        output = root / "results" / "part2" / f"{name}-{run.id}"
        output.mkdir(parents=True, exist_ok=True)
        all_predictions = np.empty_like(labels)
        for number, fold in folds.items():
            print(f"{name} | fold {number + 1}/{len(folds)}", flush=True)
            scores, predictions = train_fold(texts, labels, fold, encoder=encoder, embeddings=vectors,
                                             tokenizer=tokenizer, device=device, epochs=config["epochs"],
                                             freeze_embeddings=config["freeze_embeddings"],
                                             settings=run.config,
                                             history_path=output / f"epochs-fold-{number}.csv")
            np.savez_compressed(output / f"fold-{number}.npz", **predictions)
            all_predictions[fold.evaluation] = predictions["prediction"]
            run.log({"fold": number, "fold/fit_reviews": len(fold.fit),
                     **{f"fold/{key}": value for key, value in scores.items()}})
        run.summary["accuracy"] = accuracy_score(labels, all_predictions)
        run.summary["macro_f1"] = f1_score(labels, all_predictions, average="macro")


if __name__ == "__main__":
    main()
