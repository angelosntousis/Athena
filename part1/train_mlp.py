"""
MLP on IMDb (part1/smlp.py), scored by k-fold cross-validation on the `train` split,
with the tutorial's folds (part1/experiment.py: cross_validation_split), the same as part1/train_loglin.py.

One run = one configuration: a tokeniser (part1/tokenisation.py), a representation (part1/features.py)
and the MLP's hyperparameters. In every fold, a validation part (validation_fraction) is set aside from the
training folds; the representation is fitted on the rest, and the epoch with the lowest validation loss is kept
(training stops after `patience` epochs without improvement). The evaluation fold is used only for scoring.
The mean accuracy over the folds is logged as `accuracy`, the metric the sweep maximises.

Usage, from the repository root or from part1/:

    python part1/train_mlp.py part1/mlp_sweep_config.yaml --count 20         # create a W&B sweep and run an agent
    python part1/train_mlp.py part1/mlp_sweep_config.yaml --local --count 5  # run configurations here, without W&B
    python part1/train_mlp.py --representation tf-idf --hidden_layers 2 --hidden_dim 128 --dropout_rate 0.5   # one run

For a sweep with method `random`, --local runs the grid in a random (but fixed) order, so --count picks a random
sample. A W&B random sweep never runs out of configurations: give --count, or stop it in W&B.

Config keys (missing ones take the values in DEFAULTS):
    tokenizer:                  "word" or "bpe", optionally with a vocabulary size: "bpe_16k", "bpe16k", "word_32k"
    representation:             "token_indicators", "bow", "bow_l1", "bow_l2" or "tf-idf", optionally followed by
                                "-bigrams" (unigrams and bigrams) and/or "-nb" (Naive Bayes weighting):
                                e.g. "tf-idf-bigrams-nb"
    min_df, sublinear_tf:       passed on to the representation (see part1/features.py)
    nb_alpha:                   smoothing of the Naive Bayes weights
    hidden_layers, hidden_dim, dropout_rate:      the MLP (see part1/smlp.py)
    learning_rate, batch_size, weight_decay:      its training (AdamW)
    max_epochs, patience:       at most max_epochs; stop after `patience` epochs without a lower validation loss
    validation_fraction:        share of the training folds set aside for choosing the epoch
    folds, split_seeds, seed:   number of CV folds, seeds of the fold splits, seed of the model and the validation split
    device:                     "cpu" or "cuda"
    save_models:                save every fold's model (large for bigrams: input features x hidden_dim floats)

Every fold is saved as one JSON line in results/mlp/folds.jsonl, and every run in results/mlp/runs.jsonl.
Everything else a run produces goes to results/mlp/runs/{run}/:
    config.json, summary.json
    predictions.npz             one row per held-out review of every fold: split_seed, fold, row_id (index in the
                                `train` split), label, prediction, prob_positive
    seed-{s}/fold-{f}/model.json, model.pt   the trained MLP (load with smlp.load), if save_models
    seed-{s}/fold-{f}/features.pickle        what the representation learned from the training data (TF-IDF
                                             weights, bigram vocabulary, Naive Bayes weights), if any
"""
import argparse
import json
import os
import pickle
import random
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so that `part1` can be imported when this file is run as a script

from part1 import smlp  # noqa: E402
from part1.experiment import cross_validation_split  # noqa: E402
from part1.features import (BagOfWords, NaiveBayesWeighting, NormalisedBagOfWords, TfidfEncoder,  # noqa: E402
                            TokenIndicator)
from part1.tokenisation import get_tokenizer  # noqa: E402
from part1.train_loglin import (append_jsonl, grid, load_train_split, numbers, parse_tokenizer,  # noqa: E402
                                split_overrides)

RESULTS = ROOT / "results"
REPRESENTATIONS = {"token_indicators": TokenIndicator, "token_indicator": TokenIndicator, "bow": BagOfWords,
                   "bow_l1": NormalisedBagOfWords, "bow_l2": NormalisedBagOfWords, "tf-idf": TfidfEncoder}
DEFAULTS = dict(
    tokenizer="bpe_16k",
    representation="tf-idf",
    min_df=2,
    sublinear_tf=False,
    nb_alpha=1.0,
    hidden_layers=1,
    hidden_dim=128,
    dropout_rate=0.1,
    learning_rate=1e-4,
    batch_size=64,
    weight_decay=1e-4,
    max_epochs=10,
    patience=3,
    validation_fraction=0.1,
    folds=5,
    split_seeds=[0],
    seed=0,
    device="cpu",
    save_models=True,
)


def make_encoder(config, tokenizer):
    """The representation named in the config, e.g. "tf-idf-bigrams-nb"."""
    name = config["representation"]
    naive_bayes = name.endswith("-nb")
    name = name.removesuffix("-nb")
    bigrams = name.endswith("-bigrams")
    name = name.removesuffix("-bigrams")
    if name not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation={config['representation']}, expected one of {list(REPRESENTATIONS)}"
                         " optionally followed by -bigrams and/or -nb")
    kwargs = dict(ngram_range=(1, 2) if bigrams else (1, 1), min_df=config["min_df"])
    if name == "tf-idf":
        kwargs["sublinear_tf"] = config["sublinear_tf"]
    if name in ("bow_l1", "bow_l2"):
        kwargs["norm"] = name[-2:]
    encoder = REPRESENTATIONS[name](tokenizer, **kwargs)
    return NaiveBayesWeighting(encoder, alpha=config["nb_alpha"]) if naive_bayes else encoder


def feature_state(encoder):
    """What a representation learned from the training data: nothing for counts and indicators of unigrams."""
    state = {}
    if isinstance(encoder, NaiveBayesWeighting):
        state["naive_bayes_r"] = encoder.r
        encoder = encoder.encoder
    if isinstance(encoder, TfidfEncoder):
        state["idf"] = encoder.transformer.idf_
    if encoder._bigrams is not None:
        state["bigram_vocabulary"] = encoder._bigrams.vectorizer.vocabulary_
    return state


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cross_validate(config, texts, labels, log=print, save_to=None):
    """
    Trains and scores one configuration on every fold, calling log(record) after each fold.
    Returns the full config, the fold records, their summary and the held-out predictions.
    save_to: if given, the folder for every fold's model and learned features
    """
    config = {**DEFAULTS, **config}
    if isinstance(config["split_seeds"], int):
        config["split_seeds"] = [config["split_seeds"]]
    kind, vocab_size = parse_tokenizer(config["tokenizer"])
    tokenizer = get_tokenizer(kind, vocab_size, folder=RESULTS / "tokenizers")
    num_classes = len(np.unique(labels))

    # repeated cross-validation: k folds for every split seed, as with my_seeds in the tutorial
    splits = [(split_seed, fold, split)
              for split_seed in config["split_seeds"]
              for fold, split in cross_validation_split(list(zip(texts, labels)), k=config["folds"], seed=split_seed).items()]
    records, predictions = [], {key: [] for key in ("split_seed", "fold", "row_id", "label", "prediction", "prob_positive")}
    for split_seed, fold, split in splits:
        train_ids = np.asarray(split["training_ids"], dtype=int)  # cross_validation_split returns object arrays
        eval_ids = np.asarray(split["validation_ids"], dtype=int)
        # validation part for choosing the epoch, taken from the training folds only
        shuffled = np.random.default_rng(config["seed"]).permutation(train_ids)
        num_validation = round(config["validation_fraction"] * len(train_ids))
        validation_ids, fit_ids = shuffled[:num_validation], shuffled[num_validation:]

        set_seed(config["seed"])
        start = time.perf_counter()
        encoder = make_encoder(config, tokenizer)
        fit_texts = [texts[i] for i in fit_ids]
        if isinstance(encoder, NaiveBayesWeighting):
            encoder.fit(fit_texts, labels[fit_ids])
        else:
            encoder.fit(fit_texts)  # features are learned from the training part only

        fold_folder = Path(save_to) / f"seed-{split_seed}" / f"fold-{fold}" if save_to is not None else None
        with tempfile.TemporaryDirectory() as tmp:
            model_folder = fold_folder if fold_folder is not None and config["save_models"] else Path(tmp)
            model_folder.mkdir(parents=True, exist_ok=True)
            metrics, best_epoch, _ = smlp.train_neural_network_model(
                [(texts[i], labels[i]) for i in fit_ids],
                [(texts[i], labels[i]) for i in validation_ids],
                num_classes=num_classes,
                encoder=encoder,
                save_to=str(model_folder / "model"),
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
            model = smlp.load(str(model_folder / "model"), device=config["device"])  # the best epoch's weights
        if fold_folder is not None and feature_state(encoder):
            fold_folder.mkdir(parents=True, exist_ok=True)
            with open(fold_folder / "features.pickle", "wb") as f:
                pickle.dump(feature_state(encoder), f, protocol=pickle.HIGHEST_PROTOCOL)

        start = time.perf_counter()
        probs = smlp.predict_probs([texts[i] for i in eval_ids], model, encoder, batch_size=256)
        fold_predictions = probs.argmax(axis=1)  # the most probable class
        predict_time = time.perf_counter() - start
        predictions["split_seed"].append(np.full(len(eval_ids), split_seed))
        predictions["fold"].append(np.full(len(eval_ids), fold))
        predictions["row_id"].append(eval_ids)
        predictions["label"].append(labels[eval_ids])
        predictions["prediction"].append(fold_predictions)
        predictions["prob_positive"].append(probs[:, 1])

        record = dict(
            split_seed=split_seed,
            fold=fold,
            accuracy=accuracy_score(labels[eval_ids], fold_predictions),
            macro_f1=f1_score(labels[eval_ids], fold_predictions, average="macro"),
            best_epoch=best_epoch + 1,  # counted from 1
            epochs_run=len(metrics),
            validation_loss=metrics[best_epoch]["val_metrics"]["loss"],
            validation_accuracy=metrics[best_epoch]["val_metrics"]["accuracy"],
            train_accuracy=metrics[best_epoch]["train_metrics"]["accuracy"],
            num_features=len(encoder),
            num_parameters=sum(p.numel() for p in model.parameters()),
            train_time=train_time,
            predict_time=predict_time,
        )
        records.append(record)
        log(record)

    summary = dict(
        accuracy=float(np.mean([r["accuracy"] for r in records])),  # over every fold of every split seed
        accuracy_std=float(np.std([r["accuracy"] for r in records])),
        macro_f1=float(np.mean([r["macro_f1"] for r in records])),
        best_epoch=float(np.mean([r["best_epoch"] for r in records])),
        train_time=float(np.mean([r["train_time"] for r in records])),
        **{f"accuracy_split_seed_{seed}": float(np.mean([r["accuracy"] for r in records if r["split_seed"] == seed]))
           for seed in config["split_seeds"]},
    )
    predictions = {key: np.concatenate(values) for key, values in predictions.items()}
    return config, records, summary, predictions


def run(config, texts, labels, wandb_run=None):
    """One configuration: cross-validates it and saves (and, given a W&B run, logs) the results."""
    config = {key: numbers(value) for key, value in config.items()}

    def log(record):
        print(json.dumps(record), flush=True)
        if wandb_run is not None:
            wandb_run.log({f"fold/{key}": value for key, value in record.items()})

    # the time, plus a random suffix so that runs started in the same second do not share a folder
    run_id = wandb_run.id if wandb_run is not None else f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    folder = RESULTS / "mlp" / "runs" / run_id
    config, records, summary, predictions = cross_validate(config, texts, labels, log=log, save_to=folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", **predictions)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    for record in records:
        append_jsonl(RESULTS / "mlp" / "folds.jsonl", dict(run=run_id, config=config, **record))
    append_jsonl(RESULTS / "mlp" / "runs.jsonl",
                 dict(run=run_id, config=config, folder=str(folder.relative_to(ROOT)), **summary))
    print(json.dumps(summary), flush=True)
    if wandb_run is not None:
        wandb_run.summary.update(summary)
    return summary


def read_sweep(path):
    with open(path) as f:
        sweep = yaml.safe_load(f)
    if "method" not in sweep:
        print("The sweep config has no method; using grid")
        sweep["method"] = "grid"
    return sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_config", nargs="?", help="YAML sweep config; without it, run a single configuration")
    parser.add_argument("--local", action="store_true", help="run the sweep's configurations here, without W&B")
    parser.add_argument("--count", type=int, help="maximum number of runs")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "athena-mlp"),
                        help="W&B project (default: $WANDB_PROJECT, else athena-mlp)")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"),
                        help="W&B team or user the project belongs to (default: $WANDB_ENTITY, else your default entity)")
    argv, overrides = split_overrides(sys.argv[1:], {"--local", "--count", "--project", "--entity", "--help"})
    args = parser.parse_args(argv)

    texts, labels = load_train_split()

    if args.sweep_config is None:
        if "WANDB_SWEEP_ID" in os.environ:
            import wandb
            with wandb.init() as wandb_run:  # started by a W&B agent, which also set the config
                run(dict(wandb_run.config), texts, labels, wandb_run)
        else:
            run(overrides, texts, labels)
        return

    sweep = read_sweep(args.sweep_config)
    if args.local:
        configs = list(grid(sweep["parameters"]))
        if sweep["method"] == "random":
            random.Random(0).shuffle(configs)
        for i, config in enumerate(configs[:args.count]):
            print(f"--- run {i + 1}/{len(configs[:args.count])}: {config}", flush=True)
            run(config, texts, labels)
        return

    import wandb

    def agent_run():
        with wandb.init() as wandb_run:
            run(dict(wandb_run.config), texts, labels, wandb_run)

    sweep_id = wandb.sweep(sweep, entity=args.entity, project=args.project)
    wandb.agent(sweep_id, function=agent_run, entity=args.entity, project=args.project, count=args.count)


if __name__ == "__main__":
    main()
