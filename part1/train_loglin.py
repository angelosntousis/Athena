"""
Logistic regression on IMDb (part1/loglin.py), scored by k-fold cross-validation on the `train` split,
with the tutorial's folds (part1/experiment.py: cross_validation_split), so that other models can be scored on the same folds.

One run = one configuration: a tokeniser (part1/tokenisation.py), a representation (part1/features.py)
and the regression's hyperparameters. For every fold, the representation is fitted on the training folds only;
the tokenisers were trained on the `unsupervised` split, so they can be shared by all folds.
The mean accuracy over the folds is logged as `accuracy`, the metric the sweep maximises.

Usage, from the repository root or from part1/:

    python part1/train_loglin.py part1/loglin_sweep_config.yaml          # create a W&B sweep and run an agent
    python part1/train_loglin.py part1/loglin_sweep_config.yaml --count 20
    python part1/train_loglin.py part1/loglin_sweep_config.yaml --local  # every grid configuration, without W&B
    python part1/train_loglin.py --tokenizer bpe_16k --representation bow --regularisation l2   # one run

When a W&B agent starts this script (as the sweep's `program`), it is called with --key=value arguments
for the sweep parameters, which are read the same way as in the single-run form above.

Config keys (missing ones take the values in DEFAULTS):
    tokenizer:                  "word" or "bpe", optionally with a vocabulary size: "bpe_16k", "word_32k"
                                (no size: DEFAULT_VOCAB_SIZE)
    representation:             "token_indicator", "bow", "bow_l1", "bow_l2" or "tf-idf"
                                (bow_l1/bow_l2: counts divided by the review's L1/L2 norm); add "-nb" for Naive Bayes
                                weighting, e.g. "tf-idf-nb"
    nb_alpha:                   smoothing of the Naive Bayes weights
    ngram_range, min_df, sublinear_tf: passed on to the representation (see part1/features.py)
    regularisation:             a number: the L2 coefficient lambda (0 means no regularisation);
                                or "l1", "l2", or "no" (or "none") for no regularisation, optionally with the
                                coefficient: "l1_0.01" is the same as regularisation l1 and coefficient 0.01
    regularisation_coefficient: lambda = 1 / C (0 means no regularisation)
    epochs:                     maximum passes of the saga solver over the training folds (max_iter)
    tol:                        saga stops before `epochs` once the weights change less than this
    folds:                      number of CV folds
    split_seeds:                seeds of the fold splits (a list, or one seed); every run is scored on the k folds
                                of every seed, and `accuracy` is the mean over all of them
    seed:                       seed of the solver (the order in which saga visits the training examples)

Every fold is also saved as one JSON line in results/loglin/folds.jsonl, and every run in results/loglin/runs.jsonl.
Everything else a run produces goes to results/loglin/runs/{run}/:
    config.json, summary.json
    predictions.npz             one row per held-out review of every fold: split_seed, fold, row_id (index in the
                                `train` split), label, prediction, prob_positive
    seed-{s}/fold-{f}/model.pickle      the trained LogisticRegression (load with loglin.load)
    seed-{s}/fold-{f}/features.pickle   what the representation learned from the training folds (TF-IDF weights,
                                        bigram vocabulary); only for representations that learn something
The tokeniser is not copied: it is saved once in results/tokenizers.
"""
import argparse
import itertools
import json
import os
import sys
import pickle
import re
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so that `part1` can be imported when this file is run as a script

from part1 import loglin  # noqa: E402
from part1.experiment import cross_validation_split  # noqa: E402
from part1.features import (BagOfWords, NaiveBayesWeighting, NormalisedBagOfWords, TfidfEncoder,  # noqa: E402
                            TokenIndicator)
from part1.tokenisation import KINDS, get_tokenizer  # noqa: E402

RESULTS = ROOT / "results"
DEFAULT_VOCAB_SIZE = 32000
REPRESENTATIONS = {"token_indicator": TokenIndicator, "bow": BagOfWords, "bow_l1": NormalisedBagOfWords,
                   "bow_l2": NormalisedBagOfWords, "tf-idf": TfidfEncoder}
REGULARISATIONS = {"l1": 1.0, "l2": 0.0, "no": 0.0, "none": 0.0}  # l1_ratio passed to LogisticRegression
DEFAULTS = dict(
    tokenizer="bpe_16k",
    representation="bow",
    ngram_range=[1, 1],
    min_df=2,
    sublinear_tf=False,
    nb_alpha=1.0,
    regularisation="l2",
    regularisation_coefficient=1.0,
    epochs=100,
    tol=1e-4,
    folds=5,
    split_seeds=[0],
    seed=0,
)
# settings of gradient-descent training, which the saga solver of LogisticRegression does not have
IGNORED = ("learning_rate", "batch_size")


def parse_tokenizer(name):
    """'bpe_16k' or 'bpe16k' -> ('bpe', 16000); 'word' -> ('word', DEFAULT_VOCAB_SIZE)"""
    kind = next((kind for kind in KINDS if name.startswith(kind)), None)
    if kind is None:
        raise ValueError(f"Unknown tokenizer={name}, expected one of {KINDS}, optionally followed by a size like _16k")
    size = name[len(kind):].lstrip("_")
    if not size:
        return kind, DEFAULT_VOCAB_SIZE
    return kind, int(float(size[:-1]) * 1000) if size.lower().endswith("k") else int(size)


def make_encoder(config, tokenizer):
    name = config["representation"].removesuffix("-nb")
    if name not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation={config['representation']}, expected one of {list(REPRESENTATIONS)}"
                         " optionally followed by -nb")
    kwargs = dict(ngram_range=tuple(config["ngram_range"]), min_df=config["min_df"])
    if name == "tf-idf":
        kwargs["sublinear_tf"] = config["sublinear_tf"]
    if name in ("bow_l1", "bow_l2"):
        kwargs["norm"] = name[-2:]
    encoder = REPRESENTATIONS[name](tokenizer, **kwargs)
    if config["representation"].endswith("-nb"):
        return NaiveBayesWeighting(encoder, alpha=config["nb_alpha"])
    return encoder


def regression_args(config):
    """Maps the config's regularisation settings to the arguments of loglin.fit."""
    if config["regularisation"] not in REGULARISATIONS:
        raise ValueError(f"Unknown regularisation={config['regularisation']}, expected one of {list(REGULARISATIONS)}")
    coefficient = config["regularisation_coefficient"]
    no_penalty = config["regularisation"] in ("no", "none") or coefficient == 0
    return dict(
        C=float("inf") if no_penalty else 1 / coefficient,
        l1_ratio=REGULARISATIONS[config["regularisation"]],
        max_iter=config["epochs"],
        tol=config["tol"],
        random_state=config["seed"],
    )


def load_train_split():
    data = load_dataset("stanfordnlp/imdb", split="train")
    return list(data["text"]), np.array(data["label"])


def feature_state(encoder):
    """What a representation learned from the training texts: nothing for counts and indicators of unigrams."""
    state = {}
    if isinstance(encoder, NaiveBayesWeighting):
        state["naive_bayes_r"] = encoder.r
        encoder = encoder.encoder
    if isinstance(encoder, TfidfEncoder):
        state["idf"] = encoder.transformer.idf_
    if encoder._bigrams is not None:
        state["bigram_vocabulary"] = encoder._bigrams.vectorizer.vocabulary_
    return state


def cross_validate(config, texts, labels, log=print, save_to=None):
    """
    Trains and scores one configuration on every fold, calling log(record) after each fold.
    Returns the fold records, their summary and the held-out predictions.
    save_to: if given, the folder for every fold's model and learned features
    """
    config = {**DEFAULTS, **config}
    if isinstance(config["regularisation"], (int, float)):  # 0.1 -> l2 with coefficient 0.1
        config.update(regularisation="l2", regularisation_coefficient=float(config["regularisation"]))
    elif "_" in config["regularisation"]:  # "l1_0.01" -> l1 with coefficient 0.01
        penalty, _, coefficient = config["regularisation"].partition("_")
        config.update(regularisation=penalty, regularisation_coefficient=float(coefficient))
    if config["regularisation"] in ("no", "none") or config["regularisation_coefficient"] == 0:
        config.update(regularisation="none", regularisation_coefficient=0.0)  # the penalty does not apply
    kind, vocab_size = parse_tokenizer(config["tokenizer"])
    tokenizer = get_tokenizer(kind, vocab_size, folder=RESULTS / "tokenizers")
    args = regression_args(config)

    if isinstance(config["split_seeds"], int):
        config["split_seeds"] = [config["split_seeds"]]
    # repeated cross-validation: k folds for every split seed, as with my_seeds in the tutorial
    splits = [(split_seed, fold, split)
              for split_seed in config["split_seeds"]
              for fold, split in cross_validation_split(list(zip(texts, labels)), k=config["folds"], seed=split_seed).items()]
    records, predictions = [], {key: [] for key in ("split_seed", "fold", "row_id", "label", "prediction", "prob_positive")}
    for split_seed, fold, split in splits:
        train_ids = np.asarray(split["training_ids"], dtype=int)  # cross_validation_split returns object arrays
        eval_ids = np.asarray(split["validation_ids"], dtype=int)
        encoder = make_encoder(config, tokenizer)
        start = time.perf_counter()
        if isinstance(encoder, NaiveBayesWeighting):  # features are learned from the training folds only
            encoder.fit([texts[i] for i in train_ids], labels[train_ids])
        else:
            encoder.fit([texts[i] for i in train_ids])
        fold_folder = None
        if save_to is not None:
            fold_folder = Path(save_to) / f"seed-{split_seed}" / f"fold-{fold}"
            fold_folder.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = loglin.fit([(texts[i], labels[i]) for i in train_ids], encoder, n_jobs=None,
                               save_to=None if fold_folder is None else str(fold_folder / "model"), **args)
        train_time = time.perf_counter() - start
        if fold_folder is not None and feature_state(encoder):
            with open(fold_folder / "features.pickle", "wb") as f:
                pickle.dump(feature_state(encoder), f, protocol=pickle.HIGHEST_PROTOCOL)

        start = time.perf_counter()
        probs = loglin.predict_probs([texts[i] for i in eval_ids], model, encoder)
        fold_predictions = model.classes_[probs.argmax(axis=1)]  # the most probable class, as loglin.predict
        predict_time = time.perf_counter() - start
        predictions["split_seed"].append(np.full(len(eval_ids), split_seed))
        predictions["fold"].append(np.full(len(eval_ids), fold))
        predictions["row_id"].append(eval_ids)
        predictions["label"].append(labels[eval_ids])
        predictions["prediction"].append(fold_predictions)
        predictions["prob_positive"].append(probs[:, list(model.classes_).index(1)])

        record = dict(
            split_seed=split_seed,
            fold=fold,
            accuracy=accuracy_score(labels[eval_ids], fold_predictions),
            macro_f1=f1_score(labels[eval_ids], fold_predictions, average="macro"),
            converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            epochs_run=int(np.max(model.n_iter_)),
            num_features=len(encoder),
            nonzero_weights=int(np.count_nonzero(model.coef_)),
            train_time=train_time,
            predict_time=predict_time,
        )
        records.append(record)
        log(record)

    summary = dict(
        accuracy=float(np.mean([r["accuracy"] for r in records])),  # over every fold of every split seed
        accuracy_std=float(np.std([r["accuracy"] for r in records])),
        macro_f1=float(np.mean([r["macro_f1"] for r in records])),
        all_converged=all(r["converged"] for r in records),
        train_time=float(np.mean([r["train_time"] for r in records])),
        **{f"accuracy_split_seed_{seed}": float(np.mean([r["accuracy"] for r in records if r["split_seed"] == seed]))
           for seed in config["split_seeds"]},
    )
    predictions = {key: np.concatenate(values) for key, values in predictions.items()}
    return config, records, summary, predictions


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run(config, texts, labels, wandb_run=None):
    """One configuration: cross-validates it and saves (and, given a W&B run, logs) the results."""
    config = {key: numbers(value) for key, value in config.items()}
    ignored = {key: config.pop(key) for key in IGNORED if key in config}
    if ignored:
        warnings.warn(f"Ignoring {ignored}: LogisticRegression is trained with saga, which has no such settings")

    def log(record):
        print(json.dumps(record))
        if wandb_run is not None:
            wandb_run.log({f"fold/{key}": value for key, value in record.items()})

    # the time, plus a random suffix so that runs started in the same second do not share a folder
    run_id = wandb_run.id if wandb_run is not None else f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    folder = RESULTS / "loglin" / "runs" / run_id
    config, records, summary, predictions = cross_validate(config, texts, labels, log=log, save_to=folder)
    np.savez_compressed(folder / "predictions.npz", **predictions)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    for record in records:
        append_jsonl(RESULTS / "loglin" / "folds.jsonl", dict(run=run_id, config=config, **record))
    append_jsonl(RESULTS / "loglin" / "runs.jsonl",
                 dict(run=run_id, config=config, folder=str(folder.relative_to(ROOT)), **summary))
    print(json.dumps(summary))
    if wandb_run is not None:
        wandb_run.summary.update(summary)
    return summary


def read_sweep(path):
    """Reads a W&B sweep config, dropping the parameters that do not apply to logistic regression."""
    with open(path) as f:
        sweep = yaml.safe_load(f)
    for key in IGNORED:
        if sweep["parameters"].pop(key, None) is not None:
            print(f"Dropping sweep parameter {key}: LogisticRegression is trained with saga, which has no such setting")
    if "method" not in sweep:
        print("The sweep config has no method; using grid")
        sweep["method"] = "grid"
    return sweep


def grid(parameters):
    """Every combination of the sweep parameters (each given by `value` or `values`)."""
    names = list(parameters)
    choices = [spec["values"] if "values" in spec else [spec["value"]] for spec in parameters.values()]
    for combination in itertools.product(*choices):
        yield dict(zip(names, combination))


def numbers(value):
    """
    Converts strings that are numbers to floats, also inside lists. YAML reads 1e-4 as the string "1e-4"
    (YAML 1.1, which PyYAML follows, wants 1.0e-4), so config values may be numbers in disguise.
    """
    if isinstance(value, list):
        return [numbers(v) for v in value]
    if isinstance(value, str) and re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", value.strip()):
        return float(value)
    return value


def parse_value(text):
    """Command-line values, typed as YAML would type them (0.01 -> float, [1, 2] -> list, l2 -> str)."""
    return numbers(yaml.safe_load(text))


def split_overrides(argv, options):
    """
    Separates --key=value or --key value pairs for keys not in `options`: one configuration
    (this is how a W&B agent passes the sweep parameters). They are taken out before argparse, which would
    mistake a value with a space, like --ngram_range=[1, 2], for the positional sweep_config.
    """
    remaining, overrides, key = [], {}, None
    for token in argv:
        name, _, value = token.partition("=")
        if key is not None:
            overrides[key], key = parse_value(token), None
        elif token.startswith("--") and name not in options:
            if value:
                overrides[name[2:]] = parse_value(value)
            else:
                key = name[2:]
        else:
            remaining.append(token)
    return remaining, overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_config", nargs="?", help="YAML sweep config; without it, run a single configuration")
    parser.add_argument("--local", action="store_true", help="run the sweep's grid here, without W&B")
    parser.add_argument("--count", type=int, help="maximum number of runs")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "athena-loglin"),
                        help="W&B project (default: $WANDB_PROJECT, else athena-loglin)")
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
        for i, config in enumerate(grid(sweep["parameters"])):
            if args.count is not None and i >= args.count:
                break
            print(f"--- run {i}: {config}")
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
