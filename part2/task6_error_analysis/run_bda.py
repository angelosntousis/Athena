"""
Task 6: which review features predict where our IMDb systems succeed or fail?

Fits the tutorial's BDA model (part2.analysis.numpyro_model, a Bayesian logistic regression of
"was this prediction correct?") to the out-of-fold predictions of the 9 systems in SYSTEMS, using
the features from features.py. Run from the repository root (Athena/):

    python -m part2.task6_error_analysis.features text
    python -m part2.task6_error_analysis.features embedding
    python -m part2.task6_error_analysis.run_bda

Options: --reviews (random subset size, default 5000), --chains, --warmup, --samples, --seed,
--text-only (skip the embedding features), --out (default results/bda/).

Writes to --out:
    accuracy.csv      observed and posterior accuracy of each system on the subset
    weights.csv       each system's weight per feature: median, 95% interval, P(weight > 0)
    comparisons.csv   the design-choice comparisons in COMPARISONS: accuracy difference and the
                      difference in each feature weight, with P(first system > second)
    correlations.csv  correlations between the features, to spot near-duplicates
    convergence.txt   divergences, worst r_hat and smallest effective sample size
    accuracy.png, weights.png
    posterior.npz     every sample, shaped (chain, draw, ...), with the system, feature and review
                      ids; az.from_dict(posterior={...}) turns it into arviz data if needed

Features are standardised, so a weight is the change in log-odds of a correct prediction per
standard deviation of the feature. A system's accuracy is its average probability of being
correct over the same reviews as every other system, so accuracies are directly comparable.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from part2.task6_error_analysis.features import EMBEDDING_FEATURES, TEXT_FEATURES  # noqa: E402

PREDICTIONS = ROOT / "part2" / "predictions"

# name: (W&B run id, encoder, embedding, weights). All LSTMs are the 1-layer, 20-unit baseline.
SYSTEMS = {
    "pool-glove": ("dl4cj64u", "embed-and-pool", "glove", "frozen"),
    "pool-word2vec": ("3ut26gpz", "embed-and-pool", "word2vec", "frozen"),
    "pool-fasttext": ("rsa2q3qs", "embed-and-pool", "fasttext", "frozen"),
    "pool-fasttext-tuned": ("zr17jbgj", "embed-and-pool", "fasttext", "fine-tuned"),
    "pool-random": ("f774fbb1", "embed-and-pool", "fasttext", "random-trainable"),
    "lstm-glove": ("7e5avnkt", "lstm", "glove", "frozen"),
    "lstm-word2vec": ("3cwzsntd", "lstm", "word2vec", "frozen"),
    "lstm-fasttext": ("7h58qpb5", "lstm", "fasttext", "frozen"),
    "lstm-fasttext-tuned": ("zxn1n2hg", "lstm", "fasttext", "fine-tuned"),
}

TEXT_COLUMNS = ["label", "log_length", "negation_rate", "contrast_rate", "hedge_rate", "rare_word_rate"]
# sentiment_strength_word2vec is left out: it correlates 0.95 with fastText's, so the two weights
# could not be told apart. features.py still computes it.
EMBEDDING_COLUMNS = ["sentiment_strength_glove", "sentiment_strength_fasttext", "mismatch_glove", "mismatch_word2vec"]

# (first, second, design choice): "is first better than second, and on which reviews?"
COMPARISONS = [
    ("pool-fasttext", "pool-glove", "embedding (pool)"),
    ("pool-word2vec", "pool-glove", "embedding (pool)"),
    ("pool-fasttext", "pool-word2vec", "embedding (pool)"),
    ("lstm-fasttext", "lstm-glove", "embedding (lstm)"),
    ("lstm-word2vec", "lstm-glove", "embedding (lstm)"),
    ("lstm-fasttext", "lstm-word2vec", "embedding (lstm)"),
    ("pool-fasttext-tuned", "pool-fasttext", "fine-tuning (pool)"),
    ("lstm-fasttext-tuned", "lstm-fasttext", "fine-tuning (lstm)"),
    ("pool-fasttext-tuned", "pool-random", "pretraining (pool)"),
    ("lstm-glove", "pool-glove", "encoder"),
    ("lstm-word2vec", "pool-word2vec", "encoder"),
    ("lstm-fasttext", "pool-fasttext", "encoder"),
    ("lstm-fasttext-tuned", "pool-fasttext-tuned", "encoder"),
]


def check_systems():
    """Make sure every run id is the configuration its name claims."""
    configs = pd.read_csv(PREDICTIONS / "configurations.csv").set_index("run_id")
    for name, (run_id, encoder, embedding, weights) in SYSTEMS.items():
        row = configs.loc[run_id]
        found = (row.encoder, row.embedding, row.weights, row.fit_fraction)
        if found != (encoder, embedding, weights, 1) or (encoder == "lstm" and (row.hidden_size, row.num_layers) != (20, 1)):
            raise ValueError(f"{name}: run {run_id} is {found}")


def load_table(text_only):
    columns = ["row_id", "true_label"] + [f"{run_id}_prediction" for run_id, *_ in SYSTEMS.values()]
    table = pd.read_csv(PREDICTIONS / "all_configurations_oof.csv.gz", usecols=columns)
    for path in [TEXT_FEATURES] + ([] if text_only else [EMBEDDING_FEATURES]):
        if not path.exists():
            step = "text" if path == TEXT_FEATURES else "embedding"
            sys.exit(f"{path.name} is missing. Run: python -m part2.task6_error_analysis.features {step}")
        table = table.merge(pd.read_csv(path), on="row_id", validate="one_to_one")
    if not (table["label"] == table["true_label"]).all():
        raise ValueError("Feature labels do not match the prediction labels.")
    return table


def fit(sys_id, x, correct, args):
    """NUTS on the tutorial model, without storing one logit per row and sample."""
    import jax.numpy as jnp
    from jax import random
    from numpyro.handlers import block
    from numpyro.infer import MCMC, NUTS
    from part2.analysis import numpyro_model

    mcmc = MCMC(
        NUTS(block(numpyro_model, hide=["logit"]), target_accept_prob=0.9),
        num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains,
        chain_method="parallel", progress_bar=True,
    )
    mcmc.run(random.PRNGKey(args.seed), num_systems=len(SYSTEMS), sys_id=jnp.asarray(sys_id, dtype=jnp.int32),
             features=jnp.asarray(x), correctness=jnp.asarray(correct, dtype=jnp.int32))
    return mcmc


def interval(samples, axis=0):
    lo, mid, hi = np.percentile(samples, [2.5, 50, 97.5], axis=axis)
    return mid, lo, hi


def summarise(mcmc, systems, features, observed, out):
    from numpyro.diagnostics import summary
    s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    acc = pd.DataFrame({"system": systems, "observed": observed})
    acc["median"], acc["low_95"], acc["high_95"] = interval(s["accuracy"])
    acc.to_csv(out / "accuracy.csv", index=False, float_format="%.4f")

    rows = []
    for i, system in enumerate(systems):
        for j, feature in enumerate(features):
            w = s["w"][:, i, j]
            rows.append([system, feature, *interval(w), (w > 0).mean()])
    weights = pd.DataFrame(rows, columns=["system", "feature", "median", "low_95", "high_95", "p_positive"])
    weights.to_csv(out / "weights.csv", index=False, float_format="%.4f")

    rows, index = [], {name: i for i, name in enumerate(systems)}
    for a, b, choice in COMPARISONS:
        ia, ib = index[a], index[b]
        quantities = {"accuracy": s["accuracy"][:, ia] - s["accuracy"][:, ib]}
        quantities.update({f: s["w"][:, ia, j] - s["w"][:, ib, j] for j, f in enumerate(features)})
        for quantity, diff in quantities.items():
            rows.append([choice, a, b, quantity, *interval(diff), (diff > 0).mean()])
    comparisons = pd.DataFrame(rows, columns=["design_choice", "first", "second", "quantity",
                                              "median_difference", "low_95", "high_95", "p_first_higher"])
    comparisons.to_csv(out / "comparisons.csv", index=False, float_format="%.4f")

    stats = summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
    params = ["global_mean", "sys_shift", "w"]
    r_hat = max(float(np.max(stats[p]["r_hat"])) for p in params)
    n_eff = min(float(np.min(stats[p]["n_eff"])) for p in params)
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    verdict = "OK" if r_hat < 1.01 and divergences == 0 else "CHECK: do not trust these results yet"
    report = (f"divergences: {divergences}\nworst r_hat: {r_hat:.4f} (want < 1.01)\n"
              f"smallest effective sample size: {n_eff:.0f}\nverdict: {verdict}\n")
    (out / "convergence.txt").write_text(report)
    return acc, weights, comparisons, report


def plot(acc, weights, systems, features, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colour = {s: ("tab:blue" if s.startswith("pool") else "tab:orange") for s in systems}
    y = np.arange(len(systems))[::-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(acc["median"], y, xerr=[acc["median"] - acc["low_95"], acc["high_95"] - acc["median"]],
                fmt="none", ecolor="grey")
    ax.scatter(acc["median"], y, c=[colour[s] for s in systems], zorder=3)
    ax.set_yticks(y, systems)
    ax.set_xlabel("posterior accuracy (median, 95% interval)")
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=150)

    cols = 3
    fig, axes = plt.subplots(-(-len(features) // cols), cols, figsize=(4 * cols, 2.6 * -(-len(features) // cols)),
                             sharey=True, squeeze=False)
    for ax, feature in zip(axes.flat, features):
        w = weights[weights.feature == feature].set_index("system").loc[systems]
        ax.axvline(0, color="black", lw=0.8)
        ax.errorbar(w["median"], y, xerr=[w["median"] - w["low_95"], w["high_95"] - w["median"]],
                    fmt="none", ecolor="grey")
        ax.scatter(w["median"], y, c=[colour[s] for s in systems], zorder=3, s=15)
        ax.set_title(feature, fontsize=10)
        ax.set_yticks(y, systems, fontsize=8)
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    fig.supxlabel("weight: change in log-odds of a correct prediction per standard deviation")
    fig.tight_layout()
    fig.savefig(out / "weights.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviews", type=int, default=5000, help="Random subset of reviews (all 9 systems each)")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="Seed for the review subset and for MCMC")
    parser.add_argument("--text-only", action="store_true", help="Leave out the embedding features")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "bda")
    args = parser.parse_args()

    # Must happen before JAX starts, so that the chains run in parallel on CPU cores.
    import numpyro
    numpyro.set_host_device_count(args.chains)

    check_systems()
    table = load_table(args.text_only)
    features = TEXT_COLUMNS + ([] if args.text_only else EMBEDDING_COLUMNS)
    rng = np.random.default_rng(args.seed)
    if args.reviews < len(table):
        table = table.iloc[np.sort(rng.choice(len(table), args.reviews, replace=False))]
    args.out.mkdir(parents=True, exist_ok=True)

    corr = table[features].corr()
    corr.to_csv(args.out / "correlations.csv", float_format="%.3f")
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(features) for b in features[i + 1:] if abs(corr.loc[a, b]) > 0.7]
    for a, b, r in pairs:
        print(f"Warning: {a} and {b} are strongly correlated (r = {r:.2f}); their weights will be hard to separate.")

    systems = list(SYSTEMS)
    x = table[features].to_numpy(dtype=float)
    x = (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)  # as part2.analysis.fit_bda_model does
    sys_id = np.repeat(np.arange(len(systems)), len(table))
    correct = np.concatenate([(table[f"{SYSTEMS[s][0]}_prediction"] == table.true_label).to_numpy() for s in systems])
    observed = correct.reshape(len(systems), -1).mean(1)
    print(f"Fitting {len(systems)} systems x {len(table):,} reviews = {len(correct):,} rows, {len(features)} features")

    mcmc = fit(sys_id, np.tile(x, (len(systems), 1)), correct, args)

    acc, weights, comparisons, report = summarise(mcmc, systems, features, observed, args.out)
    plot(acc, weights, systems, features, args.out)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    np.savez_compressed(args.out / "posterior.npz", systems=systems, features=features,
                        review_ids=table.row_id.to_numpy(), **samples)

    print("\n" + report)
    print(acc.to_string(index=False, float_format="%.4f"))
    print(f"\nResults in {args.out}")


if __name__ == "__main__":
    main()
