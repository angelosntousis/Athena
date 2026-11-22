# Task 6: which reviews does each design choice get right?

A Bayesian logistic regression (the tutorial's BDA model) predicts whether each system classified each review correctly, from features of the review. Comparing the systems' weights shows where each design choice helps or hurts.

## Systems

All runs are out-of-fold predictions on the same 25,000 IMDb training reviews (Part 1 folds, seed 3, training seed 0). Every LSTM is the 1-layer, 20-unit baseline, so each comparison changes one thing.

| System | Run | Accuracy |
|---|---|---:|
| pool-glove | `dl4cj64u` | 83.09% |
| pool-word2vec | `3ut26gpz` | 84.75% |
| pool-fasttext | `rsa2q3qs` | 86.74% |
| pool-fasttext-tuned | `zr17jbgj` | 89.72% |
| pool-random (fastText vocabulary, random trained vectors) | `f774fbb1` | 89.63% |
| lstm-glove | `7e5avnkt` | 88.06% |
| lstm-word2vec | `3cwzsntd` | 88.82% |
| lstm-fasttext | `7h58qpb5` | 89.41% |
| lstm-fasttext-tuned | `zxn1n2hg` | 84.81% |

| Design choice | Compared |
|---|---|
| Embedding source | pool-glove / word2vec / fasttext; lstm-glove / word2vec / fasttext |
| Freezing vs fine-tuning (fastText only) | pool-fasttext vs pool-fasttext-tuned; lstm-fasttext vs lstm-fasttext-tuned |
| Pretraining (pooling only) | pool-fasttext-tuned vs pool-random |
| Encoder | lstm-X vs pool-X for each of the four embedding settings |

## Features and hypotheses

| Feature | Definition | Hypothesis |
|---|---|---|
| `label` | 1 = positive | GloVe misses more positive reviews |
| `log_length` | log(words) | Long reviews dilute pooling's average and strain the LSTM |
| `negation_rate` | *not, no, never, n't, cannot, nothing, nobody, none, neither, nor, without* per 100 words | Hurts pooling more than the LSTM |
| `contrast_rate` | *but, however, although, though, yet, despite, nevertheless* per 100 words | Mixed reviews hurt pooling more |
| `hedge_rate` | *somewhat, bit, almost, rather, fairly, slightly, kinda* per 100 words | Mild reviews are harder for every system |
| `rare_word_rate` | Share of content words in fewer than 20 of the fold's 18,000 fitting reviews | Hurts random vectors more than fine-tuned fastText |
| `sentiment_strength_<emb>` | \|mean cosine of the review's content words with the embedding's sentiment direction\| (opinion lexicon, positive minus negative), centred on the IMDb average | Weak signal in GloVe predicts GloVe's failures; fine-tuning helps pooling most where frozen fastText's signal is weak |
| `mismatch_<emb>` | Mean share of the 10 nearest neighbours of the review's content words that differ from fastText's (GloVe and word2vec) | Hurts GloVe, and word2vec less, but not fastText |

The BDA uses sentiment strength for GloVe and fastText only. word2vec's correlates 0.95 with fastText's, so their weights could not be separated.

Word lists are rates, so they don't just repeat length. Content words exclude NLTK's English stop words and the word lists above.

## Running it

From the repository root, with the project environment, the IMDb dataset cached, NLTK's `stopwords` and `opinion_lexicon` downloaded, and `embeddings/imdb_<model>.txt.gz` from Task 1:

```bash
python -m part2.task6_error_analysis.features text        # ~30 s, needs only the reviews
python -m part2.task6_error_analysis.features embedding   # ~1-2 min, needs the embedding files
python -m part2.task6_error_analysis.run_bda              # ~10 min on a laptop, 4 chains in parallel
```

On Snellius, `sbatch part2/task6_error_analysis/bda.sbatch` runs all three on a CPU node. Arguments after the script go to `run_bda`, for example `--reviews 10000`.

The feature files (`text_features.csv.gz`, `embedding_features.csv.gz`) are small and meant to be committed, so the BDA can be rerun anywhere without the embeddings. Results go to `results/bda/` (not tracked):

- `convergence.txt`: read first; trust the rest only if the verdict is OK
- `accuracy.csv`, `accuracy.png`: each system's posterior accuracy
- `weights.csv`, `weights.png`: each system's weight per feature, with P(weight > 0)
- `comparisons.csv`: for each design choice, the difference in accuracy and in every feature weight, with P(first system higher)
- `correlations.csv`: feature correlations; a warning is printed for any pair above 0.7
- `posterior.npz`: all samples

## Limits

- One training seed per system. The BDA's intervals cover which reviews were sampled, not training randomness. When the seed 1-2 runs finish, add them as extra systems to check that the findings hold.
- The fit uses a random subset of 5,000 reviews (45,000 rows) to keep MCMC fast; `--reviews` changes it.
