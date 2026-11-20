# Task 3 — Can averaging word embeddings beat our Part 1 IMDb classifier?

Nov 20, 2026 · Angelos

Our best embed-and-pool model reached **89.72% accuracy** with fine-tuned fastText vectors. Learning those vectors from scratch reached **89.63%** on the same reviews. Updating word vectors using IMDb sentiment labels mattered much more than starting from pretrained values. Our best Part 1 classifier still led at **92.25%**.

## How we measured it

We averaged the 300-dimensional word vectors in each review, excluding padding, then passed the average to a classifier with two 50-unit hidden layers. Averaging loses word order. For example, “good acting, bad writing” and “bad acting, good writing” produce the same vector even though the reviewer praises different things.

We reused the Part 1 five-fold split (seed 3) on all 25,000 labelled IMDb training reviews. In each fold, 18,000 reviews trained the model, 2,000 selected the epoch with the lowest validation loss, and 5,000 evaluated it. Each review contributed one held-out prediction to the scores below. We kept the embedding size, classifier, dropout (0.1), optimiser (AdamW, learning rate 0.001, weight decay 0.0001), batch size (64) and training budget (20 epochs) fixed. [Pooling encoder](embed_and_pool.py) · [Training loop](../experiment.py)

| Starting word vectors | Updated during training? | Accuracy | Macro-F1 | Run ID in the [W&B workspace](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/workspace?nw=nwuserangelosntousis) |
|---|---|---:|---:|---|
| GloVe | No | 83.09% | 0.8309 | `dl4cj64u` |
| word2vec | No | 84.75% | 0.8475 | `3ut26gpz` |
| fastText | No | 86.74% | 0.8674 | `rsa2q3qs` |
| Random, using the fastText vocabulary | Yes | 89.63% | 0.8963 | `f774fbb1` |
| fastText | Yes | **89.72%** | **0.8972** | `zr17jbgj` |

## Which design choices mattered?

**When the vectors were frozen, their source mattered.** fastText led word2vec by 1.98 percentage points and GloVe by 3.65. This agrees with our earlier observation that fastText captured review-specific meanings such as *plot* as a storyline. Each released embedding also has its own training corpus, vocabulary and token conventions, so this experiment cannot attribute the gap to any one of those differences. [Embedding analysis](../task2_explore_embeddings/TASK2.md)

**Letting IMDb labels update fastText improved accuracy by 2.99 points**, with gains on all five folds. To isolate the value of pretraining, we kept the same tokenizer, vocabulary and classifier and changed only the initial vector values. Pretrained fastText beat random initialisation on three folds and trailed on two. Overall it classified **23 more reviews correctly out of 25,000**. That 0.09-point difference gives little evidence of a practically important advantage from pretraining in this full-data setting.

## Did we improve on Part 1?

**No.** On the same folds, the Part 1 MLP with two 256-unit layers and Naive Bayes-weighted TF-IDF unigrams and bigrams led our best pooling model by **2.52 percentage points**. Its bigrams can give an adjacent pair such as *not good* its own feature. Mean pooling has no feature dedicated to that pair. This is one plausible explanation for the gap. The two systems also differ in tokenisation, feature weighting and classifier size. [Part 1 results](../../results/final_predictions/index.csv)

## Reproducing Task 3 on Snellius

This folder contains the pooling encoder, the random-initialisation control and the A100 job that runs all five configurations. The shared runner handles folds, checkpoint selection, predictions and W&B logging. [Task 3 code](embed_and_pool.py) · [Slurm job](snellius.sbatch) · [Shared runner](../project.py)

From the repository root on Snellius, with the project environment in `.venv`, the IMDb dataset cached, the Task 1 vectors in `embeddings/` and W&B configured for `nlp1-athena/Part 2`, run the following command.

```bash
sbatch part2/task3_embed_and_pool/snellius.sbatch
```

Array indices 0–2 run frozen GloVe, fastText and word2vec, respectively. Index 3 fine-tunes fastText and index 4 trains random vectors. Each index runs five folds on one A100. To repeat just one configuration, add the index to the submission command, e.g. `sbatch --array=3 part2/task3_embed_and_pool/snellius.sbatch`.

The published frozen runs came from job `29497316`, indices 0–2. The fine-tuned and random runs came from job `29497883`, indices 2 and 3, respectively. New submissions write Slurm logs to `results/imdb-pool-<job>_<index>.out` and per-fold predictions and epoch histories to `results/part2/<run-name>-<run-id>/`.
