# Task 4 — Does reading an IMDb review in order improve sentiment classification?

Nov 22, 2026 · Angelos

Our tuned LSTM reached **90.39% five-fold accuracy** on IMDb, up from **89.41%** for our original LSTM and **89.72%** for our best embed-and-pool model. Two frozen-fastText LSTMs tied at **22,597 correct predictions out of 25,000**. We report the smaller model. It uses two layers of 32 hidden units, 10% dropout between LSTM layers and 10% classifier dropout. [32-unit run](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/runs/ego2tslr) · [64-unit run](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/runs/bnx1ibm8)

## How we measured performance

The LSTM reads each review in order. A classifier with two 50-unit hidden layers predicts sentiment from the LSTM’s final hidden state. We reused the Part 1 five-fold split (seed 3) on all 25,000 labelled IMDb training reviews. Each fold used 18,000 reviews for fitting, 2,000 for choosing the epoch with the lowest validation loss out of 20, and 5,000 for evaluation. Every review contributed one held-out prediction to the reported accuracy. We used these same folds to select hyperparameters, so the winning score is a cross-validation result. The official test split played no part in this search. [LSTM encoder](lstm.py) · [Experiment code](../project.py)

## What mattered before tuning?

We first held the architecture fixed at one forward LSTM layer with 20 hidden units and varied the 300-dimensional embeddings, whether they could update during training, and how many reviews we used for fitting.

| Starting word vectors | Updated during training? | Fitting reviews per fold | Accuracy | Macro-F1 | Run ID in the [W&B workspace](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/workspace?nw=nwuserangelosntousis) |
|---|---|---:|---:|---:|---|
| GloVe | No | 18,000 | 88.06% | 0.8805 | `7e5avnkt` |
| word2vec | No | 18,000 | 88.82% | 0.8882 | `3cwzsntd` |
| fastText | No | 18,000 | **89.41%** | **0.8941** | `7h58qpb5` |
| fastText | No | 9,000 | 86.87% | 0.8687 | `2lb49t0l` |
| fastText | Yes | 18,000 | 84.81% | 0.8481 | `zxn1n2hg` |

- **Embedding source.** Frozen fastText led word2vec by 0.59 percentage points and GloVe by 1.36.
- **Embedding updates.** Freezing fastText beat fine-tuning it on every fold, by **4.60 points** overall. This motivated keeping fastText frozen throughout the hyperparameter search. We tested random initialisation for pooling only.
- **Training data.** Doubling the fitting set from 9,000 to 18,000 reviews per fold gained **2.54 points**, with a gain on every fold. The validation and evaluation sets were identical in both runs.

## Which hyperparameters helped?

**Classifier dropout** hides features in the classification head, including some of the final LSTM output. **LSTM dropout** hides some of the first recurrent layer’s outputs before they enter the second layer. We tested them separately so a change in accuracy could be traced to its location.

### First sweep — architecture, classifier dropout and optimisation

We ran all **32 configurations** in a five-parameter grid, with zero dropout inside the LSTM. Each row below compares 16 pairs of runs that differ only in the listed setting. The counts show how consistently a choice helped across the grid. [Original W&B sweep](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/sweeps/enm86hiq/workspace)

| Setting | Values compared | Higher accuracy in the matched pairs |
|---|---|---|
| AdamW learning rate | 0.0003, 0.001 | 0.001 won 15 of 16 |
| Classifier dropout | 10%, 50% | 10% won 15 of 16 |
| Training batch size | 64, 128 | 64 won 12 of 16 |
| Hidden units per LSTM layer | 32, 64 | 64 won 13 of 16 |
| LSTM layers | 1, 2 | One layer won 11 of 16 |

The **best individual run scored 90.29%** with two 32-unit layers, classifier dropout 10%, learning rate 0.001 and batch size 64. Although 64 units usually helped in matched pairs, the highest-scoring combination here used 32. [Best original run](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/runs/su7oxhim)

### Follow-up — width and dropout between LSTM layers

We kept two layers, classifier dropout 10%, learning rate 0.001 and batch size 64. We added **seven configurations** and reused the 32- and 64-unit zero-dropout controls from the original sweep. The table contains all nine width/dropout combinations. Together, the original 32 runs and the seven follow-ups required **39 configurations × 5 folds = 195 fits**. [32- and 64-unit sweep](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/sweeps/qxgmqmdx/workspace) · [128-unit sweep](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/sweeps/nnrg8v32/workspace)

| Hidden units per layer | 0% LSTM dropout | 10% LSTM dropout | 50% LSTM dropout |
|---|---:|---:|---:|
| 32 | 90.288% | **90.388%** | 90.272% |
| 64 | 89.920% | 87.760% | **90.388%** |
| 128 | 90.080% | 90.000% | 90.068% |

At **32 units**, 10% LSTM dropout gained **0.100 percentage points** over zero dropout. That is 25 additional correct predictions overall, with gains on two of five folds. At **64 units**, 50% dropout worked best. The low score at 10% included one fold at **80.04%**, while its other four folds scored 88.92–90.10%. At **128 units**, all three settings fell within 0.08 points. These runs support choosing dropout together with width. They give little evidence for a universal dropout rate. [64-unit, 10% dropout run](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/runs/tvbd8yg5)

## Did the LSTM overfit?

The clearest example came from the original **fine-tuned fastText** LSTM. On the fifth fold, training accuracy gained almost 12 points after epoch 4. Validation accuracy stayed near 85% while validation loss more than doubled. The model fit the training reviews increasingly well as its probability estimates on fresh reviews worsened. We evaluated the epoch-4 checkpoint for this fold. [Fine-tuned run](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/runs/zxn1n2hg)

| Fifth fold | Training accuracy | Validation accuracy | Validation loss |
|---|---:|---:|---:|
| Epoch 4, selected checkpoint | 87.96% | 85.45% | 0.3774 |
| Epoch 20 | 99.83% | 85.30% | 0.9585 |

## Did the LSTM improve on embed-and-pool?

**Yes, on these five folds.** The selected 32-unit LSTM scored **90.388%** against **89.724%** for fine-tuned fastText mean pooling. It classified **166 more reviews correctly**, a gain of **0.664 percentage points**. Across the five evaluation folds of 5,000 reviews, the gains were 7, 49, 28, 69 and 13 correct predictions. These are observed gains on the folds used for tuning. [Pooling run](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/runs/zr17jbgj) · [Task 3 report](../task3_embed_and_pool/TASK3.md)

The original 20-unit LSTM also beat pooling when both used the same frozen vectors, by 4.97 points with GloVe, 4.07 with word2vec and 2.68 with fastText. When both fine-tuned fastText, pooling led by 4.91 points. The encoder changes both how word order is handled and how a review is compressed, so these comparisons alone cannot attribute the gains to word order. The Part 1 classifier with weighted TF-IDF bigrams remains ahead at **92.25%**. [Part 1 predictions](../../results/final_predictions/index.csv)

## Reproducing the runs on Snellius

Run from the repository root with the project environment in `.venv`, the IMDb dataset cached, the vectors in `embeddings/` and W&B configured for `nlp1-athena/Part 2`.

- **Original five comparisons.** Run `sbatch part2/task4_recurrent_encoder/snellius.sbatch`. Array indices 0–2 use frozen GloVe, fastText and word2vec. Index 3 fine-tunes fastText. Index 4 uses half the fitting reviews. The original results came from jobs `29497316` and `29497883`. [Baseline job](snellius.sbatch)
- **Hyperparameter search.** Each array task trains one configuration on one A100. Register each YAML once, then pass the sweep path W&B prints to the agent job. Quote the path because the project name contains a space. [Sweep agent job](lstm_sweep.sbatch)

| Grid | Array argument | Completed Slurm job |
|---|---|---|
| [Original 32 configurations](lstm_sweep.yaml) | `--array=0-31%2` | `29555450` |
| [Dropout at 32 and 64 units](lstm_dropout_sweep.yaml) | `--array=0-3%2` | `29559613` |
| [Dropout at 128 units](lstm_dropout_128_sweep.yaml) | `--array=0-2%2` | `29559614` |

For example, to reproduce the original grid in a new sweep, replace `SWEEP_ID` with the ID printed by the first Python command.

```bash
source .venv/bin/activate
export WANDB_BASE_URL=https://api.forge.coreweave.com
python -m wandb sweep --entity nlp1-athena --project "Part 2" part2/task4_recurrent_encoder/lstm_sweep.yaml
sbatch --array=0-31%2 part2/task4_recurrent_encoder/lstm_sweep.sbatch "nlp1-athena/Part 2/SWEEP_ID"
```

Per-fold predictions and epoch histories are in `results/part2/<run-name>-<run-id>/` on Snellius.
