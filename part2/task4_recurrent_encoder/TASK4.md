# Task 4 — Does reading an IMDb review in order improve sentiment classification?

Nov 20, 2026 · Angelos

Our best LSTM reached **89.41% accuracy** using frozen fastText vectors. Reading the review in sequence improved on mean pooling with every frozen embedding we tried. Once we allowed each encoder its best training setup, fine-tuned mean pooling had a small lead at **89.72%**.

## What we compared

The LSTM reads the 300-dimensional word vectors in sequence. Its final 20-dimensional hidden state represents the review. We used one forward LSTM layer and the same two-layer classification head, dropout and optimiser settings as in Task 3. All five configurations used the same Part 1 folds (seed 3). Each fold reserved 2,000 reviews for choosing the lowest-validation-loss epoch out of 20 and 5,000 for evaluation. The table combines one held-out prediction for each of the 25,000 labelled training reviews. [LSTM encoder](lstm.py) · [Experiment code](../project.py)

| Starting word vectors | Updated during training? | Fitting reviews per fold | Accuracy | Macro-F1 | Run ID in the [W&B workspace](https://forge.coreweave.com/wandb/nlp1-athena/Part%202/workspace?nw=nwuserangelosntousis) |
|---|---|---:|---:|---:|---|
| GloVe | No | 18,000 | 88.06% | 0.8805 | `7e5avnkt` |
| word2vec | No | 18,000 | 88.82% | 0.8882 | `3cwzsntd` |
| fastText | No | 18,000 | **89.41%** | **0.8941** | `7h58qpb5` |
| fastText | No | 9,000 | 86.87% | 0.8687 | `2lb49t0l` |
| fastText | Yes | 18,000 | 84.81% | 0.8481 | `zxn1n2hg` |

## Which choices mattered, and how much did data size matter?

- **Embedding source.** With frozen vectors and all fitting reviews, fastText led word2vec by 0.59 percentage points and GloVe by 1.36.
- **Embedding updates.** Starting from the same fastText vectors, fine-tuning lowered LSTM accuracy by **4.60 points**. Freezing won on all five folds. Fine-tuning helped pooling in Task 3, so the right choice depended on the encoder. The random-initialisation control was run for pooling only.
- **Labelled data.** We fitted the frozen fastText LSTM on a stratified half of each fitting set, keeping the validation and evaluation reviews identical. Going from 9,000 to 18,000 fitting reviews raised accuracy by **2.54 points**, with a gain on every fold.

## Did the LSTM overfit?

**Yes, especially when we updated fastText.** On the fifth fold, training accuracy rose by nearly 12 points after epoch 4 while validation accuracy stayed close to 85%. Validation loss more than doubled, meaning the model assigned worse probabilities to the true labels even though its fraction of correct decisions barely changed. For the epoch history, open run `zxn1n2hg` in the workspace and select **Logs**.

| Fifth fold | Training accuracy | Validation accuracy | Validation loss |
|---|---:|---:|---:|
| Epoch 4, selected checkpoint | 87.96% | 85.45% | 0.3774 |
| Epoch 20 | 99.83% | 85.30% | 0.9585 |

We evaluated the epoch-4 checkpoint on that fold. Even with this checkpoint selection in every fold, the fine-tuned LSTM finished below its frozen counterpart.

## How far did recurrence take us compared with pooling?

With embeddings **frozen in both models**, switching from pooling to the LSTM gained **4.97 points with GloVe, 4.07 with word2vec and 2.68 with fastText**. With fastText **trainable in both**, pooling led by **4.91 points**. The LSTM can represent word order, but changing encoders also changes how the review is compressed. These comparisons cannot isolate the contribution of word order alone. [Matched pooling results](../task3_embed_and_pool/TASK3.md)

Across the configurations we tried, the best LSTM finished **0.31 points behind** the best pooling model on one five-fold split. We found a clear use for recurrence when word vectors must stay frozen. We found no overall accuracy gain over our strongest pooling model. Other recurrent widths, learning rates and GRUs remain untested.

## Reproducing Task 4 on Snellius

This folder contains the LSTM encoder, the stratified half-data selection and the A100 job that runs all five configurations. The shared runner handles folds, checkpoint selection, predictions and W&B logging. [Task 4 code](lstm.py) · [Slurm job](snellius.sbatch) · [Shared runner](../project.py)

From the repository root on Snellius, with the project environment in `.venv`, the IMDb dataset cached, the Task 1 vectors in `embeddings/` and W&B configured for `nlp1-athena/Part 2`, run the following command.

```bash
sbatch part2/task4_recurrent_encoder/snellius.sbatch
```

Array indices 0–2 run frozen GloVe, fastText and word2vec, respectively. Index 3 fine-tunes fastText and index 4 fits frozen fastText on half the training reviews. Each index runs five folds on one A100. To repeat just the half-data experiment, use `sbatch --array=4 part2/task4_recurrent_encoder/snellius.sbatch`.

The published frozen runs on full data came from job `29497316`, indices 3–5. The fine-tuned and half-data runs came from job `29497883`, indices 0 and 1, respectively. New submissions write Slurm logs to `results/imdb-lstm-<job>_<index>.out` and per-fold predictions and epoch histories to `results/part2/<run-name>-<run-id>/`.
