# Feature representations: IMDb examples

These examples show how different representations can lead a classifier to make different predictions for the same review. They use the team's saved **logistic regression** predictions, a 32,000-token BPE tokenizer, and split seed 0. Each review was held out from the classifier that predicted it; the fold number identifies that held-out partition of the IMDb training split.

All probabilities below are **P(positive)**. Above 50% predicts positive; below 50% predicts negative. These are model outputs, not necessarily calibrated probabilities. The excerpts are short clues; each classifier saw the **full review**.

| Representation | Input to the classifier |
| --- | --- |
| Bag-of-words (BoW) | Unigram token counts; no token order. |
| TF-IDF | Unigram frequencies weighted by how common each token is across training documents. |
| TF-IDF + bigrams + NB | Unigrams and adjacent-token pairs, TF-IDF weighting, and supervised Naive Bayes log-ratio weighting. |

A bigram can preserve local order: `not → funny` differs from `funny → not`. Plain unigram BoW and unigram TF-IDF cannot distinguish these two token sequences from their token counts alone. **TF-IDF does not capture order by itself; the bigram features add it.** BPE pairs are pairs of tokens, which can be word pieces rather than whole words.

## Example 1: negation and positive words

**Review 7367 · fold 1 · true label: negative**

> “It's not funny now.”

The review also says the writer enjoyed some reactions to the film, but criticises its writing and acting. The unigram representations see `not` and `funny` as separate features. The bigram representation can represent `not → funny` as a distinct feature; the project's BPE tokenizer produces these as adjacent tokens in this excerpt.

| Representation | P(positive) | Prediction |
| --- | ---: | --- |
| BoW | 59.00% | Positive ❌ |
| TF-IDF | 78.28% | Positive ❌ |
| TF-IDF + bigrams + NB | 1.39% | Negative ✓ |

The bigram model gets the label right, while both unigram models get it wrong. The phrase is a plausible reason local order could help, but the saved predictions alone do **not** show how much `not → funny` contributed to the decision.

## Example 2: a positive review with negative-looking words

**Review 15483 · fold 1 · true label: positive**

> “the movie's not at all bad”

The review calls the adaptation less exciting than the novel, but concludes that the film is not bad. Negation and the mix of positive and negative statements make this a useful example of context.

| Representation | P(positive) | Prediction |
| --- | ---: | --- |
| BoW | 39.31% | Negative ❌ |
| TF-IDF | 40.87% | Negative ❌ |
| TF-IDF + bigrams + NB | 73.30% | Positive ✓ |

The bigram model predicts the correct label. However, `not` and `bad` are **not adjacent** here: the tokenizer produces `not → at → all → bad`. A bigram alone does not directly encode the complete phrase's negation scope, so this example should not be presented as proof that the model understood it.

## Example 3: extra features can also hurt

**Review 1650 · fold 0 · true label: negative**

> “but not in this movie”

The reviewer praises an actor, while criticising the film. Identifying which opinion applies to the movie requires more context than a single phrase.

| Representation | P(positive) | Prediction |
| --- | ---: | --- |
| BoW | 43.94% | Negative ✓ |
| TF-IDF | 32.07% | Negative ✓ |
| TF-IDF + bigrams + NB | 59.86% | Positive ❌ |

Both unigram models get this review right, while the bigram + NB model does not. More local context can be useful without being sufficient for every review.

## How to interpret the comparison

- These are **selected examples**, not evidence that a representation always handles negation or that its probabilities are calibrated.
- The bigram system changes more than order: it also uses TF-IDF and label-dependent NB weighting. The saved final configurations also differ in classifier regularisation and training settings. Differences therefore cannot be attributed to bigrams alone.
- A controlled test of ordering would hold tokenizer, weighting, classifier and other settings fixed, changing only unigram versus unigram-plus-bigram features. Model coefficients or feature-ablation tests would be needed to investigate individual predictions more closely.
- The saved files contain **out-of-fold predictions on the training split**, not official test-set predictions. These examples were selected after inspecting predictions and should not be used as an independent final evaluation.

**Sources:** [saved prediction index](../results/final_predictions/index.csv); [feature implementation](features.py); [IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb). Review IDs are zero-based indices in the dataset's training split. The example probabilities come from the seed-0 `lr_bow`, `lr_tf-idf` and `lr_tf-idf-bigrams-nb` prediction files.

*AI assistance: Codex helped select and verify these examples and draft this analysis.*
