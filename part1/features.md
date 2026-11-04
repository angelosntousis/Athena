# Task 3: Feature representation (IMDb)

*Angelos · Implementation: [features.py](features.py)*

A tokenizer decides which pieces of a review the model can see. Feature representation decides how to turn those pieces into numbers. For IMDb sentiment classification, we compare **token indicators, bag-of-words counts and TF-IDF**, and ask whether individual tokens or adjacent pairs provide the more useful evidence.

Our strongest result is that **BPE bigram TF-IDF performed best among the configurations tested**, reaching 89.57% test accuracy with logistic regression and 90.20% with a shallow MLP. The interesting finding is not just the winning score: adjacent pieces can resolve ambiguous word fragments, while feature weighting changes how strongly repetition influences a prediction.

## From a review to a feature vector

Suppose a tokenizer produces `good`, `good`, `movie`. Before choosing the numbers, we choose which patterns become columns in the feature matrix:

| Units | Features in this example | What they retain |
| --- | --- | --- |
| Unigrams | `good`, `movie` | Individual token identity |
| Bigrams | `good → good`, `good → movie` | Identity and order of adjacent tokens |
| Unigrams + bigrams | All four features above | Both kinds of evidence |

These are **token** bigrams, not necessarily two-word phrases. BPE can split a word into several pieces, so a pair may reconstruct part of a word or connect neighbouring words. Bigrams preserve local order, but do not preserve the full review's sequence.

Once the columns are chosen, the three representations assign different values to them.

## Presence, repetition or weighted frequency?

| Representation | What a feature value means | Why use it? | Main drawback |
| --- | --- | --- | --- |
| **Token indicators** | 1 if present, otherwise 0 | A repeated phrase cannot dominate merely through repetition | Loses frequency and possible emphasis |
| **Bag-of-words counts** | Number of occurrences | Keeps repetition as evidence | Long reviews and repeated incidental phrases can produce large values |
| **TF-IDF** | Frequency weighted by training-document rarity, then L2-normalised | Controls vector magnitude and gives common features less relative weight | Rare names or fragments can receive high weights; useful repetition can be weakened |

In the example, the unigram `good` receives an indicator value of **1** and a count of **2**. Its TF-IDF value also depends on how widely it appears across the training reviews and on the other features in that review.

Our implementation uses scikit-learn's `TfidfTransformer` with smoothed IDF:

```text
IDF(f) = log((N + 1) / (df(f) + 1)) + 1
```

Here, `N` is the number of training reviews and `df(f)` counts reviews containing the feature. Optional logarithmic TF replaces a nonzero count with `1 + log(count)`. The weighted vector is then divided by its L2 norm. When unigrams and bigrams are combined, they are normalised together as **one vector**.

**Rarity is not sentiment importance.** IDF uses no sentiment labels. The classifier subsequently learns whether a feature supports positive or negative sentiment. A rare feature can still be irrelevant or misleading.

## What we compare, and what stays fixed

Our comparison separates two questions:

1. **Does weighting matter?** Compare indicators, counts and TF-IDF over the same token units and retained vocabulary.
2. **Does local order help?** Compare unigrams, bigrams and their combination under the same weighting and classifier settings.

The tutorial supplies the starting point: indicators and counts, assessed with logistic regression and a single-hidden-layer MLP. We extend it with TF-IDF, adjacent-token features and an analysis of individual predictions.

**Scope of the reported results.** These experiments use [mteb/imdb](https://huggingface.co/datasets/mteb/imdb) and a **1,000-token BPE vocabulary**, following the tutorial's tokenizer setup but training on IMDb training text. They are separate from Task 2's proposed word-level/BPE comparison with larger vocabularies and external unlabelled training text. The findings below apply to the setup actually evaluated; they should not be presented as results for those newer tokenizers. In particular, larger BPE vocabularies may reduce the need for pairs to reconstruct words.

Input features come only from `text`. Sentiment `label` is the prediction target; `label_text` must never enter the feature matrix.

## Hyperparameters and evaluation

| Setting | Values tested | What changes |
| --- | --- | --- |
| `ngram_range` | `(1, 1)`, `(2, 2)`, `(1, 2)` | Individual tokens, adjacent pairs, or both |
| `min_df` | `1`, `2`, `5` | Retains bigrams appearing in at least this many training documents |
| `sublinear_tf` | `False`, `True` | TF-IDF only: raw versus logarithmic frequency |

`min_df` filters **bigrams only**; unigram columns follow the supplied tokenizer vocabulary. Ten occurrences in one review still count as one document. This threshold trades coverage of rare patterns against feature-space size and potential noise.

We assessed **28 meaningful feature configurations** with each classifier using five shared cross-validation folds. Within every fold, the tokenizer, bigram vocabulary and IDF were fitted on that fold's training reviews only. Accuracy was the selection metric. LR used `C=1`; the MLP used one 32-unit ReLU layer, dropout 0.1 and a fixed budget of 15 epochs.

After CV selection, settings were frozen and models refitted on **24,903 cleaned training reviews**. Final evaluation used all **24,678 official test reviews** in this dataset version. No exact cleaned train/test overlap was found; near-duplicates and movie-level grouping were not assessed. Test results did not change model settings.

## What the results show

| Representation selected by CV | LR test accuracy | Shallow MLP test accuracy |
| --- | ---: | ---: |
| Indicators | 89.00% | 89.55% |
| Counts | 88.38% | 89.10% |
| TF-IDF | **89.57%** | **90.20%** |

All six selected configurations used **bigrams only**. For indicators, counts and TF-IDF respectively, selected `min_df` values were **1, 1, 2 for LR** and **2, 5, 2 for MLP**. Both TF-IDF selections used logarithmic TF.

Because these selected settings differ, the table compares complete configurations. A separate control holds the LR settings and bigram vocabulary (`min_df=2`) fixed: **indicators score 88.94%, counts 88.35%, and logarithmic TF-IDF 89.57%**. TF-IDF's advantage therefore remains when the available features are identical. Its frequency weighting and normalisation produce a more effective input representation in this comparison.

### The larger effect comes from adjacent pieces

With LR and logarithmic TF-IDF, moving from unigrams to bigrams raises accuracy from **83.58% to 89.57%**, a **5.99 percentage-point gain**. Bigrams correct 2,321 unigram errors but lose 844 unigram successes.

This benefit appears across review lengths:

| Review length (whitespace tokens) | Test reviews | Unigram accuracy | Bigram accuracy | Gain (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| Fewer than 100 | 3,139 | 83.34% | 90.28% | +6.94 |
| 100–249 | 14,465 | 83.57% | 89.72% | +6.15 |
| 250–499 | 5,239 | 83.81% | 88.89% | +5.08 |
| 500+ | 1,835 | 83.38% | 89.05% | +5.67 |

The gain is not confined to long reviews. These groups were inspected after evaluation as descriptive analysis; they did not select models. They also cannot establish whether subword reconstruction or local context is the main cause of the improvement.

Combining unigrams and bigrams scored **89.38%**, slightly below bigrams alone at **89.57%**. More features did not help this configuration. Pairs already carry some information about their constituent tokens, joint normalisation redistributes magnitude between the two feature blocks, and fixed `C=1` need not be optimal for every space. These are possible explanations, not separately tested causes.

### The improvement has a computational cost

On the 24,903 training reviews, using the frozen tokenizer and bigram `min_df=2`:

| Units | Dimensions | Median distinct active features per review | Zero entries |
| --- | ---: | ---: | ---: |
| Unigrams | 1,000 | 200 | 77.59% |
| Bigrams | 189,697 | 315 | 99.78% |
| Unigrams + bigrams | 190,697 | 515 | 99.67% |

The bigram space is much larger, yet a typical review activates only 315 of its columns. Sparse CSR matrices make this manageable by storing nonzero entries rather than the full matrix. More dimensions still mean more classifier parameters and more rare patterns to estimate.

For a fixed vocabulary, all three weighting methods share the same nonzero positions. **High sparsity is a storage property, not an explanation of predictive quality.**

## What happens inside a prediction?

Aggregate scores show which configurations perform well. Individual reviews help explain how their evidence differs. The following examples use LR, with `min_df=2` for bigrams; IDs are zero-based test rows.

For LR, we verified the exact score decomposition:

```text
positive_logit = intercept + sum(feature_value × learned_coefficient)
P(positive) = sigmoid(positive_logit)
```

A positive contribution pushes towards positive sentiment; a negative contribution pushes towards negative sentiment. Contributions add in **log-odds**, not percentage points. The probabilities below are model outputs; their calibration has not been assessed.

### Bigrams distinguish “worn” from a misleading fragment

**Review 16026, positive:** a fan says they have “worn out the tape”. Our BPE tokenizer splits `worn` into `▁wor + n`, while `worst` becomes `▁wor + st`.

| Representation | P(positive) | Prediction |
| --- | ---: | --- |
| Unigram TF-IDF | 41.13% | Negative — wrong |
| Bigram TF-IDF | 72.56% | Positive — correct |

In the unigram model, the shared piece `▁wor` contributes **−0.3953** to positive log-odds. In the bigram model, the more specific `▁wor → n` contributes only **−0.0042**. The pair distinguishes a benign word fragment from other uses of the same piece.

This is concrete evidence for the value of lexical specificity. It is only part of the prediction difference: the other features, learned coefficients and intercept also change between models.

### TF-IDF reduces a misleading repetition

**Review 22331, positive:** the reviewer asks about a hoped-for sequel. The pair `qu → el`, appearing twice, has a negative learned association.

```text
Counts:  2 × (−0.4283)        = −0.8565
TF-IDF:  0.073646 × (−1.2662) = −0.0933
```

Counts predicts **40.27% positive**, incorrectly; logarithmic TF-IDF predicts **53.08%**, narrowly correctly. Logarithmic frequency and L2 normalisation, together with IDF weighting, change the input scale and reduce this pair's negative contribution.

The TF-IDF coefficient is more negative, but its contribution is smaller because its input value is much smaller. **Inspect feature value × coefficient, rather than the coefficient alone.** This comparison illustrates the combined weighting scheme; it does not isolate the effect of IDF by itself.

### Better overall does not mean better on every review

**Review 15686, positive:** a favourable review describes faults the film *could have had*, then explains that it avoided them. Indicators predict **74.20% positive**, correctly; TF-IDF predicts **39.61%**, incorrectly. Repeated `have → been` and `could → have` pairs, alongside embarrassment-related fragments, contribute negatively in the TF-IDF model. Local features do not explicitly encode the wider counterfactual meaning. Binary presence caps repetition, and its separately learned coefficients produce a better balance here.

**Review 17893, positive:** praise for the film appears alongside disappointment about a closed video store. Counts is correct at **53.33% positive**; TF-IDF is wrong at **46.36%**. Both are near the decision boundary. Neither representation explicitly distinguishes sentiment about the film from sentiment about the store.

These failures explain why weighting alone cannot solve negation scope, counterfactuals or identifying the target of an opinion.

## What we take forward

For the tested 1,000-token BPE setup, **logarithmic TF-IDF over bigrams is the strongest feature configuration assessed**. The evidence supports retaining adjacent-token information and controlling the influence of repetition. It does not establish that TF-IDF or bigrams will be best with every tokenizer or classifier.

The length groups and review examples are descriptive evidence, not significance tests. The existing CV grid includes paired Bayesian **LR-versus-MLP comparisons within each configuration**, using a ROPE of ±0.5 percentage points. Those probabilities do not establish superiority between representations. Further model changes motivated by inspected test results need fresh held-out evidence for an independent final evaluation.

The implementation remains reusable: [features.py](features.py) accepts a trained tokenizer and returns sparse matrices for all three representations and unit choices. Fit it on training text, then reuse `transform()` for validation/test. If the team adopts Task 2's newer tokenizers, evaluate the representation choices again under that setup before transferring these conclusions.


*AI assistance: Codex helped implement the encoders, run experiments and draft the analysis.*
