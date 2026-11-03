# Task 2: Tokenisation choices (IMDb)

Nov 3, 2026 · @Giovanni

We compare two tokenisers on IMDb: a word-level tokeniser and our own BPE. The comparison answers one question we can defend to the TA, and each tokeniser's vocabulary size is chosen by cross-validation.

## Why IMDb and not SST-2

We chose IMDb because cross-validation, which the project requires, gives inflated scores on SST-2. Both datasets are on the course list ([SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb)), and both classify movie reviews as positive or negative.

**The problem:** SST-2's 67,349 training examples are mostly fragments of the same sentences; it contains, for example, both *the movie is not good at all* and *not good at all*. In cross-validation, a fragment can end up in the test part while its full sentence was used for training, so the model is tested on text it has already seen.

**What we checked**, on one 80/20 split of SST-2's training data:

- 85.5% of the test-part examples appear word for word inside a training-part example.
- The tutorial's logistic regression scored 90.1% on the test part, but only 81.4% on SST-2's official validation sentences, which it had never seen. Long test-part examples (15+ words) scored even higher (92.6%), so the gap comes from the overlap, not from sentence length.

On SST-2, cross-validation therefore overstates accuracy by about 9 to 11 points. Fixing it (grouping fragments by sentence) would be extra work, and SST-2's test labels are hidden anyway.

**IMDb avoids all of this:** every review is independent, and it has 25,000 labelled reviews for training, 25,000 for a final test, and 50,000 unlabelled reviews to train our tokenisers on. The trade-off: reviews are long (median 174 words), so encoding and training take longer than with SST-2's short sentences.

## The two tokenisers

A tokeniser cuts a review into pieces (tokens). Its vocabulary becomes the feature space of every representation in Task 3, so this choice shapes what our models can see.

| Tokeniser | Unit | Learned from | Vocabulary | With 32,000 tokens, how it cuts *An unwatchable, overhyped mess. I didn't like it.* |
| --- | --- | --- | --- | --- |
| **A. Word-level** | Whole words | 50k unlabelled IMDb reviews | Chosen from 8,000 to 64,000 (see Hyperparameters) | `an` `unwatchable` `,` `[UNK]` `mess` `.` `i` `did` `n't` `like` `it` `.` |
| **B. BPE** (tutorial algorithm) | Subword pieces | 50k unlabelled IMDb reviews | Chosen from 8,000 to 64,000 (see Hyperparameters) | `an` `unwatchable` `,` `over` `##h` `##yp` `##ed` `mess` `.` `i` `did` `n't` `like` `it` `.` |

- **A** keeps the most frequent words; any other word becomes `[UNK]` (unknown). Here *overhyped* was not frequent enough, so its meaning is lost.
- **B** starts from characters and repeatedly merges the most frequent adjacent pair. Frequent words end up whole; rare words are spelled from smaller pieces, so nothing is ever unknown. `##` marks a piece that continues a word.

## The question we answer

Comparing A and B answers one question: **are whole words enough, or do we need word pieces?**

### Why tokenisers are worth comparing at all

The tokeniser decides what the "words" of our models are. In Task 3, every feature is the presence or count of one token. In Task 4, logistic regression and the MLP learn from those features, for example one weight per token in logistic regression. So a model can only learn from what the tokeniser gives it:

- If a sentiment word becomes `[UNK]` (unknown), the model never sees it.
- If a word is cut into fragments that also appear in unrelated words, the model can learn little from them.

The tokeniser therefore puts a ceiling on what every later model can learn. That is why we test it instead of assuming.

### Whole words or word pieces? (A vs B)

**What we compare:** A (word-level) against B (BPE).

**What is identical:** the training text (the same 50k reviews), the cleaning, the splitting into words and punctuation, and the number of vocabulary slots.

**The only difference** is how each one spends its vocabulary slots:

- **A** spends every slot on a whole word. It keeps the most frequent words; any other word becomes `[UNK]`, and its meaning is lost.
- **B** spends some slots on word pieces. It can spell any word, even one it never saw, out of pieces. The price is fewer slots for whole words, and some pieces (like `##yp`) mean nothing on their own.

Example with 32,000 slots: *overhyped* is not frequent enough to get its own slot. A turns it into `[UNK]`, so the model loses it. B spells it `over` `##h` `##yp` `##ed`, so at least `over` reaches the model.

**Why this question matters:** handling rare and unseen words is the whole reason BPE exists, and it is the method the tutorial taught. Our comparison tests whether that advantage actually shows up on our data, or whether simple whole words do just as well.

**How we answer it:** we compare A and B at every candidate vocabulary size (see Hyperparameters). We expect B's advantage, if any, to be largest when the vocabulary is small, because that is when A loses the most words. With 8,000 slots, 6% of the tokens in our training reviews would become `[UNK]` under A; with 32,000 slots, only 1.5%.

## Why we do not reuse the tutorial's tokeniser

The tutorial's tokeniser was built for news articles and is far too small for our reviews. So we keep its method (BPE) but retrain it on our own data with settings suited to IMDb, instead of reusing the saved tokeniser.

**What the tutorial's tokeniser is.** In the tutorial we trained a BPE tokeniser on AG News (120,000 news articles), with a vocabulary of 1,000 tokens and `min_frequency=2`. It split text only at spaces and did not lowercase.

**Its four problems for IMDb, and what we do instead:**

| Property | Tutorial tokeniser | Why it is a problem for movie reviews | What we do instead |
| --- | --- | --- | --- |
| Training text | News articles (AG News) | BPE gives whole-word slots to the words frequent in its training text: news words like *stocks* or *minister*. Review words like *cinematography*, *overacted* or *rewatchable* are rare in news, so they get cut into fragments. | Train on 50,000 IMDb reviews, the same kind of text we classify. |
| Vocabulary size | 1,000 tokens | Far too few for 230-word reviews: most words, including most sentiment words, get split into several pieces. A piece like `##able` appears in both *watchable* and *unwatchable*, so a bag-of-words model cannot learn which sentiment it signals. | Test candidate sizes from 8,000 to 64,000 and pick the best (see Hyperparameters). |
| Upper and lower case | Kept as written | *Good*, *good* and *GOOD* become three separate tokens. The evidence for one word is split across three features, each seen less often. | Lowercase everything. |
| Where text is split | At spaces only | Punctuation stays glued to words: *mess*, *mess.* and *mess,* become three different tokens. | Split punctuation off as separate tokens (see Text cleaning). |

**What we keep from the tutorial:** the BPE algorithm itself, which is our tokeniser B. This also lets our comparison test the tutorial's method against the simplest alternative, whole words.

## Training data for our tokenisers

Tokenisers A and B are trained **only on the `unsupervised` split of [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)**: 50,000 reviews with no sentiment label.

| Split | Reviews | Labelled? | Used for |
| --- | --- | --- | --- |
| `train` | 25,000 | Yes | Cross-validation of all models |
| `test` | 25,000 | Yes | One final check of the best models |
| `unsupervised` | 50,000 | No | Training tokenisers A and B |

Why this split:

- **No leakage.** These reviews are neither in `train` nor in `test`. The tokeniser never sees a review we evaluate on, so we can train it once and reuse it unchanged in every cross-validation fold.
- **Same domain.** They are IMDb movie reviews, written like the reviews we classify.
- **Large enough.** After cleaning they contain about 13.8 million words and 103,000 distinct words. That is far more text than even our largest candidate vocabulary (64,000 tokens) needs, so the brief's concern (*your dataset may be too small to train a tokeniser*) does not apply.
- **Labels are not needed.** A tokeniser learns only which pieces are frequent, so unlabelled text is exactly what it needs.

We do not add other corpora (such as AG News or Wikipedia): they would add text from other domains to a corpus that is already large enough.

## Text cleaning and pre-tokenisation

Tokenisers A and B share the same cleaning and pre-splitting, so the only difference between them is whole words vs pieces. The same cleaning is applied to every review in every split.

1. **Remove HTML line breaks.** About 59% of `train` reviews contain `<br />` tags, which are leftover web formatting. We replace them with a space.
2. **Normalise and lowercase.** Unicode normalisation (NFKC), curly apostrophes (’) turned into straight ones ('), then lowercasing. *Good* and *good* should be one feature: case rarely changes sentiment, and merging case variants frees vocabulary slots for other words.
3. **Pre-split into words and punctuation.** Punctuation becomes its own token (`mess.` → `mess` `.`), so *mess* is one feature wherever it appears.
4. **Split clitics.** `didn't` → `did` `n't`, `kubrick's` → `kubrick` `'s` (the standard Penn Treebank convention). This gives every negation, from *didn't* and *isn't* to *wasn't*, one shared `n't` token: a strong, frequent sentiment signal for our bag-of-words models.

Two deliberate changes from the tutorial: it pre-splits on spaces only (punctuation stays attached), and it does not lowercase.

## Hyperparameters: candidates and how we choose

We do not fix the vocabulary size in advance. We test four candidate sizes for both A and B, and let cross-validation pick the best one for each tokeniser. The remaining settings are fixed, each for a stated reason.

### What we tune: vocabulary size

The vocabulary size is the number of distinct tokens a tokeniser knows. It is also the number of features every representation in Task 3 will have.

| Candidate size | Share of tokens in our training reviews kept as whole words by A | Tokens A turns into `[UNK]` |
| --- | --- | --- |
| 8,000 | 94.0% | 6.0% |
| 16,000 | 96.8% | 3.2% |
| 32,000 | 98.5% | 1.5% |
| 64,000 | 99.3% | 0.7% |

Why these four values:

- **They span the whole useful range.** At 8,000, A loses 6% of all tokens; at 64,000 it keeps nearly every word. Only 63,737 distinct words appear at least twice in the unlabelled reviews, so 64,000 is roughly "every repeated word" and larger sizes add nothing for A.
- **Each candidate doubles the previous one,** so we see the trend without testing every value in between.
- The tutorial's 1,000 is left out on purpose: at that size most words are split into fragments (see the previous section).

### What we keep fixed, and why

| Setting | Fixed value | Why we do not tune it |
| --- | --- | --- |
| `min_frequency` (how often a word or pair must appear to enter the vocabulary) | 2, as in the tutorial | It only has an effect when fewer words pass the threshold than there are vocabulary slots. 40,246 words appear at least 5 times, so for every candidate up to 32,000 the size limit is reached first: raising the threshold to 5 would not change the vocabulary. A value of 2 removes the 39,481 words that appear only once (mostly typos and rare names), which no model could learn from anyway. |
| Cleaning and splitting rules | As in Text cleaning | They follow from how sentiment is expressed (negation, punctuation, case), not from a trade-off to measure. Keeping them fixed means vocabulary size is the only setting that varies. |

### How we choose

1. Train A and B at each of the four sizes on the unlabelled reviews: 8 tokenisers in total.
2. For each of the 8, run 5-fold cross-validation on the `train` split.
3. For each tokeniser, keep the size with the highest mean accuracy. If several sizes are practically equivalent (their difference falls inside the ROPE of Task 5), keep the smallest: fewer features means faster training and less memory.
4. The chosen versions of A and B are used for the rest of the project (Tasks 3 to 5). The `test` split is never used to choose anything.

The same 8 runs also answer our question, by comparing A and B at each size.

## What we will measure, and what we expect

We judge the tokenisers in two ways: what they do to the text, and what they do to classification accuracy. The measurements are reported in Task 5 (evaluation).

**What they do to the text** (measured on the `train` reviews):

- `[UNK]` rate: the share of tokens that are unknown (only A can produce them).
- Tokens per word: how much a tokeniser fragments the text (1.0 means it never splits a word).
- Share of words split into two or more pieces.
- A few hand-picked examples of sentiment words and negations, like the one in the first table.

**What they do to accuracy:**

- Choosing the vocabulary size: the 40 cross-validation runs described under Hyperparameters.
- Afterwards, the chosen tokenisers are used across the representations of Task 3 and the models of Task 4. Every combination gets 5-fold cross-validation on `train`, with accuracy as the metric (the two classes are exactly balanced, 12,500 reviews each).
- Differences are judged with the Bayesian comparison and ROPE of Task 5.

**What we expect** (hypotheses we will confirm or reject):

- **A vs B:** B beats A at small vocabulary sizes, where A loses many words, and the gap shrinks as the size grows. At the chosen size we expect a small difference, because a 230-word review contains many clues besides any single rare word. If A and B end up practically equivalent, A is the better choice for simple models: whole words are easier to interpret.

## Limitations we state upfront

- **Lowercasing loses emphasis.** *AWFUL* and *awful* become the same token. We accept this to keep the vocabulary compact.
