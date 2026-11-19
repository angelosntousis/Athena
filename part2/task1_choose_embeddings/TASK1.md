# Task 1: Do pretrained word embeddings fit our IMDb text?

Nov 20, 2026 · @Giovanni

> **This folder is ours.** Everything in `part2/task1_choose_embeddings/` was added by us for Task 1. The original course files are directly in `part2/`.

## The short version

We compared three pretrained word embeddings: **GloVe**, **fastText** and **word2vec**. Each model was trained on text that was split into words in a slightly different way. So we split IMDb reviews once, with part 1's cleaning and splitting, and then rewrite a few tokens into each model's own spelling (for example *didn't* → `didn` `'t` for fastText).

**After that, all three models know about 99.8% of the tokens in IMDb.** Coverage therefore cannot decide between them, so we keep all three. Task 2 looks at what their vectors know, and Tasks 3–4 run every experiment with all three models to see which works best.

---

## Three terms used everywhere

- **Word embedding:** a list of 300 numbers (a *vector*) for each word, learned from billions of words of text. Words used in similar contexts get similar vectors.
- **Token:** one unit of text after splitting, such as `did`, `n't` or `!`.
- **Coverage:** the share of our tokens that the model has a vector for. A token without a vector gets one shared "unknown" vector, so its meaning is lost.

## The three candidates

We picked one model from each of the three main families. All have 300 dimensions, so dimension does not affect the comparison.

| | GloVe 2024 | fastText crawl | word2vec |
|---|---|---|---|
| Trained on | Wikipedia + news (Gigaword), 2024 | Web pages (Common Crawl) | Google News, 2013 |
| Words in the model | 1.29 million, all lowercase | 2.0 million, upper and lowercase | 3.0 million, upper and lowercase; 2.07 million of them are phrases like `New_York` |
| File | `glove.2024.wikigiga.300d.zip` | `crawl-300d-2M.vec.zip` | `word2vec-google-news-300.gz` |

**Note:** each model differs in *both* its training method and its training text. So we are comparing three **released models**, not three methods.

---

## Step 1: How we split the text

**We reuse part 1's text cleaning and splitting, but not part 1's vocabulary.**

Part 1's tokenizers (word-level and BPE) did three things:

1. **Clean the text.** Remove `<br />`, fix broken apostrophes and characters, lowercase.
2. **Split it into words.** *didn't* → `did` `n't`; punctuation becomes its own token.
3. **Map the words to a vocabulary we trained ourselves** (32k words, or BPE pieces like `##yp`).

We keep steps 1 and 2 because they were designed for IMDb in part 1. We drop step 3 because a pretrained model comes with its own vocabulary:

- **BPE pieces** such as `##yp` have no vector in a word embedding.
- **Our 32k word list** would turn rarer words into "unknown" before the model could look them up.

**Is this tokenizing or just preprocessing?** It is tokenizing. Tokenizing means splitting text into tokens; mapping tokens to vectors is a separate step. With pretrained embeddings the vocabulary is fixed by the model, so the only part of tokenization we control is how we split.

## Step 2: Each model was trained on text split differently

To find a word, we must write it the way the model saw it during training. Looking inside the three models shows that they differ:

| Our tokens | GloVe has | fastText has | word2vec has |
|---|---|---|---|
| *didn't* → `did` `n't` | `did` `n't` ✅ the same | `didn` `'t` ❌ (no `n't` at all) | `didn't` whole; `n't` only as a rare, unreliable entry |
| *can't* → `ca` `n't` | the same ✅ | `can` and `'t`; its `ca` is a different word (*circa*) | `can't` whole, and `can`, `not` |
| Punctuation `. , ! ?` | ✅ | ✅ | ❌ **no punctuation at all** |
| Numbers `10`, `1990s` | ✅ as written | ✅ as written | ❌ written with `#`: `10` → `##`, `1990s` → `####s` |
| *cannot* | ❌ missing (its tokenizer split it into `can` `not`) | ✅ | ✅ |

## Step 3: Our fix, one splitting with a few per-model rewrites

**Every model gets the same split text. Just before the lookup, a few tokens are rewritten into each model's own spelling.**

Why this way:

- **Without rewriting,** fastText would miss `n't` 33,559 times. That is most of the negations, the most important words for sentiment.
- **With a separate tokenizer per model,** the comparison would be unfair: we could not tell whether a model wins because of its vectors or because of its tokenizer.

We only rewrite a token if **all three** of these hold:
1. It is a **spelling difference**, not a missing word.
2. It affects **negation** or a **noticeable share** of tokens.
3. The rewrite **keeps the meaning** of the text.

| Model | Rewrites |
|---|---|
| GloVe | `cannot` → `can` `not` |
| fastText | `did n't` → `didn 't` (same for *is, could, was…*), `ca n't` → `can 't`, `wo n't` → `won 't` |
| word2vec | `n't` → `not`, `ca n't` → `can not`, `wo n't` → `will not`; numbers with 2+ digits → `#`; **punctuation removed** (it has no vectors, and turning 13% of every review into the "unknown" vector would blur all reviews together) |

Example: *"I didn't like it. Can't watch!! 1990s"*

| | Tokens the model receives |
|---|---|
| GloVe | `i did n't like it . ca n't watch ! ! 1990s` |
| fastText | `i didn 't like it . can 't watch ! ! 1990s` |
| word2vec | `i did not like it can not watch ####s` |

**What we did not fix, and why:**
- **Hyphenated words** (*low-budget*): only fastText stores them whole, and every part is already known.
- **Ratings like `8/10`**: only GloVe stores them whole, and they are rare (about 1,700 in 25,000 training reviews).

## Step 4: Coverage, before and after the rewrites

Measured on the 25,000 training reviews (6.85 million tokens):

| % of tokens the model knows | GloVe | fastText | word2vec |
|---|---|---|---|
| Before the rewrites | 99.82 | 99.35 | 86.54 |
| **After the rewrites** | **99.84** | **99.84** | **99.76** |
| Reviews with more than 5% unknown tokens (after) | 0.03 | 0.03 | 0.08 |
| Occurrences of sentiment-lexicon words that the model knows | 100 | 100 | 100 |

(For word2vec, the "after" figure counts only the tokens it can see: punctuation is removed.)

**What is still unknown:** almost only film names and character names (*hackenstein*), reviewers' misspellings (*dont*, *didnt*) and *gonna*. These are words the model never saw, not spelling differences, so no rewrite can recover them.

## Conclusion

**All three models can be fitted to our text, and all three know about 99.8% of it.** Coverage does not separate them, so we keep all three: Task 2 looks at what their vectors know, and Tasks 3–4 compare them by running every experiment with all three models.

One property that coverage hides: **word2vec cannot see punctuation at all.** That matters little when word vectors are averaged (Task 3), but it does matter for a recurrent encoder (Task 4), where `!`, `?` and `...` carry signal.

---

## For Tasks 3–4: how to use the embeddings

The vectors for every IMDb word (train and test) are saved per model in `embeddings/imdb_<model>.txt.gz`. They load with the tutorial's own class, and `ImdbTokenizer` does our splitting and rewrites:

```python
from part2.clslib import PretrainedEmbeddings
from part2.task1_choose_embeddings.embedding_check import ImdbTokenizer

emb = PretrainedEmbeddings("embeddings/imdb_fasttext.txt.gz", 300)   # run from Athena/
tokenizer = ImdbTokenizer(emb, "fasttext")                            # "glove", "fasttext" or "word2vec"

tokenizer.tokens("I didn't like it.<br />Overrated!")
# ['i', 'didn', "'t", 'like', 'it', '.', 'overrated', '!']
```

`tokenizer` can be passed directly to `part2.experiment.train(...)` and `predict_logprobs(...)`, together with `emb`.

Things to know:
- **Use `ImdbTokenizer`, not the tutorial's `BasicTokenizer`.** Its `remove_stopwords=True` deletes *not*, *no* and *nor* (they are in NLTK's stopword list), and `remove_punct=True` turns *don't* into `don` `t`.
- **Unknown words** (0.16–0.24% of tokens) get the average of all vectors in the file. That is the tutorial loader's default.

The embedding files are not in git (`embeddings/` is in `.gitignore`). To create them, download the three files below into `Athena/embeddings/`, then run the `export` command.

---

## Questions the TA will probably ask

**"Is this tokenizing or just preprocessing?"**
Tokenizing means splitting text into tokens. With pretrained embeddings the vocabulary is fixed by the model, so the only part we control is how we split. We kept part 1's splitting and rewrote the few tokens each model spells differently.

**"Why one tokenizer and not one per model?"**
So that the only difference between the models is their vectors. With a different tokenizer per model, we could not tell which of the two caused a difference.

**"Why not reuse part 1's BPE tokenizer?"**
BPE pieces like `##yp` have no vectors in a word embedding, and part 1's 32k vocabulary would mark rarer words as unknown before the model could look them up.

**"Why didn't you just pick the model with the best coverage?"**
Once each model gets text in its own spelling, all three know 99.8% of our tokens and 100% of the sentiment-lexicon words. Coverage does not separate them, so we kept all three and compared them in Tasks 3–4.

**"What happens to words the model doesn't know?"**
They get one shared vector, the average of all vectors (the tutorial loader's default). After the rewrites this affects at most 0.24% of tokens, almost all of them names and misspellings.

---

## How to reproduce

From `Athena/`, after downloading the three files into `embeddings/`:

| Command | What it prints |
|---|---|
| `python part2/task1_choose_embeddings/embedding_check.py conventions` | How each model spells contractions, punctuation and numbers (Step 2) |
| `python part2/task1_choose_embeddings/embedding_check.py coverage` | Coverage before and after the rewrites (Step 4) |
| `python part2/task1_choose_embeddings/embedding_check.py export` | Writes `embeddings/imdb_<model>.txt.gz` for Tasks 2–4 |

Downloads:
- GloVe: `https://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.2024.wikigiga.300d.zip` (the `nlp.stanford.edu` link in the tutorial was down)
- fastText: `https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip`
- word2vec: `https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz`
