# Task 2: What do the embeddings know about movie reviews?

Nov 20, 2026 · @Giovanni

> **This folder is ours.** Everything in `part2/task2_explore_embeddings/` was added by us for Task 2. The original course files are directly in `part2/`. Task 1 (fitting the three models to our text) is in `part2/task1_choose_embeddings/TASK1.md`.

## The short version

In Task 1, all three models (GloVe, fastText, word2vec) turned out to know about 99.8% of IMDb's tokens. Here we look at what their vectors actually capture, with two tests:

1. **Which meaning of review words does each model know?** fastText and word2vec know the *review* meaning (*plot* → storyline). GloVe often knows the *news* meaning (*plot* → conspiracy).
2. **What can none of them do?** In all three, opposites look alike (*overrated* is closest to *underrated*), and averaging word vectors cannot express *not*.

**Which model works best is measured in Tasks 3 and 4, where we run every experiment with all three models.** Going in, we expect fastText and word2vec to do better than GloVe (Test 1), and we expect all three to struggle with negation when vectors are averaged (Test 2).

---

## Setup

- **Same words for every model.** Both tests use the 13,119 words that appear at least 20 times in IMDb training reviews and that all three models know. Otherwise a model could look better just by having easier words.
- **Cosine similarity** measures how close two word vectors are: 1 means they point in the same direction, 0 unrelated, −1 opposite. "Closest words" means highest cosine.

---

## Test 1: Which meaning of a review word does the model know?

The 5 closest words to some common review words:

| Word | GloVe (Wikipedia + news) | fastText (web pages) | word2vec (Google News) |
|---|---|---|---|
| *plot* | plots, plotting, plotted, **conspiracy**, story | plots, plotline, plotting, subplots, storyline | plots, plotting, plotted, plotline, storyline |
| *acting* | acted, role, directing, **appointed**, roles | acted, act, directing, overacting, actor | acted, act, directing, behaving, overacting |
| *waste* | wastes, garbage, **disposal**, trash, **toxic** | wastes, wasted, wasting, garbage, trash | wastes, garbage, trash, rubbish, wasted |
| *lame* | **duck**, **excuses**, silly, excuse, stupid | stupid, dumb, lamest, pathetic, unoriginal | silly, inane, pathetic, unoriginal, hackneyed |
| *sick* | dying, ill, tired, **infected**, **patients** | ill, sickness, sickly, tired, miserable | ill, sickness, tired, illness, dizzy |

(**Bold** = a non-review meaning.)

**What it shows:**
- **Embeddings capture useful similarities:** *cheesy* → corny, hokey, campy; *lame* → stupid, pathetic, unoriginal (fastText).
- **The training text decides which meaning a word gets.** GloVe, trained on Wikipedia and news, puts *plot* next to *conspiracy*, *acting* next to *appointed* (as in "acting director"), and *lame* next to *duck*. fastText, trained on web pages, which include reviews and forums, gives the meaning a reviewer intends. So does word2vec, even though it was trained on news.
- **Why it matters for us:** most of a review is ordinary words like these. A model that reads *waste* as pollution gets less sentiment out of *"a waste of time"*. That is why we expect GloVe to do worst in Tasks 3–4.

## Test 2: Two blind spots that all three models share

### 2a. Opposites look alike

Average cosine between pairs of words:

| | Opposites (*good/bad*, *love/hate*, …) | Same meaning (*good/great*, …) | Random words | *overrated / underrated* |
|---|---|---|---|---|
| GloVe | 0.52 | 0.66 | 0.12 | 0.72 |
| fastText | 0.53 | 0.77 | 0.11 | 0.75 |
| word2vec | 0.51 | 0.72 | 0.07 | 0.71 |

- **Opposite words are almost as close as words with the same meaning,** and far closer than random words.
- **The worst case for us: *overrated* and *underrated* are each other's closest word, in all three models** (the `neighbours` command prints this too).
- **Why:** embeddings learn a word from its context, and opposites appear in the same contexts (*"the film was overrated / underrated"*). So the vectors know that both words are about how good a film is, but not which direction.

### 2b. Averaging cannot express *not*

An embed-and-pool encoder (Task 3) represents *"not good"* as the average of the vectors for *not* and *good*. Is that average closer to *good* or to *bad*?

| | Closer to the word itself (wrong) | Example: average of *not* + *good* |
|---|---|---|
| GloVe | 8 of 8 pairs | 0.92 to *good*, 0.68 to *bad* |
| fastText | 8 of 8 pairs | 0.83 to *good*, 0.67 to *bad* |
| word2vec | 8 of 8 pairs | 0.87 to *good*, 0.66 to *bad* |

**In every model, *not good* stays next to *good*.** Averaging loses word order, so it cannot know that *not* flips the word after it. We therefore expect the recurrent encoder in Task 4, which reads words in order, to help most on reviews with negation.

---

## What we take into Tasks 3 and 4

- **We run every experiment with all three models.** Their accuracy there is the task-specific comparison.
- **Expectation 1:** fastText and word2vec beat GloVe, because they know the review meaning of review words (Test 1).
- **Expectation 2:** with averaged vectors (Task 3), all three struggle with negation. A recurrent encoder (Task 4) should help (Test 2b).
- **Expectation 3:** without punctuation, word2vec loses `!`, `?` and `...` (Task 1). That should matter little for averaging (Task 3) but more for the recurrent encoder (Task 4).

---

## How this answers the notebook's four questions

| The notebook asks | Our answer |
|---|---|
| Is it good at representing interesting similarities for our task? | Yes: *cheesy* → corny, hokey; *lame* → stupid, unoriginal. fastText and word2vec also get the review meaning of *plot* and *waste* (Test 1). |
| Is there task-specific evidence that it beats a reasonable alternative? | Test 1 shows a task-specific difference in what the models know: GloVe has news meanings for review words. The accuracy comparison comes from Tasks 3–4, where all three models are run. |
| Do you know its limitations? | Opposites look alike, and each word has one vector whatever its meaning in context (Tests 1–2). |
| Which limitations are troubling for our task? | Sentiment flips are exactly what it misses: *overrated* vs *underrated*, and *not good* vs *good* (Test 2). |

---

## Questions the TA will probably ask

**"Why is *overrated* close to *underrated*?"**
Embeddings learn a word from its context, and the two appear in the same contexts. The vectors capture "this is about how good a film is", not the direction.

**"Why does GloVe put *plot* next to *conspiracy*?"**
It was trained on Wikipedia and news, where *plot* usually means a conspiracy. fastText was trained on web pages, where the film meaning is common.

**"So which model is best?"**
We measured that in Tasks 3 and 4 by running every experiment with all three models. Test 1 told us what to expect: GloVe's news meanings should cost it accuracy.

**"Why compare the models on the same words?"**
Otherwise one model could look better just by being tested on easier words. We used the 13,119 frequent IMDb words that all three know.

**"What would fix the blind spots?"**
A model that reads words in order, like the recurrent encoder in Task 4, can learn that *not* flips the next word. Contextual models (one vector per word *in context*) also address the one-vector-per-word problem.

---

## How to reproduce

From `Athena/`, after the Task 1 export (`python part2/task1_choose_embeddings/embedding_check.py export`):

| Command | What it prints |
|---|---|
| `python part2/task2_explore_embeddings/embedding_probe.py neighbours` | Test 1 |
| `python part2/task2_explore_embeddings/embedding_probe.py blindspots` | Test 2 |
