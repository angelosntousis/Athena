# DUTIES — NLP1 Project 1

This Markdown file records what everybody does, what we agree to compare, and what we need to finish. Our three committers are **Angelos, Giovanni, and Stephen**.

Our chosen dataset and task are **Sentiment Classification — IMDb**: [mteb/imdb on Hugging Face](https://huggingface.co/datasets/mteb/imdb).

Our aim is work worthy of a **10/10**: complete the assignment, run fair and reproducible experiments, explain our decisions, and analyse what we learn. This plan is a quality target; the course assessors determine the grade.

## 1. Initial duties

| Person | Main responsibility | Concrete deliverables |
| --- | --- | --- |
| **Stephen** | **1. Dataset selection and understanding.** Explain why IMDb fits the task. Inspect examples and compute statistics, including instances of each classification label. | A data summary, label counts and proportions, text-length plots, representative examples, duplicate/related-text audit, and documented split policy. |
| **Giovanni** | **2. Text segmentation and tokenisation.** Start from the tutorial's BPE tokeniser. Decide whether to train a new one, which hyperparameters to use, and which text it may learn from. Assess whether the available training corpus is sufficient; justify any larger resource of similar nature. | A reproducible tokenisation pipeline, saved tokeniser settings, segmentation examples, vocabulary/unknown-token statistics, and a justified training-data choice. |
| **Angelos** | **3. Feature representations.** Start with token occurrences and frequencies from the tutorial, then implement and motivate an additional representation. | Token indicators, bag-of-words counts, and TF-IDF encoders; a short explanation of their differences and small examples showing the resulting vectors. |

These duties establish a **shared pipeline**. Agree on the data, folds, tokenisation interface, feature definitions, and result format before running separate model experiments. Everyone reviews another person's contribution and must understand the full system for the oral assessment.

## 2. Model ownership and shared follow-up

After the initial work, **each of us experiments with one architecture and its hyperparameters**, using all three agreed representations.

| Person | Architecture | Initial hyperparameters to investigate |
| --- | --- | --- |
| **Stephen** | Log-linear classifier / logistic regression, following `part1/loglin.py`. | Regularisation strength `C`, convergence settings, and the effect of each feature representation. |
| **Angelos** | Single-hidden-layer MLP, following `part1/smlp.py`. | Hidden width, learning rate, weight decay, and dropout. Start with the tutorial settings. |
| **Giovanni** | **3rd model — to be decided by the team.** It must provide an additional architecture beyond logistic regression and the shallow MLP. | Agree on the model and its hyperparameters before running experiments; compare it using the same representations and evaluation protocol. |

Shared assignment steps:

4. **Experiment with architectures and hyperparameters.** Keep configurations and motivations in the experiment log. Change one factor at a time for explanatory ablations.
5. **Evaluate every model.** Use the same cross-validation folds, primary metric, and tuning budget. Reconsider what the evaluation actually measures and add complementary metrics where useful.
6. **Analyse and choose.** Compare uncertainty, practical differences, errors, and computational cost. Decide together which system is best supported by the evidence; a more complex model is not automatically better.

Stephen coordinates the evaluation tables, Giovanni coordinates the Bayesian comparisons, and Angelos coordinates the error-analysis summary. Everyone supplies results and checks the conclusions.

## 3. Start from the tutorial

Our reference materials are [tutorial.ipynb](part1/tutorial.ipynb) and [project.ipynb](part1/project.ipynb). Adapt the tutorial to sentiment classification rather than simply rerunning its AG News experiment.

- Reuse and explain the BPE training approach in `part1/experiment.py` and the token-indicator encoder in the notebook.
- Implement the bag-of-words exercise ourselves: the notebook leaves `BagOfWords` unfinished and refers to optional `part1.solutions` code. Do not assume those solutions are available.
- Use `part1/loglin.py` and `part1/smlp.py` as the architectural starting points. Expose any new tunable arguments explicitly; for example, the current logistic-regression wrapper does not expose `C`.
- Adapt the experiment loop to fit tokenisers and learned feature statistics separately inside training folds. The tutorial helper accepts a prebuilt encoder, so it cannot enforce this automatically.
- Give neural models an **inner training/validation split** for epoch selection. The tutorial selects the best epoch using its supplied validation set; using the outer evaluation fold for that purpose would bias the reported result.
- Use the tutorial's correlated Bayesian comparison approach, checking the underlying `baycomp` API and probability calculations. When combining three CV repetitions, pass `runs=3`; the local wrapper defaults to one run.

## 4. Required comparison matrix

Use the same tokenisation policy across the main matrix so that feature and architecture differences are interpretable.

| Representation | Logistic regression — Stephen | Shallow MLP — Angelos | 3rd model (TBD) — Giovanni |
| --- | --- | --- | --- |
| Token indicators: binary presence/absence | Required | Required | Required |
| Bag-of-words: raw occurrence counts | Required | Required | Required |
| TF-IDF: frequency weighted by training-corpus document rarity, with documented normalisation | Required | Required | Required |

This gives **nine system configurations**, plus a majority-class baseline fitted on each training fold. The full matrix is our chosen design for clear comparisons, not an additional requirement stated by the assignment.

For the main matrix, keep BPE settings fixed after a training-only pilot. Start from the tutorial's `vocab_size=1000` and `min_frequency=2`; treat other values as experiments, not established improvements. Fit TF-IDF vocabulary/statistics on training text only. Keep negation words such as “not”; justify any preprocessing rather than automatically removing stopwords or punctuation.

## 5. Data and evaluation protocol — agree before experiments

### Dataset facts and audit

We use **`mteb/imdb`**, the MTEB version of the English movie-review sentiment dataset. Its [current split metadata](https://huggingface.co/datasets/mteb/imdb/raw/main/README.md) lists **24,904 training reviews** and **24,678 test reviews**, with no predefined validation split. The fields are **`text`**, **`label`**, and **`label_text`**. The binary target is **0 = negative**, **1 = positive**; test labels are available.

Stephen must verify the loaded revision, actual split sizes, label mapping and proportions, empty texts, duplicates, conflicting labels, and review-length distributions. Save the dataset revision and split-local row indices or text hashes for reproducibility. Use **`text` as input** and **`label` as target**; `label_text` is a readable copy of the target and must never be included as an input feature.

Use this repository's actual counts rather than assuming the commonly quoted 25,000/25,000 counts: the card's descriptive-statistics section contains counts that differ from its split metadata. Recompute class counts rather than assuming the selected version is perfectly balanced.

Audit duplicate and near-duplicate reviews within training and check train/test text overlap under a predefined policy, without inspecting test labels to make modelling decisions. Keep exact training duplicates in the same fold, or remove them consistently before splitting; record exclusions and any residual dependence. If reliable source/movie groups become available, consider group-based evaluation as an additional question. The provided three fields do not contain movie IDs, so do not invent groups from row order.

Reviews can contain HTML line breaks such as `<br />`. Agree on whether to replace these with spaces, document that policy, and apply it identically before tokeniser training and all feature transformations. Inspect long reviews and preserve negation. Do not silently truncate inputs: these feature encoders can count tokens across a full review.

### Splits and fair comparison

1. Use the dataset's **training split for development and cross-validation**. Reserve its labelled **test split for final held-out evaluation**, excluding it from design, tuning, and error analysis. Create any validation/early-stopping subsets only from the current training fold. Limit any prior test access to the predefined text-overlap audit, without using its labels or model scores.
2. Use **5-fold stratified cross-validation**, repeated with split seeds **0, 3, and 5**, shared across every system. If grouping is possible, use stratified group folds and record any imbalance. Save the exact row IDs for every split.
3. Tune hyperparameters **inside each outer training fold**, using a small, predeclared 3-fold inner CV search. Use the same candidate-budget limit for each architecture. All learned preprocessing must also be fitted inside the inner training folds. Keep early-stopping data separate from the inner scoring fold.
4. After choosing hyperparameters, refit within the outer training data and score the outer fold once. For neural models, derive an epoch budget from inner training or use a separate early-stopping subset of outer training; never choose epochs on the outer evaluation fold.
5. **Accuracy is the primary metric**, subject to justification from the observed class balance and task. Also report macro-F1 and per-class precision/recall. Save out-of-fold predictions and confusion matrices. Record training time, inference time, and model size with comparable hardware and batch settings.
6. Set and log Python, NumPy, and PyTorch seeds. Record split seeds and model seeds separately. Use a fixed, shared model-seed policy for the main comparison, and test additional initialisation seeds for shortlisted neural systems if resources permit. Do not treat repeated predictions for the same examples as independent observations.
7. Select the final configuration from CV evidence, refit on the dataset's training split, then evaluate the reserved labelled **test split once**. Report this separately from CV. Do not use the final test result to select a different model or tune settings.

Pilot one fold first to estimate runtime and memory. Start with at most **four candidate hyperparameter configurations per architecture and representation**. If nested evaluation is too expensive, reduce and freeze the search budget before examining outer results; do not silently turn evaluation folds into tuning data.

### Bayesian estimation and ROPE — required

- Compare **paired accuracy scores on matching outer folds**, in identical run/fold order. Use the tutorial's Bayesian correlated approach to account for overlapping CV training sets; do not treat all fold scores as independent samples.
- Predeclare the main ROPE as **[-0.005, +0.005] in accuracy**, meaning differences below **0.5 percentage points** are practically negligible for this project. Motivate this as a performance-versus-complexity trade-off, rather than choosing it after seeing which model wins.
- Prioritise controlled comparisons: representations within each architecture, then architectures using the same representation. Report `P(A better)`, `P(practically equivalent)`, and `P(B better)`, with the direction of the difference explicit.
- Include posterior plots and an uncertainty interval whose definition is stated. Check that probabilities sum to approximately one and that the tutorial wrapper agrees with the underlying library; adapt it if necessary.
- As a sensitivity analysis, also show ROPE half-widths **0.0025** and **0.01**. Explain if the interpretation changes.
- Report mean and standard deviation across folds descriptively, without presenting that standard deviation as an independent-sample confidence interval. Uncertain evidence is a valid conclusion; do not force a winner.

## 6. Extensions that strengthen the work

Finish the required comparisons before adding extensions. Our central question is: **For IMDb, do richer text features help more reliably than changing the classifier architecture?**

Prioritise these two extensions:

1. **Feature ablation:** compare TF-IDF unigrams against TF-IDF unigrams + bigrams under a fixed architecture and evaluation protocol. Hypothesis: short combinations can capture sentiment cues such as negation better than isolated tokens. State whether the units are words or BPE tokens; use an unchanged tokeniser for the controlled comparison.
2. **Structured error analysis:** inspect about **30–50 training out-of-fold errors**, selected reproducibly and covering both labels. Categorise negation, contrast (“but”), mixed sentiment, sarcasm, long reviews, and other observed patterns. Examine whether errors vary with review length. Compare cases fixed or broken by the feature and architecture changes. Show representative examples without generalising from a few anecdotes.

If time remains, choose **one** further question: BPE vocabulary size (for example 1k/5k/10k), word-level versus BPE tokenisation, a learning curve, or a computational-budget comparison of the chosen architectures. Tune these choices inside training data and evaluate on the same outer folds. External tokeniser-training text needs a source, domain rationale, and overlap audit; it is optional, not automatically necessary.

These extensions should produce explanations, not just a larger score table. A well-supported negative result is useful. Pretrained transformers or large ensembles are not needed to satisfy this project's stated checklist; ask the TA before expanding the scope substantially.

## 7. Milestones and goals

| Milestone | Lead | Done when |
| --- | --- | --- |
| **A. Understand and freeze the protocol** | Stephen; all approve | Dataset audit, split policy, primary metric, ROPE, seed policy, and experiment budget are written down. Confirm the deadline and oral schedule from official course information. |
| **B. Build the shared pipeline** | Giovanni + Angelos | Fold-local tokenisation and three encoders work; a smoke run trains all three architectures; preprocessing/early-stopping leakage checks pass. |
| **C. Run required experiments** | Each model owner | All nine systems have complete paired outer-fold scores, configs, predictions, convergence notes, and runtime records. |
| **D. Explain results and extensions** | All; Giovanni coordinates statistics | Bayesian comparisons, ROPE sensitivity, the two priority extensions, and error categories are complete. Failed and inconclusive experiments are recorded. |
| **E. Final evaluation and reproducibility** | Stephen; peer reviewed | Final configuration is justified, held-out evaluation is complete, and another teammate reproduces a documented run from the README environment. |
| **F. Prepare oral assessment and blog material** | All | At most five backup slides, concise motivations/findings, source references, AI disclosure, and an oral rehearsal are ready. |

## 8. Reproducibility, teamwork, and records

- Use the README's Python 3.14 environment: `source ~/envs/nlp1-2026/bin/activate`. Record exact package versions for the final run and the commands needed to reproduce it.
- Keep reusable code separate from exploratory notebooks. Agree on locations such as `experiments/`, `configs/`, `results/`, and `reports/` before implementation; these are proposed locations, not files already created by this plan.
- Save one result record per system/run/fold: dataset revision, split IDs, tokeniser/feature settings, architecture, hyperparameters, seeds, accuracy, macro-F1, runtime, convergence, and artifact locations. Keep raw outputs so figures can be regenerated.
- Keep commits focused and reviewable. Each teammate checks another person's code and can explain their own changes. Never invent, selectively omit, or manually improve results.
- Maintain a decision log: **question → hypothesis → controlled experiment → result → uncertainty → conclusion**. Save sources and reusable figures for the eventual blog.
- Maintain an AI-assistance log with date, tool, delegated task, affected files, and human verification. **Initial entry, 2026-11-03:** Codex assisted with environment setup and drafting `DUTIES.md` from the assignment and tutorial; the team must review the plan. No classifier experiments or results were produced as part of writing this plan. Add later assistance as it occurs.
- **Feature implementation entry, 2026-11-03:** Codex helped implement token indicators, bag-of-words counts, and TF-IDF in `part1/features.py`, and created `part1/features_walkthrough.ipynb` with synthetic examples and correctness checks. Angelos must review and explain the code. This step does not produce dataset performance results.
- **Dataset change entry, 2026-11-03:** The team selected `mteb/imdb`. Codex updated this plan and the feature walkthrough to use review text, the selected dataset schema, and a final held-out test split. The feature encoder algorithms remain dataset-independent; no IMDb classifier results have been produced.

## 9. Final quality checklist

- [ ] We understand IMDb, its labels, source domain, split roles, and dependence/leakage limitations.
- [ ] We compare indicators, bag-of-words, and TF-IDF across log-linear, shallow MLP, and our additional architecture.
- [ ] We use shared cross-validation folds, training-only preprocessing/tuning, and a justified metric.
- [ ] We provide Bayesian estimation, a motivated ROPE, and honest uncertainty statements.
- [ ] We distinguish feature effects from architecture effects and support extensions with controlled comparisons.
- [ ] We analyse representative errors and report performance alongside computational cost.
- [ ] We can reproduce the experiments and trace each conclusion to saved results.
- [ ] We document sources, reused code, AI assistance, motivations, and limitations.
- [ ] Every teammate can answer tutorial questions about segmentation, BPE, features, softmax/log-linear models, MLPs, evaluation, and practical equivalence.
- [ ] We have rehearsed the oral assessment using **at most five backup slides**: **data and metrics**, **features**, **architectures**, **results**, and **extra**. These guide answers; they are not a scheduled presentation.

## Sources

- [Project 1 assignment and checklist](part1/project.ipynb).
- [Tutorial and its questions](part1/tutorial.ipynb), plus the accompanying `part1` Python modules.
- [Repository setup instructions](README.md).
- [IMDb dataset card, label examples, and original-paper citation](https://huggingface.co/datasets/mteb/imdb).
- [MTEB IMDb split metadata and schema](https://huggingface.co/datasets/mteb/imdb/raw/main/README.md).
