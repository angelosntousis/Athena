# Athena

IMDb sentiment classification for NLP1. Each part keeps the course instructions in `project.ipynb` and the team's implementation in `project.py`.

| Part | Assignment | Team code |
| --- | --- | --- |
| 1 · Sparse features, logistic regression and MLPs | [Instructions](part1/project.ipynb) · [Tutorial](part1/tutorial.ipynb) | [Experiment runner](part1/project.py) |
| 2 · Word embeddings and recurrent encoders | [Instructions](part2/project.ipynb) · [Tutorial](part2/tutorial.ipynb) | [Embedding choice](part2/task1_choose_embeddings/TASK1.md) · [Embedding analysis](part2/task2_explore_embeddings/TASK2.md) · [Embed-and-pool](part2/task3_embed_and_pool/TASK3.md) · [LSTM](part2/task4_recurrent_encoder/TASK4.md) · [Shared runner](part2/project.py) |
| 3 · Upcoming | [Part 3 folder](part3/) | Pending the assignment |

## Local setup

From the repository root, create the environment once and install the dependencies.

```bash
python3.14 -m venv ~/envs/nlp1-2026
source ~/envs/nlp1-2026/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

For later sessions, run the activation line again. The [course setup guide](docs/setup.md) covers VS Code, uv and Snellius.

## Revisit Part 1

- Inspect the training and evaluation commands with `python part1/project.py --help`. The original `python final.py` command also works.
- Find sweep settings in [configs](part1/configs/) and the selected configurations in [configs/final](part1/configs/final/). When launching a W&B CLI agent, run it from `part1/` so it can find `train_mlp.py` or `train_loglin.py`.
- Read the [representation examples](part1/reports/representations.md) alongside the [saved predictions](results/final_predictions/index.csv). The other [reports](part1/reports/), [exploratory notebooks](part1/notebooks/) and [oral exam slides](part1/slides/) are grouped by purpose.
