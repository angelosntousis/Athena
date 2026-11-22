#!/bin/bash
#SBATCH --job-name=imdb-seeds
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --array=0-15%2
#SBATCH --output=results/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stepan.maschan@student.uva.nl

# Reruns 8 fixed configurations with training seeds 1-2.
# With the published seed-0 runs, each configuration has 3 seeds on the same five folds.
# Array index = 2 * configuration + (seed - 1).

set -euo pipefail
cd "$HOME/Athena"
source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export HF_HUB_OFFLINE=1

configuration=$((SLURM_ARRAY_TASK_ID / 2))
seed=$((SLURM_ARRAY_TASK_ID % 2 + 1))

case "$configuration" in
    0) args=(--encoder embed-and-pool --embedding glove) ;;
    1) args=(--encoder embed-and-pool --embedding word2vec) ;;
    2) args=(--encoder embed-and-pool --embedding fasttext) ;;
    3) args=(--encoder embed-and-pool --embedding fasttext --weights fine-tuned) ;;
    4) args=(--encoder lstm --embedding glove) ;;
    5) args=(--encoder lstm --embedding word2vec) ;;
    6) args=(--encoder lstm --embedding fasttext) ;;
    7) args=(--encoder lstm --embedding fasttext --weights fine-tuned) ;;
    *) exit 2 ;;
esac
srun python -u -m part2.project train --device cuda --seed "$seed" "${args[@]}"
