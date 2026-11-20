#!/bin/bash
#SBATCH --job-name=imdb-lstm
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-4%2
#SBATCH --output=results/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export HF_HUB_OFFLINE=1

case "$SLURM_ARRAY_TASK_ID" in
    0) args=(--embedding glove) ;;
    1) args=(--embedding fasttext) ;;
    2) args=(--embedding word2vec) ;;
    3) args=(--embedding fasttext --weights fine-tuned) ;;
    4) args=(--embedding fasttext --half-data) ;;
    *) exit 2 ;;
esac
srun python -u -m part2.project train --encoder lstm --device cuda "${args[@]}"
