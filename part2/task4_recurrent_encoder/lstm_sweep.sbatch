#!/bin/bash
#SBATCH --job-name=imdb-lstm-sweep
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-31%2
#SBATCH --output=results/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export HF_HUB_OFFLINE=1

# Each array task claims one configuration from the same W&B grid.
srun python -u -m wandb agent --count 1 "${1:?Pass the entity/project/sweep_id as one argument}"
