#!/bin/bash
#SBATCH --job-name=imdb-bda
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out

# Computes the features and fits the Task 6 BDA on CPU (no GPU needed).
# "rome" is a Snellius CPU partition; check the ones your account can use with `sinfo -s`.
# Submit from the repository root, where results/ exists.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1

python -u -m part2.task6_error_analysis.features text
python -u -m part2.task6_error_analysis.features embedding
python -u -m part2.task6_error_analysis.run_bda "$@"
