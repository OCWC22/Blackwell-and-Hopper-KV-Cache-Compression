#!/usr/bin/env bash
#SBATCH --job-name=b200-eval
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_b200_eval.sh"
