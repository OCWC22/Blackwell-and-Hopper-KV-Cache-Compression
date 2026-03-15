#!/usr/bin/env bash
#SBATCH --job-name=b200-array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --array=0-7%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
matrix_file="${MATRIX_FILE:-$repo_root/configs/blackwell_eval_matrix.tsv}"
line_number="$((SLURM_ARRAY_TASK_ID + 2))"
row="$(sed -n "${line_number}p" "$matrix_file")"

mkdir -p "$repo_root/results" "$repo_root/logs"

if [[ -z "${row:-}" ]]; then
  echo "No matrix row found for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS=$'\t' read -r variant model context_length batch_size workload notes <<< "$row"

export RUN_ID="${RUN_ID:-${variant}-${context_length}-${batch_size}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
export VARIANT="$variant"
export MODEL="$model"
export CONTEXT_LENGTH="$context_length"
export BATCH_SIZE="$batch_size"
export WORKLOAD="$workload"
export MATRIX_NOTES="$notes"

{
  echo "variant=$VARIANT"
  echo "model=$MODEL"
  echo "context_length=$CONTEXT_LENGTH"
  echo "batch_size=$BATCH_SIZE"
  echo "workload=$WORKLOAD"
  echo "notes=$MATRIX_NOTES"
} > "$repo_root/results/${RUN_ID}.matrix.env"

exec "$repo_root/scripts/run_b200_eval.sh"
