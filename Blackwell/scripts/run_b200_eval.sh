#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${RUN_ID:-b200-${SLURM_JOB_ID:-manual}}"
result_dir="${RESULT_DIR:-$repo_root/results/$run_id}"

mkdir -p "$repo_root/logs" "$result_dir"

: "${EVAL_COMMAND:?Set EVAL_COMMAND to the exact per-node launcher command.}"

master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
master_port="${MASTER_PORT:-29500}"
gpus_per_node="${GPUS_PER_NODE:-8}"
nnodes="${SLURM_JOB_NUM_NODES:-1}"
world_size="${WORLD_SIZE:-$((nnodes * gpus_per_node))}"

{
  echo "run_id=$run_id"
  echo "repo_root=$repo_root"
  echo "result_dir=$result_dir"
  echo "master_addr=$master_addr"
  echo "master_port=$master_port"
  echo "gpus_per_node=$gpus_per_node"
  echo "nnodes=$nnodes"
  echo "world_size=$world_size"
  echo "eval_command=$EVAL_COMMAND"
} | tee "$result_dir/run.env"

scontrol show job "${SLURM_JOB_ID:-}" > "$result_dir/slurm_job.txt" 2>/dev/null || true
nvidia-smi -L > "$result_dir/nvidia_smi_L.txt"
env | sort > "$result_dir/env.txt"
git -C "$repo_root" rev-parse HEAD > "$result_dir/git_sha.txt" || true

export MASTER_ADDR="$master_addr"
export MASTER_PORT="$master_port"
export GPUS_PER_NODE="$gpus_per_node"
export NNODES="$nnodes"
export WORLD_SIZE="$world_size"

srun --ntasks="$nnodes" --ntasks-per-node=1 bash -lc '
set -euo pipefail
export NODE_RANK="${SLURM_NODEID}"
echo "node_rank=$NODE_RANK hostname=$(hostname)" >> "'"$result_dir"'/node_map.txt"
eval "'"$EVAL_COMMAND"'"
'
