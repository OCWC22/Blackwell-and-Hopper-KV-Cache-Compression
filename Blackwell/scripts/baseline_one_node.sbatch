#!/usr/bin/env bash
#SBATCH --job-name=kv-one-node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One-node multi-GPU baseline benchmark for Blackwell KV cache.
# Uses tensor parallelism across all GPUs on a single node.
# Scenario 4: longer context + more sessions on one Slurm node.
# Accepts env vars: KV_MODE, MODEL, CONTEXT_LENGTH, REQUESTS, WORKLOAD, ENGINE, GPUS, SCENARIO_ID
#
# Usage:
#   KV_MODE=fp8 sbatch scripts/baseline_one_node.sbatch
#   KV_MODE=bf16 CONTEXT_LENGTH=32768 sbatch scripts/baseline_one_node.sbatch
#   GPUS=4 KV_MODE=fp8 sbatch scripts/baseline_one_node.sbatch

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KV_MODE="${KV_MODE:-bf16}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
REQUESTS="${REQUESTS:-10}"
WORKLOAD="${WORKLOAD:-repeated_prefix}"
ENGINE="${ENGINE:-vllm}"
GPUS="${GPUS:-8}"
SCENARIO_ID="${SCENARIO_ID:-scenario_4_longer_context_more_sessions_node}"

mkdir -p "$repo_root/logs" "$repo_root/results"

echo "=== One-Node KV Baseline ==="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "GPUs: $GPUS (tensor parallel)"
echo "KV Mode: $KV_MODE"
echo "Model: $MODEL"
echo "Context: $CONTEXT_LENGTH"
echo "Start: $(date -Iseconds)"

# Run env probe first
bash "$repo_root/scripts/env_probe.sh" "$repo_root/results/env_probe.json"

# Run baseline with tensor parallelism
python3 "$repo_root/scripts/run_baseline.py" \
    --engine "$ENGINE" \
    --model "$MODEL" \
    --context-length "$CONTEXT_LENGTH" \
    --requests "$REQUESTS" \
    --kv-mode "$KV_MODE" \
    --workload-type "$WORKLOAD" \
    --tp "$GPUS" \
    --scenario-id "$SCENARIO_ID" \
    --output "$repo_root/results/baseline_${KV_MODE}_${CONTEXT_LENGTH}_tp${GPUS}_${SLURM_JOB_ID:-manual}.json"

echo "End: $(date -Iseconds)"
