#!/usr/bin/env bash
#SBATCH --job-name=kv-single-gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-GPU baseline benchmark for Blackwell KV cache.
# Accepts env vars: KV_MODE, MODEL, CONTEXT_LENGTH, REQUESTS, WORKLOAD, ENGINE
#
# Usage:
#   KV_MODE=bf16 sbatch scripts/baseline_single_gpu.sbatch
#   KV_MODE=fp8 CONTEXT_LENGTH=32768 sbatch scripts/baseline_single_gpu.sbatch
#   KV_MODE=fp8 MODEL=Qwen/Qwen3-32B sbatch scripts/baseline_single_gpu.sbatch

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KV_MODE="${KV_MODE:-bf16}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
REQUESTS="${REQUESTS:-10}"
WORKLOAD="${WORKLOAD:-repeated_prefix}"
ENGINE="${ENGINE:-vllm}"

mkdir -p "$repo_root/logs" "$repo_root/results"

echo "=== Single-GPU KV Baseline ==="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "KV Mode: $KV_MODE"
echo "Model: $MODEL"
echo "Context: $CONTEXT_LENGTH"
echo "Start: $(date -Iseconds)"

# Run env probe first
bash "$repo_root/scripts/env_probe.sh" "$repo_root/results/env_probe.json"

# Run baseline
python3 "$repo_root/scripts/run_baseline.py" \
    --engine "$ENGINE" \
    --model "$MODEL" \
    --context-length "$CONTEXT_LENGTH" \
    --requests "$REQUESTS" \
    --kv-mode "$KV_MODE" \
    --workload-type "$WORKLOAD" \
    --output "$repo_root/results/baseline_${KV_MODE}_${CONTEXT_LENGTH}_${SLURM_JOB_ID:-manual}.json"

echo "End: $(date -Iseconds)"
