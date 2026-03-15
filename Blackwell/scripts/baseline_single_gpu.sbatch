#!/usr/bin/env bash
#SBATCH --job-name=kv-single-gpu-trtllm
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-GPU baseline benchmark for Blackwell KV cache.
# Primary runtime: TensorRT-LLM. Secondary: vLLM.
# Supports Scenario 1 (longer context) and Scenario 2 (more sessions).
# Accepts env vars: KV_MODE, MODEL, CONTEXT_LENGTH, REQUESTS, WORKLOAD, ENGINE, ENGINE_DIR, OFFLOAD, SCENARIO_ID
#
# Usage:
#   KV_MODE=bf16 sbatch scripts/baseline_single_gpu.sbatch
#   KV_MODE=nvfp4 ENGINE=tensorrt_llm sbatch scripts/baseline_single_gpu.sbatch
#   KV_MODE=fp8 ENGINE=vllm sbatch scripts/baseline_single_gpu.sbatch
#   KV_MODE=nvfp4 ENGINE_DIR=/path/to/engine OFFLOAD=1 sbatch scripts/baseline_single_gpu.sbatch

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KV_MODE="${KV_MODE:-bf16}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
REQUESTS="${REQUESTS:-10}"
WORKLOAD="${WORKLOAD:-repeated_prefix}"
ENGINE="${ENGINE:-tensorrt_llm}"
ENGINE_DIR="${ENGINE_DIR:-}"
OFFLOAD="${OFFLOAD:-}"
SCENARIO_ID="${SCENARIO_ID:-scenario_1_longer_context_gpu}"

mkdir -p "$repo_root/logs" "$repo_root/results"

echo "=== Single-GPU KV Baseline ==="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Engine: $ENGINE"
echo "KV Mode: $KV_MODE"
echo "Model: $MODEL"
echo "Context: $CONTEXT_LENGTH"
echo "Start: $(date -Iseconds)"

# Run env probe first
bash "$repo_root/scripts/env_probe.sh" "$repo_root/results/env_probe.json"

# Build run_baseline.py command
CMD=(
    python3 "$repo_root/scripts/run_baseline.py"
    --engine "$ENGINE"
    --model "$MODEL"
    --context-length "$CONTEXT_LENGTH"
    --requests "$REQUESTS"
    --kv-mode "$KV_MODE"
    --workload-type "$WORKLOAD"
    --scenario-id "$SCENARIO_ID"
    --output "$repo_root/results/baseline_${KV_MODE}_${CONTEXT_LENGTH}_${SLURM_JOB_ID:-manual}.json"
)

# Add engine dir if set (TRT-LLM pre-built engine path)
if [[ -n "$ENGINE_DIR" ]]; then
    CMD+=(--engine-dir "$ENGINE_DIR")
fi

# Add offload flag if set
if [[ -n "$OFFLOAD" && "$OFFLOAD" != "0" ]]; then
    CMD+=(--offload)
fi

"${CMD[@]}"

echo "End: $(date -Iseconds)"
