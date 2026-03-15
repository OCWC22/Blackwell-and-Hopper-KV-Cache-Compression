#!/bin/bash
#SBATCH --job-name=kv-baseline
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs results

echo "=== Baseline vLLM Inference Test ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 scripts/run_baseline.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --context-lengths 8192 32768 65536 \
    --output results/baseline_$(date +%Y%m%d_%H%M%S).json

echo "End: $(date)"
