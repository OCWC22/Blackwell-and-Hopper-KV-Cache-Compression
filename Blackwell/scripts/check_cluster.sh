#!/bin/bash
# Run this first after SSH login to verify cluster environment
set -e

echo "=== Cluster Environment Check ==="
echo ""

echo "--- Node Info ---"
hostname
whoami
echo ""

echo "--- SLURM Info ---"
sinfo 2>/dev/null || echo "sinfo not available (are you on login node?)"
echo ""

echo "--- Job Queue ---"
squeue -u $USER 2>/dev/null || echo "squeue not available"
echo ""

echo "--- Shared Filesystem ---"
df -h /mnt/sharefs 2>/dev/null || df -h /fsx 2>/dev/null || echo "Shared FS not found at /mnt/sharefs or /fsx"
echo ""

echo "--- Python ---"
which python3 || which python || echo "No python found"
python3 --version 2>/dev/null || python --version 2>/dev/null
echo ""

echo "--- TensorRT-LLM Container/Module Availability ---"
# Check for common TRT-LLM containers
if command -v enroot &>/dev/null; then
    echo "enroot available (container runtime)"
    enroot list 2>/dev/null | grep -i "tensorrt\|trt" || echo "  No TRT-LLM containers found via enroot"
elif command -v docker &>/dev/null; then
    echo "docker available"
    docker images 2>/dev/null | grep -i "tensorrt\|trt" || echo "  No TRT-LLM images found via docker"
elif command -v singularity &>/dev/null; then
    echo "singularity available"
fi
# Check for module system
if command -v module &>/dev/null; then
    module avail 2>&1 | grep -i "tensorrt\|trt" || echo "  No TRT-LLM modules found"
fi
# Check nvcr.io container reference
echo "Expected container: nvcr.io/nvidia/tensorrt-llm (pull separately if needed)"
echo ""

echo "=== Done. Now run: scripts/check_gpu.sh via srun ==="
echo "  srun --gpus=1 --pty bash scripts/check_gpu.sh"
