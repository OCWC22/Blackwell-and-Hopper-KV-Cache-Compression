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

echo "=== Done. Now run: scripts/check_gpu.sh via srun ==="
echo "  srun --gpus=1 --pty bash scripts/check_gpu.sh"
