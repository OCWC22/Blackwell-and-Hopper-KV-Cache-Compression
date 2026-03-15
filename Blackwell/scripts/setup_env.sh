#!/bin/bash
# Run on login node to set up Python environment
set -e

echo "=== Setting up environment ==="

# Create conda env if conda is available
if command -v conda &> /dev/null; then
    conda create -n kvcomp python=3.11 -y
    conda activate kvcomp
fi

pip install --upgrade pip
pip install torch vllm transformers accelerate
pip install numpy pandas matplotlib

echo "=== Environment ready ==="
