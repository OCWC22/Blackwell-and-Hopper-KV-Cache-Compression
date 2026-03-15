#!/bin/bash
# Run on login node to set up Python environment.
# TensorRT-LLM is the primary hackathon runtime.
# vLLM is installed as a secondary/fallback runtime.
set -e

echo "=== Setting up environment ==="

# Create conda env if conda is available
if command -v conda &> /dev/null; then
    conda create -n kvcomp python=3.11 -y
    conda activate kvcomp
fi

pip install --upgrade pip

# Primary runtime: TensorRT-LLM + ModelOpt
# For container-based workflows, use nvcr.io/nvidia/tensorrt-llm instead.
pip install tensorrt-llm
pip install nvidia-modelopt

# Secondary runtime: vLLM (follow-up compatibility/productization path)
pip install torch vllm transformers accelerate

# Shared dependencies
pip install numpy pandas matplotlib

echo "=== Environment ready ==="
echo "Primary runtime: TensorRT-LLM (verify with: python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)')"
echo "Secondary runtime: vLLM (verify with: python3 -c 'import vllm; print(vllm.__version__)')"
