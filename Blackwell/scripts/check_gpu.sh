#!/bin/bash
# Run this ON a compute node (via srun --gpus=1 --pty bash)
# Primary runtime: TensorRT-LLM
set -e

echo "=== GPU Environment Check ==="
echo ""

echo "--- GPU Hardware ---"
nvidia-smi
echo ""

echo "--- CUDA Version ---"
nvcc --version 2>/dev/null || echo "nvcc not found"
echo ""

echo "--- PyTorch + CUDA ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    props = torch.cuda.get_device_properties(i)
    print(f'    Memory: {props.total_mem / 1e9:.1f} GB')
    print(f'    SM count: {props.multi_processor_count}')
" 2>/dev/null || echo "PyTorch not available"
echo ""

echo "--- TensorRT-LLM (primary runtime) ---"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM: {tensorrt_llm.__version__}')" 2>/dev/null || echo "TensorRT-LLM not installed"
echo ""

echo "--- ModelOpt ---"
python3 -c "import modelopt; print(f'ModelOpt: {modelopt.__version__}')" 2>/dev/null || echo "ModelOpt not installed"
echo ""

echo "--- vLLM (secondary runtime) ---"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM not installed"
echo ""

echo "--- Transformers ---"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "Transformers not installed"
echo ""

echo "=== GPU check complete ==="
