#!/usr/bin/env bash
# Environment probe for Blackwell KV runtime benchmarking.
# Writes structured JSON to results/env_probe.json.
# Run this BEFORE any benchmark to establish the support gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT="${1:-$REPO_ROOT/results/env_probe.json}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== Blackwell Environment Probe ==="
echo "Output: $OUTPUT"

# Collect cluster info (safe to fail if not on Slurm)
SLURM_JOB="${SLURM_JOB_ID:-none}"
HOSTNAME_VAL="$(hostname)"

# GPU info
GPU_MODEL="unknown"
GPU_COUNT="0"
DRIVER_VERSION="unknown"
if command -v nvidia-smi &>/dev/null; then
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs) || true
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | xargs) || true
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs) || true
    echo "GPU: $GPU_MODEL (x$GPU_COUNT)"
    echo "Driver: $DRIVER_VERSION"
else
    echo "WARNING: nvidia-smi not found"
fi

# CUDA version
CUDA_VERSION="unknown"
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo "unknown")
fi
echo "CUDA: $CUDA_VERSION"

# Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}' || echo "unknown")
echo "Python: $PYTHON_VERSION"

# Filesystem probe
SHARED_FS="none"
for path in /mnt/sharefs /fsx /scratch; do
    if [ -d "$path" ] 2>/dev/null; then
        SHARED_FS="$path"
        break
    fi
done
if [ "$SHARED_FS" = "none" ] && [ -n "${SCRATCH:-}" ] && [ -d "${SCRATCH}" ]; then
    SHARED_FS="$SCRATCH"
fi

# Emit JSON via Python for correctness and support-gate checks
python3 - "$OUTPUT" "$GPU_MODEL" "$DRIVER_VERSION" "$CUDA_VERSION" "$GPU_COUNT" \
    "$SHARED_FS" "$HOSTNAME_VAL" "$SLURM_JOB" "$PYTHON_VERSION" << 'PYEOF'
import json
import os
import sys
from datetime import datetime, timezone


def get_version(module_name):
    try:
        m = __import__(module_name)
        return getattr(m, "__version__", "installed")
    except ImportError:
        return "not_installed"
    except Exception as e:
        return f"error: {e}"


def check_fp8_kv_support():
    """Check if vLLM supports FP8 KV cache dtype."""
    try:
        from vllm.engine.arg_utils import EngineArgs
        import inspect
        sig = inspect.signature(EngineArgs.__init__)
        if "kv_cache_dtype" in sig.parameters:
            return True
        return "unknown"
    except Exception:
        return "unknown"


def check_nvfp4_kv_support():
    """Heuristic check for NVFP4 KV support in vLLM or TRT-LLM."""
    try:
        from vllm.engine.arg_utils import EngineArgs
        import inspect
        src = inspect.getsource(EngineArgs)
        if "nvfp4" in src.lower() or "fp4" in src.lower():
            return True
        return "unknown"
    except Exception:
        pass
    try:
        import tensorrt_llm
        return "check_trt_llm_manually"
    except ImportError:
        pass
    return "unknown"


output_path = sys.argv[1]
gpu_model = sys.argv[2]
driver = sys.argv[3]
cuda = sys.argv[4]
gpu_count = int(sys.argv[5]) if sys.argv[5].strip().isdigit() else 0
shared_fs = sys.argv[6]
hostname = sys.argv[7]
slurm_job = sys.argv[8]
python_ver = sys.argv[9]

is_blackwell = any(x in gpu_model.upper() for x in ["B200", "B100", "B300", "GB200"])

fp8_supported = check_fp8_kv_support()
nvfp4_supported = check_nvfp4_kv_support()

if nvfp4_supported is True:
    recommended_hot_tier = "nvfp4"
elif fp8_supported is True:
    recommended_hot_tier = "fp8"
else:
    recommended_hot_tier = "bf16"

result = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "hostname": hostname,
    "slurm_job_id": slurm_job,
    "hardware": {
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "driver_version": driver,
        "cuda_version": cuda,
        "is_blackwell": is_blackwell
    },
    "software": {
        "python_version": python_ver,
        "torch_version": get_version("torch"),
        "vllm_version": get_version("vllm"),
        "tensorrt_llm_version": get_version("tensorrt_llm"),
        "lmcache_version": get_version("lmcache"),
        "transformers_version": get_version("transformers")
    },
    "filesystem": {
        "shared_fs_path": shared_fs
    },
    "support_gate": {
        "is_blackwell": is_blackwell,
        "nvfp4_kv_supported": nvfp4_supported,
        "fp8_kv_supported": fp8_supported,
        "recommended_hot_tier": recommended_hot_tier,
        "notes": (
            "NVFP4 is Blackwell-native but KV-cache support is stack-dependent. "
            "This probe checks vLLM source heuristically. "
            "Actual verification requires loading a model with the target kv_cache_dtype."
        )
    }
}

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSupport gate result:")
print(f"  Blackwell: {is_blackwell}")
print(f"  NVFP4 KV: {nvfp4_supported}")
print(f"  FP8 KV:   {fp8_supported}")
print(f"  Recommended hot tier: {recommended_hot_tier}")
print(f"\nProbe written to: {output_path}")
PYEOF
