#!/usr/bin/env bash
# Environment probe for Blackwell KV runtime benchmarking.
# Primary check: TensorRT-LLM (NVFP4 and FP8 KV support).
# Secondary/fallback check: vLLM FP8 KV support.
# Writes structured JSON to results/env_probe.json.
# Run this BEFORE any benchmark to establish the support gate.
#
# Usage:
#   bash scripts/env_probe.sh                          # default output
#   bash scripts/env_probe.sh results/env_probe.json   # explicit path
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT="${1:-$REPO_ROOT/results/env_probe.json}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== Blackwell Environment Probe (primary: TensorRT-LLM) ==="
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


# ---------------------------------------------------------------------------
# TensorRT-LLM checks (PRIMARY)
# ---------------------------------------------------------------------------

def check_trtllm_nvfp4_support():
    """Check if TensorRT-LLM supports NVFP4 KV cache.

    Looks for KvCacheConfig, kv_cache_type, or nvfp4-related references.
    """
    try:
        import tensorrt_llm
    except ImportError:
        return "not_installed"
    try:
        import inspect
        # Check KvCacheConfig for nvfp4 references
        try:
            from tensorrt_llm import KvCacheConfig
            src = inspect.getsource(KvCacheConfig)
            if "nvfp4" in src.lower() or "fp4" in src.lower():
                return True
        except (ImportError, AttributeError):
            pass
        # Check for kv_cache_type with nvfp4 in broader source
        try:
            trtllm_dir = os.path.dirname(inspect.getfile(tensorrt_llm))
            for dirpath, _, filenames in os.walk(trtllm_dir):
                for fname in filenames:
                    if fname.endswith('.py'):
                        try:
                            with open(os.path.join(dirpath, fname)) as f:
                                content = f.read()
                            if ('nvfp4' in content.lower() or 'fp4' in content.lower()) and 'kv' in content.lower():
                                return True
                        except Exception:
                            continue
                # Only check top-level package to keep probe fast
                break
        except Exception:
            pass
        return "unknown"
    except Exception:
        return "unknown"


def check_trtllm_fp8_support():
    """Check if TensorRT-LLM supports FP8 KV cache."""
    try:
        import tensorrt_llm
    except ImportError:
        return "not_installed"
    try:
        import inspect
        try:
            from tensorrt_llm import KvCacheConfig
            src = inspect.getsource(KvCacheConfig)
            if "fp8" in src.lower():
                return True
        except (ImportError, AttributeError):
            pass
        # Broader check
        try:
            trtllm_dir = os.path.dirname(inspect.getfile(tensorrt_llm))
            for dirpath, _, filenames in os.walk(trtllm_dir):
                for fname in filenames:
                    if fname.endswith('.py'):
                        try:
                            with open(os.path.join(dirpath, fname)) as f:
                                content = f.read()
                            if 'fp8' in content.lower() and 'kv' in content.lower():
                                return True
                        except Exception:
                            continue
                break
        except Exception:
            pass
        return "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# vLLM checks (SECONDARY / FALLBACK)
# ---------------------------------------------------------------------------

def check_vllm_fp8_kv_support():
    """Check if vLLM supports FP8 KV cache dtype (fallback path)."""
    try:
        from vllm.engine.arg_utils import EngineArgs
        import inspect
        sig = inspect.signature(EngineArgs.__init__)
        if "kv_cache_dtype" in sig.parameters:
            return True
        return "unknown"
    except Exception:
        return "unknown"


def check_vllm_nvfp4_kv_support():
    """Heuristic check for NVFP4 KV support in vLLM (fallback path)."""
    try:
        from vllm.engine.arg_utils import EngineArgs
        import inspect
        src = inspect.getsource(EngineArgs)
        if "nvfp4" in src.lower() or "fp4" in src.lower():
            return True
        return "unknown"
    except Exception:
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

# Primary: TensorRT-LLM checks
trtllm_nvfp4 = check_trtllm_nvfp4_support()
trtllm_fp8 = check_trtllm_fp8_support()

# Secondary: vLLM checks (fallback)
vllm_fp8 = check_vllm_fp8_kv_support()
vllm_nvfp4 = check_vllm_nvfp4_kv_support()

# Recommended hot tier: TRT-LLM NVFP4 > TRT-LLM FP8 > vLLM FP8 > BF16
if trtllm_nvfp4 is True:
    recommended_hot_tier = "nvfp4"
    recommended_runtime = "tensorrt_llm"
elif trtllm_fp8 is True:
    recommended_hot_tier = "fp8"
    recommended_runtime = "tensorrt_llm"
elif vllm_fp8 is True:
    recommended_hot_tier = "fp8"
    recommended_runtime = "vllm"
else:
    recommended_hot_tier = "bf16"
    recommended_runtime = "tensorrt_llm" if get_version("tensorrt_llm") != "not_installed" else "vllm"

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
        "tensorrt_llm_version": get_version("tensorrt_llm"),
        "modelopt_version": get_version("modelopt"),
        "vllm_version": get_version("vllm"),
        "lmcache_version": get_version("lmcache"),
        "transformers_version": get_version("transformers")
    },
    "filesystem": {
        "shared_fs_path": shared_fs
    },
    "support_gate": {
        "is_blackwell": is_blackwell,
        "trtllm_nvfp4_kv_supported": trtllm_nvfp4,
        "trtllm_fp8_kv_supported": trtllm_fp8,
        "vllm_fp8_kv_supported": vllm_fp8,
        "vllm_nvfp4_kv_supported": vllm_nvfp4,
        "recommended_hot_tier": recommended_hot_tier,
        "recommended_runtime": recommended_runtime,
        "notes": (
            "Primary check: TensorRT-LLM NVFP4/FP8 KV support. "
            "Secondary/fallback: vLLM FP8 KV. "
            "NVFP4 is Blackwell-native but KV-cache support is stack-dependent. "
            "Actual verification requires loading a model with the target kv_cache_type."
        )
    }
}

os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSupport gate result:")
print(f"  Blackwell:          {is_blackwell}")
print(f"  TRT-LLM NVFP4 KV:  {trtllm_nvfp4}")
print(f"  TRT-LLM FP8 KV:    {trtllm_fp8}")
print(f"  vLLM FP8 KV:        {vllm_fp8}")
print(f"  Recommended runtime: {recommended_runtime}")
print(f"  Recommended hot tier: {recommended_hot_tier}")
print(f"\nProbe written to: {output_path}")
PYEOF
