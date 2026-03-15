"""Shared utilities for KV cache benchmark scripts.

Used by run_baseline.py, run_tiered_experiment.py, and compare_results.py.
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone


def get_gpu_info():
    """Query GPU info via nvidia-smi. Returns dict with gpu_model, driver, gpu_count."""
    info = {"gpu_model": "unknown", "driver_version": "unknown", "gpu_count": 0}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, timeout=10
        ).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            parts = lines[0].split(", ")
            info["gpu_model"] = parts[0] if len(parts) > 0 else "unknown"
            info["driver_version"] = parts[1] if len(parts) > 1 else "unknown"
            info["gpu_count"] = len(lines)
    except Exception:
        pass
    return info


def get_cuda_version():
    """Get CUDA version from nvcc or torch."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, timeout=10)
        for line in out.splitlines():
            if "release" in line.lower():
                # e.g. "Cuda compilation tools, release 12.8, V12.8.89"
                parts = line.split("release")
                if len(parts) > 1:
                    return parts[1].strip().split(",")[0].strip()
    except Exception:
        pass
    try:
        import torch
        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


def get_runtime_versions():
    """Get versions of vLLM, TRT-LLM, LMCache, torch."""
    versions = {}
    for pkg, mod in [("torch", "torch"), ("vllm", "vllm"),
                     ("tensorrt_llm", "tensorrt_llm"), ("lmcache", "lmcache")]:
        try:
            m = __import__(mod)
            versions[pkg] = getattr(m, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


class PowerSampler(threading.Thread):
    """Background thread that samples GPU power draw via nvidia-smi."""

    def __init__(self, interval_s=0.5):
        super().__init__(daemon=True)
        self.interval = interval_s
        self.readings = []
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5
                ).strip()
                for line in out.splitlines():
                    val = line.strip()
                    if val:
                        self.readings.append(float(val))
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)

    def average(self):
        return sum(self.readings) / len(self.readings) if self.readings else 0.0


def make_result_template():
    """Return the canonical result JSON template.

    Every benchmark JSON must include scenario_id, serving_mode, and
    cold_tier_codec so results are traceable to the four benchmark scenarios.
    """
    return {
        "run_id": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario_id": "",
        "serving_mode": "offline",
        "runtime": {
            "engine": "",
            "engine_version": "",
            "cuda_version": "",
            "driver_version": "",
            "gpu_name": "",
            "gpu_count": 1,
            "node_count": 1
        },
        "model": {
            "name": "",
            "kv_mode": "",
            "context_length": 0
        },
        "workload": {
            "type": "",
            "requests": 0,
            "concurrency": 1,
            "shared_prefix_tokens": 0,
            "generated_tokens_per_request": 128
        },
        "tiering": {
            "enabled": False,
            "hot_tier_format": None,
            "cold_tier_format": None,
            "cold_tier_codec": "none",
            "promotion_policy": None,
            "recent_window_tokens": None,
            "sink_tokens_protected": None
        },
        "metrics": {
            "ttft_ms_p50": None,
            "ttft_ms_p95": None,
            "tpot_ms_p50": None,
            "tpot_ms_p95": None,
            "tpot_ms_p99": None,
            "throughput_tokens_per_s": None,
            "peak_hbm_gb": None,
            "gpu_power_w_avg": None,
            "tokens_per_joule": None,
            "cache_hit_rate": None,
            "promotion_latency_ms_p50": None,
            "promotion_latency_ms_p95": None,
            "quality_metric_name": None,
            "quality_metric_value": None,
            "quality_delta_vs_best_baseline": None
        },
        "notes": {
            "support_gate_result": "",
            "known_limitations": []
        }
    }


def write_result_json(data, path):
    """Write result dict to JSON file, creating directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Result written to {path}")


def percentile(values, p):
    """Compute the p-th percentile of a list of values."""
    if not values:
        return None
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_v) else f
    d = k - f
    return sorted_v[f] + d * (sorted_v[c] - sorted_v[f])


def tokens_per_joule(total_tokens, avg_power_w, total_time_s):
    """Compute energy efficiency: tokens / (watts * seconds) = tokens/joule."""
    joules = avg_power_w * total_time_s
    return total_tokens / joules if joules > 0 else 0.0


def generate_run_id(kv_mode, context_length, prefix="baseline"):
    """Generate a unique run ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{kv_mode}_{context_length}_{ts}"
