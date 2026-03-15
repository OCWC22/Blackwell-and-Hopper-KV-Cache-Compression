#!/usr/bin/env python3
"""KV block loading cost benchmark — dataset-driven.

Ingests a coding-agent dataset (claudeset-community by default), grows a
simulated KV cache by accumulating tokens across sessions, and measures the
cost to load KV-shaped tensors from HBM, host RAM, and disk at each fill
level.  Measures BF16 (raw copy), FP8 (copy + cast), and emulated FP4
(copy + quantize + bit-pack) for every source tier.

Usage:
    python scripts/bench_kv_block_load.py [OPTIONS]

    # Quick test with small synthetic fallback (no HuggingFace download):
    python scripts/bench_kv_block_load.py --synthetic

    # Full run with claudeset-community:
    python scripts/bench_kv_block_load.py --dataset lelouch0110/claudeset-community

Output:
    results/kv_block_load_<timestamp>.json
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

import torch

# ---------------------------------------------------------------------------
# Reuse utilities from Blackwell bench utils when available
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BW_SCRIPTS = os.path.join(_REPO_ROOT, "Blackwell", "scripts")
if os.path.isdir(_BW_SCRIPTS):
    sys.path.insert(0, _BW_SCRIPTS)

try:
    from kv_bench_utils import get_gpu_info, get_cuda_version, percentile
except ImportError:
    # Minimal fallbacks
    def get_gpu_info():
        return {"gpu_model": "unknown", "driver_version": "unknown", "gpu_count": 0}

    def get_cuda_version():
        try:
            return torch.version.cuda or "unknown"
        except Exception:
            return "unknown"

    def percentile(values, p):
        if not values:
            return None
        s = sorted(values)
        k = (len(s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return s[f] + (k - f) * (s[c] - s[f])


# ---------------------------------------------------------------------------
# Model configs for KV geometry
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "llama-3-8b": {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
    "llama-3-70b": {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128},
}

DTYPE_BYTES = {"bf16": 2, "fp8": 1, "fp4": 0.5}


def kv_bytes(num_tokens, cfg, dtype_key):
    """Compute KV block size in bytes for a given token count and model config."""
    return int(
        cfg["num_layers"]
        * 2  # K + V
        * num_tokens
        * cfg["num_kv_heads"]
        * cfg["head_dim"]
        * DTYPE_BYTES[dtype_key]
    )


def kv_elements(num_tokens, cfg):
    """Total number of elements in a KV tensor (K+V, all layers)."""
    return cfg["num_layers"] * 2 * num_tokens * cfg["num_kv_heads"] * cfg["head_dim"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_token_counts(dataset_name):
    """Load per-session cumulative token counts from a HuggingFace dataset.

    Returns a sorted list of dicts: [{session_id, cumulative_tokens}, ...]
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    records = []
    for row in ds:
        sid = row.get("id") or row.get("session_id") or "unknown"
        stats = row.get("stats", {})
        if isinstance(stats, str):
            stats = json.loads(stats)
        tokens = stats.get("input_tokens", 0)
        if tokens and tokens > 0:
            records.append({"session_id": str(sid), "cumulative_tokens": int(tokens)})
    records.sort(key=lambda r: r["cumulative_tokens"])
    return records


def synthetic_token_counts():
    """Generate synthetic token counts for testing without a dataset."""
    targets = [512, 1024, 4096, 16384, 65536, 131072, 262144]
    return [
        {"session_id": f"synthetic_{i}", "cumulative_tokens": t}
        for i, t in enumerate(targets)
    ]


def select_measurement_points(records, max_points=20):
    """Select measurement points at roughly logarithmic intervals."""
    if len(records) <= max_points:
        return records
    min_tok = max(records[0]["cumulative_tokens"], 1)
    max_tok = records[-1]["cumulative_tokens"]
    log_min, log_max = math.log10(min_tok), math.log10(max_tok)
    targets = [10 ** (log_min + i * (log_max - log_min) / (max_points - 1))
               for i in range(max_points)]
    selected = []
    used = set()
    for target in targets:
        best_idx = min(range(len(records)),
                       key=lambda i: abs(records[i]["cumulative_tokens"] - target))
        if best_idx not in used:
            used.add(best_idx)
            selected.append(records[best_idx])
    return selected


# ---------------------------------------------------------------------------
# Transfer benchmarks
# ---------------------------------------------------------------------------
def bench_hbm_to_hbm(src_tensor, warmup=5, iters=20):
    """Benchmark HBM-to-HBM copy on the same GPU."""
    dst = torch.empty_like(src_tensor)
    # Warmup
    for _ in range(warmup):
        dst.copy_(src_tensor)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(iters):
        start_event.record()
        dst.copy_(src_tensor)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    del dst
    return latencies


def bench_ram_to_hbm(num_bytes, device, warmup=5, iters=20):
    """Benchmark pinned host RAM to HBM transfer."""
    num_elements = num_bytes // 2  # BF16 source
    host_tensor = torch.empty(num_elements, dtype=torch.bfloat16).pin_memory()
    host_tensor.fill_(1.0)

    # Warmup
    for _ in range(warmup):
        _ = host_tensor.to(device, non_blocking=False)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(iters):
        start_event.record()
        gpu_tensor = host_tensor.to(device, non_blocking=False)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        del gpu_tensor
    del host_tensor
    return latencies


def bench_disk_to_hbm(num_bytes, device, warmup=3, iters=10):
    """Benchmark disk read + H2D transfer."""
    num_elements = num_bytes // 2  # BF16
    src = torch.randn(num_elements, dtype=torch.bfloat16)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmppath = f.name
    torch.save(src, tmppath)
    del src

    # Warmup
    for _ in range(warmup):
        t = torch.load(tmppath, weights_only=True).to(device)
        del t
    torch.cuda.synchronize()

    latencies = []
    for _ in range(iters):
        # Drop page cache attempt (may need root; ignore failures)
        try:
            with open(tmppath, "rb") as f:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

        t0 = time.perf_counter()
        t = torch.load(tmppath, weights_only=True).to(device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        del t

    os.unlink(tmppath)
    return latencies


def bench_cast_fp8(src_tensor, warmup=5, iters=20):
    """Benchmark BF16 → FP8 cast on GPU."""
    fp8_dtype = torch.float8_e4m3fn

    for _ in range(warmup):
        _ = src_tensor.to(fp8_dtype)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(iters):
        start_event.record()
        _ = src_tensor.to(fp8_dtype)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    return latencies


def bench_cast_fp4_emulated(src_tensor, group_size=32, warmup=5, iters=20):
    """Benchmark emulated BF16 → FP4 quantize + bit-pack on GPU.

    This is an emulation: we quantize to INT4 range per group, compute
    per-group scales, and pack pairs of 4-bit values into uint8.
    """
    flat = src_tensor.reshape(-1).float()
    n = flat.numel()
    # Pad to group_size multiple
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))

    def _pack():
        groups = flat.view(-1, group_size)
        scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        # Quantize to [-7, 7] (INT4 range)
        quantized = (groups / scales * 7.0).round().clamp(-7, 7).to(torch.int8)
        # Bit-pack pairs into uint8: low nibble + high nibble
        even = quantized[:, 0::2] & 0x0F
        odd = (quantized[:, 1::2] & 0x0F) << 4
        packed = (even | odd).to(torch.uint8)
        return packed, scales.squeeze(1).to(torch.float16)

    # Warmup
    for _ in range(warmup):
        _pack()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(iters):
        start_event.record()
        _pack()
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    return latencies


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_cfg = MODEL_CONFIGS[args.model]

    # Load dataset
    if args.synthetic:
        print("Using synthetic token counts (no dataset download)")
        records = synthetic_token_counts()
    else:
        print(f"Loading dataset: {args.dataset}")
        records = load_token_counts(args.dataset)
        print(f"  Loaded {len(records)} sessions")

    points = select_measurement_points(records, max_points=args.max_points)
    print(f"  Selected {len(points)} measurement points")
    print(f"  Token range: {points[0]['cumulative_tokens']:,} → "
          f"{points[-1]['cumulative_tokens']:,}")

    gpu_info = get_gpu_info()
    cuda_ver = get_cuda_version()

    results = {
        "benchmark": "kv_block_loading_cost",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu_model": gpu_info["gpu_model"],
            "driver_version": gpu_info["driver_version"],
            "cuda_version": cuda_ver,
            "gpu_count": gpu_info["gpu_count"],
        },
        "dataset": {
            "name": "synthetic" if args.synthetic else args.dataset,
            "sessions_used": len(records),
            "measurement_points": len(points),
        },
        "model_config": {
            "name": args.model,
            **model_cfg,
        },
        "params": {
            "warmup_iters": args.warmup,
            "timed_iters": args.iters,
            "fp4_group_size": args.group_size,
        },
        "measurements": [],
    }

    for i, pt in enumerate(points):
        ntok = pt["cumulative_tokens"]
        n_elem = kv_elements(ntok, model_cfg)
        bytes_bf16 = kv_bytes(ntok, model_cfg, "bf16")
        bytes_fp8 = kv_bytes(ntok, model_cfg, "fp8")
        bytes_fp4 = kv_bytes(ntok, model_cfg, "fp4")

        print(f"\n[{i+1}/{len(points)}] tokens={ntok:,}  "
              f"bf16={bytes_bf16/1e6:.1f}MB  fp8={bytes_fp8/1e6:.1f}MB  "
              f"fp4={bytes_fp4/1e6:.1f}MB")

        # Check if tensor fits in GPU memory (leave 2 GB headroom for workspace)
        free_mem = torch.cuda.mem_get_info(device)[0]
        # Need ~3x for src + dst + workspace
        if bytes_bf16 * 3 > free_mem - 2 * 1024**3:
            print(f"  SKIP: tensor too large for GPU memory "
                  f"(need {bytes_bf16*3/1e9:.1f}GB, free {free_mem/1e9:.1f}GB)")
            continue

        # Allocate source tensor on GPU (BF16)
        src_gpu = torch.randn(n_elem, dtype=torch.bfloat16, device=device)

        measurement = {
            "cumulative_tokens": ntok,
            "kv_elements": n_elem,
            "kv_bytes_bf16": bytes_bf16,
            "kv_bytes_fp8": bytes_fp8,
            "kv_bytes_fp4": bytes_fp4,
            "session_id": pt["session_id"],
            "transfers": {},
            "cast_overhead": {},
        }

        # --- HBM → HBM ---
        print("  HBM→HBM ...", end="", flush=True)
        lats = bench_hbm_to_hbm(src_gpu, warmup=args.warmup, iters=args.iters)
        med = percentile(lats, 50)
        measurement["transfers"]["hbm_to_hbm"] = {
            "bf16": {
                "latency_ms_median": round(med, 4),
                "latency_ms_p5": round(percentile(lats, 5), 4),
                "latency_ms_p95": round(percentile(lats, 95), 4),
                "throughput_gbps": round(bytes_bf16 / (med / 1000) / 1e9, 2)
                if med > 0 else 0,
            }
        }
        print(f" {med:.3f}ms  {bytes_bf16/(med/1000)/1e9:.1f} GB/s")

        # --- RAM → HBM ---
        print("  RAM→HBM ...", end="", flush=True)
        lats = bench_ram_to_hbm(bytes_bf16, device, warmup=args.warmup,
                                iters=args.iters)
        med = percentile(lats, 50)
        measurement["transfers"]["ram_to_hbm"] = {
            "bf16": {
                "latency_ms_median": round(med, 4),
                "latency_ms_p5": round(percentile(lats, 5), 4),
                "latency_ms_p95": round(percentile(lats, 95), 4),
                "throughput_gbps": round(bytes_bf16 / (med / 1000) / 1e9, 2)
                if med > 0 else 0,
            }
        }
        print(f" {med:.3f}ms  {bytes_bf16/(med/1000)/1e9:.1f} GB/s")

        # --- Disk → HBM ---
        if not args.skip_disk:
            print("  Disk→HBM ...", end="", flush=True)
            # Only run disk bench for blocks up to 256 MB to keep runtime sane
            if bytes_bf16 <= 256 * 1024 * 1024:
                lats = bench_disk_to_hbm(bytes_bf16, device, warmup=2,
                                         iters=max(3, args.iters // 3))
                med = percentile(lats, 50)
                measurement["transfers"]["disk_to_hbm"] = {
                    "bf16": {
                        "latency_ms_median": round(med, 4),
                        "latency_ms_p5": round(percentile(lats, 5), 4),
                        "latency_ms_p95": round(percentile(lats, 95), 4),
                        "throughput_gbps": round(
                            bytes_bf16 / (med / 1000) / 1e9, 2)
                        if med > 0 else 0,
                    }
                }
                print(f" {med:.3f}ms  {bytes_bf16/(med/1000)/1e9:.1f} GB/s")
            else:
                print(" SKIP (>256MB)")
                measurement["transfers"]["disk_to_hbm"] = {"bf16": "skipped_too_large"}

        # --- Cast BF16 → FP8 ---
        print("  Cast BF16→FP8 ...", end="", flush=True)
        lats = bench_cast_fp8(src_gpu, warmup=args.warmup, iters=args.iters)
        med = percentile(lats, 50)
        measurement["cast_overhead"]["bf16_to_fp8"] = {
            "latency_ms_median": round(med, 4),
            "latency_ms_p5": round(percentile(lats, 5), 4),
            "latency_ms_p95": round(percentile(lats, 95), 4),
            "elements": n_elem,
        }
        print(f" {med:.3f}ms")

        # --- Cast BF16 → FP4 (emulated) ---
        print("  Cast BF16→FP4(emu) ...", end="", flush=True)
        # Use a smaller tensor if the full one is too large for the pack op
        if n_elem <= 128 * 1024 * 1024:  # ~128M elements max for pack
            lats = bench_cast_fp4_emulated(src_gpu, group_size=args.group_size,
                                           warmup=args.warmup, iters=args.iters)
            med = percentile(lats, 50)
            measurement["cast_overhead"]["bf16_to_fp4_emulated"] = {
                "latency_ms_median": round(med, 4),
                "latency_ms_p5": round(percentile(lats, 5), 4),
                "latency_ms_p95": round(percentile(lats, 95), 4),
                "elements": n_elem,
                "group_size": args.group_size,
            }
            print(f" {med:.3f}ms")
        else:
            print(" SKIP (too large for emulated pack)")
            measurement["cast_overhead"]["bf16_to_fp4_emulated"] = "skipped_too_large"

        # --- Combined: transfer + cast throughput ---
        # Effective throughput for FP8 destination = bytes_bf16 transferred + cast
        hbm_bf16_ms = measurement["transfers"]["hbm_to_hbm"]["bf16"]["latency_ms_median"]
        fp8_cast_ms = measurement["cast_overhead"]["bf16_to_fp8"]["latency_ms_median"]
        combined_fp8_ms = hbm_bf16_ms + fp8_cast_ms
        measurement["transfers"]["hbm_to_hbm"]["fp8_combined"] = {
            "transfer_plus_cast_ms": round(combined_fp8_ms, 4),
            "effective_throughput_gbps": round(
                bytes_fp8 / (combined_fp8_ms / 1000) / 1e9, 2)
            if combined_fp8_ms > 0 else 0,
            "note": "BF16 transfer + FP8 cast; effective bytes = FP8 output size",
        }

        ram_entry = measurement["transfers"].get("ram_to_hbm", {}).get("bf16", {})
        if isinstance(ram_entry, dict) and "latency_ms_median" in ram_entry:
            ram_bf16_ms = ram_entry["latency_ms_median"]
            combined_ram_fp8 = ram_bf16_ms + fp8_cast_ms
            measurement["transfers"]["ram_to_hbm"]["fp8_combined"] = {
                "transfer_plus_cast_ms": round(combined_ram_fp8, 4),
                "effective_throughput_gbps": round(
                    bytes_fp8 / (combined_ram_fp8 / 1000) / 1e9, 2)
                if combined_ram_fp8 > 0 else 0,
            }

        del src_gpu
        torch.cuda.empty_cache()

        results["measurements"].append(measurement)

    # --- RDMA (theoretical only) ---
    results["rdma_theoretical"] = {
        "note": "No live RDMA measurement; spec-sheet ceilings only",
        "infiniband_ndr_400g_gbps": 50,
        "infiniband_hdr_200g_gbps": 25,
    }

    # Write output
    os.makedirs(os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(_REPO_ROOT, "results", f"kv_block_load_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="KV block loading cost benchmark (dataset-driven)")
    parser.add_argument("--dataset", default="lelouch0110/claudeset-community",
                        help="HuggingFace dataset name")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic token counts (no download)")
    parser.add_argument("--model", default="llama-3-8b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model config for KV geometry")
    parser.add_argument("--max-points", type=int, default=15,
                        help="Max measurement points (log-spaced)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations per measurement")
    parser.add_argument("--iters", type=int, default=20,
                        help="Timed iterations per measurement")
    parser.add_argument("--group-size", type=int, default=32,
                        help="Group size for emulated FP4 quantization")
    parser.add_argument("--skip-disk", action="store_true",
                        help="Skip disk benchmarks")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
