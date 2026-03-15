#!/usr/bin/env python3
"""KV block offload/restore cost benchmark.

Measures the true cost of offloading and restoring KV cache blocks between
GPU HBM, host RAM (pinned), and disk across BF16, pre-quantized FP8, and
emulated pre-packed FP4 formats.

Sweeps block size (tokens) and batch count (blocks per transfer) to show
how per-block cost and effective throughput scale.  All buffers are
pre-allocated so the timed path measures pure DMA, not allocation.

Usage:
    # Quick synthetic test (no dataset, no disk):
    python scripts/bench_kv_block_load.py --synthetic --skip-disk

    # Full run with dataset-driven batch counts:
    python scripts/bench_kv_block_load.py --dataset lelouch0110/claudeset-community

Output:
    results/kv_block_offload_restore_<timestamp>.json
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
# Model configs for KV block geometry
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "llama-3-8b": {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
    "llama-3-70b": {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128},
}

# Bytes per element for each storage format
FORMAT_BYTES = {"bf16": 2, "fp8": 1, "fp4": 0.5}


def block_bytes(block_tokens, cfg, fmt):
    """Bytes in one KV block: layers * 2(K+V) * tokens * kv_heads * head_dim * fmt_bytes."""
    return int(
        cfg["num_layers"] * 2 * block_tokens
        * cfg["num_kv_heads"] * cfg["head_dim"] * FORMAT_BYTES[fmt]
    )


def block_elements(block_tokens, cfg):
    """Total scalar elements in one KV block (K+V, all layers)."""
    return cfg["num_layers"] * 2 * block_tokens * cfg["num_kv_heads"] * cfg["head_dim"]


# ---------------------------------------------------------------------------
# Dataset loading — drives batch count sweep
# ---------------------------------------------------------------------------
def load_session_block_counts(dataset_name, block_tokens, cfg):
    """Load dataset sessions and compute how many KV blocks each occupies."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")
    counts = []
    for row in ds:
        sid = row.get("id") or row.get("session_id") or "unknown"
        stats = row.get("stats", {})
        if isinstance(stats, str):
            stats = json.loads(stats)
        tokens = stats.get("input_tokens", 0)
        if tokens and tokens > 0:
            nblocks = max(1, math.ceil(tokens / block_tokens))
            counts.append({"session_id": str(sid), "tokens": int(tokens),
                           "num_blocks": nblocks})
    counts.sort(key=lambda r: r["num_blocks"])
    return counts


def derive_batch_counts(session_records, base_counts):
    """Merge fixed batch counts with dataset-derived block counts."""
    dataset_counts = sorted(set(r["num_blocks"] for r in session_records))
    # Pick a few representative dataset-derived counts (log-spaced)
    if len(dataset_counts) > 5:
        indices = [int(i * (len(dataset_counts) - 1) / 4) for i in range(5)]
        dataset_counts = sorted(set(dataset_counts[i] for i in indices))
    all_counts = sorted(set(base_counts) | set(dataset_counts))
    return all_counts


# ---------------------------------------------------------------------------
# Pre-allocated buffer manager
# ---------------------------------------------------------------------------
class TransferBuffers:
    """Pre-allocated GPU and pinned-host buffers for clean DMA measurement."""

    def __init__(self, max_bytes, device):
        self.device = device
        self.max_bytes = max_bytes
        # GPU buffers
        self.gpu_src = torch.empty(max_bytes, dtype=torch.uint8, device=device)
        self.gpu_dst = torch.empty(max_bytes, dtype=torch.uint8, device=device)
        # Pinned host buffer
        self.host_buf = torch.empty(max_bytes, dtype=torch.uint8,
                                    pin_memory=True)
        # Fill with non-zero data to avoid zero-page optimizations
        self.gpu_src.fill_(42)
        self.host_buf.fill_(42)

    def gpu_view(self, nbytes, target="src"):
        buf = self.gpu_src if target == "src" else self.gpu_dst
        return buf[:nbytes]

    def host_view(self, nbytes):
        return self.host_buf[:nbytes]

    def cleanup(self):
        del self.gpu_src, self.gpu_dst, self.host_buf
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Timed transfer primitives
# ---------------------------------------------------------------------------
def _stats(latencies):
    return {
        "latency_ms_median": round(percentile(latencies, 50), 4),
        "latency_ms_p5": round(percentile(latencies, 5), 4),
        "latency_ms_p95": round(percentile(latencies, 95), 4),
    }


def bench_gpu_copy(src, dst, warmup, iters):
    """Benchmark HBM-to-HBM copy with pre-allocated buffers."""
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(iters):
        start.record()
        dst.copy_(src)
        end.record()
        torch.cuda.synchronize()
        lats.append(start.elapsed_time(end))
    return lats


def bench_offload_to_host(gpu_src, host_dst, warmup, iters):
    """GPU → pinned host (D2H) with pre-allocated buffers."""
    for _ in range(warmup):
        host_dst.copy_(gpu_src)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(iters):
        start.record()
        host_dst.copy_(gpu_src)
        end.record()
        torch.cuda.synchronize()
        lats.append(start.elapsed_time(end))
    return lats


def bench_restore_from_host(host_src, gpu_dst, warmup, iters):
    """Pinned host → GPU (H2D) with pre-allocated buffers."""
    for _ in range(warmup):
        gpu_dst.copy_(host_src)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(iters):
        start.record()
        gpu_dst.copy_(host_src)
        end.record()
        torch.cuda.synchronize()
        lats.append(start.elapsed_time(end))
    return lats


def bench_offload_to_disk(gpu_src, host_tmp, disk_path, nbytes, warmup, iters):
    """GPU → pinned host → disk (D2H + raw write)."""
    # Ensure file exists and is the right size
    with open(disk_path, "wb") as f:
        f.write(b"\x00" * nbytes)

    for _ in range(warmup):
        host_tmp.copy_(gpu_src)
        torch.cuda.synchronize()
        fd = os.open(disk_path, os.O_WRONLY)
        os.write(fd, host_tmp.numpy().tobytes())
        os.close(fd)

    lats = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        host_tmp.copy_(gpu_src)
        torch.cuda.synchronize()
        fd = os.open(disk_path, os.O_WRONLY)
        os.write(fd, host_tmp.numpy().tobytes())
        os.fsync(fd)
        os.close(fd)
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)
    return lats


def bench_restore_from_disk(disk_path, host_tmp, gpu_dst, nbytes, warmup, iters):
    """Disk → pinned host → GPU (raw read + H2D)."""
    for _ in range(warmup):
        # Drop page cache
        try:
            with open(disk_path, "rb") as fh:
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        fd = os.open(disk_path, os.O_RDONLY)
        raw = os.read(fd, nbytes)
        os.close(fd)
        host_tmp[:len(raw)].copy_(
            torch.frombuffer(bytearray(raw), dtype=torch.uint8))
        gpu_dst.copy_(host_tmp)
        torch.cuda.synchronize()

    lats = []
    for _ in range(iters):
        # Drop page cache
        try:
            with open(disk_path, "rb") as fh:
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

        t0 = time.perf_counter()
        fd = os.open(disk_path, os.O_RDONLY)
        raw = os.read(fd, nbytes)
        os.close(fd)
        host_tmp[:len(raw)].copy_(
            torch.frombuffer(bytearray(raw), dtype=torch.uint8))
        gpu_dst.copy_(host_tmp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)
    return lats


# ---------------------------------------------------------------------------
# Cast / quantize benchmarks (isolated, on GPU)
# ---------------------------------------------------------------------------
def bench_cast_bf16_to_fp8(src_bf16, warmup, iters):
    """BF16 → FP8 quantize on GPU (pre-offload cast)."""
    fp8 = torch.float8_e4m3fn
    for _ in range(warmup):
        src_bf16.to(fp8)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(iters):
        s.record(); src_bf16.to(fp8); e.record()
        torch.cuda.synchronize(); lats.append(s.elapsed_time(e))
    return lats


def bench_cast_fp8_to_bf16(src_fp8, warmup, iters):
    """FP8 → BF16 dequantize on GPU (post-restore cast)."""
    for _ in range(warmup):
        src_fp8.to(torch.bfloat16)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(iters):
        s.record(); src_fp8.to(torch.bfloat16); e.record()
        torch.cuda.synchronize(); lats.append(s.elapsed_time(e))
    return lats


def bench_pack_fp4(src_bf16, group_size, warmup, iters):
    """BF16 → emulated FP4 pack on GPU (pre-offload)."""
    flat = src_bf16.reshape(-1).float()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))

    def _pack():
        groups = flat.view(-1, group_size)
        scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        q = (groups / scales * 7.0).round().clamp(-7, 7).to(torch.int8)
        even = q[:, 0::2] & 0x0F
        odd = (q[:, 1::2] & 0x0F) << 4
        return (even | odd).to(torch.uint8), scales.squeeze(1).half()

    for _ in range(warmup):
        _pack()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(iters):
        s.record(); _pack(); e.record()
        torch.cuda.synchronize(); lats.append(s.elapsed_time(e))
    return lats


def bench_unpack_fp4(packed, scales, group_size, warmup, iters):
    """Emulated FP4 unpack → FP32 on GPU (post-restore)."""
    def _unpack():
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        # Sign-extend from 4-bit
        low = (low << 4) >> 4
        high = (high << 4) >> 4
        # Interleave
        n_groups = packed.shape[0]
        half_gs = packed.shape[1]
        out = torch.empty(n_groups, half_gs * 2, dtype=torch.float32,
                          device=packed.device)
        out[:, 0::2] = low.float()
        out[:, 1::2] = high.float()
        return out * scales.unsqueeze(1) / 7.0

    for _ in range(warmup):
        _unpack()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(iters):
        s.record(); _unpack(); e.record()
        torch.cuda.synchronize(); lats.append(s.elapsed_time(e))
    return lats


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = MODEL_CONFIGS[args.model]

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    base_batch_counts = [int(x) for x in args.batch_counts.split(",")]

    # Dataset-driven batch counts
    if not args.synthetic:
        print(f"Loading dataset: {args.dataset}")
        session_records = load_session_block_counts(
            args.dataset, block_sizes[2] if len(block_sizes) > 2 else 16, cfg)
        print(f"  {len(session_records)} sessions loaded")
        batch_counts = derive_batch_counts(session_records, base_batch_counts)
        print(f"  Batch counts (merged): {batch_counts}")
    else:
        print("Using synthetic mode (fixed batch counts)")
        batch_counts = base_batch_counts
        session_records = []

    # Compute max buffer size needed
    max_block_tok = max(block_sizes)
    max_batch = max(batch_counts)
    max_bytes_bf16 = block_bytes(max_block_tok, cfg, "bf16") * max_batch
    # Cap at 2 GB to avoid OOM
    max_buf_bytes = min(max_bytes_bf16, 2 * 1024**3)

    print(f"\nMax buffer: {max_buf_bytes / 1e6:.1f} MB")
    print(f"Block sizes (tokens): {block_sizes}")
    print(f"Batch counts: {batch_counts}")

    gpu_info = get_gpu_info()
    bufs = TransferBuffers(max_buf_bytes, device)

    # Disk temp file
    disk_path = None
    if not args.skip_disk:
        tmpdir = os.path.join(_REPO_ROOT, "results")
        os.makedirs(tmpdir, exist_ok=True)
        disk_path = os.path.join(tmpdir, ".bench_tmp_block.raw")

    results = {
        "benchmark": "kv_block_offload_restore",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpu_model": gpu_info["gpu_model"],
            "driver_version": gpu_info["driver_version"],
            "cuda_version": get_cuda_version(),
            "gpu_count": gpu_info["gpu_count"],
        },
        "model_config": {"name": args.model, **cfg},
        "dataset": {
            "name": "synthetic" if args.synthetic else args.dataset,
            "sessions": len(session_records),
        },
        "params": {
            "warmup": args.warmup, "iters": args.iters,
            "block_sizes_tokens": block_sizes,
            "batch_counts": batch_counts,
            "fp4_group_size": args.group_size,
        },
        "measurements": [],
        "rdma_theoretical": {
            "note": "No live RDMA measurement; spec-sheet ceilings only",
            "infiniband_ndr_400g_gbps": 50,
            "infiniband_hdr_200g_gbps": 25,
        },
    }

    total_combos = len(block_sizes) * len(batch_counts) * 3  # 3 formats
    combo_idx = 0

    for blk_tok in block_sizes:
        for batch in batch_counts:
            for fmt in ["bf16", "fp8", "fp4"]:
                combo_idx += 1
                bpb = block_bytes(blk_tok, cfg, fmt)
                total = bpb * batch
                elem_per_block = block_elements(blk_tok, cfg)
                total_elem = elem_per_block * batch

                if total > max_buf_bytes:
                    print(f"[{combo_idx}/{total_combos}] SKIP "
                          f"blk={blk_tok}tok batch={batch} fmt={fmt} "
                          f"({total/1e6:.1f}MB > buf cap)")
                    continue

                print(f"\n[{combo_idx}/{total_combos}] blk={blk_tok}tok "
                      f"batch={batch} fmt={fmt}  "
                      f"per_block={bpb/1e3:.1f}KB  total={total/1e6:.2f}MB")

                m = {
                    "block_size_tokens": blk_tok,
                    "batch_count": batch,
                    "format": fmt,
                    "bytes_per_block": bpb,
                    "total_bytes": total,
                    "elements_per_block": elem_per_block,
                }

                gpu_src = bufs.gpu_view(total, "src")
                gpu_dst = bufs.gpu_view(total, "dst")
                host_v = bufs.host_view(total)

                # --- HBM ↔ HBM ---
                lats = bench_gpu_copy(gpu_src, gpu_dst, args.warmup, args.iters)
                med = percentile(lats, 50)
                m["hbm_copy"] = {
                    **_stats(lats),
                    "throughput_gbps": round(total / (med / 1000) / 1e9, 2)
                    if med > 0 else 0,
                }
                print(f"  HBM copy: {med:.3f}ms  "
                      f"{total/(med/1000)/1e9:.1f} GB/s")

                # --- Offload: GPU → host ---
                lats = bench_offload_to_host(gpu_src, host_v,
                                             args.warmup, args.iters)
                med = percentile(lats, 50)
                m["offload_to_host"] = {
                    **_stats(lats),
                    "throughput_gbps": round(total / (med / 1000) / 1e9, 2)
                    if med > 0 else 0,
                }
                print(f"  Offload→host: {med:.3f}ms  "
                      f"{total/(med/1000)/1e9:.1f} GB/s")

                # --- Restore: host → GPU ---
                lats = bench_restore_from_host(host_v, gpu_dst,
                                               args.warmup, args.iters)
                med = percentile(lats, 50)
                m["restore_from_host"] = {
                    **_stats(lats),
                    "throughput_gbps": round(total / (med / 1000) / 1e9, 2)
                    if med > 0 else 0,
                }
                print(f"  Restore←host: {med:.3f}ms  "
                      f"{total/(med/1000)/1e9:.1f} GB/s")

                # --- Disk paths ---
                if not args.skip_disk and total <= 128 * 1024 * 1024:
                    lats = bench_offload_to_disk(
                        gpu_src, host_v, disk_path, total,
                        min(2, args.warmup), max(3, args.iters // 3))
                    med = percentile(lats, 50)
                    m["offload_to_disk"] = {
                        **_stats(lats),
                        "throughput_gbps": round(total / (med / 1000) / 1e9, 2)
                        if med > 0 else 0,
                    }
                    print(f"  Offload→disk: {med:.3f}ms  "
                          f"{total/(med/1000)/1e9:.1f} GB/s")

                    lats = bench_restore_from_disk(
                        disk_path, host_v, gpu_dst, total,
                        min(2, args.warmup), max(3, args.iters // 3))
                    med = percentile(lats, 50)
                    m["restore_from_disk"] = {
                        **_stats(lats),
                        "throughput_gbps": round(total / (med / 1000) / 1e9, 2)
                        if med > 0 else 0,
                    }
                    print(f"  Restore←disk: {med:.3f}ms  "
                          f"{total/(med/1000)/1e9:.1f} GB/s")
                elif not args.skip_disk:
                    m["offload_to_disk"] = "skipped_too_large"
                    m["restore_from_disk"] = "skipped_too_large"

                # --- Cast overhead (format-specific) ---
                if fmt == "fp8":
                    src_bf16 = torch.randn(total_elem, dtype=torch.bfloat16,
                                           device=device)
                    lats_q = bench_cast_bf16_to_fp8(src_bf16, args.warmup,
                                                    args.iters)
                    src_fp8 = src_bf16.to(torch.float8_e4m3fn)
                    lats_dq = bench_cast_fp8_to_bf16(src_fp8, args.warmup,
                                                     args.iters)
                    m["cast_overhead"] = {
                        "quantize_bf16_to_fp8": _stats(lats_q),
                        "dequant_fp8_to_bf16": _stats(lats_dq),
                    }
                    del src_bf16, src_fp8
                    print(f"  FP8 quant: {percentile(lats_q,50):.3f}ms  "
                          f"dequant: {percentile(lats_dq,50):.3f}ms")

                elif fmt == "fp4" and total_elem <= 64 * 1024 * 1024:
                    src_bf16 = torch.randn(total_elem, dtype=torch.bfloat16,
                                           device=device)
                    lats_q = bench_pack_fp4(src_bf16, args.group_size,
                                            args.warmup, args.iters)
                    # Get packed for unpack bench
                    flat = src_bf16.reshape(-1).float()
                    pad = (args.group_size - flat.numel() % args.group_size) \
                        % args.group_size
                    if pad > 0:
                        flat = torch.nn.functional.pad(flat, (0, pad))
                    groups = flat.view(-1, args.group_size)
                    scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                    q = (groups / scales * 7.0).round().clamp(-7, 7).to(torch.int8)
                    even = q[:, 0::2] & 0x0F
                    odd = (q[:, 1::2] & 0x0F) << 4
                    packed = (even | odd).to(torch.uint8)
                    scales_h = scales.squeeze(1).half()

                    lats_dq = bench_unpack_fp4(packed, scales_h,
                                               args.group_size,
                                               args.warmup, args.iters)
                    m["cast_overhead"] = {
                        "pack_bf16_to_fp4": _stats(lats_q),
                        "unpack_fp4_to_f32": _stats(lats_dq),
                    }
                    del src_bf16, flat, groups, scales, q, even, odd, packed, scales_h
                    print(f"  FP4 pack: {percentile(lats_q,50):.3f}ms  "
                          f"unpack: {percentile(lats_dq,50):.3f}ms")

                torch.cuda.empty_cache()
                results["measurements"].append(m)

    bufs.cleanup()

    # Clean up disk temp
    if disk_path and os.path.exists(disk_path):
        os.unlink(disk_path)

    # Write output
    os.makedirs(os.path.join(_REPO_ROOT, "results"), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(_REPO_ROOT, "results",
                            f"kv_block_offload_restore_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description="KV block offload/restore cost benchmark")
    p.add_argument("--dataset", default="lelouch0110/claudeset-community")
    p.add_argument("--synthetic", action="store_true",
                   help="Fixed batch counts, no dataset download")
    p.add_argument("--model", default="llama-3-8b",
                   choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--block-sizes", default="1,4,16,64,256",
                   help="Comma-separated block sizes in tokens")
    p.add_argument("--batch-counts", default="1,4,16,64",
                   help="Comma-separated batch counts (blocks per transfer)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--group-size", type=int, default=32,
                   help="Group size for emulated FP4")
    p.add_argument("--skip-disk", action="store_true")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
