#!/usr/bin/env python3
"""Measure host-to-device KV block transfer cost for offload policy analysis.

The key insight: at concurrency=1, small transfers are latency-bound and show
artificially low bandwidth. In real serving, multiple sequences may be restoring
KV blocks concurrently, which queues up on the PCIe bus. This script sweeps:
  - Block size (tokens per block)
  - Dtype (fp16, fp8, nvfp4 → different bytes per block)
  - Concurrent transfers (1, 2, 4, 8, ... simultaneous H2D copies)

For each combination, it reports:
  - Per-block latency (wall-clock time from issue to completion)
  - Effective per-block throughput
  - Aggregate PCIe bandwidth utilization

Usage:
    # Quick single-block measurement
    python scripts/sweep_load_cost.py \
        --num-layers 48 --num-kv-heads 4 --head-dim 64 \
        --block-tokens 32 --dtypes fp16,fp8

    # Full sweep with concurrency and compute load
    python scripts/sweep_load_cost.py \
        --num-layers 48 --num-kv-heads 4 --head-dim 64 \
        --block-tokens 32 --dtypes fp16,fp8 \
        --concurrent-blocks 1,2,4,8,16,32,64 \
        --with-compute-load --per-layer
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_bench_utils import get_gpu_info, get_cuda_version, write_result_json


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep H2D KV block load cost for offload policy"
    )
    # Model KV dimensions
    p.add_argument("--num-layers", type=int, default=48,
                   help="Number of transformer layers")
    p.add_argument("--num-kv-heads", type=int, default=4,
                   help="Number of KV attention heads (after GQA)")
    p.add_argument("--head-dim", type=int, default=64,
                   help="Dimension per attention head")

    # Sweep parameters
    p.add_argument("--block-tokens", type=str, default="32",
                   help="Comma-separated block sizes in tokens to test")
    p.add_argument("--dtypes", type=str, default="fp16,fp8",
                   help="Comma-separated dtype names (fp16=2B, fp8=1B, nvfp4=0.5B)")
    p.add_argument("--concurrent-blocks", type=str, default="1,2,4,8,16,32,64",
                   help="Comma-separated number of blocks to transfer concurrently. "
                        "Simulates multiple sequences restoring KV at once.")

    # Measurement
    p.add_argument("--warmup-iters", type=int, default=20,
                   help="Warmup H2D copies before timing")
    p.add_argument("--measure-iters", type=int, default=50,
                   help="Timed iterations to average over")
    p.add_argument("--with-compute-load", action="store_true",
                   help="Run background matmul kernels during transfers")
    p.add_argument("--compute-size", type=int, default=4096,
                   help="Matrix dimension for background compute load")

    # Per-layer measurement
    p.add_argument("--per-layer", action="store_true",
                   help="Also measure single-layer transfer cost")

    p.add_argument("--output", default=None,
                   help="Output JSON path")
    return p.parse_args()


def bytes_per_element(dtype_name):
    """Return bytes per KV element for each dtype."""
    return {"fp16": 2, "bf16": 2, "fp8": 1, "nvfp4": 0.5}.get(dtype_name, 2)


def kv_block_bytes(num_layers, num_kv_heads, head_dim, block_tokens, dtype_name):
    """Compute total bytes for one KV block across all layers.

    Block stores both K and V: 2 * num_layers * num_kv_heads * head_dim * block_tokens * bpe
    For NVFP4, add ~6.25% scale overhead (1 FP8 scale per 16 values).
    """
    bpe = bytes_per_element(dtype_name)
    raw = 2 * num_layers * num_kv_heads * head_dim * block_tokens * bpe

    if dtype_name == "nvfp4":
        num_micro_blocks = (num_kv_heads * head_dim * block_tokens) // 16
        scale_bytes = num_micro_blocks * 1 + 4  # FP8 per micro-block + FP32 per-tensor
        raw += 2 * num_layers * scale_bytes

    return int(raw)


def measure_concurrent_h2d(host_bufs, gpu_bufs, warmup, iters, copy_stream):
    """Measure concurrent H2D copies: issue N copies, wait for all.

    This measures the realistic per-block cost when multiple blocks are
    being restored simultaneously (queuing on the PCIe bus).

    Returns (per_block_mean_ms, per_block_min_ms, per_block_max_ms,
             total_mean_ms, aggregate_bw_gbps).
    """
    import torch

    n = len(host_bufs)
    total_bytes = sum(h.numel() * h.element_size() for h in host_bufs)

    # Warmup
    for _ in range(warmup):
        with torch.cuda.stream(copy_stream):
            for h, g in zip(host_bufs, gpu_bufs):
                g.copy_(h, non_blocking=True)
        copy_stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_times_ms = []

    for _ in range(iters):
        start.record(copy_stream)
        with torch.cuda.stream(copy_stream):
            for h, g in zip(host_bufs, gpu_bufs):
                g.copy_(h, non_blocking=True)
        end.record(copy_stream)
        copy_stream.synchronize()
        total_times_ms.append(start.elapsed_time(end))

    total_mean = sum(total_times_ms) / len(total_times_ms)
    total_min = min(total_times_ms)
    total_max = max(total_times_ms)
    per_block_mean = total_mean / n
    per_block_min = total_min / n
    per_block_max = total_max / n
    agg_bw = (total_bytes / 1e9) / (total_mean / 1e3) if total_mean > 0 else 0

    return per_block_mean, per_block_min, per_block_max, total_mean, agg_bw, total_times_ms


def run_background_matmul(size, stop_event):
    """Background thread running continuous matmuls to load the GPU."""
    import torch
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        while not stop_event.is_set():
            torch.mm(a, b)


def main():
    args = parse_args()
    import torch
    import threading

    block_token_sizes = [int(x) for x in args.block_tokens.split(",")]
    dtypes = [x.strip() for x in args.dtypes.split(",")]
    concurrencies = [int(x) for x in args.concurrent_blocks.split(",")]

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/load_curve_{ts}.json"

    gpu_info = get_gpu_info()
    cuda_ver = get_cuda_version()

    print("=== KV Block Load Cost Sweep (with Concurrency) ===")
    print(f"Layers:       {args.num_layers}")
    print(f"KV heads:     {args.num_kv_heads}")
    print(f"Head dim:     {args.head_dim}")
    print(f"Block sizes:  {block_token_sizes} tokens")
    print(f"Dtypes:       {dtypes}")
    print(f"Concurrency:  {concurrencies} blocks")
    print(f"Warmup:       {args.warmup_iters} iters")
    print(f"Measure:      {args.measure_iters} iters")
    print(f"Compute load: {args.with_compute_load}")
    if args.per_layer:
        print(f"Per-layer:    enabled")
    print(f"Output:       {args.output}")
    print()

    # Optional background compute load
    stop_event = threading.Event()
    bg_thread = None
    if args.with_compute_load:
        print(f"Starting background matmul ({args.compute_size}x{args.compute_size})...")
        bg_thread = threading.Thread(
            target=run_background_matmul,
            args=(args.compute_size, stop_event),
            daemon=True,
        )
        bg_thread.start()
        time.sleep(0.5)

    copy_stream = torch.cuda.Stream()
    measurements = []

    for dtype_name in dtypes:
        for bt in block_token_sizes:
            nbytes = kv_block_bytes(
                args.num_layers, args.num_kv_heads, args.head_dim, bt, dtype_name
            )
            size_mb = nbytes / (1024 ** 2)

            print(f"\n=== {dtype_name}, {bt} tokens, {size_mb:.2f} MB per block ===")

            for nc in concurrencies:
                total_mb = size_mb * nc
                print(f"\n  Concurrent blocks: {nc} ({total_mb:.1f} MB total)")

                # Allocate nc independent host+device buffer pairs
                try:
                    host_bufs = [
                        torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                        for _ in range(nc)
                    ]
                    gpu_bufs = [
                        torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                        for _ in range(nc)
                    ]
                except torch.cuda.OutOfMemoryError:
                    print(f"    OOM allocating {nc} blocks, skipping")
                    break

                # Fill with non-zero data
                for h in host_bufs:
                    h.random_(0, 256)

                pb_mean, pb_min, pb_max, tot_mean, agg_bw, _ = measure_concurrent_h2d(
                    host_bufs, gpu_bufs,
                    args.warmup_iters, args.measure_iters,
                    copy_stream,
                )

                entry = {
                    "dtype": dtype_name,
                    "block_tokens": bt,
                    "block_bytes": nbytes,
                    "block_mb": round(size_mb, 3),
                    "concurrent_blocks": nc,
                    "total_transfer_mb": round(total_mb, 3),
                    "per_block_ms_mean": round(pb_mean, 4),
                    "per_block_ms_min": round(pb_min, 4),
                    "per_block_ms_max": round(pb_max, 4),
                    "total_ms_mean": round(tot_mean, 4),
                    "aggregate_bandwidth_gbps": round(agg_bw, 2),
                    "with_compute_load": args.with_compute_load,
                }

                print(f"    Per-block: {pb_mean:.4f} ms "
                      f"(min={pb_min:.4f}, max={pb_max:.4f})")
                print(f"    Total:     {tot_mean:.3f} ms for {nc} blocks")
                print(f"    Agg BW:    {agg_bw:.2f} GB/s")

                # Optional per-layer measurement at this concurrency
                if args.per_layer and nc == 1:
                    layer_bytes = nbytes // args.num_layers
                    h_layer = [torch.empty(layer_bytes, dtype=torch.uint8, pin_memory=True)]
                    g_layer = [torch.empty(layer_bytes, dtype=torch.uint8, device="cuda")]
                    h_layer[0].random_(0, 256)

                    l_pb, l_min, l_max, l_tot, l_bw, _ = measure_concurrent_h2d(
                        h_layer, g_layer,
                        args.warmup_iters, args.measure_iters,
                        copy_stream,
                    )
                    entry["per_layer_bytes"] = layer_bytes
                    entry["per_layer_ms_mean"] = round(l_pb, 4)
                    entry["per_layer_bandwidth_gbps"] = round(l_bw, 2)
                    print(f"    Per-layer: {l_pb:.4f} ms "
                          f"({layer_bytes / 1024:.1f} KB, {l_bw:.2f} GB/s)")
                    del h_layer, g_layer

                measurements.append(entry)

                # Free buffers
                del host_bufs, gpu_bufs

    if bg_thread:
        stop_event.set()
        bg_thread.join(timeout=2)

    # Write output
    output = {
        "sweep_type": "load_cost",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "num_layers": args.num_layers,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_token_sizes": block_token_sizes,
            "dtypes": dtypes,
            "concurrent_blocks": concurrencies,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "with_compute_load": args.with_compute_load,
            "per_layer": args.per_layer,
        },
        "hardware": {
            "gpu_name": gpu_info["gpu_model"],
            "gpu_count": gpu_info["gpu_count"],
            "driver_version": gpu_info["driver_version"],
            "cuda_version": cuda_ver,
        },
        "measurements": measurements,
        "rerun_command": (
            f"python scripts/sweep_load_cost.py "
            f"--num-layers {args.num_layers} --num-kv-heads {args.num_kv_heads} "
            f"--head-dim {args.head_dim} "
            f"--block-tokens {args.block_tokens} "
            f"--dtypes {args.dtypes} "
            f"--concurrent-blocks {args.concurrent_blocks} "
            f"--warmup-iters {args.warmup_iters} "
            f"--measure-iters {args.measure_iters} "
            f"{'--with-compute-load ' if args.with_compute_load else ''}"
            f"{'--per-layer ' if args.per_layer else ''}"
            f"--output {args.output}"
        ),
    }

    write_result_json(output, args.output)

    # Summary table: per-block ms at each concurrency level
    print("\n=== Load Cost Summary: Per-Block ms by Concurrency ===")
    for dtype_name in dtypes:
        for bt in block_token_sizes:
            print(f"\n  {dtype_name}, {bt} tokens:")
            header = f"    {'Conc':>6}  {'Per-blk ms':>12}  {'Total ms':>10}  {'BW GB/s':>10}"
            print(header)
            print(f"    {'-' * 44}")
            for m in measurements:
                if m["dtype"] == dtype_name and m["block_tokens"] == bt:
                    print(f"    {m['concurrent_blocks']:>6}  "
                          f"{m['per_block_ms_mean']:>12.4f}  "
                          f"{m['total_ms_mean']:>10.3f}  "
                          f"{m['aggregate_bandwidth_gbps']:>10.2f}")


if __name__ == "__main__":
    main()
