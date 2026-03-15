#!/usr/bin/env python3
"""Measure incremental prefill (recompute) cost as a function of context position
under realistic GPU saturation from concurrent decode batches.

The key insight: at batch=1, a 30B MoE model barely loads the GPU, so recompute
looks artificially expensive per-block (you're paying full kernel launch overhead
for underutilized SMs). In a real serving scenario, the GPU is already saturated
with decode work. The relevant metric is the *marginal* cost of squeezing in a
recompute while the GPU is busy.

Approach:
  - For each context position, build the KV cache up to that point.
  - Run the incremental block recompute at batch sizes 1, 2, 4, 8, 16, ...
    (simulating batched recompute of multiple evicted blocks at once).
  - Measure wall-clock time per block as batch size increases.
  - At high batch sizes, the GPU saturates and per-block cost drops to the
    real throughput-limited floor.

Usage:
    python scripts/sweep_recompute_cost.py \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 --engine torch \
        --block-size 32 \
        --positions 32,128,512,2048,8192,32768,131072 \
        --batch-sizes 1,2,4,8,16,32

    python scripts/sweep_recompute_cost.py \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 --engine torch \
        --block-size 32 --positions 128,1024,8192,65536,131072
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_bench_utils import get_gpu_info, get_cuda_version, write_result_json


DEFAULT_POSITIONS = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep incremental prefill cost vs context position and batch size"
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B-Instruct",
                   help="Model name or path")
    p.add_argument("--engine", choices=["torch"], default="torch",
                   help="Backend (torch = raw HF)")
    p.add_argument("--kv-mode", choices=["bf16", "fp8", "nvfp4"], default="bf16",
                   help="KV cache precision")
    p.add_argument("--block-size", type=int, default=32,
                   help="Tokens per KV block (the unit we'd recompute)")
    p.add_argument("--positions", type=str, default=None,
                   help="Comma-separated context positions to measure at")
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32",
                   help="Comma-separated batch sizes to test. Higher batch = "
                        "more GPU saturation, lower per-block amortized cost.")
    p.add_argument("--warmup-iters", type=int, default=2,
                   help="Warmup iterations before timing")
    p.add_argument("--measure-iters", type=int, default=5,
                   help="Timed iterations to average over")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size")
    p.add_argument("--max-seq-len", type=int, default=131104,
                   help="Maximum sequence length the model is configured for")
    p.add_argument("--output", default=None,
                   help="Output JSON path")
    return p.parse_args()


def make_dummy_tokens(batch, length, vocab_size=32000):
    """Generate deterministic dummy token IDs with given batch size."""
    import torch
    gen = torch.Generator().manual_seed(42)
    return torch.randint(100, vocab_size, (batch, length), generator=gen)


class TorchBackend:
    """Measures incremental prefill cost using HuggingFace transformers.

    Supports batched measurement: run the same incremental block for B
    independent sequences simultaneously, then report per-block amortized cost.
    This simulates the GPU saturation level of a real serving scenario.
    """

    def __init__(self, model_name, kv_mode, max_seq_len, tp_size=1):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {"bf16": torch.bfloat16, "fp8": torch.bfloat16, "nvfp4": torch.bfloat16}
        self.dtype = dtype_map.get(kv_mode, torch.bfloat16)
        self.kv_mode = kv_mode

        print(f"Loading {model_name} on {self.device} with dtype={self.dtype}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        load_kwargs = {
            "dtype": self.dtype,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        try:
            import accelerate  # noqa: F401
            load_kwargs["device_map"] = "auto"
        except ImportError:
            pass
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.vocab_size = self.model.config.vocab_size
        print("Model loaded.")

    def measure_incremental_prefill(self, context_length, block_size, batch_size,
                                     warmup_iters, measure_iters):
        """Measure time to compute one block of KV at given batch size.

        Returns dict with total time for the batch and per-block amortized cost.
        """
        import torch
        with torch.no_grad():
            total_len = context_length + block_size

            # Generate batched token sequences
            input_ids = make_dummy_tokens(batch_size, total_len, self.vocab_size).to(self.device)
            context_ids = input_ids[:, :context_length]

            # Build KV cache for the context prefix (batched)
            print(f"    Building KV cache (batch={batch_size}, ctx={context_length})...",
                  end="", flush=True)
            try:
                out = self.model(context_ids, use_cache=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f" OOM at batch={batch_size}, ctx={context_length}")
                return None
            past_kv = out.past_key_values
            torch.cuda.synchronize()
            print(" done")

            incremental_ids = input_ids[:, context_length:total_len]

            # Warmup
            for _ in range(warmup_iters):
                try:
                    self.model(incremental_ids, past_key_values=past_kv, use_cache=True)
                except torch.cuda.OutOfMemoryError:
                    del past_kv, out
                    torch.cuda.empty_cache()
                    print(f"    OOM during warmup at batch={batch_size}")
                    return None
                torch.cuda.synchronize()

            # Timed runs
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            times_ms = []

            for _ in range(measure_iters):
                start_event.record()
                self.model(incremental_ids, past_key_values=past_kv, use_cache=True)
                end_event.record()
                torch.cuda.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))

            # Clean up
            del past_kv, out, input_ids, context_ids, incremental_ids
            torch.cuda.empty_cache()

            avg_ms = sum(times_ms) / len(times_ms)
            per_block_ms = avg_ms / batch_size
            return {
                "context_position": context_length,
                "block_size_tokens": block_size,
                "batch_size": batch_size,
                "total_batch_ms_mean": round(avg_ms, 4),
                "total_batch_ms_min": round(min(times_ms), 4),
                "total_batch_ms_max": round(max(times_ms), 4),
                "per_block_ms_mean": round(per_block_ms, 4),
                "per_block_ms_min": round(min(times_ms) / batch_size, 4),
                "per_block_ms_max": round(max(times_ms) / batch_size, 4),
                "per_token_ms_mean": round(per_block_ms / block_size, 4),
                "individual_runs_ms": [round(t, 4) for t in times_ms],
            }


def main():
    args = parse_args()

    positions = (
        [int(x) for x in args.positions.split(",")]
        if args.positions
        else DEFAULT_POSITIONS
    )
    positions = [p for p in positions if p + args.block_size <= args.max_seq_len]

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if not positions:
        print("ERROR: No valid positions after filtering by max-seq-len.")
        sys.exit(1)

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/recompute_curve_{args.engine}_{args.kv_mode}_{ts}.json"

    gpu_info = get_gpu_info()
    cuda_ver = get_cuda_version()

    print("=== Recompute Cost Sweep (Batched) ===")
    print(f"Model:       {args.model}")
    print(f"Engine:      {args.engine}")
    print(f"KV mode:     {args.kv_mode}")
    print(f"Block size:  {args.block_size} tokens")
    print(f"Positions:   {positions}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warmup:      {args.warmup_iters} iters")
    print(f"Measure:     {args.measure_iters} iters")
    print(f"Output:      {args.output}")
    print()

    backend = TorchBackend(args.model, args.kv_mode, args.max_seq_len, args.tp)

    measurements = []
    for pos in positions:
        print(f"\n--- Position {pos} ---")
        for bs in batch_sizes:
            print(f"  Batch size {bs}:")
            try:
                result = backend.measure_incremental_prefill(
                    pos, args.block_size, bs,
                    args.warmup_iters, args.measure_iters,
                )
                if result is None:
                    print(f"    Skipped (OOM)")
                    measurements.append({
                        "context_position": pos,
                        "block_size_tokens": args.block_size,
                        "batch_size": bs,
                        "error": "OOM",
                    })
                    # If OOM at this batch size, skip larger batches for this position
                    break
                else:
                    measurements.append(result)
                    print(f"    Total: {result['total_batch_ms_mean']:.3f} ms, "
                          f"Per-block: {result['per_block_ms_mean']:.3f} ms, "
                          f"Per-token: {result['per_token_ms_mean']:.4f} ms/tok")
            except Exception as e:
                print(f"    FAILED: {e}")
                measurements.append({
                    "context_position": pos,
                    "block_size_tokens": args.block_size,
                    "batch_size": bs,
                    "error": str(e),
                })

    # Write output
    output = {
        "sweep_type": "recompute_cost",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "engine": args.engine,
            "kv_mode": args.kv_mode,
            "block_size_tokens": args.block_size,
            "batch_sizes": batch_sizes,
            "warmup_iters": args.warmup_iters,
            "measure_iters": args.measure_iters,
            "tp": args.tp,
            "max_seq_len": args.max_seq_len,
        },
        "hardware": {
            "gpu_name": gpu_info["gpu_model"],
            "gpu_count": gpu_info["gpu_count"],
            "driver_version": gpu_info["driver_version"],
            "cuda_version": cuda_ver,
        },
        "measurements": measurements,
        "rerun_command": (
            f"python scripts/sweep_recompute_cost.py "
            f"--model {args.model} --engine {args.engine} "
            f"--kv-mode {args.kv_mode} --block-size {args.block_size} "
            f"--positions {','.join(str(p) for p in positions)} "
            f"--batch-sizes {','.join(str(b) for b in batch_sizes)} "
            f"--warmup-iters {args.warmup_iters} "
            f"--measure-iters {args.measure_iters} "
            f"--tp {args.tp} --max-seq-len {args.max_seq_len} "
            f"--output {args.output}"
        ),
    }

    write_result_json(output, args.output)

    # Print summary table
    print("\n=== Recompute Cost Summary (per-block amortized ms) ===")
    header = f"{'Position':>10}"
    for bs in batch_sizes:
        header += f"  {'B=' + str(bs):>10}"
    print(header)
    print("-" * (12 + 12 * len(batch_sizes)))

    for pos in positions:
        row = f"{pos:>10}"
        for bs in batch_sizes:
            m = next((x for x in measurements
                      if x.get("context_position") == pos
                      and x.get("batch_size") == bs
                      and "error" not in x), None)
            if m:
                row += f"  {m['per_block_ms_mean']:>10.3f}"
            else:
                row += f"  {'---':>10}"
        print(row)


if __name__ == "__main__":
    main()
