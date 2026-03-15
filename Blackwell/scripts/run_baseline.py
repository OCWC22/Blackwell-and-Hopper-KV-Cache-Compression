#!/usr/bin/env python3
"""Baseline vLLM inference benchmark — measures decode latency, memory, throughput."""

import argparse
import json
import time
import torch
import os

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0

def run_baseline(model_name: str, context_lengths: list[int], output_path: str):
    results = {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "experiments": [],
    }

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Installing...")
        os.system("pip install vllm")
        from vllm import LLM, SamplingParams

    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=max(context_lengths),
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )

    for ctx_len in context_lengths:
        print(f"\n--- Context length: {ctx_len} ---")

        # Generate a prompt of approximately ctx_len tokens
        # ~4 chars per token as rough estimate
        prompt = "The quick brown fox " * (ctx_len // 5)

        torch.cuda.reset_peak_memory_stats()
        mem_before = get_gpu_memory_mb()

        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.perf_counter()

        mem_after = get_gpu_memory_mb()
        peak_mem = torch.cuda.max_memory_allocated() / 1e6

        output_text = outputs[0].outputs[0].text
        num_output_tokens = len(outputs[0].outputs[0].token_ids)
        elapsed = t1 - t0

        exp = {
            "context_length": ctx_len,
            "output_tokens": num_output_tokens,
            "total_time_s": round(elapsed, 3),
            "tokens_per_second": round(num_output_tokens / elapsed, 1),
            "memory_before_mb": round(mem_before, 1),
            "memory_after_mb": round(mem_after, 1),
            "peak_memory_mb": round(peak_mem, 1),
        }
        results["experiments"].append(exp)

        print(f"  Time: {elapsed:.3f}s")
        print(f"  Output tokens: {num_output_tokens}")
        print(f"  Tokens/sec: {num_output_tokens / elapsed:.1f}")
        print(f"  Peak memory: {peak_mem:.0f} MB")

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[8192, 32768, 65536])
    parser.add_argument("--output", default="results/baseline.json")
    args = parser.parse_args()

    run_baseline(args.model, args.context_lengths, args.output)
