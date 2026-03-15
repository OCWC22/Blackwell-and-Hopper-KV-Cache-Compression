#!/usr/bin/env python3
"""Generate benchmark trace JSONL files for KV serving capacity sweeps.

Downloads public datasets from HuggingFace Hub, tokenizes prompts,
buckets them by token count, fits arrival distributions from BurstGPT,
and emits JSONL trace files matching the conversation schedule schema
from actual_kv_serving_benchmark_and_synthetic_data.md.

v1 data sources:
  - ShareGPT (anon8231489123/ShareGPT_Vicuna_unfiltered) — chat prompt pool
  - BurstGPT (lzzmm/BurstGPT) — arrival timing + request-length distributions
  - OpenAI MRCR (openai/mrcr) — multi-turn recall challenge

Usage:
    # Single-turn traces from ShareGPT + BurstGPT timing
    python scripts/generate_benchmark_traces.py \\
        --datasets sharegpt,burstgpt \\
        --workload-family single_turn_active_set \\
        --num-conversations 64

    # Multi-turn traces with MRCR conversations
    python scripts/generate_benchmark_traces.py \\
        --datasets sharegpt,burstgpt,mrcr \\
        --workload-family synthetic_multi_turn_active_set \\
        --num-conversations 32

    # All defaults — both workload families, all v1 datasets
    python scripts/generate_benchmark_traces.py --all
"""

import argparse
import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate KV serving benchmark trace JSONL files"
    )
    p.add_argument(
        "--datasets",
        default="sharegpt,burstgpt,mrcr",
        help="Comma-separated dataset keys: sharegpt, burstgpt, mrcr (default: all three)",
    )
    p.add_argument(
        "--prompt-buckets",
        default="10000,20000,30000,40000,50000",
        help="Comma-separated token-count bucket targets",
    )
    p.add_argument(
        "--workload-family",
        choices=["single_turn_active_set", "synthetic_multi_turn_active_set", "both"],
        default="both",
        help="Workload family to generate (default: both)",
    )
    p.add_argument(
        "--num-conversations",
        type=int,
        default=64,
        help="Number of conversations per (workload_family, bucket)",
    )
    p.add_argument(
        "--shared-prefix-ratio",
        type=float,
        default=0.8,
        help="Fraction of prompt tokens that form the shared prefix within a reuse cluster",
    )
    p.add_argument(
        "--turns-per-conversation",
        type=int,
        default=5,
        help="Number of turns per multi-turn conversation",
    )
    p.add_argument(
        "--target-output-tokens",
        type=int,
        default=256,
        help="Target output token count per request",
    )
    p.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace tokenizer to use for prompt tokenization",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSONL traces (default: Blackwell/data/traces/)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory for downloaded datasets",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Generate all workload families with all v1 datasets (convenience flag)",
    )
    p.add_argument(
        "--sharegpt-long",
        action="store_true",
        help="Also load ShareGPT-Long (Arist12/EABF-ShareGPT-Long-3.5k) for long conversations",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without downloading or generating",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_sharegpt(cache_dir=None):
    """Load ShareGPT Vicuna unfiltered dataset.

    Returns list of dicts: [{"text": str, "source": "sharegpt"}, ...]
    """
    from datasets import load_dataset

    print("Loading ShareGPT (anon8231489123/ShareGPT_Vicuna_unfiltered)...")
    try:
        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "default",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception:
        # Some versions of the dataset use different config names
        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    prompts = []
    for row in ds:
        conversations = row.get("conversations", [])
        if not conversations:
            continue
        # Concatenate all turns into a single text block
        text_parts = []
        for turn in conversations:
            value = turn.get("value", "")
            if value:
                text_parts.append(value)
        full_text = "\n\n".join(text_parts)
        if len(full_text) > 100:  # skip trivially short
            prompts.append({"text": full_text, "source": "sharegpt"})

    print(f"  Loaded {len(prompts)} ShareGPT conversations")
    return prompts


def load_sharegpt_long(cache_dir=None):
    """Load ShareGPT Long dataset (English-only, >10K tokens).

    Returns list of dicts: [{"text": str, "source": "sharegpt_long"}, ...]
    """
    from datasets import load_dataset

    print("Loading ShareGPT-Long (Arist12/EABF-ShareGPT-Long-3.5k)...")
    ds = load_dataset(
        "Arist12/EABF-ShareGPT-Long-3.5k",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    prompts = []
    for row in ds:
        conversations = row.get("conversations", [])
        if not conversations:
            continue
        text_parts = []
        for turn in conversations:
            value = turn.get("value", "")
            if value:
                text_parts.append(value)
        full_text = "\n\n".join(text_parts)
        if len(full_text) > 100:
            prompts.append({"text": full_text, "source": "sharegpt_long"})

    print(f"  Loaded {len(prompts)} ShareGPT-Long conversations")
    return prompts


def load_burstgpt(cache_dir=None):
    """Load BurstGPT trace data for arrival distributions.

    Returns a dict with:
      - "request_lengths": list of int (token counts from traces)
      - "inter_arrival_ms": list of float (inter-arrival times in ms)
      - "think_times_ms": list of float (within-session gaps in ms)
    """
    from datasets import load_dataset

    print("Loading BurstGPT (lzzmm/BurstGPT)...")
    ds = load_dataset(
        "lzzmm/BurstGPT",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    request_lengths = []
    timestamps = []

    for row in ds:
        # BurstGPT schema: varies by subset, look for token count and timestamp fields
        tokens = (
            row.get("total_tokens")
            or row.get("input_tokens")
            or row.get("prompt_tokens")
            or row.get("Tokens")
            or row.get("tokens")
        )
        if tokens is not None:
            try:
                tokens = int(tokens)
                if tokens > 0:
                    request_lengths.append(tokens)
            except (ValueError, TypeError):
                pass

        ts = (
            row.get("timestamp")
            or row.get("Timestamp")
            or row.get("time")
        )
        if ts is not None:
            try:
                timestamps.append(float(ts))
            except (ValueError, TypeError):
                pass

    # Compute inter-arrival times
    inter_arrival_ms = []
    if len(timestamps) > 1:
        sorted_ts = sorted(timestamps)
        for i in range(1, len(sorted_ts)):
            delta = (sorted_ts[i] - sorted_ts[i - 1]) * 1000  # assume seconds → ms
            if 0 < delta < 600_000:  # filter outliers > 10 min
                inter_arrival_ms.append(delta)

    # Derive think_times from within-session patterns
    # Use inter-arrival times in the 1s–60s range as think_time proxies
    think_times_ms = [t for t in inter_arrival_ms if 1000 <= t <= 60_000]

    # If no timing data, use reasonable defaults
    if not think_times_ms:
        think_times_ms = list(np.random.lognormal(mean=8.5, sigma=1.0, size=1000))

    if not request_lengths:
        # Fallback: generate synthetic request lengths with Zipf-like distribution
        request_lengths = list(
            np.random.zipf(a=1.5, size=5000).clip(100, 50000).astype(int)
        )

    print(f"  Loaded {len(request_lengths)} request lengths, "
          f"{len(inter_arrival_ms)} inter-arrival times, "
          f"{len(think_times_ms)} think-time samples")

    return {
        "request_lengths": request_lengths,
        "inter_arrival_ms": inter_arrival_ms,
        "think_times_ms": think_times_ms,
    }


def load_mrcr(cache_dir=None):
    """Load OpenAI MRCR (Multi-turn Recall Challenge) dataset.

    Returns list of dicts with multi-turn conversation structure:
    [{"turns": [{"role": str, "content": str}, ...], "source": "mrcr"}, ...]
    """
    from datasets import load_dataset

    print("Loading MRCR (openai/mrcr)...")
    ds = load_dataset(
        "openai/mrcr",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    conversations = []
    for row in ds:
        # MRCR has multi-turn structure — extract turns
        messages = row.get("messages", [])
        if not messages:
            # Try alternative field names
            messages = row.get("conversation", [])
        if not messages:
            # Fall back to single prompt/response
            prompt = row.get("prompt", row.get("question", ""))
            if prompt:
                messages = [{"role": "user", "content": prompt}]
                answer = row.get("answer", row.get("response", ""))
                if answer:
                    messages.append({"role": "assistant", "content": answer})

        if messages:
            turns = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    turns.append({"role": role, "content": content})
            if turns:
                conversations.append({"turns": turns, "source": "mrcr"})

    print(f"  Loaded {len(conversations)} MRCR conversations")
    return conversations


# ---------------------------------------------------------------------------
# Tokenization and bucketing
# ---------------------------------------------------------------------------

def get_tokenizer(tokenizer_name):
    """Load a HuggingFace tokenizer (no model weights)."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )
    return tokenizer


def tokenize_prompts(prompts, tokenizer):
    """Tokenize prompt texts and attach token count.

    Args:
        prompts: list of dicts with "text" key
        tokenizer: HuggingFace tokenizer

    Returns:
        list of dicts with added "token_ids" and "num_tokens" keys
    """
    print(f"Tokenizing {len(prompts)} prompts...")
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt["text"], add_special_tokens=False)
        prompt["token_ids"] = token_ids
        prompt["num_tokens"] = len(token_ids)
    return prompts


def tokenize_mrcr_conversations(conversations, tokenizer):
    """Tokenize MRCR multi-turn conversations.

    Returns conversations with per-turn token counts and total token count.
    """
    print(f"Tokenizing {len(conversations)} MRCR conversations...")
    for conv in conversations:
        total_tokens = 0
        for turn in conv["turns"]:
            token_ids = tokenizer.encode(turn["content"], add_special_tokens=False)
            turn["num_tokens"] = len(token_ids)
            total_tokens += len(token_ids)
        conv["num_tokens"] = total_tokens
    return conversations


def bucket_prompts(prompts, buckets, tokenizer):
    """Assign prompts to token-count buckets.

    For prompts shorter than the target bucket, concatenate multiple prompts.
    For prompts longer than the target, truncate.

    Args:
        prompts: list of tokenized prompt dicts
        buckets: list of int bucket targets
        tokenizer: HuggingFace tokenizer for decoding concatenated tokens

    Returns:
        dict: {bucket_size: [{"text": str, "num_tokens": int, "source": str}, ...]}
    """
    print(f"Bucketing prompts into {len(buckets)} bins: {buckets}")
    rng = np.random.default_rng(42)

    # Sort prompts by length for efficient packing
    sorted_prompts = sorted(prompts, key=lambda p: p["num_tokens"], reverse=True)

    bucketed = defaultdict(list)
    tolerance = 0.05  # ±5% token count tolerance

    for target in buckets:
        lo = int(target * (1 - tolerance))
        hi = int(target * (1 + tolerance))

        # 1. Direct matches — prompts within tolerance range
        for p in sorted_prompts:
            if lo <= p["num_tokens"] <= hi:
                bucketed[target].append({
                    "text": p["text"],
                    "num_tokens": p["num_tokens"],
                    "source": p["source"],
                })

        # 2. Truncate longer prompts
        for p in sorted_prompts:
            if p["num_tokens"] > hi:
                truncated_ids = p["token_ids"][:target]
                truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                bucketed[target].append({
                    "text": truncated_text,
                    "num_tokens": target,
                    "source": p["source"],
                })
                if len(bucketed[target]) >= 500:
                    break

        # 3. Concatenate shorter prompts to reach target
        short_prompts = [p for p in sorted_prompts if p["num_tokens"] < lo]
        if short_prompts:
            for _ in range(200):  # generate up to 200 concatenated prompts
                accumulated_ids = []
                sources = set()
                indices = rng.choice(len(short_prompts), size=min(50, len(short_prompts)), replace=True)
                for idx in indices:
                    accumulated_ids.extend(short_prompts[idx]["token_ids"])
                    sources.add(short_prompts[idx]["source"])
                    if len(accumulated_ids) >= target:
                        break
                if len(accumulated_ids) >= lo:
                    final_ids = accumulated_ids[:target]
                    final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
                    bucketed[target].append({
                        "text": final_text,
                        "num_tokens": len(final_ids),
                        "source": "+".join(sorted(sources)),
                    })

        print(f"  Bucket {target}: {len(bucketed[target])} prompts")

    return bucketed


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------

def fit_burstgpt_distributions(burstgpt_data, rng):
    """Fit distributions from BurstGPT trace data.

    Returns a dict of fitted distribution parameters and sampling functions.
    """
    request_lengths = np.array(burstgpt_data["request_lengths"], dtype=float)
    think_times = np.array(burstgpt_data["think_times_ms"], dtype=float)

    # Fit log-normal to request lengths (more robust than Zipf for sampling)
    rl_log = np.log(request_lengths[request_lengths > 0])
    rl_mu = np.mean(rl_log)
    rl_sigma = np.std(rl_log)

    # Fit log-normal to think times
    tt_log = np.log(think_times[think_times > 0])
    tt_mu = np.mean(tt_log)
    tt_sigma = np.std(tt_log)

    # Try Zipf fit for request lengths using scipy
    zipf_alpha = 1.5  # default
    try:
        from scipy.stats import zipf as zipf_dist

        # MLE for Zipf on discretized lengths
        # Use method of moments: E[X] = H(s-1)/H(s) where H is generalized harmonic
        # Simple approximation: fit alpha from log-log slope
        counts, bin_edges = np.histogram(request_lengths, bins=50)
        nonzero = counts > 0
        if np.sum(nonzero) > 5:
            log_counts = np.log(counts[nonzero])
            log_bins = np.log((bin_edges[:-1][nonzero] + bin_edges[1:][nonzero]) / 2)
            # Linear regression in log-log space
            coeffs = np.polyfit(log_bins, log_counts, 1)
            zipf_alpha = max(1.01, min(3.0, -coeffs[0]))
    except ImportError:
        pass

    distributions = {
        "request_length": {
            "type": "lognormal",
            "mu": float(rl_mu),
            "sigma": float(rl_sigma),
            "zipf_alpha": float(zipf_alpha),
        },
        "think_time_ms": {
            "type": "lognormal",
            "mu": float(tt_mu),
            "sigma": float(tt_sigma),
            "median_ms": float(np.median(think_times)),
            "p95_ms": float(np.percentile(think_times, 95)),
        },
    }

    print(f"  Request length dist: lognormal(mu={rl_mu:.2f}, sigma={rl_sigma:.2f}), zipf_alpha={zipf_alpha:.2f}")
    print(f"  Think time dist: lognormal(mu={tt_mu:.2f}, sigma={tt_sigma:.2f}), "
          f"median={np.median(think_times):.0f}ms, p95={np.percentile(think_times, 95):.0f}ms")

    return distributions


def sample_think_time(distributions, rng):
    """Sample a single think_time_ms from the fitted distribution."""
    params = distributions["think_time_ms"]
    t = rng.lognormal(mean=params["mu"], sigma=params["sigma"])
    # Clamp to reasonable range: 500ms to 120s
    return float(np.clip(t, 500, 120_000))


# ---------------------------------------------------------------------------
# Prefix cluster generation
# ---------------------------------------------------------------------------

def generate_prefix_clusters(bucketed_prompts, bucket, shared_prefix_ratio, num_clusters, rng):
    """Create reuse clusters with shared prefixes and unique suffixes.

    Args:
        bucketed_prompts: list of prompt dicts for a single bucket
        bucket: target token count
        shared_prefix_ratio: fraction of tokens that form the shared prefix
        num_clusters: number of reuse clusters to create
        rng: numpy random generator

    Returns:
        list of dicts with cluster assignments and prefix/suffix structure
    """
    prefix_tokens = int(bucket * shared_prefix_ratio)
    suffix_tokens = bucket - prefix_tokens

    clustered = []
    prompts_per_cluster = max(1, len(bucketed_prompts) // num_clusters)

    for cluster_idx in range(num_clusters):
        cluster_id = f"cluster_{cluster_idx:03d}"
        start = cluster_idx * prompts_per_cluster
        end = start + prompts_per_cluster
        cluster_prompts = bucketed_prompts[start:end]

        if not cluster_prompts:
            # Recycle prompts if we run out
            cluster_prompts = [rng.choice(bucketed_prompts)]

        for prompt in cluster_prompts:
            clustered.append({
                "text": prompt["text"],
                "num_tokens": prompt["num_tokens"],
                "source": prompt["source"],
                "reuse_cluster_id": cluster_id,
                "estimated_reusable_prefix_tokens": prefix_tokens,
                "suffix_tokens": suffix_tokens,
            })

    return clustered


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------

def generate_single_turn_traces(
    clustered_prompts, bucket, shared_prefix_ratio, num_conversations,
    target_output_tokens, rng,
):
    """Generate single_turn_active_set JSONL records.

    Each conversation has exactly one turn, no think_time.
    """
    records = []
    for i in range(num_conversations):
        prompt = clustered_prompts[i % len(clustered_prompts)]
        record = {
            "conversation_id": f"conv_{i:04d}",
            "turn_index": 0,
            "workload_family": "single_turn_active_set",
            "prompt_bucket_tokens": bucket,
            "shared_prefix_ratio": shared_prefix_ratio,
            "reuse_cluster_id": prompt["reuse_cluster_id"],
            "think_time_ms": 0,
            "target_output_tokens": target_output_tokens,
            "sampling_mode": "greedy",
            "prompt_text": prompt["text"],
            "estimated_reusable_prefix_tokens": prompt["estimated_reusable_prefix_tokens"],
            "source_dataset": prompt["source"],
        }
        records.append(record)
    return records


def generate_multi_turn_traces(
    clustered_prompts, bucket, shared_prefix_ratio, num_conversations,
    turns_per_conversation, target_output_tokens, distributions, mrcr_conversations,
    rng,
):
    """Generate synthetic_multi_turn_active_set JSONL records.

    Each conversation has multiple turns with think_time gaps.
    MRCR conversations are used when available for natural multi-turn structure.
    """
    records = []

    # Split: use MRCR for some conversations, synthesized multi-turn for the rest
    mrcr_count = min(len(mrcr_conversations), num_conversations // 2) if mrcr_conversations else 0
    synthetic_count = num_conversations - mrcr_count

    # MRCR-sourced multi-turn conversations
    for i in range(mrcr_count):
        conv = mrcr_conversations[i % len(mrcr_conversations)]
        cluster_prompt = clustered_prompts[i % len(clustered_prompts)]

        for turn_idx, turn in enumerate(conv["turns"]):
            if turn["role"] != "user":
                continue
            think_time = 0 if turn_idx == 0 else sample_think_time(distributions, rng)
            record = {
                "conversation_id": f"conv_{i:04d}",
                "turn_index": turn_idx,
                "workload_family": "synthetic_multi_turn_active_set",
                "prompt_bucket_tokens": bucket,
                "shared_prefix_ratio": shared_prefix_ratio,
                "reuse_cluster_id": cluster_prompt["reuse_cluster_id"],
                "think_time_ms": round(think_time, 1),
                "target_output_tokens": target_output_tokens,
                "sampling_mode": "greedy",
                "prompt_text": turn["content"],
                "estimated_reusable_prefix_tokens": cluster_prompt["estimated_reusable_prefix_tokens"],
                "source_dataset": "mrcr",
            }
            records.append(record)

    # Synthesized multi-turn conversations from prompt pool
    for i in range(synthetic_count):
        conv_idx = mrcr_count + i
        base_prompt = clustered_prompts[conv_idx % len(clustered_prompts)]

        for turn_idx in range(turns_per_conversation):
            think_time = 0 if turn_idx == 0 else sample_think_time(distributions, rng)

            # For follow-up turns, append a short suffix to the base prompt
            if turn_idx == 0:
                prompt_text = base_prompt["text"]
            else:
                prompt_text = (
                    f"Follow-up question {turn_idx} on the previous analysis: "
                    f"Please elaborate on point {turn_idx} with specific examples "
                    f"and implementation details."
                )

            # Prefix reuse grows with each turn (earlier turns become reusable context)
            reusable_tokens = min(
                base_prompt["estimated_reusable_prefix_tokens"] + turn_idx * target_output_tokens,
                bucket,
            )

            record = {
                "conversation_id": f"conv_{conv_idx:04d}",
                "turn_index": turn_idx,
                "workload_family": "synthetic_multi_turn_active_set",
                "prompt_bucket_tokens": bucket,
                "shared_prefix_ratio": shared_prefix_ratio,
                "reuse_cluster_id": base_prompt["reuse_cluster_id"],
                "think_time_ms": round(think_time, 1),
                "target_output_tokens": target_output_tokens,
                "sampling_mode": "greedy",
                "prompt_text": prompt_text,
                "estimated_reusable_prefix_tokens": reusable_tokens,
                "source_dataset": base_prompt["source"],
            }
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_traces(records, output_dir, workload_family, bucket):
    """Write trace records to a JSONL file.

    File naming: {workload_family}_{bucket//1000}k.jsonl
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{workload_family}_{bucket // 1000}k.jsonl"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(records)} records to {filepath}")
    return filepath


def write_metadata(output_dir, metadata):
    """Write generation metadata to a JSON sidecar file."""
    filepath = os.path.join(output_dir, "trace_metadata.json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Wrote metadata to {filepath}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Resolve output dir
    if args.output_dir is None:
        # Default: Blackwell/data/traces/ relative to script location
        script_dir = Path(__file__).resolve().parent
        args.output_dir = str(script_dir.parent / "data" / "traces")

    if args.all:
        args.datasets = "sharegpt,burstgpt,mrcr"
        args.workload_family = "both"

    dataset_keys = [d.strip().lower() for d in args.datasets.split(",")]
    buckets = [int(b.strip()) for b in args.prompt_buckets.split(",")]
    workload_families = (
        ["single_turn_active_set", "synthetic_multi_turn_active_set"]
        if args.workload_family == "both"
        else [args.workload_family]
    )

    print("=" * 60)
    print("KV Serving Benchmark Trace Generator")
    print("=" * 60)
    print(f"  Datasets: {dataset_keys}")
    print(f"  Buckets: {buckets}")
    print(f"  Workload families: {workload_families}")
    print(f"  Conversations per bucket: {args.num_conversations}")
    print(f"  Shared prefix ratio: {args.shared_prefix_ratio}")
    print(f"  Turns per conversation (multi-turn): {args.turns_per_conversation}")
    print(f"  Target output tokens: {args.target_output_tokens}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Seed: {args.seed}")
    print()

    if args.dry_run:
        print("Dry run — exiting without generating traces.")
        return

    # --- Step 1: Load datasets ---
    print("--- Step 1: Loading datasets ---")
    prompt_pool = []
    burstgpt_data = None
    mrcr_conversations = []

    if "sharegpt" in dataset_keys:
        sharegpt_prompts = load_sharegpt(cache_dir=args.cache_dir)
        prompt_pool.extend(sharegpt_prompts)

    if args.sharegpt_long:
        sharegpt_long_prompts = load_sharegpt_long(cache_dir=args.cache_dir)
        prompt_pool.extend(sharegpt_long_prompts)

    if "burstgpt" in dataset_keys:
        burstgpt_data = load_burstgpt(cache_dir=args.cache_dir)

    if "mrcr" in dataset_keys:
        mrcr_conversations = load_mrcr(cache_dir=args.cache_dir)

    if not prompt_pool:
        print("ERROR: No prompt data loaded. Check dataset keys.")
        sys.exit(1)

    print(f"\nTotal prompt pool: {len(prompt_pool)} entries")

    # --- Step 2: Tokenize ---
    print("\n--- Step 2: Tokenizing ---")
    tokenizer = get_tokenizer(args.tokenizer)
    prompt_pool = tokenize_prompts(prompt_pool, tokenizer)

    if mrcr_conversations:
        mrcr_conversations = tokenize_mrcr_conversations(mrcr_conversations, tokenizer)

    # Print token count distribution
    token_counts = [p["num_tokens"] for p in prompt_pool]
    print(f"  Token count stats: min={min(token_counts)}, max={max(token_counts)}, "
          f"median={int(np.median(token_counts))}, mean={int(np.mean(token_counts))}")

    # --- Step 3: Bucket ---
    print("\n--- Step 3: Bucketing prompts ---")
    bucketed = bucket_prompts(prompt_pool, buckets, tokenizer)

    # --- Step 4: Fit BurstGPT distributions ---
    print("\n--- Step 4: Fitting BurstGPT distributions ---")
    if burstgpt_data:
        distributions = fit_burstgpt_distributions(burstgpt_data, rng)
    else:
        # Default distributions when BurstGPT is not loaded
        distributions = {
            "request_length": {
                "type": "lognormal",
                "mu": 7.0,
                "sigma": 1.5,
            },
            "think_time_ms": {
                "type": "lognormal",
                "mu": 8.5,
                "sigma": 1.0,
                "median_ms": 5000.0,
                "p95_ms": 30000.0,
            },
        }
        print("  Using default distributions (BurstGPT not loaded)")

    # --- Step 5: Generate traces ---
    print("\n--- Step 5: Generating traces ---")
    all_outputs = []
    num_clusters = max(4, args.num_conversations // 8)

    for bucket in buckets:
        if not bucketed[bucket]:
            print(f"  WARNING: No prompts for bucket {bucket}. Skipping.")
            continue

        # Generate prefix clusters
        clustered = generate_prefix_clusters(
            bucketed[bucket], bucket, args.shared_prefix_ratio, num_clusters, rng,
        )

        for family in workload_families:
            if family == "single_turn_active_set":
                records = generate_single_turn_traces(
                    clustered, bucket, args.shared_prefix_ratio,
                    args.num_conversations, args.target_output_tokens, rng,
                )
            else:
                records = generate_multi_turn_traces(
                    clustered, bucket, args.shared_prefix_ratio,
                    args.num_conversations, args.turns_per_conversation,
                    args.target_output_tokens, distributions,
                    mrcr_conversations, rng,
                )

            filepath = write_traces(records, args.output_dir, family, bucket)
            all_outputs.append({
                "file": filepath,
                "workload_family": family,
                "bucket": bucket,
                "num_records": len(records),
            })

    # --- Step 6: Write metadata ---
    print("\n--- Step 6: Writing metadata ---")
    metadata = {
        "generator": "generate_benchmark_traces.py",
        "version": "1.0.0",
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "datasets": dataset_keys,
        "prompt_pool_size": len(prompt_pool),
        "buckets": buckets,
        "workload_families": workload_families,
        "num_conversations_per_bucket": args.num_conversations,
        "shared_prefix_ratio": args.shared_prefix_ratio,
        "turns_per_conversation": args.turns_per_conversation,
        "target_output_tokens": args.target_output_tokens,
        "distributions": distributions,
        "outputs": all_outputs,
    }
    write_metadata(args.output_dir, metadata)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Generation complete.")
    print(f"  Total files: {len(all_outputs)}")
    total_records = sum(o["num_records"] for o in all_outputs)
    print(f"  Total records: {total_records}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
