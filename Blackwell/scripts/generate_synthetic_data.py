#!/usr/bin/env python3
"""Synthetic data generator for KV cache benchmarks (Layers A/B/C/D).

Generates tokenizer-aware JSONL workload files shaped to exact token buckets
with conversation scheduling metadata and restore-aware annotations.

Replaces the chars_per_token=4 inline generation in run_baseline.py,
serve_and_bench.py, and run_tiered_experiment.py.

Usage:
    # Single-turn workloads at 10K and 20K buckets
    python scripts/generate_synthetic_data.py \\
        --buckets 10000,20000 \\
        --families single_turn \\
        --conversations 1,4,8,16 \\
        --prompt-pool-file data/coding_agent/first_turn_prompts.jsonl \\
        --output-dir data/synthetic/

    # Multi-turn workloads
    python scripts/generate_synthetic_data.py \\
        --buckets 10000,20000,30000 \\
        --families multi_turn \\
        --conversations 4,8 \\
        --turns-per-conversation 5 \\
        --prompt-pool-file data/coding_agent/first_turn_prompts.jsonl \\
        --output-dir data/synthetic/
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Layer C/D record schema
# ---------------------------------------------------------------------------

@dataclass
class ConversationRecord:
    """One request in a benchmark workload (Layer C + D metadata)."""
    conversation_id: str
    turn_index: int
    workload_family: str  # single_turn_active_set | synthetic_multi_turn_active_set
    prompt_bucket_tokens: int
    shared_prefix_ratio: float
    reuse_cluster_id: str
    think_time_ms: float
    target_output_tokens: int
    sampling_mode: str  # greedy | sample
    prompt: str
    # Layer D: restore-aware
    estimated_reusable_prefix_tokens: int
    estimated_restored_kv_bytes: int
    reuse_distance_bucket: str  # immediate | short | medium | long
    expected_restore_path: str  # gpu_resident | host_restore | recompute


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

_tokenizer = None


def get_tokenizer(model_name: str):
    """Load and cache tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Loaded tokenizer: {model_name} (vocab_size={_tokenizer.vocab_size})")
    return _tokenizer


def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(text: str, target: int, tokenizer) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)[:target]
    return tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# Layer A: prompt sources
#
# TODO(engineer): Implement prompt source loading. Options:
#   1. --prompt-pool-file: load from JSONL produced by ingest_coding_datasets.py
#      (recommended — uses real coding-agent session prompts)
#   2. ShareGPT: load from HuggingFace "anon8231489123/ShareGPT_Vicuna_unfiltered"
#   3. Custom corpus: load from a local text/JSONL file with domain-specific prompts
#   4. vLLM sonnet dataset: use the vLLM benchmark's built-in sonnet prompts
#
# The prompt pool must provide enough text to fill token buckets (10K-50K).
# Short prompts will be repeated/concatenated to reach target length.
# ---------------------------------------------------------------------------


def load_prompt_pool(path: str) -> list[str]:
    """Load prompts from a JSONL file.

    Expected format: one JSON object per line with a "prompt" or "content" field.
    Produced by ingest_coding_datasets.py --output-dir data/coding_agent/.

    TODO(engineer): Add additional loaders for other prompt sources
    (ShareGPT, custom corpus, etc.) behind a --prompt-source flag.
    """
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("prompt") or record.get("content", "")
            if text:
                prompts.append(text)
    print(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def shape_prompt_to_bucket(prompt: str, target_tokens: int, tokenizer) -> str:
    """Shape a prompt to exactly target_tokens.

    If the prompt is longer, truncate. If shorter, repeat it until the
    target is reached, then truncate to exact length.

    TODO(engineer): Replace the repeat-to-fill strategy with a more
    realistic approach once real long-document corpora are available
    (e.g., ArXiv papers, code repos, technical docs). The repeat
    strategy is a placeholder that preserves tokenizer accuracy.
    """
    current = count_tokens(prompt, tokenizer)
    if current >= target_tokens:
        return truncate_to_tokens(prompt, target_tokens, tokenizer)
    # Repeat prompt text until we exceed target, then truncate
    repeats = (target_tokens // max(current, 1)) + 2
    extended = (prompt + "\n\n") * repeats
    return truncate_to_tokens(extended, target_tokens, tokenizer)


# ---------------------------------------------------------------------------
# Layer B: prefix-reuse cluster generation
# ---------------------------------------------------------------------------

def generate_reuse_clusters(
    prompt_pool: list[str],
    num_clusters: int,
    prefix_tokens: int,
    suffix_tokens: int,
    requests_per_cluster: int,
    tokenizer,
    seed: int = 42,
) -> list[dict]:
    """Generate prefix-reuse clusters from a prompt pool.

    Each cluster shares one prefix (drawn from the pool and shaped to
    prefix_tokens). Requests within a cluster share the prefix but get
    a unique suffix drawn from a different pool entry.

    Returns list of dicts with: prompt, prefix_hash, reuse_cluster_id,
    prefix_tokens, suffix_tokens.
    """
    if not prompt_pool:
        raise ValueError(
            "Prompt pool is empty. Provide --prompt-pool-file pointing to a "
            "JSONL file with prompts (see ingest_coding_datasets.py)."
        )

    rng = np.random.default_rng(seed)
    records = []
    for c in range(num_clusters):
        # Pick a prompt for the prefix and a different one for suffixes
        prefix_src = prompt_pool[c % len(prompt_pool)]
        prefix = shape_prompt_to_bucket(prefix_src, prefix_tokens, tokenizer)
        prefix_hash = hashlib.md5(prefix.encode()).hexdigest()[:12]
        cluster_id = f"cluster_{c:04d}"

        for r in range(requests_per_cluster):
            suffix_idx = (c * requests_per_cluster + r + 1) % len(prompt_pool)
            suffix_src = prompt_pool[suffix_idx]
            suffix = shape_prompt_to_bucket(suffix_src, suffix_tokens, tokenizer)
            records.append({
                "prompt": prefix + suffix,
                "prefix_hash": prefix_hash,
                "reuse_cluster_id": cluster_id,
                "prefix_tokens": prefix_tokens,
                "suffix_tokens": suffix_tokens,
            })

    return records


# ---------------------------------------------------------------------------
# Layer C: conversation scheduling
# ---------------------------------------------------------------------------

def estimate_kv_bytes(tokens: int, num_layers: int = 48,
                      num_kv_heads: int = 8, head_dim: int = 128,
                      dtype_bytes: int = 1) -> int:
    """Estimate KV cache bytes for a given token count.

    Default values are for Qwen3-30B-A3B with FP8 KV (1 byte per element).
    KV bytes = tokens * num_layers * 2 (K+V) * num_kv_heads * head_dim * dtype_bytes

    TODO(engineer): Update defaults when switching models. Key params:
      - Qwen3-30B-A3B: 48 layers, 8 KV heads, 128 head_dim
      - Kimi-K2.5:     check model config for num_hidden_layers, num_key_value_heads
      For NVFP4: dtype_bytes=0.5, for BF16: dtype_bytes=2
    """
    return tokens * num_layers * 2 * num_kv_heads * head_dim * dtype_bytes


def assign_restore_path(turn_index: int, reuse_distance: int) -> tuple[str, str]:
    """Assign reuse_distance_bucket and expected_restore_path.

    Heuristic based on how far back the prefix was last seen:
    - Same request / turn 0: gpu_resident (immediate)
    - 1-2 turns back: host_restore (short)
    - 3-5 turns back: host_restore (medium)
    - >5 turns or first seen: recompute (long)

    TODO(engineer): Calibrate these thresholds against actual TRT-LLM
    KV cache eviction behavior observed during benchmark runs. The
    boundaries depend on GPU HBM capacity and KV block size.
    """
    if reuse_distance == 0:
        return "immediate", "gpu_resident"
    elif reuse_distance <= 2:
        return "short", "host_restore"
    elif reuse_distance <= 5:
        return "medium", "host_restore"
    else:
        return "long", "recompute"


def generate_single_turn_schedule(
    prompt_pool: list[str],
    bucket_tokens: int,
    num_conversations: int,
    tokenizer,
    prefix_ratio: float = 0.8,
    output_tokens: int = 128,
    num_clusters: int = 4,
    seed: int = 42,
) -> list[ConversationRecord]:
    """Generate single-turn active-set schedule.

    All conversations share clustered prefixes. Each conversation is one turn.
    """
    prefix_tokens = int(bucket_tokens * prefix_ratio)
    suffix_tokens = bucket_tokens - prefix_tokens
    actual_clusters = min(num_clusters, num_conversations)
    requests_per_cluster = max(1, num_conversations // actual_clusters)
    remainder = num_conversations - actual_clusters * requests_per_cluster

    cluster_records = generate_reuse_clusters(
        prompt_pool=prompt_pool,
        num_clusters=actual_clusters,
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        requests_per_cluster=requests_per_cluster + (1 if remainder > 0 else 0),
        tokenizer=tokenizer,
        seed=seed,
    )
    cluster_records = cluster_records[:num_conversations]

    records = []
    for i, cr in enumerate(cluster_records):
        distance_bucket, restore_path = assign_restore_path(0, i % (requests_per_cluster + 1))
        records.append(ConversationRecord(
            conversation_id=f"st_{bucket_tokens}_{i:04d}",
            turn_index=0,
            workload_family="single_turn_active_set",
            prompt_bucket_tokens=bucket_tokens,
            shared_prefix_ratio=prefix_ratio,
            reuse_cluster_id=cr["reuse_cluster_id"],
            think_time_ms=0.0,
            target_output_tokens=output_tokens,
            sampling_mode="greedy",
            prompt=cr["prompt"],
            estimated_reusable_prefix_tokens=cr["prefix_tokens"],
            estimated_restored_kv_bytes=estimate_kv_bytes(cr["prefix_tokens"]),
            reuse_distance_bucket=distance_bucket,
            expected_restore_path=restore_path,
        ))

    return records


def sample_think_times(
    num_turns: int,
    rng: np.random.Generator,
    mean_ms: float = 2000.0,
    sigma: float = 1.0,
) -> list[float]:
    """Sample inter-turn think times from a log-normal distribution.

    Log-normal fits coding-agent behavior: most think times are short
    (user reads output), with a long right tail (user edits code).

    TODO(engineer): Once real session shapes are available from
    ingest_coding_datasets.py (session_shapes.jsonl), fit the
    distribution parameters from actual inter-turn gaps instead
    of using these defaults.
    """
    mu = np.log(mean_ms) - sigma**2 / 2
    return rng.lognormal(mu, sigma, size=num_turns).tolist()


def generate_multi_turn_schedule(
    prompt_pool: list[str],
    bucket_tokens: int,
    num_conversations: int,
    turns_per_conversation: int,
    tokenizer,
    prefix_ratio: float = 0.6,
    output_tokens: int = 128,
    seed: int = 42,
) -> list[ConversationRecord]:
    """Generate multi-turn active-set schedule.

    Each conversation has multiple turns with growing context and think-time
    gaps. Earlier turns share a prefix; later turns extend it.
    """
    rng = np.random.default_rng(seed)
    prefix_tokens = int(bucket_tokens * prefix_ratio)
    suffix_tokens_per_turn = (bucket_tokens - prefix_tokens) // turns_per_conversation

    if not prompt_pool:
        raise ValueError(
            "Prompt pool is empty. Provide --prompt-pool-file pointing to a "
            "JSONL file with prompts (see ingest_coding_datasets.py)."
        )

    records = []
    for c in range(num_conversations):
        conv_id = f"mt_{bucket_tokens}_{c:04d}"
        cluster_id = f"cluster_{c % 4:04d}"
        think_times = [0.0] + sample_think_times(
            turns_per_conversation - 1, rng
        )

        prefix_src = prompt_pool[c % len(prompt_pool)]
        prefix = shape_prompt_to_bucket(prefix_src, prefix_tokens, tokenizer)

        for t in range(turns_per_conversation):
            suffix_idx = (c * turns_per_conversation + t + 1) % len(prompt_pool)
            suffix_src = prompt_pool[suffix_idx]
            suffix = shape_prompt_to_bucket(suffix_src, suffix_tokens_per_turn, tokenizer)
            prompt = prefix + suffix
            actual_tokens = count_tokens(prompt, tokenizer)

            distance_bucket, restore_path = assign_restore_path(t, t)
            effective_prefix_ratio = prefix_tokens / max(actual_tokens, 1)

            records.append(ConversationRecord(
                conversation_id=conv_id,
                turn_index=t,
                workload_family="synthetic_multi_turn_active_set",
                prompt_bucket_tokens=bucket_tokens,
                shared_prefix_ratio=round(effective_prefix_ratio, 3),
                reuse_cluster_id=cluster_id,
                think_time_ms=round(think_times[t], 1),
                target_output_tokens=output_tokens,
                sampling_mode="greedy",
                prompt=prompt,
                estimated_reusable_prefix_tokens=prefix_tokens,
                estimated_restored_kv_bytes=estimate_kv_bytes(prefix_tokens),
                reuse_distance_bucket=distance_bucket,
                expected_restore_path=restore_path,
            ))

    return records


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_jsonl(records: list[ConversationRecord], path: str):
    """Write conversation records to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic KV cache benchmark workloads"
    )
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B",
                   help="Model name for tokenizer (default: Qwen/Qwen3-30B-A3B)")
    p.add_argument("--buckets", default="10000,20000,30000",
                   help="Comma-separated token bucket targets")
    p.add_argument("--families", default="single_turn",
                   help="Comma-separated workload families: single_turn, multi_turn")
    p.add_argument("--conversations", default="1,4,8,16",
                   help="Comma-separated conversation counts per workload")
    p.add_argument("--turns-per-conversation", type=int, default=5,
                   help="Turns per conversation for multi_turn family")
    p.add_argument("--prefix-ratio", type=float, default=0.8,
                   help="Shared prefix ratio (single_turn default: 0.8)")
    p.add_argument("--multi-turn-prefix-ratio", type=float, default=0.6,
                   help="Shared prefix ratio for multi_turn family")
    p.add_argument("--output-tokens", type=int, default=128,
                   help="Target output tokens per request")
    p.add_argument("--num-clusters", type=int, default=4,
                   help="Number of prefix-reuse clusters")
    p.add_argument("--prompt-pool-file", required=True,
                   help="JSONL file with seed prompts (from ingest_coding_datasets.py)")
    p.add_argument("--output-dir", default="data/synthetic/",
                   help="Output directory for JSONL files")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = get_tokenizer(args.model)
    prompt_pool = load_prompt_pool(args.prompt_pool_file)

    if not prompt_pool:
        print("ERROR: No prompts loaded. Check --prompt-pool-file.")
        print("Run ingest_coding_datasets.py first to generate the prompt pool.")
        sys.exit(1)

    buckets = [int(b) for b in args.buckets.split(",")]
    families = [f.strip() for f in args.families.split(",")]
    conversations = [int(c) for c in args.conversations.split(",")]
    output_dir = Path(args.output_dir)

    total_files = 0
    total_records = 0

    for family in families:
        for bucket in buckets:
            for conv_count in conversations:
                if family == "single_turn":
                    records = generate_single_turn_schedule(
                        prompt_pool=prompt_pool,
                        bucket_tokens=bucket,
                        num_conversations=conv_count,
                        tokenizer=tokenizer,
                        prefix_ratio=args.prefix_ratio,
                        output_tokens=args.output_tokens,
                        num_clusters=args.num_clusters,
                        seed=args.seed,
                    )
                    fname = f"single_turn_{bucket}_c{conv_count}.jsonl"
                elif family == "multi_turn":
                    records = generate_multi_turn_schedule(
                        prompt_pool=prompt_pool,
                        bucket_tokens=bucket,
                        num_conversations=conv_count,
                        turns_per_conversation=args.turns_per_conversation,
                        tokenizer=tokenizer,
                        prefix_ratio=args.multi_turn_prefix_ratio,
                        output_tokens=args.output_tokens,
                        seed=args.seed,
                    )
                    fname = f"multi_turn_{bucket}_c{conv_count}.jsonl"
                else:
                    print(f"WARNING: Unknown family '{family}', skipping")
                    continue

                out_path = output_dir / fname
                write_jsonl(records, str(out_path))
                total_files += 1
                total_records += len(records)

    print(f"\nDone: {total_files} files, {total_records} total records in {output_dir}")


if __name__ == "__main__":
    main()
