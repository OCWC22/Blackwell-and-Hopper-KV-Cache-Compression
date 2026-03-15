#!/usr/bin/env python3
"""Ingest and normalize coding-agent datasets from HuggingFace for KV benchmarks.

Downloads, normalizes, and profiles the top coding-agent session datasets
identified in docs/research/coding_agent_datasets_for_kv_benchmarks.md.

Outputs normalized sessions, extracted prompts (Layer A), and session shape
profiles (Layer C) for use by generate_synthetic_data.py and the benchmark
harness scripts.

Usage:
    # Ingest all three recommended datasets
    python scripts/ingest_coding_datasets.py \\
        --datasets claudeset,peteromallet,akenove \\
        --output-dir data/coding_agent/

    # Ingest only claudeset (smallest, 14.4 MB)
    python scripts/ingest_coding_datasets.py \\
        --datasets claudeset \\
        --output-dir data/coding_agent/

    # Custom cache directory
    python scripts/ingest_coding_datasets.py \\
        --datasets claudeset,peteromallet \\
        --cache-dir data/cache/ \\
        --output-dir data/coding_agent/
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class NormalizedSession:
    """Unified session schema across all coding-agent datasets."""
    session_id: str
    source_dataset: str
    model: str
    total_input_tokens: int
    total_output_tokens: int
    num_turns: int
    num_tool_calls: int
    messages: list[dict]  # [{role, content, token_count_est}]
    duration_s: float | None
    has_compaction_events: bool


# ---------------------------------------------------------------------------
# Per-dataset adapters
#
# TODO(engineer): The schemas below are best-effort based on the dataset
# catalog in docs/research/coding_agent_datasets_for_kv_benchmarks.md.
# Each adapter defensively handles missing/changed fields, but you should
# verify against the actual HuggingFace dataset viewer after first download.
# If a dataset's schema has changed, update the adapter accordingly.
#
# To add a new dataset:
#   1. Add a load_<name>(cache_dir) function returning list[NormalizedSession]
#   2. Register it in DATASET_LOADERS dict below
#   3. Document the schema in the docstring
# ---------------------------------------------------------------------------

def load_claudeset(cache_dir: str) -> list[NormalizedSession]:
    """Load lelouch0110/claudeset-community and normalize.

    Schema: session_id, uuid, cwd, model, turns (list of turn objects).
    Each turn has: type (human|assistant|exchange|compact), content/text,
    usage with cache_read_tokens.
    """
    from datasets import load_dataset

    ds = load_dataset("lelouch0110/claudeset-community", split="train",
                      cache_dir=cache_dir)
    sessions = []
    for row in ds:
        session_id = row.get("session_id", row.get("uuid", "unknown"))
        model = row.get("model", "unknown")
        turns = row.get("turns", [])
        if isinstance(turns, str):
            try:
                turns = json.loads(turns)
            except json.JSONDecodeError:
                turns = []

        messages = []
        total_input = 0
        total_output = 0
        num_tool_calls = 0
        has_compaction = False

        for t in turns:
            if not isinstance(t, dict):
                continue
            turn_type = t.get("type", "unknown")
            content = t.get("content", t.get("text", ""))
            if not isinstance(content, str):
                content = str(content) if content else ""

            usage = t.get("usage", {})
            if isinstance(usage, str):
                try:
                    usage = json.loads(usage)
                except (json.JSONDecodeError, TypeError):
                    usage = {}
            if not isinstance(usage, dict):
                usage = {}

            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            total_input += input_tokens
            total_output += output_tokens

            if turn_type == "compact":
                has_compaction = True
            if turn_type in ("tool_use", "tool_result"):
                num_tool_calls += 1

            role = "user" if turn_type == "human" else "assistant"
            messages.append({
                "role": role,
                "content": content[:10000],  # cap for memory
                "token_count_est": input_tokens + output_tokens,
            })

        sessions.append(NormalizedSession(
            session_id=str(session_id),
            source_dataset="claudeset-community",
            model=str(model),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            num_turns=len(messages),
            num_tool_calls=num_tool_calls,
            messages=messages,
            duration_s=None,
            has_compaction_events=has_compaction,
        ))

    print(f"claudeset-community: {len(sessions)} sessions loaded")
    return sessions


def load_peteromallet(cache_dir: str) -> list[NormalizedSession]:
    """Load peteromallet/my-personal-claude-code-data and normalize.

    Schema: session_id, messages (list), stats with input_tokens/output_tokens.
    """
    from datasets import load_dataset

    ds = load_dataset("peteromallet/my-personal-claude-code-data", split="train",
                      cache_dir=cache_dir)
    sessions = []
    for row in ds:
        session_id = row.get("session_id", row.get("id", "unknown"))
        stats = row.get("stats", {})
        if isinstance(stats, str):
            try:
                stats = json.loads(stats)
            except (json.JSONDecodeError, TypeError):
                stats = {}
        if not isinstance(stats, dict):
            stats = {}

        raw_messages = row.get("messages", [])
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except json.JSONDecodeError:
                raw_messages = []

        model = row.get("model", stats.get("model", "unknown"))
        total_input = stats.get("input_tokens", 0) or 0
        total_output = stats.get("output_tokens", 0) or 0
        num_tool_calls = stats.get("tool_uses", 0) or 0
        duration = stats.get("duration_s", None)

        messages = []
        for m in raw_messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content else ""
            messages.append({
                "role": role,
                "content": content[:10000],
                "token_count_est": 0,
            })

        sessions.append(NormalizedSession(
            session_id=str(session_id),
            source_dataset="peteromallet-claude-code",
            model=str(model),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            num_turns=len(messages),
            num_tool_calls=num_tool_calls,
            messages=messages,
            duration_s=duration,
            has_compaction_events=False,
        ))

    print(f"peteromallet-claude-code: {len(sessions)} sessions loaded")
    return sessions


def load_akenove_codex(cache_dir: str) -> list[NormalizedSession]:
    """Load akenove/my-personal-codex-data and normalize.

    Schema: session_id, messages, stats with input_tokens/output_tokens, model.
    """
    from datasets import load_dataset

    ds = load_dataset("akenove/my-personal-codex-data", split="train",
                      cache_dir=cache_dir)
    sessions = []
    for row in ds:
        session_id = row.get("session_id", row.get("id", "unknown"))
        stats = row.get("stats", {})
        if isinstance(stats, str):
            try:
                stats = json.loads(stats)
            except (json.JSONDecodeError, TypeError):
                stats = {}
        if not isinstance(stats, dict):
            stats = {}

        raw_messages = row.get("messages", [])
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except json.JSONDecodeError:
                raw_messages = []

        model = row.get("model", stats.get("model", "unknown"))
        total_input = stats.get("input_tokens", 0) or 0
        total_output = stats.get("output_tokens", 0) or 0
        num_tool_calls = stats.get("tool_uses", 0) or 0

        messages = []
        for m in raw_messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content else ""
            messages.append({
                "role": role,
                "content": content[:10000],
                "token_count_est": 0,
            })

        sessions.append(NormalizedSession(
            session_id=str(session_id),
            source_dataset="akenove-codex",
            model=str(model),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            num_turns=len(messages),
            num_tool_calls=num_tool_calls,
            messages=messages,
            duration_s=None,
            has_compaction_events=False,
        ))

    print(f"akenove-codex: {len(sessions)} sessions loaded")
    return sessions


# ---------------------------------------------------------------------------
# Aggregation and profiling
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "claudeset": load_claudeset,
    "peteromallet": load_peteromallet,
    "akenove": load_akenove_codex,
}

PROMPT_BUCKETS = [10_000, 20_000, 30_000, 40_000, 50_000]


def compute_percentiles(values: list[float], percentiles=(50, 75, 90, 95, 99)) -> dict:
    """Compute percentiles for a list of values."""
    if not values:
        return {f"p{p}": 0 for p in percentiles}
    sorted_v = sorted(values)
    result = {}
    for p in percentiles:
        k = (len(sorted_v) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_v) - 1)
        d = k - f
        result[f"p{p}"] = sorted_v[f] + d * (sorted_v[c] - sorted_v[f])
    return result


def merge_and_profile(sessions: list[NormalizedSession]) -> dict:
    """Compute aggregate statistics for the merged session pool."""
    if not sessions:
        return {"total_sessions": 0}

    input_tokens = [s.total_input_tokens for s in sessions]
    output_tokens = [s.total_output_tokens for s in sessions]
    turn_counts = [s.num_turns for s in sessions]
    tool_calls = [s.num_tool_calls for s in sessions]

    # Bucket distribution
    bucket_counts = {b: 0 for b in PROMPT_BUCKETS}
    bucket_counts["over_50000"] = 0
    for s in sessions:
        placed = False
        for b in PROMPT_BUCKETS:
            if s.total_input_tokens <= b:
                bucket_counts[b] += 1
                placed = True
                break
        if not placed:
            bucket_counts["over_50000"] += 1

    return {
        "total_sessions": len(sessions),
        "sources": dict(sorted(
            {ds: sum(1 for s in sessions if s.source_dataset == ds)
             for ds in set(s.source_dataset for s in sessions)}.items()
        )),
        "models": dict(sorted(
            {m: sum(1 for s in sessions if s.model == m)
             for m in set(s.model for s in sessions)}.items()
        )),
        "input_tokens": {
            "total": sum(input_tokens),
            "mean": sum(input_tokens) / len(input_tokens),
            "max": max(input_tokens),
            **compute_percentiles(input_tokens),
        },
        "output_tokens": {
            "total": sum(output_tokens),
            "mean": sum(output_tokens) / len(output_tokens),
            **compute_percentiles(output_tokens),
        },
        "turn_count": {
            "mean": sum(turn_counts) / len(turn_counts),
            "max": max(turn_counts),
            **compute_percentiles(turn_counts),
        },
        "tool_calls": {
            "total": sum(tool_calls),
            "sessions_with_tools": sum(1 for t in tool_calls if t > 0),
        },
        "sessions_with_compaction": sum(1 for s in sessions if s.has_compaction_events),
        "sessions_per_bucket": bucket_counts,
    }


# ---------------------------------------------------------------------------
# Prompt and shape extraction
# ---------------------------------------------------------------------------

def extract_first_turn_prompts(
    sessions: list[NormalizedSession],
    min_length: int = 100,
) -> list[dict]:
    """Extract first-turn user prompts suitable for Layer A prompt pool."""
    prompts = []
    for s in sessions:
        for m in s.messages:
            if m["role"] == "user" and len(m["content"]) >= min_length:
                prompts.append({
                    "prompt": m["content"],
                    "source_session_id": s.session_id,
                    "source_dataset": s.source_dataset,
                    "estimated_chars": len(m["content"]),
                })
                break  # only first user turn per session
    return prompts


def extract_session_shapes(sessions: list[NormalizedSession]) -> list[dict]:
    """Extract per-session shape profiles for Layer C scheduling."""
    shapes = []
    for s in sessions:
        user_turns = [m for m in s.messages if m["role"] == "user"]
        assistant_turns = [m for m in s.messages if m["role"] == "assistant"]
        shapes.append({
            "session_id": s.session_id,
            "source_dataset": s.source_dataset,
            "total_input_tokens": s.total_input_tokens,
            "total_output_tokens": s.total_output_tokens,
            "num_turns": s.num_turns,
            "num_user_turns": len(user_turns),
            "num_assistant_turns": len(assistant_turns),
            "num_tool_calls": s.num_tool_calls,
            "has_compaction": s.has_compaction_events,
            "max_single_turn_chars": max(
                (len(m["content"]) for m in s.messages), default=0
            ),
        })
    return shapes


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: str):
    """Write list of dicts to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"Wrote {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Ingest coding-agent datasets from HuggingFace"
    )
    p.add_argument("--datasets", default="claudeset,peteromallet,akenove",
                   help="Comma-separated dataset names: claudeset, peteromallet, akenove")
    p.add_argument("--cache-dir", default="data/cache/",
                   help="HuggingFace download cache directory")
    p.add_argument("--output-dir", default="data/coding_agent/",
                   help="Output directory for normalized data")
    p.add_argument("--min-prompt-length", type=int, default=100,
                   help="Minimum character length for extracted prompts")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = str(repo_root / args.cache_dir)
    output_dir = repo_root / args.output_dir
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(str(output_dir), exist_ok=True)

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    all_sessions = []

    for name in dataset_names:
        loader = DATASET_LOADERS.get(name)
        if loader is None:
            print(f"WARNING: Unknown dataset '{name}'. "
                  f"Available: {', '.join(DATASET_LOADERS.keys())}")
            continue
        try:
            sessions = loader(cache_dir)
            all_sessions.extend(sessions)
        except Exception as e:
            print(f"ERROR loading {name}: {e}")
            continue

    if not all_sessions:
        print("No sessions loaded. Exiting.")
        return

    # Write normalized sessions (without full message content for size)
    session_records = []
    for s in all_sessions:
        d = asdict(s)
        # Truncate messages for the merged file to keep it manageable
        d["messages"] = d["messages"][:50]  # cap at 50 turns
        session_records.append(d)
    write_jsonl(session_records, str(output_dir / "merged_sessions.jsonl"))

    # Extract prompts and shapes
    prompts = extract_first_turn_prompts(all_sessions, args.min_prompt_length)
    write_jsonl(prompts, str(output_dir / "first_turn_prompts.jsonl"))

    shapes = extract_session_shapes(all_sessions)
    write_jsonl(shapes, str(output_dir / "session_shapes.jsonl"))

    # Profile
    profile = merge_and_profile(all_sessions)
    profile_path = str(output_dir / "aggregate_profile.json")
    os.makedirs(os.path.dirname(profile_path) or ".", exist_ok=True)
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"Profile written to {profile_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"INGESTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total sessions:       {profile['total_sessions']}")
    print(f"Sources:              {profile['sources']}")
    print(f"Total input tokens:   {profile['input_tokens']['total']:,}")
    print(f"Extracted prompts:    {len(prompts)}")
    print(f"Session shapes:       {len(shapes)}")
    print(f"With compaction:      {profile['sessions_with_compaction']}")
    print(f"Bucket distribution:  {profile['sessions_per_bucket']}")


if __name__ == "__main__":
    main()
