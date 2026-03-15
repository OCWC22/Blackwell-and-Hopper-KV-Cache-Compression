# Coding Agent Datasets for KV Cache Serving Benchmarks

Retrieved and updated on `2026-03-15`.

## Purpose

This note catalogs **public coding-agent conversation datasets** that could serve as realistic prompt pools for KV cache serving benchmarks on the Blackwell and Hopper tracks.

The key motivation is that **coding-agent sessions** are among the most KV-intensive workloads in production today:

- **Long context:** individual sessions routinely hit 100K–8M+ input tokens as agents read files, receive tool outputs, and accumulate conversation history
- **Multi-turn with reusable prefixes:** system prompts, file contents, and earlier turns form large shared prefixes across subsequent requests within a session
- **Tool-use interleaving:** alternating user→assistant→tool→assistant patterns create bursty, variable-length decode sequences with frequent KV reuse opportunities
- **Context compaction events:** real Claude Code sessions include `compact` events that reset the KV working set — directly analogous to KV eviction/restore cycles

These properties make coding-agent traces a strong fit for stress-testing KV tier behavior (hot/warm/cold), restore-vs-recompute costs, and active-session concurrency limits under latency guardrails [L1][L3].

## How this relates to the existing benchmark spec

This document is a **prompt-source supplement** to `actual_kv_serving_benchmark_and_synthetic_data.md`.

It contributes to:

- **Layer A (prompt sources):** real coding-agent conversations as an alternative to ShareGPT-style chat prompts
- **Layer C (conversation schedule synthesis):** multi-turn session structure with natural think-time gaps, tool-call interleaving, and compaction events that can drive active-set scheduling
- **Layer D (restore-aware metadata):** sessions that naturally exhibit prefix reuse, context growth, and eviction-triggering length — all derivable from the raw traces

It does **not** replace the synthetic workload layers (B, D) or the benchmark framing (X-axis = `active_conversations`, slices = prompt-size buckets). Instead, it provides **real conversation shapes** that the synthetic shaping layers can use [L1][L2].

## Dataset catalog

### Tier 1 — Claude Code session exports (via DataClaw / claudeset)

These datasets contain **full agentic coding sessions** exported from Claude Code, including messages, tool calls, thinking blocks, token counts, and session metadata. They are the closest match to production coding-agent KV workloads.

#### 1. `lelouch0110/claudeset-community`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/lelouch0110/claudeset-community` |
| **Rows** | 114 sessions |
| **Size** | 14.4 MB |
| **License** | MIT |
| **Export tool** | [claudeset](https://pypi.org/project/claudeset/) |
| **Models** | Claude Opus 4.5 |
| **Schema health** | Viewer works |

**Schema:** `id`, `project`, `model`, `git_branch`, `claude_version`, `start_time`, `end_time`, `turns` (list of exchange/compact objects), `stats`, `contributor`

**Turn types:**

- **Exchange** — user message + assistant response with `thinking`, `text`, `tool_calls` (tool name, input, output), and `usage` (input/output tokens)
- **Compact** — context compression summary (directly maps to KV eviction events)

**Stats per session:** `exchanges`, `compacts`, `tool_calls`, `input_tokens`, `output_tokens`, `cache_read_tokens`

**KV benchmark suitability:** **High.** Clean schema, includes compaction events (KV eviction analog), tool call outputs (large context injections), and per-turn token accounting. The `cache_read_tokens` field directly measures prefix reuse. Community-sourced from real development tasks (debugging, refactoring, feature building, test writing).

---

#### 2. `peteromallet/my-personal-claude-code-data`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/peteromallet/my-personal-claude-code-data` |
| **Rows** | 549 sessions |
| **Size** | 82.6 MB |
| **License** | Not specified |
| **Export tool** | [DataClaw](https://github.com/banodoco/dataclaw) |
| **Models** | Claude Opus 4.6 (270), Opus 4.5 (256), Sonnet 4.6 (16), Haiku 4.5 (6), Sonnet 4.5 (1) |
| **Input tokens (total)** | 15.1B |
| **Output tokens (total)** | 4.6M |
| **Schema health** | Viewer works (load via `dataclaw-peteromallet` config) |

**Schema:** `session_id`, `project` (14 projects), `model`, `git_branch`, `start_time`, `end_time`, `messages` (role, content, timestamp, thinking, tool_uses), `stats` (user_messages, assistant_messages, tool_uses, input_tokens, output_tokens, skipped_lines)

**KV benchmark suitability:** **High.** Largest single-contributor Claude Code dataset found. 549 sessions across 14 projects provides workload diversity. Per-session token counts range from near-zero to multi-million, giving a natural long-tail distribution for active-set pressure modeling. Multiple model variants allow comparing KV profiles across model sizes.

---

#### 3. `misterkerns/my-personal-claude-code-data`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/misterkerns/my-personal-claude-code-data` |
| **Rows** | 549 sessions |
| **Size** | ~80 MB |
| **License** | Not specified |
| **Export tool** | DataClaw |
| **Models** | Claude Opus 4.6 (270), Opus 4.5 (256), Sonnet 4.6 (16), Haiku 4.5 (6), Sonnet 4.5 (1) |
| **Schema health** | Viewer broken (schema mismatch between `conversations.jsonl` and `metadata.json`) |

**Schema:** Same as `peteromallet` above — session_id, project, model, messages with tool_uses, stats with token counts.

**KV benchmark suitability:** **Medium-High.** Same schema and scale as `peteromallet`. Viewer is broken but data is loadable programmatically. Useful as a second single-contributor source for cross-user workload comparison.

---

#### 4. `REXX-NEW/my-personal-claude-code-data`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/REXX-NEW/my-personal-claude-code-data` |
| **Rows** | 10+ sessions (exact count unknown, viewer broken) |
| **License** | Not specified |
| **Export tool** | DataClaw / claudeset |
| **Models** | Claude Opus 4.5 |
| **Schema health** | Viewer broken (schema mismatch) |

**Schema:** `session_id`, `model`, `git_branch`, `start_time`, `end_time`, `messages` (role, content, timestamp, thinking, tool_uses), `stats` (user_messages, assistant_messages, tool_uses, input_tokens, output_tokens)

**Notable sessions:**

- Session with 25 user messages, 203 assistant messages, 159 tool uses, **8.87M input tokens** over ~4 hours — an extreme long-context KV stress case
- Session with 870 assistant messages — deep multi-turn coding workflow

**KV benchmark suitability:** **Medium.** Small dataset but contains extreme-length sessions (8M+ tokens) that are valuable for stress-testing KV tier boundaries and eviction behavior. Viewer is broken but raw data is accessible.

---

### Tier 2 — Codex / multi-model agent session exports

#### 5. `akenove/my-personal-codex-data`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/akenove/my-personal-codex-data` |
| **Rows** | 203 sessions |
| **Size** | ~50 MB |
| **License** | Not specified |
| **Models** | `o4-mini`, `gpt-5.3-codex`, `claude-opus-4-6-thinking` |
| **Date range** | 2026-02-20 to 2026-02-27 |
| **Projects** | `codex`, `openclaw` (trading bots, Electron apps, Solana bundler, web dev) |
| **Schema health** | Viewer works |

**Schema:** `session_id`, `model`, `git_branch`, `start_time`, `end_time`, `messages` (1 to ~1,240 messages per session), `stats` (input_tokens up to 2M+, tool_uses up to 36+), `project`, `source`

**KV benchmark suitability:** **Medium-High.** Multi-model dataset (OpenAI Codex + Claude) provides cross-runtime workload profiles. Contains both short single-shot tasks and deep multi-turn sessions. The Solana bundler session (2M+ input tokens, 36 tool uses) is a strong long-context test case. Useful for modeling workload mixes where different models have different KV footprints.

---

#### 6. `peteromallet/my-personal-codex-data`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/peteromallet/my-personal-codex-data` |
| **Rows** | Unknown (viewer broken, schema mismatch) |
| **License** | Not specified |
| **Models** | `gpt-5.3-codex` and others |
| **Schema health** | Viewer broken |

**KV benchmark suitability:** **Low-Medium.** Same schema family but viewer is broken and size is unknown. Lower priority unless aggregated with other Codex exports.

---

### Tier 3 — Agent trajectory datasets (structured but not raw sessions)

#### 7. `ByteDance-Seed/Multi-SWE-bench_trajs`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench_trajs` |
| **Downloads/month** | 16,303 |
| **License** | CC0-1.0 |
| **Format** | Custom (not auto-detected by HF viewer) |
| **Task type** | Multi-language issue resolution (SWE-bench leaderboard) |

**KV benchmark suitability:** **Medium.** Contains agent trajectories from the Multi-SWE-bench leaderboard evaluation. These are structured agent traces with tool use across multi-language codebases. The custom format requires manual parsing. Useful for understanding trajectory length distributions across real software engineering tasks, but may lack raw token counts and session-level metadata needed for KV pressure modeling.

---

#### 8. `SWE-bench/SWE-smith`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/SWE-bench/SWE-smith` |
| **Rows** | 59,136 task instances |
| **Repos** | 128 GitHub repositories |
| **Size** | 278 MB |
| **License** | Not specified |

**Schema:** `instance_id`, `patch` (up to 223K chars), `FAIL_TO_PASS`, `PASS_TO_PASS`, `image_name`, `repo`, `problem_statement` (up to 39.4K chars)

**KV benchmark suitability:** **Low-Medium.** Large-scale task dataset but contains **problem statements + gold patches**, not agent interaction traces. No multi-turn structure, no tool-use interleaving, no token accounting. Useful as a **prompt source** (problem statements as first-turn prompts) but not as a session-shape source. Could feed into Layer A of the synthetic data plan as long-context coding problem prompts.

---

### Tier 4 — General conversation datasets (not coding-specific but relevant for workload mixing)

#### 9. `lmsys/lmsys-chat-1m`

| Property | Value |
|----------|-------|
| **URL** | `https://huggingface.co/datasets/lmsys/lmsys-chat-1m` |
| **Rows** | 1,000,000 conversations |
| **Models** | 25 state-of-the-art LLMs |
| **Source** | Chatbot Arena + Vicuna demo |

**KV benchmark suitability:** **Low for coding vertical, Medium for mixed workload modeling.** Massive scale but conversations are general-purpose (not coding-focused), typically short (chat-style), and lack tool-use structure. Useful as a **background traffic generator** to model mixed coding + non-coding workloads on a shared serving cluster, but not as a primary coding-agent prompt source.

---

### Tier 5 — Training data (not public but documented)

#### 10. OmniCoder-9B training trajectories

| Property | Value |
|----------|-------|
| **Reference** | `https://huggingface.co/Tesslate/OmniCoder-9B` |
| **Trajectories** | 425,000+ curated agentic coding trajectories |
| **Source models** | Claude Opus 4.6, GPT-5.4, GPT-5.3-Codex, Gemini 3.1 Pro |
| **Scaffolding** | Claude Code, OpenCode, Codex, Droid |
| **Availability** | Training data not publicly released (model weights only) |

**KV benchmark suitability:** **High if released, N/A currently.** 425K real agentic trajectories with tool calls, error recovery, and edit diffs would be the largest coding-agent dataset by far. The multi-scaffolding coverage (Claude Code, Codex, etc.) would provide diverse workload shapes. Monitor for public release.

---

## Suitability assessment for coding-vertical KV benchmarks

### What we need from a dataset (per existing benchmark spec)

Per `actual_kv_serving_benchmark_and_synthetic_data.md`, the benchmark requires:

1. **Prompt sources** (Layer A) — realistic prompt text at coding-relevant lengths
2. **Conversation structure** (Layer C) — multi-turn scheduling with think-time gaps
3. **Restore-aware metadata** (Layer D) — prefix reuse patterns, context growth, eviction triggers
4. **Token accounting** — input/output token counts for KV pressure modeling
5. **Prompt-size buckets** — ability to bin sessions into 10K/20K/30K/40K/50K+ slices

### Dataset ranking for the coding vertical

| Rank | Dataset | Coding focus | Multi-turn | Token accounting | Prefix reuse signal | Scale |
|------|---------|-------------|------------|-----------------|-------------------|-------|
| 1 | `claudeset-community` | **Strong** | **Yes** (exchange + compact) | **Yes** (per-turn + cache_read) | **Yes** (compact events) | 114 |
| 2 | `peteromallet/claude-code-data` | **Strong** | **Yes** | **Yes** (per-session) | **Partial** (no compact type) | 549 |
| 3 | `akenove/codex-data` | **Strong** | **Yes** | **Yes** (per-session) | **Partial** | 203 |
| 4 | `REXX-NEW/claude-code-data` | **Strong** | **Yes** | **Yes** (extreme range) | **Partial** | ~10+ |
| 5 | `misterkerns/claude-code-data` | **Strong** | **Yes** | **Yes** | **Partial** | 549 |
| 6 | `Multi-SWE-bench_trajs` | **Strong** | **Yes** (trajectories) | **Unknown** | **Unknown** | Large |
| 7 | `SWE-smith` | **Strong** | **No** (single-turn) | **No** | **No** | 59K |
| 8 | `lmsys-chat-1m` | **Weak** | **Yes** | **No** | **No** | 1M |

### Recommended aggregation strategy

1. **Primary prompt pool (coding sessions):** Merge `claudeset-community` + `peteromallet/claude-code-data` + `akenove/codex-data` for ~866 coding-agent sessions with token accounting
2. **Extreme-length stress tests:** Use `REXX-NEW` sessions with 8M+ tokens for KV tier boundary testing
3. **Coding problem prompts:** Use `SWE-smith` problem statements (up to 39.4K chars) as single-turn long-context prompts for Layer A
4. **Background traffic:** Use `lmsys-chat-1m` for mixed-workload modeling if needed
5. **Session-shape extraction:** From the merged pool, extract per-session profiles: total tokens, turn count, tool-call count, max single-turn input, inter-turn gap distribution — these feed directly into Layer C conversation scheduling

### Key gaps

- **No dataset has >1,000 coding-agent sessions** — aggregation across sources is required
- **Token counts are session-level, not per-turn** in most DataClaw exports (except `claudeset-community` which has per-turn `usage`)
- **No dataset includes server-side KV cache metrics** (cache hit rate, eviction count, restore latency) — these must come from the benchmark runtime itself
- **OmniCoder training data (425K trajectories) would close the scale gap** but is not public

## Data loading notes

All DataClaw/claudeset datasets share a common loading pattern:

```python
from datasets import load_dataset

# claudeset-community (clean viewer)
ds_claudeset = load_dataset("lelouch0110/claudeset-community", split="train")

# peteromallet claude code (use dataclaw config)
ds_pete_claude = load_dataset("peteromallet/my-personal-claude-code-data", split="train")

# akenove codex
ds_codex = load_dataset("akenove/my-personal-codex-data", split="train")

# For broken-viewer datasets, load from raw files:
# ds = load_dataset("json", data_files="conversations.jsonl")
```

For session-level KV pressure profiling:

```python
def session_kv_profile(session):
    """Extract KV-relevant metrics from a coding-agent session."""
    stats = session.get("stats", {})
    return {
        "session_id": session["session_id"],
        "total_input_tokens": stats.get("input_tokens", 0),
        "total_output_tokens": stats.get("output_tokens", 0),
        "num_turns": stats.get("user_messages", 0),
        "num_tool_calls": stats.get("tool_uses", 0),
        "model": session.get("model", "unknown"),
        "duration_s": None,  # compute from start_time/end_time
    }
```

## References

- `[L1]` `actual_kv_serving_benchmark_and_synthetic_data.md` — actionable benchmark spec
- `[L2]` `synthetic_kv_working_set_benchmark_mtp_ep.md` — broader research exploration
- `[L3]` `Blackwell/CLAUDE.md` — four benchmark scenarios and execution order
- `[R1]` TensorRT-LLM `trtllm-bench` and `prepare_dataset.py` — synthetic request generation
- `[R2]` vLLM benchmarking docs — ShareGPT, prefix_repetition, long-document QA patterns
- `[R3]` BurstGPT — trace-aligned synthetic workload construction
- `[R4]` NVIDIA GenAI-Perf — TTFT, inter-token latency, goodput benchmarking
- `[R5]` DataClaw — `https://github.com/banodoco/dataclaw`
- `[R6]` claudeset — `https://pypi.org/project/claudeset/`
- `[R7]` SCBench — KV cache-centric long-context benchmark — `https://huggingface.co/papers/2412.10319`
- `[R8]` OmniCoder-9B — `https://huggingface.co/Tesslate/OmniCoder-9B`
- `[R9]` Multi-SWE-bench — `https://arxiv.org/abs/2504.02605`
- `[R10]` SWE-smith — `https://huggingface.co/datasets/SWE-bench/SWE-smith`
- `[R11]` LMSYS-Chat-1M — `https://huggingface.co/datasets/lmsys/lmsys-chat-1m`
