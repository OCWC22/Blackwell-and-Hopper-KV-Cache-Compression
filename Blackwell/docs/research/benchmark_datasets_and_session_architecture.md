# HuggingFace Dataset Map and Claude Code Session Architecture for KV Serving Benchmark

## Purpose

This document maps **specific HuggingFace datasets** to each layer of the 4-layer benchmark aggregation stack defined in `actual_kv_serving_benchmark_and_synthetic_data.md`, and defines a **5-session Claude Code execution architecture** for building and running the full KV serving capacity benchmark.

## Relation to existing docs

- `actual_kv_serving_benchmark_and_synthetic_data.md` — defines the benchmark question, aggregation stack, synthetic data layers, and conversation schedule schema. **This file is the actionable spec.**
- `coding_agent_datasets_for_kv_benchmarks.md` — catalogs coding-agent session datasets as supplementary prompt sources (Layer A).
- This document adds: dataset IDs, data preparation pipeline, session execution plan, and integration points with existing repo scripts.

---

## Part 1: HuggingFace Dataset Catalog

### v1 Core Datasets (Serving Benchmark)

| Dataset | HuggingFace ID | Layer | Role |
|---------|---------------|-------|------|
| **BurstGPT** | `lzzmm/BurstGPT` | 2 (traffic) | Arrival patterns, session timing, Zipf request-length distributions. 10.31M traces from Azure OpenAI over 213 days [R3] |
| **ShareGPT Vicuna** | `anon8231489123/ShareGPT_Vicuna_unfiltered` | 3 (reuse) | Chat-style prompt pool (~90K conversations). Classic source used by vLLM benchmarks [R2] |
| **ShareGPT Long** | `Arist12/EABF-ShareGPT-Long-3.5k` | 3 (reuse) | Long conversations (>10K tokens), English-only, pre-filtered |
| **OpenAI MRCR** | `openai/mrcr` | 3 (reuse) | Multi-turn recall challenge, 4K–1M tokens. Tests chat history reuse across turns |

### v1 Quality Backstop Datasets

| Dataset | HuggingFace ID / Source | Layer | Role |
|---------|------------------------|-------|------|
| **LongBench v1** | `THUDM/LongBench` | 4 (quality) | 6 categories, 21 tasks, bilingual. Single/multi-doc QA, summarization, code [R10] |
| **LongBench v2** | `THUDM/LongBench-v2` | 4 (quality) | 503 MCQ, 8K–2M words, deeper reasoning [R10] |
| **RULER** (pre-generated) | `rbiswasfc/ruler` | 4 (quality) | Pre-generated at 4K/8K. For custom lengths, generate via NVIDIA/RULER repo [R9] |
| **RULER** (generator) | GitHub `NVIDIA/RULER` | 4 (quality) | Generate synthetic tasks at 10K/20K/30K/40K/50K to match benchmark prompt buckets [R9] |

### v2 Expansion Datasets (Future)

| Dataset | HuggingFace ID | Role |
|---------|---------------|------|
| ShareGPT 52K | `RyokoAI/ShareGPT52K` | Additional chat prompt pool |
| UltraChat 200K | `HuggingFaceH4/ultrachat_200k` | Multi-turn instructional conversations |
| Long-Data-Collections | `togethercomputer/Long-Data-Collections` | Long document source (books, 180K–1M tokens) |
| NarrativeQA | `deepmind/narrativeqa` | Story-based long document QA |
| LongBench-Pro | `caskcsg/LongBench-Pro` | 1,500 samples, 8K–256K tokens |
| InfiniteBench | GitHub `OpenBMB/InfiniteBench` | 12 tasks, 100K+ tokens |
| longctx_bench | GitHub `henryzhongsc/longctx_bench` | Compression/eviction tradeoff backstop [R11] |

### Coding-Agent Datasets (Supplementary)

See `coding_agent_datasets_for_kv_benchmarks.md` for the full catalog. Top picks:

| Dataset | HuggingFace ID | Scale |
|---------|---------------|-------|
| claudeset-community | `lelouch0110/claudeset-community` | 114 sessions |
| peteromallet claude-code | `peteromallet/my-personal-claude-code-data` | 549 sessions |
| akenove codex | `akenove/my-personal-codex-data` | 203 sessions |

---

## Part 2: Data Preparation Pipeline

### Pipeline overview

```
generate_benchmark_traces.py

1. load_datasets()
   ├── load ShareGPT → extract conversation text
   ├── load BurstGPT → extract timing distributions (think_time, request_length)
   └── load MRCR → extract multi-turn conversation structures

2. tokenize_and_bucket(prompts, tokenizer, buckets)
   ├── tokenize all prompts with target model tokenizer
   ├── for short prompts: concatenate turns to reach bucket target
   ├── for long prompts: truncate to bucket boundary
   └── assign to nearest bucket (±5% tolerance)

3. fit_burstgpt_distributions(burstgpt_data)
   ├── fit log-normal distribution on request lengths
   ├── fit log-normal arrival time distribution
   └── extract think_time_ms distribution from conversation intervals

4. generate_prefix_clusters(bucketed_prompts, shared_prefix_ratio)
   ├── for each bucket: group prompts into reuse clusters
   ├── assign shared prefix + unique suffix per cluster
   └── assign reuse_cluster_id

5. synthesize_conversation_schedule(
       bucketed_prompts, timing_distributions, workload_family)
   ├── if single_turn: one prompt per conversation, no think_time
   ├── if multi_turn: multiple turns with think_time gaps from BurstGPT
   └── emit JSONL records with full schema

6. write_traces(records, output_dir)
   └── one file per (workload_family, prompt_bucket)
       e.g., single_turn_active_set_10k.jsonl
```

### Output JSONL schema (per record)

Each JSONL record matches the conversation schedule schema from the benchmark spec (Lines 314–327 of `actual_kv_serving_benchmark_and_synthetic_data.md`):

```json
{
  "conversation_id": "conv_0042",
  "turn_index": 0,
  "workload_family": "single_turn_active_set",
  "prompt_bucket_tokens": 10000,
  "shared_prefix_ratio": 0.8,
  "reuse_cluster_id": "cluster_03",
  "think_time_ms": 0,
  "target_output_tokens": 256,
  "sampling_mode": "greedy",
  "prompt_text": "...",
  "estimated_reusable_prefix_tokens": 8000,
  "source_dataset": "sharegpt"
}
```

### CLI usage

```bash
# Single-turn traces from ShareGPT + BurstGPT timing
python scripts/generate_benchmark_traces.py \
    --datasets sharegpt,burstgpt \
    --workload-family single_turn_active_set \
    --num-conversations 64

# Multi-turn traces with MRCR conversations
python scripts/generate_benchmark_traces.py \
    --datasets sharegpt,burstgpt,mrcr \
    --workload-family synthetic_multi_turn_active_set \
    --num-conversations 32

# All defaults — both workload families, all v1 datasets
python scripts/generate_benchmark_traces.py --all

# Custom tokenizer and buckets
python scripts/generate_benchmark_traces.py \
    --datasets sharegpt,burstgpt,mrcr \
    --prompt-buckets 10000,20000,30000,40000,50000 \
    --workload-family both \
    --num-conversations 64 \
    --shared-prefix-ratio 0.8 \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --output-dir data/traces/
```

### Dependencies

```
pip install datasets transformers numpy scipy
```

- `datasets` — HuggingFace Hub download
- `transformers` — tokenizer only (no model loading)
- `numpy` — distribution fitting and sampling
- `scipy` — Zipf fitting from BurstGPT (optional, fallback to numpy)

### Output artifacts

```
Blackwell/data/traces/
├── single_turn_active_set_10k.jsonl
├── single_turn_active_set_20k.jsonl
├── single_turn_active_set_30k.jsonl
├── single_turn_active_set_40k.jsonl
├── single_turn_active_set_50k.jsonl
├── synthetic_multi_turn_active_set_10k.jsonl
├── synthetic_multi_turn_active_set_20k.jsonl
├── synthetic_multi_turn_active_set_30k.jsonl
├── synthetic_multi_turn_active_set_40k.jsonl
├── synthetic_multi_turn_active_set_50k.jsonl
└── trace_metadata.json
```

---

## Part 3: Claude Code Session Architecture

### Session 1: Data Preparation

**Goal**: Download datasets and generate benchmark trace files

**Estimated scope**: One Claude Code session

1. Install dependencies: `pip install datasets transformers numpy scipy`
2. Create `Blackwell/data/` directory structure
3. Run `generate_benchmark_traces.py` for single_turn workload across all buckets
4. Run again for multi_turn workload
5. Verify: spot-check JSONL files for correct token counts, prefix structure, timing distributions

**Verification checks:**
- JSONL records parse correctly
- Token counts match bucket targets (±5%)
- `think_time_ms` values in multi-turn traces are non-zero and vary (not all identical)
- `reuse_cluster_id` values repeat across records (prefix reuse structure exists)
- `trace_metadata.json` contains valid distribution parameters

**Existing scripts used**: None directly (this is data prep)

**Output artifacts**: `data/traces/*.jsonl`, `data/traces/trace_metadata.json`

---

### Session 2: Single-Turn Capacity Sweep

**Goal**: Run `single_turn_active_set` workload, find max active conversations per bucket

1. Verify env probe: `bash scripts/env_probe.sh`
2. For each prompt bucket (10K → 50K):
   - Feed generated trace to `scripts/serve_and_bench.py`
   - Sweep `--sweep-concurrency 1,2,4,8,16,32`
   - Use `--p95-tpot-limit-ms 100` (or configured target)
   - Pass `--workload-type repeated_prefix --prefix-ratio 0.8`
3. Collect all results JSONs to `results/single_turn_sweep/`
4. Run `scripts/compare_results.py` across buckets

**Integration with `serve_and_bench.py`:**

The generated JSONL traces from Session 1 provide the prompt text and prefix structure. The script's existing `generate_prompts()` function can be replaced by loading prompts from the JSONL file, preserving all sweep logic:

```bash
# Example: feed 10K single-turn trace to serve_and_bench
python scripts/serve_and_bench.py \
    --engine tensorrt_llm --kv-mode nvfp4 \
    --context-length 10000 --prefix-ratio 0.8 \
    --sweep-concurrency 1,2,4,8,16,32 \
    --p95-tpot-limit-ms 100 \
    --output results/single_turn_sweep/st_10k_nvfp4.json
```

**Existing scripts used**: `env_probe.sh`, `serve_and_bench.py`, `compare_results.py`

**Output artifacts**: `results/single_turn_sweep/*.json`

---

### Session 3: Multi-Turn Active Set Sweep

**Goal**: Run `synthetic_multi_turn_active_set`, measure reuse under realistic conversation patterns

1. Same sweep structure as Session 2 but with multi-turn traces
2. Key differences:
   - `think_time_ms` gaps between turns (from BurstGPT distributions)
   - Follow-up turns that partially reuse earlier context
   - Growing prefix reuse across conversation turns
3. Focus metrics:
   - `cache_hit_rate` — how much KV is reused vs recomputed
   - `restored_kv_bytes` — volume of KV restored from host
   - `promotion_latency_ms` — cost of promoting cold KV to hot tier
4. Run `scripts/compare_results.py` single-turn vs multi-turn

**Existing scripts used**: `serve_and_bench.py`, `compare_results.py`

**Output artifacts**: `results/multi_turn_sweep/*.json`

---

### Session 4: Quality Backstop Evaluation (Required)

**Goal**: Verify long-context quality is preserved under tiered-KV configuration

1. Download LongBench v1/v2:
   ```python
   from datasets import load_dataset
   ds_lb1 = load_dataset("THUDM/LongBench", split="test")
   ds_lb2 = load_dataset("THUDM/LongBench-v2", split="test")
   ```

2. Generate RULER tasks at benchmark bucket lengths (10K/20K/30K/40K/50K):
   - Clone `NVIDIA/RULER`, configure `scripts/synthetic.yaml` for target lengths
   - Run `prepare_data.py` to generate task instances

3. Run quality evaluation under three configurations:
   - BF16 baseline (no tiering)
   - NVFP4 hot-tier (no offload)
   - NVFP4 + host offload (full tiered config)

4. Compute `quality_delta_vs_best_baseline` for each config

5. **Gate**: if quality delta > 1%, flag for investigation before continuing benchmark

**Existing scripts used**: `run_baseline.py` (quality evaluation mode), `run_tiered_experiment.py`

**Output artifacts**: `results/quality_backstop/*.json`

---

### Session 5: Analysis and Decision Memo

**Goal**: Aggregate all results, produce capacity curves, write recommendation

1. Run `scripts/compare_results.py` across all sessions:
   ```bash
   python scripts/compare_results.py \
       results/single_turn_sweep/*.json \
       results/multi_turn_sweep/*.json \
       results/quality_backstop/*.json
   ```

2. For each (workload_family, prompt_bucket), extract:
   - `max_active_conversations_at_target` (where p95 TPOT < limit)
   - `peak_hbm_gb` at max capacity point
   - `cache_hit_rate` at max capacity point

3. Generate comparison tables: baseline vs tiered for each metric

4. Write decision memo (`results/decision_memo.md`) with:
   - Capacity improvement (%) per bucket
   - HBM reduction (%) per bucket
   - Quality delta per config
   - Bottleneck identification (HBM-bound? latency-bound? bandwidth-bound?)
   - Recommendation: which tiered config to deploy

**Existing scripts used**: `compare_results.py`

**Output artifacts**: `results/decision_memo.md`

---

## Part 4: Integration Points with Existing Repo Scripts

### How generated traces connect to the existing harness

| Trace field | Maps to | Used by |
|-------------|---------|---------|
| `prompt_text` | Prompt input to serving engine | `serve_and_bench.py` → `generate_prompts()` |
| `prompt_bucket_tokens` | `--context-length` | `serve_and_bench.py`, `run_baseline.py` |
| `shared_prefix_ratio` | `--prefix-ratio` | `serve_and_bench.py`, `run_baseline.py` |
| `target_output_tokens` | `--max-tokens` | `serve_and_bench.py`, `run_baseline.py` |
| `think_time_ms` | Inter-request delay in multi-turn scheduling | Future: extended `serve_and_bench.py` |
| `reuse_cluster_id` | Groups requests that share a prefix | Future: APC hit-rate analysis |
| `estimated_reusable_prefix_tokens` | Expected `cache_hit_rate` baseline | `compare_results.py` |

### Extending `serve_and_bench.py` to consume trace JSONL

The existing `generate_prompts()` function (line 192–215 of `serve_and_bench.py`) generates synthetic repeated-prefix prompts. To use the generated traces instead, add an optional `--trace-file` argument that loads prompts from JSONL:

```python
# Future extension (not implemented yet):
# if args.trace_file:
#     prompts = load_trace_prompts(args.trace_file)
# else:
#     prompts = generate_prompts(args.context_length, ...)
```

This keeps the existing harness working unchanged while allowing the generated traces to be fed in when ready.

---

## Verification Criteria

1. `generate_benchmark_traces.py` runs end-to-end with `--datasets sharegpt` and produces valid JSONL
2. JSONL records match the conversation schedule schema from `actual_kv_serving_benchmark_and_synthetic_data.md` (Lines 314–327)
3. Token counts in generated traces match bucket targets (±5%)
4. BurstGPT timing distributions produce realistic `think_time_ms` values (not all zeros, not all identical)
5. Quality backstop datasets (LongBench, RULER) load correctly from HuggingFace Hub
6. Doc references only existing repo scripts and consistent dataset IDs

---

## Sources

### External sources

- **[R2] vLLM Benchmark CLI** — `https://docs.vllm.ai/en/latest/benchmarking/cli/`
- **[R3] BurstGPT** — `https://arxiv.org/html/2401.17644v3` — `lzzmm/BurstGPT`
- **[R4] NVIDIA GenAI-Perf** — `https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai_perf.html`
- **[R9] NVIDIA RULER** — `https://github.com/NVIDIA/RULER` — `rbiswasfc/ruler`
- **[R10] LongBench** — `https://github.com/THUDM/LongBench` — `THUDM/LongBench`, `THUDM/LongBench-v2`
- **[R11] longctx_bench** — `https://github.com/henryzhongsc/longctx_bench`

### Repo-local sources

- **[L1]** `Blackwell/scripts/serve_and_bench.py` — online serving benchmark with concurrency sweeps
- **[L2]** `Blackwell/scripts/run_baseline.py` — offline baseline benchmark
- **[L3]** `Blackwell/scripts/kv_bench_utils.py` — shared utilities and result schema
- **[L4]** `Blackwell/scripts/run_tiered_experiment.py` — tiered KV experiment runner
- **[L5]** `Blackwell/scripts/compare_results.py` — result comparison and analysis
- **[L6]** `Blackwell/docs/research/actual_kv_serving_benchmark_and_synthetic_data.md` — actionable benchmark spec
- **[L7]** `Blackwell/docs/research/coding_agent_datasets_for_kv_benchmarks.md` — coding-agent dataset catalog
