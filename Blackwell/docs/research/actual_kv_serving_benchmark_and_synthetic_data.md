# Actual KV Serving Benchmark and Synthetic Data Plan

## Purpose

This note defines the **actual benchmark** we should run for the Blackwell track and the **synthetic data sources** we can use to build it.

This document is intentionally narrower than `synthetic_kv_working_set_benchmark_mtp_ep.md`.

This version:

- keeps the benchmark centered on **interactive KV-serving pressure**
- keeps the Blackwell primary path centered on **TensorRT-LLM** and the existing repo harnesses [L1][L2][L3]
- uses **existing benchmark patterns and datasets** where possible instead of inventing a bespoke text workload from scratch [R1][R2][R3][R4]
- treats **EP / MoE routing** as **out of scope** for the benchmark v1 in this file

## What this file replaces in practice

Use this file as the **actionable benchmark spec**.

Treat `synthetic_kv_working_set_benchmark_mtp_ep.md` as the broader research exploration.

The practical simplification is:

- keep **KV working-set pressure**
- keep **restore vs recompute cost**
- keep **interactive concurrency**
- drop **EP / MoE skew** from the required benchmark scope for now

## Benchmark question

The main question is:

- **How many active conversations can one Blackwell GPU sustain, at fixed prompt-size slices, while preserving interactive latency and useful KV reuse / restore behavior?** [L1][L3][R5][R6]

That aligns with the repo's Blackwell Scenario 2 and Scenario 3 framing around:

- more sessions at fixed latency target
- better TTFT on reused prefixes
- lower peak HBM
- longer effective context [L3][L4]

## Scope and runtime honesty

This benchmark stays on the Blackwell track:

- **Primary path:** TensorRT-LLM on Blackwell / B200 [L3][L4]
- **Follow-up comparison path:** vLLM with FP8 KV and optional LMCache [L3][L4]

This benchmark is about **serving capacity** and **KV behavior**.

It is not a claim that we are building a new serving stack. That stays consistent with the repo guidance and the existing harnesses `serve_and_bench.py`, `run_baseline.py`, and `run_tiered_experiment.py` [L1][L2][L3].

## Primary benchmark framing

### Primary X-axis

Use:

- **`active_conversations`**

Interpretation:

- each conversation has reusable or restorable KV state
- each conversation competes for GPU KV capacity and restore bandwidth
- the benchmark asks whether interactive latency remains within guardrails as the active set grows [L1][R6][R7]

### Prompt-size slices

Use separate slices for first-turn prompt sizes:

- **10K tokens**
- **20K tokens**
- **30K tokens**
- **40K tokens**
- **50K tokens**

These are workload slices, not the main X-axis.

### Workload families

Run two workload families:

- **`single_turn_active_set`**
- **`synthetic_multi_turn_active_set`**

This matches the repo's existing split between repeated-prefix and reuse-oriented workloads and keeps the benchmark focused on reusable KV state rather than generic offline throughput [L1][L2].

## What we should borrow from existing benchmarks

We do **not** need to invent every component ourselves.

We can consolidate from existing benchmark ecosystems.

### 1. TensorRT-LLM synthetic request generation

TensorRT-LLM documents `trtllm-bench` as a throughput and latency benchmark tool and documents `prepare_dataset.py` for generating synthetic request datasets such as a fixed-length `token-norm-dist` workload with configurable input mean, output mean, and standard deviation [R1].

That gives us a clean source for:

- fixed-length synthetic sweeps
- controlled input/output length distributions
- offline calibration workloads before we run the online active-conversation benchmark [R1]

### 2. vLLM benchmark datasets and benchmark patterns

vLLM documents:

- online serving benchmarks against **ShareGPT**
- **custom JSONL** prompt datasets with a `prompt` field
- a **long document QA** benchmark with prefix caching
- a **prefix caching** benchmark
- a synthetic **`prefix_repetition`** dataset with configurable prefix length, suffix length, number of prefixes, and output length [R2]

That gives us a good menu of reusable workload patterns:

- ShareGPT-style prompt pools
- prefix-repetition synthetic workloads
- long-document QA reuse patterns
- custom JSONL prompt files for our own synthetic traces [R2]

### 3. BurstGPT trace-aligned synthetic generation

BurstGPT is useful because it is explicit about how to build a serving workload when you do **not** have private production traces.

BurstGPT describes:

- request lengths following a **Zipf distribution** in their study
- using a prompt pool sampled from existing dialogue corpora
- truncating prompts to match the target request-length distribution
- evaluating serving systems with latency, throughput, jitter, and failure-oriented metrics [R3]

That is exactly the kind of reasoning we need here:

- use public prompt pools
- fit serving-relevant distributions
- do not pretend we have real customer traces [R3]

### 4. NVIDIA GenAI-Perf metric framing

NVIDIA GenAI-Perf explicitly presents TTFT, request latency, inter-token latency analysis, and goodput-style benchmarking under metric constraints / service level objectives [R4].

That is useful for our benchmark because our primary KPI is not just raw throughput; it is **capacity under latency guardrails** [R4][L1].

### 5. vLLM automatic prefix caching examples

vLLM's Automatic Prefix Caching documentation is directly relevant because it explicitly identifies two example workload families where prefix reuse matters most:

- repeated queries over the same long document
- multi-round conversation over a shared chat history [R8]

That is nearly identical to the reuse-heavy operating regime we care about for KV reuse, restore, and offload benchmarking [R8].

### 6. RULER and LongBench as long-context quality backstops

RULER provides synthetic long-context tasks with configurable sequence length and task complexity, and it is designed to test long-context capability beyond simple needle-in-a-haystack retrieval [R9].

LongBench v2 provides realistic long-context tasks spanning single-document QA, multi-document QA, long dialogue history understanding, code repo understanding, and long structured data understanding, with contexts ranging from 8k to 2M words [R10].

These are not serving benchmarks, but they are strong candidates for **quality-retention backstops** so that a tiered-KV benchmark does not optimize latency while ignoring long-context task utility [R9][R10].

### 7. longctx_bench as a compression / eviction tradeoff backstop

`longctx_bench` is directly relevant because it is a benchmark focused on long-context capable approaches including token eviction, KV cache quantization, and prompt compression, with explicit emphasis on tradeoffs rather than only speed [R11].

It is not a serving-capacity benchmark by itself, but it is useful as a **secondary validation layer** when we eventually compare offload, eviction, or codec-style approaches [R11].

## Benchmark aggregation stack

We should aggregate these benchmarks into a single stack rather than choosing only one.

### Layer 1: serving KPI and SLO layer

Use:

- **repo online harness** via `Blackwell/scripts/serve_and_bench.py` [L1]
- **GenAI-Perf metric framing** for TTFT, inter-token latency, request latency, throughput, and goodput under constraints [R4][R7]

Role:

- defines the main serving KPI
- keeps the benchmark focused on user-visible latency and goodput instead of raw tokens/sec only
- gives us the right language for p95/p99 guardrails [R4][R7]

### Layer 2: synthetic traffic and request-length layer

Use:

- **BurstGPT** for arrival patterns, session structure, and request / response token distributions [R3]
- **TensorRT-LLM `prepare_dataset.py` / `trtllm-bench`** for controlled synthetic calibration workloads [R1]

Role:

- BurstGPT supplies trace-aligned traffic structure and session metadata [R3]
- TensorRT-LLM synthetic datasets supply controlled calibration sweeps that are easy to reproduce [R1]

This means the actual benchmark should use:

- **trace-aligned scheduling from BurstGPT-style distributions**
- **controlled token buckets from TensorRT-LLM-style synthetic datasets**

### Layer 3: reuse and offload stress layer

Use:

- **repo repeated-prefix workloads** in `serve_and_bench.py` and `run_baseline.py` [L1][L2]
- **vLLM benchmark CLI prefix-repetition and prefix-caching patterns** [R2]
- **vLLM APC workload guidance** for long-document query reuse and multi-round conversation reuse [R8]

Role:

- directly stresses the KV reuse path
- gives us reusable prefixes with configurable repeat structure
- makes the benchmark relevant for restore-vs-recompute and host-offload behavior [R2][R8]

This is the most important layer for the Blackwell KV offload hypothesis.

### Layer 4: long-context quality-retention layer

Use:

- **LongBench / LongBench v2** [R10]
- **RULER** [R9]
- optionally **longctx_bench** when evaluating compression / eviction tradeoffs [R11]

Role:

- verifies that long-context ability is not collapsing while we optimize serving capacity
- provides task-facing sanity checks beyond latency-only measurements [R9][R10][R11]

These should be treated as **backstop evaluations**, not the main serving benchmark.

## What we should actually compose for the Blackwell KV offload benchmark

The practical answer is:

### Benchmark core

Use this as the benchmark core:

- **online concurrency sweep** from the repo harness [L1]
- **goodput / latency guardrails** from GenAI-Perf framing [R4]
- **repeated-prefix and synthetic multi-turn active-set workloads** from this repo and vLLM prefix-caching patterns [L1][L2][R2][R8]
- **restore-aware metrics** from the TensorRT-LLM disaggregated serving and KV connector model [R5][R6]

### Synthetic data source mix

Use this as the data source mix:

- **BurstGPT-derived scheduling distributions** for sessions, request timing, and request-length realism [R3]
- **ShareGPT / custom JSONL prompt pools** for chat-style prompts, as already supported by vLLM benchmarking [R2]
- **prefix-repetition synthetic data** for controlled reuse clusters [R2]
- **long-document QA prompts** for document-reuse slices [R2][R8]

### Backstop evaluation set

Use this as the secondary evaluation set:

- **RULER** for synthetic long-context stress [R9]
- **LongBench** for realistic long-context task quality [R10]
- **longctx_bench** for compression / eviction tradeoff checks when needed [R11]

## Recommended decision

For the current Blackwell track, the best benchmark aggregation is:

- **BurstGPT + repo repeated-prefix harness + vLLM prefix-repetition / APC patterns + GenAI-Perf metric framing** as the main benchmark stack [R2][R3][R4][R8][L1][L2]
- **LongBench / RULER** as quality-retention backstops [R9][R10]
- **longctx_bench** as a secondary comparison source when the work shifts from pure offload to broader compression / eviction tradeoff evaluation [R11]

That gives us:

- realistic or trace-aligned request timing [R3]
- controllable reuse structure [R2][R8]
- serving-facing latency and goodput KPIs [R4][R7]
- long-context quality guardrails [R9][R10]
- an optional compression-tradeoff benchmark when needed [R11]

## What is out of scope for this file

The following are explicitly **not required** for the benchmark v1 defined here:

- expert parallel routing skew
- MoE hotspot simulation
- expert-load CV / Gini metrics
- all-to-all expert dispatch analysis

If we want those later, they can remain in the broader research note.

## Synthetic data plan

We should build the synthetic data in **layers**.

### Layer A: prompt sources

Use one or more of the following:

- **ShareGPT-style conversations** for chat-like prompts, as already supported by vLLM benchmarking examples [R2]
- **custom JSONL prompt pools** with a `prompt` field, also directly supported by vLLM benchmarking [R2]
- **repo-generated repeated-prefix prompts** for controlled reuse-heavy experiments, matching the current repo harness direction [L1][L2]
- **long-document QA prompt sets** for long-context reuse scenarios, following the vLLM long-document QA benchmark pattern [R2]

### Layer B: prompt shaping

Convert prompt pools into benchmark prompts using:

- **fixed prompt buckets** at 10K / 20K / 30K / 40K / 50K tokens
- **shared-prefix ratios** for reuse-heavy cases, matching the repo's repeated-prefix harness direction [L1][L2]
- **prefix-repetition** style workloads when we want a controlled number of reusable prefixes, following vLLM's `prefix_repetition` benchmark dataset design [R2]
- **Zipf-like request length fitting** when we want a more trace-like prompt-length distribution, following BurstGPT's workload construction logic [R3]

### Layer C: conversation schedule synthesis

For the actual active-set benchmark, synthetic data should include conversation scheduling metadata rather than only prompt text.

Each record should contain at least:

- **`conversation_id`**
- **`turn_index`**
- **`workload_family`** = `single_turn_active_set` | `synthetic_multi_turn_active_set`
- **`prompt_bucket_tokens`**
- **`shared_prefix_ratio`**
- **`reuse_cluster_id`**
- **`think_time_ms`**
- **`target_output_tokens`**
- **`sampling_mode`**

This is the minimum needed to reproduce the active conversation pattern that the repo actually cares about [L1][L2][R3].

### Layer D: restore-aware metadata

Because this repo cares about tiered KV behavior, synthetic request records should also support restore-aware analysis fields such as:

- **`estimated_reusable_prefix_tokens`**
- **`estimated_restored_kv_bytes`**
- **`reuse_distance_bucket`**
- **`expected_restore_path`** = `gpu_resident` | `host_restore` | `recompute`

These fields are benchmark metadata. They do not have to be perfectly known before the run, but they should be recorded or estimated so the results can be grouped later [R6][R7].

## Actual benchmark matrix

For each prompt bucket:

- **10K**
- **20K**
- **30K**
- **40K**
- **50K**

run:

- **`single_turn_active_set`**
- **`synthetic_multi_turn_active_set`**

and sweep:

- **`active_conversations`** = `1, 2, 4, 8, 16, 32, ...`

### `single_turn_active_set`

Use this for:

- first-pass capacity sweeps
- repeated-prefix reuse measurement
- restore-line-rate instrumentation
- TTFT / TPOT guardrail finding [L1][R4]

### `synthetic_multi_turn_active_set`

Use this for:

- persistent conversation identity
- think-time gaps
- follow-up turns that partially reuse earlier context
- more realistic active working-set pressure [R3][L1]

## Prompt format and stop conditions

To keep results comparable across buckets:

- use one canonical chat template per model family
- keep output limit fixed per run configuration
- keep temperature and sampling settings fixed inside a sweep
- do not mix multiple stop policies inside one benchmark slice [R2][R4]

## Metrics

### Headline KPI

The main KPI should be:

- **`max_active_conversations_at_target`**

where the target is expressed as latency guardrails such as:

- **p95 TTFT <= target**
- **p95 TPOT <= target**
- **p99 TPOT <= guardrail**

This is directly aligned with the repo's existing `serve_and_bench.py` sweep logic, which already stops when p95 TPOT exceeds a configured threshold [L1].

### Required latency and throughput metrics

Always report:

- **`ttft_ms_p50`**
- **`ttft_ms_p95`**
- **`ttft_ms_p99`**
- **`tpot_ms_p50`**
- **`tpot_ms_p95`**
- **`tpot_ms_p99`**
- **`throughput_tokens_per_s`**
- **`request_throughput_per_s`**
- **`inter_token_latency_ms_p50`**
- **`inter_token_latency_ms_p95`**
- **`goodput_at_slo`**

These are consistent with vLLM serving benchmarks and NVIDIA GenAI-Perf's benchmark framing around TTFT, TPOT / ITL, and constraint-based serving evaluation [R2][R4].

### Required KV and memory metrics

Always report:

- **`peak_hbm_gb`**
- **`cache_hit_rate`**
- **`restored_kv_bytes`**
- **`recomputed_kv_bytes`**
- **`promotion_latency_ms_p50`**
- **`promotion_latency_ms_p95`**
- **`effective_line_rate_gbps`**
- **`prefill_visible_restore_ms`**
- **`decode_visible_restore_stall_ms`**

These are the metrics that connect the synthetic schedule to actual tiered-KV behavior [R6][R7][L1][L2].

### Stability metrics

Always report:

- **request failures**
- **token-latency jitter**
- **timeout / aborted request count**

BurstGPT explicitly argues that stability and reliability should be part of serving evaluation rather than only average latency and average throughput [R3].

## Synthetic data sources we should actually use

### Recommended source mix for v1

Use this mix in order:

#### 1. Controlled repeated-prefix synthetic prompts

Use the repo's existing repeated-prefix prompt strategy as the default calibration path because it is already wired into `serve_and_bench.py` and `run_baseline.py` [L1][L2].

#### 2. ShareGPT-derived prompt pool

Use a ShareGPT-style prompt pool for a second benchmark view because vLLM already documents ShareGPT-backed serving benchmarks and prefix-caching benchmarks [R2].

#### 3. Prefix-repetition synthetic dataset

Use a prefix-repetition dataset for highly controlled reuse clusters because vLLM already exposes it as a benchmark dataset family with explicit prefix and suffix lengths [R2].

#### 4. Long-document QA style prompt set

Use long-document QA prompts when we want long-context reuse rather than generic short chat behavior, following the documented vLLM benchmark pattern [R2].

### Recommended source mix for v2

If we need more workload realism later, add BurstGPT-style length fitting:

- sample request lengths from a target distribution
- truncate prompt pools to match that distribution
- preserve the benchmark's synthetic honesty by labeling it as trace-aligned, not real-trace replay [R3]

## Suggested result schema additions

The current repo result schema already includes core serving and KV metrics such as `ttft_ms_*`, `tpot_ms_*`, `cache_hit_rate`, and `promotion_latency_ms_*` [L3].

Add fields such as:

- **`active_conversations`**
- **`workload_family`**
- **`prompt_bucket_tokens`**
- **`turns_per_conversation`**
- **`think_time_ms_mean`**
- **`reuse_cluster_count`**
- **`reuse_distance_bucket_counts`**
- **`restored_kv_bytes`**
- **`recomputed_kv_bytes`**
- **`effective_line_rate_gbps`**
- **`prefill_visible_restore_ms`**
- **`decode_visible_restore_stall_ms`**
- **`goodput_at_slo`**
- **`request_failures`**
- **`token_latency_jitter_ms`**

## Recommended implementation order

### 1. Keep the existing harnesses

Do not replace the repo benchmark scripts.

Use:

- `Blackwell/scripts/run_baseline.py` for controlled offline-style prompt sweeps [L2]
- `Blackwell/scripts/serve_and_bench.py` for online concurrency sweeps and target-based stopping [L1]
- `Blackwell/scripts/run_tiered_experiment.py` for reuse / tiering comparisons [L4]

### 2. Add a synthetic trace generator

Create a synthetic data generator that emits JSONL records with the conversation scheduling metadata described above.

### 3. Add prompt-source adapters

Add adapters for:

- repeated-prefix synthetic prompts
- ShareGPT/custom JSONL prompt pools
- prefix-repetition prompt generation
- long-document QA prompt generation [R2]

### 4. Add restore-aware metrics

Extend the result schema and per-request tracing so restored bytes, visible restore cost, and effective line rate are captured in the same output artifact [R6][R7][L3].

## One-sentence benchmark description

> We benchmark Blackwell KV-serving capacity under interactive active-conversation pressure using controlled repeated-prefix and trace-aligned synthetic prompt sources, and we measure latency, throughput, restore cost, and goodput without requiring EP / MoE simulation or private customer traces.

## Sources

### External sources

- **[R1] TensorRT-LLM Benchmarking Default Performance**
  - URL: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html
  - Retrieved: 2026-03-15
- **[R2] vLLM Benchmark CLI**
  - URL: https://docs.vllm.ai/en/latest/benchmarking/cli/
  - Retrieved: 2026-03-15
- **[R3] BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems**
  - URL: https://arxiv.org/html/2401.17644v3
  - Retrieved: 2026-03-15
- **[R4] NVIDIA GenAI-Perf / Goodput docs**
  - URL: https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2591/user-guide/docs/perf_analyzer/genai-perf/docs/goodput.html
  - Retrieved: 2026-03-15
- **[R5] TensorRT-LLM Disaggregated Serving docs**
  - URL: https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html
  - Retrieved: 2026-03-15
- **[R6] TensorRT-LLM KV Cache Connector docs**
  - URL: https://nvidia.github.io/TensorRT-LLM/features/kv-cache-connector.html
  - Retrieved: 2026-03-15
- **[R7] NVIDIA GenAI Performance Analyzer**
  - URL: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai_perf.html
  - Retrieved: 2026-03-15
- **[R8] vLLM Automatic Prefix Caching**
  - URL: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
  - Retrieved: 2026-03-15
- **[R9] NVIDIA RULER**
  - URL: https://github.com/NVIDIA/RULER
  - Retrieved: 2026-03-15
- **[R10] LongBench / LongBench v2**
  - URL: https://github.com/THUDM/LongBench
  - Retrieved: 2026-03-15
- **[R11] longctx_bench / KV Cache Compression, But What Must We Give in Return?**
  - URL: https://github.com/henryzhongsc/longctx_bench
  - Retrieved: 2026-03-15

### Repo-local sources

- **[L1] `Blackwell/scripts/serve_and_bench.py`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/scripts/serve_and_bench.py
  - Retrieved: 2026-03-15
- **[L2] `Blackwell/scripts/run_baseline.py`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/scripts/run_baseline.py
  - Retrieved: 2026-03-15
- **[L3] `Blackwell/scripts/kv_bench_utils.py`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/scripts/kv_bench_utils.py
  - Retrieved: 2026-03-15
- **[L4] `Blackwell/scripts/run_tiered_experiment.py`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/scripts/run_tiered_experiment.py
  - Retrieved: 2026-03-15
- **[L5] `Blackwell/docs/research/synthetic_kv_working_set_benchmark_mtp_ep.md`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/docs/research/synthetic_kv_working_set_benchmark_mtp_ep.md
  - Retrieved: 2026-03-15
