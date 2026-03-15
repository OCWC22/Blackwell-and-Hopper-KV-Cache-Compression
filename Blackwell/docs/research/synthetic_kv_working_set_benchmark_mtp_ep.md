# Synthetic KV Working-Set Benchmark with MTP Acceptance and EP Activation Imbalance

## Purpose

This note defines a structured benchmark methodology for the current research problem:

- how to benchmark **KV cache working-set size** under active interactive usage
- how to account for **MTP acceptance behavior** when speculative decoding changes decode efficiency
- how to account for **EP / MoE expert-activation imbalance** when routed-expert skew changes serving cost
- how to build a **synthetic workload** that is defensible when real customer traces are unavailable

This is a benchmark-design document, not a claim that the current repo already measures every field below.

## Executive Summary

The benchmark should be framed as an **interactive serving benchmark** whose primary independent variable is:

- **`active_conversations`** = the number of simultaneously live conversations whose KV state must remain reusable or restorable within the target latency envelope

The benchmark should not be framed primarily as a generic long-context benchmark. Prompt length matters, but it should be treated as a **workload slice** rather than the primary X-axis.

To make the synthetic benchmark more realistic, we should explicitly model three serving-relevant distributions:

- **KV working-set behavior**
- **MTP acceptance behavior**
- **EP / expert-activation imbalance**

This decomposition is motivated by public sources that document:

- DeepSeek-V3 as an MoE model with **671B total parameters** and **37B activated parameters per token**, plus **multi-token prediction** and **auxiliary-loss-free load balancing** [R1]
- vLLM MTP as a speculative-decoding path whose utility depends on speculative-token verification behavior [R2]
- SGLang MTP performance in terms of **acceptance length**, with a published case study showing **82.0 tokens/s/rank** and **average acceptance length 2.44** for a 4-token MTP window versus **60.4 tokens/s/rank** baseline in their DeepSeek-V3 setup [R3]
- SGLang expert parallelism as a serving path where tokens are dynamically routed across experts and devices, making expert-load skew a systems issue rather than just a model-training issue [R4]
- DeepSeek's load-balancing work as evidence that expert-load imbalance is important enough to explicitly monitor and counteract [R1][R5]

## Why This Benchmark Exists

Existing long-context benchmarks do not cleanly answer the operational question we care about:

- **How many active conversations can the system sustain, at a fixed first-turn prompt size, while preserving interactive latency and useful KV reuse?**

For the Blackwell track, this stays aligned with the repo's primary objective:

- more concurrent sessions at fixed latency target
- lower peak HBM
- better TTFT on reuse-heavy traffic
- longer effective context

Those goals are already the Blackwell track's serving-facing success criteria, and the existing online benchmark harness is already oriented around concurrency sweeps and repeated-prefix traffic rather than pure offline context-length measurement [L1][L2][L3].

## Scope and Runtime Honesty

This document does **not** change the Blackwell track routing rule:

- **Primary runtime path:** TensorRT-LLM on Blackwell / B200, with NVFP4 as the primary hot-KV thesis when support-gated and available [L2][L3]
- **Follow-up compatibility path:** vLLM with FP8 KV plus optional LMCache [L2][L3]

MTP and EP are included here because they help define the **shape of the workload** and the **serving characteristics we should trace**, not because they replace the primary runtime path.

## Core Benchmark Framing

### Primary X-Axis

Use:

- **`active_conversations`**

Interpretation:

- each conversation has a persistent identity
- each conversation contributes live KV state to the working set
- each conversation may send additional turns after a think-time gap
- the system is evaluated on whether those active conversations still fit within the latency target

### Prompt-Length Slices

Use separate workload slices for first-turn prompt lengths:

- **10K tokens**
- **20K tokens**
- **30K tokens**
- **40K tokens**
- **50K tokens**

These are workload slices, not the main X-axis.

### Workload Families

Use two workload families with the same prompt format and stop conditions:

- **`single_turn_active_set`**
- **`synthetic_multi_turn_active_set`**

This is consistent with the repo's current distinction between repeated-prefix and reuse-focused workloads and with the desire to evaluate both single-turn and synthetic multi-turn behavior [L1][L4][L5].

## The Three Distributions We Need To Model

### 1. KV Working-Set Behavior

Trace and model:

- **`active_conversations`**
- **`conversation_turn_depth`**
- **`reuse_distance`**
- **`resident_kv_bytes_gpu`**
- **`restored_kv_bytes`**
- **`cache_hit_rate`**
- **`promotion_latency_ms`**

Why this matters:

- the repo's primary Blackwell scenarios are about **more sessions**, **longer context**, or both [L2][L3]
- the current online harness already measures concurrency and latency while the shared JSON schema already includes `cache_hit_rate` and `promotion_latency_ms_*` fields, so this is an extension of the current benchmark direction rather than a new benchmark category [L1][L4]

### 2. MTP Acceptance Behavior

Trace and model:

- **`accepted_tokens_per_verify_step`**
- **`acceptance_length_mean`**
- **`acceptance_length_p50`**
- **`acceptance_length_p95`**
- **`acceptance_rate`** = accepted draft tokens / proposed draft tokens
- conditioning variables:
  - prompt length
  - turn index
  - output position
  - sampling mode
  - active batch size
  - working-set pressure

Why this matters:

- vLLM's MTP docs describe MTP as a speculative-decoding path with configurable speculative depth via `num_speculative_tokens` [R2]
- SGLang's MTP documentation makes **acceptance length** the key serving metric, because the speedup depends on how many speculative tokens are accepted on each verification step [R3]
- public published results show that acceptance behavior is not a constant; it changes with workload and system configuration [R3]

### 3. EP / Expert-Activation Imbalance

If the target model is MoE, trace and model:

- **`tokens_per_expert`** over a decode window
- **`expert_load_cv`** = std / mean across experts
- **`expert_load_gini`** or another skew score
- **`hot_expert_share`** = share handled by the hottest experts
- **`a2a_bytes`**
- **`a2a_time_ms`**
- **`dispatch_stall_ms`**

Why this matters:

- DeepSeek-V3 is an MoE model with routed experts and explicit load-balancing machinery [R1][R5]
- SGLang's EP documentation states that expert parallelism requires dynamic token routing across experts and devices, which makes routing skew and communication imbalance part of the serving problem [R4]

## Synthetic Workload Design Principle

We should **not** try to synthesize "customer-like text" directly.

We should instead synthesize the **state distributions that dominate serving cost**:

- conversation activity
- prefix reuse
- KV reuse distance
- MTP acceptance regime
- EP / expert-load skew regime

That is the cleanest defensible surrogate when real customer traces are unavailable.

## Synthetic Workload Schema

Each synthetic conversation/request record should include the following fields.

### Conversation Fields

- **`conversation_id`**
- **`conversation_mode`** = `single_turn` | `synthetic_multi_turn`
- **`turn_index`**
- **`think_time_ms`**
- **`active_window_id`**

### Prompt and Reuse Fields

- **`first_turn_prompt_tokens`**
- **`followup_prompt_tokens`**
- **`reuse_cluster_id`**
- **`shared_prefix_ratio`**
- **`reuse_distance_bucket`** = `short` | `medium` | `long`

### Decode-Behavior Fields

- **`expected_mtp_regime`** = `high_accept` | `medium_accept` | `low_accept`
- **`target_output_len_bucket`**
- **`sampling_mode`**

### MoE / EP Fields

- **`expected_ep_skew_regime`** = `balanced` | `moderate_hotspot` | `severe_hotspot`
- **`moe_model_family`**
- **`expert_parallel_enabled`**

## Operational Interpretation of the Synthetic Regimes

### MTP Acceptance Regimes

Use three synthetic decode regimes.

#### `high_accept`

Characteristics:

- more templated or formulaic continuations
- follow-up questions that stay close to a shared scaffold
- lower branching entropy in the next-token path

Expected effect:

- longer average acceptance length
- higher decode efficiency from speculative verification

#### `medium_accept`

Characteristics:

- normal general chat or QA
- mixed predictability
- moderate branching

Expected effect:

- moderate acceptance length
- moderate speculative gain

#### `low_accept`

Characteristics:

- branchy reasoning
- tool-use-like or code-edit-like continuations
- more unpredictable next-token paths

Expected effect:

- shorter acceptance length
- weaker speculative gain

These are **benchmark regimes**, not claims that any public source gives universal acceptance-rate constants for all deployments. Public sources show that acceptance behavior is workload-sensitive and should be treated as a measured distribution, not a fixed benchmark constant [R2][R3].

### EP Skew Regimes

Use three synthetic expert-routing regimes.

#### `balanced`

Characteristics:

- expert traffic is spread relatively evenly
- no severe all-to-all hotspot

#### `moderate_hotspot`

Characteristics:

- some experts dominate
- communication and dispatch overhead become more uneven

#### `severe_hotspot`

Characteristics:

- a small set of experts absorb a disproportionate share of tokens
- all-to-all and expert dispatch costs dominate more strongly

These are again synthetic serving regimes, not a claim that public papers disclose customer expert-routing histograms.

## Benchmark Matrix

For each first-turn prompt bucket:

- **10K**
- **20K**
- **30K**
- **40K**
- **50K**

run both:

- **`single_turn_active_set`**
- **`synthetic_multi_turn_active_set`**

and sweep:

- **`active_conversations`** = `1, 2, 4, 8, 16, 32, ...`

For each run, stratify or annotate by:

- **MTP acceptance regime**
- **EP skew regime**
- **reuse distance regime**

## Primary Metrics

### Headline KPI

- **`max_active_conversations_at_target`**

Recommended guardrail targets should be expressed as latency constraints such as:

- p95 TTFT within target
- p95 TPOT within target
- p99 TPOT within guardrail

This matches the repo's Scenario 2 / Scenario 3 framing and the existing online harness behavior that sweeps concurrency and stops when p95 TPOT exceeds a limit [L1][L3].

### Supporting Metrics

Always report:

- **`ttft_ms_p50`**
- **`ttft_ms_p95`**
- **`tpot_ms_p50`**
- **`tpot_ms_p95`**
- **`tpot_ms_p99`**
- **`throughput_tokens_per_s`**
- **`peak_hbm_gb`**
- **`cache_hit_rate`**
- **`promotion_latency_ms_p50`**
- **`promotion_latency_ms_p95`**
- **`acceptance_length_mean/p50/p95`**
- **`accepted_tokens_per_verify_step`**
- **`expert_load_cv`**
- **`expert_load_gini`**
- **`hot_expert_share`**
- **`a2a_time_ms`** where available

## What The Current Repo Already Supports

The current Blackwell harness already gives us a useful starting point:

- `Blackwell/scripts/serve_and_bench.py` already performs **concurrency sweeps**, uses **repeated-prefix prompts**, and stops the sweep when **p95 TPOT** exceeds a threshold [L1]
- `Blackwell/scripts/kv_bench_utils.py` already defines a canonical result schema with fields for `cache_hit_rate` and `promotion_latency_ms_p50/p95` [L4]
- `Blackwell/scripts/run_baseline.py` and `Blackwell/scripts/run_tiered_experiment.py` already define repeated-prefix / reuse-oriented workloads that can be extended into conversation-aware synthetic traces [L5][L6]

## What The Current Repo Does Not Yet Support

The current repo does **not** yet fully expose:

- persistent conversation scheduling in the online benchmark harness
- MTP acceptance-length tracing
- expert-load-skew tracing for EP / MoE routing
- explicit synthetic workload metadata fields in result JSON

So this document should be read as a **benchmark methodology target**, not as a statement that the implementation is already complete.

## Recommended Next Implementation Steps

### 1. Extend the result schema

Add fields such as:

- `active_conversations`
- `conversation_mode`
- `turns_per_conversation`
- `first_turn_prompt_tokens`
- `followup_prompt_tokens`
- `think_time_ms`
- `working_set_kv_bytes_est`
- `restored_kv_bytes`
- `recomputed_kv_bytes`
- `acceptance_length_mean/p50/p95`
- `expert_load_cv`
- `expert_load_gini`
- `hot_expert_share`

### 2. Add conversation-aware scheduling

Extend the online benchmark harness so it can schedule **persistent conversations**, not just concurrency-limited independent requests.

### 3. Add synthetic regime controls

Add explicit knobs for:

- MTP acceptance regime
- EP skew regime
- reuse distance regime
- think-time distribution

### 4. Keep runtime honesty

Do not turn this into a claim that the Blackwell primary result depends on replacing TRT-LLM with a different serving engine. For the Blackwell track, the point is to benchmark serving capacity honestly while preserving the primary TensorRT-LLM path [L2][L3].

## Assumptions and Limitations

### What this benchmark can claim

- it is a structured synthetic benchmark for **active KV working-set pressure**
- it explicitly accounts for speculative-decode efficiency via **MTP acceptance behavior**
- it explicitly accounts for MoE routed-expert pressure via **EP activation imbalance**
- it is more defensible than a benchmark that varies only prompt length and concurrency

### What this benchmark cannot claim

- it is **not** proof that the synthetic workload matches undisclosed customer traces exactly
- it is **not** proof that one public acceptance-rate result transfers directly to another model, runtime, or deployment
- it is **not** proof that public MoE routing behavior distributions match a private production system

### The honesty rule

Because customer traces are unavailable, the benchmark should be described as a **behavioral surrogate** that attempts to match the distributions most likely to dominate serving cost:

- active working-set size
- reuse distance
- MTP acceptance length
- EP routing skew

## Suggested One-Sentence Benchmark Description

> We benchmark active KV working-set capacity under interactive serving, parameterized by conversation activity, reuse distance, speculative acceptance behavior, and expert-routing skew, so that we can approximate customer-relevant serving pressure without claiming access to private customer traces.

## Sources

### External sources

- **[R1] DeepSeek-V3 Technical Report**
  - URL: https://arxiv.org/abs/2412.19437
  - Retrieved: 2026-03-15
- **[R2] vLLM MTP (Multi-Token Prediction) docs**
  - URL: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
  - Retrieved: 2026-03-15
- **[R3] SGLang: Accelerating SGLang with Multiple Token Prediction**
  - URL: https://lmsys.org/blog/2025-07-17-mtp/
  - Retrieved: 2026-03-15
- **[R4] SGLang Expert Parallelism docs**
  - URL: https://docs.sglang.io/advanced_features/expert_parallelism.html
  - Retrieved: 2026-03-15
- **[R5] Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts**
  - URL: https://arxiv.org/html/2408.15664v1
  - Retrieved: 2026-03-15

### Repo-local sources

- **[L1] `Blackwell/scripts/serve_and_bench.py`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/scripts/serve_and_bench.py
  - Retrieved: 2026-03-15
- **[L2] `Blackwell/README.md`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/README.md
  - Retrieved: 2026-03-15
- **[L3] `Blackwell/BLACKWELL_24H_PRD.md`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/BLACKWELL_24H_PRD.md
  - Retrieved: 2026-03-15
- **[L4] `Blackwell/scripts/kv_bench_utils.py`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/scripts/kv_bench_utils.py
  - Retrieved: 2026-03-15
- **[L5] `Blackwell/scripts/run_baseline.py`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/scripts/run_baseline.py
  - Retrieved: 2026-03-15
- **[L6] `Blackwell/scripts/run_tiered_experiment.py`**
  - URL: file:///Users/chen/Documents/GitHub/Hopper-KV-Cache-Compression/Blackwell/scripts/run_tiered_experiment.py
  - Retrieved: 2026-03-15
