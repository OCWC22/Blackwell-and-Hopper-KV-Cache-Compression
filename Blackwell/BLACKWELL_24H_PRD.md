# Blackwell Hackathon PRD

## Objective

Show that on B200 / Blackwell, a vLLM + LMCache compatible hot/cold KV lifecycle improves reuse-heavy long-context serving by achieving at least one of:
- >=20% lower peak HBM
- materially better TTFT on repeated-prefix traffic
- >=25% more concurrent sessions at fixed p95 target
- materially longer effective context

while keeping:
- p95 TPOT regression <= 10%
- p99 TPOT regression <= 15%
- quality delta <= 1%

This repo is testing a KV-lifecycle controller around existing serving stacks (vLLM, LMCache), not building a new engine.

## Four Benchmark Scenarios

| Scenario | Question | KV Pressure | Primary Metrics |
|----------|----------|-------------|-----------------|
| **1** — Longer context, one GPU | How far can one GPU go? | KV bytes/session | peak HBM, TTFT, p95/p99 TPOT, max effective context |
| **2** — More sessions, one GPU | How many concurrent sessions? | Live KV replicas | max sessions at p95, throughput, TTFT under reuse, cache hit rate |
| **3** — Both, one GPU **(PRIMARY)** | Many users + long prompts? | Both context × sessions | max sessions at large context, peak HBM, p95/p99 TPOT, TTFT on reused prefixes, quality delta |
| **4** — Both, one node | Same idea at node level? | Node aggregate | aggregate sessions/node, throughput, HBM, power efficiency |

**Scenario 3 is the main goal. Scenario 4 is the follow-up. Scenarios 1 and 2 are explanatory baselines.**

## Models

| Role | Model |
|------|-------|
| Primary | `Qwen/Qwen3-30B-A3B` (or `Qwen/Qwen3-32B`) |
| Smoke test | `Qwen/Qwen3-8B-Instruct` |

## Workloads

| Role | Workload | Definition |
|------|----------|------------|
| Primary | `repeated_prefix` | Same long system prompt / agent scaffold / document prefix, many requests with short differing suffixes |
| Secondary | `multi_turn_reuse` | Multi-turn conversation reusing prior context |
| Secondary | `repeated_long_doc` | Same long document, different questions |

Primary contexts: 8192, 32768

## Primary KPI

Achieve at least one of:

| Target | Threshold |
|--------|-----------|
| More concurrent sessions | ≥25% higher max sessions at fixed latency target |
| Lower peak HBM | ≥20% reduction vs best non-tiered baseline |
| Better TTFT | Materially better on repeated-prefix traffic |
| Longer effective context | Materially longer at same GPU memory budget |

While keeping:

| Guard rail | Threshold |
|------------|-----------|
| p95 TPOT regression | ≤10% vs best non-tiered baseline |
| p99 TPOT regression | ≤15% vs best non-tiered baseline |
| Quality delta | ≤1% vs bf16 baseline on chosen eval |

These are hackathon-grade thresholds, not production guarantees.

## Non-Goals

- Do not build a new inference engine
- Do not replace vLLM or LMCache
- Do not treat KVTC as a hot-path representation — it is a cold-tier codec candidate
- Do not run multi-node jobs before the single-GPU path works
- Do not claim NVFP4 hot-KV support in vLLM unless explicitly verified at runtime
- Do not block on NVFP4 or KVTC integration — the core experiment is vLLM + LMCache reuse

## Deliverables

1. `results/env_probe.json` — support gate result
2. Aligned baseline results (bf16, fp8, nvfp4 if supported)
3. First tiered KV result (hot GPU + host RAM cold tier)
4. One promotion-policy ablation (demand vs eager)
5. One benchmark comparison table (via `scripts/compare_results.py`)
6. One bottleneck summary
7. Exact rerun commands for every result

## Acceptance Criteria

- Every result JSON matches the canonical schema from `scripts/run_baseline.py`
- `scripts/compare_results.py` can read every result and produce the comparison table
- Results include quality and latency together, not memory alone
- Every run captures model, hardware, context, batch size, KV mode, and workload type
- The benchmark ladder is honest (support-gate results recorded, no overclaims)
- The repo can be handed to another engineer without oral explanation

## 5-Hour Execution Plan

### Hour 0:00–0:30 — Environment Probe + Support Gate

Run `scripts/env_probe.sh` to produce `results/env_probe.json`.

Must capture: GPU model, driver, CUDA, Python, vLLM version, TRT-LLM version, LMCache version, filesystem paths, NVFP4/FP8 KV support status.

Decision: full ladder (NVFP4 supported) or reduced ladder (FP8 only) or stop (neither).

Verify baseline harness: `python scripts/run_baseline.py --kv-mode bf16 --context-length 8192 --requests 2`

### Hour 0:30–1:30 — Aligned Single-GPU Baselines (Scenarios 1 & 2)

Model: `Qwen/Qwen3-30B-A3B` (smoke test: `Qwen/Qwen3-8B-Instruct`)

**Scenario 1 runs (context sweep, concurrency=1):**

| Variant | Context | scenario_id |
|---------|---------|-------------|
| bf16 | 8192 | scenario_1_longer_context_gpu |
| fp8 | 8192 | scenario_1_longer_context_gpu |
| bf16 | 32768 | scenario_1_longer_context_gpu |
| fp8 | 32768 | scenario_1_longer_context_gpu |

**Scenario 2 runs (concurrency sweep, context=8k):**

| Variant | Concurrency | scenario_id |
|---------|-------------|-------------|
| vLLM baseline | 1,2,4,8,16,32 | scenario_2_more_sessions_gpu |
| vLLM + LMCache | 1,2,4,8,16,32 | scenario_2_more_sessions_gpu |

Output: `results/baseline_{variant}_{context}_*.json`

### Hour 1:30–3:00 — First vLLM + LMCache Result (Scenario 3 — PRIMARY)

Run vLLM with real LMCache cold-tier reuse via `LMCacheConnectorV1`:
- Hot tier: vLLM FP8 KV cache on GPU (or NVFP4 if support-gated)
- Cold tier: LMCache-managed host RAM via CPU offloading
- Connector: `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
- Config: `LMCACHE_CONFIG_FILE=configs/lmcache_config.yaml`
- Promotion policy: demand (default)
- Protection: 4 sink tokens, 128 recent tokens

**Scenario 3 benchmark matrix:**

| Variant | Context | Concurrency | Workload | scenario_id |
|---------|---------|-------------|----------|-------------|
| vLLM baseline | 8k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |
| vLLM + LMCache | 8k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |
| vLLM baseline | 32k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |
| vLLM + LMCache | 32k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |
| optional LMCache + KVTC | 8k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |

Commands:
```bash
# Tiered experiment with LMCache offloading
python scripts/run_tiered_experiment.py --use-lmcache --kv-mode fp8 --requests 10

# Concurrent user sweep (primary KPI)
python scripts/serve_and_bench.py --kv-mode fp8 --use-lmcache \
    --sweep-concurrency 1,2,4,8,16,32 --p95-tpot-limit-ms 100
```

Must record: offload latency, restore/promotion latency, cold store size, cache hit rate, TTFT cold vs warm, max concurrent sessions at p95 target, tokens/joule.

This step validates the serving economics hypothesis. The question is whether KV reuse via LMCache improves TTFT and concurrency, not whether offloading exists.

### Hour 3:00–3:45 — Policy Ablation

Compare demand promotion vs eager promotion using the same tiered experiment script.

Output: two tiered result JSONs, one comparison via `scripts/compare_results.py`.

### Hour 3:45–4:30 — One-Node Run (Scenario 4, only after Scenario 3 success)

**Scenario 4 benchmark matrix (same as Scenario 3, one full node):**

| Variant | Context | Concurrency | scenario_id |
|---------|---------|-------------|-------------|
| vLLM baseline | 8k, 32k | 4,8,16 | scenario_4_longer_context_more_sessions_node |
| vLLM + LMCache | 8k, 32k | 4,8,16 | scenario_4_longer_context_more_sessions_node |

Submit `scripts/baseline_one_node.sbatch` only after single-GPU success. Do not change multiple variables at once.

### Hour 4:30–5:00 — Decision Memo + Artifacts

Produce:
- `results/comparison.md` — benchmark table from `scripts/compare_results.py`
- `results/benchmark_summary.md` — one-page summary of findings
- `results/bottleneck_summary.md` — key bottlenecks and recommendations
- One-paragraph continue/pivot/kill recommendation
- Exact rerun commands for every result

## Full Benchmark Matrix

### Scenario 1 — Longer context on one GPU
- Context sweep: 8k, 32k, 64k
- Concurrency: 1 (fixed)
- Compare: vLLM baseline, vLLM FP8 KV, vLLM + LMCache, optional + KVTC

### Scenario 2 — More sessions on one GPU
- Context: 8k (fixed)
- Concurrency sweep: 1, 2, 4, 8, 16, 32
- Compare: vLLM baseline, vLLM + LMCache

### Scenario 3 — Longer context + more sessions on one GPU (PRIMARY)
- Contexts: 8k, 32k
- Concurrency: 4, 8, 16
- Workload: repeated_prefix
- Compare: vLLM baseline, vLLM + LMCache, optional LMCache + KVTC

### Scenario 4 — Longer context + more sessions on one node
- Same as Scenario 3
- One full node (8x GPUs)
- Only run after Scenario 3 is stable

## Kill Criteria

Stop or pivot if:

1. Hot-tier format support is not available and blocks the entire stack
2. Restore/promotion cost dominates so badly that reuse no longer helps
3. p95 latency blows up without enough capacity gain to compensate
4. Quality loss is obvious and cannot be fixed with simple protection
5. Implementation becomes engine-rewrite territory
6. One-node execution is unstable and obscures the result

If blocked: fall back to nearest stable baseline, preserve the serving-capacity experiment, keep moving.

## Scope Cutting Order

Cut in this order if time is short:

1. Multi-node — cut first
2. Full KVTC integration — cut second
3. Giant models — cut third
4. **Never cut:** aligned baseline table, restore timing, one serving-facing KPI

## References

- `TIERED_KV_ARCHITECTURE.md`: architectural specification (facts vs hypotheses vs measurements)
- `blackwell_kv_hackathon_context.md`: execution brief, validation ladder, source notes
- NVIDIA NVFP4: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA NVFP4 KV cache: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- NVIDIA TRT-LLM KV reuse: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- vLLM quantized KV cache: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- LMCache docs: <https://docs.lmcache.ai/>
