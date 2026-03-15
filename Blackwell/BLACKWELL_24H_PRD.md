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

## Workstreams

### WS0: Support Gate and Environment (20 min)

Run `scripts/env_probe.sh` to produce `results/env_probe.json`.

Must capture: GPU model, driver, CUDA, Python, vLLM version, TRT-LLM version, LMCache version, filesystem paths, NVFP4/FP8 KV support status.

Decision: full ladder (NVFP4 supported) or reduced ladder (FP8 only) or stop (neither).

### WS1: Baseline Harness (45 min)

Harden `scripts/run_baseline.py` to accept `--model`, `--context-length`, `--requests`, `--concurrency`, `--kv-mode`, `--workload-type`, `--engine`, `--output`.

Must emit stable JSON with: TTFT p50/p95, TPOT p50/p95/p99, throughput, peak HBM, GPU power, cache hit rate, quality placeholder.

Verify with: `python scripts/run_baseline.py --kv-mode bf16 --context-length 8192 --requests 2`

### WS2: Aligned Baselines (60 min)

Model: `Qwen/Qwen3-30B-A3B` (fallback: `Qwen/Qwen3-32B`)

Workload: `repeated_prefix`

| Variant | Context | Runs |
|---------|---------|------|
| bf16 | 8192 | 1 |
| fp8 | 8192 | 1 |
| fp8 + lmcache | 8192 | 1 |
| bf16 | 32768 | 1 |
| fp8 | 32768 | 1 |
| fp8 + lmcache | 32768 | 1 |

Output: `results/baseline_{variant}_{context}.json`

### WS3: First Tiered Experiment — vLLM + LMCache (75 min)

Run vLLM with LMCache cold-tier reuse enabled:
- Hot tier: vLLM FP8 KV cache on GPU
- Cold tier: LMCache-managed host RAM
- Promotion policy: demand (default)
- Protection: 4 sink tokens, 128 recent tokens

Must record: offload latency, restore/promotion latency, cold store size, cache hit rate, TTFT cold vs warm.

This step validates the serving economics hypothesis. The question is whether KV reuse via LMCache improves TTFT and concurrency, not whether offloading exists.

### WS4: Policy Ablation (45 min)

Compare demand promotion vs eager promotion using the same tiered experiment script.

Output: two tiered result JSONs, one comparison via `scripts/compare_results.py`.

### WS5: One-Node Scaling (30 min, only after WS2 succeeds)

Submit `scripts/baseline_single_gpu.sbatch` for bf16/fp8/nvfp4 via env vars.

If single-GPU works, submit `scripts/baseline_one_node.sbatch` for the same matrix.

Do not change multiple variables at once.

### WS6: Decision Memo (30 min)

Produce:
- `results/comparison.md` — benchmark table from `scripts/compare_results.py`
- One-paragraph bottleneck analysis
- One-paragraph continue/pivot/kill recommendation
- Exact rerun commands

## Evaluation Matrix

Minimum viable matrix:

| Variant | Model | Context | Workload | Notes |
|---------|-------|---------|----------|-------|
| bf16_default | Qwen/Qwen3-30B-A3B | 8192 | repeated_prefix | baseline ceiling |
| fp8_kv | Qwen/Qwen3-30B-A3B | 8192 | repeated_prefix | stable hot-tier path |
| fp8_kv + lmcache | Qwen/Qwen3-30B-A3B | 8192 | repeated_prefix | cold-tier reuse |
| tiered_demand | Qwen/Qwen3-30B-A3B | 8192 | repeated_prefix | demand promotion policy |
| tiered_eager | Qwen/Qwen3-30B-A3B | 8192 | repeated_prefix | eager promotion policy |
| bf16_default | Qwen/Qwen3-30B-A3B | 32768 | repeated_prefix | long-context baseline |
| fp8_kv | Qwen/Qwen3-30B-A3B | 32768 | repeated_prefix | long-context fp8 |
| fp8_kv + lmcache | Qwen/Qwen3-30B-A3B | 32768 | repeated_prefix | long-context cold-tier reuse |

Concurrency / request-rate sweep (stop when p95 becomes unacceptable):
- `request_rate`: 1, 2, 5, 10, 20
- `max_concurrency`: 1, 2, 4, 8, 16, 32

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
