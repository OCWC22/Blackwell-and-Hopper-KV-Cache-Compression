# Blackwell Hackathon PRD

## Objective

Show that on B200 / Blackwell, **TensorRT-LLM with NVFP4 KV cache and host memory offloading** improves reuse-heavy long-context serving by achieving at least one of:
- >=20% lower peak HBM
- materially better TTFT on repeated-prefix traffic
- >=25% more concurrent sessions at fixed p95 target
- materially longer effective context

while keeping:
- p95 TPOT regression <= 10%
- p99 TPOT regression <= 15%
- quality delta <= 1%

This repo is testing TRT-LLM NVFP4 hot-tier efficiency plus KV offload/reuse lifecycle on Blackwell, not building a new engine. vLLM + LMCache is the follow-up compatibility/productization path.

## Four Benchmark Scenarios

| Scenario | Question | KV Pressure | Primary Metrics |
|----------|----------|-------------|-----------------|
| **1** — Longer context, one GPU | How far can one GPU go? | KV bytes/session | peak HBM, TTFT, p95/p99 TPOT, max effective context |
| **2** — More sessions, one GPU | How many concurrent sessions? | Live KV replicas | max sessions at p95, throughput, TTFT under reuse, cache hit rate |
| **3** — Both, one GPU **(PRIMARY)** | Many users + long prompts? | Both context x sessions | max sessions at large context, peak HBM, p95/p99 TPOT, TTFT on reused prefixes, quality delta |
| **4** — Both, one node | Same idea at node level? | Node aggregate | aggregate sessions/node, throughput, HBM, power efficiency |

**Scenario 3 is the main goal. Scenario 4 is the follow-up. Scenarios 1 and 2 are explanatory baselines.**

## Models

| Role | Model | Notes |
|------|-------|-------|
| Primary | `Qwen/Qwen3-30B-A3B` (or `Qwen/Qwen3-32B`) | Main benchmark model |
| Smoke test | `Qwen/Qwen3-8B-Instruct` | Harness validation |
| Stretch | `moonshotai/Kimi-K2.5` | 8x H200 verified [R10], node-level Scenario 4 only, NOT for single-GPU proof |

Kimi K2.5 is a stretch goal. Cut it first if time is short. Do not attempt on single-GPU Blackwell.

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
| More concurrent sessions | >=25% higher max sessions at fixed latency target |
| Lower peak HBM | >=20% reduction vs best non-tiered baseline |
| Better TTFT | Materially better on repeated-prefix traffic |
| Longer effective context | Materially longer at same GPU memory budget |

While keeping:

| Guard rail | Threshold |
|------------|-----------|
| p95 TPOT regression | <=10% vs best non-tiered baseline |
| p99 TPOT regression | <=15% vs best non-tiered baseline |
| Quality delta | <=1% vs bf16 baseline on chosen eval |

These are hackathon-grade thresholds, not production guarantees.

## Non-Goals

- Do not build a new inference engine
- Do not replace TensorRT-LLM as the primary runtime
- Do not start with vLLM + LMCache as the primary path — TRT-LLM is primary
- Do not treat KVTC as a hot-path representation — it is a cold-tier codec candidate
- Do not run multi-node jobs before the single-GPU path works
- Do not attempt Kimi K2.5 on single-GPU — node-level stretch only
- Do not block on KVTC integration — the core experiment is TRT-LLM NVFP4 + offload

## Full Benchmark Matrix

### Scenario 1 — Longer context on one GPU

| Variant | Engine | KV Mode | Offload | Notes |
|---------|--------|---------|---------|-------|
| TRT-LLM BF16 | tensorrt_llm | bf16 | no | Baseline |
| TRT-LLM FP8 | tensorrt_llm | fp8 | no | Baseline |
| TRT-LLM NVFP4 | tensorrt_llm | nvfp4 | no | Primary |
| TRT-LLM NVFP4 + offload | tensorrt_llm | nvfp4 | host | Primary thesis |
| TRT-LLM NVFP4 + offload + KVTC | tensorrt_llm | nvfp4 | host+kvtc | Optional |

Context sweep: 8k, 32k, 64k. Concurrency: 1 (fixed).

### Scenario 2 — More sessions on one GPU

| Variant | Engine | KV Mode | Offload | Notes |
|---------|--------|---------|---------|-------|
| TRT-LLM baseline | tensorrt_llm | nvfp4 | no | Baseline |
| TRT-LLM NVFP4 + offload | tensorrt_llm | nvfp4 | host | Primary thesis |
| TRT-LLM NVFP4 + offload + KVTC | tensorrt_llm | nvfp4 | host+kvtc | Optional |

Context: 8k (fixed). Concurrency sweep: 1, 2, 4, 8, 16, 32.

### Scenario 3 — Longer context + more sessions on one GPU (PRIMARY)

| Variant | Engine | KV Mode | Offload | Notes |
|---------|--------|---------|---------|-------|
| TRT-LLM baseline | tensorrt_llm | nvfp4 | no | Baseline |
| TRT-LLM NVFP4 + offload (demand) | tensorrt_llm | nvfp4 | host | Primary thesis |
| TRT-LLM NVFP4 + offload (eager) | tensorrt_llm | nvfp4 | host | Ablation |
| TRT-LLM NVFP4 + offload + KVTC | tensorrt_llm | nvfp4 | host+kvtc | Optional |

Contexts: 8k, 32k. Concurrency: 4, 8, 16. Workload: repeated_prefix.

### Scenario 4 — Longer context + more sessions on one node

Same variants as Scenario 3, one full node (8x GPUs). Only run after Scenario 3 is stable.

Additional stretch variant:

| Variant | Engine | KV Mode | Model | Notes |
|---------|--------|---------|-------|-------|
| TRT-LLM NVFP4 Kimi K2.5 | tensorrt_llm | nvfp4 | moonshotai/Kimi-K2.5 | Stretch, 8x H200 verified, cut first |

## Deliverables

1. `results/env_probe.json` — support gate result
2. Aligned TRT-LLM baseline results (bf16, fp8, nvfp4)
3. TRT-LLM NVFP4 + offload result (hot GPU + host RAM secondary tier)
4. One promotion-policy ablation (demand vs eager)
5. One benchmark comparison table (via `scripts/compare_results.py`)
6. One bottleneck summary
7. Exact rerun commands for every result

## Acceptance Criteria

- Every result JSON matches the canonical schema from `scripts/run_baseline.py`
- Every result JSON contains `engine: "tensorrt_llm"` for primary path runs
- `scripts/compare_results.py` can read every result and produce the comparison table
- Results include quality and latency together, not memory alone
- Every run captures engine, model, hardware, context, batch size, KV mode, offload status, and workload type
- The benchmark ladder is honest (support-gate results recorded, no overclaims)
- The repo can be handed to another engineer without oral explanation

## 5-Hour Execution Plan

### Hour 0:00-0:30 — Environment Probe + Support Gate

Run `scripts/env_probe.sh` to produce `results/env_probe.json`.

Must capture: GPU model, driver, CUDA, Python, TensorRT-LLM version, ModelOpt version, filesystem paths, NVFP4/FP8 KV support status.

Decision: full ladder (TRT-LLM NVFP4 supported) or reduced ladder (TRT-LLM FP8 only) or fallback (vLLM FP8) or stop (nothing works).

Verify baseline harness:
```bash
python scripts/run_baseline.py --engine tensorrt_llm --kv-mode bf16 --model Qwen/Qwen3-8B-Instruct --context-length 8192 --requests 2
```

### Hour 0:30-1:30 — TRT-LLM Aligned Single-GPU Baselines (Scenarios 1 & 2)

Model: `Qwen/Qwen3-30B-A3B` (smoke test: `Qwen/Qwen3-8B-Instruct`)

**Scenario 1 runs (context sweep, concurrency=1):**

| Variant | Context | scenario_id |
|---------|---------|-------------|
| TRT-LLM bf16 | 8192 | scenario_1_longer_context_gpu |
| TRT-LLM fp8 | 8192 | scenario_1_longer_context_gpu |
| TRT-LLM nvfp4 | 8192 | scenario_1_longer_context_gpu |
| TRT-LLM bf16 | 32768 | scenario_1_longer_context_gpu |
| TRT-LLM fp8 | 32768 | scenario_1_longer_context_gpu |
| TRT-LLM nvfp4 | 32768 | scenario_1_longer_context_gpu |

**Scenario 2 runs (concurrency sweep, context=8k):**

| Variant | Concurrency | scenario_id |
|---------|-------------|-------------|
| TRT-LLM NVFP4 baseline | 1,2,4,8,16,32 | scenario_2_more_sessions_gpu |
| TRT-LLM NVFP4 + offload | 1,2,4,8,16,32 | scenario_2_more_sessions_gpu |

Commands:
```bash
python scripts/run_baseline.py --engine tensorrt_llm --kv-mode bf16 --context-length 8192 --requests 10 --scenario-id scenario_1_longer_context_gpu --output results/trtllm_baseline_bf16_8192.json
python scripts/run_baseline.py --engine tensorrt_llm --kv-mode fp8 --context-length 8192 --requests 10 --scenario-id scenario_1_longer_context_gpu --output results/trtllm_baseline_fp8_8192.json
python scripts/run_baseline.py --engine tensorrt_llm --kv-mode nvfp4 --context-length 8192 --requests 10 --scenario-id scenario_1_longer_context_gpu --output results/trtllm_baseline_nvfp4_8192.json
```

Output: `results/trtllm_baseline_{variant}_{context}_*.json`

### Hour 1:30-3:00 — TRT-LLM NVFP4 + Offload (Scenario 3 — PRIMARY)

Run TensorRT-LLM with NVFP4 KV cache and host memory offloading:
- Hot tier: TRT-LLM NVFP4 KV cache on GPU
- Secondary tier: Host RAM via TRT-LLM KV offload/eviction
- Promotion policy: demand (default)
- Protection: 4 sink tokens, 128 recent tokens

**Scenario 3 benchmark matrix:**

| Variant | Context | Concurrency | Workload | scenario_id |
|---------|---------|-------------|----------|-------------|
| TRT-LLM NVFP4 baseline | 8k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |
| TRT-LLM NVFP4 + offload | 8k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |
| TRT-LLM NVFP4 baseline | 32k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |
| TRT-LLM NVFP4 + offload | 32k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |
| optional + KVTC | 8k | 4,8,16 | repeated_prefix | scenario_3_longer_context_more_sessions_gpu |

Commands:
```bash
# Tiered experiment with TRT-LLM NVFP4 + host offload
python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --requests 10

# Concurrent user sweep (primary KPI)
python scripts/serve_and_bench.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host \
    --sweep-concurrency 1,2,4,8,16,32 --p95-tpot-limit-ms 100
```

Must record: offload latency, restore/promotion latency, host memory usage, cache hit rate, TTFT cold vs warm, max concurrent sessions at p95 target.

This step validates the serving economics hypothesis.

### Hour 3:00-3:45 — Policy Ablation

Compare demand promotion vs eager promotion using the same tiered experiment script:

```bash
python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --promotion-policy demand --output results/trtllm_nvfp4_offload_demand.json
python scripts/run_tiered_experiment.py --engine tensorrt_llm --kv-mode nvfp4 --offload-to-host --promotion-policy eager --output results/trtllm_nvfp4_offload_eager.json
python scripts/compare_results.py --output results/comparison.md
```

### Hour 3:45-4:30 — One-Node Run (Scenario 4, only after Scenario 3 success)

**Scenario 4 benchmark matrix (same as Scenario 3, one full node):**

| Variant | Context | Concurrency | scenario_id |
|---------|---------|-------------|-------------|
| TRT-LLM NVFP4 baseline | 8k, 32k | 4,8,16 | scenario_4_longer_context_more_sessions_node |
| TRT-LLM NVFP4 + offload | 8k, 32k | 4,8,16 | scenario_4_longer_context_more_sessions_node |

Submit via Slurm:
```bash
ENGINE=tensorrt_llm KV_MODE=nvfp4 SCENARIO_ID=scenario_4_longer_context_more_sessions_node sbatch scripts/baseline_one_node.sbatch
```

Only after single-GPU success. Do not change multiple variables at once.

Optional Kimi K2.5 stretch (node-level only):
```bash
ENGINE=tensorrt_llm KV_MODE=nvfp4 MODEL=moonshotai/Kimi-K2.5 SCENARIO_ID=scenario_4_longer_context_more_sessions_node sbatch scripts/baseline_one_node.sbatch
```

### Hour 4:30-5:00 — Decision Memo + Artifacts

Produce:
- `results/comparison.md` — benchmark table from `scripts/compare_results.py`
- `results/benchmark_summary.md` — one-page summary of findings
- `results/bottleneck_summary.md` — key bottlenecks and recommendations
- One-paragraph continue/pivot/kill recommendation
- Exact rerun commands for every result

## Kill Criteria

Stop or pivot if:

1. Hot-tier format support is not available and blocks the entire stack
2. Restore/promotion cost dominates so badly that reuse no longer helps
3. p95 latency blows up without enough capacity gain to compensate
4. Quality loss is obvious and cannot be fixed with simple protection
5. Implementation becomes engine-rewrite territory
6. One-node execution is unstable and obscures the result

If blocked: fall back to nearest stable baseline (TRT-LLM FP8, then vLLM FP8), preserve the serving-capacity experiment, keep moving.

## Scope Cutting Order

Cut in this order if time is short:

1. **Kimi K2.5** — cut first (stretch goal, node-level only)
2. **Multi-node** — cut second
3. **KVTC integration** — cut third
4. **vLLM + LMCache follow-up** — cut fourth
5. **Never cut:** aligned TRT-LLM baseline table, restore timing, one serving-facing KPI

## References

- `TIERED_KV_ARCHITECTURE.md`: architectural specification (facts vs hypotheses vs measurements)
- `blackwell_kv_hackathon_context.md`: execution brief, validation ladder, source notes
- `[R1]` NVIDIA NVFP4: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- `[R2]` NVIDIA TRT-LLM KV reuse: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- `[R3]` vLLM quantized KV cache: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- `[R4]` vLLM production-stack KV cache: <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- `[R5]` LMCache docs: <https://docs.lmcache.ai/>
- `[R6]` KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- `[R7]` NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- `[R8]` TRT-LLM KV cache system: <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html> and <https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html>
- `[R9]` ModelOpt PTQ: <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md>
- `[R10]` Kimi K2.5 vLLM recipe (8x H200 verified): reference as stretch only
