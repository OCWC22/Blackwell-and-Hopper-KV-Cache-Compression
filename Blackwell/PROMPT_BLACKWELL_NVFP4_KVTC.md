# Blackwell TRT-LLM NVFP4 + Offload Hackathon Prompt

You are acting as a senior GPU systems engineer and inference-runtime researcher.

## Task

Validate that a Blackwell-first TensorRT-LLM + NVFP4 KV cache with host offloading improves reuse-heavy long-context serving on B200.

## Goal

Use TensorRT-LLM with NVFP4 KV cache as the primary Blackwell hot-tier path and TRT-LLM host memory offloading as the secondary tier. The system should minimize decode latency, maximize effective context capacity, and preserve quality. KVTC is a cold-tier codec candidate for compressing offloaded KV blocks. vLLM + LMCache is the follow-up compatibility/productization path.

## Environment Assumptions

- Blackwell / B200 hardware is available.
- **TensorRT-LLM** is the primary serving engine with native NVFP4 KV cache support.
- **NVFP4** (E2M1 + FP8 E4M3 microscaling) is the primary hot-tier KV representation, Blackwell-native.
- **TRT-LLM KV reuse/eviction/offload** manages KV lifecycle including host memory offloading.
- **ModelOpt** is available for PTQ if needed for NVFP4 checkpoint preparation [R9].
- **KVTC** is available as a cold-tier compression candidate, not a hot-tier format.
- **vLLM + LMCache** is the follow-up path, not the primary runtime.

## Models

| Role | Model | Notes |
|------|-------|-------|
| Primary | `Qwen/Qwen3-30B-A3B` or `Qwen/Qwen3-32B` | Main benchmark model |
| Smoke test | `Qwen/Qwen3-8B-Instruct` | Use for harness validation |
| Stretch | `moonshotai/Kimi-K2.5` | 8x H200 verified, node-level Scenario 4 only |

## What To Do

1. Write the architecture for a tiered KV hierarchy:
   - Tier 0: TRT-LLM NVFP4 KV cache (primary Blackwell hot-tier path)
   - Tier 0 fallback: TRT-LLM FP8 KV cache (if NVFP4 not supported)
   - Tier 1: Host memory via TRT-LLM KV offload/eviction/reuse (optional KVTC compression)
2. Propose the promotion path from host offloaded KV back to GPU hot tier
3. Define which tokens or windows should remain protected if quality drops
4. Define an experiment matrix (see below)
5. State exactly how to measure:
   - `p50/p95/p99` decode latency
   - throughput
   - HBM footprint
   - effective context capacity
   - quality deltas
   - offload/restore latency
   - cache hit rate
6. Propose the minimal implementation order for a weekend hackathon

## Experiment Matrix

### Primary (TRT-LLM)

| Variant | Engine | KV Mode | Offload | scenario_id |
|---------|--------|---------|---------|-------------|
| TRT-LLM BF16 baseline | tensorrt_llm | bf16 | no | scenario_1/2/3 |
| TRT-LLM FP8 baseline | tensorrt_llm | fp8 | no | scenario_1/2/3 |
| TRT-LLM NVFP4 baseline | tensorrt_llm | nvfp4 | no | scenario_1/2/3 |
| TRT-LLM NVFP4 + offload (demand) | tensorrt_llm | nvfp4 | host | scenario_3 (PRIMARY) |
| TRT-LLM NVFP4 + offload (eager) | tensorrt_llm | nvfp4 | host | scenario_3 |
| TRT-LLM NVFP4 + offload + KVTC | tensorrt_llm | nvfp4 | host+kvtc | scenario_3 (optional) |

### Follow-Up (vLLM + LMCache)

| Variant | Engine | KV Mode | Cold Tier | scenario_id |
|---------|--------|---------|-----------|-------------|
| vLLM FP8 baseline | vllm | fp8 | none | scenario_3 |
| vLLM FP8 + LMCache | vllm | fp8 | lmcache | scenario_3 |

## Expected TRT-LLM Outputs

Each run must produce a JSON result file containing at minimum:

```json
{
  "engine": "tensorrt_llm",
  "model": "Qwen/Qwen3-30B-A3B",
  "kv_mode": "nvfp4",
  "offload_enabled": true,
  "offload_target": "host",
  "cold_tier_codec": "none",
  "promotion_policy": "demand",
  "context_length": 8192,
  "batch_size": 4,
  "scenario_id": "scenario_3_longer_context_more_sessions_gpu",
  "p50_latency_ms": 0.0,
  "p95_latency_ms": 0.0,
  "p99_latency_ms": 0.0,
  "tokens_per_second": 0.0,
  "peak_hbm_gb": 0.0,
  "ttft_cold_ms": 0.0,
  "ttft_warm_ms": 0.0,
  "cache_hit_rate": 0.0,
  "quality_delta_vs_bf16": 0.0,
  "max_concurrent_sessions": 0,
  "offload_latency_ms": 0.0,
  "restore_latency_ms": 0.0
}
```

## Output Format

- 1 paragraph summary
- architecture diagram in ASCII
- experiment matrix table
- prioritized task list
- top 5 implementation risks
- top 5 likely wins

## Hard Rules

- Do not recommend replacing the inference engine.
- Do not start with vLLM + LMCache as the primary runtime path — TRT-LLM is primary.
- Do not claim NVFP4 hot-KV support unless explicitly verified at runtime via support gate.
- Do not use `KVTC` as a hot format — it is a cold-tier codec candidate.
- Do not attempt Kimi K2.5 on single-GPU — node-level stretch only.
- Do not run multi-node before single-GPU success.
- Optimize for serving economics (sessions, HBM, TTFT), not compression ratio alone.
- Be explicit about what is Blackwell-specific vs what works on any GPU.

## Key References

- `[R1]` NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- `[R2]` NVIDIA TRT-LLM KV cache early reuse: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- `[R3]` vLLM quantized KV cache: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- `[R4]` vLLM production-stack KV cache: <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- `[R5]` LMCache docs: <https://docs.lmcache.ai/>
- `[R6]` KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- `[R7]` NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- `[R8]` TRT-LLM KV cache system: <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html> and <https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html>
- `[R9]` ModelOpt PTQ: <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md>
- `[R10]` Kimi K2.5 vLLM recipe (8x H200 verified): reference as stretch only
