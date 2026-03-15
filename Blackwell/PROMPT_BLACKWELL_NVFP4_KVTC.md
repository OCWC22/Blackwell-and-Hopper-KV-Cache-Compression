# Blackwell Tiered KV Runtime Hackathon Prompt

You are acting as a senior GPU systems engineer and inference-runtime researcher.

## Task

Validate that a Blackwell-first vLLM + LMCache tiered KV runtime improves reuse-heavy long-context serving on B200.

## Goal

Use vLLM with FP8 KV cache as the stable hot-tier path and LMCache as the cold/warm reusable KV layer. The system should minimize decode latency, maximize effective context capacity, and preserve quality. NVFP4 is a Blackwell-aware optional enhancement if runtime support is verified. KVTC is a cold-tier codec candidate for compressing reusable KV.

## Environment Assumptions

- Blackwell / B200 hardware is available.
- `vLLM` is the serving engine with FP8 KV cache as the stable documented hot-tier path.
- `LMCache` is the cold/warm reusable KV layer for KV lookup/inject/offload/sharing.
- NVFP4 is a Blackwell-native format — only use as hot-tier enhancement if runtime support is explicitly verified in vLLM.
- `KVTC` is available as a cold-tier compression candidate, not a hot-tier format.

## What To Do

1. write the architecture for a tiered KV hierarchy:
   - Tier 0: vLLM FP8 KV cache (stable documented hot-tier path)
   - Tier 0b: NVFP4-aware hot path (only if runtime support verified)
   - Tier 1: LMCache-managed cold/reusable KV (host RAM first, optional KVTC compression)
2. propose the promotion path from LMCache cold tier back to vLLM hot tier
3. define which tokens or windows should remain protected if quality drops
4. define an experiment matrix:
   - `BF16` baseline
   - `FP8` KV baseline
   - `FP8` KV + LMCache cold-tier reuse
   - optional: Blackwell NVFP4 path if runtime support verified
5. state exactly how to measure:
   - `p50/p95` decode latency
   - throughput
   - HBM footprint
   - effective context capacity
   - quality deltas
6. propose the minimal implementation order for a weekend hackathon

## Output Format

- 1 paragraph summary
- architecture diagram in ASCII
- experiment matrix table
- prioritized task list
- top 5 implementation risks
- top 5 likely wins

## Hard Rules

- Do not recommend replacing the inference engine.
- Do not claim NVFP4 hot-KV support in vLLM unless explicitly verified at runtime.
- Do not use `KVTC` as a hot format — it is a cold-tier codec candidate.
- Optimize for serving economics (sessions, HBM, TTFT), not compression ratio alone.
- Be explicit about what is Blackwell-specific vs what works on any GPU.

## Key References

- NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- NVIDIA TRT-LLM KV cache early reuse: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- vLLM quantized KV cache: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- LMCache docs: <https://docs.lmcache.ai/>
- ModelOpt PTQ examples: <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md>
- TRT-LLM KV cache reuse: <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html>
