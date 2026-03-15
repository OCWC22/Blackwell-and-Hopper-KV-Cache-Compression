# Blackwell NVFP4 + KVTC Hackathon Prompt

You are acting as a senior GPU systems engineer and inference-runtime researcher.

## Task

Design and validate a Blackwell-native KV-cache runtime for long-context LLM serving.

## Goal

Use native `NVFP4` for the hot active-KV path and `KVTC` for a warm or cold reusable-KV tier. The system should minimize decode latency, maximize effective context capacity, and preserve quality.

## Environment Assumptions

- Blackwell hardware is available.
- `vLLM` or an equivalent serving engine is available.
- `LMCache` may be used as a baseline storage or offload layer, but we are not trying to rebuild `LMCache`.
- Native `NVFP4` support is available.
- `KVTC` is available as a transform-coding reference for compact storage.

## What To Do

1. write the architecture for a dual-tier KV hierarchy:
   - Tier 0: resident active `NVFP4` KV
   - Tier 1: `KVTC`-compressed reusable or stale KV
2. propose the promotion path from `KVTC` back into `NVFP4`
3. define which tokens or windows should remain protected if quality drops
4. define an experiment matrix versus:
   - `BF16` or default KV
   - `FP8` KV
   - native `NVFP4` KV
   - `LMCache` or raw offload baseline if relevant
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
- Do not use `KVTC` as the hot format by default unless latency proves acceptable.
- Optimize for latency and quality first, then storage ratio.
- Be explicit about what is Blackwell-specific.

## Key References

- NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- NVIDIA TRT-LLM KV cache early reuse: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- vLLM quantized KV cache: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- LMCache docs: <https://docs.lmcache.ai/>
- ModelOpt PTQ examples: <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md>
- TRT-LLM KV cache reuse: <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html>
