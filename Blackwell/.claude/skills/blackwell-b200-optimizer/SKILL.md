# blackwell-b200-optimizer

Use this skill when the task involves Blackwell-specific performance tuning, memory layout, promotion strategy, or hardware-aware runtime decisions.

## Blackwell Guardrails

- Optimize for Blackwell first.
- TensorRT-LLM is the primary hackathon runtime. NVFP4 KV cache is the primary Blackwell hot-tier thesis.
- FP8 is the fallback if NVFP4 is not supported in TRT-LLM.
- Secondary memory offload (host RAM) is the warm/cold tier. KVTC compresses the secondary tier.
- vLLM + LMCache is the follow-up compatibility/productization path.
- Focus on HBM pressure, eviction/offload latency, promotion latency, and quality preservation.
- Keep fallback paths available when proposing aggressive optimizations.
- Do not import Hopper-only assumptions unless explicitly justified.
- For hardware specs, bandwidth budgets, and ISA details, reference the `b200-architecture` skill.

## Priorities

1. Get TRT-LLM + NVFP4 hot-KV baseline working first.
2. Add secondary-tier offload after hot-tier baseline is stable.
3. Use profiling evidence to justify eviction/promotion and runtime claims.
4. Treat `KVTC` as a secondary-tier codec; do not use as hot-path format until latency proves acceptable.
5. vLLM + LMCache optimization is a follow-up after TRT-LLM results are in.

## Common Wins

- clear hot versus warm KV boundaries
- deterministic promotion logging
- protection policies for recent or sink-sensitive token regions
- profiling-guided reduction of promotion stalls

## Sources

- For hardware specs, bandwidth budgets, and ISA details, see the `b200-architecture` skill
- NVIDIA Blackwell Tuning Guide: <https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html>
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
