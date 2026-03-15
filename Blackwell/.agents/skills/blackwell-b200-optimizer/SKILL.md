# blackwell-b200-optimizer

Use this skill when the task involves Blackwell-specific performance tuning, memory layout, promotion strategy, or hardware-aware runtime decisions.

## Blackwell Guardrails

- Optimize for Blackwell first.
- Treat `NVFP4` as the native hot-KV format.
- Focus on HBM pressure, promotion latency, and quality preservation.
- Keep fallback paths available when proposing aggressive optimizations.
- Do not import Hopper-only assumptions unless explicitly justified.
- For hardware specs, bandwidth budgets, and ISA details, reference the `b200-architecture` skill.

## Priorities

1. Reduce hot-tier memory pressure before chasing exotic compression ideas.
2. Keep the resident `NVFP4` path simple enough to validate.
3. Use profiling evidence to justify promotion and runtime claims.
4. Treat `KVTC` as a secondary representation unless hot-path latency proves acceptable.

## Common Wins

- clear hot versus warm KV boundaries
- deterministic promotion logging
- protection policies for recent or sink-sensitive token regions
- profiling-guided reduction of promotion stalls

## Sources

- For hardware specs, bandwidth budgets, and ISA details, see the `b200-architecture` skill
- NVIDIA Blackwell Tuning Guide: <https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html>
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
