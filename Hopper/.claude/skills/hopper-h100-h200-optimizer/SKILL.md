# hopper-h100-h200-optimizer

Use this skill when the task involves Hopper-specific performance tuning, memory layout, kernel strategy, or hardware-aware runtime decisions.

## Hopper Guardrails

- Optimize for H100 and H200 first.
- Do not assume Blackwell-only native FP4 or NVFP4 execution paths.
- Focus on memory bandwidth, cache behavior, vectorization, TMA overlap, and efficient dequantization.
- Keep fallback paths available when proposing aggressive optimizations.

## Priorities

1. Reduce memory traffic before chasing exotic kernel ideas.
2. Keep packed storage layouts simple enough to validate.
3. Use profiling evidence to justify optimization claims.
4. Prefer compute in FP8, BF16, or FP16 after unpacking or dequantization.
5. Treat dequant latency and grouping strategy as first-class optimization targets.

## Common Wins

- contiguous layouts for KV blocks
- vectorized pack and unpack kernels
- scale granularity that balances quality and bandwidth
- persistent or fused paths when profiling shows launch overhead matters
