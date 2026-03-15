# blackwell-b200-optimizer

Use this skill when the task involves Blackwell-specific performance tuning, memory layout, promotion strategy, or hardware-aware runtime decisions.

## Blackwell Guardrails

- Optimize for Blackwell first.
- Treat `NVFP4` as the native hot-KV format.
- Focus on HBM pressure, promotion latency, and quality preservation.
- Keep fallback paths available when proposing aggressive optimizations.
- Do not import Hopper-only assumptions unless explicitly justified.

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
