# nsys-ncu-profiler

Use this skill when preparing or analyzing Nsight Systems or Nsight Compute profiling runs for the Blackwell track.

## Profiling Rules

- Profile only on an allocated GPU node.
- Keep captures scoped to the kernels or phases that matter.
- Save traces and summaries under `results/` so they can be shared later.
- Use profiling evidence to explain promotion or decode bottlenecks, not to decorate guesses.

## Workflow

1. Decide whether timeline tracing or kernel-level analysis is the right first tool.
2. Add a Slurm-safe profiling entrypoint when needed.
3. Keep output filenames deterministic.
4. Note which promotion, decode, or memory ranges to inspect first.
5. Summarize what to compare before and after an optimization.
