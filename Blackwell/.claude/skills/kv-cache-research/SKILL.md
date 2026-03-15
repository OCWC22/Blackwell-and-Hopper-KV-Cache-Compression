# kv-cache-research

Use this skill when the task is about Blackwell experiment design, benchmark structure, ablations, or interpreting memory-versus-latency tradeoffs for KV-cache runtime work.

## Research Frame

- Compare aligned baselines in this order: `BF16` or default, `FP8`, native `NVFP4`, then `NVFP4 + KVTC`.
- Track latency, throughput, memory use, and quality retention together.
- Keep context lengths, prompts, seeds, and model settings aligned across variants.
- Capture enough metadata so results are comparable across runs and hardware.
- Do not treat `KVTC` as a hot-path win until latency is measured.

## Recommended Workflow

1. Identify the current baseline path in the repo.
2. Define the next missing experiment component instead of designing the entire roadmap at once.
3. Make result schemas explicit and machine-readable.
4. Keep ablations focused on one independent variable per run when possible.
5. Note assumptions that could bias interpretation.

## Useful Outputs

- config templates
- run manifests
- JSON result schemas
- concise comparison reports
