# kv-cache-research

Use this skill when the task is about Blackwell experiment design, benchmark structure, ablations, or interpreting memory-versus-latency tradeoffs for the TRT-LLM + NVFP4 + offload tiered KV runtime.

## Blackwell Benchmark Ladder

Always compare aligned baselines in this order on TRT-LLM:

1. `BF16` or default KV baseline
2. `FP8` KV baseline
3. `NVFP4` KV baseline (primary Blackwell thesis)
4. `NVFP4 + offload` (secondary tier with host memory)
5. `NVFP4 + offload + KVTC` (compressed secondary tier)
6. eviction/promotion and protection ablations

vLLM + LMCache experiments are a follow-up compatibility path.

## Required Metrics (Every Run)

| Metric | Description |
|--------|-------------|
| TTFT | time to first token |
| p50 TPOT | median time per output token |
| p95 TPOT | tail time per output token |
| throughput | tokens per second |
| HBM footprint | peak GPU memory |
| cache-hit rate | prefix cache hit percentage |
| promotion latency | time to restore cold KV → hot tier (when applicable) |
| accuracy delta | quality vs best higher-precision baseline |

## Evaluation Matrix (from PRD)

- Models: one Llama-family model, one Qwen-family model if time allows
- Contexts: `8k`, `32k`, `64k`
- Modes: `BF16`, `FP8`, `NVFP4`, `NVFP4 + offload` (TRT-LLM primary); `FP8 + LMCache` (follow-up)
- Ablations: recent-window protection, sink-token protection, eager vs demand promotion

## Research Frame

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

## Sources

- `TIERED_KV_ARCHITECTURE.md`: full tiered architecture spec with benchmark ladder and metrics
- `BLACKWELL_24H_PRD.md`: evaluation matrix and workstreams
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
