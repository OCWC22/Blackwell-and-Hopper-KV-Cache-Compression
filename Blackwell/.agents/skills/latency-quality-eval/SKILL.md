# latency-quality-eval

Use this skill when the task involves measuring performance or quality retention for the Blackwell NVFP4 + KVTC KV runtime.

## Evaluation Rules

- Compare `BF16`, `FP8`, `NVFP4`, and `NVFP4 + KVTC` with aligned prompts and generation settings.
- Use stable seeds when the framework allows it.
- Record latency, throughput, memory, and quality in the same artifact.
- Keep outputs machine-readable and easy to diff.

## Required Metrics (Every Run)

| Metric | Description |
|--------|-------------|
| TTFT | time to first token |
| p50 TPOT | median time per output token |
| p95 TPOT | tail time per output token |
| throughput | tokens per second |
| HBM footprint | peak GPU memory |
| cache-hit rate | prefix cache hit percentage |
| promotion latency | time to promote KVTC → NVFP4 (when applicable) |
| accuracy delta | quality vs best higher-precision baseline |

## Accuracy Benchmarks

For hackathon, use small subsets, not full sweeps:

- One code benchmark: LiveCodeBench or MBPP subset
- One long-context benchmark: Ruler 64K or LongBench subset

NVIDIA used LiveCodeBench, MMLU-PRO, MBPP, and Ruler 64K for NVFP4-KV claims (<1% accuracy loss). The KVTC paper used GSM8K, LiveCodeBench, LongBench, MMLU, Qasper, and RULER.

## Workflow

1. Make the benchmark CLI explicit.
2. Ensure output files capture model, hardware, context length, and KV mode.
3. Repeat runs when noise could dominate conclusions.
4. Summarize tradeoffs instead of reporting a single best number without context.

## Sources

- NVIDIA NVFP4 KV cache blog (measured baselines): <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- KVTC paper quality results (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- `TIERED_KV_ARCHITECTURE.md`: metrics table and benchmark ladder
