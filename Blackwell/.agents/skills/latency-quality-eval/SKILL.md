# latency-quality-eval

Use this skill when the task involves measuring performance or quality retention for the Blackwell KV runtime.

## Evaluation Rules

- Compare `BF16`, `FP8`, `NVFP4`, and `NVFP4 + KVTC` with aligned prompts and generation settings.
- Use stable seeds when the framework allows it.
- Record latency, throughput, memory, and quality in the same artifact.
- Keep outputs machine-readable and easy to diff.

## Suggested Metrics

- `p50` decode latency
- `p95` decode latency
- end-to-end generation time
- tokens per second
- peak GPU memory or HBM footprint
- promotion count or hit rate when applicable
- text similarity or task-specific score versus the best higher-precision baseline

## Workflow

1. Make the benchmark CLI explicit.
2. Ensure output files capture model, hardware, context length, and KV mode.
3. Repeat runs when noise could dominate conclusions.
4. Summarize tradeoffs instead of reporting a single best number without context.
