# Hopper HFP4 Research Prompt

You are acting as a senior GPU systems engineer and low-precision runtime researcher.

## Task

Design and validate a Hopper-compatible FP4-like KV-cache runtime that attempts to achieve some Blackwell-like quality and memory behavior without native `NVFP4` hardware.

## Goal

Store KV in a packed FP4-like format, reconstruct directly into `FP8` compute, reduce KV memory and bandwidth, and preserve quality with low enough decode overhead to be practical.

## Environment Assumptions

- Target research question is Hopper `H100` and `H200` behavior.
- Hopper does not have native `NVFP4`.
- `vLLM` `FP8` KV is the primary active-KV baseline.
- `LMCache` may be used as an optional cold-tier or offload baseline.
- Relevant references include direct FP8 to FP4 conversion on Hopper, `InnerQ`, `KIVI`, `KVQuant`, `IntactKV`, `KVSink`, `QServe`, and `FlashAttention-3`.

## What To Do

1. propose an HFP4 packed-KV representation for Hopper
2. explain how direct reconstruction into `FP8` should work
3. choose a grouping strategy:
   - `InnerQ`-style inner-dimension grouping
   - or an alternative with justification
4. add quality-protection mechanisms:
   - sink or pivot-token protection
   - recent-window protection
   - optional K or V asymmetric handling
5. define the benchmark matrix versus:
   - `BF16` or default KV
   - `FP8` KV baseline
   - optional `LMCache` offload
6. define the exact metrics:
   - `p50/p95` decode latency
   - tokens per second
   - HBM footprint
   - quality delta
7. recommend whether the idea is worth continuing after weekend validation

## Output Format

- 1 paragraph summary
- algorithm sketch
- decode-path bottleneck analysis
- experiment matrix table
- prioritized task list
- explicit kill criteria

## Hard Rules

- Do not claim native `NVFP4` on Hopper.
- Keep compute `FP8`-oriented.
- Avoid unnecessary `BF16` round-trips.
- Treat dequantization overhead as the main systems risk.
- Optimize for the latency-quality tradeoff, not just compression ratio.
