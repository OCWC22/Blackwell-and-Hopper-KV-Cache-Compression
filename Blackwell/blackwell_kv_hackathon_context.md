# Blackwell KV Runtime Hackathon Context

Retrieved and updated on `2026-03-14`.

## Thesis

This is the practical hackathon path:

- keep active or hot KV in native `NVFP4`
- move stale, reusable, or oversized KV into a `KVTC` warm or cold tier
- optimize the promotion path from `KVTC` back into `NVFP4`
- preserve quality while lowering HBM pressure and avoiding decode-latency regressions

The repo should treat `vLLM` as the serving baseline and `LMCache` as the storage or offload baseline. The wedge is the Blackwell-native KV runtime and policy layer, not a replacement inference engine.

## What We Are Actually Validating In 24 Hours

We need to answer these questions, in order:

1. Can we run aligned `BF16`, `FP8`, and native `NVFP4` baselines on the same model and prompts?
2. Does `NVFP4` give the expected memory reduction without unacceptable quality loss on our actual workload?
3. Can `KVTC` act as a warm or cold tier for stale or reusable KV without giving back the win in promotion latency?
4. Which policy is best for the weekend:
   - eager promotion
   - demand fetch
   - protected recent window
   - protected sink or pivot tokens

If we cannot answer those four questions with clean data, the demo is still too speculative.

## Hardware-Aware Framing

Blackwell changes the equation because the hot format is no longer an emulation:

- `NVFP4` is native on Blackwell and should be treated as the real active-KV primitive.
- NVIDIA describes `NVFP4` as using a low-precision data format with microscaling that keeps accuracy materially better than naive 4-bit schemes.
- NVIDIA also documents Blackwell KV-cache optimizations for longer context windows and earlier KV reuse behavior.

The practical implication is simple:

- the weekend path should not waste time reinventing native `NVFP4`
- the real systems question is how to tier and promote KV without losing the latency win

## Baseline Ladder

Always benchmark in this order:

1. `BF16` or default KV baseline
2. `FP8` KV baseline
3. native `NVFP4` baseline
4. optional `LMCache` CPU or local-disk offload baseline if it is already easy to run
5. `NVFP4 + KVTC` tiering path
6. `NVFP4 + KVTC + protected recent window`
7. `NVFP4 + KVTC + protected sink or pivot tokens`

This is the minimum ladder that lets us say whether the custom runtime policy is real.

## Architecture Hypothesis

Use a two-tier KV hierarchy first:

```text
Tier 0: resident active KV in NVFP4
    |
    | decode consumes from here
    v
Tier 1: stale / reusable / oversized KV in KVTC form
    |
    | promotion path reconstructs what is needed
    v
Back into Tier 0 NVFP4 for active decode
```

Optional extensions only after the first tiering path works:

- `LMCache` as the persistence or offload baseline
- node-local storage for warm cache artifacts
- multi-GPU or disaggregated follow-up if the single-GPU path is stable

## Metrics

Every run should record:

- model and revision
- context length
- batch size
- KV mode: `BF16`, `FP8`, `NVFP4`, or `NVFP4 + KVTC`
- `p50` decode latency
- `p95` decode latency
- tokens per second
- peak HBM usage
- promotion count or hit rate if applicable
- quality delta relative to the best higher-precision baseline

If we cannot compare quality and latency in the same artifact, we will make bad decisions under time pressure.

## 24-Hour Execution Plan

### Phase 1: make the baselines real

- harden the baseline harness
- run `BF16`, `FP8`, and `NVFP4`
- verify output schema and reproducibility

### Phase 2: define the tiering surface

- decide what is eligible for `KVTC`
- decide what must stay resident in `NVFP4`
- define the promotion interface and logging

### Phase 3: run small ablations

- recent-window protection on versus off
- sink or pivot protection on versus off
- eager promotion versus demand promotion

### Phase 4: profile before making stronger claims

- use `nsys` or `ncu` only after the baseline table exists
- inspect decode stalls and promotion overhead
- kill ideas that save memory but lose badly on `p95`

## What Not To Do

- do not treat `KVTC` as a guaranteed hot-path win before proving latency
- do not blur Blackwell-native behavior with Hopper emulation
- do not claim a breakthrough if it only beats a strawman baseline
- do not drop quality measurement because the memory numbers look good

## Concrete Deliverables

The hackathon output should contain:

1. a clean benchmark table for `BF16`, `FP8`, and `NVFP4`
2. one first-pass `NVFP4 + KVTC` result, even if it is partial
3. one profile or bottleneck explanation that points to the next iteration
4. one short explanation of whether the idea deserves follow-on work

## Source Notes

Use these as the primary references for this track.

- `[R1]` NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"
  - <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- `[R2]` NVIDIA, "5x Faster Time to First Token with NVIDIA TensorRT-LLM KV Cache Early Reuse"
  - <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- `[R3]` vLLM quantized KV cache docs
  - <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- `[R4]` vLLM production-stack KV cache docs
  - <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- `[R5]` LMCache docs
  - <https://docs.lmcache.ai/>
- `[R6]` KVTC OpenReview page
  - <https://openreview.net/forum?id=tMiBQXQ0Cm>
