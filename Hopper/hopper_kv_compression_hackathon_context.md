# Hopper KV Runtime Research Context

Retrieved and updated on `2026-03-14`.

## Thesis

This is the Hopper research path:

- use deployed `H100` and `H200` fleets as the real target
- store KV in a packed FP4-like representation
- reconstruct directly into `FP8`-oriented compute on Hopper
- use hardware-aware grouping plus protected-token policies to preserve quality
- optimize the active decode path first, then add colder storage tiers

The wedge is not "another cache layer." The wedge is low-latency active-KV compression and reconstruction for Hopper-class inference.

## Core Research Question

Can Hopper achieve something close to Blackwell-like KV economics without native `NVFP4` by combining:

- FP4-like packed storage
- direct `FP8` reconstruction
- better grouping and scale reuse
- protected recent windows
- protected sink or pivot tokens

That is the question the repo should stay centered on.

## Hardware Facts That Matter

The bottleneck is not subtle.

- H100 SXM HBM bandwidth is about `3.35 TB/s`.
- H200 HBM3e bandwidth is about `4.8 TB/s`.
- PCIe Gen5 x16 is only about `63 GB/s` theoretical per direction, even though the bidirectional headline is about `128 GB/s`.

Practical implication:

- host refill is tens of times slower than local HBM access
- any packed format that adds too much dequant overhead loses the real game
- the win condition is smaller KV with acceptable decode cost, not smaller KV at any price

## Baseline Ladder

Always benchmark in this order:

1. `BF16` or default KV baseline
2. `FP8` KV baseline
3. first packed FP4-like path
4. packed FP4-like path plus direct `FP8` reconstruction
5. add protected recent window
6. add protected sink or pivot tokens
7. compare grouping strategies:
   - naive grouping
   - `InnerQ`-style inner-dimension grouping
8. optionally add `LMCache` as a colder tier after the hot path is real

If the packed representation does not stay competitive with the `FP8` baseline, it is not the right path.

## Research Stack

Treat the stack like this:

- `vLLM`: serving engine and `FP8` KV baseline
- `LMCache`: optional cold-tier or persistence baseline
- our runtime: packed FP4-like storage, direct `FP8` reconstruction, and quality protection

This separation matters because we are not trying to rewrite the whole serving stack.

## What We Need To Learn

We need concrete answers to these questions:

1. what grouping strategy makes dequant cheap enough on Hopper?
2. which tokens must stay higher precision to keep quality stable?
3. how much memory do we save versus `FP8`?
4. what happens to `p50` and `p95` decode latency?
5. is there a realistic path to a cold-tier extension after the hot decode path works?

## Implementation Hypothesis

Start with an HFP4-style packed layout:

```text
packed 4-bit values
  +
small per-group scale metadata
  +
direct reconstruction into FP8-oriented compute
  +
quality protection for token regions that are known to be sensitive
```

The first runtime pass should avoid unnecessary `BF16` round-trips and keep the active-path interface narrow enough to profile.

## Metrics

Every run should record:

- model and revision
- context length
- batch size
- KV mode: `BF16`, `FP8`, or packed FP4-like
- `p50` decode latency
- `p95` decode latency
- tokens per second
- peak HBM usage
- quality delta relative to the higher-precision baseline
- grouping strategy
- protection policy settings

## Research Phases

### Phase 1: baseline and packed format

- harden the baseline harness
- run `BF16` and `FP8`
- define the packed layout and correctness checks

### Phase 2: direct reconstruction

- reconstruct directly into `FP8`-oriented compute
- measure the dequant or unpack cost
- compare against the `FP8` baseline before adding more complexity

### Phase 3: quality protection

- recent-window protection on versus off
- sink or pivot protection on versus off
- inner-dimension grouping versus naive grouping

### Phase 4: optional colder-tier work

- only after the active decode path is understood
- evaluate whether `LMCache` or another colder tier is worth integrating

## Kill Criteria

Kill or pause the idea if:

- the packed path saves memory but loses badly on `p95` decode latency
- the quality delta stays clearly worse than the `FP8` baseline even after protection policies
- the implementation requires enough complexity that it no longer looks like a Hopper-compatible product path

## Paper Stack

Prioritize these sources in roughly this order:

- `[R1]` Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs
  - <https://arxiv.org/abs/2603.02731>
- `[R2]` InnerQ: Faster 4-bit KV Cache Quantization with Inner-Dimension Grouping
  - <https://arxiv.org/abs/2602.23200>
- `[R3]` QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving
  - <https://arxiv.org/abs/2405.04532>
- `[R4]` KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
  - <https://arxiv.org/abs/2402.02750>
- `[R5]` KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
  - <https://arxiv.org/abs/2401.18079>
- `[R6]` IntactKV
  - <https://aclanthology.org/2024.findings-acl.560/>
- `[R7]` KVSink: Attention Sinks are Efficient Context Compressors for KV Cache Compression
  - <https://arxiv.org/abs/2508.04257>
- `[R8]` FlashAttention-3 blog and paper resources
  - <https://tridao.me/blog/2024/flash3/>
- `[R9]` NVIDIA H100 product page
  - <https://www.nvidia.com/en-us/data-center/h100/>
- `[R10]` NVIDIA H200 product page
  - <https://www.nvidia.com/en-us/data-center/h200/>
- `[R11]` NVIDIA Hopper architecture in depth
  - <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>
- `[R12]` PCI-SIG bandwidth table reference
  - <https://pcisig.com/sites/default/files/files/PCIe%206.0%20Webinar_Final_.pdf>
