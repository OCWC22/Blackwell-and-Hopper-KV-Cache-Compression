# Blackwell 24-Hour PRD

## Objective

Validate a Blackwell-native KV runtime that keeps hot KV in `NVFP4`, uses `KVTC` as a warm or cold tier, and shows a defensible memory-versus-latency-versus-quality tradeoff on `B200` hardware.

## Why This Matters

- it is the most credible hackathon path on the hardware we actually have
- it produces a product-shaped demo instead of a speculative kernel story
- it can roll forward into a company wedge around KV-memory runtime infrastructure

## Non-Goals

- do not build a new inference engine
- do not replace `vLLM`
- do not replace `LMCache`
- do not treat `KVTC` as a guaranteed hot-path representation
- do not run giant multi-node jobs before the single-node path is real

## Deliverables

1. aligned `BF16`, `FP8`, and `NVFP4` baseline results
2. first `NVFP4 + KVTC` tiering result
3. one explicit promotion-policy ablation
4. one profile or bottleneck explanation
5. one short decision memo: continue, pivot, or kill

## Acceptance Criteria

- results are machine-readable
- results include quality and latency together
- every run captures model, hardware, context, batch size, and KV mode
- the benchmark ladder is honest
- the repo can be handed to another engineer without extra oral context

## Workstreams

### Workstream 0: environment and cluster verification

Tasks:

1. verify B200 visibility on the target cluster or node
2. capture `nvidia-smi`, driver, CUDA, and Slurm metadata
3. record shared-storage and scratch-path assumptions
4. make a single source of truth for run commands

### Workstream 1: baseline harness hardening

Tasks:

1. ensure `scripts/run_baseline.py` writes stable JSON
2. ensure `scripts/baseline_inference.sh` is safe on shared Slurm
3. add explicit KV mode metadata
4. verify that repeated runs are comparable

### Workstream 2: single-GPU Blackwell ladder

Tasks:

1. run `BF16` or default KV
2. run `FP8`
3. run native `NVFP4`
4. compare memory footprint, `p50`, `p95`, throughput, and quality

### Workstream 3: one-node B200 sweep

Tasks:

1. scale the aligned run matrix to one full B200 node if the stack supports it
2. capture node-level metadata
3. compare single-GPU versus one-node behavior
4. confirm that the runbook works without manual repair

### Workstream 4: first `NVFP4 + KVTC` integration

Tasks:

1. define the resident hot tier
2. define what goes to `KVTC`
3. log promotion and protection policy settings
4. run one first integrated result

### Workstream 5: promotion-policy ablations

Tasks:

1. eager promotion versus demand promotion
2. protected recent window on versus off
3. protected sink or pivot tokens on versus off
4. document what changed in latency and quality

### Workstream 6: profiling and bottleneck analysis

Tasks:

1. capture at least one `nsys` or `ncu` run
2. identify whether the bottleneck is decode, promotion, memory traffic, or framework overhead
3. record the top three next optimizations

### Workstream 7: decision memo and handoff

Tasks:

1. summarize the benchmark table
2. summarize the bottleneck analysis
3. decide whether the idea should continue after the hackathon
4. leave exact commands and files for the next engineer

## Evaluation Matrix

Minimum matrix:

- models: one Llama-family model, one Qwen-family model if time allows
- contexts: `8k`, `32k`, `64k`
- modes: `BF16`, `FP8`, `NVFP4`, `NVFP4 + KVTC`
- policy ablations: recent-window protection, sink or pivot-token protection, eager versus demand promotion

## Kill Criteria

- `NVFP4 + KVTC` loses materially on `p95` decode latency and does not recover enough memory to matter
- quality drops clearly versus the `NVFP4` baseline even after basic protection policies
- the implementation surface is too unstable to hand off cleanly

## References

- `TIERED_KV_ARCHITECTURE.md`: full architectural specification for NVFP4 hot + KVTC warm/cold tiering
- `blackwell_kv_hackathon_context.md`: execution brief, validation ladder, and source notes
- NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- NVIDIA TRT-LLM KV cache early reuse: <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- vLLM quantized KV cache: <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- LMCache docs: <https://docs.lmcache.ai/>
