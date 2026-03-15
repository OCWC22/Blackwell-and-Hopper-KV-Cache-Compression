# Blackwell KV Runtime Hackathon Context

Retrieved and updated on `2026-03-15`.

## Four Benchmark Scenarios

| Scenario | Question | scenario_id |
|----------|----------|-------------|
| 1 — Longer context, one GPU | How far can one GPU go? | `scenario_1_longer_context_gpu` |
| 2 — More sessions, one GPU | How many concurrent sessions? | `scenario_2_more_sessions_gpu` |
| **3 — Both, one GPU (PRIMARY)** | Many users + long prompts? | `scenario_3_longer_context_more_sessions_gpu` |
| 4 — Both, one node | Same idea at node level? | `scenario_4_longer_context_more_sessions_node` |

Scenario 3 is the primary target. Scenario 4 is the follow-up. Scenarios 1 and 2 are explanatory baselines.

## Thesis

This repo validates that on Blackwell/B200, **TensorRT-LLM with NVFP4 KV cache and host offloading** improves serving economics for reuse-heavy long-context inference.

The primary runtime path:
- **TensorRT-LLM** is the serving engine with native NVFP4 KV cache support as the primary Blackwell hot-tier path
- **NVFP4** (E2M1 + FP8 E4M3 microscaling) is the primary hot-tier KV representation, Blackwell-native
- **TRT-LLM KV reuse/eviction/offload** manages KV lifecycle including host memory offloading for cold/stale prefixes
- **KVTC** is a cold-tier codec candidate for compressing offloaded KV blocks before host storage

The follow-up compatibility path:
- **vLLM** with FP8 KV cache as the stable productization engine
- **LMCache** as the cold/warm reusable KV layer for KV lookup/inject/offload/sharing
- **KVTC** as a cold-tier codec candidate within LMCache

The metric of success is not compression ratio alone, but lower HBM pressure, more concurrent sessions, better TTFT under reuse, or longer effective context.

## What We Are Actually Validating In 24 Hours

Can Blackwell + TRT-LLM NVFP4 KV + host offloading improve one of these on a single GPU and then one node:
- lower peak HBM
- better TTFT on repeated-prefix traffic
- more concurrent sessions at fixed latency target
- longer effective context

without unacceptable p95/p99 or quality regressions?

We need to answer these questions, in order:

1. Can we run aligned `BF16` and `FP8` baselines on TensorRT-LLM with the same model and prompts?
2. Does NVFP4 KV give expected memory reduction over FP8 without unacceptable quality loss?
3. Does TRT-LLM host offloading improve TTFT and concurrency on repeated-prefix traffic?
4. Which eviction / promotion policy works best: demand vs eager?

If we cannot answer those questions with clean data, the demo is still too speculative.

## Hardware-Aware Framing

Blackwell/B200 makes the NVFP4 KV lifecycle economically interesting:

- TensorRT-LLM has native NVFP4 KV cache support on Blackwell, documented and optimized [R1][R7]
- TRT-LLM KV cache reuse can deliver up to 5x faster TTFT via early reuse of computed KV blocks [R2][R8]
- NVFP4 is Blackwell-native (E2M1 + FP8 E4M3 microscaling) — 4-bit KV with hardware-accelerated dequant
- KVTC is a transform codec (PCA decorrelation + adaptive quantization + entropy coding) — relevant as a cold-tier compression candidate, not as a hot-tier format [R6]

The practical implication:

- the hackathon path uses TRT-LLM NVFP4 KV as the primary hot tier
- the real systems question is whether NVFP4 + host offloading improves serving economics
- vLLM + LMCache is the follow-up compatibility/productization path

## Models

| Role | Model | Notes |
|------|-------|-------|
| Primary | `Qwen/Qwen3-30B-A3B` or `Qwen/Qwen3-32B` | Main benchmark model |
| Smoke test | `Qwen/Qwen3-8B-Instruct` | Use for harness validation |
| Stretch | `moonshotai/Kimi-K2.5` | 8x H200 verified, node-level Scenario 4 only, NOT for single-GPU proof |

Kimi K2.5 is a stretch goal. It has been verified on 8x H200 nodes [R10]. Do not attempt on single-GPU Blackwell. Cut Kimi first if time is short.

## Baseline Ladder

Always benchmark in this order:

1. TRT-LLM BF16 baseline (default KV)
2. TRT-LLM FP8 KV baseline
3. TRT-LLM NVFP4 KV baseline
4. TRT-LLM NVFP4 KV + host offload
5. One promotion / eviction-policy ablation (demand vs eager)
6. One-node run only after single-GPU success
7. Optional: TRT-LLM NVFP4 + offload + KVTC cold-tier compression
8. Follow-up: vLLM FP8 baseline
9. Follow-up: vLLM FP8 + LMCache cold-tier reuse

Do not block on steps 7-9. Steps 1-6 are the core experiment.

## Architecture

### Primary Path: TRT-LLM + NVFP4

```text
Request arrives
  |
TensorRT-LLM engine
  |
NVFP4 hot KV on GPU
  - Blackwell-native E2M1 + FP8 E4M3 microscaling
  - hardware-accelerated dequant
  |
TRT-LLM KV reuse / eviction / offload
  - KV cache early reuse for repeated prefixes
  - eviction of stale/cold KV blocks
  |
Secondary memory (host RAM)
  - offloaded cold KV blocks
  - optional KVTC compression before storage
  |
Restore / promotion back to NVFP4 hot tier
  - demand or eager policy
  - optional KVTC decompress on restore
```

### Follow-Up Path: vLLM + LMCache

```text
Request arrives
  |
vLLM serving engine
  |
FP8 hot KV on GPU
  - stable documented path
  |
LMCache reusable KV layer
  - lookup / inject / async store
  |
Cold / warm storage (host RAM)
  - KVTC as compression candidate
  |
Restore / promotion back to FP8 hot tier
```

## Benchmark Variants

| Variant | Engine | KV Mode | Offload | Notes |
|---------|--------|---------|---------|-------|
| TRT-LLM BF16 | tensorrt_llm | bf16 | no | Baseline |
| TRT-LLM FP8 | tensorrt_llm | fp8 | no | Baseline |
| TRT-LLM NVFP4 | tensorrt_llm | nvfp4 | no | Primary hot tier |
| TRT-LLM NVFP4 + offload | tensorrt_llm | nvfp4 | host | Primary thesis |
| TRT-LLM NVFP4 + offload + KVTC | tensorrt_llm | nvfp4 | host+kvtc | Optional |
| vLLM FP8 | vllm | fp8 | no | Follow-up |
| vLLM FP8 + LMCache | vllm | fp8 | lmcache | Follow-up |

## Metrics

Every run should record:

- engine and version
- model and revision
- context length
- batch size
- KV mode: `BF16`, `FP8`, `NVFP4`
- offload enabled: yes/no
- cold-tier codec: none, KVTC
- `p50` decode latency
- `p95` decode latency
- `p99` decode latency
- tokens per second
- peak HBM usage
- promotion count or hit rate if applicable
- quality delta relative to the best higher-precision baseline

If we cannot compare quality and latency in the same artifact, we will make bad decisions under time pressure.

## Execution Ladder (11 steps)

1. **Env probe** — `bash scripts/env_probe.sh` → `results/env_probe.json`
2. **Support gate** — determine BF16/FP8/NVFP4 KV support from probe
3. **Baseline harness** — verify `run_baseline.py --engine tensorrt_llm` schema
4. **TRT-LLM BF16** — `--engine tensorrt_llm --kv-mode bf16`
5. **TRT-LLM FP8** — `--engine tensorrt_llm --kv-mode fp8`
6. **TRT-LLM NVFP4** — `--engine tensorrt_llm --kv-mode nvfp4`
7. **TRT-LLM NVFP4 + offload** — `--engine tensorrt_llm --kv-mode nvfp4 --offload-to-host`
8. **+ KVTC** — optional cold-tier compression on offloaded blocks
9. **Ablation** — demand vs eager promotion policy
10. **One-node** — Scenario 4, only after single-GPU success
11. **Decision memo** — comparison table + bottleneck summary + recommendation

## KPI Targets

Achieve at least one of:
- >=20% lower peak HBM vs best non-tiered baseline
- Materially better TTFT on repeated-prefix traffic
- >=25% more concurrent sessions at fixed p95 target
- Materially longer effective context at same HBM budget

Guards:
- p95 TPOT regression <= 10% vs best non-tiered baseline
- p99 TPOT regression <= 15%
- Quality delta <= 1% vs BF16 baseline

## One Sentence by 4 PM

"On the same B200 hardware, our TensorRT-LLM NVFP4 KV cache with host offloading served more reuse-heavy long-context traffic by reducing KV memory pressure and reusing computed prefixes, while keeping latency and quality within acceptable bounds."

## Follow-Up Compatibility Path: vLLM + LMCache

After the TRT-LLM primary results are stable, the follow-up path validates the same thesis on the vLLM + LMCache stack for productization:

### How the LMCache KV connector works

vLLM and LMCache both hash token blocks (256 tokens per block by default). vLLM passes block hashes to LMCache via the KV connector interface. LMCache returns a bitmask of available blocks. No coordination or block map persistence needed.

Hierarchical lookup order: GPU memory -> CPU memory (LMCache) -> remote storage.

### Server launch — vLLM + LMCache (follow-up)

```bash
LMCACHE_CONFIG_FILE=configs/lmcache_config.yaml \
vllm serve Qwen/Qwen3-30B-A3B \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### LMCache environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LMCACHE_CONFIG_FILE` | — | Path to YAML config |
| `LMCACHE_LOCAL_CPU` | `True` | Enable CPU memory backend |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | `20.0` | CPU offloading buffer in GB |
| `LMCACHE_CHUNK_SIZE` | `256` | Token chunk size per block |

## What Not To Do

- do not start with vLLM + LMCache as the primary path — TRT-LLM is primary
- do not claim NVFP4 hot-KV support unless explicitly verified at runtime via support gate
- do not treat `KVTC` as a hot-path format — it is a cold-tier codec candidate
- do not blur Blackwell-native behavior with Hopper emulation
- do not claim a breakthrough if it only beats a strawman baseline
- do not drop quality measurement because the memory numbers look good
- do not attempt Kimi K2.5 on single-GPU — node-level stretch only

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
- `[R7]` NVIDIA, "Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache"
  - <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- `[R8]` TRT-LLM KV cache system docs
  - <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html>
  - <https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html>
- `[R9]` ModelOpt PTQ examples
  - <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md>
- `[R10]` Kimi K2.5 vLLM recipe (8x H200 verified) — reference as stretch only
