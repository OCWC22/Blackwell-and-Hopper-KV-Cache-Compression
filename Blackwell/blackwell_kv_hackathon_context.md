# Blackwell KV Runtime Hackathon Context

Retrieved and updated on `2026-03-14`.

## Thesis

This repo validates that on Blackwell/B200, a hot/cold KV lifecycle can improve serving economics for reuse-heavy long-context inference.

The primary runtime path:
- **vLLM** is the serving engine with FP8 KV cache as the stable documented hot-tier path
- **LMCache** is the cold/warm reusable KV layer for KV lookup/inject/offload/sharing
- **NVFP4** is a Blackwell-aware optional hot-tier enhancement — only use if runtime support is explicitly verified
- **KVTC** is a cold-tier codec candidate for compressing reusable KV in LMCache

The metric of success is not compression ratio alone, but lower HBM pressure, more concurrent sessions, better TTFT under reuse, or longer effective context.

## What We Are Actually Validating In 24 Hours

Can Blackwell + vLLM hot KV + LMCache cold reusable KV improve one of these on a single GPU and then one node:
- lower peak HBM
- better TTFT on repeated-prefix traffic
- more concurrent sessions at fixed latency target
- longer effective context

without unacceptable p95/p99 or quality regressions?

We need to answer these questions, in order:

1. Can we run aligned `BF16` and `FP8` baselines on vLLM with the same model and prompts?
2. Does FP8 KV give expected memory reduction without unacceptable quality loss?
3. Can LMCache cold-tier reuse improve TTFT on repeated-prefix traffic?
4. Which reuse / promotion policy works best: demand vs eager?

If we cannot answer those questions with clean data, the demo is still too speculative.

## Hardware-Aware Framing

Blackwell/B200 makes the tiered KV lifecycle economically interesting:

- vLLM FP8 KV cache is the stable, documented hot-tier path
- LMCache integrates with vLLM and supports KV lookup/inject/offload/sharing for reusable prefixes
- NVFP4 is Blackwell-native (E2M1 + FP8 E4M3 microscaling) — relevant as a hot-tier enhancement if vLLM runtime support is verified, not as an assumed baseline
- KVTC is a transform codec (PCA decorrelation + adaptive quantization + entropy coding) — relevant as a cold-tier compression candidate, not as a hot-tier format

The practical implication:

- the weekend path uses vLLM FP8 KV as the stable hot tier, not undocumented NVFP4 assumptions
- the real systems question is whether LMCache cold-tier reuse improves serving economics

## Baseline Ladder

Always benchmark in this order:

1. vLLM BF16 baseline
2. vLLM FP8 KV baseline
3. vLLM FP8 KV + LMCache cold-tier reuse
4. one promotion / reuse-policy ablation (demand vs eager)
5. one-node run only after single-GPU success
6. optional: NVFP4 baseline if runtime support verified
7. optional: KVTC as cold-tier compression if codec ready

Do not block on steps 6-7. Steps 1-4 are the core experiment.

## Architecture

```text
Request arrives
  ↓
vLLM hot KV path on GPU
  - documented stable path: FP8 KV cache
  - Blackwell-aware future path: NVFP4 hot-KV if runtime support is proven
  ↓
LMCache reusable KV layer
  - lookup / inject / async store
  ↓
Cold reusable tier
  - host RAM first
  - KVTC as compression candidate
  ↓
restore / promotion only on reuse
```

Optional extensions only after the first tiering path works:

- KVTC compression in the cold tier
- NVFP4 hot-tier enhancement if runtime support verified
- multi-GPU or disaggregated follow-up if the single-GPU path is stable

## Metrics

Every run should record:

- model and revision
- context length
- batch size
- KV mode: `BF16`, `FP8`, `FP8 + LMCache`, or optional `NVFP4`
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
- run `BF16` and `FP8` on vLLM
- verify output schema and reproducibility

### Phase 2: enable LMCache cold-tier reuse

- configure LMCache integration with vLLM
- run FP8 KV + LMCache on repeated-prefix workload
- measure TTFT improvement and cache hit rate

### Phase 3: run small ablations

- recent-window protection on versus off
- sink or pivot protection on versus off
- eager promotion versus demand promotion

### Phase 4: profile before making stronger claims

- use `nsys` or `ncu` only after the baseline table exists
- inspect decode stalls and promotion overhead
- kill ideas that save memory but lose badly on `p95`

## What Not To Do

- do not claim NVFP4 hot-KV support in vLLM unless explicitly verified at runtime
- do not treat `KVTC` as a hot-path format — it is a cold-tier codec candidate
- do not blur Blackwell-native behavior with Hopper emulation
- do not claim a breakthrough if it only beats a strawman baseline
- do not drop quality measurement because the memory numbers look good

## Concrete Deliverables

The hackathon output should contain:

1. a clean benchmark table for `BF16`, `FP8`, and `FP8 + LMCache`
2. one tiered reuse result showing TTFT improvement or HBM reduction
3. one profile or bottleneck explanation that points to the next iteration
4. one defendable sentence: "On the same B200 hardware, our Blackwell-first vLLM + LMCache tiered KV setup served more reuse-heavy long-context traffic by reducing repeated prefill work and/or lowering peak KV memory pressure, while keeping latency and quality within acceptable bounds."

## Concrete vLLM + LMCache Integration

### How the KV connector works

vLLM and LMCache both hash token blocks (256 tokens per block by default). vLLM passes block hashes to LMCache via the KV connector interface. LMCache returns a bitmask of available blocks. No coordination or block map persistence needed — even across multiple vLLM instances.

Hierarchical lookup order: GPU memory → CPU memory (LMCache) → remote storage. Cache miss at each level cascades down.

### Server launch — single-GPU with LMCache

```bash
LMCACHE_CONFIG_FILE=configs/lmcache_config.yaml \
vllm serve Qwen/Qwen3-30B-A3B \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### Server launch — full 8xB200 node with LMCache

```bash
LMCACHE_CONFIG_FILE=configs/lmcache_config.yaml \
vllm serve Qwen/Qwen3-30B-A3B \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --tensor-parallel-size 8 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### Server launch — NVFP4 hot tier (support-gated)

```bash
LMCACHE_CONFIG_FILE=configs/lmcache_config.yaml \
vllm serve Qwen/Qwen3-30B-A3B \
  --kv-cache-dtype nvfp4 \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

If `--kv-cache-dtype nvfp4` is rejected, fall back to `fp8`.

### LMCache environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LMCACHE_CONFIG_FILE` | — | Path to YAML config |
| `LMCACHE_LOCAL_CPU` | `True` | Enable CPU memory backend |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | `20.0` | CPU offloading buffer in GB |
| `LMCACHE_CHUNK_SIZE` | `256` | Token chunk size per block |

### Concurrent user benchmarking

Use `scripts/serve_and_bench.py` to answer "can we serve more users?"

```bash
# FP8 baseline (no LMCache)
python scripts/serve_and_bench.py --kv-mode fp8 --tp 1

# FP8 + LMCache cold tier
python scripts/serve_and_bench.py --kv-mode fp8 --use-lmcache --tp 1

# NVFP4 + LMCache (support-gated)
python scripts/serve_and_bench.py --kv-mode nvfp4 --use-lmcache --tp 1

# Full 8xB200 node
python scripts/serve_and_bench.py --kv-mode fp8 --use-lmcache --tp 8

# Concurrency sweep (stop at p95 TPOT limit)
python scripts/serve_and_bench.py --kv-mode fp8 --use-lmcache \
    --sweep-concurrency 1,2,4,8,16,32 --p95-tpot-limit-ms 100
```

### Tiered experiment with real LMCache offloading

```bash
python scripts/run_tiered_experiment.py --use-lmcache --kv-mode fp8 --requests 10
python scripts/run_tiered_experiment.py --use-lmcache --kv-mode nvfp4 --requests 10
```

### SLA/SLO targets

| Target | Metric |
|--------|--------|
| More concurrent users | Max sessions at fixed p95 TPOT |
| Longer context | Max context length at same HBM budget |
| Energy efficiency | Tokens per joule (tok/J) |
| TTFT improvement | Warm vs cold TTFT on repeated prefix |
| HBM reduction | Peak HBM with tiering vs without |

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
