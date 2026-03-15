# Blackwell KV Runtime: Tiered Architecture

This document separates upstream documented facts from repo hypotheses from things that must be measured here. Do not conflate them.

---

## Bucket 1: Upstream Documented Facts

These are verified in public documentation. Each has a source citation.

### NVFP4 is Blackwell-native

- NVFP4 is a 4-bit floating-point format native to Blackwell SM120 GPUs
- Representation: E2M1 payload with FP8 E4M3 scale per 16-value micro-block, FP32 per-tensor second-level scale
- Dequantization: to FP8 before attention and context math
- Memory: roughly 50% reduction vs FP8 KV cache
- Quality: <1% accuracy loss documented on LiveCodeBench, MMLU-PRO, MBPP, and Ruler 64K
- Benefits: up to 3x lower TTFT, 20% higher cache hit rate vs FP8 in Blackwell studies
- **Source:** `[R1]` NVIDIA, "Introducing NVFP4" — <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- **Source:** `[R7]` NVIDIA, "NVFP4 KV Cache" — <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>

### TensorRT-LLM has KV reuse, offloading, and connectors

- KV Cache Early Reuse: enables prefix sharing and cache reuse across requests
- KV Cache Connector: persistence/reuse interface for KV blocks
- Prioritized eviction: configurable KV eviction policies
- Offloading: supports KV movement between GPU and host memory
- NVFP4 KV cache: TRT-LLM release notes validate Qwen3-32B and Qwen3-30B-A3B with NVFP4
- Host cache: `KvCacheConfig.host_cache_size` enables host memory as secondary KV storage
- Block reuse: `KvCacheConfig.enable_block_reuse` enables KV block sharing across requests
- **Source:** `[R2]` NVIDIA, "5x Faster TTFT" — <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- **Source:** `[R8]` TRT-LLM KV Cache Reuse — <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html>
- **Source:** `[R8]` TRT-LLM KV Cache System — <https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html>

### vLLM supports FP8 KV cache (follow-up path)

- vLLM documents FP8 KV cache quantization as a first-class path
- vLLM supports ModelOpt NVFP4 checkpoints for model weights, but public KV docs are FP8-centered
- vLLM prefix caching enables KV reuse for repeated prefixes
- **Source:** `[R3]` vLLM quantized KV cache — <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- **Source:** `[R4]` vLLM production-stack KV cache — <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>

### LMCache supports KV offloading, compression, and sharing (follow-up path)

- Pinned CPU RAM as hot host cache
- Async loading for I/O overlap
- `remote_serde` compression hook (CacheGen today)
- Controller APIs for compress/decompress
- KV sharing across requests and sessions
- **Source:** `[R5]` LMCache docs — <https://docs.lmcache.ai/>

### KVTC achieves high compression with quality retention

- Pipeline: PCA-based decorrelation → adaptive quantization via dynamic programming → entropy coding (nvCOMP DEFLATE)
- Compression: roughly 20x average vs 16-bit KV, 40x+ in some cases
- Calibration: PCA basis computed once per model; same basis reused across compression ratios with only bit allocation changing
- Quality: strong reasoning and long-context accuracy retention; <1 score point loss at 16x, stable up to 32x-64x
- Pre-RoPE compression yields better quality than post-RoPE
- **Source:** `[R6]` KVTC, "KV-Cache Compression with Learned Transform Coding" (ICLR 2026) — <https://openreview.net/forum?id=tMiBQXQ0Cm>

### ModelOpt provides NVFP4 quantization recipes

- Combined FP8 + NVFP4-KV quantization recipe via `mtq.NVFP4_KV_CFG`
- Deployment via TensorRT-LLM (primary) or vLLM (secondary)
- **Source:** `[R9]` ModelOpt PTQ examples — <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md>

---

## Bucket 2: Repo Hypothesis

These are things this repo proposes but has **not yet measured**. They are testable claims, not established facts.

**Core claim: We are validating serving economics, not compression ratio alone.**

### Primary architecture (hackathon path)

```
                ┌──────────────────────────────┐
                │           Request            │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │       TensorRT-LLM           │
                │   primary hackathon runtime  │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   Hot KV tier (GPU memory)   │
                │  primary: NVFP4 KV cache     │
                │  fallback: FP8 KV cache      │
                └──────────────┬───────────────┘
                               │
                      eviction │ reuse/restore
                               │
                               ▼
                ┌──────────────────────────────┐
                │  TRT-LLM KV reuse / offload  │
                │  eviction + host cache mgmt  │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Secondary memory (host RAM)  │
                │  KVTC compression candidate  │
                │  on the offloaded tier       │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  restore / promotion on hit  │
                │ back into active NVFP4 tier  │
                └──────────────────────────────┘
```

### Follow-up compatibility architecture (productization path)

```
                ┌──────────────────────────────┐
                │           Request            │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │            vLLM              │
                │   follow-up serving engine   │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │       Hot KV tier (GPU)      │
                │  stable path: FP8 KV cache   │
                └──────────────┬───────────────┘
                               │
                      reuse miss│reuse hit
                               │
                               ▼
                ┌──────────────────────────────┐
                │           LMCache            │
                │ KV lookup / inject / store   │
                │ cold/warm reusable KV layer  │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │      Cold / warm storage     │
                │ host RAM first               │
                │ KVTC as codec candidate      │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  restore / promotion on hit  │
                │ back into active hot tier    │
                └──────────────────────────────┘
```

### Why longer context and more sessions are different KV-pressure modes

**Longer context** (Scenario 1) puts pressure on **KV bytes per session**. A single session with 64K context on a 32B model may consume 8-16 GB of KV alone in FP16, leaving little room for model weights or other sessions. NVFP4 directly reduces this per-session footprint by ~50% vs FP8.

**More sessions** (Scenario 2) puts pressure on **total live KV replicas**. Even at short context, 32 concurrent sessions each with their own KV blocks can exhaust HBM. Offloading stale KV to host memory and restoring on reuse keeps more sessions alive without OOM.

**Both** (Scenario 3, PRIMARY) is the hardest: context_length × active_sessions. This is where NVFP4 hot-tier + host offload is most valuable — NVFP4 compresses per-session KV, and offload manages the total active set.

### Primary hypothesis

On Blackwell/B200, an NVFP4 hot KV tier plus host-offloaded secondary tier can improve serving economics for reuse-heavy long-context inference. The hot decode path uses NVFP4 KV in TensorRT-LLM, and stale/evicted KV moves to host memory for later restore.

### Specific claims to test

1. **NVFP4 hot KV reduces HBM pressure** — NVFP4 KV cache gives meaningful HBM reduction vs FP8 with acceptable quality loss on Blackwell
2. **Host offload extends serving capacity** — offloading evicted KV to host memory and restoring on reuse lets the GPU serve more sessions or longer context
3. **Protection policies preserve quality** — protecting first 4 sink tokens and last 128 recent tokens maintains quality within 1% of baseline
4. **Demand promotion beats eager promotion** — restoring KV only on cache hit is more efficient than pre-loading all cold KV
5. **Compressed cold tier beats raw cold tier** — KVTC compression in the secondary tier saves enough host memory or bandwidth to be worth the codec overhead

### Architecture hypothesis

```
Tier 0 — Hot KV (GPU memory)
  Primary: TRT-LLM NVFP4 KV cache
  Fallback: TRT-LLM FP8 KV cache

Tier 1 — Secondary offload (host memory)
  Managed by TRT-LLM KV offload / eviction / reuse
  Optional compression via KVTC codec candidate

Goal:
  Improve serving economics via NVFP4 hot-tier efficiency
  and KV reuse/offload, not compression ratio alone.

Promotion Path:
  host blob → decompress if KVTC → restore to GPU → decode continues
  Cost paid once on reuse, not every token
```

### What this repo is NOT building

- A new inference engine
- A new distributed cache protocol
- A general-purpose KV compression framework
- A replacement for TensorRT-LLM, vLLM, or LMCache

---

## Bucket 3: Must Be Measured Here

These metrics must be produced by this repo to validate or reject the hypothesis. No result is accepted without these measurements.

### Primary metrics (every run)

| Metric | Description | Unit |
|--------|-------------|------|
| TTFT | Time to first token | ms (p50, p95) |
| TPOT | Time per output token | ms (p50, p95, p99) |
| Throughput | Output tokens per second | tok/s |
| Peak HBM | Maximum GPU memory allocated | GB |
| GPU power | Average power draw during run | W |
| Cache hit rate | Fraction of requests hitting prefix cache | 0.0-1.0 |
| Promotion latency | Time to restore offloaded KV to hot tier | ms (p50, p95) |
| Quality delta | Accuracy difference vs bf16 baseline | % |

### Serving-outcome metrics (tiered runs)

| Metric | Description |
|--------|-------------|
| Max concurrent sessions | At fixed latency target |
| Effective context length | Longest context that fits within HBM budget |
| TTFT improvement | On cache-hit vs cache-miss requests |
| HBM reduction | Peak HBM with tiering vs without |
| Secondary tier size | Bytes stored in host RAM |
| Offload latency | Time to move KV to secondary tier |

### Success thresholds (hackathon-grade)

| Target | Threshold |
|--------|-----------|
| Serving capacity improvement | ≥25% more concurrent sessions OR ≥20% lower peak HBM |
| TTFT improvement | Materially better on repeated-prefix traffic |
| p95 TPOT regression | ≤10% vs best non-tiered baseline |
| p99 TPOT regression | ≤15% vs best non-tiered baseline |
| Quality delta | ≤1% vs bf16 baseline on chosen eval |

### Protection policy ablations

| Dimension | Values to test |
|-----------|---------------|
| Promotion strategy | demand vs eager |
| Sink tokens protected | 0, 4 |
| Recent window size | 0, 64, 128, 256 |
| Secondary tier format | raw tensor vs KVTC (if codec ready) |

---

## Benchmark Ladder

Always run in this order. Do not skip steps.

| Step | Variant | Purpose | Gate |
|------|---------|---------|------|
| 1 | TRT-LLM BF16 / default KV | Accuracy ceiling, latency/memory floor | Always run |
| 2 | TRT-LLM FP8 KV | Stable hot-tier baseline | Always run |
| 3 | TRT-LLM NVFP4 KV | Primary Blackwell hot-tier thesis | Only if support gate passes |
| 4 | TRT-LLM NVFP4 + host offload | Secondary tier validation | After step 3 |
| 5 | TRT-LLM NVFP4 + offload + KVTC | Compressed secondary tier | Only if codec ready |
| 6 | vLLM FP8 KV (follow-up) | Compatibility comparison | If time permits |
| 7 | vLLM FP8 + LMCache (follow-up) | Productization comparison | If time permits |

### Accuracy benchmarks

For hackathon, use small subsets:
- One code benchmark (LiveCodeBench or MBPP subset)
- One long-context benchmark (Ruler 64K or LongBench subset)

---

## Implementation Notes

### Pre-RoPE vs Post-RoPE

**Explicit research question.** KVTC paper applies compression before RoPE for best quality. Many serving stacks only expose post-RoPE cached K tensors. If only post-RoPE KV is available:
- Prototype still works
- Expect a quality gap vs paper results
- Treat as explicit limitation in reporting

### KVTC calibration workflow

1. Capture representative prompts: chat/system prefixes, agent scaffolds, code prefixes, long-document prefixes
2. Run PCA calibration once per model (one-time cost)
3. Reuse PCA basis across compression ratios: 8x / 16x / 32x
4. Bit allocation changes via dynamic programming per target ratio
5. Validate quality on subset benchmarks at each ratio
6. Select operating point based on quality-vs-density tradeoff

### Stack priority

1. **Primary hackathon path:** TensorRT-LLM with NVFP4 hot KV and host offload
2. **Secondary tier first:** Prove offload helps with raw host-RAM before adding KVTC codec
3. **Do not couple risks:** Tiering hypothesis and compression hypothesis are independent. Test tiering first.
4. **Follow-up comparison:** vLLM + LMCache after primary TRT-LLM results are stable

---

## Rules

- Do not replace the inference engine
- Do not start with vLLM + LMCache as the primary hackathon path
- Do not use KVTC as the always-hot representation first
- Keep attention sink tokens protected
- Keep a recent-token window protected
- Treat pre-RoPE vs post-RoPE interception as explicit research question
- Pay decode/reconstruction cost on promotion, not every token
- Do not blur Blackwell-native behavior with Hopper emulation
- Do not claim breakthrough if it only beats strawman baseline
- Do not drop quality measurement because memory numbers look good
- Do not claim NVFP4 hot-KV support unless verified on the chosen stack
- Do not couple KVTC integration risk with tiering risk on day zero
- Do not use KVTC paper as evidence for TRT-LLM behavior
- Do not use NVIDIA NVFP4 docs as proof of vLLM hot-KV support

---

## Sources

- `[R1]` NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"
  - <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- `[R2]` NVIDIA, "5x Faster Time to First Token with TensorRT-LLM KV Cache Early Reuse"
  - <https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/>
- `[R3]` vLLM quantized KV cache documentation
  - <https://docs.vllm.ai/usage/quantization/quantized_kvcache/>
- `[R4]` vLLM production-stack KV cache documentation
  - <https://docs.vllm.ai/projects/production-stack/en/latest/user_manual/kv_cache/index.html>
- `[R5]` LMCache documentation
  - <https://docs.lmcache.ai/>
- `[R6]` KVTC, "KV-Cache Compression with Learned Transform Coding" (ICLR 2026)
  - <https://openreview.net/forum?id=tMiBQXQ0Cm>
- `[R7]` NVIDIA, "Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache"
  - <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
  - ModelOpt PTQ examples: <https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md>
- `[R8]` TensorRT-LLM KV Cache Reuse and KV Cache System
  - <https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html>
  - <https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html>
