# Blackwell KV Runtime: Tiered Architecture

## Core Thesis

We are not choosing between KVTC and NVFP4.

We are building:
- **Tier 0** hot KV in NVFP4 on Blackwell
- **Tier 1** warm/cold reusable KV in KVTC format
- **Promotion path:** KVTC decode → FP8 staging → NVFP4 active cache

## Goal

Reduce HBM pressure and preserve accuracy while improving TTFT, concurrency, and effective context on Blackwell.

---

## Tier Definitions

### Tier 0 — Hot Active KV (GPU HBM)

- Format: NVFP4 (native Blackwell SM120)
- Representation: E2M1 payload, FP8 E4M3 scale per 16-value micro-block, FP32 per-tensor second-level scale
- Dequantization: to FP8 before attention and context math
- Memory: roughly 50% reduction vs FP8 KV cache
- Quality: <1% accuracy loss documented on LiveCodeBench, MMLU-PRO, MBPP, and Ruler 64K
- Benefits: up to 3× lower TTFT, 20% higher cache hit rate vs FP8 in Blackwell studies

### Tier 1 — Warm/Cold Reusable KV (Host RAM / Local SSD / Remote)

- Format: KVTC bitstream
- Pipeline: PCA-based decorrelation → adaptive quantization via dynamic programming → entropy coding (nvCOMP DEFLATE)
- Compression: roughly 20× average vs 16-bit KV, 40×+ in some cases
- Calibration: PCA basis computed once per model on calibration set; same basis reused across 8× / 16× / 32× ratios with only bit allocation changing
- Quality: strong reasoning and long-context accuracy retention per KVTC paper (ICLR 2026)
- Source: https://openreview.net/forum?id=tMiBQXQ0Cm

### Promotion Path

```
Tier 1 (KVTC blob)
    │
    ▼ KVTC decode
FP8 staging buffer
    │
    ▼ NVFP4 pack (microscaling)
Tier 0 active blocks
    │
    ▼ Blackwell attention dequantizes NVFP4 → FP8
Attention / context computation
```

Promotion cost is paid once on reuse, not every token.

---

## Protection Policies

### Attention Sink Tokens
- First 4 tokens of every sequence are never compressed
- These carry disproportionate attention mass

### Recent Token Window
- Final 128 tokens (configurable) stay in Tier 0 uncompressed
- Live decode tail is never KVTC-compressed
- KVTC ablations show this improves quality

### What to KVTC-Compress
- Reused prefixes (system prompts, chat templates)
- Stale conversation turns
- Agent scaffolding tokens
- Cached document prefixes
- Long-context KV that has aged past the recent window

---

## Pre-RoPE vs Post-RoPE

**Explicit research question.** KVTC paper applies compression before RoPE for best quality. Many serving stacks only expose post-RoPE cached K tensors. If only post-RoPE KV is available:
- Prototype still works
- Expect a quality gap vs paper results
- Treat as explicit limitation in reporting

---

## Implementation Paths

### Hackathon Path (fastest)

**TensorRT-LLM + ModelOpt NVFP4-KV + custom KV connector**

1. Use ModelOpt combined quantization recipe:
   ```python
   import modelopt.torch.quantization as mtq

   quant_cfg = mtq.FP8_DEFAULT_CFG
   quant_cfg["quant_cfg"].update(mtq.NVFP4_KV_CFG["quant_cfg"])

   def forward_loop(model):
       for data in calib_set:
           model(data)

   model = mtq.quantize(model, quant_cfg, forward_loop)
   ```
2. Deploy checkpoint with TensorRT-LLM
3. Use TensorRT-LLM KV Cache Connector for persistence/reuse
4. Write custom connector that saves/loads KVTC blobs for stale reusable prefixes

### Product Path (LMCache integration)

**LMCache manages host/disk/remote tiers with KVTC codec**

1. Leave active Blackwell KV in NVFP4 inside the serving engine
2. Add KVTC as new compression/serde method in LMCache
3. LMCache already has: pinned CPU RAM as hot host cache, async loading for I/O overlap, remote_serde compression hook (CacheGen today), controller APIs for compress/decompress
4. Replace CacheGen with KVTC codec for higher compression density
5. Promotion on prefix hit or idle-session reuse triggers KVTC decode → FP8 → NVFP4

### vLLM Compatibility Note

vLLM KV-cache quantization today is centered on FP8 (with llm-compressor calibration). vLLM does support ModelOpt NVFP4 checkpoints, but public KV docs are FP8-focused. For clean NVFP4-KV demo, TensorRT-LLM is safer. For LMCache integration, use vLLM FP8 as baseline and treat NVFP4 as separate compatibility track.

---

## Benchmark Ladder

Always run in this order:

| Step | Variant | Purpose |
|------|---------|---------|
| 1 | BF16 / default KV | Accuracy ceiling, latency/memory floor |
| 2 | FP8 KV | First low-precision baseline |
| 3 | NVFP4 KV | Native Blackwell hot-tier baseline |
| 4 | Raw host offload | Offload baseline (LMCache or manual) |
| 5 | NVFP4 hot + KVTC warm/cold | Tiered system (the thesis) |

### Metrics (every run must record)

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token |
| p50 TPOT | Median time per output token |
| p95 TPOT | Tail time per output token |
| throughput | Tokens per second |
| HBM footprint | Peak GPU memory |
| cache-hit rate | Prefix cache hit percentage |
| promotion latency | Time to promote KVTC → NVFP4 |
| accuracy delta | Quality vs best higher-precision baseline |

### Accuracy Benchmarks

For hackathon, use small subsets, not full sweeps:
- One code benchmark (LiveCodeBench or MBPP subset)
- One long-context benchmark (Ruler 64K or LongBench subset)

---

## KVTC Calibration Workflow

1. **Capture representative prompts**: chat/system prefixes, agent scaffolds, code prefixes, long-document prefixes
2. **Run PCA calibration** once per model (one-time cost)
3. **Reuse PCA basis** across compression ratios: 8× / 16× / 32×
4. **Bit allocation** changes via dynamic programming per target ratio
5. **Validate** quality on subset benchmarks at each ratio
6. **Select** operating point based on quality-vs-density tradeoff

---

## Success Criteria

We win only if NVFP4 + KVTC beats raw offload or plain NVFP4 on:
- HBM efficiency
- TTFT / hit-rate
- or concurrency

**without breaking quality.**

---

## Rules

- Do not replace the inference engine
- Do not use KVTC as the always-hot representation first
- Keep attention sink tokens protected
- Keep a recent-token window protected
- Treat pre-RoPE vs post-RoPE interception as explicit research question
- Pay decode/reconstruction cost on promotion, not every token
- Do not blur Blackwell-native behavior with Hopper emulation
- Do not claim breakthrough if it only beats strawman baseline
- Do not drop quality measurement because memory numbers look good

---

## Sources

- R1: NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"
- R2: NVIDIA, "5x Faster Time to First Token with TensorRT-LLM KV Cache Early Reuse"
- R3: vLLM quantized KV cache documentation
- R4: vLLM production-stack KV cache documentation
- R5: LMCache documentation (https://docs.lmcache.ai/)
- R6: KVTC, "KV-Cache Compression with Learned Transform Coding" (https://openreview.net/forum?id=tMiBQXQ0Cm)
- R7: NVIDIA ModelOpt NVFP4-KV quantization documentation
- R8: TensorRT-LLM KV Cache Connector documentation
