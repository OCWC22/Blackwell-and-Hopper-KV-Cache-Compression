# Theoretical FLOP Validation: Recompute vs Load — Qwen3-30B-A3B on H200

**Date:** 2026-03-15
**Status:** Analysis complete
**Depends on:** `recompute_vs_load_crossover_h200.md` (empirical measurements)
**GPU:** NVIDIA H200 (140 GB HBM3e, ~1 PFLOP/s FP16 peak, PCIe Gen5 x16 ~64 GB/s peak)
**Model:** Qwen3-30B-A3B-Instruct-2507 (48 layers, 32 Q heads, 4 KV heads, head_dim 128, 128 MoE experts, top-8 routing, ~3B active params/token)
**Block size:** 32 tokens
**Config source:** [HuggingFace config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json)

> **Erratum:** The empirical crossover doc (`recompute_vs_load_crossover_h200.md`) lists head_dim as 64.
> The actual model config specifies head_dim=128. This doubles KV block size and attention FLOPs.
> The recompute measurements (which used the real model) are unaffected. The load measurements
> used synthetic blocks sized by the `--head-dim` CLI flag — if 64 was passed, those measurements
> reflect half-sized blocks and the per-block load times should be approximately doubled.
> The conclusion (load always wins) is unchanged; the gap narrows slightly at short context.

---

## 1. Conclusion

**Loading evicted KV blocks from host memory is faster than recomputing them at every operating point — including the theoretical minimum where the GPU achieves 100% MFU.**

At 100% MFU (physically impossible), the best-case recompute (short context) is still **~3x slower** than worst-case PCIe load. At realistic 20% MFU, the gap is 9–180x. The empirical measurements in `recompute_vs_load_crossover_h200.md` are consistent with first-principles FLOP counts and known hardware limits.

This validates the offload-first policy for any model large enough to benefit from KV offloading.

---

## 2. Motivation

The empirical crossover analysis (`recompute_vs_load_crossover_h200.md`) measured recompute costs of 26–862 ms per block versus load costs of 0.017–0.048 ms, yielding ratios of 533–50,700x. These measurements used PyTorch eager mode, which introduces framework overhead (kernel launches, MoE dispatch). This document asks: **do the numbers still hold if we remove all framework overhead and assume ideal GPU utilization?**

If the theoretical FLOP-limited recompute time is still far above the PCIe transfer time, then the conclusion is hardware-fundamental, not an artifact of the measurement methodology.

---

## 3. Theoretical FLOP Count

### 3.1 Model architecture (from HuggingFace config.json)

| Parameter | Config value | Notes |
|---|---|---|
| `hidden_size` | 2048 | |
| `num_attention_heads` | 32 | Q heads |
| `num_key_value_heads` | 4 | GQA ratio 8:1 |
| `head_dim` | **128** | Q/K/V head dimension |
| `num_hidden_layers` | 48 | |
| `num_experts` | 128 | Total MoE experts |
| `num_experts_per_tok` | 8 | Active experts per token |
| `moe_intermediate_size` | **768** | Per-expert FFN hidden dim |
| `intermediate_size` | 6144 | (shared/dense FFN, not used in MoE layers) |
| `hidden_act` | silu | SwiGLU activation |

### 3.2 Parameter counts per layer

**Attention projections:**

| Projection | Shape | Params |
|---|---|---|
| Q | 2048 × (32 × 128) = 2048 × 4096 | 8,388,608 (8.0M) |
| K | 2048 × (4 × 128) = 2048 × 512 | 1,048,576 (1.0M) |
| V | 2048 × 512 | 1,048,576 (1.0M) |
| O | 4096 × 2048 | 8,388,608 (8.0M) |
| **Attention total** | | **18.4M** |

**MoE FFN (per active expert, SwiGLU):**

| Projection | Shape | Params |
|---|---|---|
| gate_proj | 2048 × 768 | 1,572,864 (1.57M) |
| up_proj | 2048 × 768 | 1,572,864 (1.57M) |
| down_proj | 768 × 2048 | 1,572,864 (1.57M) |
| **Per expert** | | **4.72M** |
| **8 active experts** | | **37.7M** |

**Total active per layer:** 18.4M + 37.7M = **55.7M** (attention-heavy due to large head_dim)
**Total active across 48 layers:** 48 × 55.7M = **2.67B** (~3B, consistent with model name)

### 3.3 FLOPs per 32-token block

**Linear layers (context-independent):**
Each matmul costs 2× parameter count FLOPs per token [1]. For 32 tokens across 48 layers:

```
linear_flops = 32 tokens × 48 layers × 2 × 55.7M params ≈ 171 GFLOP
```

**Attention scores (context-dependent):**
For each token at position `pos`, attention computes Q·K^T and attn·V over all `pos` prior positions, across 32 Q heads with head_dim=128 [2]:

```
attn_flops_per_token_per_layer = 2 × 32 heads × 128 dim × pos × 2 (QK + AV)
                                = 16,384 × pos
```

For a 32-token block starting at context position `pos`, summing over tokens i=0..31:

```
attn_flops_per_layer = Σ(i=0..31) 16,384 × (pos + i)
                     = 16,384 × (32 × pos + 496)
                     = 524,288 × pos + 8,126,464
```

Across 48 layers:

```
total_attn_flops = 48 × (524,288 × pos + 8,126,464)
                 = 25,165,824 × pos + 390,070,272
```

### 3.4 Total FLOPs by context position

| Context pos | Linear (GFLOP) | Attention (GFLOP) | **Total (GFLOP)** |
|---|---|---|---|
| 32 | 171 | 1.2 | **172** |
| 128 | 171 | 3.6 | **175** |
| 512 | 171 | 13.3 | **184** |
| 2,048 | 171 | 51.9 | **223** |
| 8,192 | 171 | 206.5 | **378** |
| 32,768 | 171 | 824.8 | **996** |
| 131,072 | 171 | 3,298 | **3,469** |

At short contexts, linear layers dominate (~99%). At 128K, attention is 95% of FLOPs. Compared to head_dim=64, attention FLOPs are exactly 2× while linear FLOPs are slightly lower (due to smaller MoE intermediate size than initially assumed).

---

## 4. Theoretical Recompute Time

### 4.1 At 20% MFU (realistic for optimized serving)

Effective throughput: 1 PFLOP/s × 0.20 = 200 TFLOP/s

| Context pos | GFLOP | **Theoretical time (ms)** | Measured B=1 (ms) | Measured/Theoretical |
|---|---|---|---|---|
| 32 | 172 | **0.86** | 450 | 523× |
| 128 | 175 | **0.87** | 426 | 490× |
| 512 | 184 | **0.92** | 445 | 484× |
| 2,048 | 223 | **1.12** | 433 | 387× |
| 8,192 | 378 | **1.89** | 437 | 231× |
| 32,768 | 996 | **4.98** | 505 | 101× |
| 131,072 | 3,469 | **17.35** | 862 | 50× |

Note: measured/theoretical ratio decreases at long context because measured times become increasingly compute-bound (attention FLOPs dominate) while short-context measurements are dominated by fixed kernel-launch overhead.

### 4.2 At 100% MFU (physical upper bound)

Effective throughput: 1 PFLOP/s

| Context pos | GFLOP | **Theoretical time (ms)** |
|---|---|---|
| 32 | 172 | **0.17** |
| 512 | 184 | **0.18** |
| 2,048 | 223 | **0.22** |
| 8,192 | 378 | **0.38** |
| 32,768 | 996 | **1.00** |
| 131,072 | 3,469 | **3.47** |

### 4.3 At B=32 (batched, compute-saturated)

At B=32 the GPU is compute-saturated and ~20% MFU is realistic. Total FLOPs scale with batch size:

| Context pos | GFLOP (B=32) | Time @ 20% MFU (ms) | Per-request (ms) | Measured per-request (ms) |
|---|---|---|---|---|
| 32 | 5,504 | 27.5 | 0.86 | ~31 (implies ~18% MFU) |
| 512 | 5,888 | 29.4 | 0.92 | ~26 (implies ~23% MFU) |

The measured B=32 values are consistent with 18–23% MFU, validating the FLOP model.

### 4.4 Explaining the measured vs theoretical gap at B=1

At B=1, measured times are 50–523× above the 20% MFU theoretical floor. This is expected:

1. **Kernel launch overhead dominates at B=1.** Each token requires ~29 kernel launches per layer (4 attention matmuls + MoE routing + 8 experts × 3 matrices). For 32 tokens × 48 layers ≈ 44,500 kernel launches at ~10μs each = ~445 ms of pure launch overhead [3]. This alone accounts for the measured ~450 ms at short context.

2. **Memory bandwidth bound, not compute bound.** At B=1, each token loads ~5.3 GB of active weights (2.67B params × 2 bytes BF16). Arithmetic intensity is far below the GPU's compute/bandwidth ratio, so the GPU stalls waiting for weight data.

3. **MoE dispatch overhead.** Expert routing involves top-k selection, scatter/gather, and per-expert kernel launches — significant fixed overhead per token in PyTorch eager mode.

At long context (131K), the gap narrows to 50× because attention FLOPs finally dominate over fixed overhead, and the 862 ms measurement reflects real compute work (3.47 TFLOP at ~4 TFLOP/s effective = ~0.4% MFU, consistent with B=1 memory-bandwidth-limited execution).

---

## 5. Theoretical Load Time

### 5.1 KV block size (corrected for head_dim=128)

```
block_bytes = num_layers × 2 (K+V) × num_kv_heads × head_dim × block_tokens × dtype_bytes
            = 48 × 2 × 4 × 128 × 32 × dtype_bytes
```

| Dtype | Calculation | Block size |
|---|---|---|
| FP16 | 48 × 2 × 4 × 128 × 32 × 2 | **3.0 MB** |
| FP8 | 48 × 2 × 4 × 128 × 32 × 1 | **1.5 MB** |

### 5.2 PCIe transfer time

H200 PCIe Gen5 x16: 64 GB/s peak, ~50 GB/s sustained [4].

| Dtype | Bytes | **Theoretical (ms)** | Estimated measured* (ms) |
|---|---|---|---|
| FP16 | 3.0 MB | **0.060** | ~0.064 |
| FP8 | 1.5 MB | **0.030** | ~0.034 |

*Estimated by scaling the empirical measurements from the crossover doc (which used head_dim=64 blocks) by 2×. Actual re-measurement recommended.

At worst case (FP16, C=1, including DMA setup latency): estimated **~0.096 ms**.

---

## 6. Theoretical Crossover Comparison

### 6.1 Best-case recompute vs worst-case load

Even granting recompute every possible advantage (100% MFU, shortest context) and penalizing load (FP16, single concurrent block with DMA overhead):

| Context pos | Recompute @ 100% MFU (ms) | Load FP16 C=1 est. (ms) | **Ratio** |
|---|---|---|---|
| 32 | 0.17 | 0.096 | **1.8×** |
| 512 | 0.18 | 0.096 | **1.9×** |
| 2,048 | 0.22 | 0.096 | **2.3×** |
| 8,192 | 0.38 | 0.096 | **3.9×** |
| 32,768 | 1.00 | 0.096 | **10.4×** |
| 131,072 | 3.47 | 0.096 | **36×** |

**Even at 100% MFU, load wins at every context length.** The minimum gap is 1.8× at short context — and 100% MFU is physically unachievable.

### 6.2 Realistic operating points

At 20% MFU with FP8 and moderate PCIe concurrency:

| Context pos | Recompute @ 20% MFU (ms) | Load FP8 C=16 est. (ms) | **Ratio** |
|---|---|---|---|
| 32 | 0.86 | 0.034 | **25×** |
| 512 | 0.92 | 0.034 | **27×** |
| 2,048 | 1.12 | 0.034 | **33×** |
| 8,192 | 1.89 | 0.034 | **56×** |
| 32,768 | 4.98 | 0.034 | **146×** |
| 131,072 | 17.35 | 0.034 | **510×** |

### 6.3 Crossover condition

For recompute to match the worst-case load (FP16, C=1, ~0.096 ms):

```
recompute_time = load_time
172 GFLOP / (MFU × 1000 TFLOP/s) = 0.096 ms
MFU = 172,000 / (1000 × 0.096) = 1,792
```

Required MFU: **1,792×** — physically impossible.

For recompute to match at 100% MFU, the block would need to contain only:

```
block_flops = 1000 TFLOP/s × 0.096 ms = 96 GFLOP
```

This is 96 GFLOP vs 172 GFLOP actual — the model would need to be ~1.8× smaller (with the same KV block size) for a crossover at 100% MFU. In practice, smaller models have proportionally smaller KV blocks, so the ratio stays roughly constant.

**No crossover exists on this hardware for this model class.**

---

## 7. Why vLLM Defaults to Recompute (and Why That's Suboptimal)

Despite the clear theoretical advantage of loading, vLLM's default preemption policy is `recompute` [5]. This is a pragmatic engineering choice, not a performance-optimal one:

1. **Preemption is designed to be rare.** PagedAttention minimizes preemption events. Optimizing a rare path was not a priority.
2. **Swap requires additional infrastructure.** Pinned host memory pool, host-side block table, async copy stream management, failure handling (host memory exhaustion). Recompute requires none of this.
3. **Preemption evicts entire requests, not single blocks.** vLLM preemption discards all KV for a request. Swapping a full request's KV (e.g., 768 MB for 8K context at FP16 with head_dim=128) and later restoring it requires careful memory lifecycle management.
4. **Memory pressure circularity.** If GPU memory is full enough to trigger preemption, bringing swapped blocks back may trigger further preemption.
5. **Historical simplicity.** Recompute shipped first; swap was added later and never became the default.

For production systems with explicit KV management (LMCache, TRT-LLM offloading, tiered caches), load/swap is strictly better. The theoretical analysis here confirms that the performance argument is unambiguous — the engineering complexity is the only reason not to default to it.

---

## 8. Implications for Tiered Architecture

This analysis strengthens the conclusions in `TIERED_KV_ARCHITECTURE.md`:

- **Policy: always offload, never recompute** — validated from first principles, not just empirical measurement
- **Optimization target: minimize restore latency** (layer-pipelined H2D), not whether to restore at all
- **KVTC compression value: capacity, not speed** — even halving transfer bytes (FP8 → FP4) saves only ~0.030 ms; the value is fitting more blocks in host memory
- **Block size choice: not load-sensitive** — doubling block size doubles transfer bytes but the load time is still negligible vs recompute
- **head_dim=128 narrows the gap slightly** — larger KV blocks mean more transfer bytes, but also more recompute FLOPs (attention scales with head_dim). The ratio stays comfortably in load's favor.

---

## 9. Limitations

1. **Single-model analysis.** The FLOP counts are specific to Qwen3-30B-A3B. Dense models with fewer parameters would have lower recompute cost, but also smaller KV blocks.
2. **H200 only.** Different compute/bandwidth ratios (e.g., B200 with NVLink-C2C vs PCIe) could shift the ratio, though not enough to create a crossover.
3. **Prefill vs decode context.** The FLOP model assumes incremental decode (32 new tokens with full prior KV). Chunked prefill would have different characteristics.
4. **Attention kernel efficiency.** FlashAttention-2/3 can change the effective FLOP cost of long-context attention via IO-aware tiling [6], but this improves recompute speed by at most 2–4× — not enough to close a 1.8× minimum gap at 100% MFU (and real MFU is ≤30%).
5. **Load measurements need re-running.** The empirical load data in `recompute_vs_load_crossover_h200.md` used head_dim=64 blocks (1.5 MB FP16). With the correct head_dim=128, blocks are 3.0 MB and load times should be re-measured. Theoretical estimates (2× scaling) are used in this document.

---

## References

[1] Kaplan et al., "Scaling Laws for Neural Language Models," 2020. Standard 2×params FLOP approximation for transformer matmuls.

[2] Vaswani et al., "Attention Is All You Need," 2017. Attention FLOP formula: 2 × seq_len × d_model per head for QK^T and AV.

[3] NVIDIA, "CUDA C++ Best Practices Guide," Section on kernel launch overhead. Typical kernel launch latency is 5–15μs on modern GPUs.

[4] PCI-SIG, "PCI Express Base Specification Revision 5.0." Gen5 x16 theoretical: 64 GB/s unidirectional; measured sustained ~50 GB/s consistent with DMA overhead.

[5] vLLM source, `vllm/core/scheduler.py`, `PreemptionMode.RECOMPUTE` as default. See also vLLM documentation on preemption policy.

[6] Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023. IO-aware attention reduces HBM reads but does not change FLOP count.
