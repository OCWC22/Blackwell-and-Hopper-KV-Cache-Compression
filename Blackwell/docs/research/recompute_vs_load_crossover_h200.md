# Recompute vs Load Crossover Analysis — Qwen3-30B-A3B on H200

**Date:** 2026-03-15
**GPU:** NVIDIA H200 (140 GB HBM3e)
**Model:** Qwen3-30B-A3B-Instruct-2507 (48 layers, 4 KV heads, head_dim 64, 128 experts MoE)
**Block size:** 32 tokens
**Max context tested:** 131,072 tokens (128K)

---

## Motivation

When a KV block is evicted from GPU memory, there are two ways to recover it on a cache miss:

1. **Recompute** — re-run the forward pass for those tokens (requires all prior KV to be present)
2. **Load from host** — transfer the offloaded block back over PCIe

The cost tradeoff depends on context position, GPU saturation, and PCIe contention. We measured both curves under varying concurrency to find the crossover point that determines the offload policy.

---

## Methodology

### Recompute cost (`sweep_recompute_cost.py`)

For each (context_position, batch_size) pair:
1. Build full KV cache up to `context_position` (batched)
2. Measure wall-clock time to compute the next 32-token block incrementally
3. Report **per-block amortized cost** = total_time / batch_size

Batch size simulates GPU saturation: at B=1 the GPU is underutilized; at B=32 it approaches realistic serving load.

### Load cost (`sweep_load_cost.py`)

For each (dtype, concurrent_blocks) pair:
1. Allocate N pinned host buffers + N device buffers, each sized to one full KV block (all 48 layers, K+V)
2. Issue all N H2D copies on a dedicated copy stream
3. Measure total wall-clock time and report **per-block amortized cost** = total_time / N

Concurrent blocks simulates PCIe queuing: at C=1 the bus is latency-bound; at C=128 it saturates.

Background matmul load (4096x4096 FP16) ran continuously during all load measurements to simulate realistic GPU compute contention.

---

## Results

### Recompute cost: per-block amortized (ms)

| Context Position | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|---|
| 32 | 450 | 269 | 162 | 91 | 51 | 31 |
| 128 | 426 | 264 | 152 | 87 | 47 | 26 |
| 512 | 445 | 268 | 151 | 84 | 45 | 26 |
| 2,048 | 433 | 257 | 153 | 85 | 48 | 28 |
| 8,192 | 437 | 265 | 161 | 94 | 58 | OOM |
| 32,768 | 505 | 315 | 196 | OOM | — | — |
| 131,072 | 862 | OOM | — | — | — | — |

Key observations:
- Per-block cost scales roughly as 1/B (near-perfect batching efficiency up to B=16)
- Context position has moderate effect at short contexts, strong effect at 32K+ (attention cost dominates)
- OOM limits max batch size at long contexts (KV cache memory pressure)
- **Floor:** ~26 ms per block at B=32, ctx=512

### Load cost: per-block amortized (ms)

**FP16 (1.5 MB per block):**

| Concurrent Blocks | Per-block ms | Total ms | Aggregate BW (GB/s) |
|---|---|---|---|
| 1 | 0.048 | 0.048 | 32.6 |
| 2 | 0.040 | 0.079 | 39.8 |
| 4 | 0.035 | 0.142 | 44.4 |
| 8 | 0.033 | 0.265 | 47.5 |
| 16 | 0.032 | 0.516 | 48.8 |
| 32 | 0.032 | 1.017 | 49.5 |
| 64 | 0.032 | 2.027 | 49.7 |
| 128 | 0.032 | 4.052 | 49.7 |

**FP8 (0.75 MB per block):**

| Concurrent Blocks | Per-block ms | Total ms | Aggregate BW (GB/s) |
|---|---|---|---|
| 1 | 0.034 | 0.034 | 22.9 |
| 2 | 0.026 | 0.053 | 29.8 |
| 4 | 0.022 | 0.086 | 36.5 |
| 8 | 0.019 | 0.153 | 41.0 |
| 16 | 0.018 | 0.289 | 43.6 |
| 32 | 0.018 | 0.563 | 44.7 |
| 64 | 0.018 | 1.118 | 45.0 |
| 128 | 0.017 | 2.215 | 45.5 |

Key observations:
- PCIe saturates at ~50 GB/s (FP16) / ~45 GB/s (FP8) around 16-32 concurrent blocks
- Per-block cost floor: **0.032 ms** (FP16) / **0.017 ms** (FP8)
- Even at C=1 (worst case), load cost is 0.048 ms (FP16) / 0.034 ms (FP8)

---

## Crossover Analysis

### Gap at every operating point

| Operating Point | Recompute (ms) | Load (ms) | Ratio |
|---|---|---|---|
| Best case recompute (B=32, ctx=512) | 25.6 | — | — |
| Best case load (FP8, C=128) | — | 0.017 | — |
| **Best recompute vs worst load** | 25.6 | 0.048 | **533x** |
| **Worst recompute vs best load** | 862 | 0.017 | **50,700x** |
| Typical serving (B=8, ctx=8K) vs (FP8, C=16) | 94 | 0.018 | **5,200x** |

### Crossover matrix

Every cell in the (batch_size × concurrency) matrix is **LOAD wins**. No crossover exists for this model on this hardware.

```
           C=1    C=8    C=32   C=128
  B=1     LOAD   LOAD   LOAD   LOAD
  B=4     LOAD   LOAD   LOAD   LOAD
  B=16    LOAD   LOAD   LOAD   LOAD
  B=32    LOAD   LOAD   LOAD   LOAD
```

---

## Policy Conclusion

**For Qwen3-30B-A3B on H200: always offload, never recompute.**

The minimum recompute cost (~26 ms at B=32, short context) is still ~800x more expensive than the maximum load cost (~0.032 ms at C=1, FP16). The gap is structural:

- **Recompute** requires a full forward pass through 48 transformer layers with MoE expert routing — even for just 32 tokens, this involves billions of FLOPs.
- **Load** transfers 0.75–1.5 MB over PCIe — a fraction of the bus's 50 GB/s capacity.

### When might a crossover exist?

The crossover would require recompute cost to drop below ~0.05 ms per block. This might occur with:

1. **Very small models** (1-3B dense) where a single forward pass is sub-millisecond
2. **Dedicated recompute hardware** (e.g., a separate small model that approximates KV)
3. **Extremely degraded PCIe** (shared bus with NVMe, multi-GPU traffic, broken lanes)
4. **Very large blocks** where transfer size grows but recompute stays cheap (unlikely — both scale with block size)

For any model large enough to benefit from KV offloading in the first place, loading wins.

### Implications for the tiered architecture

This validates the offload-first design in `TIERED_KV_ARCHITECTURE.md`:

- **Tier 0 (hot GPU):** NVFP4/FP8 KV cache — active decode path
- **Tier 1 (host RAM):** offloaded KV blocks — restore on demand
- **Policy:** always offload evicted blocks; never recompute; restore via H2D on cache hit
- **Optimization focus:** minimize restore *latency* (layer-pipelined restore via `wait_for_layer_load`), not whether to restore at all
- **Compression (KVTC):** reduces Tier 1 storage and transfer bytes, but adds GPU-side decompress cost; the transfer savings are small relative to the recompute gap, so KVTC's value is in *capacity* (fitting more blocks in host RAM), not in making load faster than recompute

---

## Reproduction

```bash
# Load cost sweep (fast, ~30s)
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_load_cost.py \
    --num-layers 48 --num-kv-heads 4 --head-dim 64 \
    --block-tokens 32 --dtypes fp16,fp8 \
    --concurrent-blocks 1,2,4,8,16,32,64,128 \
    --with-compute-load --per-layer \
    --output results/load_curve_h200_block32_conc.json

# Recompute cost sweep (~10 min)
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_recompute_cost.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --engine torch \
    --kv-mode bf16 --block-size 32 \
    --positions 32,128,512,2048,8192,32768,131072 \
    --batch-sizes 1,2,4,8,16,32 \
    --warmup-iters 1 --measure-iters 3 \
    --max-seq-len 131104 \
    --output results/recompute_curve_h200_bf16_block32_batched.json

# Crossover analysis
python scripts/plot_crossover.py \
    --recompute results/recompute_curve_h200_bf16_block32_batched.json \
    --load results/load_curve_h200_block32_conc.json \
    --block-size 32 \
    --output results/crossover_plot_h200_block32_saturated.png \
    --output-json results/crossover_policy_h200_block32_saturated.json
```

---

## Data Files

- `results/recompute_curve_h200_bf16_block32_batched.json` — full recompute sweep
- `results/load_curve_h200_block32_conc.json` — full load sweep with concurrency
- `results/crossover_policy_h200_block32_saturated.json` — crossover analysis output
