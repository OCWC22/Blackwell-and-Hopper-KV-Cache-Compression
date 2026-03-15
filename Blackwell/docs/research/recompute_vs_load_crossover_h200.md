# Recompute vs Load Crossover Analysis — Qwen3-30B-A3B on H200

**Date:** 2026-03-15
**GPU:** NVIDIA H200 (140 GB HBM3e)
**Model:** Qwen3-30B-A3B-Instruct-2507 (48 layers, 4 KV heads, head_dim 64, 128 experts MoE)
**Block sizes tested:** 32 tokens, 512 tokens
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

Batch size simulates GPU saturation: at B=1 the GPU is underutilized; at B=1024 it approaches realistic serving load where the MoE experts and attention are fully utilized.

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

**Low batch (B=1 to B=32):**

| Context Position | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|---|
| 32 | 450 | 269 | 162 | 91 | 51 | 31 |
| 128 | 426 | 264 | 152 | 87 | 47 | 26 |
| 512 | 445 | 268 | 151 | 84 | 45 | 26 |
| 2,048 | 433 | 257 | 153 | 85 | 48 | 28 |
| 8,192 | 437 | 265 | 161 | 94 | 58 | OOM |
| 32,768 | 505 | 315 | 196 | OOM | — | — |
| 131,072 | 862 | OOM | — | — | — | — |

**High batch (B=32 to B=1024) — GPU approaching saturation:**

| Context Position | B=32 | B=64 | B=128 | B=256 | B=512 | B=1024 |
|---|---|---|---|---|---|---|
| 32 | 32.1 | 17.8 | 10.2 | 5.6 | 3.4 | **2.3** |
| 128 | 25.8 | 14.3 | 7.9 | 4.9 | **3.3** | OOM |
| 512 | 25.7 | 14.3 | 8.6 | **5.6** | OOM | — |
| 2,048 | 28.0 | **17.1** | OOM | — | — | — |
| 8,192 | OOM | — | — | — | — | — |

Key observations:
- Per-block cost continues to drop roughly as 1/B up to the OOM limit
- At B=1024, ctx=32: **2.33 ms per block** — the GPU is finally well-utilized
- At B=512, ctx=128: **3.31 ms per block**
- OOM is the binding constraint, not GPU compute saturation — the model's KV cache fills HBM before the SMs are fully loaded
- At long contexts (8K+), even B=32 OOMs because KV cache for 32 sequences × 8K tokens exceeds remaining HBM
- **True compute floor is likely sub-1ms** but unreachable due to memory limits on a single GPU

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

### 512-token blocks

Larger blocks increase the load cost (more bytes) but also increase the recompute work per block. This tests whether the ratio changes.

#### Recompute cost: per-block amortized (ms), block=512 tokens

| Context Position | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---|---|---|---|---|---|---|
| 512 | 825 | 478 | 278 | 162 | 91 | 59 | **42** |
| 1,024 | 746 | 412 | 229 | 133 | 82 | 57 | **43** |
| 2,048 | 751 | 418 | 232 | 137 | 87 | **61** | OOM |
| 4,096 | 745 | 425 | 241 | 145 | 96 | **70** | OOM |
| 8,192 | 750 | 441 | 256 | 163 | **115** | OOM | — |
| 16,384 | 793 | 476 | 292 | **199** | OOM | — | — |
| 32,768 | 860 | 548 | **364** | OOM | — | — | — |
| 65,536 | 997 | **685** | OOM | — | — | — | — |
| 131,072 | **1295** | OOM | — | — | — | — | — |

OOM hits earlier than with 32-token blocks because each block's KV is 16x larger.

#### Load cost: per-block (ms), block=512 tokens

| Dtype | Block size | C=1 | C=8 | C=32 | C=128 | BW at saturation |
|---|---|---|---|---|---|---|
| FP16 | 24.0 MB | 0.471 | 0.460 | 0.459 | 0.459 | 54.9 GB/s |
| FP8 | 12.0 MB | 0.243 | 0.234 | 0.232 | 0.231 | 54.5 GB/s |

At 512-token blocks the transfers are throughput-bound from C=1 — PCIe is already saturated at ~55 GB/s. Concurrency barely matters.

#### Comparison: 32-token vs 512-token blocks

| Metric | Block=32 | Block=512 | Ratio |
|---|---|---|---|
| **Load FP16 (C=1)** | 0.048 ms | 0.471 ms | 9.8x (tracks 16x size) |
| **Load FP8 (C=1)** | 0.034 ms | 0.243 ms | 7.1x |
| **Recompute floor** | 2.3 ms (B=1024) | 42 ms (B=64) | 18x |
| **Best recompute / worst load** | 48x | 89x | — |

Larger blocks make loading *relatively* more expensive (closer to BW-limited), but recompute also gets more expensive because the forward pass now covers 512 tokens of incremental attention. The ratio stays firmly in load's favor.

---

## Crossover Analysis

### Gap across operating points

| Operating Point | Recompute (ms) | Load (ms) | Ratio |
|---|---|---|---|
| Low batch (B=1, ctx=32) vs idle PCIe | 450 | 0.048 | **9,375x** |
| Mid batch (B=32, ctx=512) vs saturated PCIe | 25.7 | 0.032 | **803x** |
| High batch (B=512, ctx=128) vs saturated PCIe | 3.3 | 0.032 | **103x** |
| **Best recompute (B=1024, ctx=32) vs worst load (FP16, C=1)** | **2.3** | **0.048** | **48x** |
| **Best recompute (B=1024, ctx=32) vs best load (FP8, C=128)** | **2.3** | **0.017** | **135x** |
| Typical serving (B=8, ctx=8K) vs (FP8, C=16) | 94 | 0.018 | **5,222x** |

### Scaling trend

The per-block recompute cost drops roughly as 1/B:

```
B=1:    ~450 ms
B=32:   ~26 ms    (17x from B=1)
B=128:  ~8 ms     (56x from B=1)
B=512:  ~3.3 ms   (136x from B=1)
B=1024: ~2.3 ms   (196x from B=1)
```

Extrapolating: to reach 0.03 ms (load parity), you'd need B≈15,000 at short context — physically impossible on a single H200 due to HBM limits. The per-block recompute cost is bounded below by the sequential latency of the 48-layer forward pass even with perfect parallelism.

### Crossover matrix

Every cell is **LOAD wins**. No crossover exists for this model on this hardware.

```
              C=1    C=8    C=32   C=128
  B=1        LOAD   LOAD   LOAD   LOAD
  B=32       LOAD   LOAD   LOAD   LOAD
  B=128      LOAD   LOAD   LOAD   LOAD
  B=512      LOAD   LOAD   LOAD   LOAD
  B=1024     LOAD   LOAD   LOAD   LOAD
```

---

## Policy Conclusion

**For Qwen3-30B-A3B on H200: always offload, never recompute.**

Even at the most aggressive achievable batch size (B=1024, short context), recompute costs 2.3 ms per block — still **48x** more expensive than the worst-case load (0.048 ms, FP16, C=1). The gap is structural:

- **Recompute** requires a sequential forward pass through 48 transformer layers with MoE expert routing. Even with perfect batch parallelism, the latency is bounded by the serial depth of the model. At B=1024 the GPU is well-utilized but each block still takes 2.3ms because the computation is irreducibly deep.
- **Load** transfers 0.75–1.5 MB over PCIe — a sub-microsecond operation at line rate. The H200's PCIe Gen5 x16 delivers ~50 GB/s sustained.
- **OOM is the binding constraint**: before the GPU is fully saturated, HBM fills with KV cache. At ctx=8K, even B=32 OOMs. This means the high-batch regime where recompute gets cheap is exactly the regime where you can't fit the KV cache anyway — making offload even more necessary, not less.

### When might a crossover exist?

The crossover would require recompute cost to drop below ~0.05 ms per block. This might occur with:

1. **Very small dense models** (1-3B) where a single forward pass is sub-millisecond and the model depth is shallow enough that even B=1 recompute is fast
2. **Speculative/draft model recompute** — using a much smaller model to approximate KV, accepting quality loss
3. **Extremely degraded PCIe** — shared bus with NVMe, multi-GPU cross-traffic, or broken lanes dropping effective BW below 1 GB/s
4. **NVLink-connected remote GPU** acting as KV store — if load path goes over NVLink instead of PCIe, latency could be higher than local recompute for very small blocks
5. **Massive block sizes** (thousands of tokens) where transfer volume grows but model parallelism can absorb more compute — though both costs scale with block size, so this is unlikely to flip the ratio

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

# Recompute cost sweep — low batch (~5 min)
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_recompute_cost.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --engine torch \
    --kv-mode bf16 --block-size 32 \
    --positions 32,128,512,2048,8192,32768,131072 \
    --batch-sizes 1,2,4,8,16,32 \
    --warmup-iters 1 --measure-iters 3 \
    --max-seq-len 131104 \
    --output results/recompute_curve_h200_bf16_block32_batched.json

# Recompute cost sweep — high batch (~5 min)
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_recompute_cost.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --engine torch \
    --kv-mode bf16 --block-size 32 \
    --positions 32,128,512,2048,8192,32768,131072 \
    --batch-sizes 32,64,128,256,512,1024,2048 \
    --warmup-iters 1 --measure-iters 3 \
    --max-seq-len 131104 \
    --output results/recompute_curve_h200_bf16_block32_highbatch.json

# Crossover analysis — 32-token blocks
python scripts/plot_crossover.py \
    --recompute results/recompute_curve_h200_bf16_block32_batched.json \
                results/recompute_curve_h200_bf16_block32_highbatch.json \
    --load results/load_curve_h200_block32_conc.json \
    --block-size 32 \
    --output results/crossover_plot_h200_block32_saturated.png \
    --output-json results/crossover_policy_h200_block32_saturated.json

# 512-token blocks — load sweep (fast)
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_load_cost.py \
    --num-layers 48 --num-kv-heads 4 --head-dim 64 \
    --block-tokens 512 --dtypes fp16,fp8 \
    --concurrent-blocks 1,2,4,8,16,32,64,128 \
    --with-compute-load --per-layer \
    --output results/load_curve_h200_block512_conc.json

# 512-token blocks — recompute sweep (~10 min)
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_recompute_cost.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --engine torch \
    --kv-mode bf16 --block-size 512 \
    --positions 512,1024,2048,4096,8192,16384,32768,65536,131072 \
    --batch-sizes 1,2,4,8,16,32,64,128,256,512,1024 \
    --warmup-iters 1 --measure-iters 3 \
    --max-seq-len 131584 \
    --output results/recompute_curve_h200_bf16_block512_batched.json

# Crossover analysis — 512-token blocks
python scripts/plot_crossover.py \
    --recompute results/recompute_curve_h200_bf16_block512_batched.json \
    --load results/load_curve_h200_block512_conc.json \
    --block-size 512 \
    --output results/crossover_plot_h200_block512.png \
    --output-json results/crossover_policy_h200_block512.json
```

---

## Data Files

### 32-token blocks
- `results/recompute_curve_h200_bf16_block32_batched.json` — recompute sweep B=1..32
- `results/recompute_curve_h200_bf16_block32_highbatch.json` — recompute sweep B=32..1024
- `results/load_curve_h200_block32_conc.json` — load sweep C=1..128

### 512-token blocks
- `results/recompute_curve_h200_bf16_block512_batched.json` — recompute sweep B=1..1024
- `results/load_curve_h200_block512_conc.json` — load sweep C=1..128

### Analysis
- `results/crossover_policy_h200_block32_saturated.json` — crossover output, block=32
- `results/crossover_policy_h200_block512.json` — crossover output, block=512
