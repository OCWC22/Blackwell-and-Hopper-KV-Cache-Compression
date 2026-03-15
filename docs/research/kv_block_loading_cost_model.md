# KV Block Loading Cost Model and Roofline Benchmark

## Purpose

This document defines how to measure the cost of loading a KV cache block into GPU HBM from different storage tiers and in different data formats. The results feed eviction and promotion policy decisions: knowing the cost to load a block tells the policy whether to restore or recompute, and which tier to evict to.

The key insight is that KV block loading has two regimes:

- **Latency bound** (small blocks): per-block overhead dominates (kernel launch, metadata lookup, page table update). Transfer time is negligible.
- **Throughput bound** (large blocks): link bandwidth dominates. Per-block overhead is amortized.

This gives a roofline model for each {source tier, data type} combination.

## Cost Matrix

Every combination we measure:

| Source → Dest | BF16 (2 B/elem) | FP8 (1 B/elem, cast) | FP4 (0.5 B/elem, emulated pack) |
|---|---|---|---|
| **HBM → HBM** (same GPU) | `cudaMemcpy D2D` | copy + FP8 cast | copy + INT4 quantize + bit-pack + per-group scales |
| **Host RAM → HBM** (PCIe) | pinned DMA H2D | DMA + FP8 cast on GPU | DMA + quantize + pack on GPU |
| **Disk (NVMe) → HBM** | read → pinned → DMA | read → DMA + cast | read → DMA + quantize + pack |
| **RDMA → HBM** (remote) | theoretical ceiling | theoretical ceiling | theoretical ceiling |

**Cast/pack always happens on the GPU** after the raw bytes land in HBM. The transfer moves the source-format bytes; the destination-format conversion is a separate GPU kernel. This means:

- Transfer cost scales with **source dtype size** (BF16 moves 2x the bytes of FP8)
- Cast cost scales with **element count** (same regardless of source size)
- For FP4 destination: cast cost includes per-group scale computation

For pre-quantized storage (data already stored as FP8 or FP4 on the source tier), the transfer moves fewer bytes and no cast is needed. The benchmark measures both paths.

## Hardware Bandwidth Ceilings

These are spec-sheet numbers to be validated by measurement:

| Link | Theoretical Peak | Expected Practical | Notes |
|---|---|---|---|
| H100 SXM HBM3 | 3.35 TB/s | 2.5–3.0 TB/s | Device-to-device copy on same GPU |
| H200 HBM3e | 4.8 TB/s | 3.5–4.2 TB/s | Device-to-device copy on same GPU |
| PCIe Gen5 x16 | 63 GB/s (one-way) | 25–32 GB/s | Pinned H2D; practical is well below theoretical |
| NVMe Gen4 x4 | 7 GB/s | 3–5 GB/s | Sequential read; depends on filesystem and page cache |
| NVLink (H100 8-way) | 450 GB/s/dir | 350–400 GB/s | Intra-node, relevant for tensor parallel |
| InfiniBand NDR 400G | 50 GB/s/port | 40–45 GB/s | Theoretical only; no live measurement |
| InfiniBand HDR 200G | 25 GB/s/port | 20–22 GB/s | Theoretical only; no live measurement |

The critical ratio: **host refill is ~53x slower than H100 HBM** and **~76x slower than H200 HBM**. This means any eviction to host RAM or disk must be justified by a large enough reuse window to amortize the restore cost.

## KV Block Geometry

A KV block stores keys and values for a contiguous range of tokens across all layers:

```
block_bytes = num_layers × 2 (K+V) × block_size_tokens × num_kv_heads × head_dim × dtype_bytes
```

Example block sizes for representative models with `block_size = 16 tokens`:

| Model | Layers | KV Heads | Head Dim | BF16 (bytes) | FP8 (bytes) | FP4 (bytes + scales) |
|---|---|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | 2.1 MB | 1.05 MB | 0.53 MB + 33 KB |
| Llama-3-70B | 80 | 8 | 128 | 5.2 MB | 2.6 MB | 1.3 MB + 82 KB |
| DeepSeek-V3 (dense equiv) | 61 | 128 | 128 | 128 MB | 64 MB | 32 MB + 2 MB |

Scale metadata for FP4: one FP16 scale per group (group_size = 32 elements default), so `scale_bytes = total_elements / group_size × 2`.

At typical vLLM block sizes (16 tokens), a single Llama-3-8B block is ~1-2 MB — well into the throughput-bound regime for HBM copies but near the latency-bound knee for PCIe and disk.

## Roofline Model

### Framing

The roofline for KV block loading follows the same logic as the compute roofline but replaces FLOP/s with GB/s and arithmetic intensity with block size:

```
effective_throughput(block_bytes) = min(
    bandwidth_ceiling,                        # throughput bound
    block_bytes / T_overhead                  # latency bound
)
```

Where:
- `bandwidth_ceiling` = link bandwidth (different per source tier)
- `T_overhead` = fixed per-transfer overhead (kernel launch, page table, metadata)
- Crossover knee: `block_bytes_knee = T_overhead × bandwidth_ceiling`

### What the roofline tells us

- **Below the knee** (small blocks): adding more bytes per block is "free" in latency terms — you're paying the overhead regardless. Batching multiple small blocks into one transfer wins here.
- **Above the knee** (large blocks): cost scales linearly with bytes. This is where dtype reduction (FP8 vs BF16, FP4 vs FP8) gives proportional speedup.
- **Different tiers have different knees:** HBM has a tiny knee (~KB range), PCIe has a larger knee (~64-256 KB range), disk has the largest (~MB range).

### Data-driven measurement

Instead of measuring at fixed synthetic sizes, the benchmark ingests real coding-agent datasets and measures at the KV cache sizes that naturally arise from real conversation token counts:

1. Load sessions from `lelouch0110/claudeset-community` (114 sessions, MIT license)
2. Sort sessions by cumulative token count
3. At each measurement point, create a KV-shaped tensor matching the cumulative tokens
4. Measure transfer time from each source tier in each dtype
5. Plot effective throughput vs KV cache size

This gives roofline curves grounded in real workload sizes rather than arbitrary powers-of-two.

## Cast and Dequant Cost

The cast/pack cost is measured separately from transfer cost because they can overlap via CUDA streams:

| Conversion | Operation | Expected Cost | Notes |
|---|---|---|---|
| BF16 → FP8 | elementwise truncation | ~0.01 ms/MB | Near-zero relative to any transfer |
| BF16 → FP4 (emulated) | quantize + bit-pack + per-group scales | ~0.1–0.5 ms/MB | Nontrivial; multiple passes |
| FP4 → FP8 (reconstruct) | unpack + scale multiply | ~0.05–0.2 ms/MB | Relevant for cold-tier reads |

**Overlap opportunity:** While block N is being cast on the GPU, block N+1 can be transferring from host/disk on a separate CUDA stream. The benchmark measures both isolated cost and pipeline-overlapped cost.

## Benchmark Methodology

### Dataset ingestion

The benchmark uses real coding-agent session data to drive KV cache growth:

```python
from datasets import load_dataset
ds = load_dataset("lelouch0110/claudeset-community", split="train")
```

Each session's `stats.input_tokens` gives the cumulative token count. The benchmark:

1. Sorts sessions by input token count to get a monotonic growth curve
2. Selects measurement points at logarithmic intervals (1K, 4K, 16K, 64K, 256K, 1M+ tokens)
3. At each point, allocates a KV-shaped tensor: `[2, num_layers, num_kv_heads, num_tokens, head_dim]`
4. Measures all {source, dtype} transfer paths with CUDA event timing

### Per-measurement-point protocol

For each measurement point with `N` cumulative tokens:

1. Allocate source tensor in BF16 on GPU (baseline shape)
2. **HBM→HBM:** copy to a second GPU allocation; time with CUDA events
3. **RAM→HBM:** copy to pinned host memory, then time H2D transfer
4. **Disk→HBM:** write to temp file, drop page cache, time read + H2D
5. **Cast overhead:** time BF16→FP8 and BF16→FP4(emulated) separately on GPU
6. Record: `cumulative_tokens`, `kv_bytes_{bf16,fp8,fp4}`, `latency_ms`, `throughput_gbps`, `cast_ms`

### Warm-up and statistical robustness

- 5 warm-up iterations (untimed) to stabilize GPU clocks and fill caches
- 20 timed iterations per measurement point
- Report: median, p5, p95 of latency and throughput

## Output Schema

```json
{
  "benchmark": "kv_block_loading_cost",
  "hardware": { "gpu_model": "...", "hbm_bandwidth_tb_s": 3.35, ... },
  "dataset": { "name": "claudeset-community", "sessions_used": 114 },
  "model_config": { "num_layers": 32, "num_kv_heads": 8, "head_dim": 128 },
  "measurements": [
    {
      "cumulative_tokens": 1024,
      "kv_bytes_bf16": 67108864,
      "kv_bytes_fp8": 33554432,
      "kv_bytes_fp4": 16777216,
      "transfers": {
        "hbm_to_hbm": { "bf16": { "latency_ms_median": ..., "throughput_gbps": ... }, ... },
        "ram_to_hbm": { ... },
        "disk_to_hbm": { ... }
      },
      "cast_overhead": {
        "bf16_to_fp8_ms": ...,
        "bf16_to_fp4_ms": ...
      },
      "dataset_provenance": { "session_id": "...", "turn_index": 3 }
    }
  ]
}
```

## Roofline Plots

The benchmark produces two plot families:

### 1. Throughput roofline

- **X-axis:** KV cache size (bytes, log scale) — driven by dataset token accumulation
- **Y-axis:** effective throughput (GB/s, log scale)
- **One subplot per source tier** (HBM, RAM, Disk)
- **Lines per dtype** (BF16, FP8, FP4)
- **Horizontal ceiling** = theoretical bandwidth for that tier
- **Annotated knee point** where latency bound meets throughput bound

### 2. Latency growth curve

- **X-axis:** cumulative tokens (log scale)
- **Y-axis:** per-block load latency (ms, log scale)
- **Lines per {source, dtype}**
- Shows when and where each combination transitions from overhead-dominated to bandwidth-dominated

## References

- `[L1]` `Blackwell/docs/research/synthetic_kv_working_set_benchmark_mtp_ep.md` — block restore cost model
- `[L2]` `Blackwell/docs/research/actual_kv_serving_benchmark_and_synthetic_data.md` — benchmark spec
- `[L3]` `Blackwell/docs/research/coding_agent_datasets_for_kv_benchmarks.md` — dataset catalog
- `[L4]` `Blackwell/scripts/kv_bench_utils.py` — shared benchmark utilities
- `[R1]` H100 product page — https://www.nvidia.com/en-us/data-center/h100/
- `[R2]` H200 product page — https://www.nvidia.com/en-us/data-center/h200/
- `[R3]` PCI-SIG bandwidth table — https://pcisig.com
- `[R4]` Hopper architecture in depth — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
