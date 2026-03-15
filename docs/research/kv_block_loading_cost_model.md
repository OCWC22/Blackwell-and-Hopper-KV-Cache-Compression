# KV Block Offload and Restore Cost Model

## Purpose

This document defines how to measure the cost of **offloading** KV cache blocks from GPU HBM to lower tiers (host RAM, disk) and **restoring** them back. The results feed eviction and promotion policy: knowing the round-trip cost per block tells the policy whether to evict, where to evict to, and whether restoring is cheaper than recomputing.

The key insight is that block transfers have two regimes:

- **Latency bound** (small transfers): per-transfer overhead dominates (kernel launch, page table update, DMA setup). Transfer time is negligible.
- **Throughput bound** (large transfers): link bandwidth dominates. Overhead is amortized.

This gives a roofline model for each {tier, direction, format} combination.

## What We Measure

### Four timed paths

| Path | Direction | What it measures |
|---|---|---|
| `offload_to_host` | GPU → pinned host RAM | D2H DMA into pre-allocated pinned buffer |
| `restore_from_host` | pinned host RAM → GPU | H2D DMA into pre-allocated GPU buffer |
| `offload_to_disk` | GPU → host → disk | D2H + raw `os.write` (no serialization) |
| `restore_from_disk` | disk → host → GPU | raw `os.read` + H2D (no deserialization) |

Plus `hbm_copy` (GPU → GPU same device) as a bandwidth ceiling reference.

All paths use **pre-allocated buffers** — no allocation in the timed loop. This isolates pure DMA cost from PyTorch allocation overhead.

### Three storage formats

| Format | Offload path | Restore path | Bytes/elem |
|---|---|---|---|
| **BF16** | copy raw 2B/elem | copy raw 2B/elem | 2 |
| **FP8** (pre-quantized) | quantize on GPU → copy 1B/elem | copy 1B/elem → dequant on GPU | 1 |
| **FP4** (emulated pack) | pack on GPU → copy 0.5B/elem + scales | copy 0.5B/elem → unpack on GPU | 0.5 + scale overhead |

For pre-quantized formats, the **link transfers fewer bytes** because quantization happens before offload (or after restore). Cast overhead is measured separately so it can be overlapped with the next block's transfer.

### Sweep axes

- **Block size (tokens):** 1, 4, 16, 64, 256 — matches typical vLLM/TRT-LLM block sizes
- **Batch count:** 1, 4, 16, 64 blocks per transfer — shows batching benefit
- **Format:** BF16, FP8, FP4

Dataset sessions drive additional batch counts: each session's token count is converted to a block count, giving realistic batch sizes.

## Hardware Bandwidth Ceilings

| Link | Theoretical Peak | Expected Practical | Notes |
|---|---|---|---|
| H100 SXM HBM3 | 3.35 TB/s | 2.5–3.0 TB/s | Device-to-device copy on same GPU |
| H200 HBM3e | 4.8 TB/s | 3.5–4.2 TB/s | Device-to-device copy on same GPU |
| PCIe Gen5 x16 | 63 GB/s (one-way) | 25–32 GB/s | Pinned DMA; practical well below theoretical |
| NVMe Gen4 x4 | 7 GB/s | 3–5 GB/s | Sequential; depends on filesystem |
| InfiniBand NDR 400G | 50 GB/s/port | 40–45 GB/s | Theoretical only; no live measurement |
| InfiniBand HDR 200G | 25 GB/s/port | 20–22 GB/s | Theoretical only; no live measurement |

**Critical ratio:** host round-trip is ~53x slower than H100 HBM and ~76x slower than H200 HBM. Disk round-trip is ~500x slower than HBM. Eviction to host or disk must be justified by enough reuse to amortize restore cost.

## KV Block Geometry

```
block_bytes = num_layers × 2 (K+V) × block_tokens × num_kv_heads × head_dim × fmt_bytes
```

| Model | Layers | KV Heads | Head Dim | BF16 block (16 tok) | FP8 block | FP4 block + scales |
|---|---|---|---|---|---|---|
| Llama-3-8B | 32 | 8 | 128 | 2.1 MB | 1.05 MB | 0.53 MB + 33 KB |
| Llama-3-70B | 80 | 8 | 128 | 5.2 MB | 2.6 MB | 1.3 MB + 82 KB |

At vLLM block size = 16 tokens, a Llama-3-8B block is ~1-2 MB — throughput-bound for HBM, near the knee for PCIe.

## Roofline Model

```
effective_throughput(transfer_bytes) = min(
    bandwidth_ceiling,                          # throughput bound
    transfer_bytes / T_overhead                 # latency bound
)
```

- `T_overhead` = fixed per-transfer cost (kernel launch, DMA setup, page table)
- `bandwidth_ceiling` = link bandwidth per tier
- Crossover knee: `transfer_bytes_knee = T_overhead × bandwidth_ceiling`

**Below the knee:** batching multiple blocks into one transfer amortizes overhead.
**Above the knee:** reducing bytes per element (FP8 vs BF16) gives proportional speedup.
**Different tiers have different knees:** HBM ~KB, PCIe ~64-256 KB, disk ~MB.

## Cast / Quantize Cost

Measured separately from transfer because cast can overlap with the next transfer:

| Conversion | Direction | Operation | Expected Cost |
|---|---|---|---|
| BF16 → FP8 | offload (pre-quantize) | elementwise truncation | ~0.01 ms/MB |
| FP8 → BF16 | restore (dequantize) | elementwise widen | ~0.01 ms/MB |
| BF16 → FP4 | offload (emulated pack) | quantize + bit-pack + per-group scales | ~0.1–0.5 ms/MB |
| FP4 → FP32 | restore (unpack) | unpack + scale multiply | ~0.05–0.2 ms/MB |

## Benchmark Script

`scripts/bench_kv_block_load.py` — see docstring for usage.

Key design decisions:
- **Pre-allocated buffers** for GPU src, GPU dst, and pinned host. No allocation in timed loop.
- **Raw binary I/O** for disk path. No `torch.save/load` serialization overhead.
- **CUDA event timing** for GPU-involved paths; `time.perf_counter` for end-to-end disk paths.
- **Separate cast measurement** per format so cast can be analyzed independently or as pipeline overlap.
- **Dataset-driven batch counts** from `claudeset-community` session sizes.

## Output

JSON with per-{block_size, batch_count, format} measurements including:
- `offload_to_host`, `restore_from_host`, `offload_to_disk`, `restore_from_disk`, `hbm_copy`
- Each with: `latency_ms_median`, `latency_ms_p5`, `latency_ms_p95`, `throughput_gbps`
- `cast_overhead` with quantize and dequantize timings

## Plots

`scripts/plot_kv_load_roofline.py` produces three plots:

1. **Throughput roofline** — effective GB/s vs transfer size, with bandwidth ceilings
2. **Latency curves** — ms vs transfer size, offload and restore side-by-side
3. **Batch scaling** — throughput vs batch count at fixed block size, showing batching benefit

## References

- `[L1]` `Blackwell/docs/research/synthetic_kv_working_set_benchmark_mtp_ep.md` — block restore cost decomposition
- `[L2]` `Blackwell/docs/research/coding_agent_datasets_for_kv_benchmarks.md` — dataset catalog
- `[L3]` `Blackwell/scripts/kv_bench_utils.py` — shared benchmark utilities
- `[R1]` H100 product page — https://www.nvidia.com/en-us/data-center/h100/
- `[R2]` H200 product page — https://www.nvidia.com/en-us/data-center/h200/
- `[R3]` Hopper architecture — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
