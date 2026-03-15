# B200 Architecture and ISA Reference

Use this skill when: making hardware-aware decisions about KV cache layout, memory bandwidth budgets, tensor core utilization, NVFP4 dequantization costs, promotion path sizing, or any optimization that depends on B200 first principles.

## B200 Hardware Specifications

### Die and Process
- **Process:** TSMC 4NP, dual-die chiplet design
- **Transistors:** 208 billion (across two dies)
- **Compute capability:** 10.0 (sm_100)
- **Each die:** 80 physical SMs, 74 enabled → **148 SMs total**
- **Architecture:** Blackwell (5th gen Tensor Cores)

### Compute Throughput (per GPU)
| Precision | Peak Performance |
|-----------|-----------------|
| FP4 (NVFP4) dense | 10 PFLOPS |
| FP4 (NVFP4) sparse (2:1) | 20 PFLOPS |
| FP8 / FP6 | 10 PFLOPS (dense) |
| INT8 | 10 PETAOPS |
| FP16 / BF16 | 5 PFLOPS |
| TF32 | 2.5 PFLOPS |
| FP32 (CUDA cores) | 80 TFLOPS |
| FP64 | 40 TFLOPS |

Key ratio: **FP4 is 2× FP8 throughput** at the tensor core level.

### Memory
| Parameter | Value |
|-----------|-------|
| HBM3e capacity | 192 GB (96 GB per die, 4 × 24 GB stacks per die) |
| HBM3e bandwidth | **8 TB/s** total (4096-bit bus per die) |
| L2 cache | **126 MB** (partitioned across dies) |
| L1 / shared / texture (combined) | 256 KB per SM |
| Max shared memory per SM | 228 KB (configurable carveout) |
| Register file per SM | 256 KB (4 × 16,384 × 32-bit) |
| TMEM per SM | 256 KB (dedicated tensor memory) |

### Interconnect
| Interface | Bandwidth |
|-----------|-----------|
| NVLink 5 (18 links × 100 GB/s) | **1.8 TB/s** bidirectional per GPU |
| PCIe Gen6 x16 | ~256 GB/s bidirectional |
| NVLink 5 per link | 100 GB/s (200 Gbps SerDes × 4 pairs) |
| DGX B200 aggregate (8 GPU) | 14.4 TB/s GPU-to-GPU |

### Power
- **TDP:** 1,000 W (air or liquid cooling)

---

## Fifth-Generation Tensor Core ISA

### New MMA Instructions
- PTX opcode: `tcgen05.mma` (replaces Hopper's `wgmma`)
- **Warp-level, single-thread MMA** — each thread issues MMA independently
- Eliminates warp-wide instruction barriers from Hopper
- Instruction latency: ~11 cycles, nearly constant across tile sizes and precisions
- SASS instruction families: DMMA (FP64), HMMA (FP16/BF16), QMMA (FP8/INT8), OMMA (FP4)

### Supported Data Types on Tensor Cores
| SASS Opcode | Data Types | Bits |
|-------------|-----------|------|
| DMMA | FP64 | 64 |
| HMMA | FP16, BF16, TF32 | 16/32 |
| QMMA | FP8 (E4M3, E5M2), INT8 | 8 |
| OMMA | FP4 (NVFP4, MXFP4), FP6 | 4/6 |

### Tensor Memory (TMEM)
- 256 KB per SM, 512 columns × 128 lanes × 32 bits
- Dedicated instructions: `tcgen05.cp`, `tcgen05.ld`, `tcgen05.st`
- **Read bandwidth:** 16 TB/s per SM
- **Write bandwidth:** 8 TB/s per SM
- 58% reduction in cache miss latency vs H200 (~420 cycles vs ~1000)

### Microscaling Formats
| Format | Block Size | Scale Type | Scale Per |
|--------|-----------|------------|-----------|
| NVFP4 | 16 values | FP8 E4M3 + FP32 | micro-block + tensor |
| MXFP4 | 32 values | E8M0 | block |
| MXFP6 | 32 values | E8M0 | block |
| MXFP8 | 32 values | E8M0 | block |

NVFP4 accuracy advantage: dual-scale (FP8 per 16 + FP32 per tensor) yields ~5% better accuracy than MXFP4 single-scale for KV cache on Llama 3.3 70B.

---

## Hardware Decompression Engine
- Throughput: **800 GB/s** hardware-accelerated decompression
- Programmable via nvCOMP library
- Reduces CPU overhead for compressed dataset loading
- Relevant for KVTC entropy-coded bitstream decompression on promotion path

---

## NVFP4 KV Cache: Hardware Path

### Quantization Flow
```
New token K/V (FP8/BF16)
    │
    ▼ quantize to NVFP4
NVFP4 KV cache in HBM
    │
    ▼ dequantize NVFP4 → FP8
FP8 values for attention computation
```

### NVFP4 Format Detail
```
┌─────────────────────────────────────────────────┐
│ 16 values × E2M1 (4-bit payload)                │
│ + 1 × FP8 E4M3 micro-block scale               │
│ + 1 × FP32 per-tensor scale (amortized)         │
└─────────────────────────────────────────────────┘
Storage: 4 bits/value + scale overhead
Effective: ~50% memory reduction vs FP8 KV cache
```

### Measured Benefits (NVIDIA benchmarks)
- **Memory:** 50% reduction vs FP8 → effectively doubles context / batch
- **TTFT:** up to 3× lower vs FP8 KV cache
- **Cache hit rate:** 20% higher vs FP8
- **Accuracy:** <1% loss on LiveCodeBench, MMLU-PRO, MBPP, Ruler 64K
- **Per-user latency:** 2.5× lower on GB200 vs H200

---

## First Principles for KV Cache Workloads

### Memory Bandwidth Budget
KV cache attention is **memory-bandwidth bound**, not compute bound.

For decode (one new token attending to N cached tokens):
- Each attention head reads K and V for all N cached tokens
- With NVFP4: read 0.5 bytes/value (4 bits) + scale overhead
- With FP8: read 1 byte/value
- With BF16: read 2 bytes/value

**Bandwidth savings directly translate to decode latency reduction** because the operation is limited by HBM read speed (8 TB/s on B200).

### Back-of-envelope: KV Cache Size
For a model with:
- `n_layers` layers, `n_heads` KV heads, `d_head` head dimension, `seq_len` tokens

```
KV bytes = 2 × n_layers × n_heads × d_head × seq_len × bytes_per_value
```

| Format | bytes_per_value | 70B model, 64K context (approx) |
|--------|----------------|--------------------------------|
| BF16 | 2.0 | ~40 GB |
| FP8 | 1.0 | ~20 GB |
| NVFP4 | ~0.5 | ~10 GB |
| KVTC 20× | ~0.1 | ~2 GB |

### Promotion Bandwidth Cost
Promoting KVTC → FP8 → NVFP4 for N tokens:
1. **Read KVTC blob from host:** limited by PCIe Gen6 (~256 GB/s) or NVLink if GPU-to-GPU
2. **KVTC decode (PCA inverse + dequant):** compute-bound, uses CUDA cores
3. **NVFP4 pack (FP8 → NVFP4 quantize):** lightweight, tensor core or CUDA core
4. **Write to HBM:** limited by HBM bandwidth (8 TB/s)

Host → GPU transfer is the bottleneck. For 1 GB of KVTC data over PCIe Gen6: ~4 ms. Over NVLink 5: ~0.6 ms.

### Decode Latency vs Promotion Latency
- Decode latency per token: microseconds (memory-bound attention kernel)
- Promotion latency per prefix: milliseconds (host transfer + decompress + repack)
- **Rule:** promotion cost is paid once per prefix reuse, amortized over all tokens decoded from that prefix

### Tensor Core Utilization for KV Cache
- Prefill (long prompt, batch matmul): **compute-bound**, tensor cores matter
- Decode (autoregressive, one token at a time): **memory-bound**, HBM bandwidth matters
- Promotion (KVTC decompress + repack): **mixed** — PCA inverse is compute, host transfer is I/O

### Key Hardware Ratios for Decision-Making
| Ratio | Value | Implication |
|-------|-------|-------------|
| HBM BW / PCIe BW | 8000 / 256 = **31×** | Host offload is 31× slower than HBM access |
| HBM BW / NVLink BW | 8000 / 1800 = **4.4×** | Cross-GPU is 4.4× slower than local HBM |
| NVFP4 / FP8 memory | 0.5× | Doubles effective KV capacity |
| KVTC / FP8 memory | 0.05-0.1× | 10-20× density, but promotion cost |
| FP4 / FP8 compute | 2× PFLOPS | Extra headroom for promotion compute |
| L2 cache | 126 MB | Can cache hot KV heads; 2.5× larger than H100 |

---

## Comparison: B200 vs H100/H200 for KV Cache

| Feature | H100 | H200 | B200 |
|---------|------|------|------|
| HBM capacity | 80 GB HBM3 | 141 GB HBM3e | 192 GB HBM3e |
| HBM bandwidth | 3.35 TB/s | 4.8 TB/s | 8 TB/s |
| L2 cache | 50 MB | 50 MB | 126 MB |
| Native NVFP4 | No | No | **Yes** |
| Native FP8 TC | Yes | Yes | Yes |
| FP4 TC | No | No | **Yes (OMMA)** |
| Tensor cores | 456 | 456 | 528 |
| NVLink BW | 900 GB/s | 900 GB/s | 1.8 TB/s |
| Decompression engine | No | No | **Yes (800 GB/s)** |
| TMEM per SM | No | No | **256 KB** |
| Compute capability | 9.0 | 9.0 | 10.0 |

**Key Blackwell advantages for KV cache:**
1. Native NVFP4 eliminates dequantize-compute-requantize overhead for hot KV
2. 2.4× more HBM bandwidth than H100 → faster decode for large KV caches
3. 2.5× larger L2 cache → more KV heads can stay in fast cache
4. Hardware decompression engine → potential acceleration of KVTC decode on promotion
5. TMEM → reduced latency for tensor core operand staging

---

## Sources

- NVIDIA Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- NVIDIA Blackwell Architecture: https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- NVIDIA NVFP4 KV Cache Blog: https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache
- NVIDIA Blackwell Ultra Blog: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- Blackwell Microbenchmarking (arXiv:2512.02189): https://arxiv.org/html/2512.02189v1
- Chips and Cheese B200 Analysis: https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut
- NVIDIA DGX B200: https://www.nvidia.com/en-us/data-center/dgx-b200/
- Blackwell B200 Datasheet: https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf
- SemiAnalysis Tensor Core Evolution: https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
- Transformer Engine FP8/FP4: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html

Retrieved: 2026-03-15
