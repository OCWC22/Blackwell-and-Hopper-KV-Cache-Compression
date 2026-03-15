# KVTC Algorithm Reference

Use this skill when: implementing KVTC compression or decompression, reasoning about PCA calibration, dynamic programming bit allocation, reconstruction error analysis, understanding quality-compression tradeoffs, or evaluating limitations from first principles.

This is a deep technical reference grounded in the KVTC paper (arXiv:2511.01815, ICLR 2026). For workflow guidance (calibration steps, protection policies, LMCache integration), see `kvtc-codec`. For the tiered architecture context (NVFP4 hot + KVTC warm/cold), see the `TIERED_KV_ARCHITECTURE.md` doc.

---

## Pipeline Overview

```
Compression:
  KV tensor
    → remove RoPE (keys only)
    → center: subtract μ
    → PCA project: z = V^T (x - μ)
    → DP bit allocation under budget B
    → quantize PCA coefficients (per-group microscaling)
    → pack quantized symbols
    → DEFLATE compress (nvCOMP, GPU)
    → bitstream

Decompression:
  bitstream
    → DEFLATE decompress (nvCOMP, GPU)
    → unpack
    → dequantize
    → inverse PCA: x̂ = V z + μ
    → FP8/BF16 KV tensor
```

---

## Stage 1 — PCA Calibration (One-Time Per Model)

### Procedure

```
Input:  Model M, calibration dataset D
Output: Orthonormal basis V, mean μ

1. Forward representative prompts from D through M
2. Collect KV caches at each layer and attention head
3. For each sampled token position t:
   - Skip if t ∈ {0, 1, ..., s-1}         (attention sinks, s=4)
   - Skip if t ∈ {T-w, T-w+1, ..., T-1}   (recent window, w=128)
   - For keys: undo RoPE positional rotation
   - Concatenate K and V across all layers and heads into matrix X
4. Compute mean: μ = mean(X, axis=0)
5. Center: X_c = X - μ
6. SVD: X_c = U Σ V^T
7. Store V (orthonormal basis) and μ as calibration artifacts
```

### Key Design Decisions

**Why SVD on centered data?**
- SVD decomposes the data into orthogonal components ordered by decreasing variance
- High-variance components capture the most information; low-variance components can be truncated or coarsely quantized
- Centering ensures the first principal component captures variance, not mean offset

**Why shared basis across heads?**
- The paper empirically observes that keys across different attention heads can be aligned into a shared latent space using orthogonal transformations (Procrustes alignment)
- Head-specific projections are rotations of a common subspace, not completely distinct information
- This justifies computing a single PCA basis V shared across heads, rather than per-head bases

**Why remove RoPE before calibration?**
- RoPE (Rotary Position Embedding) applies position-dependent rotations to key vectors
- These rotations distort the apparent low-rank structure: keys at different positions look artificially different even when their content is similar
- Removing RoPE restores the content-based correlation structure that PCA can exploit
- Post-RoPE compression works but with a measurable quality gap
- **Critical implementation question:** Does your serving stack expose pre-RoPE KV or only post-RoPE cached K? If only post-RoPE, document the quality gap as an explicit limitation

**Calibration efficiency:**
- Time: <10 minutes for a 12B model on a single H100
- Storage: V + μ = only 2.4% of model parameters for Llama-3.3-70B
- The same PCA basis V is reused across ALL compression ratios (8×, 16×, 32×, 64×). Only the bit allocation changes when switching ratios

---

## Stage 2 — Dynamic Programming Bit Allocation

### Objective

Minimize Frobenius reconstruction error under a global bit budget:

```
minimize  ||X - X̂||²_F

subject to  Σᵢ bits_i ≤ B_total

where bits_i ∈ {0, 1, 2, ..., b_max} for each PCA component i
```

### Why This Works

After PCA projection, the data is decorrelated: each PCA component has independent variance. The total reconstruction error decomposes as a sum of per-component errors:

```
||X - X̂||²_F = Σᵢ error_i(bits_i)
```

where `error_i(b)` is the quantization distortion of component i when allocated b bits. This decomposition makes the problem separable and solvable by DP.

### DP Recurrence

```
State: (component_index i, remaining_bits r)

// Base case: no components left
cost(d, r) = 0   for all r   (d = total number of components)

// Recursive case:
cost(i, r) = min over b ∈ {0, 1, ..., min(b_max, r)} of:
               error_i(b) + cost(i+1, r - b)

// Optimal allocation:
bits_i* = argmin_b [ error_i(b) + cost(i+1, r - b) ]
```

**Properties:**
- Components are sorted by decreasing variance (PCA ordering)
- `error_i(b)` is precomputed for each component and bit level
- Zero-bit assignment (b=0) means complete truncation of that component
- Trailing low-variance components typically get 0 bits
- High-variance leading components get more bits
- Complexity: O(d × B × b_max) where d = number of PCA components

### Quantization Within Components

- Per-group scaling inspired by microscaling data formats
- Shared scaling factors across groups of PCA coefficients
- The quantization step size for each component at each bit level determines `error_i(b)`

### Key Insight: Ratio Independence

Since PCA basis V is fixed (computed once during calibration), bit allocation only depends on:
1. Component variances (fixed for a given model)
2. Target compression ratio (determines B_total)

Changing from 8× to 32× only reruns the DP allocation, not the SVD calibration. This means you can sweep compression ratios cheaply.

---

## Stage 3 — Entropy Coding

### Codec
- DEFLATE (RFC 1951): combines LZ77 and Huffman coding
- GPU-accelerated via **nvCOMP** library for parallel compress/decompress directly on GPU
- No CPU roundtrip needed for entropy coding

### Compression Ratio Breakdown
```
Total compression ≈ PCA truncation × quantization × entropy coding

Example at 20× total:
- PCA truncation + quantization: ~16×
- DEFLATE entropy coding:        ~1.25× additional
- Combined:                      ~20×
```

For some models and contexts, total compression reaches 40×+ (when PCA reveals strong redundancy).

### Incremental Compression

KVTC does not compress the entire context at once. Compression is applied incrementally:

```
Every c=16 new tokens:
  1. Sliding window shifts
  2. Oldest chunk (tokens leaving the w=128 window) is compressed
  3. Compressed chunk appended to KVTC bitstream
  4. Window stays in 112–128 token range
```

This means the KVTC bitstream grows incrementally as the context extends.

---

## Stage 4 — Decompression (Promotion Path)

```
KVTC bitstream
  → nvCOMP DEFLATE decompress (GPU, parallel)
  → unpack quantized symbols
  → dequantize PCA coefficients
  → inverse PCA: x̂ = V z + μ
  → output: FP8 or BF16 KV tensor
```

### Layer-Wise Decompression

The inverse PCA transform is applied **layer-wise**, not all-at-once. This enables early generation: the model can start attending to decompressed KV from early layers while later layers are still being decompressed. This is important for minimizing promotion latency.

### Fidelity

Decompressed KV is at the target precision (FP8/BF16). Inference runs on full-precision decompressed caches with no modification to the attention mechanism. KVTC does not alter the structure of KV cache or change how attention is calculated.

---

## Protected Tokens

### Attention Sinks (s=4)

- The first 4 tokens of every sequence are never compressed
- These carry disproportionate attention mass across all layers and all heads
- The reconstruction error as a function of position shows initial tokens have the highest error
- **Ablation:** Compressing attention sink tokens collapses accuracy at high compression ratios (≥16×)

### Sliding Window (w=128)

- The most recent 128 tokens stay uncompressed in the active KV cache
- These tokens are actively being attended to during generation
- KVTC compresses tokens as they age out of the window
- **Ablation:** Without the sliding window, accuracy degrades significantly at high ratios

### Both Protections Are Essential

The ablation studies in the paper show that both protections are necessary:
- Removing sink protection: accuracy collapse at ≥16×
- Removing window protection: accuracy degradation at ≥16×
- With both: stable accuracy up to 32×–64× compression

---

## Quality Results (From Paper)

### Models Tested
- Llama 3.1 (8B, 70B)
- Llama 3.3 70B
- Mistral NeMo (12B)
- MN-Minitron (8B)
- R1-Qwen 2.5 (1.5B, 7B)

### Benchmarks
- Reasoning: GSM8K (8-shot CoT), MATH-500, AIME25
- Coding: LiveCodeBench
- Long context: LongBench, RULER (VT), LITM (Lost in the Middle), NIAH (Needle in a Haystack)
- General: MMLU, Qasper

### Key Results

**At 16× compression (~20× after DEFLATE):**
- Within **<1 score point** (accuracy or F1) of vanilla on all benchmarks
- Consistently across all tested models

**At 32×–64× compression:**
- Still maintains high accuracy
- Some tasks show compressed version matching or exceeding vanilla (attributed to CoT variability)

**vs Baselines:**
- GEAR (quantization): shows degradation at just 5× compression
- KIVI (quantization): shows degradation at just 5× compression
- H2O (token eviction): performs poorly as generic KV compressor
- TOVA (token eviction): performs poorly as generic KV compressor
- SVD-based methods: KVTC outperforms at equivalent compression ratios

### TTFT Improvement
- At 8K context: up to **8× reduction in TTFT** vs full recomputation
- Loading compressed cache + decompress is faster than rerunning prefill

---

## Known Limitations (From Paper)

### 1. Pre-RoPE Requirement
Best quality requires intercepting KV before RoPE application. Many serving stacks (vLLM, TensorRT-LLM) only expose post-RoPE cached K tensors. If only post-RoPE KV is available:
- Compression still works
- Expect a measurable quality gap vs paper results
- Document as explicit limitation in any evaluation

### 2. Calibration Dataset Dependency
PCA basis quality depends on calibration set diversity. If the calibration set does not represent the target workload distribution, compression may be suboptimal. The paper does not provide exact calibration set sizes or selection criteria beyond "representative" and "diverse."

### 3. Production Evaluation Gap
The paper explicitly acknowledges: "Evaluating larger models under conditions that more accurately mirror production is left as future work." The benchmarks are academic evaluations, not production-serving scenarios with concurrent requests, variable context lengths, and dynamic batching.

### 4. Not a Hot-Path Format
KVTC is designed for storage and transport, not as the active decode format. Decompression latency (DEFLATE + inverse PCA) makes it unsuitable for per-token random access. It should be used as a warm/cold tier, not as the always-resident KV representation.

### 5. Fixed PCA Basis
One basis per model. If the model is fine-tuned, adapted (LoRA), or used with significantly different prompt distributions than the calibration set, the PCA basis may be suboptimal. Recalibration is cheap (<10 min) but requires awareness.

### 6. Hyperparameter Sensitivity
- Chunk size c=16: affects compression/decompression granularity and overhead
- Window size w=128: affects quality protection vs compression ratio
- Sink count s=4: standard but may need adjustment for models with different attention patterns
- These are fixed in the paper's evaluation but may need tuning for production

### 7. Compatibility Scope
KVTC is compatible with token eviction methods (orthogonal) but has not been validated in combination with:
- Speculative decoding
- KV cache merging across requests
- Disaggregated serving architectures
- Real-time streaming with very short chunks

---

## LMCache Integration Path

KVTC does not alter KV cache structure or attention computation. Integration:

```
LMCache remote_serde hook (currently CacheGen):
  Replace with KVTC codec:

  encode(kv_tensor, pca_basis, bit_alloc) → compressed_bytes
  decode(compressed_bytes, pca_basis) → kv_tensor

LMCache manages: host/disk/remote storage, async I/O, cache eviction
KVTC provides: the codec (compress/decompress)
```

The KVTC paper's own end-to-end evaluation uses LMCache/vLLM for host-memory storage, with KVTC as the compression layer. The authors explicitly note that compressing KV for GPU HBM storage (rather than offload) would bring extra latency benefits but is more involved.

---

## Sources

- arXiv:2511.01815 — "KV Cache Transform Coding for Compact Storage in LLM Inference" (Staniszewski, Łańcucki)
- ICLR 2026 — OpenReview: aNVKROYpLB (conference paper, published proceedings)
- Authors: Konrad Staniszewski (NVIDIA / University of Warsaw), Adrian Łańcucki (NVIDIA)
- MarkTechPost summary (Feb 2026): "NVIDIA Researchers Introduce KVTC Transform Coding Pipeline"
- Moonlight literature review: "KV Cache Transform Coding for Compact Storage in LLM Inference"

Retrieved: 2026-03-15
