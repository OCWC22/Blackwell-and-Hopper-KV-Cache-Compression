# NVFP4 Format Reference

Use this skill when: implementing NVFP4 quantization or dequantization, reasoning about quantization error, comparing NVFP4 vs MXFP4, understanding the microscaling architecture from first principles, or working on KV cache format decisions that depend on E2M1 precision characteristics.

This is a deep technical reference. For workflow guidance (ModelOpt recipe, deployment paths), see `modelopt-nvfp4-quantization`. For hardware specs (tensor core throughput, bandwidth), see `b200-architecture`.

---

## E2M1 Bit Layout

```
[ S | E₁ E₀ | M₀ ]
  1    2 bits   1 bit

Exponent bias = 1
Total: 4 bits per value
```

No NaN or Inf encodings — all 16 bit patterns (8 positive, 8 negative) represent valid numbers.

### Full Representable Values Table

| Bits (S E₁E₀ M₀) | Sign | Exponent (raw) | Exponent (debiased) | Mantissa | Value |
|-------------------|------|-----------------|---------------------|----------|-------|
| 0 00 0 | + | 0 (subnormal) | — | 0.0 | **+0.0** |
| 0 00 1 | + | 0 (subnormal) | — | 0.5 | **+0.5** |
| 0 01 0 | + | 1 | 0 | 1.0 | **+1.0** |
| 0 01 1 | + | 1 | 0 | 1.5 | **+1.5** |
| 0 10 0 | + | 2 | 1 | 1.0 | **+2.0** |
| 0 10 1 | + | 2 | 1 | 1.5 | **+3.0** |
| 0 11 0 | + | 3 | 2 | 1.0 | **+4.0** |
| 0 11 1 | + | 3 | 2 | 1.5 | **+6.0** |

Negative values (S=1) mirror the positive set: {-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0}

**Dynamic step sizes:**
- Between 0 and 2: step = 0.5
- Between 2 and 4: step = 1.0
- Between 4 and 6: step = 2.0

This non-uniform spacing concentrates precision near zero where most weight and activation values cluster. The coarser spacing at larger magnitudes creates higher relative error for near-maximal values — the core problem addressed by the "Four Over Six" technique.

**Subnormal handling:** Exponent bits `00` with mantissa `0` = 0.0, with mantissa `1` = 0.5. These use a fixed exponent of 2^(1-bias) = 2^0 = 1 with implicit leading zero (0.M instead of 1.M).

---

## Two-Level Microscaling Architecture

NVFP4 uses a two-level scaling scheme because E4M3 (max = 448) alone cannot cover the full dynamic range of a tensor. The two levels work together:

### Level 1 — Per-Tensor FP32 Scale (α)

Maps the entire tensor into the representable range of FP4 × FP8 E4M3:

```
α = max(|X|) / (M_FP4 × M_FP8)
  = max(|X|) / (6 × 448)
  = max(|X|) / 2688
```

Where:
- `M_FP4 = 6` (largest E2M1 value)
- `M_FP8 = 448` (largest E4M3 normal value: 2^8 × 1.75)

The global amax requires a full pass through device memory before quantization begins.

### Level 2 — Per-Block FP8 E4M3 Scale (Δ)

One scale per 16 contiguous values. Maps block values into FP4 representable range:

```
Δ_b = cast_E4M3( amax(block_b) / (α × 6) )
```

**Why E4M3 for block scale:**
- E4M3 has 4 exponent bits (bias 7) and 3 mantissa bits
- Representable range: subnormal min = 2^(-6) × 0.125 = 0.001953125, normal max = 448
- The 3 mantissa bits give fractional precision between powers of two
- This means the scale can track block amax closely (utilization ratio ρ ≈ 1)

**Why E8M0 is worse (MXFP4):**
- E8M0 has 8 exponent bits but 0 mantissa bits
- Every representable value is a power of two (1, 2, 4, 8, 16, ...)
- When block amax falls between powers of two, the scale rounds to the nearest power
- This leaves FP4 codebook entries unused, increasing quantization error
- Measured impact: NVFP4 achieves ~5% better accuracy than MXFP4 on Llama 3.3 70B KV cache

**Why two levels instead of one:**
- If using only E4M3 (max 448): tensor values beyond 448 × 6 = 2688 would clip
- If using only FP32: per-block FP32 scales would cost 32 bits per 16 values = 2 extra bits/value (vs 0.5 for E4M3)
- Two-level scheme: FP32 normalizes tensor range once, E4M3 handles local variation cheaply

---

## Quantization Algorithm

```
Input:  X (tensor of values in FP8/BF16/FP16/FP32)
Output: {q_i} (4-bit packed E2M1), {Δ_b} (FP8 E4M3 per block), α (FP32 per tensor)

1. Compute global amax:
   amax_global = max(|X|)
   // Requires one pass through device memory

2. Compute global encode scale:
   S_enc = (6.0 × 448.0) / amax_global
   // Inverse of α: used to scale values into quantizable range

3. Compute global decode scale:
   α = amax_global / (6.0 × 448.0)
   // Stored as FP32, used for dequantization

4. For each block b of 16 contiguous values:
   a. Compute block amax in scaled space:
      amax_b_scaled = max(|block_b|) × S_enc

   b. Compute block decode scale:
      Δ_b_decode = amax_b_scaled / 6.0

   c. Cast to E4M3:
      Δ_b = cast_E4M3(Δ_b_decode)
      // Rounding to nearest E4M3 representable value

   d. Compute effective block encode scale:
      Δ_b_enc = 6.0 / Δ_b
      // Inverse of the cast scale for encoding

   e. Quantize each value:
      q_i = round_to_nearest_E2M1(x_i × S_enc × Δ_b_enc)
      // Maps scaled value to nearest of {0, 0.5, 1, 1.5, 2, 3, 4, 6}

5. Store: {q_i} packed (2 values per byte), {Δ_b}, α
```

### Dequantization Formula

```
x̂_i = q_i × Δ_b × α
```

Where:
- `q_i` is the E2M1 quantized value (one of the 16 representable numbers)
- `Δ_b` is the E4M3 block scale (shared across 16 elements)
- `α` is the FP32 tensor scale (shared across entire tensor)

On Blackwell hardware, this dequantization from NVFP4 to FP8 happens automatically in the tensor core data path before attention computation. The `cast_E2M1` and scale application are fused into the OMMA instruction.

---

## The "Four Over Six" Improvement (arXiv:2512.02010)

### Problem
Standard NVFP4 quantization always maps block amax to the maximum FP4 value (6). Because the step size between 4 and 6 is 2.0, values near 5.0 round to either 4 or 6 with up to 1.0 absolute error. This creates disproportionate error for near-maximal values.

### Solution
For each block, evaluate two candidate scales:
1. Map block amax to FP4 max = **6** (standard)
2. Map block amax to FP4 value = **4** (alternative, finer grid)

Select whichever yields lower MSE for that block.

When mapped to max=4, the step sizes become: 0.33 (0–1.33), 0.67 (1.33–2.67), 1.33 (2.67–4). This trades off dynamic range for finer precision.

### Why E4M3 Enables This
E4M3 can represent both 6-mapping and 4-mapping scales precisely because it has mantissa bits. E8M0 (power-of-two only) cannot represent the fractional ratio 4/6 = 0.667, so this technique is impossible with MXFP4.

### Overhead
- Inference: <2% overhead (one extra comparison per block)
- Training: <15% overhead
- Code: github.com/mit-han-lab/fouroversix (CUDA, Triton, PyTorch reference)

---

## NVFP4 vs MXFP4 Comparison

| Property | NVFP4 | MXFP4 (OCP MX) |
|----------|-------|-----------------|
| Payload format | E2M1 (4-bit) | E2M1 (4-bit) |
| Block size | 16 values | 32 values |
| Scale format | FP8 E4M3 | FP8 E8M0 |
| Scale precision | Fractional (mantissa bits) | Power-of-two only |
| Second-level scale | FP32 per-tensor | None |
| Effective bits/value | ~4.5 | ~4.25 |
| Scale utilization (ρ) | ≈ 1.0 | < 1.0 (gaps between powers of 2) |
| Accuracy (KV cache) | Baseline | ~5% worse (Llama 3.3 70B) |
| 4/6 technique | Supported | Impossible |
| PTX qualifier | `mxf4nvf4 .block16` | `mxf4 .block32` |
| Hardware | Blackwell SM100/SM120 | Blackwell SM100/SM120 |

---

## Storage Efficiency

### Per-Block Overhead
```
16 values × 4 bits = 64 bits (E2M1 payload)
1 scale × 8 bits   =  8 bits (E4M3 block scale)
                    = 72 bits total per 16 values
                    =  9 bytes per 16 values
                    =  4.5 bits/value effective
```

### Per-Tensor Overhead
```
1 scale × 32 bits = 32 bits (FP32 tensor scale)
Amortized over millions of values: negligible
```

### Memory Comparison
| Format | Bits/value | vs BF16 reduction | vs FP8 reduction |
|--------|-----------|-------------------|------------------|
| BF16 | 16 | — | — |
| FP8 | 8 | 50% | — |
| NVFP4 | 4.5 | 72% | 44% |
| MXFP4 | 4.25 | 73% | 47% |

### KV Cache Impact
For a model with `n_layers` layers, `n_kv_heads` KV heads, `d_head` head dim, `seq_len` tokens:
```
KV_bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × bits_per_value / 8
```

Example: Llama 3.1 70B (80 layers, 8 KV heads, 128 dim) at 64K context:
| Format | KV Cache Size |
|--------|--------------|
| BF16 | ~20.5 GB |
| FP8 | ~10.2 GB |
| NVFP4 | ~5.8 GB |

---

## Known Limitations

1. **Blackwell-only:** NVFP4 requires SM100/SM120 tensor cores. No native hardware support on Hopper (H100/H200). On Hopper, FP4 is emulation only with no performance benefit
2. **Extra memory pass:** Global amax computation requires reading the entire tensor before quantization begins. This adds latency proportional to tensor size
3. **Non-uniform step sizes:** The 0.5/1.0/2.0 step pattern creates higher relative error at larger magnitudes. The "Four Over Six" technique mitigates but does not eliminate this
4. **Coarse codebook:** Only 16 representable values per sign (8 positive + 0). Distributions with fine structure or heavy tails may suffer
5. **Scale quantization error:** Casting block scale to E4M3 introduces its own rounding error. When amax_b falls between E4M3 representable values, some FP4 range is wasted
6. **KV cache specific:** Dequantization to FP8 before attention means the attention computation itself runs at FP8 precision, not higher. Quality is bounded by FP8 attention accuracy
7. **Dynamic quantization:** For KV cache, each new token's K/V is quantized on the fly. The global amax may shift as new tokens arrive, but the per-tensor scale is typically fixed at model load time (calibrated via ModelOpt)

---

## Sources

- arXiv:2509.25149 — "Pretraining Large Language Models with NVFP4" (NVIDIA, Sep 2025)
- arXiv:2512.02010 — "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling" (MIT/NVIDIA, Dec 2025)
- NVIDIA Technical Blog — "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference" (Jan 2026)
- NVIDIA Transformer Engine docs — "Using FP8 and FP4 with Transformer Engine"
- NVIDIA NVFP4 KV Cache Blog — "Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache"
- github.com/mit-han-lab/fouroversix — CUDA/Triton/PyTorch reference implementation

Retrieved: 2026-03-15
