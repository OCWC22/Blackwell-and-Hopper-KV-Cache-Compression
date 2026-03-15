# KVTC Codec

Use this skill when: implementing KVTC calibration, PCA basis computation, compression or decompression pipelines, entropy coding, LMCache serde integration, or evaluating compression ratios for warm and cold KV tiers.

## Core Rules

- KVTC is a transform coder: PCA-based decorrelation → adaptive quantization → entropy coding
- PCA basis is computed once per model on a calibration set and reused across compression ratios
- Bit allocation across PCA components uses dynamic programming; only the allocation changes when switching 8× / 16× / 32× targets
- Entropy coding uses nvCOMP DEFLATE or equivalent fast GPU-friendly codec
- Compression applies before RoPE for best quality; post-RoPE interception is a known quality tradeoff and explicit limitation
- Average compression is roughly 20× vs 16-bit KV; some sequences reach 40×+
- Source: KVTC paper (ICLR 2026), https://openreview.net/forum?id=tMiBQXQ0Cm

## Calibration Workflow

1. Capture representative reusable prompts: chat and system prefixes, agent scaffolds, code prefixes, long-document prefixes
2. Run forward pass to collect KV activations at each layer and head
3. Compute PCA basis per layer and head group from collected activations
4. Store PCA matrices as calibration artifact (reusable across ratios)
5. Run DP-based bit allocation for each target compression ratio
6. Validate quality on small benchmark subset at each ratio
7. Select operating point based on quality-versus-density tradeoff

## Protection Policies

- First 4 tokens (attention sinks) are never compressed
- Final 128 tokens (recent window) stay uncompressed in Tier 0
- Live decode tail is never KVTC-compressed
- Only compress stable KV: reused prefixes, stale turns, agent scaffolds, cached documents

## LMCache Integration

- LMCache exposes remote_serde hook (CacheGen today)
- Replace CacheGen with KVTC codec for higher density
- LMCache controller APIs support explicit compress and decompress calls
- Keep codec modular: encode(kv_tensor, pca_basis, bit_alloc) → bytes; decode(bytes, pca_basis) → fp8_tensor

## Things To Avoid

- Using KVTC as hot-path format before proving promotion latency is acceptable
- Conflating KVTC paper results with LMCache or vLLM behavior without measuring
- Compressing every token including live decode tail
- Redoing PCA calibration when only changing compression ratio
- Assuming post-RoPE compression matches paper quality without validating
