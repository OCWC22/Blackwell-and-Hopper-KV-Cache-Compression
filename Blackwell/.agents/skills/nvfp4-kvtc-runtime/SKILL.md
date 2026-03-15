# nvfp4-kvtc-runtime

Use this skill when implementing or reviewing the Blackwell-first TRT-LLM + NVFP4 + offload tiered KV runtime.

## Mental Model

- TensorRT-LLM is the primary hackathon runtime.
- NVFP4 KV cache is the primary Blackwell hot-tier thesis.
- Secondary memory offload (host RAM, disk) is the warm/cold tier for evicted or reusable KV.
- `KVTC` is a compression codec on the secondary tier.
- The main systems risk is promotion latency (secondary → hot), not whether compression exists in principle.
- Quality protection is mandatory if the tiering policy becomes more aggressive.
- vLLM + LMCache is the follow-up compatibility/productization path — not the hackathon primary.

## Implementation Guidance

1. Build and run TRT-LLM engine with NVFP4 KV cache as the hot-tier baseline.
2. Enable TRT-LLM KV offload to host memory as the secondary tier.
3. Define what stays resident in the hot tier (NVFP4, or FP8 fallback) and what is eligible for offload.
4. Add KVTC as a candidate codec on the offloaded tier for higher density.
5. Make eviction/promotion logging explicit, including policy and hit or miss counts.
6. Keep the hot and secondary paths modular so policy work can evolve later.
7. Measure promotion overhead alongside memory savings.

## Things To Avoid

- claiming `KVTC` is a hot-path win before measuring latency
- starting with vLLM + LMCache as the primary runtime before TRT-LLM results are in
- hiding policy assumptions inside unrelated code

## TensorRT-LLM NVFP4 Hot-Tier Path (Primary)

### Engine Build

TRT-LLM loads an NVFP4-KV checkpoint exported by ModelOpt and builds an engine with NVFP4 KV cache support.

### Support gate

1. Run `scripts/env_probe.sh` → check `results/env_probe.json` for TRT-LLM version and NVFP4-KV support
2. If supported: build TRT-LLM engine with NVFP4 KV cache
3. If not supported: fall back to FP8 KV cache in TRT-LLM
4. If TRT-LLM is unavailable: fall back to vLLM FP8 as compatibility path

### Full ladder with TRT-LLM

bf16 → fp8 → nvfp4 → nvfp4+offload → nvfp4+offload+kvtc

## TRT-LLM KV Host Offload (Secondary Tier)

TRT-LLM supports offloading KV cache to host memory when HBM is under pressure. This is the secondary tier.

- Evicted KV blocks move from GPU HBM to host RAM
- Promotion brings offloaded KV back to GPU when needed for decode
- KVTC compresses offloaded KV for higher host-RAM density

## vLLM + LMCache Integration (Follow-Up Path)

### KV Connector

vLLM integrates with LMCache via `LMCacheConnectorV1`:

```bash
LMCACHE_CONFIG_FILE=configs/lmcache_config.yaml \
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### How it works

- vLLM hashes token blocks (256 tokens default) and passes hashes to LMCache
- LMCache returns a bitmask of available blocks — no coordination needed
- Hierarchical lookup: GPU → CPU (LMCache) → remote storage
- Cache miss at each level cascades down

### Environment variables

- `LMCACHE_CONFIG_FILE`: path to YAML config
- `LMCACHE_LOCAL_CPU=True`: enable CPU memory backend
- `LMCACHE_MAX_LOCAL_CPU_SIZE=20.0`: CPU buffer in GB
- `LMCACHE_CHUNK_SIZE=256`: tokens per block

### Python API (offline inference)

```python
from vllm import LLM
from vllm.config import KVTransferConfig

llm = LLM(
    model="Qwen/Qwen3-30B-A3B",
    kv_cache_dtype="fp8",
    enable_prefix_caching=True,
    kv_transfer_config=KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    ),
)
```

## SLA/SLO Targets

These are the metrics the hackathon validates:

| Target | Metric | How to measure |
|--------|--------|----------------|
| More concurrent users | Max sessions at fixed p95 TPOT | TRT-LLM benchmark with concurrency sweep |
| Longer context windows | Max context at same HBM budget | Sweep context length until OOM |
| Energy efficiency | Tokens per joule (tok/J) | Benchmark computes automatically |
| TTFT improvement | Warm vs cold on repeated prefix | Measure with and without offload tier |
| HBM reduction | Peak HBM with tiering vs without | Compare baseline vs tiered results |

## Sources

- `TIERED_KV_ARCHITECTURE.md`: full tiered architecture spec including promotion path and protection policies
- NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- TensorRT-LLM docs: <https://nvidia.github.io/TensorRT-LLM/>
- LMCache docs: <https://docs.lmcache.ai/>
- vLLM LMCache examples: <https://docs.vllm.ai/en/latest/examples/others/lmcache/>
