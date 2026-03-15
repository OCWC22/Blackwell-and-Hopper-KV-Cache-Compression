# nvfp4-kvtc-runtime

Use this skill when implementing or reviewing the Blackwell-first vLLM + LMCache tiered KV runtime.

## Mental Model

- vLLM FP8 KV cache is the stable hot-tier path.
- `LMCache` manages the cold/warm reusable KV layer.
- `NVFP4` is an optional Blackwell enhancement if runtime support is verified.
- `KVTC` is a cold-tier codec candidate.
- The main systems risk is promotion latency, not whether compression exists in principle.
- Quality protection is mandatory if the tiering policy becomes more aggressive.

## Implementation Guidance

1. Define what stays resident in the hot tier (FP8, or NVFP4 if verified).
2. Define what is eligible for cold-tier storage via `LMCache`.
3. Make promotion logging explicit, including policy and hit or miss counts.
4. Keep the hot and warm path modular so policy work can evolve later.
5. Measure promotion overhead alongside memory savings.

## Things To Avoid

- claiming `KVTC` is a hot-path win before measuring latency
- mixing `LMCache` baseline claims with `KVTC` paper claims
- hiding policy assumptions inside unrelated code

## vLLM + LMCache Integration

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

## NVFP4 Hot-Tier Path

NVFP4 is a Blackwell-native 4-bit format (E2M1 + FP8 E4M3 microscaling). It is relevant as a hot-tier enhancement, not as the default assumed path.

### Support gate

1. Run `scripts/env_probe.sh` → check `results/env_probe.json` for `nvfp4_kv_supported`
2. If supported: use `--kv-cache-dtype nvfp4` in vLLM
3. If not supported: fall back to `--kv-cache-dtype fp8`
4. Can combine with LMCache: NVFP4 hot + LMCache cold

### Full ladder with NVFP4

bf16 → fp8 → fp8+lmcache → nvfp4 → nvfp4+lmcache

## SLA/SLO Targets

These are the metrics the hackathon validates:

| Target | Metric | How to measure |
|--------|--------|----------------|
| More concurrent users | Max sessions at fixed p95 TPOT | `serve_and_bench.py --sweep-concurrency` |
| Longer context windows | Max context at same HBM budget | Sweep `--context-length` until OOM |
| Energy efficiency | Tokens per joule (tok/J) | `serve_and_bench.py` computes automatically |
| TTFT improvement | Warm vs cold on repeated prefix | `run_tiered_experiment.py --use-lmcache` |
| HBM reduction | Peak HBM with tiering vs without | Compare baseline vs tiered results |

## Sources

- `TIERED_KV_ARCHITECTURE.md`: full tiered architecture spec including promotion path and protection policies
- NVIDIA NVFP4 overview: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- NVIDIA NVFP4 KV cache blog: <https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache>
- KVTC paper (ICLR 2026): <https://openreview.net/forum?id=tMiBQXQ0Cm>
- LMCache docs: <https://docs.lmcache.ai/>
- vLLM LMCache examples: <https://docs.vllm.ai/en/latest/examples/others/lmcache/>
- vLLM KV cache offloading: <https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.2/tutorials/kv_cache.html>
