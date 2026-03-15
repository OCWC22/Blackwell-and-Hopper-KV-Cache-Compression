# Promotion Policy

Use this skill when: designing tier promotion logic, configuring protected token policies, choosing between eager and demand promotion, implementing hit and miss rate logging, or ablating promotion strategies.

## Mental Model

```
Tier 0: FP8 on HBM (active decode consumes from here; or NVFP4 if support-gated)
    ↑ promotion
Tier 1: LMCache-managed cold tier (host RAM / local SSD / remote store; optional KVTC codec)
```

Promotion path: cold tier decode → FP8 restore → active blocks (or → NVFP4 pack if supported)

Promotion cost is paid once on reuse, not every token.

## Promotion Strategies

### Eager Promotion
- Pre-promote on prefix match detection
- Lower latency on cache hit but wastes bandwidth if prefix is not actually used
- Best for predictable workloads with known prefix patterns

### Demand Promotion
- Promote only on cache miss during decode
- Higher first-hit latency but no wasted bandwidth
- Best for diverse workloads with unpredictable reuse

### Hybrid
- Eager for high-confidence prefix matches (system prompts, agent scaffolds)
- Demand for uncertain or low-frequency reuse patterns
- Requires confidence scoring or frequency tracking

## Protection Policies

### Attention Sink Tokens
- First 4 tokens of every sequence stay in Tier 0 uncompressed
- These carry disproportionate attention mass across all layers
- Count is configurable but 4 is the standard default

### Recent Token Window
- Final 128 tokens stay in Tier 0 uncompressed
- Window size is configurable
- Live decode tail is never KVTC-compressed
- KVTC paper ablations show quality improvement from this protection

### What to Compress
- Reused prefixes: system prompts, chat templates
- Stale conversation turns beyond the recent window
- Agent scaffolding tokens
- Cached document prefixes
- Long-context KV that has aged past the recent window

### What to Never Compress
- Live decode tail
- Attention sink tokens
- Actively generating sequence tokens

## Logging Requirements

Every promotion event should log:
- Promotion count per request
- Cache hit and miss rate
- Promotion latency (p50, p95)
- HBM delta before and after promotion
- Protection policy settings active during the run
- Compression ratio of the promoted KVTC blob

## Ablation Matrix

| Ablation | Variable | Values |
|----------|----------|--------|
| A1 | Promotion strategy | eager, demand, hybrid |
| A2 | Recent window | 0, 64, 128, 256 tokens |
| A3 | Sink protection | off, 4 tokens, 8 tokens |
| A4 | Compression ratio | 8×, 16×, 32× |

Run each ablation against the FP8-only baseline (or best practical baseline) measuring latency and quality together.

## Things To Avoid

- Compressing live decode tail
- Promoting without logging latency and hit rate
- Hiding policy configuration inside code instead of explicit config files
- Running ablations without the FP8-only baseline for comparison
- Optimizing promotion latency before having a working end-to-end path
