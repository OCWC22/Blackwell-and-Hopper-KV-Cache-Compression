# Hopper-Aware KV Cache Compression Runtime
## Hackathon Context Brief for Onboarding Engineers and Coding Agents

## 0) One-line thesis
We are building an **accuracy-preserving, low-latency, hardware-aware KV-cache compression runtime for Hopper-class inference systems**, using **Blackwell/NVFP4 as the design north star** but **not pretending Hopper has native NVFP4**.

The practical goal is to make **Hopper behave more like a lower-precision, memory-richer serving platform** by shrinking KV-cache bytes, reducing offload traffic, and carefully controlling decompression overhead during decode.

---

## 1) What we are actually trying to prove
### Core claim
For long-context inference, the KV cache becomes a first-order systems bottleneck. If we compress KV intelligently and place it across the right memory tiers, we can improve:

- **decode latency**
- **effective GPU memory capacity**
- **PCIe / host-memory pressure**
- **context length or concurrency at the same memory budget**

while preserving model quality.

### The exact angle
This is **not** a generic “KV compression” project.

This is:

> **A Hopper-aware KV runtime that chases Blackwell-like low-precision memory economics without requiring native Blackwell FP4 hardware.**

### What “similar to Blackwell” means here
On Blackwell, NVFP4 is native and hardware-accelerated. On Hopper, it is not. So our Hopper strategy is:

- use **FP8 / FP16 compute paths** that Hopper supports natively
- use **compressed or FP4-like storage representations** for KV when profitable
- **decompress or dequantize late** and only where needed
- keep the hottest / most sensitive KV regions in higher precision

So the north star is:

```text
Blackwell: native FP4/NVFP4 math + storage
Hopper: compressed low-bit KV storage + FP8/FP16 compute
```

---

## 2) Reality check: what is true on Hopper vs Blackwell
### Hopper facts
H100 adds native **FP8 Tensor Cores** and supports FP8 formats `E4M3` and `E5M2`; NVIDIA says FP8 halves storage requirements and doubles throughput versus FP16/BF16 on Hopper.[R1]

### Blackwell facts
Blackwell adds native support for **FP4-family formats**, including **NVFP4**, and NVIDIA describes NVFP4 as a Blackwell-native 4-bit floating-point path with micro-block scaling designed to reduce accuracy loss at ultra-low precision.[R2]

### Implication for our architecture
We must be explicit:

- **Hopper does not have native NVFP4 compute**
- therefore our hackathon architecture for Hopper is **not “run NVFP4 on Hopper”**
- instead, it is **“borrow the system idea behind Blackwell low-precision serving”**:
  - shrink KV bytes aggressively
  - control precision by tier / token / layer
  - hide decompression overhead
  - reduce movement over slower links

---

## 3) Why this matters: KV cache is big enough to dominate serving
KV cache size is large enough to become the memory and bandwidth bottleneck in long-context serving.

NVIDIA states that for **Llama 3 70B**, a **128k-token KV cache for batch size 1** uses about **40 GB** of memory, and that it scales linearly with the number of users.[R3]

That means the serving problem is not just model weights. It is:

```text
weights + KV cache + movement cost + decode-time retrieval latency
```

LMCache and vLLM already attack part of this problem by offloading or reusing KV caches.[R4][R5][R6]

Our job is to see whether **compressed KV storage plus hardware-aware placement** can outperform raw offload approaches on Hopper-class systems.

---

## 4) Baseline mental model
### Today’s common stack
```text
Model runs in vLLM
→ KV cache lives on GPU until pressure rises
→ offload path moves KV to CPU or disk via LMCache / connectors
→ cache is later fetched back or reused
```

vLLM’s production stack explicitly documents KV-cache offloading with LMCache to move KV from GPU memory to CPU or disk.[R6]

LMCache’s own docs position it as a reusable KV-cache layer for vLLM and similar serving engines, with reported **3–10x TTFT / delay savings** in some reuse-heavy workloads.[R5]

### What the baseline still leaves unsolved
LMCache helps with:

- KV reuse
- offloading to cheaper tiers
- reducing recomputation

But the core limitations remain:

- **raw KV bytes are still large**
- **host-side movement can still be expensive**
- **decode may stall on retrieval / decompression / rehydration**
- **the hottest tokens and the coldest tokens do not deserve the same treatment**

---

## 5) What our runtime adds
### High-level idea
We insert a **compression-aware, latency-aware KV policy layer** between the serving engine and the underlying memory tiers.

### Proposed architecture
```text
LLM decode / attention
        │
        ▼
KV policy runtime
  ├─ importance scoring
  ├─ precision / compression selection
  ├─ placement decision
  ├─ prefetch scheduling
  └─ decompression / dequant orchestration
        │
        ▼
Memory tiers
  ├─ Tier 0: GPU HBM (hot KV)
  ├─ Tier 1: optional peer / near-GPU tier if available
  ├─ Tier 2: host memory / pinned buffers
  └─ Tier 3: NVMe / durable cache tier
```

### Precision / storage concept
For Hopper-class serving:

- **Tier 0 hot KV**: keep in **FP16 or FP8**
- **Tier 1 warm KV**: optionally keep in **compressed block format** or FP4-like packed storage
- **Tier 2 cold KV**: store in **stronger compression**
- **Tier 3 archive / reusable KV**: store in **max-compression mode** if reuse economics justify it

### Decode path concept
```text
request arrives
→ prefill generates KV
→ runtime classifies KV blocks by importance and recency
→ hot blocks remain resident
→ warm / cold blocks are compressed and placed off the fast path
→ decode requests issue async prefetches
→ only needed blocks are rehydrated just-in-time
→ compute continues on FP8/FP16 Hopper-supported kernels
```

---

## 6) The research inspiration: KVTC
The strongest recent reference point is **KV Cache Transform Coding (KVTC)**.

OpenReview describes KVTC as a transform-coding approach for compact on- and off-GPU KV storage using:

- **PCA-based feature decorrelation**
- **adaptive quantization**
- **entropy coding**

and reports **up to 20x compression while maintaining reasoning and long-context accuracy**, with **40x+ in some use cases**.[R4]

### Why KVTC matters to us
KVTC is important because it changes the framing from:

```text
“Where do we store raw KV?”
```

to:

```text
“How do we store much smaller KV with acceptable rehydration cost?”
```

### But our angle is different
KVTC is the compression building block. Our hackathon angle is the **systems layer around it**:

- when to compress
- how much to compress
- where to place the result
- what to keep uncompressed
- how to hide rehydration latency
- how to tune for Hopper’s actual hardware paths

---

## 7) What “hardware-aware” means in this project
“Hardware-aware” is not a vibe. It means the runtime makes decisions based on real device constraints.

### On Hopper, we care about:
- **HBM capacity pressure**
- **decode-time memory traffic**
- **PCIe / host-path bytes moved**
- **async overlap between compute and rehydration**
- **whether decompress/dequant cost erases the bandwidth win**
- **which precision paths are natively fast on Hopper**

### Practical translation
A block should only be compressed if:

```text
saved_bytes / transfer_cost > rehydration_cost + scheduling_overhead + accuracy_risk
```

Not mathematically exact, but this is the right decision rule.

### Hopper-aware policy intuition
- **hot recent tokens** should stay easy to access
- **attention-critical tokens** should stay higher precision
- **older / colder KV** can be aggressively compressed
- **everything depends on whether the block will be needed soon**

---

## 8) What we are optimizing for
### Primary objective
**Lower end-to-end decode latency while preserving quality.**

### Secondary objectives
- reduce **KV memory footprint**
- reduce **bytes moved off GPU**
- increase **max feasible context length**
- increase **concurrency** at a fixed HBM budget
- reduce **TTFT / refill penalty** when reusable KV is offloaded

### Optimization targets
We should measure and optimize:

1. **Decode latency**
   - p50
   - p95 / p99
   - per-token latency under long contexts

2. **Memory footprint**
   - GPU HBM used by KV
   - host memory used by KV
   - compressed bytes per token

3. **Traffic**
   - bytes transferred over host path
   - refill / reload volume
   - prefetch hit rate

4. **Quality**
   - benchmark score delta versus baseline
   - long-context degradation
   - reasoning degradation

5. **System overhead**
   - compress cost
   - decompress / dequant cost
   - metadata / index overhead

---

## 9) Accuracy strategy: where we cannot be sloppy
If compression destroys attention fidelity, the project is dead.

### Non-negotiable principle
**Not all KV should be treated equally.**

### Blocks we should protect
At minimum, protect:

- **very recent tokens**
- **attention sink / anchor tokens**
- **high-reuse or high-attention blocks**
- any layers shown to be unusually sensitive

### Working policy idea
```text
critical blocks   → FP16 / FP8, stay near compute
warm blocks       → lightly compressed / low-bit packed
cold blocks       → aggressively compressed
archive blocks    → strongest compression allowed
```

### Why this is justified
KVTC’s reported success comes from exploiting redundancy while maintaining long-context accuracy, but our runtime still needs a **safety policy** around what not to crush.[R4]

---

## 10) The exact Hopper-vs-Blackwell framing we should use publicly
Use this framing consistently:

> **We are targeting Hopper-class serving constraints today, while using Blackwell’s native low-precision serving model as the architectural north star.**

Say this, not this:

### Good
- “Hopper-aware compressed KV runtime”
- “Blackwell-inspired low-precision KV architecture”
- “NVFP4-like memory economics for Hopper-class serving”
- “compressed KV storage on Hopper, native FP4 path on Blackwell later”

### Bad
- “We run NVFP4 on Hopper”
- “This is a Blackwell-only project”
- “We just do quantization”

---

## 11) What we think the benchmark should look like
### Baselines
At minimum compare against:

1. **vLLM baseline without our runtime**
2. **vLLM + LMCache / KV offload baseline**
3. **our runtime with compression disabled but placement logic enabled**
4. **our full runtime**

### Workloads
Use long-context workloads where KV actually matters:

- **32k / 64k / 128k** contexts
- long multi-turn chat
- code editing / reusable prefix cases
- retrieval-heavy prompts

### Models
Prefer models relevant to the hackathon and easy to run in the available stack.

### Metrics to record
- decode latency
- TTFT
- throughput
- HBM usage
- host memory usage
- bytes moved during refill / offload
- quality delta

---

## 12) Why SLURM access matters
We need SLURM because this is a **real systems experiment**, not a notebook trick.

SLURM access lets us:

- run **multi-GPU jobs** repeatably
- test **long-running long-context workloads**
- compare baselines under consistent resource allocation
- collect meaningful latency, memory, and throughput traces
- stress the runtime under more realistic contention and scheduling conditions

The point is not “we need random GPU time.”
The point is:

> **we need controlled cluster execution to evaluate whether hardware-aware compressed KV actually beats raw offload strategies for Hopper-class serving.**

---

## 13) Concrete engineering workstreams
### Workstream A — KV block model and policy engine
Build the abstraction for:

- KV block metadata
- temperature / hotness
- recency
- estimated sensitivity
- placement target
- compression mode

### Workstream B — Compression backends
Implement pluggable storage backends:

- no compression
- light quant / pack mode
- FP4-like packed mode for storage only
- stronger transform-coded mode

### Workstream C — Rehydration path
Build async path for:

- prefetch
- dequant / decompress
- host-to-device staging
- overlap with decode

### Workstream D — Measurement harness
Need instrumentation for:

- bytes written / read by tier
- block hit / miss rate
- prefetch timing
- compression ratio
- decode latency breakdown

### Workstream E — Evaluation harness
Automate experiments across:

- context length
- batch size
- policy ablations
- protected-token window sizes
- compression strengths

---

## 14) Candidate first implementation path
### Phase 1 — Prove policy + measurement
- use existing vLLM + LMCache baseline
- add a shim/runtime to track KV block decisions
- start with simple tiering + instrumentation
- no fancy codec yet

### Phase 2 — Add lightweight compression
- add packed / low-bit block storage
- rehydrate back to FP8 / FP16 compute path on Hopper
- measure real latency trade-off

### Phase 3 — Add stronger compression mode
- integrate a KVTC-like or transform-coded cold-path mode
- use it only for cold / reusable KV
- protect hot / recent / sensitive blocks

### Phase 4 — Add async overlap and smarter policy
- prefetch sooner
- overlap decode and rehydration
- tune thresholds to beat the baseline on actual workloads

---

## 15) Risks and failure modes
### Risk 1 — Compression saves memory but hurts latency
If decompression / staging is too expensive, the system loses even if memory footprint improves.

### Risk 2 — Accuracy cliffs
If the wrong tokens or layers are compressed too hard, long-context quality drops fast.

### Risk 3 — Metadata overhead / fragmentation
Too many tiny blocks or complex metadata can kill practical gains.

### Risk 4 — Wrong benchmark
If workloads do not stress KV enough, results will look inconclusive.

### Risk 5 — We accidentally build a storage demo, not a serving win
The success criterion is not “compression ratio.” It is **serving improvement with acceptable quality**.

---

## 16) Success criteria
We should define success as something like:

- measurable KV memory reduction
- measurable reduction in host-path bytes moved
- neutral or improved decode latency at relevant long contexts
- minimal quality regression on chosen evals
- clear explanation of why this is especially relevant for Hopper-class serving

If we only get a better compression ratio with worse serving, that is not success.

---

## 17) The pitch in one paragraph
We are building an **accuracy-preserving, low-latency, hardware-aware KV-cache compression runtime for Hopper-class inference systems**. The core idea is to treat Blackwell’s native FP4/NVFP4 serving model as the architectural north star, but implement a practical Hopper version using **FP8/FP16 compute, compressed low-bit KV storage, selective protection of sensitive blocks, and tier-aware placement across GPU and host memory**. We want to prove that **compressed KV plus smart placement and rehydration** can beat raw KV offload approaches on the workloads that matter: long-context, decode-heavy, memory-bound serving.

---

## 18) Short prompt you can paste into an agent
```markdown
We are building a Hopper-aware KV-cache compression runtime for long-context LLM serving.

Goal:
- preserve accuracy
- reduce decode latency
- reduce KV memory footprint
- reduce offload traffic

Important truth:
- Hopper has native FP8, not native NVFP4
- Blackwell/NVFP4 is the design north star, not the immediate hardware assumption
- On Hopper, we use compressed or FP4-like storage for KV when profitable, then rehydrate into Hopper-supported FP8/FP16 compute paths

Baseline:
- vLLM
- vLLM + LMCache / KV offload

What our runtime adds:
- per-block KV policy
- precision / compression selection
- tier-aware placement
- async prefetch + rehydration
- instrumentation for latency / memory / traffic / quality

Primary success criterion:
- serving win, not just compression ratio
```

---

## 19) Source notes
### [R1] NVIDIA Hopper Architecture In-Depth
- H100 adds native FP8 Tensor Cores
- FP8 formats E4M3 and E5M2
- FP8 halves storage and doubles throughput vs FP16/BF16 on Hopper
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

### [R2] Introducing NVFP4 for Efficient and Accurate Low-Precision Inference
- Blackwell introduces NVFP4
- NVFP4 uses E2M1 plus scaling strategy for improved low-precision accuracy
- https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

### [R3] Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing
- Example: Llama 3 70B, 128k context, batch size 1, about 40 GB KV cache
- Grace Hopper / Grace Blackwell NVLink-C2C example and unified memory context
- https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/

### [R4] KV Cache Transform Coding for Compact Storage in LLM Inference
- KVTC uses PCA decorrelation, adaptive quantization, entropy coding
- reports up to 20x compression while maintaining long-context and reasoning accuracy
- https://openreview.net/forum?id=aNVKROYpLB

### [R5] LMCache documentation
- reusable KV-cache layer for serving engines
- docs claim 3–10x delay savings in some workloads when combined with vLLM
- https://docs.lmcache.ai/

### [R6] vLLM production-stack KV cache offloading docs
- documents LMCache-based KV offloading from GPU memory to CPU or disk
- https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.2/tutorials/kv_cache.html
