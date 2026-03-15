<PROMPT>
Does TensorRT have an implementation for dual-batch overlap so that you can do the forward pass and load a kvcache from a preemption at the same time? 



<REPLY>
Yes, but not in the exact “generic preemption-resume dual-batch overlap” sense.

TensorRT-LLM has **three related mechanisms**, and they are easy to mix up:

1. **Overlap Scheduler**: this is **not** KV-cache load overlap. It overlaps **CPU-side scheduler/postprocessing work for step *n*** with **GPU compute for step *n+1***. NVIDIA’s docs explicitly describe it as overlapping CPU tasks like stop-criteria checks and scheduling with GPU computation, and it is enabled by default. ([NVIDIA GitHub][1])

2. **Disaggregated serving KV-transfer overlap**: this is the closest built-in thing to what you’re asking. In disaggregated mode, TensorRT-LLM can **overlap KV-cache transmission with model inference**, but the docs are careful: the overlap is for **multiple independent requests**. While one request is sending/receiving KV blocks, other requests can compute. There is also an env var, `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP`, and by default it is **off = overlap enabled**. ([NVIDIA GitHub][2])

3. **KV Cache Connector API**: this is the lower-level hook for offload/reload. The connector API has async-oriented methods like `start_load_kv`, `wait_for_layer_load`, and `get_finished`, and the scheduler API can report whether loading is async via `get_num_new_matched_tokens(...)->(num_tokens, is_async)`. That means TensorRT-LLM’s architecture does support **layer-aware/background KV loading coordinated with forward execution**. ([NVIDIA GitHub][3])

The catch: the **reference example is not production-grade overlap**. NVIDIA’s own example connector uses synchronous `torch.load` / `torch.save`, and the docs say real implementations should move that to background threads or async I/O so the GPU doesn’t stall. So the framework supports it, but the sample implementation does **not** give you a turnkey high-performance preemption-resume path. ([NVIDIA GitHub][3])

So the practical answer is:

* **If you mean normal in-process serving:** **No, not as a documented built-in “resume a preempted request while simultaneously doing that same request’s forward pass” feature.** The standard overlap scheduler is not that. ([NVIDIA GitHub][1])
* **If you mean TensorRT-LLM has primitives to do it:** **Yes.** The KV Cache Connector interface is clearly designed so a worker can start loading KV, then gate per-layer execution with `wait_for_layer_load`, which is exactly the sort of machinery you’d use for overlapped restore. ([NVIDIA GitHub][3])
* **If you mean a shipped built-in overlap path today:** **The closest official built-in is disaggregated serving**, where generation can overlap KV transfer with inference for other independent requests, not a generic preemption-reload feature on a single aggregated executor. Also, NVIDIA notes that overlap scheduler is **not supported for disaggregated context servers yet**, which shows there are still scheduler limitations around these paths. ([NVIDIA GitHub][2])

So the clean CTO answer is:

**TensorRT-LLM has the plumbing, and it has a built-in KV-transfer/inference overlap path in disaggregated serving, but it does not appear to expose a polished, first-class “preempt request → reload KV → overlap reload with same-request forward” feature in standard serving out of the box.** You’d likely implement that on top of **KV Cache Connector** or via the **disaggregated generation executor** path. ([NVIDIA GitHub][3])

If you want, I can map this directly to LMCache/vLLM-style preemption semantics and show what TensorRT-LLM is still missing.

[1]: https://nvidia.github.io/TensorRT-LLM/features/overlap-scheduler.html "Overlap Scheduler — TensorRT LLM"
[2]: https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html "Disaggregated Serving — TensorRT LLM"
[3]: https://nvidia.github.io/TensorRT-LLM/features/kv-cache-connector.html "KV Cache Connector — TensorRT LLM"



Not in the way you’re asking.

**Answer:** **Dynamo does not currently expose a “dual-batch overlap” path for TensorRT-LLM where one batch is doing forward while another batch is restoring/loading KV cache from a preemption at the same time.** The closest thing Dynamo documents is **disaggregated KV transfer overlap**, and even there the behavior is backend-specific: **SGLang** can let decode begin while KV transfer continues in parallel, but for **TRTLLM** and **vLLM** the docs say prefill is synchronous and decode waits before proceeding. ([GitHub][1])

A few important distinctions:

* **Dynamo-level overlap:** Dynamo’s disaggregated serving design says KV transfer is non-blocking at the framework level, but the backend section explicitly says only **SGLang** supports the “background task / begin decode immediately while transfer proceeds” behavior. **TRTLLM** uses `opaque_state`, and Dynamo says **decode waits for prefill to complete**. ([GitHub][1])
* **TensorRT-LLM overlap scheduler:** TRT-LLM release notes mention an **overlap scheduler** that overlaps **prepare inputs and model forward by decoding 1 extra token**. That is a decode-path scheduling optimization, **not** a documented mechanism for overlapping a forward pass with **KV-cache reload/onboarding from preemption or host/offloaded storage**. ([NVIDIA GitHub][2])
* **Secondary/offloaded KV in TRT-LLM:** TRT-LLM exposes a secondary memory pool / host cache and an `onboardBlocks` behavior for bringing offloaded blocks back before reuse, but I did **not** find documentation saying this onboarding is overlapped with a separate active forward batch the way you described. The executor docs expose the knobs and timing metrics for KV transfer, but not that specific concurrent restore+forward workflow. ([NVIDIA GitHub][3])

So the clean CTO answer is:

**For Dynamo + TRTLLM today: no documented implementation of “forward on one batch while simultaneously reloading a preempted request’s KV cache” as a first-class feature.**
**For Dynamo + SGLang disaggregated serving: yes, there is documented overlap of decode with KV transfer, but that is inter-worker KV transfer overlap, not clearly the same thing as preemption restore/onboard from secondary storage.** ([GitHub][1])

If you want, I can do the next step and audit **where this would have to be implemented** across:

1. Dynamo router/scheduler,
2. TRT-LLM executor / batch manager,
3. KVBM or LMCache / secondary-memory restore path,
4. CUDA stream/event scheduling for true restore-vs-forward overlap.

[1]: https://github.com/ai-dynamo/dynamo/blob/main/docs/design-docs/disagg-serving.md?utm_source=chatgpt.com "dynamo/docs/design-docs/disagg-serving.md at main"
[2]: https://nvidia.github.io/TensorRT-LLM/release-notes.html?utm_source=chatgpt.com "Release Notes — TensorRT LLM"
[3]: https://nvidia.github.io/TensorRT-LLM/_cpp_gen/executor.html?utm_source=chatgpt.com "Executor — TensorRT LLM"
