# Nemotron-H request memory admission

Metal text-only Nemotron-H serving checks a conservative memory estimate before starting each request. The scheduler clamps this family to one active slot and waits for preceding work to finish before sampling active allocations. A request rejected by this guard never runs GPU forward work, and a streaming request receives HTTP 503 before SSE headers are committed. `/health` remains available; smaller requests can run afterwards.

## Budget and errors

The default budget is the minimum positive finite value among the device's recommended working-set size, the **actually configured** wired-memory cap, and the current MLX allocator cap. The `get_wired_limit()` MLX query reports a device recommendation, not the configured wired cap; the runtime records the latter separately.

`MLXCEL_REQUEST_MEMORY_BUDGET=24G` can lower that budget, never raise it or supply a missing runtime budget. Size suffixes follow the existing binary-unit grammar (for example `1.5G`); explicit zero, empty, `none`, negative, nonfinite, and overflowing values fail server startup. Unknown dimensions, an unknown budget, or arithmetic overflow fail request admission closed.

```json
{
  "error": {
    "message": "request_memory_budget_exceeded: estimated_bytes=..., budget_bytes=...; shorten the prompt/output or lower retained cache usage",
    "type": "request_memory_budget_exceeded",
    "code": "request_memory_budget_exceeded"
  }
}
```

The error message reports estimated and budget bytes (an unknown estimate uses `18446744073709551615`; an unknown budget uses `0`). Existing context-length checks take precedence and remain HTTP 400. Memory admission logs include prompt/output lengths, largest-forward span, resident allocations and the component breakdown.

## What is estimated

- Loaded, normalized model dimensions and embedding output dtype, not a checkpoint-specific size table.
- Current MLX **active** memory, including model weights and retained prompt snapshots. The shared Metal Nemotron-H constructor materializes sanitized weights (including fused adapter graphs) in place before serving, including with `--no-warmup`, so a lazy-weight graph cannot masquerade as zero residency. Reclaimable allocator free-cache is neither charged as permanent residency nor subtracted from active memory.
- Full cold attention KV for prompt plus planned output, rounded to allocation granularity, with room for growth copies and snapshots; float32 recurrent state and convolution state. With context shifting disabled, the context stop bounds output growth. With shifting enabled no unsupported trim credit is taken for Nemotron-H's model-owned caches.
- The largest padded forward in the actual `PrefillPlan`, respecting model chunking opt-out and history-boundary splits, for residuals, MLP/MoE work, logits, and a conservative float32 attention-score fallback.
- Float32 SSD linear buffers and bounded internal-chunk scratch. The SSD internal chunk is independent of the server prefill chunk.

Guarded requests currently prefill cold without restoring or donating prompt snapshots, and skip background prompt-cache warmups: snapshot restoration could otherwise perform request work while the previous request is still active. Retained snapshots remain counted in the resident baseline. This deliberately favors safety over cache-hit latency and can reject requests that would fit with optimistic cache reuse or fused kernels. Clear or reduce retained prompt caches, or shorten prompt/output bounds, rather than expecting the override to bypass the device ceiling.

## Limits

This is not a universal OOM guarantee or a device-wide reservation. Other processes, driver allocations, fragmentation, and unexpected kernel workspace can consume memory after admission. CUDA/ROCm/CPU, other model families, multimodal paths, speculative decoding, distributed handoffs, and multi-slot reservations are outside this policy. Both scheduler-backed serving and legacy `--no-batch` serving use the guard. Other unsupported paths retain their existing behavior. Metal Nemotron-H runtime (unfused) LoRA fails at model construction with a clear error: its arbitrary-rank float32 activation workspace is not modeled. Use a fused adapter checkpoint instead. The shared Nemotron-H constructor also eagerly materializes weights and rejects staged runtime adapters when embedded by VLM/VoiceChat; those models do not acquire this request admission policy. The guard does not change model numerics or retry a request after an allocation failure.
