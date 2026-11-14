# PR #3049: Bound the materializing prefill SDPA on Metal

**Date**: 2026-11-14
**Status**: PR open (closes #3033, supersedes #3034); M3 Ultra verified, M5+ verification pending

## Executive Summary

When MLX cannot run a fused attention kernel for an SDPA call, its graph fallback computes `q @ k^T` over the whole query axis and holds the `[B, heads, L, L]` score matrix in one buffer. mlxcel already split the query axis of such calls to keep that buffer within `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` (#672), but the gate returned early on every non-CUDA build. On Metal a 26.6K-token prompt on a 24-head, head_dim 256 model therefore asked for one 33.9 GB buffer, past the M4 Pro's 30.15 GB Metal limit, and the request failed (#3033).

This PR runs the gate on Metal. Whether a call materializes is asked of MLX itself (`ScaledDotProductAttention::use_fallback`) for the mask mode the call issues, so a fused kernel is never replaced on any GPU generation. Metal's default budget is 256 MiB rather than CUDA's 1024 because MLX keeps about ten chunks alive at once. On an M3 Ultra a 28K single-pass prefill drops from 66.5 to 44.4 GB peak footprint (qwen3.5-4b) and from 114.0 to 78.5 GB (qwen3.8-27b), with identical greedy output and prefill time within 1.2%. Gemma 4, whose calls MLX fuses, is untouched. Verification on M5 or later is still pending.

## 1. Change Summary

| Area | Change |
|---|---|
| `layers.rs` chunk gate | `ChunkGateBackend` (Cuda, Metal, Unchunked) resolved at runtime. The budget is checked first, then the backend rule: CUDA's cuDNN mirrors are unchanged, Metal asks MLX per mask mode, softcap always materializes. A debug line records each chunked call. |
| `layers.rs` budget | Default 1024 MiB on CUDA, 256 MiB on Metal; `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` overrides both. `MLXCEL_METAL_ATTENTION_CHUNK=0` turns the Metal gate off. |
| `attention` / `causal_attention` | The gate runs before the M5 (Metal 4) route, which issues the same SDPA. |
| `window_sdpa.rs` | `sdpa_mode_is_fused` adds causal and unmasked arms to the existing per-thread fused-verdict memo. |
| Tests | `layers_metal_chunk_tests.rs`: per-mode verdict, kill-switch parse, head_dim 256 fp16 chunked parity, and an ignored op-level peak-memory probe. The existing CUDA gate test passes its backend explicitly. |
| Docs and harness | Env var docs, CHANGELOG, benchmark doc with CSV raw data, `scripts/bench/metal_sdpa_chunk/` (CLI and server rerun scripts). |

## 2. Technical Decisions

### 2.1 Ask MLX, per call mode, instead of tabulating

MLX's fused support on Metal depends on head dim, query length, whether `q_len == k_len`, the mask mode, dtype and the GPU generation (NAX on M5 and later). On this M3 Ultra a causal-mode call at head_dim 256 with `q_len == k_len >= 2048` is fused, while an array-mask call over the same shapes is not; on M5 both fuse. A Rust table would drift with every MLX pin bump. The bridge already exposed `use_fallback` through `sdpa_uses_fused_kernel`, so the gate asks it with the exact flags of the call it is about to make. The rejected #3034 design asked about the array-mask arm for every call, which would have replaced the fused causal kernel with chunked fallbacks.

The probe is consulted only after the budget check says the call would be over budget, so decode and small prefill calls never pay for it, and its verdict is memoized per shape. A probe error answers "not fused", which chunks: chunking is row-exact, so a wrong "no" costs speed, never memory or output.

### 2.2 Runtime backend, not the `metal` feature

A plain `cargo build` on macOS builds the Metal backend without the `metal` Cargo feature (#1988), so a `cfg!(feature = "metal")` gate would be inert in the default build. The gate uses `hardware::gpu_backend_kind()`. CUDA keeps its compile-time test so its behavior is byte for byte the same.

### 2.3 A lower budget on Metal

The first measurement at the shared 1024 MiB budget cut one `[1, 24, 28000, 256]` call from 43.5 to only 26.0 GB of MLX memory. MLX lets up to ten command buffers run ahead (`MAX_ACTIVE_TASKS`) and returns a chunk's score buffer to the allocator only when its command buffer completes, so the live transient is about ten chunks, not one. At 256 MiB the same call peaks at 9.8 GB and runs in 1.41 s against 2.08 s unchunked. The cost shows up on the default 2048-token chunked prefill, whose later chunks also fall back: four times as many chunks cost 2 to 3.5% of prefill time there, where 1024 MiB was neutral. Evaluating each chunk synchronously would bound the transient to one chunk, but it adds a host sync per chunk inside the forward, so it was not used.

### 2.4 Gate ahead of the M5 route

On M5 `attention` used to return through `metal4_attention` before any gate. That route issues the same `fast::scaled_dot_product_attention` (or the softcap composite), and MLX still falls back on M5 for some calls (head_dim 512 with an array mask, softcap). Since the gate fires only on a fallback, moving it first changes nothing for fused M5 calls.

## 3. Verification

M3 Ultra, 512 GB, release build. The baseline is the same binary with `MLXCEL_METAL_ATTENTION_CHUNK=0`, which is main's Metal dispatch. Greedy, 48 tokens.

| Case | Before | After | Output |
|---|---|---|---|
| qwen3.5-4b 28707 tokens, single pass (CLI / server) | 66.46 / 66.65 GB | 44.38 / 44.59 GB | identical |
| qwen3.8-27b 28749 tokens, single pass (CLI / server) | 114.05 / 114.21 GB | 78.48 / 78.65 GB | identical |
| gemma-4-12b 28710 tokens (fused, 0 gated calls) | 23.57 GB, 33.65 s | 23.56 GB, 33.56 s | identical |
| Op level `[1, 24, 28000, 256]` array mask | 43.48 GB, 2.08 s | 9.84 GB, 1.41 s | diff 0 |
| 4.5K tokens, budget forced to 16 MiB (qwen, gemma) | | 24 and 88 gated calls | identical |

Finding: on this build the Qwen 3.5 family's full-attention layers pass an explicit causal array mask (`Qwen35Model::forward_internal`), so their single-pass prefill is the unfused call the reporter hit, on both the CLI and server paths. The reference figures quoted in the #3034 review (63.8 GB / 18.7 s and 109.7 GB / 85.5 s) are that unfused baseline. Unit tests: 143 passed across `layers::`, `window_sdpa` and `causal`; clippy and the env-var and benchmark-data checks pass.

## 4. Unverified and Open Items

- **M5 and later.** Expected: the Qwen array-mask calls fuse there and the gate does not fire. The benchmark doc carries the rerun commands and a pending section.
- **The reporter's machine.** The 33.9 GB single buffer is removed (each chunk's scores stay within 256 MiB), but the M4 Pro run itself was not repeated.
- **Remaining peak.** Most of the 44 to 78 GB left is a separate full-sequence transient, with logits as the suspect, tracked in another issue.
- **Chunked-prefill cost.** The 2 to 3.5% prefill cost on the default 2048-token prefill chunk comes from the lower budget. If it matters, raise the budget for that case or skip chunking there once the over-budget margin is small.
- **Server debug lines.** The server's log filter does not surface the gate's debug lines; the server rows rely on peak memory.

## 5. Learning Points

- On Metal, the memory a lazily chunked computation holds is set by how many command buffers MLX lets run ahead, not by the chunk size alone. Measure the peak; do not assume one chunk is live.
- Ask "is this call fused?" for the exact call: the same shapes can give different answers in different mask modes.
- A kill switch that skips the gate before any decision doubles as an exact A/B baseline on the same binary.
