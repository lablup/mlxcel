# PR #2185: Gemma 4 MTP verify byte-identical to decode on CUDA

**Date**: 2026-10-07
**Status**: Implemented and verified on CUDA (GB10); Metal verification pending in #2158
**Risk**: Medium (CUDA verify path for both Gemma 4 pairs; the history-boundary prefill split and the `attend` gate refactor also reach Metal 31B serving)

## Summary

Gemma 4 MTP never engaged on CUDA: the startup block-versus-chain exactness probe failed on the first 8-token draw for `gemma-4-12b-it-4bit` and `gemma-4-31b-it-4bit` with their assistants, so every pairing declined to classic decode. An attribution matrix over `MLXCEL_QMV_MULTIROW` and `MLXCEL_SDPA_VECTOR_LARGE_D`, an `MLXCEL_COMPILED_QGELU_MLP` arm, and a temporary query/key capture inside the attention layer found three causes. The multirow qmv kernel, the issue's main suspect, changed no cell.

1. The per-row verify gate in `Attention::attend` matched only the 31B geometry, so the 12B ran its M=4 block batched and left the single-query `sdpa_vector` kernel classic decode takes at head_dim 256. `TextConfig::mtp_requires_linear_singleton` now also covers the 12B geometry on CUDA (`mtp_requires_linear_singleton_on(cuda)` keeps both answers testable), and `attend` reads a construction-time flag routed through `verify_attention::attend_verify_rows`.
2. The 12B's 8-bit MLP ran the compiled GeGLU graph at decode and the eager activation for a multi-token block. `compiled_gelu_approx_mlp_forward` now takes the compiled graph on CUDA for inputs under 8 rows, the qmv window where the matmul itself is per-row exact.
3. MLX's CUDA RoPE picks `rope_single` for `B=1, L=1` and the general kernel otherwise, and the two differ in the last float bit. Row-wise verify on CUDA (`mtp_row_rope`) runs the Q and K `reshape, norm, transpose, RoPE` chain one row at a time (`head_rows_like_decode`).

Chat output then still diverged on one of three 256-token prompts per pair: classic serving splits a chat prefill at the history boundary when the prompt cache is on (#1143), and the MTP burst bypassed that prefill. The scheduler now reports the boundary (`history_boundary_split`) to the B=1 MTP slice and burst, and the row-wise prefill forwards the same partition (`mtp_prefill_ranges`).

## Design notes

- The 12B pair on CUDA inherits the 31B restrictions the predicate already drives: B=1 linear verify, rotating verify buffer, no tree rounds, no buffered snapshot donation, and the long buffered probe draw.
- The RoPE fix replays decode's own call per row instead of patching MLX's `rope.cu`, which would have moved classic decode numerics for every CUDA model.
- The compiled-GeGLU gate also reaches Gemma 1/2/3 on CUDA for inputs under 8 rows (short prefills, suffix chunks). The existing fallback test now uses 16 rows so it still exercises the fallback.
- A sliding layer without a ring cursor takes the per-row path only while its keys fit the window; past that an unbuffered ring is not in decode order. Serving and the probe always enable the buffer, so this is a fallback.
- `mtp_prefill_ranges` forwards the boundary segment as one chunk and the suffix in `prefill_chunk_size` chunks from the boundary, matching `capture_history_boundary_snapshot` followed by the classic prefill. Batched bursts pass no boundary.

## Verification on GB10 (CUDA, release, `--features cuda`)

Driver 580.178.04, kernel 7.0.0-1019-nvidia, MLX pin `81ba1c6a`.

- Probe with default draws (8, 8, 8, 1056), both pairs, all four `MULTIROW` x `LARGE_D` cells: pass. Before: both pairs failed in every cell.
- Chat, temperature 0, 256 tokens, three prompts per pair, default config: byte-identical with MTP engaged. Before the boundary fix 2 of 3 per pair matched; with `--no-cache-prompt` 3 of 3.
- `speculative_parity --ignored`: `greedy_parity_mtp_gemma4_31b`, `greedy_parity_mtp_gemma4_unified_12b`, both batched Gemma 4 tests and `greedy_parity_dflash_qwen35_4b` pass; Qwen 3.5 DFlash still declines with the #1935 reason.
- Lib tests: gemma4 (250), scheduler burst/slice/prompt-cache selectors (127), and the new core qmv and compiled-GeGLU bitwise tests. Clippy `-D warnings` on both crates and `cargo fmt --check` clean.
- Throughput, three interleaved rounds with a classic-vs-classic null arm, prompt cache off: 12B MTP +106% to +122% over classic (13.2 to 28 tok/s), 31B +80% to +91% (8.3 to 15.8 tok/s), null spread -5.1% to +6.2%, acceptance 0.63 and 0.64, identical output bytes across all arms of each pair. The CPU-quiet host gate was skipped because the shared CI runner stayed busy.

## Not verified on this host

Metal. Shared-path changes listed for #2158: the `attend` gate refactor, the no-cursor sliding fallback, the history-boundary prefill split, and `mtp_requires_linear_singleton` consulting `cuda_is_available()`.

## Follow-up

Qwen 3.5 DFlash's CUDA residual (#1935) may be the same RoPE kernel choice; untested here. The first MTP bonus token still comes from an M=prompt LM-head projection against classic's last-position one; it matched in every run. Classic decode on a repeated prompt depends on prompt-cache state (a whole-prompt hit produced different text than the cold request), which MTP does not reproduce because it never adopts a buffered snapshot.
