# Gemma 4 MTP on CUDA: why the verify block missed decode, and the fix (GB10, 2026-10-07)

Issue #2160. Before this work the startup block-versus-chain exactness probe failed on the first 8-token draw for both Gemma 4 pairs on GB10, so Gemma 4 MTP never engaged on CUDA. This record holds the attribution, the three causes it found, the fix, and the measurements that show the probe now passes and greedy output matches classic decode.

## Host

NVIDIA GB10 (sm_121), driver 580.178.04, kernel 7.0.0-1019-nvidia, MLX pin `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`, release profile, `--features cuda`, base `origin/main` `0ee13dfe`. Checkpoints under `/home/inureyes/models/mlx/`: `gemma-4-12b-it-4bit` + `gemma-4-12b-it-assistant-4bit`, `gemma-4-31b-it-4bit` + `gemma-4-31b-it-assistant-bf16`. Block width 4, no `MLXCEL_MTP_ALLOW_INEXACT`.

## Attribution matrix

`data/gemma4-mtp-cuda-gb10-2026-10-07/matrix/matrix.tsv` has every cell. The short draw is one 8-token prompt; the localized first divergence comes from the probe's own failure rerun with layer capture.

| Stage | Pair | `QMV_MULTIROW` unset / `=0` | `SDPA_VECTOR_LARGE_D` unset | `SDPA_VECTOR_LARGE_D=0` |
|---|---|---|---|---|
| before | 12B | identical in every cell | layer 0 sliding, attention output, pos 0 | layer 0 sliding, MLP output, pos 0 |
| before | 31B | identical in every cell | layer 6 sliding, attention output, pos 2 | layer 56 sliding, attention output, pos 3 |
| before, `MLXCEL_COMPILED_QGELU_MLP=0` | 12B | identical | layer 0 sliding, attention output, pos 0 | layer 5 full, attention output, pos 0 |
| after attention gate + MLP gate | 12B | identical | layer 44 sliding, attention output, pos 3 | pass (short draw) |
| after attention gate + MLP gate | 31B | identical | layer 6 sliding, attention output, pos 2 | layer 56 sliding, attention output, pos 3 |
| final | 12B | identical | pass, draws 8, 8, 8, 1056 | pass, draws 8, 8, 8, 1056 |
| final | 31B | identical | pass, draws 8, 8, 8, 1056 | pass, draws 8, 8, 8, 1056 |

**`MLXCEL_QMV_MULTIROW` changes nothing in any cell**, before or after: the multirow qmv kernel is per-row bit-identical at these shapes, as its contract states, and step 3 of the issue (fixing or retrying around it) is not needed. `qmv_multirow_matches_per_row_qmv_bitwise_gemma4_12b_mlp_shapes` pins that at the 12B MLP shapes for 4-bit and 8-bit.

The residual that survived the first two fixes moved with `LARGE_D` but was not the fused SDPA kernel: a temporary capture of Q, K, V and the attention core inside the layer (`matrix/probe-capture-q-k-v-sdpa.txt`) put the first divergence on the **query** (12B layer 5 pos 3; 31B layer 6 pos 2, and layer 7 pos 2 with `LARGE_D=0`), before any attention ran.

## Causes and fixes

1. **12B attention gate.** `Gemma4Attention::attend` only took the per-row verify path for the 31B geometry (32/4/512 full, 32/16/256 sliding). The 12B (16 query heads, 8 sliding KV heads at 256, 1 global KV head at 512) ran its M=4 block batched, which on CUDA leaves the single-query `sdpa_vector` kernel classic decode takes at head_dim 256 (`supports_sdpa_vector` needs `q_len < 4`), and picks a different reduction at head_dim 512. Now a flag computed at construction from `TextConfig::mtp_requires_linear_singleton` selects the row-wise path, and the predicate also covers the 12B geometry when CUDA is available. The 12B pair on CUDA inherits the 31B restrictions: B=1 linear verify, rotating verify buffer, no tree rounds, no buffered snapshot donation.
2. **8-bit compiled GeGLU.** Single-token decode runs the compiled gate/up/GELU/down graph for the 12B's 8-bit MLP, while a multi-token block took the op-at-a-time activation (#680 gating), and the two round differently. On CUDA, inputs with fewer than 8 rows (the qmv window) now take the compiled graph too.
3. **RoPE kernel choice.** MLX's CUDA RoPE uses `rope_single` for a row-contiguous `B = 1, L = 1` input and the general kernel otherwise. Measured in float32 on random input, every row of an L=4 rotation differs from the same row rotated alone, in the last bit; RMSNorm rows do not differ. After bf16 rounding that flips a few query or key elements every few layers. Row-wise verify on CUDA now runs the Q and K `reshape, norm, transpose, RoPE` chain one row at a time at `offset + row`, which is the call classic decode makes.

With all three, the probe passes every draw on both pairs, including the 1,056-token buffered draw, in all four `MULTIROW` x `LARGE_D` cells.

## Serving: chat prompts and the history boundary

The first default-config chat parity run (three 256-token prompts, temperature 0, `freegen_parity.py`) matched on two prompts per pair and diverged about 40 tokens into the third (`parity-*-pre-boundary/`). With `--no-cache-prompt` all three matched on both pairs (`parity-*-nocache/`), and through `/v1/completions` with the same token ids the MTP output, classic output and a model-level chain trajectory were identical. A trajectory replay at model level (verify blocks along the classic trajectory, full and partial accepts, 64 to 99 rounds) was byte-identical throughout.

The cause is the prompt KV, not the verify: with the prompt cache on, classic serving prefills a chat prompt in two forwards, the history segment then the suffix (`capture_history_boundary_snapshot`, #1143), and the MTP burst bypasses that prefill and forwarded the prompt in one piece. The third prompt is the longest (40 tokens against 31 and 35), which fits the split's 32-token minimum prefix; the per-prompt boundary was not logged. The scheduler now reports the boundary (`history_boundary_split`, same gates as the capture) to the B=1 MTP slice and burst, and the row-wise prefill forwards the segment as one chunk, then the suffix in `prefill_chunk_size` chunks from the boundary. After that, default-config chat parity is 3 of 3 on both pairs (`parity-*-default/`).

The first bonus token is still sampled from an M=prompt LM-head projection in the MTP prefill against classic's last-position projection. It matched in every run here; it is the same partition class and is noted rather than changed.

## Throughput

THROUGHPUT_PLACEHOLDER

## Not covered here

- Metal. The 12B predicate arm and the RoPE and MLP changes are CUDA-gated; the attention gate refactor, the no-cursor sliding fallback and the history-boundary prefill split reach Metal 31B serving and are listed for the Apple Silicon pass (#2158).
- Qwen 3.5 DFlash still declines on CUDA with the #1935 reason (verified, unchanged). Its residual may be the same RoPE kernel choice; that was not tested.
- No local Qwen 3.8 MTP drafter, so that pairing is unmeasured.
