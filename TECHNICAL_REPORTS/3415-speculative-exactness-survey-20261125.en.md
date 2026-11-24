# PR #3415: Speculative exactness survey and a byte-exact Inkling verify block

**Date**: 2026-11-25
**Status**: Draft PR open (closes #3409). Validated on an M3 Ultra (512 GB) on 21 local checkpoints; narrow lib suites, clippy and the contract checks pass

## Executive Summary

PR #3405 made an Inkling 2 verify block equal decode bit for bit, and found that the 8-token startup probe could not see two of the five causes. This PR checks the other drafter-capable families for the same causes, turns the two one-off diagnostics into a shared helper, fixes the one family whose cause is dense gemv/gemm (Inkling), and gives the shared probe a long-context draw for the families that have a sliding window.

The survey result is mostly reassuring. Every quantized family with an exactness gate is exact on the narrow quantized kernel at widths 2, 4 and 8, after short prompts and after 600 and 2100 tokens. Inkling is now exact as well. Gemma 4 26B A4B still declines, in its MoE feed-forward. The ungated token-only verify round (prompt lookup) is inexact on every model it takes, as its documentation says.

## 1. Survey

Method, tables and skipped checkpoints are in `docs/benchmark_results/speculative-exactness-survey-m3ultra-2026-11-25.md`. The headline verdicts:

| Family | Checkpoints | Verdict |
|---|---|---|
| Qwen 3.5 (DFlash, MTP) | qwen3.5-4b-4bit, qwen3.8-27b-4bit | exact on the narrow kernel, all widths and lengths |
| qwen4_exp (MTP) | Qwen3.8-Flash-Next-4bit-mtp | same |
| Gemma 4 (MTP) | 12B 4-bit, 31B 4-bit | same |
| Gemma 4 MoE (MTP) | 26B A4B 4-bit | declines on both kernels; layer 0 MLP output on narrow |
| GLM-4.7-Flash (MTP) | glm-4.7-flash-4bit | exact on the narrow kernel |
| LFM2 (DSpark) | d1-3b-mlx-8bit | exact on the narrow kernel |
| Inkling (MTP) | inkling-0.6b bf16 | declined at layer 0 before; exact on the default kernel after |
| Token-only round | DeepSeek V3 (4), xing4_0 (2), arcee, k2_horizon, OLMoE, GLM | differs at row 0 everywhere; no gate |

Two things follow. The dense gemv/gemm cause only bites where the projections are dense, and Inkling 0.6B is the only gated bf16 checkpoint on disk besides Inkling 2. And no gated family has a long-context cause: none that passes at 8 tokens fails at 600 or 2100. Drafter-on text equals classic text on a prompt past the 2048-token prefill chunk for the four MTP pairs with an offline round loop.

## 2. Shared diagnostic

`models::speculative_exactness_diag` is the Gemma 4 probe trace, moved and widened:

- `capture` / `with_capture`: a thread-local sink. A `capture` call outside a scope is one thread-local read, so it can stay in a serving forward.
- `Stage::of`: the same record built from a tensor a forward already returns, for DFlash targets that hand back per-layer hidden states.
- `locate`: the first stage, then the first row, that differs, with byte counts. A malformed trace is an error, never equality.
- `report`: one line per stage with every row's differing byte count.

Users: the Gemma 4 startup probe (unchanged behavior), the Inkling startup probe (new: a declined probe now names the layer), the DeepSeek V3 layer loop, the survey test and the Inkling 2 per-visit test.

## 3. Inkling

The diagnostic put the first divergence at the layer 0 output, row 0. Four causes:

| Cause | Fix |
|---|---|
| gemv for one row, gemm for several, in every dense product | `row_invariant::linear` / `dense` inside a model scope |
| Short convolution: window times taps at one token, `conv1d` for several | the one-token form per position inside a verify block |
| Routed experts change kernel with the token count | `row_invariant::per_position` over the MoE |
| Past the window, block row r sees r extra masked old keys | per-row attention from each row's window start |

The attention lives in `mlxcel_core::inkling_layer`, which the Inkling MTP drafter also uses. The helpers are plain `matmul` outside a scope, so the drafter is unchanged. The verify forward marks its block with an explicit scope (`verify_block_scope`) rather than the speculative append scope, because Inkling's verify does not enter that scope.

Decode with the kernel is 244.4 tok/s against 239.3 with MLX `matmul` (8 ABAB pairs, faster in 6, within 1% in 2), so the change is on by default; `MLXCEL_INKLING_ROW_INVARIANT=0` is the A/B switch.

A side finding: `row_invariant_matvec` took its inputs through `device` pointers, and MLX binds an array of fewer than 8 elements as `constant`. The tiny Inkling test model (hidden size 4) made the Metal library fail to build and aborted the process. The kernel now indexes its inputs; the arithmetic is unchanged.

## 4. Long-context draw

`models::speculative_exactness_draws` owns the plan every family probe runs: the short draws, then one draw of `sliding_window + 32` tokens when the family declares a window. Inkling, Laguna and Muse Glimmer declare one; Gemma 4 keeps the draw it had, now through the shared plan. Families without a window get the same plan as before, so their startup cost is unchanged, and a test counts the forwards to hold that. `MLXCEL_MTP_PROBE_LONG_CONTEXT` drops the draw (`0`) or forces a length on every family.

## 5. Tests

- `a_seeded_mismatch_is_reported_at_its_first_layer_and_row`: the helper on a toy forward with one flipped element.
- `the_long_draw_catches_the_sliding_window_case`: a stand-in sliding layer with the Inkling 2 defect passes the short draws and fails the long one at row 1.
- `a_family_without_a_boundary_pays_nothing_for_the_long_draw`.
- `the_verify_block_matches_the_chain_on_metal` (Inkling tiny model, 64-wide heads): equal with the change, different from layer 0 without it.
- `real_checkpoint_probe_fails_without_row_invariant_and_passes_with_it` (ignored, real checkpoint).
- `a_tiny_row_or_weight_does_not_break_the_kernel_build` (core).

## 6. Limits

- MTP throughput on Inkling is not measured: the only published drafter pairs with Inkling-Small, which this run could not load. A verify round now costs more than the inexact one did, and #1549 measured MTP on Inkling-Small as slower than classic decode already.
- The quantized Inkling-Small checkpoints were not probed.
- The long draw on Laguna and Muse Glimmer is covered by their tiny-model suites only.
- Inkling 2's probe still draws short prompts; that code belongs to #3410.
- Gemma 4 26B A4B: which MoE product differs was not traced.
