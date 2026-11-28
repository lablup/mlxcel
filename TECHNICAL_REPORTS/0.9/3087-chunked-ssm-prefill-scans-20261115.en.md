# Technical Report: PR #3087 - Chunk the Mamba2 SSD and block the Mamba1 graph scan

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Medium

## Executive Summary

Falcon-H1, GraniteMoeHybrid and PLaMo 2 computed the Mamba2 SSD over the whole prefill piece, holding several f32 `[B, H, L, L]` segsum copies per layer. A 28K single pass of granite-4.0-h-tiny, falcon-h1-tiny or plamo-2-1b therefore ran out of memory on an M5 Max. All three now call a shared `chunked_ssd_scan` with a 256-token chunk. The single passes complete (17.10, 9.99 and 25.62 GB), and at 2048-token chunks the peak drops (9.85 to 7.46, 3.99 to 2.34, and 12.21 to 9.90 GB) while prefill gets 1.6 to 2.7 times faster. Jamba's graph-fallback selective scan now builds its state increments per 256-step block instead of for the whole piece.

## 1. Problem Statement

`ssm_step` in falcon_h1 and granitemoehybrid, and the plamo2 scan, used the unchunked SSD formulation. Its working set grows with the square of the piece length: at 128 heads and L = 2048, one copy is about 2 GiB. `mamba2::ssm_attn` and nemotron_h already chunked theirs (#3073).

## 3. Technical Decisions

- **One shared scan.** `src/models/ssd_scan.rs` replaces three duplicated `segsum`/`repeat_axis` implementations. NemotronH's `nemotron_h_ssm_attn` becomes a thin wrapper over it with per-chunk evaluation, so its behavior is unchanged. The 256-token chunk is mlx-lm's `ssm_attn` default, and it is independent of the prefill piece.
- **#3063 behavior is preserved.** Final states are still scheduled with `schedule_prefill_state`, and conv tails still use `materialized_state_tail`. A single pass longer than 2048 tokens also schedules each window's output and state.
- **Arithmetic order changes.** The chunked scan sums in a different order than the whole-piece scan. Teacher-forced logit traces over 16,896 positions per model (forward width 2112) put decided-position disagreement at:
  - granite-4.0-h-tiny: 4 of 16,148 (0.025%)
  - falcon-h1-tiny: 0 of 12,168
  - plamo-2-1b: 0 of 9,400

  Perplexity moves by -0.098% to +0.007%. Greedy output at 2048 chunks is byte-identical for Granite and PLaMo 2. For Falcon-H1 tiny it diverges in output that is already degenerate on both arms.
- **`mamba.rs` is unchanged.** Its graph fallback already projects one timestep at a time, so the `[B, L, d_inner, N]` buffer the issue cites does not exist there.

## 7. Change Summary

| Checkpoint | Single pass, main → branch | 2048-chunk peak, main → branch | 2048-chunk prefill |
|---|---|---|---|
| granite-4.0-h-tiny-4bit | OOM → 17.10 GB | 9.85 → 7.46 GB | 1.6-2.5x faster |
| falcon-h1-tiny-90m-instruct-4bit | OOM → 9.99 GB | 3.99 → 2.34 GB | 2.6-2.7x faster |
| plamo-2-1b | OOM → 25.62 GB | 12.21 → 9.90 GB | 1.5-1.6x faster |
| ai21-jamba-reasoning-3b-4bit (control, fused kernel on Metal) | 6.45 → 6.45 GB | 3.25 → 3.25 GB | unchanged |

New tests:
- the chunked scan against the unchunked scan in f32 and f16, with and without prior state;
- bit-identical results across evaluation policies;
- Jamba's blocked graph scan, bit-identical, plus a forced-fallback prefill against the kernel.

The family unit tests and `tests/mamba2_hybrid_*` pass. Details are in `docs/benchmark_results/ssm-prefill-scan-chunking-m5max-2026-11-15.md`.

## 8. Follow-up Actions

- Two GraniteMoeHybrid fused-MoE tests (`granite_hybrid_fused_kernel_matches_the_gather_path_at_one_token`, `granite_hybrid_fused_arm_routes_one_token_through_the_fused_kernel`) fail in parallel runs and pass alone. This PR does not touch that path.
