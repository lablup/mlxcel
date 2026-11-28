# PR #3461: Raon-1 mini decode, load and prompt-reuse performance (issue #3458)

**Date**: 2026-11-27

**Status**: PR open (closes #3458). Measured on an M3 Ultra against `main` at `477564e2e`. 24 files, +2025 / -39. CUDA and ROCm were not run.

**Risk Level**: Medium (plain greedy text can change at a near tie; new model-owned cache snapshots; device-side NVFP4 transcode)

## Executive Summary

Six follow-ups from the Raon-1 mini port. Three are kept, one is a validation, and two were measured and not kept. A repeated 2132-token chat turn's TTFT falls from 1.00 s to 0.04 s, single-stream decode rises from 33.5 to 54.2 tok/s, and the ModelOpt NVFP4 load falls from 63 to 70 s to 2.2 s. Batched DFlash2 and a single host read per sampled verify block were measured to lose and are not in the tree.

## 1. Problem Statement

- **No prompt reuse.** Raon's state is RDA conv windows plus f32 recurrent state and MLA latents. The server had no model-owned snapshot for it, so even identical repeats prefilled cold.
- **Slow decode step.** 33.5 tok/s against a roughly 8 ms weight floor. The issue assumed many small dispatches.
- **Slow ModelOpt load.** About 70 s, dominated by host-side work.
- **DFlash2 on NVFP4 unchecked** after #3453 and #3455 merged separately.
- **Batched DFlash2 and sampled-verify host syncs** were open questions from #3455.

## 2. Technical Decisions

### 2.1 The decode bottleneck was one kernel, not dispatch count

Profiling showed the step bound by a one-row mHC selector product that ran MLX's transposed gemv kernel: 255 us against 35 us for the row-major form, at 50 sites per token. Taking the row-major form for one-row products gives 51.5 tok/s alone; compiled mHC site graphs add the rest and are bit-identical to eager. The changed summation order of that product means plain greedy text can differ from earlier builds at a near tie (seen on both NVFP4 exports on one essay prompt, not on the original checkpoint). The Transformers trace still has 0 of 578 decided disagreements on the prefill and decode paths. `MLXCEL_RAON_DECODE_FUSION=mhc` keeps the earlier bytes at 34.5 tok/s.

Fused RDA projections, the issue's suggested fix, were dropped: +0.6 to +0.8% and not bit-identical at the real width (1008 of 1024 prefill trace rows changed).

### 2.2 Snapshots refuse rather than guess

Raon donates and restores exact-prefix snapshots (RDA conv window, f32 state and offset; MLA latent pair). A restore refuses another family, a snapshot missing a layer, and RDA state whose rank, batch or offset does not match, and the request prefills cold instead. The reasoning effort is already part of the cache key (#3445), and a scheduler test confirms another effort misses. A DFlash2 burst stores no snapshot.

To make a later hit equal a cold run, cold server chat now splits its prefill at the history boundary: a miss costs 1.14 s instead of 1.00 s, and cold server text can differ from `mlxcel generate` at a near tie.

### 2.3 Device-side NVFP4 transcode

The 63 to 70 s load was 50 s of host weight copy, 10 s of scale decode and 3 s of re-encode. Canonical triplets are now reinterpreted on the device; a loader that already decoded the scales keeps the host path. Planes are byte-identical to the host transcode, except that scale byte `0xFF` (a NaN) is no longer rewritten to `0x7F`. Resident memory (16.39 GB) and decode speed are unchanged.

### 2.4 Two items dropped on measurement

| Item | Measurement | Decision |
|---|---|---|
| Batched DFlash2 | An exact verify block costs 42 to 59% more for two rows than one, against 9% for a batched decode step; the ceiling is 13 to 22% below classic batched decode at four rows | `supports_batched` stays `false`; no B > 1 code |
| One host read per sampled verify block | 49.5 against 48.7 ms per verify round (rows past the first mismatch are sampled for nothing) | `sampled_verify.rs` unchanged |

Since #3459, the served default `presence_penalty` 1.5 also keeps default requests out of batched speculative windows, so batched DFlash2 would need per-row token history as well as a cheaper exact verify.

### 2.5 Default DFlash2 width 5 to 3

With the faster decode step, width 5 fell to 0.97x plain decode on prose. Width 3 averages 1.18x over nine prompt and checkpoint pairs against 1.11x for width 5.

## 3. Validation

| Check | Result |
|---|---|
| Transformers trace, 4 x 256 corpus | 0 / 578 decided disagreements, prefill and decode paths |
| Greedy drafted vs plain, CLI, three checkpoints | 54 / 54 equal |
| Server, cold / cache hit / drafted | 36 texts, 0 differ |
| DFlash2 on both NVFP4 exports | gate allows widths 3, 5, 7; 36 / 36 drafted runs equal plain |
| Snapshot round trip and refusals, scheduler hit and effort miss, compiled vs eager mHC, device transcode route | unit tests |

The full local `make verify` passed on the branch merged with `d43915faf`; targeted suites, fmt and clippy passed after the final merge with main. The server measurements in the benchmark doc ran before #3459, so at presence penalty 0.

## 4. Follow-up

- Batched DFlash2 (needs a cheaper exact multi-row verify and per-row token history in batched windows): filed separately.
- The second identical request still misses and only the third hits, because the scheduler's idle warm-up supersedes the boundary entry. This is shared scheduler policy: filed separately.
- CUDA and ROCm were not run.
