# PR #3455: Raon-1 mini DFlash2 speculative drafting (issue #3436)

**Date**: 2026-11-26

**Status**: PR open (closes #3436). Validated on an M3 Ultra. 34 files, +3794 / -33.

**Risk Level**: Medium (new drafter and target capture; the verify forward of a shipped model changes when a drafter is passed, guarded by the exactness gate)

## Executive Summary

KRAFTON ships a DFlash2 speculator (`KRAFTON/Raon-1-mini-dflash2`) that drafts a block of tokens from the four mHC residual streams of five target layers. This PR adds the drafter to `mlxcel-core`, makes `raon1_mini` a `DFlashTargetModel`, and wires both into `mlxcel generate` and `mlxcel-server` through the existing `--draft-model` / `--model-draft` flags. Greedy drafted text equals plain decode in every run, and width 5 (the default) runs 1.30x to 1.76x faster than plain decode.

## 1. Problem Statement

### 1.1 Background

The drafter is a 5-layer sliding-window transformer (`block_size 8`) with a short grouped conv and a top-16 candidate selector. It reads `[n, 51200]` features: the 4 mHC streams of width 2560 after target layers 1, 6, 11, 16 and 21. It has no embedding or head of its own and borrows the target's.

### 1.2 Existing gaps

- **No feature tap.** `raon1_mini` (#3225) exposed only logits; nothing could read the per-stream residuals mid-forward.
- **Verify was not decode.** Raon's multi-token forward rounds differently from one-token decode (RDA chunked recurrence, batched matmuls), so a verify forward that scores several positions at once could pick a different greedy token than plain decode. The exactness gate (`mtp_exactness_gate`) would then refuse drafting.
- **Sampling.** DFlash in mlxcel was greedy only. Raon's `generation_config.json` samples (`temperature 1.0`, `top_p 0.95`, `top_k 20`), so a greedy-only drafter would rarely run on default requests.

## 2. Technical Decisions

### 2.1 A verify forward that is byte-identical to decode

| Option | Pros | Cons |
|---|---|---|
| Batched verify over all draft positions | Fastest | Different rounding from decode; the exactness gate fails |
| **Per-position RDA and MLA, row-invariant dense products** | Same bytes as one-token decode, so greedy output cannot change | Slower verify than a fully batched one |

RDA conv, recurrence and MLA attention step one position at a time inside the verify forward. The dense products (RDA projections, the mHC selector, router, SiTU MLPs, head) go through core `row_invariant`, which gives the same bytes for a row whether it is computed alone or in a block. `MLXCEL_RAON_ROW_INVARIANT=1` routes every forward through the row-invariant kernel, which drafts faster (about 2.1x on the code prompt) but changes plain greedy text, so it stays off.

### 2.2 Sampled verify as the speculative rule

The draft is deterministic, so a sampled request does not need the probability-ratio acceptance test. The round loop samples each verify row with the same sampler and token history plain decode would use, and accepts while the sample equals the draft. Each emitted token is then a plain sample from the target's distribution. This lives in `drafter/dflash/sampled_verify.rs` and is opt-in per drafter (`Drafter::supports_sampled_verify`). Samplers that carry state between calls (Mirostat, adaptive-p) are declined: the server falls back to classic decode and the CLI refuses the combination, because per-row sampler calls would update that state differently from plain decode (a review finding, fixed in `3523d76a4`).

### 2.3 Capture only the drafter's window at prefill

The first version captured `[L, 51200]` features for every prompt row. On a long prompt that is gigabytes for a drafter that reads 2048 positions. `Drafter::context_window`, `Drafter::skip_context` and `Raon1MiniModel::limit_prefill_capture` restrict the prefill tap to the last window; other `DFlashTargetModel` implementations get a no-op hook (security review MEDIUM, `ad9d715ee`). MLX slices share their source buffer, so the full-length tap tensors live until the first draft and peak memory drops by about half, not to zero.

## 3. Implementation Details

- `src/lib/mlxcel-core/src/drafter/raon_dflash2/`: config validation with named errors (mismatched `target_layer_ids` or stream count, i32 overflow in dimension products), sanitize that refuses the vendor NVFP4 key format, grouped block conv that never crosses a block boundary, non-causal attention over grouped KV heads, the selector chain. A feature-width mismatch is `DraftFailed`, not a panic; the `add_context` panic is now an error.
- `src/models/raon1_mini_speculative.rs`: tap capture in layer then stream order, the decode-identical verify forward, rollback that restores recorded RDA states (`RdaStep`, `forward_recording` in `raon_rda.rs`) and trims the MLA cache. The request's reasoning-effort alpha (#3435) is carried in the burst caches; server runs at high and low effort each match their plain output and differ from each other.
- Small edits to shared Raon modules: `raon_rda.rs`, `raon_mhc.rs`, `raon1_mini_moe.rs`, `raon1_mini.rs`, `raon1_mini_lm.rs`.
- Width defaults to 5; widths above the trained 8 are refused.
- Docs: `docs/supported-models.md`, `docs/environment-variables.md`, `CHANGELOG.md`, benchmark doc.

## 4. Validation

| Check | Result |
|---|---|
| Greedy, offline, 3 prompts (short, code, 3137 tokens) x widths 3/5/7 x 2 rounds | 21 of 21 equal to plain decode |
| Greedy, server, effort high and low | each equal to plain server output |
| After the prefill-window change, width 5, short and 3284-token prompts | still equal |
| Sampled distribution | Markov round-loop tests match the joint distribution at widths 2 and 4; presence penalty sees accepted drafts |
| Rollback after a partial accept | equal to a fresh decode |

Speed-up over plain decode (ABAB under the GPU lock, `docs/benchmark_results/raon-dflash2-m3ultra-2026-11-26.md`):

| Prompt | Width 3 | Width 5 | Width 7 |
|---|---|---|---|
| Short | 1.39x | 1.30x | 1.10x |
| Code | 1.60x | 1.76x | 1.62x |
| Long | 1.76x | 1.69x | 1.43x |

Acceptance ranges from 0.26 to 0.74. Sampled width 5 ran at 1.51x (one run). The full local `make verify` gate passed on babf65660, and the repository contract targets passed after the merge with main.

## 5. Follow-up

- Concurrent batches decode classically: there is no batched DFlash for this target.
- The drafter is not checked against the NVFP4 exports from #3453; the exactness gate decides at startup.
- Sampled verify syncs to the host once per verify row.
- The existing distribution-preservation test named in the issue was not run against the sampled path; new Markov tests cover it instead.
- CUDA and ROCm were not run.
