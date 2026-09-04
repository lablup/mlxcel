# Technical Report: PR #1621 - feat(bench): dispatch speculative_bench MTP past Gemma 4 Unified

**Date**: 2026-09-04
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (both previously-failing pairings measured on M1 Ultra; the M5 Max re-run is pending on that host)
**Languages**: Rust
**Risk Level**: Low (benchmark harness only; no inference path changes, one diagnostic helper widened to `pub`)

---

## Executive Summary

`run_mtp` matched `LoadedModel::Gemma4Unified` and bailed on every other variant with `std::mem::discriminant`, so `speculative_bench` could produce MTP numbers for exactly one checkpoint. Every adapter it needed already existed and the server burst path already dispatched them, so the harness was the only place pretending the support was absent.

The variant match now mirrors `src/server/batch/speculative_burst.rs`, and `REACHABLE_PAIRINGS` gains the Qwen 3.8 pairing. Both pairings that previously failed now produce numbers.

A separate defect found on the way turned out to matter more than the one filed: `resolve_model_dir` could not find any checkpoint on a host whose store is `models/mlx/<name>`, so a full sweep there measured nothing while reporting no error.

---

## 1. Problem Statement

### 1.1 Background

`speculative_bench --sweep` walks `REACHABLE_PAIRINGS` and, for an MTP row, calls `run_mtp`. That function opened with a match that accepted one variant:

```rust
let unified = match &model {
    LoadedModel::Gemma4Unified(u) => u,
    other => anyhow::bail!(
        "MTP bench currently supports a Gemma 4 Unified target; \
         load_model returned a different variant ({:?})",
        std::mem::discriminant(other)
    ),
};
```

### 1.2 What that cost

Two pairings. The Gemma 4 31B MTP row failed at all three K values, not for want of a checkpoint: `models/gemma-4-31b-it-4bit` is `Gemma4ForConditionalGeneration` and loads as `LoadedModel::Gemma4VLM`, while the 12B is `Gemma4UnifiedForConditionalGeneration`. Qwen 3.8 MTP could not be measured at all, and a catalog entry alone would not have helped because the same match rejected the target.

Every piece needed already existed: per-variant `&dyn LanguageModel` selection and per-variant `MtpTarget` adapter selection in `speculative_burst.rs`, the `Gemma4VLMtpTargetAdapter` the 31B target needs, the `Qwen35MtpTargetAdapter` the Qwen 3.8 target needs, `qwen3_5_mtp` drafter resolution, and `LanguageModel` on the enum itself including `release_sequence_state_by_id`.

### 1.3 The defect the issue did not mention

`resolve_model_dir` probed `<CARGO_MANIFEST_DIR>/models/<name>` and then `../mlxcel-internal/models/<name>`. On a host whose checkpoint store is `models/mlx/<name>`, and whose sibling checkout has the same layout, neither probe hits. Every sweep row then fell through to the "checkpoint missing on disk" skip, so the sweep completed and measured nothing while reporting no error at all. That is a worse failure mode than the one filed, because it is silent.

---

## 2. Technical Decisions

### 2.1 One generic timed run, not one per variant

`MtpGenerator` is generic over `MtpTarget`, so the warm-up burst, the timed generate, the acceptance capture and the sequence release moved into a single `run_mtp_timed` generic over `T: MtpTarget`, which each arm calls with its own adapter constructor. Only the `MtpTarget` construction is variant-specific. The drafter is bound against the enum's own `LanguageModel` impl and released through the same reference, so no arm carries its own copy of the lifecycle.

### 2.2 `pub`, not `pub(crate)`, for the shared label

The plan had been to widen `model_variant_label` to `pub(crate)`. That does not work: `speculative_bench` is a separate crate from the `mlxcel` library, and `pub(crate)` stops at the library boundary. The function is now `pub` with a re-export at the crate root, documented at both sites. The alternative was a second copy of the label table in the bench, which would drift from the server's.

The exported surface is one `fn(&LoadedModel) -> &'static str` returning a diagnostic name, so the addition is small and non-breaking. The shared table reports anything outside the speculative-capable set as `other`, so the bench's unsupported message names the target directory alongside the label; widening that table was out of scope.

### 2.3 The unsupported-variant check runs before any drafter IO

As first written the match sat after `validate_target_compat`. An unsupported target with a mismatched hidden size therefore failed with a *drafter* message, and the acceptance criterion's "name the loaded variant and the supported families" message was unreachable for exactly the inputs it existed to describe. The check now runs before the drafter is loaded, the same ordering the burst path uses.

### 2.4 Every failure stays a row status

A drafter that loads but fails `validate_target_compat`, a missing checkpoint, `block_size < 2`, and the Metal-only guard for Qwen 3.5 MTP exactness all record a per-row status string rather than aborting the sweep. A CUDA sweep records the reason instead of emitting a number the runtime cannot vouch for.

---

## 3. Validation

Apple M1 Ultra 128 GB, Metal, mlxcel 0.7.0-beta.1, MLX pin `9a795735`, `--kind mtp --block-size 4 --max-tokens 128`. Both of these fail on the pre-change binary.

```
gemma-4-31b-it-4bit + gemma-4-31b-it-assistant-bf16
  tok/s=14.8  rounds=34  acceptance_rate=0.529  mean_accepted_len=1.59

qwen3.8-27b-4bit + qwen3.8-27b-mtp-4bit
  tok/s=16.9  rounds=47  acceptance_rate=0.504  mean_accepted_len=1.51
```

The full `--sweep --batch 1 --max-tokens 128` completes 16 rows: 4 baselines, 9 measured MTP rows at K=2/4/8, 3 DFlash rows still deferred on their own known blocker, and no row carrying a status beginning `MTP bench currently supports`. A `gemma-3-4b-it-4bit` target under `--kind mtp` reports the loaded variant, the target directory and the supported families as a row status rather than aborting.

### 3.1 The result is a negative one, and it is the host

| Pairing | Baseline | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| Gemma 4 31B + MTP assistant | 19.9 | 18.4 | 14.9 | 14.9 |
| Gemma 4 Unified 12B + MTP assistant | 38.4 | 36.0 | 28.5 | 29.3 |
| Qwen 3.8 27B + MTP head | 24.9 | 22.5 | 17.2 | 9.3 |

Every ratio is below 1.00x, where M5 Max reads 1.57x at K=4 for the Unified 12B pairing. Acceptance is not the explanation: the same pairing accepts *more* on M1 Ultra than on M5 Max at the same K (39.6% and mean accepted length 1.19, against 35.0% and 1.05) and still lands at 0.74x.

The verify round is the difference. `docs/benchmark_results/speculative-decoding-m1ultra-2026-08-19.md` measures a block-4 verify round at 2.70 classic decode steps on M1 Ultra against 1.50 on M3 Ultra and 1.27 on M5 Max, and identifies M1 Ultra as the first host on Apple GPU generation 13. A mean accepted length near 1.2 cannot clear break-even against a 2.70-step verify. The static gate that declines B=1 MTP on this generation exists for the same reason, so these rows confirm a known property of the host rather than reporting a regression.

### 3.2 Gates

`cargo test --workspace --profile test-fast --features metal,accelerate`: 10507 passed, 0 failed. `cargo clippy --bin speculative_bench --features metal,accelerate -- -D warnings` and `cargo fmt --all -- --check` clean. All 13 CI jobs pass.

One full-workspace run before the final one failed on `vision::inkling_vl::tests::mixed_prefill_scatter_order_is_normalized_text_then_image_then_audio`, a file this PR does not touch. It passes 3 of 3 in isolation and the whole `--lib` binary passes twice at the same test count, so it is intermittent and unrelated. Filed separately rather than absorbed here.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed (code PR) | 4 |
| Lines added | 268 |
| Lines removed | 65 |

### Changes by category

- `src/bin/speculative_bench.rs`: per-variant adapter dispatch, the generic `run_mtp_timed`, the Qwen 3.8 pairings, the hoisted unsupported-variant check, the `models/mlx/<name>` probes in `resolve_model_dir`.
- `src/server/batch/speculative_burst.rs`: visibility of `model_variant_label` only.
- `src/lib.rs`: the crate-root re-export.
- `docs/benchmarks.md`: the supported-target set.

### Landed separately

`benchmarks/metal_m1ultra_spec_2026-09-04.csv` (16 rows) and the `docs/benchmark_results/model_tests.md` update are on `bench/0.7.0-refresh` (PR #1617) as commit `f4ff37522`, because benchmark artifacts belong to that branch rather than to a code PR. That split means this PR can merge and close the issue while the documentation half is still unmerged; the PR body says so explicitly.

### Related issues

Closes #1613. Related: #154 (Gemma 4 Unified MTP drafter), #1165 (`qwen3_5_mtp` drafter and its Metal exactness gate), #638 (the K sweep behind `--k-values`).

---

## 5. Follow-up Actions

- The M5 Max re-run of the two newly-reachable pairings is pending on that host. The M5 Max table in `model_tests.md` was left as it stands rather than being overwritten with M1 Ultra numbers, with its "unsupported target" note replaced by a statement that the harness restriction is lifted.
- The three `--kind dflash` rows keep their existing deferral (DFlash loader plus a public `Qwen3NextCache` API).
- Batched (`B > 1`) MTP adapters and Inkling pairings remain out of scope, the latter for want of a local checkpoint.

### Transferable lesson

The filed issue described a harness that refused to measure. The more expensive defect found while fixing it was a harness that measured nothing and said so nowhere: `resolve_model_dir` silently resolved every pairing to a skip on this host's layout. A benchmark that reports zero rows is obvious; one that completes with every row skipped looks like a clean run. When a sweep depends on path discovery, the discovery failure needs to be as loud as a measurement failure.
