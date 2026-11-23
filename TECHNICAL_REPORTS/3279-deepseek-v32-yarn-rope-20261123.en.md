# PR #3279: Apply YaRN RoPE in DeepSeek V3.2 attention and indexer

**Date**: 2026-11-23
**Status**: PR open (closes #3178). Unit tests run on ROCm gfx1151 only. The long-context perplexity re-run on DeepSeek-V3.2-4bit is pending on an M3 Ultra and has not been done.

## Executive Summary

`deepseek_v32` (and `glm_moe_dsa`, which reuses its attention) rotated the MLA `q_pe` / `k_pe` and the DSA indexer queries and keys with plain RoPE, although DeepSeek-V3.2 checkpoints declare YaRN (`factor` 40 over `original_max_position_embeddings` 4096). The `rope_scaling` block was read only for the softmax-scale correction. Past 4096 positions the low-frequency pairs rotated up to 40 times too fast, which matches the perplexity cliff reported in #3178 (7.3 behind 4096 tokens, 1403 behind 8192, 2080 behind 16384 on an M3 Ultra with the fp16 cache).

The PR adds `DsaRope` (`src/models/deepseek_v32_rope.rs`), which resolves the whole `rope_scaling` block once per model through the shared `RopeScalingKind::resolve` / `yarn_rope_setup` path and is used at all six rotation sites. Configs with no block or a `default` block keep the exact plain `fast_rope` call. Unit tests show the fix at the tensor level. Whether YaRN alone closes the perplexity gap is not yet known, because the checkpoint cannot load on the ROCm host where the work was done.

## 1. Problem Statement

#3178 measured teacher-forced perplexity on `mlx-community/DeepSeek-V3.2-4bit` with the fp16 cache: 7.3 behind 4096 tokens, 1403 behind 8192, 2080 behind 16384. Chunking the prefill into 2048-token forwards did not change the 4096 figure much (7.32 against 7.51), so chunking was ruled out as the main cause. The breakage starts right after `original_max_position_embeddings`, which pointed at RoPE scaling.

The issue also noted that `docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md` records 16384-token rows (int8 against fp16 top-k overlap) measured on this broken reference, so those rows are suspect until re-run.

## 2. Root Cause

Upstream mlx-lm builds both the attention RoPE and the indexer RoPE with `initialize_rope(..., scaling_config=args.rope_scaling)`. For the shipped config (`yarn`, `factor` 40, `original_max_position_embeddings` 4096, `beta_fast` 32, `beta_slow` 1, `mscale` 1.0, `mscale_all_dim` 1.0) that returns `YarnRoPE`.

mlxcel's `RopeScaling` struct parsed only `type`, `factor` and `mscale_all_dim`, and those fed the softmax-scale correction. Every rotation called `fast_rope(.., rope_base, 1.0, offset)`:

- `MLAAttention::forward`: `q_pe` and `k_pe`.
- `MLAAttention::forward_batched_decode`: `q_pe` and `k_pe` through `fast_rope_batched`.
- `Indexer`: the key and the query.

With `qk_rope_head_dim` 64 there are 32 frequency pairs. The YaRN correction ramp leaves pairs 0 to 10 unchanged (extrapolation), divides pairs from 23 up by 40 (interpolation), and blends the pairs between. Plain RoPE therefore agrees with YaRN only on the high-frequency pairs. Inside 4096 positions the low-frequency pairs have not rotated far enough for the error to matter; past 4096 they reach angles the model never saw in pre-training.

## 3. Change Summary

| File | Change |
|---|---|
| `src/models/deepseek_v32_rope.rs` (new, 193 lines) | `DsaRope`: resolves the block, validates `qk_rope_head_dim`, applies the table and the YaRN magnitude multiplier |
| `src/models/deepseek_v32.rs` | `rope_scaling` becomes `RopeScalingSpec`; the private `RopeScaling` struct is removed; both MLA forward paths call `DsaRope`; one `DsaRope` is built per model (full model and pipeline stage model) and duplicated into each layer |
| `src/models/deepseek_v32_indexer.rs` | Indexer key and query use the same `DsaRope`, keeping `indexer_rope_interleave` as the `traditional` flag; `load_with_rope` takes the shared instance |
| `src/models/glm_moe_dsa.rs` | `rope_parameters` maps to a full `RopeScalingSpec` through `RopeScalingSpec::from_lookup` instead of three hand-picked keys |
| `src/models/rope_overrides.rs` | `resolve_base` "used by" comment lists `DsaRope` |
| `docs/llama-server-compat.md` | DeepSeek V3.2 moves from "implements YaRN outside the shared path" to the list of families that honor `--rope-*` / `--yarn-*` overrides |
| `src/models/deepseek_v32_rope_tests.rs` (new, 505 lines) | Regression tests (section 5) |
| `CHANGELOG.md` | `[Unreleased]` Fixed entry, which states that the perplexity re-run is pending |

`DsaRope` has three behaviours:

1. `from_args` rejects a `qk_rope_head_dim` that is zero, odd, or wider than `min(index_head_dim, qk_nope_head_dim + qk_rope_head_dim)`. It renames `deepseek_yarn` to `yarn` (upstream accepts both) and calls `RopeScalingKind::resolve` with no training-context fallback.
2. `apply` / `apply_batched` call `fast_rope_with_freqs` / `fast_rope_batched_with_freqs` when the scheme has a frequency table, and otherwise the plain `fast_rope` / `fast_rope_batched` with the resolved base and scale.
3. `magnitude` multiplies only the first `dims` features by `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)`, as upstream `YarnRoPE.__call__` does. For the shipped config this is exactly 1.0, so it returns early and adds no work.

The softmax-scale correction in `load_mla_attention_with_rope` is unchanged. Upstream applies both the correction and the magnitude multiplier, so neither is counted twice.

## 4. Technical Decisions

**Reuse the shared YaRN resolver instead of the DeepSeek V2 copy.** `deepseek_v2.rs` has its own YaRN implementation. The PR goes through `rope_utils` (`RopeScalingSpec`, `RopeScalingKind::resolve`, `yarn_rope_setup`) instead, as the issue proposed. This avoids a third YaRN implementation, and it makes the `--rope-*` / `--yarn-*` runtime overrides reach DeepSeek V3.2 and GLM-5: the counter the model worker checks is bumped by the shared resolver, so these families are now accepted instead of refused. The documentation change records this.

**Build once, duplicate per layer.** The table is resolved once in `DeepSeekV32Model::from_weights` and once in the pipeline stage loader, then each attention block and indexer takes a `duplicate()` handle (a lazy copy of a 32-element array). Loading does not re-run the `powf` loop per layer. The single-block loaders `load_mla_attention` and `Indexer::load` are kept only under `#[cfg(test)]`.

**Keep the plain path bit-exact.** With no block or a `default` block, `DsaRope::apply` makes the same `fast_rope(x, dims, traditional, base, 1.0, offset)` call as before. `absent_rope_scaling_keeps_the_plain_rope_call` asserts a maximum difference of exactly 0.0 for both single and batched paths. This matters for GLM-5: the public configs (`zai-org/GLM-5`, `mlx-community/GLM-5-4bit`) carry `rope_parameters` `{rope_theta: 1000000, rope_type: default}`, so their output does not change.

**No training-context fallback.** The shared `--yarn-orig-ctx` sentinel documentation describes a fallback to `max_position_embeddings`. `DsaRope` passes `None`, so a missing `original_max_position_embeddings` falls back to 4096, which is what upstream `YarnRoPE` does. This is chosen for parity with upstream; the deviation from the general documentation is not documented (a LOW item from review).

**Map the whole GLM-5 block.** Upstream sets `self.rope_scaling = self.rope_parameters`. The old mapping copied three keys and dropped the rest, so a GLM-5 YaRN block would have lost `original_max_position_embeddings` and the betas. Using `RopeScalingSpec::from_lookup` keeps every key.

**Validate the rope width before allocation.** Security review found that `from_args` runs before any weight shape is checked, and that `yarn_rope_setup` allocates `qk_rope_head_dim / 2` entries with no upper bound, so a malformed config could make the loader allocate gigabytes. Commit 70ad89cd rejects a zero, odd, or too-wide `qk_rope_head_dim` with a named error before the table is built.

## 5. Tests and Verification

`src/models/deepseek_v32_rope_tests.rs`:

- `yarn_table_matches_the_upstream_formula_at_checkpoint_geometry`: at `qk_rope_head_dim` 64 with the shipped block, pairs 0, 10, 16, 20, 24 and 31 match values computed independently in float64 from upstream's `YarnRoPE` formula (relative error under 1e-5), for both `yarn` and `deepseek_yarn`, and the magnitude multiplier is exactly 1.0.
- `absent_rope_scaling_keeps_the_plain_rope_call`: bit-exact plain path (section 4).
- `yarn_magnitude_scales_only_the_rotated_dims`: with `mscale` 2.0 against `mscale_all_dim` 1.0 the multiplier matches upstream (1.26948) and touches only the rotated features.
- `prefill_and_decode_cache_yarn_rotated_keys_past_the_original_context` and `batched_decode_caches_yarn_rotated_keys_per_row`: cached `k_pe` and indexer keys on both sides of `original_max_position_embeddings` match a reference from `yarn_rope_setup` plus `fast_rope_with_freqs`, and differ from plain RoPE. The batched case puts one row on each side of the boundary.
- `indexer_queries_and_keys_rotate_with_yarn`: the same check for indexer queries and keys.
- `yarn_model_batched_decode_matches_sequential`: batched decode matches sequential decode under YaRN.
- `mla_queries_rotate_with_the_key_table_past_the_original_context`: added during implementation review. Cached keys alone cannot show a wrong `q_pe` rotation, so this test uses RoPE's relative-position property: a two-token prefill at positions 0, 1 and at a far offset, and a batched decode with two rows far apart, must produce the same attention output. Reverting either `q_pe` site alone fails it (drift 1.45e-3 against a 1e-4 tolerance; under 1e-5 with the fix). Before this test, reverting both `q_pe` sites left every `deepseek_v32` and `glm_moe_dsa` test green.
- `malformed_rope_width_is_refused_before_the_table_is_built`: widths 0, 7 and 14 are refused.

With the six rotation sites reverted to main and the tests kept, the cache and indexer tests fail (cached `k_pe` off by up to 2.04, indexer queries by 11.4).

Commands run on ROCm gfx1151:

- `cargo test --lib --profile test-fast --features rocm -- deepseek_v32 glm_moe_dsa rope_utils rope_overrides`: 99 passed (run twice).
- `cargo clippy --lib --tests --features rocm -- -D warnings`, `cargo fmt --check`, `scripts/ci/check_used_by.py`: clean.

Workspace-wide tests and clippy were not run locally, and nothing was run on Metal or CUDA.

**Test lock fix (b92ca275).** The five forward tests ran decode steps without the shared MLX test lock. Under parallel test threads they incremented the process-wide MLA decode-path counters while `a_forced_int4_slot_keeps_the_indexer_key_at_eight_bits` measured those counters as a delta, and that test failed once. The forward tests now hold `embedding_test_support::mlx_test_guard()`, like the other DeepSeek V3.2 test files.

## 6. Review Outcomes

**Implementation review** (PR comment 7087109122): one HIGH finding, the untested `q_pe` sites, fixed by the relative-position test in 65288130. Three LOW items left open: the undocumented 4096 training-context fallback, the shared YaRN helper ignoring upstream's `truncate` key (no shipped DeepSeek-V3.2 or GLM-5 config sets it), and a private `yarn_get_mscale` in `deepseek_v32.rs` that duplicates `rope_utils::yarn_get_mscale`.

**Security and performance review** (PR comment 7087141070): no CRITICAL or HIGH findings. The review confirmed that the per-field screen in `YarnParams::resolve` rejects missing, zero, negative and non-finite values without panicking, that the magnitude ratio cannot divide by zero, that the correction ramp cannot underflow, and that decode adds no work at the shipped config. One MEDIUM finding (unbounded table allocation from a large `qk_rope_head_dim`) was fixed by the validation in 70ad89cd. Two LOW items remain unfixed, both in pre-existing shared code:

- Values that pass the per-field screen can still overflow together: a `factor` near `f32::MAX` produces a NaN table entry, and an extreme `mscale` makes the ratio inf / inf. A finite-and-positive check on the finished table and multiplier would close this.
- The softmax-scale correction reads `factor` and `mscale_all_dim` with no finite check, so an `inf` factor leaves the YaRN table on its fallback while the softmax scale becomes inf. The correction also ignores a `--rope-scaling` override that replaces the table.

## 7. Pending Work and Limitations

- **Perplexity re-run is not done.** The ROCm gfx1151 host (30 GB) cannot load DeepSeek-V3.2-4bit. The #3178 acceptance criteria for fp16 perplexity behind 8192 and 16384, the #3163 top-k overlap re-run at 16384, and the update to `docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md` are pending on an M3 Ultra. The command is `MLXCEL_TRACE_PREFILL_CHUNK=2048 logit_trace models/deepseek-v3.2-4bit corpus.txt 64 8 8 8192 > fp16-8k.tsv`. Until it runs, YaRN is the confirmed tensor-level defect but not a confirmed full explanation of the perplexity cliff. If perplexity stays high, the issue lists the next suspects: the indexer top-k and sparse prefill mask past `index_topk` 2048, the decode gather, and the chunked prefill context pass, bisected with `MLXCEL_DSA_DENSE`.
- **Test location.** The issue asked for the regression test in `deepseek_v32_tests.rs`; it lives in the new `deepseek_v32_rope_tests.rs`, next to the module it tests.
- **Out of scope: DeepSeek V3.** `src/models/deepseek_v3.rs` has the same defect on main. `DeepSeekV3Attention` calls plain `fast_rope` / `fast_rope_batched` for `q_pe` / `k_pe` and reads `rope_scaling` only for the softmax scale, while DeepSeek-V3 checkpoints carry the same YaRN block. It needs its own issue.

## 8. Learning Points

- A config block that is parsed partially can hide a missing feature. Here `rope_scaling` was read, so the model looked YaRN-aware, but only the softmax-scale half was wired. Parsing the full shared spec type (`RopeScalingSpec`) makes the missing half visible.
- YaRN with plain RoPE as a fallback is invisible in short-context tests, because the two tables agree on the high-frequency pairs and the low-frequency error grows slowly. Tests need an offset past `original_max_position_embeddings`.
- Checking cached keys does not cover queries, because the query rotation is never stored. A relative-position invariance check (same tokens at shifted positions must attend identically) tests the query side without a reference implementation.
- Tests that run forward passes in this crate touch process-wide counters, so they must hold `mlx_test_guard()` even when they assert nothing about those counters.
