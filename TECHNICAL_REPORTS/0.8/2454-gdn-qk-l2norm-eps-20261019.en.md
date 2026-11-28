# PR #2454: Reference l2norm eps for GDN / KDA q/k via normalize_qk

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with Qwen3.5-4B and Qwen3-Next-80B-A3B; PR open

## Summary

Closes #2361. The gated-delta (GDN) and Kimi Delta Attention (KDA) q/k normalization is `l2norm(x) = x / sqrt(sum(x^2) + 1e-6)`. mlxcel computes it with the fused `rms_norm`, which adds its eps to `mean(x^2)`, so the matching eps is `1e-6 / head_k_dim`. Qwen3-Next, Qwen 3.5 and Kimi Linear passed a flat `1e-6`. This PR adds a shared `normalize_qk` that mirrors mlx-lm `gated_delta.normalize_qk` (ml-explore/mlx-lm#1896) and routes all six call sites plus Kimi K3 through it.

## Problem

With a flat `1e-6` added to `mean(x^2)`, the effective l2 eps is `1e-6 * head_k_dim`, 128 times too large at `head_dim = 128`. For ordinary rows the difference is small. For rows with a small norm (entries around 1e-5) the old formula is off by 91 percent relative to a true l2norm. Kimi K3 already used the right value through a private `qk_norm_eps`, so the families disagreed with each other as well as with upstream.

## Technical decisions

- **One helper, three pieces.** `gated_delta.rs` gains `normalize_qk(q, k, inv_scale, eps)`, the constant `QK_L2NORM_EPS = 1e-6`, and `normalize_qk_rms_eps`, which computes `eps * inv_scale^2` in f64 from the f32 `inv_scale`, as mlx-lm does with Python floats. `scaled_fast_rms_norm_no_weight` now only backs `normalize_qk` and is narrowed to `pub(crate)`.
- **Every call site switched.** Qwen3-Next (one site), Qwen 3.5 (the main forward, the speculative capture forward that MTP and DFlash verify through, and the test-only `debug_prefill_no_cache`; the VL and MoE wrappers inherit it), Kimi Linear KDA (one site, `delta_head_dim` scale) and Kimi K3.
- **Kimi K3 kept bit-identical.** Its private `qk_norm_eps`, the stored `qk_eps` field and the stale "Kimi Linear is left as is" comment are removed. A test replays the removed path and checks bit equality at the published `delta_head_dim` of 128. At non-power-of-two head dims the two f32 eps expressions can differ by 1 ulp, so identity is claimed only for 128.

## Changes

- `src/models/gated_delta.rs`: `normalize_qk`, `QK_L2NORM_EPS`, `normalize_qk_rms_eps`, visibility narrowing.
- `src/models/qwen3_next.rs`, `src/models/qwen3_5.rs`, `src/models/kimi_linear.rs`, `src/models/kimi_k3.rs`: call sites switched.
- `src/models/gated_delta_normalize_qk_tests.rs` (new), `src/models/kimi_k3_tests.rs`: tests.
- `CHANGELOG.md`: Unreleased / Fixed entry.

## Verification

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`, test binaries under `gpu-lock` with `--test-threads=1 --include-ignored`.

| Filter | Result |
|---|---|
| normalize_qk | 4/4 |
| models::kimi_k3 | 26/26 |
| models::qwen3_next | 11/11 |
| models::qwen3_5 | 51/51 |
| models::kimi_linear | 10/10 |
| models::gated_delta | 19/21 (the 2 failures are already on main) |

The small-norm test feeds q/k entries around 1e-5 at `head_dim = 128`: `normalize_qk` is within 1.6e-7 relative error of a hand-computed f64 l2norm, and the old flat-eps arm is 0.91 off on the same rows. The two gated_delta failures (`gated_delta_step_output_shape_and_dtype`, `gated_delta_ops_single_token_output_shape`) fail an output-dtype assertion the same way on a main build.

Teacher-forced logit traces over `docs/architecture.md`, decided gap 2. The reverted arm is this branch with the six source files checked out from main, rebuilt.

| Checkpoint | Width | Top-1 disagreement | Decided disagreement | ppl reverted / fixed |
|---|---|---|---|---|
| Qwen3.5-4B-4bit | 1 | 2/200 | 0/79 | 19.213 / 19.199 |
| Qwen3.5-4B-4bit | 5 | 4/300 | 0/134 | 15.289 / 15.276 |
| Qwen3.5-4B-4bit | 256 | 18/1024 | 0/385 | 23.072 / 23.063 |
| Qwen3-Next-80B-A3B-4bit | 1 | 8/200 | 0/91 | 27.351 / 27.140 |
| Qwen3-Next-80B-A3B-4bit | 5 | 28/300 | 0/51 | 980.26 / 987.42 |
| Qwen3-Next-80B-A3B-4bit | 256 | 27/1024 | 0/409 | 24.109 / 24.059 |

A null arm (the reverted binary run twice on Qwen3.5-4B) is byte-identical at all three widths, so the movement comes from the eps change. No decided position disagrees anywhere, and perplexity drops in 5 of 6 rows. Greedy 128-token `mlxcel generate --temp 0` is fluent in both arms on both checkpoints.

There is no mlx-lm reference arm: the oracle venv has only the mlx CPU backend here, and `scripts/mlx_lm_logit_trace.py` did not finish the first of four forwards on Qwen3.5-4B in 70 minutes. Kimi-Linear-48B is not on this host, so it has no trace; its change is the same call swap and is covered by the kimi_linear unit tests.

## Follow-ups

- Qwen3-Next-80B at width 5 behind a 512-token context reads ppl about 980 in both arms, with every in-chunk position bad (mean NLL 6 to 9), while width 1 behind the same context reads 27. This looks like an existing problem in Qwen3-Next multi-token forwards behind a cache, the shape speculative verify runs at. It is not caused or fixed by this PR.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 8 |
| Lines added | 262 |
| Lines deleted | 52 |

| Commit | Message |
|--------|---------|
| `11e9b5f2` | fix(gdn): use the reference l2norm eps for q/k via normalize_qk |
| `92421992` | docs: add changelog entry for the GDN q/k l2norm eps fix |

## Related

- ml-explore/mlx-lm#1896: upstream `normalize_qk`.
