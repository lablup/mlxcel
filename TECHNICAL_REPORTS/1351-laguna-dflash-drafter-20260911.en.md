# Technical Report: PR #1351 - feat(speculative): add the Laguna DFlash drafter

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: -
**Status**: Completed (Linux/CUDA host; the `metal,accelerate` workspace gate was not runnable here, the block-versus-chain exactness probe declines this host by default, and the throughput criterion is not met at any block size on this host)
**Languages**: Rust, Markdown
**Risk Level**: Medium (a new drafter family in `mlxcel-core`, a `SpeculativeTarget` impl on the Laguna target, a generalized server DFlash burst target trait that the Qwen 3.5 path now also goes through, and a new offline `mlxcel generate --draft-kind dflash` arm)

---

## Executive Summary

Poolside ships a DFlash speculator for every Laguna release, but mlxcel's DFlash machinery was hard-wired to the Qwen 3.5 drafter shape. This PR adds `mlxcel_core::drafter::laguna_dflash` (fused QKV, per-head softplus gate, `aux_hidden_norms`, sliding-window context attention), implements `SpeculativeTarget` on the Laguna target with rollback across dense and rotating caches, routes `model_type: laguna` drafters through `load_drafter`, and wires the pairing into both `mlxcel-server` and offline `mlxcel generate`. The pairing runs behind the same measured block-versus-chain exactness gate as the LFM2 and Muse Glimmer arms. On Laguna XS 2.1 NVFP4 with the published drafter on a GB10 the probe declines (107246 of 200704 logit bytes differ at the first verify position), so DFlash is off by default there; with `MLXCEL_MTP_ALLOW_INEXACT=1` greedy output equals classic decode except at bf16 logit ties, code completions accept 2.6 to 3.9 proposals per round, and throughput is at or below classic decode at every block size 2 to 16 on this host, before and after #1795 (best 1.00x at block 8, n=3, range inside the off arm's; 0.88x at the default block 16), because the multi-row verify runs launch-bound against a graph-replayed classic step.

---

## 1. Problem Statement

### 1.1 Background

The DFlash round loop (`DFlashGenerator`) and its `SpeculativeTarget` trait were target-agnostic, but the only drafter implementation (`dflash::DFlashDraftModel`) expected split `q/k/v_proj`, no gate, no QK norm on the drafter side of the checkpoint, a growing `KVCache` for the context, and the Qwen 3.5 target hooks. `poolside/Laguna-XS-2.1-DFlash` declares `model_type: laguna` with a fused `self_attn.qkv_proj`, per-head `q_norm`/`k_norm`, a per-head `g_proj` gate, one RMSNorm per captured target layer, a sliding window of 512, and `dflash_config { block_size 16, mask_token_id 12, num_target_layers 40, target_layer_ids [1, 13, 25, 33, 39], causal true }`.

### 1.2 Existing Issues

- **Issue 1**: `load_drafter` built the Qwen drafter for every `DrafterKind::Dflash` and failed on missing keys for the Laguna checkpoint.
- **Issue 2**: `LagunaModel::forward_with_capture` and `LagunaCache::trim` were staged by #1347 but nothing implemented `SpeculativeTarget` for the family.
- **Issue 3**: Offline `mlxcel generate --draft-kind dflash` returned a hard error for every target; the server burst was bound to a `Qwen35DFlashTarget` trait.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A drafter forward that is subtly wrong still produces fluent text (verification hides it) | High | Medium |
| Rollback on the sliding `RotatingKVCache` after a partial accept corrupts later positions | High | Medium |
| Multi-token verify and single-token decode disagree on quantized kernels | Medium | High |

---

## 2. Technical Review

### 2.1 Security

Config values from the drafter's `config.json` are validated before any weight is read: layer counts, GQA divisibility, window bounds, `mask_token_id` inside the vocabulary, strictly increasing `target_layer_ids` inside `num_target_layers`, `draft_vocab_size == vocab_size`, and `causal == true`. The sanitizer checks the weight key set exactly and refuses a layer that carries both fused and split projections or an incomplete split set. `validate_target_compat` refuses a target whose depth or vocabulary differs from the drafter's contract before bind.

### 2.2 Performance

Measured on a GB10 (sm_121), NVFP4 target, bf16 drafter, greedy, 128 tokens, `mlxcel generate` with `MLXCEL_MTP_ALLOW_INEXACT=1` (the exactness probe declines this host):

| Prompt | Classic tok/s | DFlash block 16 tok/s | Mean accepted length | Greedy ids |
|-------|------|------|------|------|
| chat 0 (retry wrapper, `<think>` channel) | 29.11 | 11.66 | 1.12 | differ at 76 (tie) |
| chat 1 (Rust LRU cache) | 26.86 | 12.24 | 1.84 | identical |
| chat 2 (TypeScript debounce) | 29.96 | 13.59 | 1.29 | differ at 83, 91 (ties) |
| code 0 (`retry_with_backoff` body, no template) | 28.70 | 24.81 | 3.27 | differ at 101 (tie) |
| code 1 (`lru_get` body) | 30.81 | 27.13 | 3.88 | differ at 69 (one-ulp tie) |
| code 2 (`debounce` body) | 32.35 | 21.87 | 2.63 | differ at 103 (one-ulp tie) |

**Throughput A/B (same binary, feature off versus on per block size; raw code prompts, no chat template, 200 tokens; GPU lock held for the whole sweep; measured after the merge of main `0c21aed2` with #1795; decode tok/s as the CLI reports it).** The GPU was exclusive (0 percent at start) but the host was not otherwise idle: another session's `cargo build` / `rustc` ran at full CPU throughout (load 2.6 to 3.9), the same for both arms. A first sweep before #1795 on an idle host (load 0.10) gave the same picture (block 8 at 0.97x, block 16 at 0.86x).

| Configuration | n | tok/s mean (min to max) | Mean accepted | Device ms/round | Drafter build ms/round | vs off |
|---|---|---|---|---|---|---|
| code 0 (`retry_with_backoff`), off | 5 | 31.18 (30.41 to 31.98) | | | | |
| code 0, block 2 | 3 | 16.19 (16.12 to 16.25) | 0.84 | 80 | 30 | 0.52x |
| code 0, block 3 | 3 | 21.61 (21.52 to 21.77) | 1.52 | 83 | 30 | 0.69x |
| code 0, block 4 | 3 | 27.05 (26.61 to 27.53) | 2.21 | 85 | 31 | 0.87x |
| code 0, block 5 | 3 | 28.33 (26.21 to 29.51) | 2.55 | 89 | 34 | 0.91x |
| code 0, block 6 | 3 | 30.46 (30.05 to 31.26) | 2.90 | 92 | 33 | 0.98x |
| code 0, block 7 | 3 | 30.24 (30.12 to 30.47) | 3.06 | 97 | 34 | 0.97x |
| code 0, block 8 | 3 | 31.26 (31.07 to 31.36) | 3.33 | 101 | 34 | 1.00x |
| code 0, block 9 | 3 | 30.58 (30.37 to 30.75) | 3.33 | 106 | 33 | 0.98x |
| code 0, block 10 | 3 | 30.02 (29.68 to 30.19) | 3.42 | 109 | 35 | 0.96x |
| code 0, block 12 | 3 | 28.94 (28.66 to 29.41) | 3.52 | 116 | 37 | 0.93x |
| code 0, block 16 (default) | 3 | 27.41 (26.90 to 27.90) | 3.55 | 127 | 35 | 0.88x |
| code 1 (`lru_get`), off | 3 | 31.80 (31.50 to 32.38) | | | | |
| code 1, block 4 | 3 | 26.10 (25.74 to 26.47) | 2.08 | 83 | 31 | 0.82x |
| code 1, block 6 | 3 | 30.42 (30.11 to 30.64) | 2.92 | 91 | 34 | 0.96x |
| code 1, block 8 | 3 | 29.95 (29.60 to 30.56) | 3.17 | 102 | 34 | 0.94x |
| code 1, block 10 | 3 | 30.16 (29.88 to 30.58) | 3.44 | 109 | 35 | 0.95x |

No configuration is a net win on this host, before or after #1795. The best width, block 8, is 1.00x on code 0 with its whole range (31.07 to 31.36) inside the off arm's (30.41 to 31.98), and 0.94x on code 1; the default block 16 is 0.88x. There is no sub-8 cliff on the NVFP4 path (5, 6 and 7 sit on a smooth curve; `fp_qmv` has no 2/4/8 accumulator dispatch) and no crossover past 8. The default block size stays at 16; a served-width change for Laguna belongs to #1797.

Attribution (block 8, per round): 30 to 35 ms of host-side drafter graph construction, about 3 ms of target graph construction, and 101 ms of synchronized device work for 4.33 emitted tokens, against 31 to 32 ms per classic token. The device cost of a verify block is a fixed 77 ms plus about 3.1 ms per row (80 ms at 2 rows, 127 ms at 16), 2.5x a single-token step even at 2 rows: the classic step is graph-replayed while the multi-row verify runs eagerly and launch-bound. Neither #1795 fix applies here (the drafter was already bf16; Laguna has no gated-delta layers), so the numbers did not move. That is a property of the CUDA backend on this host, not of the drafter, and out of scope here. Even perfect acceptance at block 8 would reach about 17 ms per token (1.8x); at the measured 3.3 to 3.6 accepted, 0.88x to 1.00x.

The exactness probe, re-run after #1795, still declines this host at block 16 (107246 of 200704 logit bytes) and at block 8 (107466).

The drafter's per-position accuracy along the reference path (probe b, shadow drafter): code 0 gives 0.88, 1.00, 0.75, 0.62, 0.38, 0.25 for d_0 to d_5 (mean accepted prefix 4.25 over 8 rounds); chat 0 gives 0.88, 0.50, 0.25, 0.12 (mean 1.38). Poolside's own numbers with a bf16 target are 3.55 to 4.57 on GSM8K, HumanEval, EvalPlus and Math.

Server path: the same request through `mlxcel-server --draft-model ... --draft-kind dflash` logged `rounds=60 proposed_tokens=835 accepted_tokens=67`, the same counters as the offline run, at 11.75 tok/s.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The Qwen 3.5 server burst now runs through the generalized `DFlashBurstTarget` trait; its behavior is unchanged (last prompt row as the first hidden, same cache factory).
- **New Dependencies**: none.
- **Compatibility**: `MLXCEL_PRINT_TOKEN_IDS` is a new opt-in print on `mlxcel generate`. `MLXCEL_KEEP_BF16` and `MLXCEL_CUDA_F16_NORMALIZE` now also govern the Laguna drafter's weight dtype so it matches the target.

### 2.4 Code Quality

- **Test Coverage**: 10 core unit tests (config contract and bounds, sanitizer, context window, in-block causality, window visibility, RoPE sensitivity, offset continuity, fused q/k/v row order, load-time shape checks), 2 binary greedy-invariant tests (oracle drafter with forced accept lengths 0, 1, 2 and full across the window wrap; the real drafter with random weights), 1 detection test, and an ignored real-checkpoint probe.
- **Code Complexity**: the drafter is a sibling module of `dflash`, sharing `DFlashMlp` and the sampling helpers; no changes to the round loop.
- **Technical Debt**: the gate is the shared MTP one, so its decline log line still says "MTP declined"; the burst and the offline arm add a DFlash-named line after it.

---

## 3. Technical Decisions

### 3.1 Follow the vLLM implementation where the issue text disagrees

**Context:** the issue body specified `ctx_qkv = qkv_proj(ctx_i)` and a fixed 511-entry context view for every block row. The merged reference (`vllm/model_executor/models/laguna_dflash.py`, vllm-project/vllm#46853) passes the projected context through each layer's `input_layernorm` before the K/V projection and keeps the sliding window as a per-query compute-time limit.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Option A: issue text | matches the ticket | not what the checkpoint is served with |
| **Chosen: Option B: vLLM semantics** | matches the served implementation and the training mask's block layout | deviates from the ticket in two places, which the PR names |

**Rationale:** the published checkpoint's norm weights are all exactly 1.0, so the `input_layernorm` on unit-RMS context rows is numerically a no-op today, but a future Laguna drafter with trained norm weights would differ; the per-query window is what `create_causal_mask_with_window_full` gives for free.

**Trade-offs:** the training mask (`speculators/models/dflash/attention.py`) anchors the window lower bound at the bonus position; at contexts under 512 tokens the two agree, and the difference at the far edge of a long window is not measurable here.

### 3.2 Temporal context buffer instead of a ring

**Context:** the drafter caches only the context K/V; the proposal K/V is concatenated per forward.

**Rationale:** a temporal buffer capped at `window - 1` makes the per-query window a plain additive `[block, prior + block]` mask and keeps `offset` as the absolute target position for RoPE. A ring would have required unwrapping before every multi-row attention.

### 3.3 Buffered rotating caches on the target

**Context:** rollback after a partial accept must trim the sliding layers' `RotatingKVCache`.

**Rationale:** `enable_speculative_buffer(block_size)` keeps the verify block inside the temporal region, so `trim` is a pointer rewind and the next append overwrites the rejected tail. The tiny-model greedy-invariant test forces accept lengths 0, 1, 2 and full across a window of 6, so every trim length hits the caches after the wrap.

---

## 4. Implementation Details

### 4.1 Architecture Changes

```
[Before]
load_drafter(Dflash) -> DFlashDrafter (Qwen 3.5 shape) -> DFlashGenerator -> SpeculativeTarget (Qwen35Model only)

[After]
load_drafter(Dflash) -> model_type == laguna ? LagunaDFlashDrafter : DFlashDrafter
DFlashGenerator -> SpeculativeTarget (Qwen35Model | LagunaModel / LagunaWrapper)
mlxcel generate --draft-kind dflash -> generate_dflash::run_offline_dflash (Laguna)
mlxcel-server DFlash burst -> DFlashBurstTarget (Qwen 3.5, Qwen 3.5 VLM, Laguna)
```

### 4.2 Key Code Changes

**File: `src/lib/mlxcel-core/src/drafter/laguna_dflash/attention.rs`**: fused `qkv_proj` split into q, k, v rows; per-head `q_norm` / `k_norm` before RoPE; context rows older than `window - 1` dropped before projection with the cache offset advanced past them; only the context K/V enters `LagunaDFlashContextCache`; `create_causal_mask_with_window_full(L, prior, window)` over `[context | block]`; `softplus(g_proj(x))` per head in f32.

**File: `src/models/laguna_speculative.rs`**: `make_speculative_caches(block_size)`, `forward_speculative` (capture after the listed layers), `rollback_speculative_cache` (trim every cache by `block_size - (accepted + 1)`), and the `SpeculativeTarget` impls on `LagunaModel` and `LagunaWrapper`.

**File: `src/server/batch/speculative_burst.rs`**: `Qwen35DFlashTarget` becomes `DFlashBurstTarget` with a `block_size`-aware cache factory and a `dflash_first_hidden` hook; Laguna seeds the drafter with every captured prompt row, Qwen 3.5 keeps the last row.

### 4.3 Data Model Changes

None.

---

## 5. Learning Points

### 5.1 Block-versus-chain ties on quantized kernels

**Concept:** a `T = K` verify block and `K` single-token decodes reduce the same dot products in different orders on the `M >= 2` and `M = 1` quantized matmul kernels. Where the target's top-2 logits are equal in bf16, the argmax can differ.

**Application in this PR:** the ignored `laguna_real_checkpoint_probe` compares both arms on the real checkpoint and prints the top-2 margins at every disagreement: every one of the seven observed disagreements across five prompts sat at a chain-arm margin of 0.0 or 0.125 (one bf16 ulp at logits of 21 to 34). The production gate (`dflash_exactness_allows`) compares the two arms byte for byte on synthetic inputs and declines the host when they differ, which on this GB10 they do.

### 5.2 Measuring a drafter without rejection feedback

**Concept:** with an oracle keeping the round loop on the reference path, a shadow drafter records the real drafter's proposals every round, which yields per-position accuracy independent of earlier rejections.

**Application in this PR:** this is what separated "the drafter is weak on `<think>` prose" (d_1 at 0.50) from "the drafter is broken" (d_1 at 1.00 on code), and what ruled out the RoPE base, the rotated dims, three in-block mask variants, the context norms, and a capture-layer off-by-one as causes.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `DFlash` | block-diffusion drafter: one masked forward proposes `block_size - 1` tokens | the drafter family added here |
| `aux_hidden_norms` | one RMSNorm per captured target layer ahead of `fc` | Laguna-specific context path |
| `RotatingKVCache::enable_speculative_buffer` | temporal slack for verify-then-trim on sliding layers | target rollback |

### Related PRs/Issues

- Issue #1347: Laguna family port (the target this drafter pairs with)
- vllm-project/vllm#46853: Laguna DFlash in vLLM (reference implementation)

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 23 |
| Lines added | +3293 |
| Lines deleted | -83 |
| Tests added | 15 |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Code Quality | 3 | new drafter module, target impl, burst trait generalization |
| Performance | 1 | block-size sweep documented; default kept at the checkpoint's 16 |
| Documentation | 1 | `docs/supported-models.md` DFlash row |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `154aafa1` | feat | add the Laguna DFlash drafter and target |
| `d64c73af` | merge | integrate Laguna into main's `DFlashTargetModel` design with the exactness gate |
| `912ac708` | fix | validate projection rows at load and drop a per-round copy |
| `63af4faf` | fix | close the implementation-review findings (offline pairing and greedy guards, requested block width, pre-`fc` window drop, slack rule, routing predicate) |
| `a73e249f` | fix | bound untrusted config, `greedy_only` on the server, K/V-only context projection, sanitizer shape checks |
| `edd959bb` | docs | throughput A/B (pre-#1795) and split phase diagnostics |
| `08e09036` | merge | main at `0c21aed2` (#1795) |
| `aad08e1c` | fix | load the drafter through the shared dtype policy |
| `76f16daf` | test | track the oracle drafter's reference position |
| `fa8f1919` | test | add a real-checkpoint DFlash probe and a RoPE sensitivity test |

---

## 8. Follow-up Actions

### Required

- [ ] Decide whether the Qwen 3.5 DFlash arm should run the same measured gate Laguna, LFM2 and Muse Glimmer now run (it keeps the permissive default).
- [ ] Run the `metal,accelerate` workspace gate and the block-size sweep on an Apple Silicon host, where the classic step is not graph-replayed and the verify overhead may differ.
- [ ] Graph capture (or `mlx::compile`) for the multi-row verify forward and the drafter forward on the CUDA backend, which is where the GB10 ceiling comes from; a runtime change, not a drafter one.

### Monitoring Required

- Acceptance length per request (`DFlash diagnostics` log line, `spec_decode_*` counters) on thinking-channel traffic, where the drafter accepts 1.1 to 1.8 per round.

### Future Improvements

- Batched (B > 1) Laguna DFlash; fixed-anchor window option if a long-context measurement shows a difference.

---

## Appendix

### A. Test Results

- `cargo test -p mlxcel-core --profile test-fast --features cuda --lib -- drafter::laguna_dflash`: 7 passed.
- `cargo test --profile test-fast --features cuda --lib -- models::laguna_dflash_tests models::detection_tests::laguna_dflash models::laguna_tests`: 24 passed, 1 ignored (the real-checkpoint probe).
- `cargo test -p mlxcel-core --profile test-fast --features cuda --lib -- drafter::dflash drafter::laguna_dflash drafter::tests`: 88 passed.
- `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` and the `-p mlxcel-core` variant: clean; `cargo fmt --all -- --check`: clean.
- `cargo test --workspace --profile test-fast --features metal,accelerate`: not runnable on this Linux/CUDA host; unrun.

### B. Performance Benchmarks

See 2.2. Commands: `MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate -m models/mlx/laguna-xs-2.1-nvfp4 [--draft-model models/mlx/laguna-xs-2.1-dflash --draft-kind dflash [--draft-block-size N]] [--no-chat-template] -p ... -n 128 --temp 0`.

### C. References

- vLLM `laguna_dflash.py`, `qwen3_dflash.py`, `v1/spec_decode/dflash.py` at `fb5138c3`.
- `vllm-project/speculators` `models/dflash/{core,attention,utils}.py` (training block layout and mask).
- `poolside/Laguna-XS-2.1-DFlash` model card (acceptance lengths with a bf16 target).
