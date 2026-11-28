# Technical Report: PR #3275 - Aleph Alpha Kolibri-1 (kolibri1) hybrid sliding/full MoE decoder

**Date**: 2026-11-23
**Reviewer**: -
**Status**: PR open (closes #2357); verified on M3 Ultra against the vendored mlx-vlm kolibri1 module
**Languages**: Rust, Python (oracle trace script)
**Risk Level**: Low to Medium (new model family; the shared AFMoE attention block, prefill masks and expert stacking helper were refactored, with AFMoE output unchanged and its 32 tests passing)

---

## Executive Summary

This PR adds Kolibri-1 (`model_type: kolibri1`), a 50-layer decoder that interleaves four sliding-window layers with one full-attention layer and uses a 384-expert top-6 MoE plus one shared expert in every layer. It is structurally AFMoE without the attention output gate, so it reuses AFMoE's attention block, shared MLP, weight validators and hybrid prefill masks. The family runs through `mlxcel generate`, `mlxcel-server` and `mlxcel arch`, joins native batched decode, projects only the sampled row at prefill, and donates model-state snapshots for prompt-cache reuse. On `models/kolibri-1-mlx-4bit` (45 GB on disk, 41.9 GB resident, about 80 tok/s greedy on M3 Ultra) a teacher-forced 1024-token prefill agrees with mlx-vlm on 1024 of 1024 top-1 positions, and decode behind a 1024-token prefix, with every sliding ring wrapped, disagrees on 1 of 388 decided positions.

---

## 1. Problem Statement

### 1.1 Background

Kolibri-1 ships as `nativ-community/Kolibri-1-MLX-4bit` (affine 4-bit, group size 64). `layer_types` repeats four `sliding_attention` layers and one `full_attention` layer ten times. Sliding layers rotate with RoPE inside a `RotatingKVCache(max_size = 513, keep = 0)`. Full layers are NoPE and use a plain `KVCache`. Per-head Q/K RMSNorm runs before RoPE, and four norms per layer form a sandwich. mlxcel had no `kolibri1` detection arm.

### 1.2 Existing Issues

- **Router semantics differ from DeepSeek-V3**: the router selects on raw logits plus `e_score_correction_bias`, not on sigmoid plus bias. The wrong rule picks a different expert set and still generates fluent text, so a quality check by eye does not catch it.
- **Unstacked experts**: the MLX conversion ships 174,253 tensors, with 384 separate expert planes per projection per layer and the bias named `moe.router.expert_bias`.
- **CI rules for new families**: since #3090 a new `impl LanguageModel` must implement `last_row_logits` or be listed, and must join batched decode or be pinned in `batched_decode_coverage_tests.rs`.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Router selects on sigmoid plus bias | High: wrong experts on every token, fluent but degraded output | Medium for a port that copies DeepSeek-V3; covered by router fixtures and the trace |
| RoPE applied on full layers | High: silent quality loss at long context | Medium; the `--rope-on-full` trace arm fails the gate |
| A short or malformed expert stack reaches `gather_qmm`, which does not range-check a positive index | High: out-of-bounds GPU read | Low; refused at load |
| AFMoE refactor changes AFMoE output | High: regression in an existing family | Low; AFMoE tests unchanged and passing |

---

## 2. Technical Review

### 2.1 Security

The implementation and security reviews treated `config.json` and tensors as untrusted. Neither found a CRITICAL or HIGH issue.

| Issue | Severity | Status |
|-------|----------|--------|
| Stacked expert tensor with fewer planes than `num_experts` | Medium | Refused at load by AFMoE's `validate_stacked_experts` |
| `layer_types` of the wrong length or holding a value other than the two attention kinds | Medium | Refused at config validation |
| Quantized router, non-default `rope_scaling`, `attention_bias` | Medium | Refused rather than silently mis-run |
| Any tensor whose shape disagrees with the config | Medium | `validate_weights` runs after sanitize, before any tensor reaches MLX |
| `drain_numbered_expert_planes` receives planes of mismatched shapes: `mlx::core::stack` throws across the C++ bridge and aborts the process | Medium | Open; the helper is shared with other families, so a shared-helper fix is left for a follow-up |

### 2.2 Performance

| Area | Value |
|------|-------|
| Greedy decode, `kolibri-1-mlx-4bit`, M3 Ultra | 77 to 83 tok/s |
| Resident memory after load | 41.9 GB |
| Load time | 10 to 12 s |

Stacking is materialized one layer at a time inside `sanitize_weights`, while that layer's 384 source planes are the only ones alive. Deferring it would copy all layers inside the first forward and raise peak memory.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. AFMoE's attention block gained an optional gate and explicit geometry, and its hybrid prefill masks moved into a shared `hybrid_prefill_masks`; AFMoE output is unchanged.
- **New Dependencies**: none.
- **Not tested**: the MXFP4 conversion (not on this host) and the EAGLE3 drafter (out of scope).

### 2.4 Code Quality

- **Tests**: 19 tests in `src/models/kolibri1_tests.rs`, plus the pinned entries in `batched_decode_coverage_tests.rs` and the `models::afmoe` suite (32) rerun after the refactor.
- **Structure**: `kolibri1.rs` (config, attention layer, model), `kolibri1_moe.rs` (router and routed MoE), `kolibri1_weights.rs` (sanitize and validate), `kolibri1_lm.rs` (`LanguageModel` impl), `kolibri1_tests.rs`.
- **Technical debt (open MEDIUM)**: the `LanguageModel` impl in `kolibri1_lm.rs` duplicates much of AFMoE's. A shared implementation would remove that duplication, but changing AFMoE's trait surface was out of scope for a new-family PR.

---

## 3. Technical Decisions

### 3.1 Reuse AFMoE instead of a standalone block

**Context:** Kolibri-1 differs from AFMoE only in having no attention output gate and in its router rule.

| Option | Pros | Cons |
|--------|------|------|
| Copy AFMoE's attention, MLP and validators | No change to AFMoE | Two copies of the same code to keep correct |
| **Chosen: add an optional gate and explicit geometry to AFMoE's attention block and lift the prefill masks into `hybrid_prefill_masks`** | One implementation; validators and masks shared | Touches a shipped family, so AFMoE's 32 tests and output must stay unchanged |

The router and the cache plumbing are not shared: the router rule differs, and the model-owned state reuses gemma3's `Cache` enum (`Rotating` for sliding layers, `KVCache` for full layers).

### 3.2 Router semantics

The router `mlp.gate` is unquantized and runs in f32. It computes logits, selects the top 6 on `logits + e_score_correction_bias`, reorders the six by descending selection value, and takes their weights from `sigmoid(logits)` with no bias. Weights are renormalized only when `norm_topk_prob` is set (false on the release), and there is no routed scaling factor. Logits are cast to f32 when `head_dtype` is `float32`. The router fixtures use inputs where sigmoid plus bias picks a different set than raw logits plus bias, so the DeepSeek-V3 rule fails them.

### 3.3 Full layers are NoPE

Full-attention layers apply no RoPE; only sliding layers rotate. The trace's `--rope-on-full` arm shows the gate separates the two (section 4.3).

### 3.4 Stack experts at load, bounds-checked

`sanitize_weights` renames the bias, accepts pre-stacked `mlp.experts.{proj}.*` and `mlp.switch_mlp.*` exports (idempotent), and otherwise drains numbered planes through the shared `drain_numbered_expert_planes`, which looks every key up by name and never scans the 174k-entry map. A stack with fewer planes than `num_experts` is refused afterwards by `validate_stacked_experts`, because the gather behind `gather_qmm` does not range-check a positive index.

---

## 4. Implementation Details

### 4.1 Forward pass per layer

```
h   = x + post_attn_norm(attn(input_layernorm(x)))
out = h + post_ffn_norm(moe(post_attention_layernorm(h)))
moe(z) = shared_mlp(z) + sum_{e in top6} w_e * expert_e(z)
```

Sliding layers: per-head Q/K RMSNorm, then RoPE, then attention over a `RotatingKVCache(513)`. Full layers: the same norms, no RoPE, plain `KVCache`. Prefill builds a windowed mask for sliding layers and a causal mask for full layers through `hybrid_prefill_masks`.

### 4.2 Wiring

`ModelType::Kolibri1`, the `"kolibri1"` detection arm, `LoadedModel`, `model_metadata`, the registry, the memory estimate, the tensor-parallel model lists, the snapshot sizing family list in `snapshot_sizing.rs`, `kv_snapshot.rs` `Used by:` annotations, `batched_decode_coverage_tests.rs` (`EXPECTED_NATIVE`), `docs/supported-models.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/turbo-kv-cache.md`, `docs/environment-variables.md` (the `MLXCEL_FUSED_MOE` list and the attention-cache snapshot families) and `CHANGELOG.md`. The fused single-token decode MoE kernel applies to the routed experts when `MLXCEL_FUSED_MOE` is on.

### 4.3 Real-checkpoint validation (`models/kolibri-1-mlx-4bit`, M3 Ultra)

`mlxcel generate -p "Hello" -n 50 --temp 0` gives coherent reasoning text at 83 tok/s. A 792-token prompt gives an accurate outline of `docs/architecture.md` at 77 tok/s. `mlxcel-server` `/v1/chat/completions` with the same prompt returns the same text as the CLI.

Teacher-forced traces against the vendored mlx-vlm module (`scripts/kolibri1_oracle_trace.py`, corpus `docs/architecture.md`, `--decided 2`, mlxcel with `MLXCEL_FUSED_MOE=0`):

| Reference | Candidate | Top-1 disagreement | Decided | Perplexity |
|---|---|---|---|---|
| mlx-vlm, one 1024-token forward | mlxcel prefill | 0 / 1024 | 0 / 383 | 31.24 vs 31.24, max delta 0 |
| same | mlx-vlm `--rope-on-full` | 601 / 1024 | 155 / 383 (40.5%), fails | 116.28 |
| mlx-vlm chunks 4 to 7 of a `256 8 8 1024` plan | mlxcel width-one decode behind a 1024-token prefix, rings wrapped | 57 / 1024 | 1 / 388 (0.26%) | 18.10 vs 18.08 |
| same | mlx-vlm `--rope-on-full` | 685 / 1024 | 220 / 388 (56.7%), fails | 155.36 |

The trace's id-less `PREFILL` arm is not used: like AFMoE, this family opts out of chunked prefill, so an id-less multi-token forward starts a new prompt and the traced chunk loses its context.

---

## 5. Learning Points

### 5.1 A fluent model is not a correct model

Selecting experts on sigmoid plus bias produced fluent text. Only a teacher-forced comparison against the reference, with fixtures built so the two rules diverge, separates them. Build the fixture where the wrong rule gives a different answer.

### 5.2 Prefill agreement does not prove decode agreement

The prefill arm never wraps a sliding ring. Decode behind a 1024-token prefix does, and is the arm where a wrong ring offset or a stray RoPE on full layers shows up. Run both.

### 5.3 A negative arm calibrates the gate

`--rope-on-full` fails at 40.5% and 56.7% decided disagreement, against 0% and 0.26% for the correct port. That gap is what makes the passing numbers meaningful.

### 5.4 A native exception across the bridge is an abort

`mlx::core::stack` throws on mismatched shapes, and the exception crosses the C++/Rust bridge, so a malformed checkpoint aborts the process instead of returning an error. Validate shapes before calling into MLX where the call can throw.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| NoPE | A layer with no positional encoding | Kolibri-1 full-attention layers |
| `e_score_correction_bias` | A per-expert selection-only bias | Added for top-k selection, not to the weights |
| Teacher-forced trace | Feed the reference's tokens to both models and compare logits per position | The validation method |

### Related PRs/Issues

- Issue #2357: the requirements this PR implements.
- AFMoE: the family this one reuses.
- #3090: `last_row_logits` coverage rule.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 25 in the first two commits, plus `docs/environment-variables.md` and the two reports |
| Tests added | 19 (`models::kolibri1`) |

### Changes by Category

| Category | Summary |
|----------|---------|
| Model | `kolibri1.rs`, `kolibri1_moe.rs`, `kolibri1_weights.rs`, `kolibri1_lm.rs`, `kolibri1_tests.rs` |
| Refactor | AFMoE attention gate option and geometry, shared `hybrid_prefill_masks`, expert plane helper use |
| Wiring | Detection, loaders, registry, metadata, memory estimate, TP lists, snapshot sizing |
| Tooling | `scripts/kolibri1_oracle_trace.py` |
| Documentation | supported models, continuous batching, turbo KV cache, environment variables, CHANGELOG |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `0f0329df` | feat | add Kolibri-1 (kolibri1) hybrid sliding/full MoE |
| `5888260a` | docs | record Kolibri-1 trace validation against mlx-vlm |

---

## 8. Follow-up Actions

### Future Improvements

- Share the `LanguageModel` impl between AFMoE and Kolibri-1 (open MEDIUM).
- Make `drain_numbered_expert_planes` check plane shapes before `mlx::core::stack` so a malformed checkpoint returns an error instead of aborting (open MEDIUM, shared helper).
- Test the MXFP4 conversion and the EAGLE3 drafter.

---

## Appendix

### A. Unit tests

`models::kolibri1` (19), `models::afmoe` (32), `coverage_tests` (18), and clippy on lib and tests with `-D warnings`.
