# PR #2475: Implementation review of Qwen3.8-Flash-Next (qwen4_exp) support

> **Hardening update, 2026-10-23:** The sections below preserve the original 2026-10-20 review and its then-open items. Those runtime limitations have since been addressed: anonymous batches retain independent rows; OptiQ folded norm scales load correctly; bf16 reductions, recurrent precision, native indexed QSA and MRoPE match the updated reference. The real 80.8 GB checkpoint now passes CLI and server text/image/video/long-context checks, streaming, 13,308 tests, and paired performance measurements on M3 Ultra. See the [current validation report](../docs/benchmark_results/qwen4-exp-m3-ultra-2026-10-23.md) for results, raw data, source revisions and remaining feature exclusions.

**Date**: 2026-10-20
**Status**: Reviewed; four fixes pushed on the PR branch; three items left for a decision or for the 128 GB machine
**Languages**: Rust, Python (fixture generator only)
**Risk Level**: Medium (a new model family behind its own `qwen4_exp` detection arm; no shared function changed; the real 80.8 GB checkpoint has not been run)

## Conclusion

The PR implements every item of issue #2406 that can be checked without the real checkpoint, and it is wired into the code flow the same way the Qwen3.5 VLM is: detection, `model_metadata`, the registry (`mlxcel arch`), `LoadedModel`, `VlmRuntimeRef::Qwen`, `qwen_video_runtime`, `model_type_has_native_video` and the CLI video placeholder count. Nothing is a placeholder: every config rejection upstream makes has a named error, the four additions over Qwen3.5 (zero-centered RMSNorm, gated hyper-connections, the hashed n-gram PLE, QSA) are real implementations pinned to mlx-vlm outputs, and the out-of-scope inputs (`ple_storage`, FP8 `weight_scale_inv`, non-empty `deepstack_visual_indexes`) are refused by name.

The review found no CRITICAL or HIGH issue. It fixed two MEDIUM and three LOW items (one reuse miss, one test gap against `docs/adding-models.md`, dead code, a trivial test helper, imprecise docs) and leaves three items open:

- The `Used by:` lists of the shared helpers the port now calls do not name Qwen4-Exp. The change is a comment edit in shared files, which this review was not allowed to make.
- `forward_batched` without sequence ids runs every row against the one fallback state. Qwen3.5 does the same, and the engine always passes ids, so this is a family-wide decision rather than a Qwen4-Exp fix.
- Upstream computes four narrow quantized projections row by row during prefill (`_row_invariant_quantized_projection`); the port uses the batched projection. Only the real-checkpoint logit trace can say whether that matters.

## 1. What the PR is meant to do

Issue #2406 asks for `model_type: qwen4_exp` (`Qwen4ExpForConditionalGeneration`) as a VLM for text, image and video on the CLI and the server, ported from mlx-vlm's `mlx_vlm/models/qwen4_exp`, with fixture-parity tests for each component, a 4-layer model that matches upstream logits, one test per config validation error, and a run of `mlx-community/Qwen3.8-Flash-Next-OptiQ-2bit` plus a teacher-forced logit trace. The last two need the 128 GB machine; the PR body lists them as still open.

The review checked the PR against the issue, the PR body, `CONTRIBUTING.md`, `docs/code-guidelines.md` and `docs/adding-models.md`, compared each component against upstream at mlx-vlm `9527f2da` (`language.py`, `config.py`, `qwen4_exp.py`, `qsa_kernel.py`), and ran Codex's reviewer (`codex review --base origin/main`) as a second opinion.

## 2. Findings

| Severity | Location | Finding | Status |
|----------|----------|---------|--------|
| MEDIUM | `src/models/qwen4_exp_ple.rs:252` (`ShardedEmbedding::lookup`) | Per-shard gathers were joined with a left-deep chain of binary `concatenate`, one node per touched shard (up to 128), instead of the existing `mlxcel_core::concatenate_many`, whose doc says it exists to replace exactly that chain | Fixed, `fee17250` |
| MEDIUM | `src/models/qwen4_exp_config_tests.rs:209` | `docs/adding-models.md` ("Quantization Parameter Bounds") asks a per-prefix quantization block to be tested with the shared `HOSTILE_QUANT_PARAMS` list through the real loader, with a positive control; the PR tested one pair (bits 0) in one position | Fixed, `9bf50f43` |
| MEDIUM | `src/models/gated_delta.rs:54`, `:752`; `src/models/switch_layers.rs:1961`; `src/models/qwen_mrope_state.rs:77`-`184` | `docs/code-guidelines.md` ("Shared Function Comments", "When adding a new model that uses an existing shared function") wants Qwen4-Exp added to the `Used by:` lists of `GatedDeltaCache`, `gated_delta_update`, `moe_weighted_sum` and the `MRopeState` methods | Open: shared files, see section 4 |
| LOW | `src/models/qwen4_exp_ple.rs` (`PleLayer::from_weights`) | A check that the head vocabularies fit the padded table could never fire: `padded_vocab_size` is their sum rounded up | Fixed, `fee17250` |
| LOW | `src/models/qwen4_exp_parity_tests.rs` | `mrope_for` only forwarded three fields to `InterleavedMRoPE::new`; the first model test rebuilt what `tiny_model` / `prompt_ids` already build | Fixed, `22dbfacd` |
| LOW | `CHANGELOG.md`, `docs/supported-models.md`, `src/models/qwen4_exp_qsa.rs:15`, `tests/fixtures/qwen4_exp/generate_fixtures.py:9` | "dense up to 2048 tokens" is off by three: selection starts at 513 complete blocks, the 2052nd token. The state list omitted the pooled block keys. The generator named a nonexistent `src/models/qwen4_exp/parity_tests.rs` | Fixed, `7bb7127b` |
| LOW (Codex P2) | `src/models/qwen4_exp.rs:415` (`forward_rows_sequential`) | With `seq_ids = None` and more than one row, every row reads and writes the same fallback state | Open: same as `Qwen35Model::forward_batched_rows_sequential` (`src/models/qwen3_5.rs:1889`), see section 4 |
| LOW | `src/models/qwen4_exp_hyper.rs:102`, `qwen4_exp_gated_delta.rs:151`-`152`, `qwen4_exp_moe.rs:133` | Upstream routes `block_inject_weight`, `in_proj_b`, `in_proj_a` and `shared_expert_gate` through `_row_invariant_quantized_projection` (mlx-vlm `mlx_vlm/models/qwen4_exp/language.py:44`, used at `:809`-`810`, `:1194`, `:1203`) so a quantized prefill row computes what decode computes; the port uses the batched projection | Open: measure with the logit trace, see section 4 |

## 3. Fixes

### 3.1 One concatenate for the n-gram table gather

`ShardedEmbedding::lookup` routes each n-gram id to its row shard, gathers per shard, and reorders by source position. The join was:

```rust
gathered = Some(match gathered {
    None => rows,
    Some(previous) => concatenate(&previous, &rows, 0),
});
```

Each node copies everything joined so far, so a chunk touching many of the 128 shards copied the growing prefix once per shard. It now collects the per-shard rows and calls `mlxcel_core::concatenate_many(&parts, 0)` once (`src/models/qwen4_exp_ple.rs:273`); the final `take` is unchanged, so row order is the same. The parity tests cover it: the PLE fixture's table has three shards and the ids of the test sequence span them (`ple_layer_matches_upstream_and_split_prefill_equals_one_shot`, and the 4-layer model tests).

The same commit drops the padded-table check in `PleLayer::from_weights`, which compared the last head's offset plus size against `padded_vocab_size = total.div_ceil(divisor) * divisor` and so could not fail.

### 3.2 Hostile quantization pairs through the loader

`QuantizationTable::from_config` (`src/models/qwen4_exp_config.rs:548`) already bounds every entry with `validate_quantization_params`, the default and each override. The test now proves it for the whole shared list: `hostile_quantization_pairs_are_refused_as_default_and_as_override` drives the five `HOSTILE_QUANT_PARAMS` pairs through `text_config_from_full`, the loader's config entry, once as the top-level default the text config inherits and once as a module override, checks that each error names the module and blames the right field (`quantization.bits (` or `quantization.group_size (`), and uses an honest inherited block as the positive control. It replaces the single bits-0 row of the `rejects!` table.

### 3.3 Test helpers and docs

`mrope_for` is inlined at its two QSA tests, and `tiny_model_matches_upstream_logits_for_prefill_and_eight_decode_steps` uses `tiny_model` / `prompt_ids` like the other model tests. The docs now say that attention stays dense through a query's 2051st token and turns sparse from its 2052nd (`QsaIndexer::select` returns `None` while `key_len / compress_ratio <= block_topk`, `src/models/qwen4_exp_qsa.rs`), list the pooled block keys the index cache keeps, and point the fixture generator at `src/models/qwen4_exp_parity_tests.rs`.

## 4. Open items

### 4.1 `Used by:` lists (needs a shared-file edit)

Proposed change, comment only:

- `src/models/gated_delta.rs:54` (`GatedDeltaCache`) and `:752` (`gated_delta_update`): `Used by: Qwen3Next, Qwen3.5, Qwen4-Exp, KimiLinear`.
- `src/models/switch_layers.rs:1961` (`moe_weighted_sum`): add `Qwen4-Exp` to the list.
- `src/models/qwen_mrope_state.rs:77`-`184` (`MRopeState` methods): `Used by: Qwen2VL, Qwen3VL, Qwen3VLMoe, Qwen3.5, Qwen4-Exp`.

The PR also calls `qwen3_next::{SwitchGLU, MLP}`, `qwen3_5::mrope_position_source`, `qwen3_vl::{InterleavedMRoPE, rotate_half}` and `merge::merge_llava_any`; those carry no `Used by:` list today, or one that names families generically.

### 4.2 Batched forward without sequence ids

The engine passes the row ids on every multi-row call (`Engine::forward_rows`, `src/lib/mlxcel-core/src/engine/mod.rs:440`, and `Engine::prefill_cohort`, `src/lib/mlxcel-core/src/engine/prefill.rs:191`), so the CLI and server paths are not affected. A caller of `LanguageModel::forward_batched` with B > 1, such as `examples/profile_batched_decode.rs`, would mix rows. Qwen3.5's `forward_batched_rows_sequential` has the same behavior, so the fix (refuse multi-row calls without ids, or give each row a transient state) should be decided once for the model-owned families.

### 4.3 Row-invariant quantized projections

`_row_invariant_quantized_projection` uses `singleton_quantized_linear` (mlx-vlm `mlx_vlm/models/quantized_verifier.py:392`), a `gather_qmm` call that keeps the single-row `qmv` arithmetic for every row of a prefill. It only differs from the batched projection when the module is quantized, so the float32 fixtures cannot see it. By the checkpoint description in `docs/supported-models.md`, the OptiQ quant keeps the hyper-connections (`block_inject_weight`) at 8 bits and `in_proj_b` / `in_proj_a` at the 4-bit default, with the router and shared-expert gate in bf16. The teacher-forced trace in the PR's "Still open" list will show whether prefill positions disagree; porting it would need a shared single-row quantized projection helper.

## 5. Decisions checked as implemented

- QSA graph form: `QsaIndexer::build_mask` scatters the selected blocks onto the token axis, adds the query's incomplete tail block, and falls back to the causal row when a query has at most `block_topk` complete blocks, as upstream `build_mask` does; the KV and index caches grow together, so the mask's `key_len` matches the fetched keys.
- Row-by-row batching: `supports_batched_prefill`, `supports_padded_prefill` and `supports_paged_decode_backend` are `false`, and `Qwen4ExpVLModel` delegates all three (the trait default for padded prefill is `true`, which a wrapper must not inherit).
- PLE host hashing: `Qwen4ExpModel::host_tokens` reads the ids back once per forward, only when a layer has PLE.
- Fixture devices: `generate_fixtures.py` sets the CPU as the default device (`:610`) and runs only the float16 q/k normalization case under `mx.stream(mx.gpu)` (`:267`); `qk_normalization_matches_upstream_where_qwen35_normalization_does_not` pins that case bit for bit and checks that Qwen3.5's `normalize_qk` misses it by more than 1e-3.
- No final norm: `forward_hidden` returns `hyper_connection_mixer.collapse(...)` straight to the LM head, as upstream returns `self.hyper_connection_mixer(hidden_states)`.
- `Qwen4ExpMoeBlock` duplicates the routing of `SparseMoeBlock::forward` with `softmax_precise` and top-k over the gates, as upstream `Qwen3_5MoeSparseMoeBlock` does, and loads the routed experts, shared expert, router and shared-expert gate each with its own declared pair; `check_expert_packing` refuses a plane packed at another width.

## 6. Change summary

| Item | Value |
|------|-------|
| Files changed by the review | 7 |
| Lines added / deleted | +62 / -45 |
| Tests | one guard test replaces one `rejects!` row; the `qwen4_exp` filter still selects 48 tests |

| Commit | Message |
|--------|---------|
| `fee17250` | refactor(models): gather Qwen4-Exp PLE shard rows in one concatenate |
| `9bf50f43` | test(models): walk hostile quant pairs through the Qwen4-Exp loader |
| `22dbfacd` | test(models): inline the Qwen4-Exp MRoPE helper, reuse tiny_model |
| `7bb7127b` | docs: state the exact Qwen4-Exp QSA dense threshold and index state |

The fixes were checked with the narrow `qwen4_exp` unit-test filter, `cargo clippy --features metal,accelerate --lib --tests -- -D warnings` and `cargo fmt --all -- --check`; the workspace-wide suites are left to the orchestrator.

## 7. Follow-ups

- The real-checkpoint items of #2406 on the 128 GB machine: `mlxcel generate` text, `--image`, a prompt past 4096 tokens, the same through `mlxcel-server`, and the teacher-forced logit trace with the forced-dense arm. The trace also settles section 4.3.
- The `Used by:` edit in section 4.1.
- A decision for model-owned families on `forward_batched` without ids (section 4.2).

## Related

- Issue #2406; upstream Blaizzy/mlx-vlm#2032 (checked against mlx-vlm `9527f2da`).
