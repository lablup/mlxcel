# Technical Report: PR #1070 - feat(models): add LocateAnything (locateanything) VLM support

**Date**: 2026-08-07
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed (the `pbd` parallel-box-decoding path and coordinate-token-to-box post-processing are explicitly deferred follow-ups)
**Languages**: Rust, Markdown
**Risk Level**: Medium (new model family; a loader crash, a process abort, and a quadratic DoS were found and fixed before merge)

---

## Executive Summary

PR #1070 adds LocateAnything (`model_type: locateanything`), NVIDIA's roughly 3B generative grounding VLM, closing issue #847. The model pairs a MoonViT vision tower and an MLP connector with a Qwen2 text decoder that mlxcel already supports, so the new work is the tower's two numeric deltas from Kimi-VL, the connector, the image-token splice, and the control-plane wiring. The PR reached merge over 7 commits and 29 files (+4007/-14 per the merge commit), 4 of which were review-driven fixes for defects that only the real checkpoint exposed: a hard MLX shape error from mixed 4/8-bit quantization, a missing `tokenizer.json`, a loader ordering bug that silently defeated an existing bf16-to-f16 conversion, and a quadratic tokenizer-registration cost that could wedge the loader for tens of minutes on a hostile `added_tokens_decoder`. All four are fixed and regression-tested. Real-checkpoint validation on `mlx-community/LocateAnything-3B-4bit` reproduces the model card's own documented output byte for byte.

---

## 1. Problem Statement

### 1.1 Background

LocateAnything's architecture is not novel to mlxcel: `docs/adding-models.md`'s reuse-first approach applies directly, because two of its three components already existed in the tree.

- **Vision tower**: MoonShot MoonViT-SO-400M, the same tower Kimi-VL (`kimi_vl` / `kimi_k25`) embeds. LocateAnything's upstream `vision.py` diverges from Kimi-VL's in exactly two numeric places: LayerNorm epsilon `1e-5` instead of `1e-6` (already a config field), and the block MLP's GELU is `nn.GELU(approx="precise")`, MLX's tanh approximation, instead of the exact erf form.
- **Text backbone**: `models::Qwen2Model`, reused verbatim. Qwen2 (`model_type: qwen2`) was already fully supported.
- **New surface**: the connector, the native-resolution image processor, the `<image-N>` marker splice, and the control-plane wiring (`ModelType::LocateAnythingVLM`, the `"locateanything"` detection arm, `LoadedModel`, `VlmRuntimeRef`, the loader route, the `generate_vlm` summary line, the TP arch-string table).

Grounding output is ordinary text. The checkpoint interleaves `<ref>`/`<box>` markers and 1001 coordinate tokens `<0>`..`<1000>` (ids 151677..152677) into its answer, so plain autoregressive decode is sufficient; no special detokenization runs. The checkpoint's parallel-box-decoding head (`pbd`, multi-token-prediction with `n_future_tokens: 6`) and coordinate-token-to-box post-processing were scoped out of issue #847 from the start and stay out of scope here.

### 1.2 Reuse Without Duplication: the MoonViT Delta

Rather than fork `src/vision/encoders/kimi_vl.rs`, the PR extends it. The block MLP activation became a config-carried enum:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoonViTMlpActivation {
    #[default]
    Gelu,
    GeluTanh,
}
```

`mlxcel_core` exposes no elementwise tanh-approximate GELU: both `gelu` and, despite its name, `gelu_approx` are erf-based. `GeluTanh` is therefore synthesized from primitives (`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`), the same pattern `models::kokoro::ops::gelu_new` already uses. The enum's `#[default]` is `Gelu` and the config field carries `#[serde(default)]`, so Kimi-VL's `config.json`, which has no such key, deserializes to prior behavior unchanged. `tests/kimi_vl_parity.rs` was not weakened: the only change there is the now-mandatory field in a test config literal.

### 1.3 Two Real-Checkpoint Gaps Synthetic Parity Would Have Missed

Both were found by running the actual checkpoint, not by reading `mlx_vlm`.

**Mixed 4/8-bit quantization.** `mlx-community/LocateAnything-3B-4bit` is not uniform 4-bit. The model card explains why: pure 4-bit on the tied `embed_tokens` destroys coordinate-token precision. Its `mlx_lm --quant-predicate mixed_4_8` conversion stores 18 of the 36 layers' `v_proj` at 8 bits while `q_proj` / `k_proj` stay at 4. Every per-tensor loader reconciles its own width from tensor shapes and is unaffected, but `FusedQKVLinear::from_weights_separate` concatenates the three packed planes along axis 0 and infers a single width from `q_proj`, which is a hard MLX shape error for a mixed layer (4-bit `q` is `[2048, 256]`, 8-bit `v` is `[256, 512]`). This is not LocateAnything-specific: any family on the shared fused-QKV loader hits the same wall on a `mixed_4_8` conversion.

**No `tokenizer.json` anywhere.** Neither `nvidia/LocateAnything-3B` nor its MLX conversions export a fast tokenizer, only `vocab.json` + `merges.txt` + `added_tokens.json` + `tokenizer_config.json`. `load_tokenizer` had nothing to read, so the model could not load at all.

### 1.4 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A `mixed_4_8` checkpoint fails to load with an opaque MLX shape error | High | Certain for this checkpoint, latent for any future family sharing the fused-QKV loader |
| A model directory with no `tokenizer.json` cannot load at all | High | Certain for this checkpoint (both NVIDIA's original and every MLX conversion) |
| A malformed `config.json` or `preprocessor_config.json` causes a panic or an unbounded allocation instead of a clean `Err` | High | Realized 4 separate ways during review (division by zero, connector-width truncation, unbounded resize, missing `in_token_limit` ceiling) |
| A large `added_tokens_decoder` wedges the loader with no error and no timeout | High | Realized during review: cleanly quadratic, 4x cost per doubling |
| An inverted `special` default silently strips added tokens from `skip_special_tokens` decode | Medium | Latent for any Qwen2-family checkpoint shipping only `added_tokens.json`; LocateAnything itself unaffected because its own decoder is fully flagged |

---

## 2. Technical Review

### 2.1 The Mixed-Precision Loader (`src/loading/vlm_locateanything_quant.rs`)

`densify_mixed_precision_qkv` walks every `self_attn` prefix, reconciles each of `q_proj` / `k_proj` / `v_proj`'s quantization layout via `mlxcel_core::layers::reconcile_quantization_layout`, and dequantizes all three planes of a layer whenever their bit widths or group sizes disagree:

```rust
// SAFETY: `w` and `s` are borrowed from live map entries for the
// duration of the call, and `b_ptr` is either null or borrowed from a
// live entry in the same map.
let dense = unsafe { mlxcel_core::dequantize(w, s, b_ptr, group_size, bits, mode) };
```

Dequantization is exact: it is the stored representation's own definition, so it changes no value the model computes with. A test asserts `max_abs_diff(before, after) == 0.0` for each of the three planes on a synthetic mixed layer.

The alternative, requantizing the narrow (4-bit) planes up to the wider (8-bit) width, was tried and rejected on measurement: MLX's affine quantizer snaps the group scale onto the larger-magnitude edge rather than using a plain `(max - min) / (2^bits - 1)`, so a 4-bit group does not land exactly on the 8-bit grid. A round trip through requantization perturbed a synthetic plane by 3.7e-3, versus exact for dequantization. The cost of dequantizing is roughly 190 MB on the released 3B checkpoint (18 affected layers' QKV planes going from packed 4/8-bit to dense bf16), applied only to genuinely mixed layers; uniform 4-bit and uniform 8-bit checkpoints each have a positive-control test proving `densify_mixed_precision_qkv` leaves them untouched. Fixing this at the root needs a `FusedQKVLinear` that can hold per-projection quantization widths, which is a shared-layer change left as a follow-up, not scoped to this PR.

### 2.2 The Tokenizer Fallback (`src/tokenizer/mod.rs`)

`build_qwen2_bpe_tokenizer` reconstructs the tokenizer the way `transformers`' `Qwen2Converter` does, component for component: byte-level BPE with no unk token and no subword prefix or suffix, an NFC normalizer, `Sequence[Split(PRETOKENIZE_REGEX, isolated), ByteLevel(add_prefix_space, use_regex=false)]` as the pre-tokenizer (the regex copied verbatim from `transformers/models/qwen2/tokenization_qwen2.py`), a ByteLevel decoder, and a ByteLevel post-processor with `trim_offsets = false`.

The fallback is gated on `is_qwen2_slow_tokenizer_dir`, which requires both `vocab.json` and `merges.txt` to exist *and* `tokenizer_config.json`'s `tokenizer_class` to name `Qwen2Tokenizer` or `Qwen2TokenizerFast`:

```rust
fn is_qwen2_slow_tokenizer_dir(model_path: &Path) -> bool {
    if !model_path.join("vocab.json").exists() || !model_path.join("merges.txt").exists() {
        return false;
    }
    // ... tokenizer_class must name the Qwen2 tokenizer
}
```

`vocab.json` + `merges.txt` is the generic GPT-2 slow-tokenizer pair, so the class check exists specifically to keep another family from being silently tokenized with Qwen2's rules. That is not a hypothetical: `moondream2`'s real checkpoint directory ships that exact GPT-2 file pair but declares `CodeGenTokenizer` in `tokenizer_class`, so it is excluded twice over (once by the class mismatch, and its `tokenizer.json` branch upstream in `load_tokenizer` returns unconditionally before this check is ever reached). This was checked against the real local `moondream2` model directory, not inferred from reading the code. In `load_tokenizer`, the new branch sits last, after `tokenizer.json`, `tokenizer.model`, tiktoken vocabularies, and `tokenizer.jsonl`.

Added tokens are registered in ascending id order and then verified against the ids the checkpoint declares, since `tokenizers` assigns added-token ids sequentially from the base vocab size and any drift would silently shift every later token. LocateAnything's 1038 added tokens include the 1001 coordinate tokens that carry its box output, so a shift there would corrupt every box rather than fail loudly. The construction was validated against a `transformers` `Qwen2Tokenizer` oracle built from the real checkpoint directory on 7 cases: a rendered ChatML prompt, a `<ref>`/`<box>` grounding string, repeated whitespace and newline runs, tabs, CJK and accented Latin text, and the apostrophe contractions the regex special-cases. All 7 encode token-exact and decode round-trip exactly.

### 2.3 Image-Token Splice and Control-Plane Wiring

`<image-N>` is what the chat template renders and is plain text rather than a vocabulary token, so `multimodal::locateanything_prompt` rewrites the rendered prompt into `<img> + <IMG_CONTEXT> * (grid_h * grid_w / merge_length) + </img>` and re-encodes it, matching upstream's `re.sub`. A token-level splice covers `--no-chat-template` runs. The runtime then verifies the encoded stream carries exactly as many `<IMG_CONTEXT>` ids as the tower will emit feature rows before scattering them via `vision::merge::merge_llava`, so a drift cannot silently place image features into text positions.

`"locateanything"` has to win over the generic `"qwen2"` arm at the top level of `src/models/detection.rs`, because the LocateAnything text sub-config also declares `model_type: "qwen2"`; without the dedicated arm the grounding VLM would detect and load as a text-only Qwen2 model. `LocateAnything's backbone is `llama`-family for the TP arch-string table (`src/distributed/tensor_parallel/inference.rs`), though tensor parallelism is refused earlier for all VLM-kind models, so this entry only keeps the dispatch table total rather than enabling anything new.

### 2.4 Compatibility

- **Breaking changes**: none. This is a purely additive model family.
- **Shared code touched**: `src/vision/encoders/kimi_vl.rs` gained the config-carried activation field; Kimi-VL and Kimi-VL 2.5 behavior is unchanged because the default is the prior hardcoded value.
- **Convention followed, not changed**: `language_weights_subset` (`src/loading/vlm_locateanything.rs`) copies the entire text-stack weight subset (`mlxcel_core::copy(value)` per tensor, roughly 1.8 GB for this checkpoint) rather than moving or aliasing it, matching the existing `vlm_internvl.rs` and `vlm_lfm2_vl.rs` loaders. Changing that is a shared-convention question across all three loaders, not a defect specific to this PR.

---

## 3. Defects Found and Fixed During Review

Four commits after the initial feature commit fixed defects the real checkpoint (or a hostile-config sweep) exposed. Each is recorded here with its severity as assessed during review.

### 3.1 HIGH: bf16-to-f16 Conversion Ran Before Mixed-Precision Densification

`load_locateanything_vlm`'s Apple Silicon bf16-to-f16 pass (`models::convert_bf16_weights_with_keep`, keeping `.scales` / `.biases` at bf16) originally ran *before* `densify_mixed_precision_qkv`. MLX's `dequantize` returns an array carrying the scales' dtype, and the scales are bf16, so every dense q/k/v plane the densification pass inserted was a fresh bf16 tensor created *after* the only pass that would have converted it: 54 tensors (18 mixed layers x 3 projections) on the released checkpoint, left at bf16. That is exactly the M5 JIT crash the conversion pass exists to avoid, defeated by ordering.

The fix reorders the two passes so densification runs first; its output is converted along with the MoonViT tower and the connector, and because the densified planes no longer carry `.scales` / `.biases` of their own, the keep-predicate no longer exempts them. A follow-up commit extracted the two inline passes into `reconcile_mixed_precision_weights(weights, group_size, bits, mode, convert_bf16: bool)`, taking the Apple Silicon gate as a parameter instead of reading `hardware::get_hardware()` internally, specifically so a test could force the conversion branch independent of the host it runs on. Two regression tests quantize a bf16-sourced dense plane the way the released checkpoint's mixed layers do (confirming the scales stay bf16, the premise of the bug), densify it, and assert the densified planes land on f16 only when `convert_bf16` is `true`; the control test with the flag `false` proves the f16 dtype in the first test comes from the conversion pass and not from densification itself. This mattered because the CUDA box the PR was developed on never exercises the Apple Silicon branch regardless of test placement, so without the parameterized extraction the fix would have shipped without a test that could actually run.

### 3.2 MEDIUM: Added-Token `special` Defaults Inverted HuggingFace's Convention

`read_added_tokens_sorted` defaulted an added token's `special` flag to `true` in both branches it handles. HuggingFace's own `AddedToken` defaults `special` to `false`. The `added_tokens.json` branch was worse: that file is a flat `content -> id` map carrying no flags at all, so every one of its entries was forced special regardless of what it represented.

`AddedVocabulary`'s `special_tokens_set` is insert-only, a property `src/tokenizer/mod.rs` already documents for `demote_tool_parser_markers` in the context of issue #778. A content-bearing token registered special under the old default could never be demoted afterward, and was silently stripped from every `decode(.., skip_special_tokens = true)` call. Both defaults are now `false`. LocateAnything itself is unaffected either way, because its `added_tokens_decoder` carries an explicit `special` flag on all 1038 entries; the bug was latent for any other Qwen2-family checkpoint that ships only `added_tokens.json`. Two unit tests cover both branches: `added_tokens_decoder_defaults_special_to_false` (an explicit `special: true` survives; a missing key defaults to non-special) and `added_tokens_json_fallback_tokens_are_not_special` (nothing in that file justifies marking an entry special).

### 3.3 HIGH: Division by Zero and Truncation from `merge_kernel_size`

`merged_token_count` computed `(merge_kernel_size[0] * merge_kernel_size[1]) as i32` and divided by it. `merge_kernel_size: [65536, 65536]` is a `usize` product of exactly 2^32, which the narrowing `as i32` cast truncates to 0, and the next line panicked with "attempt to divide by zero." That pair is square and non-zero, so it passed both existing guards (`.max(1)` and the square-kernel check). The fix forms the merge product with `saturating_mul`, clamps it into `1..=i32::MAX` before the cast, and forms the patch product in `i64` so two `i32` grid sides cannot overflow either.

A second instance one layer up hit the connector: `(hidden_size * merge_h * merge_w) as i32` truncated the same way. With `hidden_size: 1152` and the same `[65536, 65536]` merge, `input_dim` became 0, which reaches `LocateAnythingConnector::forward`'s `reshape(image_features, &[-1, 0])` and throws inside MLX; a C++ throw across the cxx boundary aborts the process rather than unwinding into a `Result`. This became `connector_input_dim`, using `checked_mul` plus `i32::try_from` and returning a descriptive `Err` naming the offending config values instead of a truncated zero.

`patch_size` and `merge_kernel_size` are rejected outright at the loader (`to_moonvit_config`) rather than clamped, because both values are consumed twice by consumers that must agree: the processor derives the patch grid from them, while the MoonViT conv patch-embed and patch merger are sized from the `KimiVLVisionConfig` built alongside. Clamping only one side would turn a loud failure into a quiet desync between the two. The bounds (128 for `patch_size`, 16 per merge axis) sit well outside the released geometry (`patch_size: 14`, `merge_kernel_size: [2, 2]`), which a dedicated test asserts stays inside the range unchanged.

### 3.4 HIGH: Quadratic Tokenizer Registration (Denial of Service)

`build_qwen2_bpe_tokenizer` originally registered added tokens one call per token. Every `Tokenizer::add_tokens` / `add_special_tokens` call ends in `AddedVocabulary::refresh_added_tokens`, which rebuilds both Aho-Corasick automatons over every token accumulated so far, and `add_tokens` additionally clones the whole existing token set first. N separate calls therefore cost N full rebuilds: cleanly quadratic, measured at a 4x cost per doubling on the pinned `tokenizers` 0.22.2 in a release build (500 entries 46.9 ms, 1000 162 ms, 2000 609 ms, 4000 2.42 s, 8000 10.1 s, 16000 46.1 s). Extrapolating that quadratic curve to a 100,000-entry `added_tokens_decoder`, a value nothing prevented a hostile `tokenizer_config.json` from declaring, points to on the order of half an hour with no error and no timeout. LocateAnything's own 1038 tokens already paid 175 ms for this instead of 0.5 ms. It was also a regression against `main`, where added tokens previously arrived only through `tokenizer.json`, whose deserializer makes a single batched call.

The fix batches registration over consecutive runs of equal `special` flag, not a single specials group plus a single normals group: `tokenizers` hands out ids in the order tokens are passed, so regrouping is not id-preserving (`[A(special), B(normal), C(special)]` declared as 7, 8, 9 would come back A=7, C=8, B=9 under a two-group split). Runs keep the sorted order intact, so a uniformly special checkpoint (LocateAnything, and the common case generally) collapses to a single call, while a perfectly alternating sequence degrades to the previous per-token cost without ever moving an id. End to end through the loader on synthetic checkpoints, the fix measured 1038 tokens at 197.6 ms to 1.5 ms, 4000 at 2.54 s to 4.8 ms, and 16000 at 44.0 s to 23.3 ms, with the resulting added-token tables and encodings identical in every case. Run-batching was verified id-identical against the old per-token loop across 33 flag patterns: all-special, all-plain, both alternations, the `[special, normal, special]` trap, mixed runs, the real 1038-token LocateAnything shape, and 24 seeded random sequences.

The same commit added `validate_dense_vocab_ids`, rejecting a `vocab.json` whose ids are not exactly `0..len`. `BPE::get_vocab_size()` reports `vocab.len()` regardless of which ids are actually occupied, and `AddedVocabulary` starts assigning added-token ids from that count. A gap below `len` therefore aliases an added token onto an id the base vocab already owns: measured directly with `{"Ġ":0,"h":1,"Ġh":2,"i":3,"COLLIDE":5}` plus an added token declaring id 5, `token_to_id("COLLIDE")` and the added token both resolve to 5 after registration, and `decode([5])` stops returning `COLLIDE`. The library exposes no way to steer the starting id, so refusing a sparse file is the only sound response; all 65 `vocab.json` files in the local model set are dense, so no real checkpoint is turned away.

### 3.5 MEDIUM: Additional Hardening

- **Unbounded resize.** `LocateAnythingProcessor` rounds each image side up to the next multiple of `merge * patch`, bounded only by a 511-patch grid envelope. A declared `patch_size: 100000` asked `resize_exact` for a roughly 200000x200000 RGB buffer (over 100 GB), OOM-aborting the process; the `in_token_limit` downscale could not help, since `(w / p) * (h / p)` is 0 once `p` exceeds the image. `patch_size` and `merge_kernel_size` rejection (3.3) closes this at the loader; `LocateAnythingProcessor::new`'s public constructor keeps a backstop clamp pinned equal to the loader's constants by a cross-module test.
- **Unbounded `in_token_limit`.** Read from `preprocessor_config.json`, the only value that engages the downscale at all; left unbounded, a single image was bounded only by the grid envelope, roughly 614 MB of f32 patch data at `patch_size: 14` plus the resized RGB image and the MLX array copy. Clamped rather than rejected, since the tower never sees this value and nothing downstream can desync.
- **Silent vocabulary aliasing.** Covered by `validate_dense_vocab_ids` above (3.4); listed separately because it is a distinct failure mode (silent wrong output) from the DoS it was fixed alongside.

---

## 4. The `unsafe` Block: Audit

`dequantize_plane_in_place` (`src/loading/vlm_locateanything_quant.rs`) is the one `unsafe` block the PR introduces:

```rust
let b_ptr = weights
    .get(&format!("{prefix}.biases"))
    .and_then(|b| b.as_ref())
    .map(|r| r as *const mlxcel_core::MlxArray)
    .unwrap_or(std::ptr::null());
let dense = unsafe { mlxcel_core::dequantize(w, s, b_ptr, group_size, bits, mode) };
```

This is the established cxx nullable-`biases` bridge pattern, with 17-plus existing call sites elsewhere in the tree. `b.as_ref()` yields `None` for a null `UniquePtr`, producing a genuinely null pointer rather than a dangling one; `w` and `s` are borrowed from live `WeightMap` entries for the duration of the call. `reconcile_quantization_layout` validates `group_size`, `bits`, and the `mode` string before `dequantize_plane_in_place` is ever reached, so a malformed checkpoint's mixed layer produces a clean `Err` upstream rather than reaching this call with unvalidated inputs. No new soundness surface is introduced.

---

## 5. Validation Evidence

### 5.1 Real-Checkpoint Generation

GB10, CUDA (sm_121), release build, `mlx-community/LocateAnything-3B-4bit`, on the COCO image the model card's own usage example uses:

```
$ ./target/release/mlxcel generate -m models/mlx/locateanything-3b-4bit \
    --image 000000039769.jpg -p "Detect all objects in the image." -n 128 --temp 0.0
LocateAnything: inserted 1 image block(s) (414 total image tokens)
<ref>object</ref><box><64><152><273><244></box><box><520><160><580><392></box>
```

The first box, `<64><152><273><244>`, is byte-identical to the box the model card documents for this image (it labels the same region `remote`; the 4-bit conversion's `object` label instead of `remote` is the semantic generalization the card itself documents). A referring query, "the cat on the right," returns `<541><49><1000><778>`, which maps to roughly (346, 24, 640, 372) px against a COCO ground-truth box of (345, 23, 640, 368).

### 5.2 Tokenizer Oracle Parity

Separately validated against a `transformers` `Qwen2Tokenizer` oracle built from the real checkpoint directory: 7 cases (ChatML prompt, grounding string, whitespace/newline runs, tabs, CJK, contractions) all encode token-exact and decode round-trip exactly.

### 5.3 Full Test Suite and Merge Gate

Per-area suite counts at merge: `locateanything` 53 passed, `tokenizer` 75 passed (5 ignored), `kimi_vl` 25 passed, `detection` 80 passed, `model_metadata` 8 passed, `loading::` 255 passed, `vision::` 314 passed (17 ignored), `multimodal::` 187 passed (17 ignored), `kimi_vl_parity` 3 passed. `cargo clippy --features cuda --lib --tests -- -D warnings` clean, `cargo fmt --all` clean.

The full CUDA merge gate on the merge candidate reported 7365 passed, 5 failed; those same 5 fail identically on `main` and were control-run there (4 are TF32 artifacts that pass with `MLX_ENABLE_TF32=0`, 1 is a pre-existing integer underflow in `resolve_paged_block_budget`). The PR added 20 net passing tests.

---

## 6. What Was Deferred

- **The `pbd` parallel-box-decoding path.** Multi-token-prediction heads (`n_future_tokens: 6`) for faster box emission. Out of scope for issue #847 from the start.
- **Coordinate-token-to-box post-processing.** Boxes are currently returned as raw `<N>` coordinate tokens in generated text; mapping them to pixel or normalized coordinates is left to the caller (or a follow-up).
- **Apple Silicon bf16-to-f16 exercise.** The conversion branch is now unit-tested via the parameterized `reconcile_mixed_precision_weights`, but was not exercised end to end, because this checkpoint was developed and validated on a Linux CUDA box.
- **`FusedQKVLinear` per-projection widths.** The dequantize-on-mixed-precision workaround (2.1) removes the immediate blocker but leaves the shared loader unable to represent per-projection quantization widths natively; any other family converted with `mixed_4_8` pays the same 190 MB-class cost until that is addressed.

---

## 7. Lessons

- **A checkpoint that validates against a synthetic parity harness can still fail to load.** Both defining gaps in this PR (mixed 4/8-bit quantization, the missing `tokenizer.json`) were invisible to any test built from a well-formed synthetic config; they surfaced only when the actual released checkpoint was loaded.
- **A correctly reasoned pass ordering is not guaranteed by two individually correct passes.** The bf16-ordering bug was not a logic error in either `densify_mixed_precision_qkv` or the bf16-to-f16 conversion; each was correct in isolation. The bug was purely in which ran first, and it silently defeated a mitigation the loader's own documentation says exists specifically to avoid an M5 JIT crash.
- **A default that inverts an upstream library's convention is not neutral.** `AddedToken::special` defaulting to `true` instead of HuggingFace's `false` had no effect on this PR's own checkpoint (whose decoder is fully flagged) but was a live bug for every other checkpoint on the same code path, and the insert-only `special_tokens_set` meant the bug was unrecoverable once triggered, not just wrong.
- **A square, non-zero value can still overflow.** `merge_kernel_size: [65536, 65536]` passed every guard written to catch degenerate merge kernels (non-square, zero) precisely because those guards were not written with a narrowing-cast overflow in mind.
