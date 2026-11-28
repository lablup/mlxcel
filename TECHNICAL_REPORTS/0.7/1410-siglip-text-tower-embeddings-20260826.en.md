# Technical Report: PR #1410 - SigLIP text tower on /v1/embeddings

**Date**: 2026-08-26
**Author**: mlxcel contributors
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: Low

---

## Executive Summary

PR #1410 (issue #1341, epic #1348) serves the SigLIP text tower through `POST /v1/embeddings` and `mlxcel embed`. A `google/siglip-base-patch16-224` checkpoint previously reached the embedding dispatcher and stopped at `not yet supported`; it now loads and returns 768-wide unit-norm vectors.

Two things in this port are worth a future maintainer's attention, and neither is the forward pass itself.

The first is that SigLIP's pooling is a consequence of its padding rather than a separate step. Its pad token and its EOS token are the same id, so right-padding every input to the 64 learned positions puts `</s>` at position 63 for every input, and a fixed slice at 63 reproduces the reference's pooling for a five-token caption and a sixty-four-token one alike. An implementation that reached for `PoolingMode::LastToken` and the attention mask would pool the same token at a different index and, because the tower is unmasked, get a different vector.

The second is that the epic's shared acceptance gate does not fit this family. It expects unrelated sentences to score below 0.5 cosine; SigLIP scores 0.707. That was investigated rather than adjusted, and the investigation is the substance of this report: an independently written NumPy implementation of the reference reproduces the same 0.707, and the family's whole text space turns out to sit on a floor of roughly 0.65. The absolute threshold is unreachable for a correct implementation, so the gate now asserts the margin the issue actually specified.

## 1. Problem Statement

`src/models/detection.rs` already routed `model_type: siglip` to `ModelType::SiglipText`, and `src/model_metadata.rs` already registered it as `ModelKind::Embedding`, both landed with the embedding foundation in #1353. What did not exist was any text tower: `src/vision/encoders/siglip.rs` implemented the vision side only, its encoder block was private, and `VisionEmbeddings` is a patch convolution with no token or position lookup and no projection head.

Three constraints made this more than a transcription.

**The encoder block already carried decisions that a copy would fork.** `gelu_pytorch_tanh` in that file is evaluated in f32 specifically to avoid a bf16 `x^3` overflow, and `select_mlp_activation` encodes a three-way compatibility matrix across SigLIP, CLIP and Idefics2 callers. A second copy of the block in the text module would have started identical and drifted on the first fix applied to one of them.

**The config is mostly absent.** The base checkpoint's `text_config` declares five keys: `hidden_size`, `intermediate_size`, `num_attention_heads`, `vocab_size`, `model_type`. The other four values the tower needs come entirely from defaults, and a wrong default does not fail loudly; the model still loads and still emits plausible unit vectors.

**No reference implementation was installed.** PyTorch, transformers and even NumPy were absent from the build host, so there was no oracle to check absolute values against, only self-consistency.

## 2. Technical Decisions

### 2.1 Share the encoder block, and prove the sharing did not move the vision path

The alternative was a copy, rejected for the drift reason above. Sharing costs one thing: `EncoderLayer::forward` gained an `Option<&MlxArray>` mask parameter that no caller currently passes `Some` to, since both the vision towers and the text tower run unmasked.

Adding a parameter to a numeric kernel is exactly the kind of change whose safety is easy to assert and hard to demonstrate, so it was demonstrated instead. A deterministic one-block fixture (an LCG-seeded weight map at `hidden = 8`, `intermediate = 16`, two heads) was run through the **pre-change** `EncoderLayer::forward(x)` and its 32 outputs captured. The mask parameter was then added and the same fixture asserted against those captured values through `forward(x, None)`. The ordering is what makes it evidence rather than a tautology: the numbers existed before the code they now guard.

A golden alone would not catch a `forward` that accepted the mask and ignored it, so a second test asserts both halves of the plumbing: an all-zero additive mask must reproduce the maskless path to 1e-6, and a mask blocking the last key column must move the output by more than 1e-4. The first half alone passes trivially for an implementation that drops the parameter; the second half is what fails it.

`EncoderLayer::from_weights_parts` takes an `EncoderBlockShape` (hidden size, head count, LayerNorm epsilon) instead of a `&VisionConfig`, so the text tower does not have to fabricate a `patch_size` and an `image_size` it has no opinion about.

### 2.2 Let the padding do the pooling

The reference pools `last_hidden_state[:, -1, :]` and comments that EOS is "sticky". Three facts turn that into a fixed slice:

| Fact | Source | Consequence |
|---|---|---|
| `pad_token` and `eos_token` are both `</s>` | `tokenizer_config.json` | position 63 holds `</s>` for every input, however short |
| `model_input_names: ["input_ids"]` | `tokenizer_config.json` | the reference processor emits no attention mask, so the tower is unmasked |
| `model_max_length: 64` and `max_position_embeddings` 64 | `tokenizer_config.json`, `SiglipTextConfig` default | inputs cap at 63 tokens plus the trailing `</s>` |

So `pad_to_max_length()` returns `Some(64)`, the engine pads every micro-batch to exactly that width, and `embed` slices index 63. `default_pooling()` reports `LastToken` for the startup log only; `1_Pooling/config.json` is not consulted and `MLXCEL_EMBEDDING_POOLING` does not apply, both deliberate.

The trap a future change should avoid: pooling through the shared `pool()` helper with `PoolingMode::LastToken` would select the last *real* token rather than index 63. For a short caption those are different positions holding the same token id, and because attention is unmasked the hidden states at those two positions differ. The result would look plausible and be wrong.

### 2.3 Do not inherit the vision side's activation default

`VisionHiddenActivation` derives `Default` as `ExactGelu`, deliberately, so older vision checkpoints that declare no `hidden_act` keep exact-erf GELU. `SiglipTextConfig`'s default is `gelu_pytorch_tanh`. Reusing the enum's `Default` would therefore have silently selected the wrong GELU for exactly the checkpoint this issue targets, which declares no `hidden_act`.

`SigLipTextArgs::hidden_act` is `Option<VisionHiddenActivation>`, where `None` (missing or explicit null) means `gelu_pytorch_tanh` and a declared string still resolves through the shared `select_mlp_activation`. All three paths are tested.

### 2.4 Build the oracle rather than skip absolute validation

With no reference framework installed, the honest options were to ship self-consistency only, or to construct an independent implementation. NumPy and `tokenizers` installed cleanly into a scratch virtualenv, so `siglip_text_reference.py` reimplements the reference forward pass directly over the checkpoint's own safetensors and `tokenizer.json`: tokenize, truncate to 63 plus `</s>`, pad to 64, token plus position embedding, twelve unmasked pre-norm blocks, final LayerNorm, `head`, L2 normalize.

| Path | Comparison against the NumPy reference |
|---|---|
| engine path, in-test, twelve pinned components | 2.7e-8 |
| `mlxcel embed` process, full 768-wide vector | 4.5e-5 |
| `mlxcel embed` repeated in a second process | bit-identical |

What this proves: the MLX op composition, the weight-key mapping, the tokenizer path, the truncation and padding rules and the f32 accumulation all reproduce a second, independently coded evaluation of the same architecture.

What it does not prove: both implementations were written by the same author from the same reading of the reference, so a shared misreading of the architecture would agree with itself. The independent evidence against that is semantic rather than numeric, and it is the ordering in section 5: a tower with, say, the wrong pooling index or a transposed projection would not place `cat`/`kitten` 0.26 above `cat`/`car engine`.

The two mlxcel paths differ from each other by about 4e-5 while each is bit-stable on its own, which is MLX selecting kernels per process. That spread is why the pinned tolerance is 2e-4 rather than the 1e-7 the in-test measurement alone would justify.

### 2.5 Treat the failing similarity threshold as a family mismatch, not a defect

The epic's shared gate expects unrelated sentences below 0.5 cosine. The first real-checkpoint run returned 0.707366 for `a photo of a cat` against `a diagram of a car engine`.

The NumPy oracle returns 0.707365 for the same pair, which rules out the implementation. Measuring the space rather than the single pair explains it:

| Statistic over six sentences (animals, machinery, finance, food, physics) | Value |
|---|---|
| unrelated pairs measured | 14 |
| minimum | 0.5187 |
| maximum | 0.7250 |
| mean | 0.6531 |
| the one related pair, `cat` against `kitten` | 0.9664 |

SigLIP's text tower is trained contrastively against images and never against other texts. Nothing in its objective pushes two unrelated captions apart, so the text space is anisotropic and its cosine floor sits near 0.65. An absolute 0.5 threshold is unreachable for a correct implementation of this family, and passing it would have been evidence of a bug, not of health.

The gate now asserts what the issue actually specified, a margin of at least 0.1 (observed 0.259), plus a loose ceiling that a genuinely collapsed tower would still breach, plus the absolute NumPy pin from 2.4. The geometry is documented in `docs/embeddings.md` so an operator ranks by margin instead of thresholding the score the way a sentence-transformers encoder invites.

### 2.6 Serialize the MLX tests, and discard the numbers taken without the lock

While this work was in review, a sibling unit measured concurrent MLX forward passes in `cargo test` corrupting each other: two byte-identical rows inside one batch scored a cosine of 0.999912 instead of 1.0. `EmbeddingModel` is documented as single-threaded and the product honors that (the server owns one embedding worker thread, `mlxcel embed` runs on the main thread), so the hazard is confined to the test harness, which runs test functions on a thread pool.

Every tolerance in these gates is tighter than 1e-4, so all of them were meaningless without a lock. One `OnceLock<Mutex<()>>` in `siglip_text::test_guard` now covers both new test modules, deliberately not one per module: they drive the same encoder block, so per-module locks would still let them run against each other. A poisoned lock is recovered rather than propagated, so one genuine failure does not cascade into every later test and hide which one broke.

Re-measured under the lock, three consecutive runs were identical to every printed digit:

| Quantity | Runs 1, 2, 3 |
|---|---|
| cosine of two identical inputs | 1.000000000 |
| `cat` against `kitten` | 0.966430 |
| `cat` against `car engine` | 0.707366 |
| margin | 0.259065 |
| padded batch against unpadded single input | 5.96e-8 |
| NumPy parity, twelve pinned components | 2.7e-8 |

### 2.7 One unguarded test in the module defeated the lock

Adding the guard exposed a second failure the corrupted-cosine report had not described. Repeating `cargo test --lib vision::encoders::siglip` aborted the process mid-run in roughly one attempt in four:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  cudaStreamEndCapture(stream, &handle_) failed: operation failed due to a previous error during capture
```

Two of the four tests had reported and the other two never ran, so this was not a teardown artifact: CUDA graph capture failed while a test was still executing. The cause was that the guard was incomplete. `pytorch_tanh_gelu_matches_hugging_face_f32_golden` predates this PR, lives in the same module, and evaluates MLX ops, so it kept running concurrently with the guarded block tests and could interleave with their graph capture. A lock only serializes the tests that take it.

Guarding that one pre-existing test removed the class entirely: over ten repeats afterwards, all four tests reported `ok` in ten of ten runs, with zero mid-run aborts.

Three of those ten still exited non-zero, all with a different and strictly post-results failure:

```
  what():  Destroy(handle_) failed: driver shutting down
```

That one fires after every test result has printed, so it cannot affect an outcome, and it is not this PR's. A control filter containing none of this PR's code, `cargo test --lib embeddings::`, reproduced the identical message in one run out of ten. It is a CUDA graph-handle destructor racing driver shutdown on this host, aggravated by three sibling units sharing the GPU, and it matches the pre-existing full-suite abort already known on this box.

| Filter | Runs | All tests reported ok | Mid-run capture abort | Post-results teardown abort |
|---|---|---|---|---|
| `vision::encoders::siglip`, before guarding the pre-existing test | 4 | 3 | 1 | 0 |
| `vision::encoders::siglip`, after | 10 | 10 | 0 | 3 |
| `embeddings::` control, no code from this PR | 10 | 10 | 0 | 1 |

## 3. Implementation Details

| File | Change |
|---|---|
| `src/models/siglip_text.rs` | New. `SigLipTextArgs`, `sanitize_siglip_text_weights`, `SigLipTextModel` with `encode` and the `EmbeddingModel` impl, `load_siglip_text_model`, and the shared `test_guard`. 309 lines. |
| `src/models/siglip_text_tests.rs` | New. Eight synthetic-tower and config tests plus two real-checkpoint gates that soft-skip when the checkpoint is absent. 635 lines. |
| `src/vision/encoders/siglip.rs` | Encoder block made `pub(crate)`; optional mask threaded through `EncoderLayer::forward` and `VisionAttention::forward_impl`; `EncoderBlockShape` and `from_weights_parts` added; all six internal call sites pass `None`. |
| `src/vision/encoders/siglip_block_tests.rs` | New. The pre-refactor golden and the mask no-op / mask-effective pair. 237 lines. |
| `src/embeddings/loader.rs` | `ModelType::SiglipText` arm constructs the tower; the two dispatcher parameters are no longer underscore-prefixed. |
| `src/models/mod.rs` | Module declaration and re-export. |
| `docs/supported-models.md` | Embedding models table row plus the fixed-width note. |
| `docs/embeddings.md` | Family notes section, and the current-status paragraph reworded so it no longer claims no family has landed. |

Weight sanitization drops `vision_model.*`, `logit_scale`, `logit_bias` and any `position_ids` buffer; everything else is used under the `text_model.` prefix, which a `SiglipTextModel` export keeps because the exported module owns a `text_model` attribute in both layouts.

One deviation from the issue body: it specified `position_embedding: UniquePtr<MlxArray>`, a raw table. `UnifiedEmbedding` is used instead. For a dense checkpoint it is the same lookup, it additionally loads a quantized conversion, and it mirrors how `VisionEmbeddings` reads the vision-side position table.

## 4. Test Coverage

| Test | What it pins |
|---|---|
| `text_config_defaults_match_the_reference_config` | all four undeclared defaults, and that the activation default is tanh rather than the vision enum's exact-erf |
| `text_config_overrides_are_read_including_projection_and_activation` | declared overrides, explicit null, and the no-`text_config` layout |
| `sanitize_drops_vision_and_logit_keys` | the four dropped key shapes |
| `pooling_takes_the_last_position_and_not_cls_or_mean` | that the pooled vector is `head(h[:, L-1, :])` and disagrees with both `head(h[:, 0, :])` and `head(mean)` |
| `every_position_reaches_the_pooled_slot` | bidirectional reach, and reproducibility of a repeated forward |
| `trait_surface_reports_fixed_width_padding_and_last_token_pooling` | the whole `EmbeddingModel` surface |
| `encode_rejects_more_tokens_than_learned_positions` | the position-table overflow error and that a shorter batch is accepted |
| `embed_rejects_image_inputs` | the image rejection |
| `siglip_base_detects_pads_to_64_and_keeps_trailing_eos` | detection, pad id, the 64-wide padding, and that truncation keeps `</s>` |
| `siglip_base_text_tower_passes_the_embedding_gate` | limits, unit norm, real-token accounting, the margin, the batch-vs-single agreement and the NumPy pin |
| `encoder_block_shared_with_vision_is_unchanged` | the pre-refactor golden |
| `an_all_attend_mask_is_a_no_op_and_a_blocking_mask_is_not` | both halves of the mask plumbing |

The real-checkpoint gates soft-skip when the checkpoint is absent, following `src/embeddings/real_checkpoint_tests.rs`, so a machine without it still runs the other ten.

## 5. Real-Checkpoint Results

`google/siglip-base-patch16-224`, `mlxcel embed`, kernels cached, repeated in two processes and bit-identical between them:

| Quantity | Value |
|---|---|
| detected family | `SiglipText` |
| vector width | 768 |
| `max_length` | 64 |
| `prompt_tokens` for four prompts | 33, not the 256 padded slots |
| cosine, two identical prompts | 1.000000000 |
| cosine, `cat` against `kitten` | 0.966439 |
| cosine, `cat` against `car engine` | 0.707386 |
| maximum element-wise difference from the NumPy reference | 4.461e-5 |

`mlxcel-server -m google/siglip-base-patch16-224` then `POST /v1/embeddings` with the same four inputs returned the same vectors, unit norm, and `usage.prompt_tokens` 33.

## 6. Validation Summary

| Command | Result |
|---|---|
| `cargo test --profile test-fast --features cuda --lib models::siglip_text` | 10 passed, 0 failed, three consecutive runs identical |
| `cargo test --profile test-fast --features cuda --lib vision::encoders::siglip` | 4 passed, 0 failed, ten of ten repeats |
| `cargo test --profile test-fast --features cuda --lib embeddings::` | 62 passed, 0 failed |
| `cargo test --profile test-fast --features cuda --lib models::detection` | 40 passed, 0 failed |
| `cargo check --profile test-fast --features cuda --all-targets` | exit 0 |
| `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |
| `cargo build --profile test-fast --features cuda --bins` | exit 0 |

Three of the ten `vision::encoders::siglip` repeats exited non-zero after every test had reported `ok`, on the pre-existing `Destroy(handle_) failed: driver shutting down` teardown race described in 2.7. The `embeddings::` control reproduced it once in ten runs with none of this PR's code, so it is environmental on this host rather than a property of these tests.

No performance measurement was taken; the epic runs one performance pass at the end on a quiet machine, and this host was running three sibling units concurrently.

## 7. Change Summary

| Metric | Value |
|---|---|
| Files changed | 8 |
| Lines added | 1,323 |
| Lines deleted | 32 |
| Tests added | 12 |
| Pre-existing tests brought under the lock | 1 |

## 8. Follow-up Actions

- Image embeddings through the SigLIP vision tower and its attention-pooling head are out of scope here and remain unserved; `supports_images()` is false and the error message says so.
- SigLIP 2 (`siglip2`) text towers are not detected and are not covered by this port.
- The per-module test lock does not serialize across modules. Every family in epic #1348 is adding its own, so the full lib suite will still run one family's real-checkpoint forwards against another's, and 2.7 shows what that costs: a single unguarded MLX test in the same module aborted the process in one run in four. A crate-wide MLX test guard is the actual fix and is worth filing once the epic's families have landed.
- The `Destroy(handle_) failed: driver shutting down` abort at process exit is pre-existing, reproduces on unrelated filters, and makes a passing run exit non-zero roughly one time in ten under concurrent GPU load. It deserves its own issue: results are unaffected, but it turns a green suite red at random and will be misread as a flaky test.
- The NumPy oracle lives in a scratch directory, not in the repository. The twelve pinned components in the real-checkpoint gate are what survives; regenerate the oracle if the tower's numerics are ever reworked.
- The roughly 4e-5 difference between the `mlxcel embed` process and the test binary on the same input is per-process kernel selection rather than a defect, but it sets the floor for any tolerance a future gate on this family can claim.

## References

- Issue #1341: serve the SigLIP text tower through `/v1/embeddings`.
- Epic #1348: embedding family ports.
- Issue #1353 and PR #1408: the embedding foundation this builds on.
- `docs/embeddings.md`: detection, pooling, limits, and the SigLIP family notes.
- `docs/supported-models.md`: the Embedding models table.
