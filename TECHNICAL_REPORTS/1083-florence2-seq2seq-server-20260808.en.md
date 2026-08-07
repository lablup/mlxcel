# Technical Report: PR #1083 - feat(server): serve Florence-2 through a seq2seq worker loop

**Date**: 2026-08-08
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Completed (three request-boundary findings fixed in review; four shared single-stream-worker limitations recorded rather than fixed)
**Languages**: Rust, Markdown
**Risk Level**: Medium (a family that the server previously refused outright is now reachable over HTTP; every other family's wire shape and worker path is untouched)

---

## Executive Summary

Florence-2 worked through the CLI and nowhere else. `mlxcel-server` refused the checkpoint at startup with a named error added by #856, because mlxcel's generation engine is decoder-only everywhere except Whisper's ASR pipeline and Florence-2 is BART-style seq2seq: one encoder pass over the fused vision-plus-prompt sequence, then an autoregressive decode that cross-attends to the cached encoder output. Handing that to a decoder-only worker would have served garbage from the trait-completeness forward, so the refusal was the correct interim behavior rather than a bug.

This lands the serving path. A dedicated single-stream worker (`src/server/florence2_worker.rs`) branches off the same `mpsc` request channel the batched worker uses, in both spawn paths, before any scheduler starts. The CLI's answer renderer moved into the library so that `message.content` over HTTP is the same string the CLI prints, byte for byte, which is what makes the acceptance criterion checkable at all. The structured coordinates ride alongside as `message.florence2_result`, following the `reasoning_content` optional-field convention the server already had.

The interesting part is not the worker loop, which is short. It is that per-request isolation is structural rather than enforced: the encoder output and the seq2seq decode cache are created inside `run_task_with_cancel` and dropped when it returns, so there is no shared state to leak. The test that pins this does not merely assert that two sequential requests agree with a fresh run; it also proves that decoding against the two requests' encoder outputs yields measurably different logits, so a leaked cache would change the answer rather than coincide with it.

---

## 1. Problem Statement

### 1.1 Background

Epic #850 landed Florence-2 across #852 (BART seq2seq stack), #853 (DaViT tower), #854 (fusion), #855 (processor and post-processing), and #856 (CLI pipeline), and #1082 added the quantized load path. All of it was reachable only from `mlxcel generate`. `start_server` bailed:

```
Florence-2 is an encoder-decoder (seq2seq) VLM that mlxcel-server cannot serve
yet. Run it through the CLI instead: mlxcel generate -m <model> --image <image>
-p '<CAPTION>' (or another task marker such as <OCR> or <OD>).
```

### 1.2 Existing issues

- **The family was unreachable over HTTP.** Every deployment that speaks OpenAI-compatible chat completions had no way to use Florence-2 at all, including the quantized conversions #1082 had just unblocked.
- **The response shape was an open design question.** OpenAI's chat schema has no place for bounding boxes, quad boxes, polygons, or OCR regions. The issue named this as a decision the work had to settle, not a detail to improvise.
- **Two security requirements were handed forward from #855 and had to be honored by any server surface.** `preprocess_with_sizes` takes an already-decoded `DynamicImage`, so decompression-bomb defense cannot live in the processor. And `Florence2Task::expand` interpolates caller-supplied text into the encoder prompt for 7 of the 15 task modes; the CLI sidesteps that by only accepting the operator's own flag, a server cannot.

### 1.3 Risk assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A Florence-2 checkpoint reaching a decoder-only worker loop after the refusal is removed | High (silently serves garbage) | Certain if any spawn path is missed |
| Encoder state shared or leaked across sequential requests on one loaded model | High (cross-contaminated answers, silent) | Low, but undetectable without a distinguishability test |
| Untrusted task-prompt text reaching `Florence2Task::expand` unvalidated | High (marker smuggling into the encoder prompt) | Certain without an explicit boundary check |
| Image decode running before the admission limits | High (decompression bomb) | Certain if the worker skips `decode_request_images` |
| HTTP answers drifting from CLI answers as the renderer is duplicated | Medium (breaks the acceptance criterion quietly) | Certain over time with two copies |

---

## 2. Technical Review

### 2.1 Security

Both #855 handoffs are honored, and both were re-verified during review rather than taken on the PR's word.

**Checklist:**

- [x] Input validation: `validate_task_input` bounds and shape-checks every input-taking mode at the request boundary.
- [x] Resource bounds: image bytes decode through `decode_request_images`, which applies `current_image_input_limits()` (payload size, dimension caps, decode-allocation cap) before any pixel work. The HTTP boundary applies the payload bound a second time in `try_collect_image_data_with_limits`.
- [x] No new authentication or authorization surface.
- [x] Logging carries no request text or image bytes.

**The task-input boundary, in detail.** `parse_task_prompt` admits only a recognized marker out of fifteen. On top of that, for the 7 input-taking modes: at most 2048 bytes, no control characters, and then a per-class rule. The four region tasks require exactly `<loc_a><loc_b><loc_c><loc_d>` with each bin in `0..=999`, parsed by a strict scanner that rejects junk between, before, or after the tokens, non-numeric or over-long digit runs, and out-of-range values. The three free-text tasks reject `<` and `>` outright, so no location, sequence, or task marker can be smuggled into the encoder prompt.

A drift guard pins the partition: `input_taking_task_set_matches_takes_input` walks `Florence2Task::ALL` and asserts the validator's two task sets agree with `takes_input()`, so adding a sixteenth task that takes input cannot silently bypass the boundary.

**Issues found:**

| Issue | Severity | Status |
|-------|----------|--------|
| Cancelled requests paid a full encoder pass before the first cancellation poll | Medium | Fixed in `cf4faa86` |
| Media, image-cardinality, and finish-reason mapping had no unit coverage | Medium | Fixed in `cf4faa86` |
| `MAX_TASK_INPUT_BYTES` comment claimed a "before tokenization" property the server path does not have | Low | Fixed in `cf4faa86` |
| `response_format` accepted and silently ignored (shared with the other single-stream workers) | Medium | Documented, not fixed (see 8) |
| `--max-queue-depth` not honored on any single-stream worker | Medium | Recorded, out of scope (see 8) |
| Declared vs resolved image cardinality not cross-checked | Low | Recorded, matches the documented MLX-worker convention (see 8) |

An initial review pass reported that malformed client requests return HTTP 500 because the chat route labels generation errors `server_error`. That did not reproduce. `ErrorResponse::new` hard-codes `StatusCode::BAD_REQUEST`; only the `error.type` string reads `server_error`, which is a pre-existing cosmetic inaccuracy across the whole server and not specific to this path.

### 2.2 Performance

Serving is one request at a time. The issue permits this for a first landing if it is documented, and it is, in `docs/supported-models.md` and in the worker's module doc. The reason it is not merely a shortcut: the encoder pass has a different cost profile from the decode loop, so dropping Florence-2 into the continuous-batching admission logic would be a guess, and no batched admission policy for it has been designed or measured. No concurrent-throughput property is claimed.

Serial service is exactly why the cancellation gap mattered. `generate_greedy_with_cancel` polls the flag once per decode step, which sounds sufficient until you notice the poll only begins *after* the encoder. A request queued behind another whose client had already disconnected still paid for the DaViT tower and the bidirectional BART pass over the fused 577-plus-prompt sequence, which is the expensive half. The fix is a flag check at the top of the handler.

The two-phase generation timeout was checked for interaction and is safe. Phase 1 blocks indefinitely by design so long prefills are never aborted, and the Florence-2 worker emits nothing until generation completes, so the whole run sits in Phase 1 and the bounded decode-hang window never applies to it.

### 2.3 Compatibility and dependencies

- **Breaking changes**: none. `florence2_result` and `GenerationResult.structured_output` are both `Option`, and the response field carries `skip_serializing_if = "Option::is_none"`, so every non-Florence-2 response serializes exactly as before. A test asserts the key is absent, not merely null.
- **New dependencies**: none.
- **Compatibility**: `mlxcel-server` accepts the checkpoint at startup and reaches a serving state. The startup refusal is gone; the same `get_model_type` check now gates only the text-only `"Hello"` warmup, which cannot run against an image-task model and would only have logged a spurious failure.

### 2.4 Code quality

The renderer relocation is the structural point. `render_task_result` moved out of `src/commands/generate_florence2.rs` into `src/models/florence2/render.rs`, so the CLI prints and the server returns the same function's output. Byte-identity between the two surfaces is then a property of the code rather than a claim to re-verify on every change. The move also dropped the old `_ => String::new()` catch-all: `Florence2TaskResult` is `#[non_exhaustive]`, which forced a wildcard on the out-of-crate CLI copy but not on the in-crate one, so a new result variant now fails to compile instead of silently rendering as empty.

Test coverage added: 516 lines across four files. `florence2_render_tests.rs` covers both the text and JSON forms of all five result variants; `florence2_worker_tests.rs` covers the region parser, the validator across all fifteen tasks, and (after review) the three mapping guards; `florence2_tests.rs` adds the sequential-isolation test.

---

## 3. Technical Decisions

### 3.1 Where the structured coordinates go

**Context:** OpenAI's chat schema has no field for boxes, polygons, or OCR regions, and the issue required this to be settled rather than improvised.

**Alternatives considered:**

| Option | Pros | Cons |
|--------|------|------|
| Serialize JSON into `content` | One field, no extension | Makes the common case unreadable for a standard client |
| Text only | Zero wire change | Forces every consumer to reimplement what `postprocess.rs` already does |
| **Chosen: both, side by side** | Standard clients see sensible text; coordinate-aware clients read JSON | One non-OpenAI field to document |

**Rationale:** recorded by the maintainer on issue #1073. The implementation follows the extension-field convention the server already had (`reasoning_content`, itself mirroring vLLM) rather than inventing a new one, and documents the field in `docs/responses-api.md` under a new "mlxcel extension fields" section.

**Trade-off:** the field rides the non-streaming chat-completions surface only. Streaming and `/v1/responses` return the rendered text. Both are documented.

### 3.2 Who owns the encoder cache

**Context:** the issue described "a seq2seq worker variant that runs the encoder pass once per request, caches the encoder output, and drives the cross-attention decode loop."

**What was done instead:** that pipeline already existed as `Florence2Model::generate_greedy` from #856. The worker reuses the model-owned pipeline, with a cancellation hook added, rather than building a second encoder-cache layer inside the server.

**Rationale:** this is what makes per-request isolation structural. The encoder output, the seq2seq cache, and the preprocessed pixel tensor are all local to one `run_task_with_cancel` call and dropped when it returns. There is no server-side cache object whose lifetime someone could get wrong later. Reimplementing the cache in the server would have created exactly the shared-state hazard the issue's own "Technical Considerations" warned about.

### 3.3 A built-in chat template for a model that is not a chat model

**Context:** Florence-2 checkpoints ship a plain BART `tokenizer_config.json` with no chat template, no `chat_template.jinja`, and no `chat_template.json`. Verified across all five local conversions.

**Problem:** the generic `User:\n\nAssistant: ` fallback would prepend `User:` to the task prompt, and `parse_task_prompt` requires the string to begin with a task marker. Every request would have been rejected.

**Chosen:** a built-in template keyed on `model_type == "florence2"`, joining the precedent set by Jina VLM. It emits the messages' text verbatim, string content directly and typed content lists as their `text` parts, with no role prefix and no generation prompt. Image parts travel out of band as pixels; Florence-2 has no image placeholder token to render.

**Trade-off:** multi-message requests concatenate, so a conversation with more than the single task-carrying user message is rejected downstream by the task parser with a message listing the valid markers. Acceptable for a model with no conversational semantics, and documented in the template's own comment. An operator `--chat-template` override still takes precedence, which is the existing resolution order.

### 3.4 One delta chunk for streaming

Post-processing needs the complete decode: `<loc_*>` tokens only become pixel coordinates once the whole answer is parsed against the original image size. Streaming the raw token sequence would emit text that does not match the non-streaming `content` for the same request. The worker therefore sends the whole rendered answer as a single `delta.content` chunk followed by the `finish_reason` chunk, and says so in the docs.

---

## 4. Implementation Details

### 4.1 Architecture change

```
[Before]
start_server --> get_model_type == Florence2VLM --> anyhow::bail!

[After]
start_server --> is_florence2 flag (gates the text-only warmup only)
     |
     v
spawn_model_worker_with_batch_config / spawn_legacy_model_worker
     |
     +-- LoadedModel::DiffusionGemma --> diffusion worker loop --> return
     +-- LoadedModel::Llada2Moe      --> llada2 worker loop    --> return
     +-- LoadedModel::Florence2VLM   --> florence2 worker loop --> return
     |
     v
BatchScheduler (decoder-only families only)
```

The branch sits after `LoadedModel` construction and before any scheduler setup, in both MLX spawn paths. Tensor-parallel and pipeline-parallel requests route through the same `spawn_model_worker_with_batch_config`, so they hit the same branch. The XLA worker never builds a `LoadedModel` at all; it loads through `XlaBatchEngine`, which cannot load this architecture and fails at load with a named error rather than serving anything.

### 4.2 Key code changes

**`src/server/florence2_worker.rs` (new, 419 lines including the review follow-up)**

One request end to end: cancellation check, media rejection, image-cardinality check, `parse_task_prompt`, `validate_task_input`, bounded image decode, `run_task_with_cancel`, then `render_task_result` into `content` and `structured_task_json` into `florence2_result`. Every failure path sends a single `GenerateEvent::Error` and returns, so one bad request never tears down the worker.

**`src/models/florence2/render.rs` (new)**

`render_task_result` (moved from the CLI, unchanged in behavior) and `structured_task_json`. The JSON key names mirror upstream `Florence2Processor.post_process_generation` (`bboxes`, `quad_boxes`, `polygons`, `labels`, `bboxes_labels`, `polygons_labels`) so code written against the HuggingFace or mlx-vlm dict shape ports directly.

**`src/models/florence2/{model,runtime}.rs`**

`generate_greedy_with_cancel` and `run_task_with_cancel` add an `Option<&AtomicBool>` polled once per decode step. The existing entry points delegate with `None`, so the CLI path is unchanged. `Florence2RunOutput` gains `prompt_tokens` for the usage block.

**Review follow-up, `cf4faa86`**

```rust
// Serving is serial, so a request can sit in the channel while its client
// goes away. The per-step poll inside `generate_greedy_with_cancel` only
// starts after the encoder pass, which is the expensive half (DaViT tower
// plus the bidirectional BART encoder over the fused sequence), so an
// already-abandoned request would still pay for it. Drop it here instead.
if cancelled.load(Ordering::Relaxed) {
    let _ = response_tx.send(GenerateEvent::Error(
        FLORENCE2_CANCELLED_BEFORE_START_MSG.to_string(),
    ));
    return;
}
```

The same commit extracted `reject_media`, `reject_image_count`, and `florence2_finish_reason` as pure functions with tests, matching the `reject_audio_video` / `diffusion_finish_reason_str` shape the sibling single-stream worker already uses. Before that, the two rejection-message constants were `pub(crate)` with no test referencing them, and the "length" versus "stop" choice was untested.

---

## 5. Learning Points

### 5.1 Serving a seq2seq model on a decoder-only engine

**Concept:** a decoder-only serving engine assumes generation state is a growing KV cache over one token sequence. A seq2seq model has two kinds of state with different lifetimes: the encoder output, computed once and constant for the request, and the decoder's self-attention cache, which grows per step. Cross-attention K/V is projected once from the encoder output and then pinned.

**Application here:** rather than teaching the scheduler about a second cache kind, the family declares `supports_batching() == false` and gets its own loop. That is the same move DiffusionGemma and LLaDA-2 made for a different reason, so the tree now has three single-stream workers sharing one channel protocol and one set of conventions.

**Where it generalizes:** any model whose generation is model-owned rather than step-owned. The channel protocol (`ModelRequest` in, `Token` / `Done` / `Error` out) is the entire integration surface.

### 5.2 Testing an isolation property so the test can actually fail

**Concept:** "request B after request A equals request B alone" is a weak assertion on its own. It passes trivially if the two requests happen to produce the same answer, or if the model is insensitive to the input under test.

**Application here:** `sequential_requests_reuse_no_encoder_state` adds a second half. It decodes one step against request A's encoder output and against request B's, and asserts the logits differ by more than 1e-6. That converts the first assertion from "these agreed" into "these agreed even though a leak would have made them disagree." The test names this in its own failure message, so a future reader who breaks the distinguishability half is told why the isolation half no longer means anything.

### 5.3 Byte-identity as a structural property

**Concept:** an acceptance criterion of the form "the HTTP answer equals the CLI answer" is only checkable if there is one implementation of the answer.

**Application here:** moving the renderer into the library made the criterion true by construction instead of true by inspection. The alternative, two renderers kept in sync by discipline, degrades on the first change to either.

---

## 6. Further Learning

### Key terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `seq2seq` | Encoder-decoder generation with cross-attention against a fixed encoder output | The architectural reason the server refused this family for two releases |
| `Florence2SeqCache` | The dual cache holding one-shot cross-attention K/V and the growing decoder self-attention cache | Per-request state whose lifetime is the isolation guarantee |
| `ImageInputLimits` | Payload, dimension, and decode-allocation caps applied before image decode | The decompression-bomb boundary #855 handed forward |
| `skip_serializing_if` | serde attribute omitting a `None` field entirely | What keeps `florence2_result` from changing any other family's wire shape |
| `<loc_N>` | Florence-2's 1000-bin coordinate tokens, marked special in `tokenizer.json` | Both the output format the parser consumes and the injection vector the validator blocks |

### Related technologies

- **Florence-2** (Microsoft): task-prompted vision foundation model. https://huggingface.co/microsoft/Florence-2-base-ft
- **mlx-vlm** `processing_florence2.py`: the post-processing dict shape `structured_task_json` mirrors. https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py

### Related PRs and issues

- Issue #1073: this work.
- Issue #856 / PR #1071: added the startup refusal this PR removes, and the CLI pipeline the worker reuses.
- Issue #855: handed forward both security requirements.
- PR #1082: quantized Florence-2 load path; its checkpoints are served through this path unchanged.
- Issue #217 phase 3 / #546: the DiffusionGemma and LLaDA-2 single-stream workers this one follows.
- Issue #633: dispatch-thread pre-tokenization, whose interaction with `MAX_TASK_INPUT_BYTES` the review corrected in the docs.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 22 |
| Lines added | +1483 |
| Lines deleted | -219 |
| Test lines added | 516 |
| New unit tests | 30 (12 renderer/JSON, 15 worker boundary and mapping, 1 model isolation, 2 chat template) |

### Changes by category

| Category | Count | Summary |
|----------|-------|---------|
| Feature | 1 | Florence-2 served over OpenAI-compatible HTTP |
| Security | 2 | Bounded image decode and task-input validation at the request boundary |
| Code quality | 3 | Renderer moved into the library; mapping guards extracted and tested; exhaustive match restored |
| Performance | 1 | Cancelled requests dropped before the encoder pass |
| Documentation | 4 | `supported-models.md`, `responses-api.md` extension-field section, `MAX_TASK_INPUT_BYTES` correction, stale tensor-parallel comment |

### Related commits

| Hash | Type | Message |
|------|------|---------|
| `fbdc37db` | feat | serve Florence-2 through a seq2seq worker loop |
| `cf4faa86` | fix | harden the Florence-2 seq2seq worker request boundary |

---

## 8. Follow-up Actions

### Recorded, not fixed here

These four are properties of the single-stream worker class, not of this PR. DiffusionGemma and LLaDA-2 have the first, second, and fourth identically, and fixing them for one family only would leave the tree inconsistent. They belong in one issue that covers all three workers.

- **`--max-queue-depth` is not honored.** `AppState::can_accept_request()` reads `batch_metrics.queue_depth()`, which only the `BatchScheduler` updates. Requests queue on an unbounded `mpsc` channel carrying decoded image payloads, so admission control does not apply. The legacy worker passes `usize::MAX` for this explicitly, so the behavior is at least deliberate there.
- **`response_format` is accepted and ignored.** `options.structured` is consumed only in `batch/scheduler.rs`. Documented for Florence-2 in `supported-models.md` as part of this review; the same statement is missing for the diffusion workers.
- **Usage `prompt_tokens` excludes image feature tokens.** `Florence2RunOutput.prompt_tokens` is the encoder's text token count. The real fused encoder sequence carries 577 projected image tokens ahead of it on `base-ft`, so reported prompt usage understates the prefill by roughly that much. Documented on the field. Reporting the fused length would mean plumbing it out of `Florence2Model::encode`.
- **Declared versus resolved image cardinality is not cross-checked.** The image resolver drops an unresolvable `image_url` tolerantly, so a request declaring two images where one fails resolution is accepted as a one-image request. `MediaRequestMetadata` retains both counts, and its own doc records that the MLX and diffusion workers ignore it while XLA validates it. Florence-2 matches the documented convention.

### Monitoring after deployment

- Worker log line `Florence-2 seq2seq worker ready` confirms the branch was taken; its absence with a Florence-2 checkpoint loaded means the model reached a scheduler.
- Rate of `Florence-2 task prompt:` and `Florence-2 requires exactly one image` errors indicates clients sending conversational rather than task-marker requests.
- Wall time per request is dominated by the encoder pass, so queue latency under concurrency grows linearly with depth by design.

### Future improvements

- Batched admission for the seq2seq path, which needs a measurement of the encoder-versus-decode cost split first, per the repo's performance-issue completion criteria.
- Incremental streaming would require a post-processor that can parse a `<loc_*>` prefix, which is not obviously worth it for answers this short.
- `/v1/responses` could carry `florence2_result` if a consumer asks for it.

---

## Appendix

### A. Test results

```
cargo clippy --profile test-fast --features metal,accelerate --lib --tests   clean
cargo fmt --check                                                             clean
cargo test --profile test-fast --features metal,accelerate --lib florence2    186 passed
cargo test --profile test-fast --features metal,accelerate --lib server::     1655 passed, 8 ignored
```

### B. Real-checkpoint validation

Run against `Florence-2-base-ft-bf16` and `Florence-2-base-ft-4bit` with COCO `val2017/000000039769.jpg` (640x480). HTTP answers for `<CAPTION>`, `<OD>`, and `<CAPTION_TO_PHRASE_GROUNDING>` are byte-identical to the CLI answers from the same binaries. Five rejection paths returned HTTP 400 with named errors and the worker kept serving afterward: a 20000x20000 PNG at 380 KB on the wire rejected before decode, a malformed region input, a 2799-byte oversized input, an angle-bracket smuggling attempt, and an unknown task marker. Streaming and repeat-request isolation were verified in the same session. Full transcripts are in the PR body.

### C. References

- `docs/supported-models.md`, Florence-2 entry: serving semantics and the documented limitations.
- `docs/responses-api.md`, "mlxcel extension fields": the `florence2_result` wire contract.
