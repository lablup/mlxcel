# Technical Report: PR #1740 - feat(gemma4_unified): accept video and audio in the same prompt

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation plus review follow-up cycle
**Status**: Completed (validated on Apple Silicon / Metal against `models/gemma-4-12b-it-4bit`; the `--workspace` gate was not run locally because the machine is shared, and the narrow scopes plus CI cover it)
**Languages**: Rust, Markdown
**Risk Level**: Medium (one family gains a capability, a refusal every generation route shared moves to the HTTP boundary, and two non-generating fronts gain a media gate they never had)

---

## Executive Summary

`Gemma4UnifiedModel::merge_multimodal` has always taken images, video frames, audio features and an audio mask in one call. Neither entry point would hand it all four: the CLI errored on `--video` with `--audio`, and the server refused twice, once in `prepare_chat_request_with_cache` and again in the worker. A clip with its soundtrack is the natural input for an encoder-free unified checkpoint, so the refusal is lifted for `gemma4_unified` and kept, byte for byte, for every other family.

The interesting part is not the lifting, it is where the check had to end up. Request preparation renders a template and never sees the loaded model, so it cannot separate a checkpoint that merges both modalities from one that consumes each alone. Moving the check into `media_capability_rejection` at the HTTP boundary put it in front of a fetch, and that immediately exposed two fronts that render a chat body without ever calling the boundary: the disaggregated router and the three prompt-inspection routes. The third commit closes both. Along the way the audio decode moved ahead of the video decode, because a few kilobytes of malformed audio should not buy a full ffmpeg decode of every clip plus a patch projection over every sampled frame.

The cleanest evidence that the two expansions coexist is arithmetic: the combined prompt is 637 tokens, which is exactly 537 (video only) plus 95 for the audio run plus the 5 extra text tokens the longer prompt contributes.

---

## 1. Where the refusal had to move, and why

### 1.1 `prepare_chat_request_with_cache` cannot make this call

The old server-side refusal lived in `src/server/chat_request.rs`, as an unconditional `anyhow::bail!("Combined video and audio inputs are not supported")` fired the moment `declared_audio > 0 && declared_videos > 0`. That function renders a chat template and resolves media parts. It never sees the loaded model, so it has no way to distinguish an encoder-free unified checkpoint whose `merge_multimodal` scatters both modalities into one token stream from a family that takes video alone and audio alone with no merge path for the pair. Teaching it the difference would mean threading the loaded model into request preparation, which is the wrong direction: the answer is a property of the checkpoint, and the server already computes that once at startup.

The check now lives in `media_capability_rejection` (`src/server/media.rs`), which already reads the `ModelMediaSupport` flags detected by `detect_model_media_support` and runs at the HTTP boundary, before any byte of a referenced image, audio or video payload is fetched. `prepare_chat_request_with_cache` keeps only a comment where the bail was, naming the new owner.

### 1.2 `ModelMediaSupport::video_with_audio` is a flag, not a conjunction

`src/server/state.rs` gains one field:

```rust
pub struct ModelMediaSupport {
    pub image: bool,
    pub audio: bool,
    pub video: bool,
    pub video_with_audio: bool,
}
```

`video_with_audio` is deliberately not `audio && video`. That conjunction is wrong for Gemma 4 VL, Kimi-VL, Inkling and Qwen-VL, all of which consume each modality on its own and none of which has a merge path for both. `detect_model_media_support` sets the flag with `matches!(model_type, ModelType::Gemma4Unified)`, and `LoadedModel::supports_video_with_audio` in `src/loaded_model_capabilities.rs` is the worker-side sibling with the same single arm. The five `server::startup` detection tests assert the negative for Gemma 4 VL, Kimi-VL 2.5, Inkling, Qwen3.5 VL and the missing-config fallback, and the positive only for `gemma4_unified`.

### 1.3 Ordering inside the boundary check

The combination arm is placed last, after the three per-modality arms:

```rust
if !support.image && !request.image_urls().is_empty() { return refuse("image"); }
if !support.audio && !request.audio_inputs().is_empty() { return refuse("audio"); }
if !support.video && !request.video_urls().is_empty() { return refuse("video"); }
if !support.video_with_audio && !request.video_urls().is_empty() && !request.audio_inputs().is_empty() { /* 400 invalid_request_error */ }
```

The order is load-bearing. A text-only checkpoint sent video and audio has to hear `audio input is not supported`, which names a capability it will never have, rather than a combination refusal it could not act on either. `media_capability_rejection_reports_the_missing_modality_before_the_combination` pins that.

The refusal keeps its exact string, now the shared constant `COMBINED_VIDEO_AUDIO_REFUSAL`, its `invalid_request_error` type and its 400 status, so `/v1/chat/completions` and `/v1/responses` are unchanged by the move.

### 1.4 The Anthropic envelope: a client error, not a capability gap

`routes::anthropic` rendered every media-capability rejection as `501 not_supported_error`. That is right for a per-modality refusal and wrong for the combination: on such a checkpoint each modality is accepted on its own and only the pair has no merge path, so the client fixes it by dropping one part. `media_rejection_response` keys off `is_combined_video_audio_rejection` (status plus the shared constant, so the route carries no copy of the string) and keeps the boundary's 400 for that one case. The arm is unreachable through `/v1/messages` today because `AnthropicContentBlock` has no video or audio variant; it is written anyway so the status is already right if that schema gains one, and the two new tests exercise the switch directly for the same reason.

---

## 2. Building the combined prompt out of the single-modality helpers

### 2.1 Per-modality helpers on both surfaces

Both the worker and the CLI were factored into helpers before the combined builder was written, and the combined builder calls exactly those:

| Step | Server (`src/server/model_worker.rs`) | CLI (`src/commands/generate_vlm.rs`) |
|---|---|---|
| Companion images | `gemma4_unified_server_images` | `gemma4_unified_cli_images` |
| Video decode | `gemma4_unified_server_decode_videos` | `gemma4_unified_cli_decode_videos` |
| Frame patchify plus placeholder expansion | `gemma4_unified_server_video_frames` | `gemma4_unified_cli_video_frames` |
| Audio decode and chunking | `gemma4_unified_server_audio_features` | `gemma4_unified_cli_audio` |
| Audio run expansion | `gemma4_unified_server_expand_audio_run` | (inside `gemma4_unified_cli_audio`) |

Reuse rather than duplication is the whole argument for byte-identity on the single-modality paths. `prepare_gemma4_unified_audio_embeddings`, `prepare_gemma4_unified_video_embeddings` and `prepare_gemma4_unified_video_and_audio_embeddings` now run the same code over the same inputs, so a video-only or audio-only prompt cannot drift from what it produced before this path existed. The validation in section 4 confirms that empirically, but the structure is what makes it true rather than lucky.

`get_input_embeddings_with_video_and_audio` in `src/vision/gemma4_unified.rs` is the fourth and widest wrapper, and the only one that reaches every argument of `merge_multimodal`. It adds no logic of its own.

### 2.2 Images must expand before the video frames

The order is not cosmetic. `expand_gemma4_image_tokens` counts a placeholder as `image_token_id` **or** `boi_token_id`, and `expand_gemma4_unified_video_tokens` frames every emitted frame with its own `boi_token_id`. Expanding images second would therefore count each emitted video frame as an image placeholder, and the prompt would either fail the image cardinality check with a count the caller cannot explain (`Gemma4 prompt has N image placeholder(s) but M image(s) were provided`) or, when the counts happen to line up, expand against the wrong runs. Both combined builders carry that reason in their doc comments rather than merely stating the order.

### 2.3 The placeholder ordering constraint, and the audio split

The audio run must expand **after** the image and video runs, so the three placeholder streams land in the order `merge_multimodal` scatters them: `merge_llava` on `image_token_id`, then `merge_llava` on `video_token_id` against the running embeddings, then a `masked_scatter` on `audio_token_id`. The video runs splice in after BOS and the audio run lands before the last `<end_of_turn>` (issue #437), so the two address disjoint ids at disjoint insertion points.

That constraint collides with a cheaper one: validation should run first. `gemma4_unified_server_audio` was therefore split in two. `gemma4_unified_server_audio_features` decodes the clip, enforces `require_single_server_audio_clip`, and chunks the waveform into `audio_samples_per_token` frames, touching the prompt not at all. `gemma4_unified_server_expand_audio_run` does nothing but mutate the prompt. The combined builder runs them at opposite ends:

```rust
let audio_input = gemma4_unified_server_audio_features(unified, audio_data)?;   // validation, first
let decoded_videos = gemma4_unified_server_decode_videos(videos)?;
let processed_images = gemma4_unified_server_images(unified, prompt_tokens, images, image_soft_tokens)?;
let video_frames = gemma4_unified_server_video_frames(unified, prompt_tokens, &decoded_videos)?;
gemma4_unified_server_expand_audio_run(unified, prompt_tokens, audio_input.num_frames, end_of_turn_token_id);  // last
```

Both failures the decode can raise, a clip count other than one and a waveform the WAV reader rejects, are decided by bytes the client already sent. Ordering the decode last let a few kilobytes of malformed audio buy a full ffmpeg decode of every clip plus a patch projection over every sampled frame before the request was refused. The audio-only path already validated first; this stops the combined path from being the cheaper way to buy that work.

One asymmetry is deliberate. The audio-only path warns and drops the audio when `embed_audio` is `None`; the combined path refuses with `MISSING_AUDIO_EMBEDDER_REFUSAL`. By the time that check could fire, the video runs are already in the prompt, so answering from video alone would answer a question the caller did not ask, and nothing in a 200 would tell them half their input vanished. Neither `ModelMediaSupport` nor `LoadedModel::supports_video_with_audio` can catch it: both key on the model type, not on which weights actually loaded, so the code holding the model is the first place that knows.

---

## 3. The pre-fetch media gate

Moving the refusal to the boundary made a latent problem visible: `prepare_chat_request_with_cache` downloads every `image_url` and `input_audio` payload and opens every `video_url` as part of rendering. Refusing after it means fetching a client-named URL first. Two fronts did exactly that, and neither ever reaches a model worker, so the worker backstop is no help.

**`router_front`.** `route_chat` refused media only after preparation returned, by asking `has_declared_media` of the resolver's output. The disaggregated router is text-only for every checkpoint behind it, so that download could never be used for anything. `request_declares_media` now asks the request itself, before the render:

```rust
fn request_declares_media(request: &ChatCompletionRequest) -> bool {
    !request.image_urls().is_empty()
        || !request.audio_inputs().is_empty()
        || !request.video_urls().is_empty()
}
```

The post-resolution `has_declared_media` check is kept as a backstop, so a future translation step that synthesized a media part after the request was inspected is still refused rather than silently dropped. The new test drives the predicate with an `image_url` at `http://169.254.169.254/`, an `input_audio` part and a `video_url`, plus a text-only negative.

**`prompt_inspection`.** `/apply-template` and both `chat/completions/input_tokens` paths never refused at all: they reported a token count for media the checkpoint has no tower for, having fetched it first. `render_chat_prompt` now runs `media_capability_rejection` ahead of `validate_chat_tool_inputs`, the order `/v1/chat/completions` uses, so the three inspection routes answer the way the generating route answers. Three tests cover it: an image part against a text-only stub refused on all three paths with `501 not_supported_error` and no `prompt` or `input_tokens` field in the body, the video plus audio combination still refused here, and a text-only body unaffected.

---

## 4. Validation

All runs are against `models/gemma-4-12b-it-4bit` on Apple GPU (Metal), greedy (`--temp 0`), compared against the same commands on the pre-change binary. The second commit re-ran the whole set against a binary rebuilt from that tree and reproduced every number.

### 4.1 Token accounting is exactly additive

| Run | Expansion line | Total prompt tokens |
|---|---|---|
| `--video clip.mp4` | `expanded 1 video(s) into 8 frame slot(s)` | 537 |
| `--audio speech.wav` | `expanded audio into 93 soft tokens` | 114 |
| both | `expanded 1 video(s) into 8 frame slot(s) and audio into 93 soft tokens` | 637 |

637 = 537 + 95 + 5: the video-only prompt, plus 95 for the audio run (93 soft tokens framed by BOA and EOA), plus the 5 extra text tokens the longer prompt (`Describe the video and transcribe what is said.`) contributes. Nothing is lost and nothing is double-counted, which is the cleanest evidence available that the two expansions do not clobber each other.

The single-modality runs are byte-identical to the pre-change binary, expansion line and generated text both. The video-only run generates `The video shows a black screen with a white text "The video is black" appearing in the center.`; the audio-only run generates `The red square moves from the left side of the frame to the right side.`

The combined run, re-run at `-n 160`, stops on its own after 75 tokens with `The video shows a person in a black shirt and pants standing in front of a white wall. They are holding a white object in their hands and are moving it from left to right. The person is speaking in a clear and articulate voice.` followed by `The transcript of the video is as follows: "The frame moves from the left side of the frame to the right side."`, reproducing the soundtrack almost exactly. Before this change the same command exited 1 with `Error: Combined --video and --audio inputs are not supported yet`.

### 4.2 Server

`mlxcel-server --port 19349` with `MLXCEL_VIDEO_DIR_ALLOWLIST` pointed at the fixture directory. `POST /v1/chat/completions` carrying a `video_url` part (`file://` absolute path), an `input_audio` part (base64 wav) and the same text prompt, `max_tokens` 64, `temperature` 0, returned HTTP 200 with `prompt_tokens` 637, matching the CLI exactly, and content `The video shows a person's hand moving from the left side of the frame to the right side.` followed by an attempted transcript. Single-modality controls on the same server returned 200 at 114 (audio only, exact transcript) and 537 (video only) prompt tokens, matching their CLI counterparts token for token.

### 4.3 Negative control

`generate -m models/gemma-4-e4b-it-4bit ... --video clip.mp4 --audio speech.wav` still exits 1 with `Combined --video and --audio inputs are not supported yet`. That checkpoint is `model_type: gemma4` and loads a 12-layer Conformer audio encoder, so it genuinely consumes video alone and audio alone. It is the strongest available control for the flag being narrower than `audio && video`, because it is a real checkpoint on which the conjunction would have been true.

### 4.4 Second pass: the whole set re-run on the branch head

The pass above was taken on a release binary before `06141567` landed. After that commit (the pre-fetch media gate and the audio-decode-before-video reordering described in section 3) the whole set was re-run against `target/test-fast/mlxcel` and `target/test-fast/mlxcel-server`, built under `[profile.test-fast]`, which inherits release and keeps `opt-level = 3`. Same checkpoint, same Apple GPU (Metal), same greedy decode.

The fixtures had to be regenerated, because the machine rebooted and `/tmp` was cleared: a 4.25 s 320x240 clip of a red square moving left to right, and a macOS `say` wav resampled to 16 kHz mono, 55550 samples, 3.5 s, which chunks to 87 soft tokens where the first-pass wav gave 93. That turns the second pass into a check of the arithmetic rather than a repeat of the same totals.

| Run | Expansion line | CLI total prompt tokens | Server `prompt_tokens` |
|---|---|---|---|
| `--video clip.mp4` | `expanded 1 video(s) into 8 frame slot(s)` | 537 | 537 |
| `--audio speech.wav` | `expanded audio into 87 soft tokens` | 108 | 108 |
| both | `expanded 1 video(s) into 8 frame slot(s) and audio into 87 soft tokens` | 631 | 631 |

631 = 537 + 89 + 5: the same video-only prompt, the audio run at 87 soft tokens framed by BOA and EOA, and the same 5 extra text tokens, against 637 = 537 + 95 + 5 in the first pass. The additive accounting survives a fixture whose audio length changed, which is the property it was there to demonstrate.

The single-modality generated texts are byte-identical to the ones recorded in 4.1, on a binary three commits further on: video-only still gives `The video shows a black screen with a white text "The video is black" appearing in the center.` and audio-only still gives `The red square moves from the left side of the frame to the right side.` The combined CLI run stopped on its own inside `-n 160`, and the negative control is unchanged: `models/gemma-4-e4b-it-4bit` exits 1 with `Error: Combined --video and --audio inputs are not supported yet`.

Server on port 19349, `MLXCEL_VIDEO_DIR_ALLOWLIST` pointed at the fixture directory, returned HTTP 200 on all three requests at the prompt-token counts in the table, each matching its CLI count token for token. At `max_tokens` 200 the combined request finished with `finish_reason` `stop` after 70 completion tokens, content `The video shows a person's hand moving a small, white, rectangular object across a dark surface. The object is being moved in a repetitive, back-and-forth motion. The background is dark and out of focus.` then `The transcript of the video is as follows:` then `"I'm from the left side to the right side."`

The same combined body posted to `/apply-template` and to `/v1/chat/completions/input_tokens` returned 200. That is the positive side of the gate added in `06141567`: those routes now refuse a body the generating route refuses, and they still admit the one family that can consume the combination rather than sweeping it up with everyone else.

**One divergence, stated rather than hidden.** The combined free-text reply differs between the CLI and the server, and between the two passes, on token-identical prompts under greedy decoding. The first-pass record in 4.1 and 4.2 already shows the same CLI-versus-server divergence on this prompt. The evidence that the two expansions compose is therefore the token accounting and the byte-identical single-modality replies, not the wording of the combined reply.

### 4.5 Test coverage added

| Module | What it pins |
|---|---|
| `vision::merge` | 1 test: the two-op composition `merge_multimodal` runs (`merge_llava` on `video_token_id`, then `masked_scatter` on `audio_token_id`) with a distinct constant per modality, asserting each run receives its own features and the text rows survive |
| `multimodal::vlm_runtime` | 3 tests: the video splice (after BOS) and the audio splice (before the last `<end_of_turn>`) compose in either order, and a template-rendered prompt expands each placeholder id independently |
| `server::media` | 3 tests: the refusal for a family that takes each modality alone, admission under `video_with_audio`, single-modality requests left alone, and the per-modality refusal winning over the combination refusal on a text-only checkpoint |
| `server::startup` | 5 detection tests gain `video_with_audio` assertions, positive for `gemma4_unified` and negative for Gemma 4 VL, Kimi-VL 2.5, Inkling, Qwen3.5 VL and the missing-config fallback |
| `server::chat_request` | the old rejection test is rewritten to assert preparation no longer emits the combined refusal |
| `server::router_front` | 1 test: the pre-fetch predicate refuses an image, audio or video part from the request itself, and passes text |
| `server::routes::prompt_inspection` | 3 tests: a media part refused rather than fetched and counted on all three routes, the combination still refused, a text-only body unaffected |
| `server::routes::anthropic` | 2 tests: the combination renders as `400 invalid_request_error`, a per-modality refusal keeps `501 not_supported_error` |

### 4.6 Gates

`cargo fmt --all -- --check` passes and `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` exits 0 with no diagnostics. Narrow scopes under `--profile test-fast --features metal,accelerate` are all green: `server::chat_request` 104 passed, `server::media` 75, `server::startup` 74, `server::router_front` 28, `server::routes::prompt_inspection` 16, `server::routes::anthropic` 17, `multimodal::vlm_runtime` 53, `vision::merge` 5, `vision::gemma4_unified` 19.

---

## 5. What was not verified

- **The full `--workspace` gate was not run locally.** The machine is shared with seven other build agents and the target directory is in use. The scopes in 4.6 cover every module this change touches, and the merge gate runs the rest.
- **The audio fixture is synthetic.** No speech sample exists under `tests/fixtures`, so it is macOS `say` TTS at 16 kHz speaking `The red square moves from the left side of the frame to the right side.`, paired with a 4.25 s generated clip of a red square moving left to right (8 frames at the default 2.0 fps). The model reports that clip's frames as black in every video run, including the pre-change baseline, so the visual descriptions above carry almost no information about visual correctness. What they do establish is that the prompt assembles, the token counts are additive, and the audio reaches the backbone.
- **Attention dilution was measured, not fixed.** At the default 2.0 fps the 528 video tokens dominate a short prompt, and a probe phrased `What sentence is spoken in the audio? Reply with the sentence only.` answered `The audio is not provided.` The same probe at `fps` 0.25 (382 prompt tokens) answered `The red square moves from the left side of the frame to the right side.`, the transcript verbatim. Read together with the additive token accounting and the byte-identical single-modality runs, that is attention dilution against a fixture whose frames carry no information, not a plumbing defect. It is reported as measured; no claim is made that the default fps is the right one for a mixed prompt.
- **The combined reply's wording is not reproducible, and is not the evidence.** On token-identical prompts under greedy decoding the combined free-text reply differs between the CLI and the server, and between the two validation passes (4.2, 4.4). Nothing here investigates that, and nothing here rests on it: what the two passes pin is the token accounting and the byte-identical single-modality replies.
- **One audio clip per request, unchanged.** `require_single_server_audio_clip` still refuses any other count, and the combined path validates it first.
- **The Anthropic combination arm is unreachable today.** `AnthropicContentBlock` has no video or audio variant, so the arm is exercised by unit test only.
- **No other family was widened.** `video_with_audio` has exactly one `true` arm, in two places that mirror each other, and the negative control is a real checkpoint rather than a synthetic config.

---

## 6. Files

| File | Role |
|---|---|
| `src/vision/gemma4_unified.rs` | `get_input_embeddings_with_video_and_audio`, the widest wrapper over `merge_multimodal`; `MISSING_AUDIO_EMBEDDER_REFUSAL` |
| `src/server/state.rs` | `ModelMediaSupport::video_with_audio` |
| `src/server/startup.rs` | `detect_model_media_support` sets the flag for `ModelType::Gemma4Unified` only |
| `src/loaded_model_capabilities.rs` | `LoadedModel::supports_video_with_audio`, the worker-side sibling |
| `src/server/media.rs` | `COMBINED_VIDEO_AUDIO_REFUSAL`, the combination arm of `media_capability_rejection`, `is_combined_video_audio_rejection` |
| `src/server/chat_request.rs` | the preparation-level bail removed, with the new owner named in its place |
| `src/server/model_worker.rs` | per-modality helpers, the audio decode / expand split, `prepare_gemma4_unified_video_and_audio_embeddings`, the backstop arm |
| `src/commands/generate_vlm.rs` | CLI helpers and `compute_gemma4_unified_video_and_audio_embeddings` |
| `src/server/router_front.rs` | `request_declares_media`, refusing before the render fetches |
| `src/server/routes/prompt_inspection.rs` | `media_capability_rejection` ahead of the tool guard in `render_chat_prompt` |
| `src/server/routes/anthropic.rs` | `media_rejection_response`, the 400 / 501 switch |
| `src/multimodal/vlm_runtime.rs` | `VlmPreparationSummary::Gemma4VideoAudio` |
| `docs/supported-models.md`, `src/main.rs` | the family entry and the `--video` / `--audio` help text |
