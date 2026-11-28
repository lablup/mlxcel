# Technical Report: PR #1759 - feat(server): send sampled video frames as ordered images

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation plus review and security follow-up cycle
**Status**: Completed (validated on Apple Silicon / Metal against `models/gemma-3-4b-it-4bit`, `models/lfm2-vl-450m-4bit`, `models/qwen2.5-vl-3b-instruct-4bit` and `models/gemma-4-e4b-it-4bit`; the `--workspace` gate was not run locally because the machine is shared, and the narrow scopes plus CI cover it)
**Languages**: Rust, Markdown
**Risk Level**: Medium (a refusal that stood for every family but a handful is lifted, one media-support flag splits in two, and a request is now rewritten in place before it renders)

---

## Executive Summary

Video input reached exactly the families that had built a temporal path for it. Everything else, which is most of the vision-language catalogue, answered a clip with a hard refusal: `--video input is currently only supported by ...` on the CLI, HTTP 400 `video_url content blocks are not supported by model '...'` on the server. The odd part is that those same checkpoints read a sequence of still images without complaint. The capability was there; only the plumbing said no.

This change spends that observation. A checkpoint with a vision tower and no native video path now has its clip decoded at the requested fps, evenly subsampled to at most `--video-max-frames` frames with the first and the last always kept, PNG-encoded, and spliced back into the request as that many ordered `image_url` parts behind one sentence saying they are a video. Nothing downstream is taught anything new. The template emits one image placeholder per frame, the per-request image budget counts them, the soft-token accounting counts them, and the prompt-cache multimodal digest hashes their bytes, because by the time any of that runs the frames are images and not a special case.

Where the rewrite happens is the whole design. It sits on one seam, after `media_capability_rejection` at the HTTP boundary and before `prepare_chat_request_with_cache` renders. Earlier than that and it would run before the capability gate had decided whether the checkpoint may see a clip at all; later and the template would already have been rendered against a `video_url` part the model has no placeholder for.

The cleanest evidence that the substitution is faithful is a control rather than a description: at four frames, the same question asked through `video_url` (1112 prompt tokens) and asked with those same four frames handed over as four `image_url` parts (1097) returns the identical answer, and the 15-token gap is the lead sentence and nothing else.

---

## 1. What "native" had to mean, and where the answer lives

### 1.1 One list, read by two fronts

Before this change the set of video-capable families was written down twice. `server::startup::detect_model_media_support` carried an eleven-arm `matches!` to set `ModelMediaSupport { video }`, and `commands/generate_vlm::compute_vlm_embeddings` dispatched on the loaded model with its own arms, falling through to an error message that listed the families by name in prose. Two lists that must agree, in different files, with no mechanism keeping them together: the failure mode is a family that the server admits and the CLI refuses, or the reverse.

The predicate now lives once, in `src/models/detection.rs`, next to `get_model_type`:

```rust
pub fn model_type_has_native_video(model_type: ModelType) -> bool
```

`detect_model_media_support` calls it to compute `video_native`, and `commands::generate` calls it to decide whether `--video` needs the fallback. A family that grows a temporal path is added in one place and both fronts follow. The comment above it names the two dispatch sites that must be updated alongside it, which is the discovery mechanism this repository uses for shared functions.

`model_type_is_vision_capable` is exported next to it for the same reason: the binary crate's `--video` handling needs the `config.json`-only VLM predicate the model registry already computes, and re-deriving it in the CLI would be a third list.

### 1.2 `ModelMediaSupport::video` splits, and keeps its old meaning

`video: bool` answered one question, "may this request carry a `video_url` part", and that question still has one answer. But the two ways of answering it differ in what actually reaches the model, not merely in whether the request is admitted, so the flag becomes two:

```rust
pub video_native: bool,
pub video_frames_fallback: bool,

pub const fn video(&self) -> bool { self.video_native || self.video_frames_fallback }
```

`media_capability_rejection` reads `support.video()` and behaves exactly as before. `expand_video_parts_to_frames` reads `video_frames_fallback` and returns `Ok(0)` untouched for a native family, which is what keeps `prepared.videos` populated for the checkpoints that want it. `/props` reports `video()` because the wire-level answer to "does this server take video" is unchanged; the log line is what distinguishes the two.

The fallback is keyed on `multimodal && !video_native` rather than on a family list, so a VLM landing tomorrow gains it the day it lands, and a family that later grows a real temporal path silently stops using the substitute instead of doing both.

### 1.3 The one exclusion

Muse Glimmer is excluded by name. The CLI already refuses `--video` for it in `validate_muse_glimmer_cli_unsupported_options`, and admitting the clip on the HTTP boundary alone would leave the two fronts disagreeing about one checkpoint, which is the exact failure 1.1 exists to prevent. Both guards lift together when the family is qualified for multi-image prompts.

---

## 2. The rewrite

### 2.1 Positions first, then the decode

`expand_video_parts_to_frames_with_allowlist` collects `(message index, part index, VideoUrl)` for every `video_url` part before it decodes anything, because the decode is async and the splice shifts every index after it. Doing both in one pass would walk a list that is moving underneath the walker.

The decode itself runs under `tokio::task::spawn_blocking`, matching the image path: ffmpeg plus PNG encoding is seconds of CPU on a long clip and has no business on a Tokio worker.

### 2.2 Decode only what survives

The first version decoded the clip at the requested fps and subsampled afterwards, which is the shape the issue described. That is a 48x peak-memory amplification and the security pass was right to refuse it: `load_video_source` answers an fps with up to `FPS_MAX_FRAMES` (768) full-resolution frames held at once, and the fallback keeps 16 of them. At the shipped caps (`MLXCEL_VIDEO_MAX_PIXELS` 4096x4096, `MLXCEL_VIDEO_MAX_DURATION_SEC` 600) that is roughly 38 GB of RGB to send sixteen pictures, and no unusual `video_url.fps` is needed to get there: the default 2.0 fps reaches the 768 ceiling on any clip past about six minutes. The fallback also turns this on for essentially every image-capable VLM rather than the five native families, so the exposure is the whole catalogue rather than a corner.

`load_video_source_frames_fallback` probes the container once, computes the count `target_fps` alone would have sampled so the caller can still report "kept of sampled", and decodes only `min(sampled, max_frames)` frames. The frames it keeps are the same even spread over the clip, because a uniform sample of a uniform sample is one, and 5.4 is the measurement that says so rather than the argument. `subsample_evenly` still runs afterwards and is now almost always the identity; it is kept because `max_frames` is a cap and not a promise, and because the CLI and the tests exercise it directly.

`subsample_evenly` is generic over the element type so the same spacing can be applied to decoded frames or to encoded buffers, but every caller uses it on the cheap one.

The spacing is `round(i * (len - 1) / (max - 1))` for `i in 0..max`, exposed separately as `evenly_spaced_indices` because `uniform_indices` answers the neighbouring question (which frames to decode) rather than this one (which decoded frames to keep). First and last are always kept, and the result is strictly increasing whenever `max_frames <= len` because the step is then at least 1.

PNG rather than JPEG: the frames are re-decoded by the image path a moment later, and a lossy round trip would put artifacts in front of the vision tower that the source clip does not have.

### 2.3 The splice, back to front

`apply_video_frame_expansion` applies the expansions in reverse order so an earlier splice cannot move a later part's index. Each `video_url` part is replaced by a text part carrying `Here is a video as a sequence of N frames in chronological order.` followed by N `image_url` parts holding `data:image/png;base64,...`, in chronological order. Parts that surrounded the clip keep their positions relative to it, which is what makes a request mixing pictures and a clip come out in the order the caller wrote it.

The function is split from the I/O half deliberately: everything above it is allowlist resolution, ffmpeg and encoding, and everything in it is the ordering contract. Tests drive this half directly with synthetic bytes, so the contract is covered on a host with no ffmpeg.

### 2.4 The budget refusal names the frames, and comes before the bill

The frames become ordinary images and spend the ordinary per-request image budget. `validate_image_count` would refuse an over-budget request a moment later, but its message reports an image count the caller never sent, which reads as a server bug rather than as a `--video-max-frames` set too high for the deployment. `video_frame_budget_rejection` refuses first, naming the injected frame count, the caller's own image count and the limit, and pointing at the two flags that resolve it.

Where that check runs matters as much as what it says. The first version ran it once, after every clip in the body had been resolved, decoded and PNG-encoded. Nothing caps the number of `video_url` parts and the JSON body limit is roughly 1.4 GB, so a body of tiny video references bought a full probe and decode per clip before the refusal it was always going to get. Two checks now stand in front of that work: a clip-count guard before the first ffprobe, valid because every clip yields at least one frame image so more clips than `max_images_per_request` can never be served, and the frame check moved inside the loop so a body that goes over budget stops at the clip that broke it. `video_expansion_refuses_more_clips_than_the_image_budget_before_decoding` pins the first one by pointing the URLs at nothing: reaching the resolver would report a different error.

The `info` line reporting a clip as sent sits behind the per-clip refusal, so a refused request does not also leave a log saying the clip that broke the budget was sent.

None of this is the enforcement, only the friendly pre-empt. The frames are real `image_url` parts by the time the media path sees them, so `validate_image_count` and `validate_resolved_image_count` still run per image and the budget cannot be slipped past.

---

## 3. The seams

| Front | Wired | Why |
|---|---|---|
| `routes/chat.rs` | yes | after `media_capability_rejection`, before `validate_chat_tool_inputs` and the render. `chat_completions` is the single entry feeding both the streaming and non-streaming paths, so one call covers both. |
| `routes/responses.rs` | yes | on the translated `ChatCompletionRequest`, on the same seam, before either handler reads it. |
| `routes/prompt_inspection.rs` | yes | these routes exist to answer "what prompt would the generating route build for this body", so they have to run the same substitution. Applied to a clone, since the handlers borrow the request and never generate. |
| `routes/anthropic.rs` | no | `AnthropicContentBlock` has variants text, image, document, tool_use and tool_result, and no video one, so `anthropic_request_to_chat` cannot produce a `VideoUrl` part. Wiring it would add an unreachable call. |
| `router_front.rs` | no | `route_chat` refuses everything `request_declares_media` matches, `video_urls()` included, before rendering, because the disaggregated path is text-only for pool-backed families. There is nothing there for the expansion to run on. |

The issue's implementation plan named the last two. Both are stated in the PR body with their reasons, so a reviewer does not read the absence as an omission.

The worker guard in `model_worker::prepare_request_video_embeddings` is kept and reworded. It is now a backstop rather than the refusal a caller normally meets: an image-capable checkpoint without a native path has its clip rewritten at the HTTP boundary, so its request arrives with no videos at all, and reaching that arm means a route skipped the expansion. That is a wiring bug, and the message says so.

---

## 4. The CLI

`expand_cli_videos_to_frames` in `src/commands/generate.rs` is the same transformation against a different representation: the frames are written to the system temp directory as PNGs, their paths are appended to `args.generation.image`, the lead sentence is prepended to `user_prompt`, and `args.generation.video` is cleared so `compute_vlm_embeddings` never sees a clip on this path.

It calls the same `load_video_source_frames_fallback` through `VideoSource::from_path`. The memory argument is weaker here, since this is one local process rather than a service, but the parity argument is not: with two decoders the CLI and the server were free to disagree about which frames of the same clip the model saw at the same `--fps` and `--video-max-frames`, and an answer that differs between the two fronts is a bug report nobody can reproduce.

Three placement constraints:

- It runs after every validator that reads `--video` (the pipeline-parallel check, `--output-audio`, `--layout-detections`, the Muse Glimmer guard), so none of them changes meaning.
- It runs before the prompt is rendered, so `load_cli_prompt` counts one image content part per frame.
- The `TempFile` RAII guards are bound to a variable that lives until `run_generate_once` returns, which is after the vision tower has read the PNGs. Dropping them earlier would unlink the files under the reader.

It declines, returning `Ok(None)`, for a native family, for a checkpoint with no vision tower, and for Muse Glimmer. The refusal in `generate_vlm.rs` is reworded for the only case that still reaches it: a checkpoint with neither a native path nor a vision tower.

---

## 5. Validation

Release binaries built from this tree (`cargo build --release --features metal,accelerate`, exit 0), on M5 Max / macOS 27.0. Fixture: an 8 s 448x448 25 fps H.264 clip of a 128x128 red square crossing a white background left to right, built with `ffmpeg -f lavfi -i "color=c=white:s=448x448:d=8:r=25" -f lavfi -i "color=c=red:s=128x128:d=8:r=25" -filter_complex "[0:v][1:v]overlay=x='(W-w)*t/8':y=(H-h)/2"`, first and last frames visually confirmed. It is a scratch file and is not committed.

The first attempt at the fixture used `drawbox` with a `t`-dependent `x` and produced eight seconds of blank white, which every model dutifully described as blank. It is worth recording because the run looked like a successful validation: the log line was right, the frame count was right, the token growth was right, and only the answers gave it away. The frames were checked by eye before the numbers below were trusted.

### 5.1 CLI, four runs, all exit 0

| Case | Checkpoint | What ran |
|---|---|---|
| Fallback | `gemma-3-4b-it-4bit`, `--video-max-frames 8` | `model_type=Gemma3VLM has no native video path; sending 8 of 16 sampled frames from <clip> as ordered images`, then `Loaded 8 image(s).` and `Expanded 8 <image> token(s) to 256 tokens each`; the prompt opens with the lead sentence |
| Fallback | `lfm2-vl-450m-4bit`, default cap | `model_type=Lfm2VL ... sending 16 of 16 sampled frames`, `Loaded 16 image(s).`, `LFM2-VL: inserted 16 image block(s) (3136 total image tokens)` |
| Native control | `qwen2.5-vl-3b-instruct-4bit` | no fallback line; `Loaded 1 Qwen-VL video(s) (16 total frames after sampling)`, `1 video block(s) (2048 video tokens)` |
| Native control | `gemma-4-e4b-it-4bit` | no fallback line; `Loaded 1 video(s) (16 total frames after sampling)`, `Gemma4: expanded 1 video(s) into 16 frame slot(s) (1079 total tokens)` |

### 5.2 Server, token accounting

`mlxcel-server -m models/gemma-3-4b-it-4bit --port 19322 --video-max-frames 8`, `MLXCEL_VIDEO_DIR_ALLOWLIST` pointed at the clip's directory, `/v1/chat/completions`, HTTP 200 throughout:

| Request | `usage.prompt_tokens` |
|---|---|
| text only | 19 |
| text plus one still PNG | 279 |
| text plus `video_url` | 2114 |

2114 - 19 = 2095, against 8 x 260 = 2080 for the frames plus 15 for the lead sentence. That is the growth the issue asked for, to the token.

Both log lines appeared, one at startup and one per request: `model_type=Gemma3VLM: no native video path; video_url content blocks will be served as ordered sampled frames` and `model gemma-3-4b-it-4bit has no native video path; sending 8 of 16 sampled frames from <clip> as ordered images`.

`models/gemma-4-e4b-it-4bit` answered the same body natively at 1080 prompt tokens, matching its CLI run's 1079 frame-slot tokens, and logged the fallback line zero times. `GET /props` reports `modalities.video: true` for both.

### 5.3 The faithfulness control

At `--video-max-frames 4` with greedy decoding on `gemma-3-4b-it-4bit`, the same question asked two ways:

- through `video_url`, the fallback doing the work: 1112 prompt tokens
- with those same four frames sent by the caller as four `image_url` parts: 1097 prompt tokens

Identical answer, both times. The 15-token delta is the lead sentence. The rewritten request is the plain multi-image request the model would have received had the caller extracted the frames themselves, which is the claim the change rests on.

### 5.4 The bounded decode changes nothing the model sees

Every run in 5.1 through 5.3 was performed twice: once against the original decode-then-subsample, and once against `load_video_source_frames_fallback`. Not one number moved. The same `sending 8 of 16 sampled frames`, the same 19 / 279 / 2114 prompt tokens, the same 1080 on the native control with the fallback line still absent, and the same 1112 against 1097 with the same answer both times.

That is the interesting part of the security fix. "A uniform sample of a uniform sample is one" is exact arithmetic, but the two paths round twice and once respectively, so it was fair to ask whether the surviving frames shifted by an index. Against this clip and these caps they did not, and the token identity across four checkpoints is what says so.

### 5.5 Test coverage added

- `src/multimodal/video_tests.rs`: `subsample_evenly_keeps_first_and_last`, `subsample_evenly_identity_when_under_cap`, `subsample_evenly_indices_match_formula` (37 sampled, cap 16, indices `[0, 2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29, 31, 34, 36]`), `subsample_evenly_indices_are_strictly_increasing`, `frames_to_png_round_trips_every_frame_in_order`.
- `src/server/chat_request_tests.rs`: `video_part_expands_to_ordered_image_parts_with_lead_text`, `video_expansion_preserves_surrounding_images`, `video_expansion_skipped_for_native_video_model`, `video_expansion_is_a_no_op_without_video_parts`, `video_expansion_changes_mm_digest_when_frames_change`, `video_frame_budget_refusal_names_the_frames`, `video_frames_lead_text_names_the_frame_count`, `video_expansion_refuses_more_clips_than_the_image_budget_before_decoding`, and the ffmpeg-backed `video_part_expands_through_a_real_clip`.
- `src/server/startup_tests.rs`: `media_support_marks_image_only_vlms_as_frames_fallback`, `media_support_keeps_native_video_families_native`, `media_support_gives_the_vit_gemma4_vlm_no_frames_fallback`, `media_support_denies_the_frames_fallback_to_text_only_and_muse_glimmer`.
- `src/commands/generate_tests.rs`: `cli_video_fallback_declines_native_and_text_only_checkpoints`, `cli_video_fallback_appends_frame_images_and_clears_video`.

The two ffmpeg-backed additions are `#[ignore]` by repository convention (#1172) and were run for real under `MLXCEL_TEST_VIDEO=1 ... -- --include-ignored`; both pass.

### 5.6 Gates

`cargo test --profile test-fast --features metal,accelerate` over `--lib multimodal::video` (40 passed), `--lib server::chat_request` (112), `--lib server::startup` (78), `--lib server::media` (75), `--lib server::routes` (394), `--lib server::cli_input` (144), and `--bin mlxcel commands::generate` (85). `commands::generate_tests` lives in the binary crate, not the lib, so `--lib commands::generate_tests` matches zero tests; the `--bin` scope is the correct one.

`cargo clippy --lib --tests` and `--bins` with `-D warnings`, `cargo fmt --all -- --check`, and `cargo check --lib --tests` all clean. GitHub CI on the rebased head is fully green, `cargo-clippy` and `cargo-fmt` included.

---

## 6. What was not verified

- **The full `--workspace` gate was not run locally.** The machine is shared with seven other build agents. The scopes in 5.5 cover every module this change touches, and CI runs the rest.
- **No checkpoint articulated the motion.** `gemma-3-4b-it-4bit` describes a red square on white and calls it constant; `lfm2-vl-450m-4bit` walks the frames and describes a red square in each. The native paths do no better on the same clip: Qwen2.5-VL calls it "a static image ... no changes or movements". This is the checkpoints' multi-image temporal reasoning, not the substitution, and 5.3 is what separates the two. No claim is made here that the fallback lets a small model reason about motion, and the documentation says the same thing.
- **Qwen2.5-VL is a native control here, not a fallback case.** The issue names it as the fallback checkpoint to validate against, which was true when the issue was filed. Qwen-VL gained a native video path in #1166, so in this tree it exercises the untouched-native half instead, and `lfm2-vl-450m-4bit` took its place as the second fallback family.
- **The Anthropic and disaggregated-router fronts are unreachable for video** and are covered by reasoning about their types and guards rather than by a request that gets there.
- **`--video-fps` on the server is not a per-request override.** A request's own `video_url.fps` still wins; the flag supplies the default when the request omits it.

---

## 7. Files

| File | Role |
|---|---|
| `src/multimodal/video.rs` | `DEFAULT_FALLBACK_MAX_FRAMES`, `MIN_FALLBACK_MAX_FRAMES`, `evenly_spaced_indices`, `subsample_evenly`, `frames_to_png`, `load_video_source_frames_fallback` |
| `src/models/detection.rs`, `src/models/mod.rs` | `model_type_has_native_video`, `model_type_is_vision_capable`, the single list both fronts read |
| `src/server/state.rs` | `ModelMediaSupport::video_native` / `video_frames_fallback` and the `video()` accessor |
| `src/server/startup.rs` | `detect_model_media_support` sets both flags; `video_max_frames` / `video_fps` on `ServerStartupConfig` and the clamp in `build_server_config` |
| `src/server/chat_request.rs` | `VideoFramesFallback`, `expand_video_parts_to_frames(_with_allowlist)`, the clip-count and per-clip budget checks, `video_frame_budget_rejection`, `apply_video_frame_expansion` |
| `src/server/routes/chat.rs`, `routes/responses.rs`, `routes/prompt_inspection.rs` | the three wiring points |
| `src/server/media.rs` | `resolve_video_url` widened to `pub(crate)`; `media_capability_rejection` reads `support.video()` |
| `src/server/model_worker.rs` | the native-video guard kept as a backstop, reworded |
| `src/commands/generate.rs` | `CliVideoFrames`, `expand_cli_videos_to_frames`, the call site in `run_generate_once` |
| `src/commands/generate_vlm.rs` | the refusal reworded for the only case that still reaches it |
| `src/main.rs`, `src/bin/mlx_server.rs`, `src/commands/serve.rs`, `src/server/cli_input.rs`, `src/server/config.rs` | `--video-max-frames` and `--video-fps` across the three fronts |
| `docs/supported-models.md`, `docs/llama-server-compat.md`, `docs/environment-variables.md` | the native / fallback distinction, the `video_url` section, the two new env vars |
