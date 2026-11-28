# Technical Report: PR #1777 - fix(video): align the CLI and server video-frames fallback

**Date**: 2026-09-11
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review)
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low-Medium (touches the CLI prompt layout, server startup state and three chat routes; real-run parity verified)

---

## Executive Summary

PR #1759 added the video-frames fallback: for a vision checkpoint with no native temporal path, a clip is decoded, subsampled, encoded as PNGs and spliced into the request as ordered images. Its reviews left five items, and all five are fixed here. The CLI now announces each clip with its own lead sentence, placed ahead of that clip's frames, exactly as the server does. CLI frame files are private. The server does no process spawns or path canonicalization on a request's worker thread, stops cloning the whole request to render a prompt, and has a decode that a disconnect can stop before it starts.

The visible outcome is prompt parity. For the same two clips, the CLI and the server now report the same prompt size (2,129 tokens on `gemma-3-4b-it-4bit`, in single-model and router mode alike). Before this PR the CLI read 2,115, one lead sentence short.

---

## Problem Statement

The CLI summed the frame count across every `--video` and prepended one sentence for all clips. The server instead emitted one sentence per clip, at that clip's position. The same two clips therefore reached the model as one undivided run of frames on the CLI and as two announced clips on the server. The other four items were:

- frame PNGs written 0644 into the shared temp directory;
- `ffmpeg_available()` and the allowlist canonicalization run on a Tokio worker;
- prompt inspection cloning the whole request, base64 images included;
- a `spawn_blocking` decode that kept running after the client left.

---

## Change Summary

- **Per-clip layout on the CLI.** `video_frames_lead_text` moved to `multimodal::video` and both fronts call it. `CliVideoFrames` holds one entry per clip, and `CliPromptMedia` renders `[lead A, frames A, lead B, frames B, question]`.
- **One flattening.** For templates without image content items, both fronts join text parts through `flatten_template_text`. Review found the CLI had been joining with `\n\n` and the server with no separator. For internvl3, deepseek-vl2, fastvlm, molmo and dots.ocr the two fronts would still have rendered different prompts.
- **`PrivateTempDir`.** A std-only per-run 0700 directory with 0600 `create_new` files. The directory prefix and every file name must be plain names. The directory is removed on drop, and the guard outlives the vision-tower read.
- **Startup state.** `resolve_video_request_inputs` resolves the allowlist, warms the ffmpeg probe and warns about writable allowlist directories. It runs at server start and, in router mode, at each model load. The result lives on `AppState`. `extract_chat_video_paths` returns early through a non-allocating `has_video_urls()`.
- **By-value rendering.** `render_chat_prompt` takes the request by value.
- **Cancellation.** `expand_request_video_parts` creates a `CancellationToken` only for requests with clips to expand, holds its drop guard in the handler future, and checks it before the loop, per clip, and at the start of the blocking decode.

---

## Technical Decisions

**Render the sentence with its clip rather than prepend it to `-p`.** The issue proposed prepending the sentences to the user's prompt. The CLI's template path emits every image item before the text item, so prepended sentences would still have landed after all the frames: the exact layout the issue set out to remove. The revert arm shows it (`<IMG>×6 ... 5 frames ... What moves?`). `-p` is left untouched, which also keeps Florence-2's raw task prompt intact.

**The cancellation check that matters is at decode start.** The token's only canceller is its own drop guard, which fires when the handler future is dropped. Dropping the future already stops the clip loop, so the loop checks only matter for a caller that keeps polling after cancelling. The check at the start of the blocking task does matter: a decode still queued behind other blocking work now never begins. The 499 `client_closed_request` mapping is kept as defensive code, and the doc comments say no live route receives it.

**Read the allowlist once, at startup.** This takes a process spawn and a `canonicalize` per entry off every request. The cost is operator-visible: a directory created after startup needs a restart for the fallback. That is documented. The native-video path still reads the allowlist per request, off the request thread, and aligning it is left for a follow-up.

---

## Validation

- **Gate.** On c5204bf5 the workspace gate passed: 123 binaries, 11,024 passed. On the review-fix commit, 121 binaries ran clean. The `mlxcel-core` lib binary aborted with `Discarded (victim of GPU error/recovery) (kIOGPUCommandBufferCallbackErrorInnocentVictim)`, the error macOS reports when another process's GPU fault forces a recovery. Re-run alone, it passed 1,731 of 1,731. Clippy, fmt and the contract checks are clean.
- **Revert arms.** Every new test fails by name with its fix reverted. This includes the CLI and server parity tests, which fail against the old `\n\n` join.
- **Real run.** `gemma-3-4b-it-4bit` with two synthetic clips (motion checked on the first and last frames) and `--video-max-frames 4`:
  - The CLI and the server both report 2,129 prompt tokens.
  - A router-mode server reports the same 2,129.
  - `/apply-template` shows two sentences and eight image markers.
  - The same run before this PR gives 2,115 on the CLI.

---

## Learning Points

- **An issue's layout fix has to be checked against the renderer, not the prompt string.** Prepending text looked right until the renderer's item order was read.
- **Parity has two paths.** The content-list path and the flattened-text path each needed the same answer. The planned Gemma run could only ever exercise the first; review found the second by reading which families' templates lack image items.
- **A GPU recovery in another process is now a test-binary abort.** At the current MLX pin, a discarded command buffer throws where it used to be swallowed. A gate run on a shared machine can fail for a reason outside the tree. The signature to recognize is `InnocentVictim`, and the fix is to re-run the affected binary alone.

---

## Follow-ups

- The server's native-video `write_video_temp_file` still writes a 0644 file, and a disconnect mid-write can leave it behind.
- The native-video path still reads the allowlist per request.
- `/v1/chat/completions/input_tokens` counts rendered template text without image soft tokens: 57 for this body, before and after this PR.
- A template that accepts `video` or `audio` items but not `image` still gets a content list on the server and flattened text on the CLI.
