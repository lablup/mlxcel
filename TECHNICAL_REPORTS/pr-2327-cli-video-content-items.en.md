# PR #2327: Render video content items in the CLI for templates without image support

**Date**: 2026-10-16
**Status**: Implemented; revert check on the CLI test and a real checkpoint (`qwen3-vl-2b-instruct-4bit`) verified before merge; the video-only-template case itself is covered only by a synthetic template (see "Not verified and follow-up")
**Risk**: Low (prompt construction only; templates with an image branch render as before, and the server change only removes items the template could not render)

## Summary

Issue #1789 fixes a CLI and server prompt mismatch for chat templates that have a `video` branch and no `image` branch.

1. `apply_vlm_chat_template` (`src/commands/generate.rs`) now builds the typed content list when the template handles image, video or audio, and decides the items per modality. Clip lead-text sentences are kept.
2. The server (`src/server/chat_request.rs`) keeps its raw-render gate and gains `drop_unsupported_media_items`, which removes typed `image` and `video` items the template has no branch for.
3. Two tests pin the same prompt from both sides: `cli_video_only_template_renders_the_server_prompt` and `video_only_template_renders_the_cli_prompt`.

## Problem

The CLI flattened the turn to text whenever the template lacked an `image` branch. A template with a `video` branch and no `image` branch therefore got the video item from the server's raw render but nothing from the CLI. For a request with one video and the question "What moves?", the server rendered `<|user|><VIDEO>What moves?<|end|><|assistant|>` and the CLI rendered `<|user|>What moves?<|end|><|assistant|>`. The model saw different prompts depending on the front.

## Technical decisions

- Keep the list-shape gate as "any of image, video, audio", the same condition as the server's raw-render gate. Narrowing the server gate to a per-modality check broke `vlm_chat_template_omits_video_when_template_lacks_video_support`, so the gate stays wide and the items are filtered inside the list instead.
- Decide items per modality in the CLI: image items and fallback frames only with an image branch, video items only with a video branch, audio only with an audio branch. Fallback-frame groups still emit their lead sentence when frames are dropped, so both fronts keep the same text.
- Add `drop_unsupported_media_items` on the server rather than change the CLI to emit everything, so a template that renders only one modality gets the same items from both fronts. Audio items are not touched there: audio stays flattened on the server.
- Test with a synthetic template because no known checkpoint has a video branch without an image branch. The same template string is pinned on both sides so the two prompts are compared against one literal.

## Changes

- `src/commands/generate.rs`: `emit_image`, `emit_video` and `emit_audio` are computed up front; the flatten gate becomes "none of the three"; image items and fallback frames are gated on `emit_image`.
- `src/server/chat_request.rs`: `drop_unsupported_media_items` is called on the raw-JSON messages before rendering.
- `src/commands/generate_tests.rs` (new test, 59 lines) and `src/server/chat_request_tests.rs` (new test, 46 lines).

## Behavior change

For a template with a `video` branch and no `image` branch, the CLI now includes the video item, and both the CLI and the server omit an image item the template cannot render. Templates with an image branch, and templates with none of the three branches, render as before.

## Verification

- Revert check: with the fix reverted, `cli_video_only_template_renders_the_server_prompt` fails, rendering `<|user|>What moves?<|end|><|assistant|>` instead of `<|user|><VIDEO>What moves?<|end|><|assistant|>`. The server twin passes with or without the change, because the server already raw-rendered the video, so the CLI test carries the revert evidence.
- `commands::generate` (87 passed) and `server::chat_request` (133 passed) with `--features cuda --profile test-fast -- --test-threads=1`. clippy with `-D warnings`, `cargo fmt --check` and the contract scripts are clean.
- Real checkpoint `qwen3-vl-2b-instruct-4bit` with an ffmpeg `testsrc` clip: the CLI-rendered prompt and the server `POST /apply-template` output are byte-identical, and `mlxcel generate --video` runs normally.

## Not verified and follow-up

- No local or known checkpoint has a video branch without an image branch (Qwen-VL and Gemma 4 templates have both), so the real-checkpoint run covers the existing image-plus-video path only. The video-only case rests on the synthetic template test.
- Audio is still flattened on the server and the CLI emits audio items only with an audio branch, so a template with only an audio branch is not pinned by a parity test.
