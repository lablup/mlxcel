# PR #2993: Turn pictures upright by their EXIF orientation

**Date**: 2026-11-14
**Status**: Completed (PR open at report time; closes #2992)

## Executive Summary

mlxcel handed models the pixels a picture stores and ignored the EXIF `Orientation` tag. A portrait phone photo is stored on its side with tag 6 or 8, so it reached the model turned: positions were described wrong and rotated text was misread. mlx-vlm and transformers both call `PIL.ImageOps.exif_transpose` in `load_image`, and the checkpoints were measured through that. PR #2993 applies the tag right after decoding, in the one shared decode that every server family and `mlxcel generate` use, in the Kimi K3 prompt sizing that reads the header before the decode, and in the commands that open a picture themselves. Pictures without a tag decode to the same pixels as before. Review found no CRITICAL or HIGH issue; the hardening pass added a WebP test for all eight tags and a Metal real-model check. The PR as submitted was 16 files, +550/-21, most of it tests; hardening added a 47-line WebP test.

## 1. Problem Statement

The `image` crate (0.25.10) decodes the stored pixels and leaves orientation to the caller (`ImageDecoder::orientation`, `DynamicImage::apply_orientation`). `decode_request_image_with_limits` in `src/server/model_worker.rs` called `reader.decode()` and returned. That one function serves chat `image_url`, `/v1/embeddings` and `/v1/rerank` image items, `/v1/images/edits` references (all through `media::fetch_labeled_image`), the diffusion, Florence-2 and Nemotron-parse workers, the OpenXLA preprocessors, and `mlxcel generate --image` (through `decode_image_payloads_with_limits`). The CLI commands `embed`, `rerank`, `generate-image`, the diffusion path of `generate`, and the YOLO11 and RT-DETRv2 detectors called `image::open` or `image::load_from_memory` directly and had the same gap.

## 2. Change Summary

| Area | Change |
|---|---|
| `multimodal/image_orientation.rs` (new) | `exif_orientation(bytes)` reads the tag from a header-only parse; `turn_upright`, `load_upright_from_memory`, `open_upright`; `upright_dimensions` for header-only callers |
| `server/model_worker.rs` | The shared decode applies the tag after `reader.decode()` succeeds |
| `server/chat_request.rs` | `planned_image_size` (extracted from `kimi_k3_image_prompts`) checks limits on the stored size and returns the upright size |
| `commands/{embed,rerank,generate_image,generate_vlm}.rs`, `vision/detection/{yolo11,rt_detr_v2}` | `image::open` / `load_from_memory` swapped for the upright helpers |
| Tests | Pictures built in memory (PNG `eXIf`, JPEG `APP1`, WebP `EXIF`), stored pixels computed from the EXIF definition, all tags 1 to 8 |
| Docs | `docs/llama-server-compat.md`, `docs/embeddings.md` |

The other detectors (RF-DETR, PP-DocLayoutV3), the OCR commands, and Florence-2 already read bytes and decode through the shared function, so they inherit the fix without a change.

## 3. Technical Decisions

### 3.1 The tag is read in a second header pass

The obvious single-pass form is `into_decoder()`, `decoder.orientation()`, then `DynamicImage::from_decoder`. It was rejected because `ImageReader::decode` reserves the output buffer against `max_alloc` (`limits.reserve(decoder.total_bytes())`) before it decodes, and `into_decoder` skips that reservation. Keeping `reader.decode()` leaves the decompression-bomb and allocation checks exactly as they were. The orientation parse runs only after a successful decode, so its header and metadata parsing already passed the request's limits once.

The cost is a second header parse per picture. For PNG and WebP it reads chunks up to the image data. For JPEG, `JpegDecoder::new` in image 0.25.10 copies the whole input into a `Vec` before reading headers, so each JPEG is copied once more. The input is bounded by `--max-image-payload-size`, and a memcpy of that size is small next to the decode and the vision encoder.

### 3.2 An unreadable tag never refuses a picture

No tag, a value outside 1 to 8, a format without EXIF, and metadata the parser rejects all mean "use the picture as stored". The decode is what decides whether a request is valid; the tag is a display hint, and rejecting a decodable picture over it would be a regression for clients that send malformed EXIF today.

### 3.3 Limits apply to the stored pixels

Width, height and allocation limits are checked by the decoder on the stored frame, then a quarter turn swaps width and height. Kimi K3 follows the same rule: `planned_image_size` checks the stored header size against the limits and returns the upright size, so the planned `<|media_pad|>` run matches the grid the processor builds from the turned picture. A quarter turn allocates a second buffer of the same size for a moment; that copy is not counted against `max_alloc`, so the transient peak per picture is about twice the decoded size.

### 3.4 Visible behavior changes

A tagged picture now reaches every model upright, matching mlx-vlm, transformers and image viewers. A picture whose pixels are already upright but which still carries a stale tag is turned, as it is in those tools. Detection boxes, and the box outputs of Florence-2 and Nemotron-parse, are now in the upright frame the user sees, not the stored frame. Benchmark binaries and diagnostics, which read repository fixtures, and video frames, which carry no EXIF, are unchanged.

## 4. Verification

- Unit tests: all eight tags through PNG `eXIf` (pixel exact), JPEG `APP1` (block tolerance) and WebP `EXIF` (pixel exact, added in hardening); the decode alone is asserted to return the stored pixels, so a test cannot pass by accident. Undefined tags (0, 9, 600) and non-images are left as stored.
- The shared decode test and the Kimi K3 sizing test fail when the fix is removed (mutation runs recorded in the PR).
- macOS M3 Ultra (Metal), the merged tree: the targeted lib tests (120 passed), `cargo clippy --lib --tests -D warnings`, and a release build pass.
- Real model on Metal, `gemma-3n-e2b-it-4bit`, greedy, shapes fixture: the upright JPEG and the tag 3, 6 and 8 JPEGs all answer "triangle at the bottom, square and circle at the top"; the same tag 6 pixels without the tag answer "square at the bottom". The server (`POST /v1/chat/completions`, base64 `image_url`) gives the same split for tag 6 and its untagged twin.
- The contributor's GB10 runs cover Qwen3-VL-4B and gemma-3-4b on JPEG, PNG and WebP, and 35 checkpoints on an untagged picture with no change beyond known decode nondeterminism.

## 5. Learning Points

- The `image` crate does not apply EXIF orientation on its own in any of `open`, `load_from_memory` or `ImageReader::decode`. Any new entry point should go through `decode_request_images_with_limits` (bytes from a request) or `image_orientation::open_upright` / `load_upright_from_memory` (CLI paths), not `image::open`.
- Code that sizes a prompt from the header before decoding must use `upright_dimensions`, or it will disagree with the decoded grid on tags 5 to 8.

## 6. Not Verified

`embed`, `rerank`, `generate-image` references and the YOLO11 and RT-DETRv2 detectors were not run on real checkpoints with tagged input; they are one-line swaps to the tested helpers. No real phone photo was used; all pictures are synthetic. HEIC and AVIF are not enabled `image` features, so their tags are not in scope.
