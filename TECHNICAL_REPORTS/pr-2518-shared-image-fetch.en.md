# PR #2518: One shared image fetch/decode helper for the media routes

**Date**: 2026-10-21
**Status**: Completed (PR open)

## Executive Summary

PR #2518 (closes #1692) replaces three private copies of the image URL fetch and decode chain with one helper, `fetch_labeled_image` in `src/server/media.rs`. The copies were in the embeddings, rerank and images routes. The issue named only the first two. The images route copy arrived later, with #2497. Behavior does not change: every route returns the same status, error type and message text as before. Before this PR no test asserted these messages, so the PR adds four helper tests. Review found one MEDIUM test-strength issue, which was fixed. The security check found nothing.

## 1. Change

| Area | Change |
|---|---|
| `src/server/media.rs` | `fetch_labeled_image(url, label, limits) -> Result<DynamicImage, ErrorResponse>`: `try_read_image_url_with_limits`, then `decode_request_images_with_limits`, then `pop()` |
| `routes/embeddings.rs`, `routes/rerank.rs`, `routes/images.rs` | Private chains replaced by calls. Each caller still builds its label (`input[{index}]`, the rerank label, `image[{index}]`). |
| `src/server/media_tests.rs` | 4 tests: success path, unsupported scheme, read failure, decode failure |

The helper produces four messages, each as 400 `invalid_request_error`:

| Arm | Message |
|---|---|
| Unsupported scheme | `{label}: unsupported image URL scheme` |
| Read error | `{label}: failed to read image: {err:#}` |
| Decode error | `{label}: {err:#}` |
| Empty decode | `{label}: image decoded to nothing` |

## 2. Technical Decisions

- **Third copy included.** The issue asked for one shared helper. Leaving the `images.rs` copy would have kept the duplication the issue wanted removed.
- **Route checks stay in the route.** `images.rs` still runs its own checks before it fetches: the empty-list check, `validate_image_count`, the per-item empty check, and the raw-path rejection. Those checks belong to that route's contract, not to the fetch.
- **Callers keep their labels.** The label is the only difference between the copies, so passing it in keeps every string byte-identical.
- **Test strength (review, MEDIUM).** At first the decode-failure test checked only the label prefix. That check would also pass for the empty-decode arm. The test now asserts the full message, `documents[0]: Failed to decode any images`. The read-failure test still checks only the prefix, because the rest of that message comes from the underlying reader error.

## 3. Verification

| Check | Result |
|---|---|
| `server::media` | 82 passed |
| `server::routes::embeddings`, `rerank`, `images` | 22, 20, 17 passed |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | Clean |

All runs used `MLXCEL_BUILD_METAL=0 --profile test-fast --lib` on macOS. The security check confirmed that each route passes the same `ImageInputLimits` value through. No route gained a new way to read local files or reach the network.

## 4. Open Items

The review found these. They predate this PR, which does not fix them:
- **Broken PNG fixture.** The `tiny_png_bytes()` fixture in `media_tests.rs` (the same data as `PNG_BASE64` in `prompt_count_route_tests.rs`) has a bad IDAT CRC and does not decode. No current test decodes it, so no assertion is weakened. Its `/// A 1x1 PNG` comment overstates what it is. `TINY_PNG` in `rerank_tests.rs` is a valid replacement.
- **Order-dependent audio test.** `audio_acquisition_preserves_order_and_rejects_any_invalid_item` fails about 1 run in 11. `read_audio_input` treats raw base64 as a file path when a shared test media root is installed, so the result depends on test order.
