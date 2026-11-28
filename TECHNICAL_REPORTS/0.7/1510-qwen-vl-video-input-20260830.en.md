# Technical Report: PR #1510 - qwen-vl-video-input

**Date**: 2026-08-30

**Status**: Completed with runtime qualification limits

**Languages**: Rust, Markdown

**Risk Level**: Medium

## Executive Summary

PR #1510 adds code-path support for video input across the Qwen-VL family by connecting the CLI, OpenAI-compatible server, Qwen processor, placeholder expansion, MRoPE position construction, and visual embedding merge paths. The implementation is intentionally fail-closed: decoded media, rendered placeholders, visual grids, projected feature rows, and final token positions must match exactly before generation can start.

The change is validated by focused Rust tests, formatting, diff checks, non-workspace clippy, and hosted static CI observed before the report commit. Real ffmpeg-backed video decoding and Qwen3.5/Qwen3.8 checkpoint generation were not run because `ffmpeg`/`ffprobe` and local checkpoint files were unavailable in the worktree.

## 1. Problem Statement

### 1.1 Background

Issue #1166 was split from the Qwen3.8 qualification work because the model family already publishes a video contract in its configs and chat templates, while mlxcel still rejected `--video` for Qwen-VL. The existing implementation carried image and video token IDs through loading, but only image preprocessing, image placeholder expansion, and image-oriented Qwen MRoPE were wired into the runtime path.

### 1.2 Existing Issues

- **CLI/server rejection**: Qwen-VL models were not routed to any video-capable embedding path, so video requests failed even when the checkpoint declared `video_token_id` and rendered `<|video_pad|>` placeholders.
- **Missing cardinality contract**: The old Qwen path only counted image blocks and had no strict validation that declared videos, decoded frames, placeholder spans, projected feature rows, and final embedding positions agreed.
- **MRoPE drift risk**: Multiple Qwen wrappers carried duplicated visual-position builders, making it easy for image and video semantics to diverge between families.
- **Silent media loss risk**: Without fail-closed checks, malformed mixed media prompts could silently drop videos, clamp frame counts, or assign visual features to the wrong token span.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Qwen video requests keep failing despite model support | High | High |
| Mismatched placeholder and feature counts corrupt generation inputs | High | Medium |
| Unbounded or silently clamped video sampling hides request failures | Medium | Medium |
| Real checkpoint behavior differs from code-path tests | Medium | Medium until ffmpeg/checkpoint qualification runs |

## 2. Change Summary

| Category | Count | Summary |
|----------|-------|---------|
| CLI/server integration | 4 areas | CLI `--video`, server video request preparation, media-support detection, and help/error text now include Qwen-VL variants. |
| Qwen processor/runtime | 5 areas | Processor sidecar parsing, mixed image/video patch preprocessing, strict media-token expansion, shared MRoPE, and image/video embedding merge are connected. |
| Tests | 7 focused groups | Added or updated Qwen token ID, sidecar bounds, video padding, frame policy, media order, placeholder cardinality, and CLI/server capability tests. |
| Documentation | 1 file | `docs/supported-models.md` now describes Qwen-VL video support and clearly separates code-path support from real video checkpoint qualification. |

### Statistics

| Item | Value |
|------|-------|
| Files changed | 24 |
| Lines added | +1897 |
| Lines deleted | -595 |
| Primary commit | `78177cfb` `feat(models): add Qwen-VL video input` |

## 3. Technical Decisions

### 3.1 Use one mixed-media Qwen path instead of per-family forks

Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL MoE, Qwen3.5-VL, and Qwen3.5 MoE share the same relevant video contract: a Qwen visual processor, `video_token_id`, visual grid metadata, and MRoPE over temporal and spatial axes. PR #1510 therefore centralizes placeholder expansion and MRoPE construction in `src/multimodal/qwen_vl.rs`, then calls that helper from each family wrapper.

The main trade-off is that the shared helper is stricter than some ad hoc prompt shapes might expect. That is deliberate: a malformed prompt/media ordering problem is safer as an invalid request than as a generation with shifted visual embeddings.

### 3.2 Treat over-limit video sampling as an error, not a clamp

The video loader now accepts a `FrameSamplingPolicy` and Qwen processors pass a policy derived from `processor_config.json`. When Qwen's policy would exceed `max_frames`, the request returns a named `SampledFramesTooMany` error instead of silently truncating or clamping.

This preserves the checkpoint contract and makes validation failures visible. It also prevents a user from believing the model saw the full requested clip when the runtime actually discarded frames.

### 3.3 Preserve rendered prompt order for mixed media

The runtime scans rendered token IDs to derive whether the prompt expects image or video media first, rather than assuming images-before-videos whenever placeholders exist. If no placeholders exist, it falls back to the legacy insertion order of images followed by videos.

The scanner rejects adjacent cross-kind visual runs with no framing token, because the production Qwen MRoPE scanner otherwise cannot distinguish whether two consecutive visual spans are one media block or two media blocks.

### 3.4 Disable Qwen vision cache when videos are present

Image-only Qwen requests keep the existing request-scoped vision cache behavior. Video requests bypass that cache because decoded frame sequences are request-specific and can be large; caching them under an image key would risk stale media reuse or high memory pressure.

## 4. Implementation Details

### 4.1 End-to-end data flow

```text
CLI/server request
  -> render image/video content parts
  -> decode videos through ffmpeg-backed loader with Qwen frame policy
  -> preprocess mixed Qwen media into patch rows and visual grids
  -> expand or validate image/video token placeholders exactly
  -> compute shared Qwen MRoPE positions with T kept separate from H/W merge
  -> merge visual embeddings into image and video token positions
  -> generate with expanded prompt tokens
```

### 4.2 Key files

- `src/multimodal/video.rs` adds `FrameSamplingPolicy`, fail-closed over-max behavior, and policy-aware video loading entry points.
- `src/vision/processors/qwen2_vl.rs` adds Qwen video sidecar parsing, mixed image/video preprocessing, temporal padding, and video grid generation.
- `src/multimodal/qwen_vl.rs` adds strict mixed-media placeholder expansion and shared Qwen image/video MRoPE position construction.
- `src/multimodal/vlm_runtime.rs` derives mixed media order from rendered prompts, connects Qwen media preprocessing to embedding generation, and expands Qwen preparation summaries with video counts.
- `src/commands/generate_vlm.rs` and `src/server/model_worker.rs` route Qwen-VL video requests through the new shared helper.
- `src/vision/qwen*_vl*.rs` wrappers use the shared MRoPE helper and merge visual embeddings for both image and video token IDs.
- `docs/supported-models.md` documents the supported code path and explicitly states that real ffmpeg-backed Qwen3.8 video Q&A remains separate runtime qualification unless that exact run is cited.

## 5. Technical Review

### 5.1 Correctness

The main correctness boundary is exact visual cardinality. The implementation checks placeholder counts, media ordering, expanded run lengths, grid divisibility by `spatial_merge_size`, positive grid dimensions, checked token-count multiplication, processor frame bounds, temporal padding, and adjacent visual-run ambiguity before generation.

The finalizer review added direct tests for the runtime prompt-order scanner after finding that the lower-level insertion tests did not cover the rendered-token scanner itself.

### 5.2 Security

No new shell-string construction was introduced for user input. Video decoding continues through the existing `ffmpeg` pipeline and the new Qwen entry points pass structured paths and policy values through the existing loader APIs. The review found no credential handling, authentication, SQL, web rendering, or sensitive logging changes.

The user-facing file-path exposure remains consistent with the existing media loader behavior: errors include canonical media source descriptions to diagnose failed local video loads. That is appropriate for this local/server operator surface but should remain part of normal deployment threat review if the server is exposed to untrusted tenants.

### 5.3 Performance

The patch avoids silent over-sampling by enforcing Qwen's frame cap before preprocessing. It also avoids caching video frame embeddings under image cache keys. The main cost is expected and inherent: Qwen video preprocessing resizes and normalizes every sampled frame and creates one patch tensor for all media in the request.

No benchmark was run. The implementation adds bounded allocation checks for visual token counts and uses existing per-request video limits to control decode size and duration.

## 6. Learning Points

### 6.1 Qwen-VL MRoPE video semantics

For Qwen-VL video, temporal grid size remains a real axis. `spatial_merge_size` divides only the H/W axes, while T is expanded as frame-group positions. Treating video as a flat image-token run without preserving T would make long clips positionally wrong even when placeholder counts match.

### 6.2 Code-path support is not runtime qualification

The implementation proves the Rust integration contracts compile and pass focused unit tests. It does not prove that a Qwen3.8 checkpoint produces a correct video answer, because that requires ffmpeg, a local checkpoint, and an actual generation run on suitable hardware.

## 7. Validation Record

| Check | Result | Notes |
|-------|--------|-------|
| `cargo fmt --all -- --check` | Pass | Ran after rebase and after final test addition. |
| `git diff --check origin/main..HEAD` | Pass | No whitespace errors in the 24-file implementation diff. |
| `cargo test --lib qwen_vl` | Pass | 13 passed, 11 ignored serial-MLX tests, 7276 filtered. |
| `cargo test --lib video_processor_config` | Pass | 2 passed. |
| `cargo test --lib preprocess_video_pads_frames_to_temporal_patch_grid` | Pass | 1 passed. |
| `cargo test --lib smart_nframes_policy` | Pass | 2 passed. |
| `cargo test --lib detect_model_media_support_recognises_qwen35_vlm_video` | Pass | 1 passed. |
| `cargo test --lib qwen35_vl_token_ids` | Pass | 8 passed. |
| `cargo test --lib qwen_media_order_from_prompt` | Pass | 2 passed. |
| `cargo test cli_video_content_part_count_enables_qwen_vl_videos` | Pass | Named `src/main.rs` test passed; other enumerated targets had 0 matching tests under the filter. |
| `cargo clippy --lib --tests -- -D warnings` | Pass | Non-workspace clippy, 1m29s. |
| `cargo check --lib --features xla-iree` | Skipped by environment | Local escalated run reached `mlxcel-xla` build script and stopped because `IREE_DIST` is unset. Hosted `OpenXLA feature compile` later passed before the report commit. |
| Hosted checks before report commit | Partial pass, one pending | Static setup, `cargo-fmt`, `cargo-deny`, and `OpenXLA feature compile` passed; hosted `cargo-clippy` was still pending when the report workflow proceeded because the report commit would trigger a new run. |
| `command -v ffmpeg` / `command -v ffprobe` | Unavailable | Real decode validation could not run. |
| `nvidia-smi` outside sandbox | Available | Reports NVIDIA GB10, driver 580.173.02, CUDA 13.0. |
| Local checkpoint lookup | Unavailable | No `models/` or `models/mlx` tree in the worktree, so no real Qwen3.5/Qwen3.8 generation run was performed. |

## 8. Follow-up Actions

### Required before claiming real model qualification

- [ ] Install `ffmpeg` and `ffprobe` on the validation host and run a real Qwen-VL video decode through CLI and server paths.
- [ ] Place the intended Qwen3.5/Qwen3.8 checkpoint under the project model directory and run a fixed video prompt that verifies answer quality, placeholder expansion, prompt token accounting, and MRoPE positions.
- [ ] Compare CLI and server prompt rendering for the same mixed image/video request using a concrete checkpoint and media fixture.

### Monitoring after merge

- Watch for user reports where Qwen video requests fail with frame-bound errors; those are expected fail-closed diagnostics when a clip exceeds the processor sidecar's limits.
- Watch server memory use on long clips because video preprocessing intentionally materializes sampled frames and patch rows per request.

## Appendix

### Related issue and PR

- Issue #1166: Qwen-VL video input support.
- PR #1510: Implements the code path and tests described in this report.
