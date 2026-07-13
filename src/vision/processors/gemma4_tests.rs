// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Unit tests for the Gemma 4 image / video preprocessor.
//!
//! These tests use synthetic in-memory frames so they run on any
//! platform without an actual video file or `ffmpeg` install.

use std::process::Command;

use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

use super::*;

fn synthetic_rgb(width: u32, height: u32, fill: u8) -> DynamicImage {
    let img = RgbImage::from_pixel(width, height, image::Rgb([fill, fill, fill]));
    DynamicImage::ImageRgb8(img)
}

fn frames(count: usize, width: u32, height: u32) -> Vec<DynamicImage> {
    (0..count)
        .map(|i| synthetic_rgb(width, height, (i * 7 % 200) as u8))
        .collect()
}

#[test]
fn process_videos_emits_one_features_block_per_video() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(8, 320, 240), frames(12, 256, 192)];
    let features = processor.process_videos(&videos, None);
    assert_eq!(features.len(), 2);
    assert_eq!(features[0].num_frames(), 8);
    assert_eq!(features[1].num_frames(), 12);
}

#[test]
fn process_videos_clamps_long_inputs_to_video_num_frames() {
    let mut processor = Gemma4Processor::new(16, 280, 3);
    processor.set_video_config(70, 4).unwrap();
    let videos = vec![frames(20, 224, 224)];
    let features = processor.process_videos(&videos, None);
    assert_eq!(features.len(), 1);
    assert_eq!(
        features[0].num_frames(),
        4,
        "downsampled to video_num_frames"
    );
    assert_eq!(features[0].frame_timestamps.len(), 4);
}

#[test]
fn process_videos_pixel_values_have_expected_shape() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(2, 192, 192)];
    let features = processor.process_videos(&videos, None);
    let first = &features[0].frames[0];
    let shape = mlxcel_core::array_shape(first.pixel_values.as_ref().unwrap());
    assert_eq!(shape.len(), 4, "pixel_values must be (1, C, H, W)");
    assert_eq!(shape[0], 1, "batch axis");
    assert_eq!(shape[1], 3, "channel axis is 3 (RGB)");
    let side_mult = (processor.pooling_kernel_size * processor.patch_size) as i32;
    assert!(
        shape[2] % side_mult == 0,
        "H must be divisible by side_mult"
    );
    assert!(
        shape[3] % side_mult == 0,
        "W must be divisible by side_mult"
    );
}

#[test]
fn process_videos_all_frames_share_video_dimensions() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(6, 320, 240)];
    let features = processor.process_videos(&videos, None);
    let mut shapes = features[0]
        .frames
        .iter()
        .map(|f| mlxcel_core::array_shape(f.pixel_values.as_ref().unwrap()));
    let first = shapes.next().unwrap();
    for shape in shapes {
        assert_eq!(
            shape, first,
            "every frame in the same video must share H and W"
        );
    }
}

#[test]
fn process_videos_num_soft_tokens_within_budget() {
    let mut processor = Gemma4Processor::new(16, 280, 3);
    processor.set_video_config(70, 32).unwrap();
    let videos = vec![frames(2, 384, 288)];
    let features = processor.process_videos(&videos, None);
    let budget = processor.video_max_soft_tokens;
    assert!(
        features[0].num_soft_tokens_per_frame <= budget,
        "soft tokens per frame {} exceeds budget {}",
        features[0].num_soft_tokens_per_frame,
        budget
    );
    assert!(
        features[0].num_soft_tokens_per_frame > 0,
        "non-empty frames must produce at least one soft token"
    );
}

#[test]
fn process_videos_per_video_fps_drives_timestamps() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(4, 224, 224), frames(4, 224, 224)];
    let features = processor.process_videos(&videos, Some(&[2.0, 1.0]));

    // First video sampled at 2 fps: t = [0, 0.5, 1.0, 1.5]
    assert!((features[0].frame_timestamps[1] - 0.5).abs() < 1e-6);
    // Second video sampled at 1 fps: t = [0, 1, 2, 3]
    assert!((features[1].frame_timestamps[1] - 1.0).abs() < 1e-6);
}

#[test]
fn process_videos_invalid_fps_falls_back_to_default() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(2, 224, 224)];
    let features = processor.process_videos(&videos, Some(&[0.0]));
    // Default fps = 2.0 so timestamp[1] should be 0.5.
    assert!((features[0].frame_timestamps[1] - 0.5).abs() < 1e-6);
}

#[test]
fn process_videos_handles_empty_input() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let features = processor.process_videos(&[], None);
    assert!(features.is_empty());
}

#[test]
fn process_videos_handles_single_frame_video() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let videos = vec![frames(1, 224, 224)];
    let features = processor.process_videos(&videos, None);
    assert_eq!(features.len(), 1);
    assert_eq!(features[0].num_frames(), 1);
    assert_eq!(features[0].frame_timestamps, vec![0.0]);
}

#[test]
fn set_video_config_rejects_unsupported_soft_tokens() {
    let mut processor = Gemma4Processor::new(16, 280, 3);
    assert!(processor.set_video_config(99, 32).is_err());
    // Sanity: supported values pass.
    for &n in SUPPORTED_VIDEO_SOFT_TOKENS {
        let mut p = Gemma4Processor::new(16, 280, 3);
        assert!(p.set_video_config(n, 32).is_ok());
    }
}

#[test]
fn process_videos_existing_image_path_unchanged() {
    // Sanity: adding the video path must not alter the image path.
    // We confirm Gemma4ImageInput shapes match what Gemma 4 image
    // wiring already expects.
    let processor = Gemma4Processor::new(16, 280, 3);
    let images = vec![synthetic_rgb(224, 224, 128)];
    let processed = processor.preprocess(&images);
    assert_eq!(processed.len(), 1);
    assert!(processed[0].num_soft_tokens > 0);
    let shape = mlxcel_core::array_shape(processed[0].pixel_values.as_ref().unwrap());
    assert_eq!(shape.len(), 4);
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 3);
}

// ─── Full-pipeline pixel-content test (requires ffmpeg) ──────────────────────

/// Return true if ffmpeg is on PATH. Matches the gating pattern used in
/// video_tests.rs so tests skip cleanly on systems without ffmpeg.
fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Synthesize a single-color video (R, G, B) of `frames` frames at `fps` into
/// `out`. Returns true on success. Writes per-frame PNGs to a temp dir and
/// encodes with ffmpeg.
fn synth_solid_color_video(
    out: &std::path::Path,
    frames: usize,
    fps: u32,
    width: u32,
    height: u32,
    r: u8,
    g: u8,
    b: u8,
) -> bool {
    let tmp_dir = std::env::temp_dir().join(format!("mlxcel-gemma4-solid-{r}-{g}-{b}-{fps}fps"));
    if std::fs::create_dir_all(&tmp_dir).is_err() {
        return false;
    }

    for i in 0..frames {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(width, height, Rgb([r, g, b]));
        let fp = tmp_dir.join(format!("frame_{i:03}.png"));
        if img.save(&fp).is_err() {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return false;
        }
    }

    let pattern = tmp_dir.join("frame_%03d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            &fps.to_string(),
            "-i",
        ])
        .arg(&pattern)
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(out)
        .status()
        .ok();

    let _ = std::fs::remove_dir_all(&tmp_dir);
    status.map(|s| s.success()).unwrap_or(false)
}

/// Full-pipeline test: synthesize a solid-color video → load_video →
/// Gemma4Processor::process_videos → verify the output NCHW tensor has correct
/// channel ordering (channel 0 = R, 1 = G, 2 = B) and that pixel values
/// match the input color after de-normalization.
///
/// The processor uses `rescale_factor = 1/255`, so to de-normalize we
/// multiply back by 255. Tolerance of ±20 accounts for YUV420 codec
/// round-trip plus the resize interpolation applied by `process_videos`.
///
/// This test catches channel-order bugs: if R and B were swapped in the
/// NCHW packing, channel 0 would contain ~50/255 ≈ 0.196 instead of
/// ~200/255 ≈ 0.784 — well outside the ±20/255 tolerance.
#[test]
fn process_videos_pixel_values_match_input_color() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // Input color: R=200, G=100, B=50. Channels differ enough that any
    // permutation (BGR, BRG, GBR, …) is distinguishable.
    const TARGET_R: u8 = 200;
    const TARGET_G: u8 = 100;
    const TARGET_B: u8 = 50;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const TOLERANCE: f32 = 20.0; // 8-bit counts after de-normalization

    let video_path = std::env::temp_dir().join("mlxcel-gemma4-pixel-values-match.mp4");
    let _ = std::fs::remove_file(&video_path);
    if !synth_solid_color_video(
        &video_path,
        10,
        10,
        WIDTH,
        HEIGHT,
        TARGET_R,
        TARGET_G,
        TARGET_B,
    ) {
        eprintln!("SKIP: could not synthesize solid-color video");
        return;
    }

    // Load video frames using the same path as the production code.
    let frames = crate::multimodal::video::load_video(&video_path, Some(2.0), None);
    let _ = std::fs::remove_file(&video_path);

    let frames = match frames {
        Ok(f) => f,
        Err(e) => {
            eprintln!("SKIP: load_video failed: {e}");
            return;
        }
    };

    assert!(!frames.is_empty(), "should have at least one decoded frame");

    // Run through the Gemma4 preprocessor — patch_size=16, max_soft_tokens=280,
    // pooling_kernel_size=2 matches real Gemma 4 config.
    let processor = Gemma4Processor::new(16, 280, 2);
    let features = processor.process_videos(&[frames], None);

    assert_eq!(features.len(), 1, "one video → one features block");
    assert!(!features[0].frames.is_empty(), "features must have frames");

    // Inspect the first frame's pixel_values tensor (shape [1, 3, H, W]).
    let first_frame = &features[0].frames[0];
    let arr = first_frame.pixel_values.as_ref().unwrap();

    // Evaluate and extract raw bytes (f32 little-endian / native-endian).
    mlxcel_core::eval(arr);
    let raw = mlxcel_core::array_to_raw_bytes(arr);
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect();

    let shape = mlxcel_core::array_shape(arr);
    assert_eq!(shape.len(), 4, "pixel_values must be (1, C, H, W)");
    assert_eq!(shape[0], 1, "batch axis");
    assert_eq!(shape[1], 3, "channel axis (RGB)");
    let c = shape[1] as usize;
    let h = shape[2] as usize;
    let w = shape[3] as usize;
    assert_eq!(floats.len(), c * h * w, "float count must match C*H*W");

    // Compute per-channel mean over the spatial plane (NCHW layout).
    // Channel c occupies floats[c*h*w .. (c+1)*h*w].
    let plane_size = h * w;
    let mean_r: f32 = floats[..plane_size].iter().sum::<f32>() / plane_size as f32;
    let mean_g: f32 = floats[plane_size..2 * plane_size].iter().sum::<f32>() / plane_size as f32;
    let mean_b: f32 = floats[2 * plane_size..].iter().sum::<f32>() / plane_size as f32;

    // De-normalize: multiply by 255 to convert back to 8-bit counts.
    let mean_r_8bit = mean_r * 255.0;
    let mean_g_8bit = mean_g * 255.0;
    let mean_b_8bit = mean_b * 255.0;

    assert!(
        (mean_r_8bit - TARGET_R as f32).abs() <= TOLERANCE,
        "channel 0 (R) mean after de-norm: {mean_r_8bit:.1} != expected {TARGET_R} (±{TOLERANCE}). \
         Channel order may be wrong (e.g., BGR)."
    );
    assert!(
        (mean_g_8bit - TARGET_G as f32).abs() <= TOLERANCE,
        "channel 1 (G) mean after de-norm: {mean_g_8bit:.1} != expected {TARGET_G} (±{TOLERANCE})"
    );
    assert!(
        (mean_b_8bit - TARGET_B as f32).abs() <= TOLERANCE,
        "channel 2 (B) mean after de-norm: {mean_b_8bit:.1} != expected {TARGET_B} (±{TOLERANCE})"
    );
}

// ---------------------------------------------------------------------------
// Per-request image soft-token budget override (issue #777)
// ---------------------------------------------------------------------------

/// Emulate the placeholder expansion the runtime performs, so the tests can
/// assert the prompt's image-token run and the emitted feature rows agree.
///
/// Mirrors `vlm_runtime::expand_gemma4_image_tokens`: each image becomes
/// `<boi>` + `image_token * num_soft_tokens` + `<eoi>`.
fn expanded_placeholder_count(inputs: &[Gemma4ImageInput], image_token_id: i32) -> usize {
    const BOI: i32 = -1;
    const EOI: i32 = -2;
    let mut expanded: Vec<i32> = Vec::new();
    for input in inputs {
        expanded.push(BOI);
        expanded.extend(std::iter::repeat_n(image_token_id, input.num_soft_tokens));
        expanded.push(EOI);
    }
    expanded.iter().filter(|&&t| t == image_token_id).count()
}

/// The soft-token count the vision tower will actually emit for an image is
/// `patch_h * patch_w / pooling_kernel_size^2`. Recompute it straight from the
/// patch grid so the test does not just re-read the field it is validating.
fn soft_tokens_from_patch_grid(input: &Gemma4ImageInput, pooling_kernel_size: usize) -> usize {
    let (patch_h, patch_w) = input.patch_grid;
    (patch_h * patch_w) / pooling_kernel_size.pow(2)
}

#[test]
fn image_soft_token_budget_ladder_is_shared_with_video() {
    assert_eq!(SUPPORTED_IMAGE_SOFT_TOKENS, SUPPORTED_VIDEO_SOFT_TOKENS);
    assert_eq!(SUPPORTED_IMAGE_SOFT_TOKENS, &[70, 140, 280, 560, 1120]);
}

#[test]
fn patch_grid_at_top_budget_stays_inside_the_vit_position_table() {
    // The gemma4 ViT indexes its learned per-axis position table with the raw
    // patch x/y ids (`build_patch_position_ids` in encoders/gemma4.rs), gathered
    // with `mlxcel_core::take`, which WRAPS on an out-of-range index rather than
    // faulting. So an oversized patch grid would silently produce wrong position
    // embeddings. The table default is `position_embedding_size = 10_240`. Pin
    // that even a pathological aspect ratio at the top ladder budget keeps every
    // single-axis patch index below it, mirroring the gemma4_unified bound test.
    const VIT_POSITION_TABLE_SIZE: usize = 10_240;
    let processor = Gemma4Processor::new(16, 280, 3);
    // Extreme aspect ratios drive one axis toward `max_side_length`.
    for (w, h) in [(8192u32, 64u32), (64u32, 8192u32), (16384u32, 32u32)] {
        let image = synthetic_rgb(w, h, 100);
        let out = processor.preprocess_with_budget(std::slice::from_ref(&image), Some(1120));
        let (patch_h, patch_w) = out[0].patch_grid;
        assert!(
            patch_h < VIT_POSITION_TABLE_SIZE && patch_w < VIT_POSITION_TABLE_SIZE,
            "patch grid {:?} for a {w}x{h} image at budget 1120 must stay inside the \
             {VIT_POSITION_TABLE_SIZE}-wide position table",
            (patch_h, patch_w)
        );
    }
}

#[test]
fn preprocess_with_budget_yields_strictly_increasing_soft_tokens() {
    // Matches every shipped gemma4 checkpoint: patch 16, pooling 3, default 280.
    let processor = Gemma4Processor::new(16, 280, 3);
    let image = synthetic_rgb(640, 480, 120);

    let counts: Vec<usize> = [70usize, 280, 1120]
        .iter()
        .map(|&budget| {
            let out = processor.preprocess_with_budget(std::slice::from_ref(&image), Some(budget));
            assert_eq!(out.len(), 1);
            out[0].num_soft_tokens
        })
        .collect();

    assert!(
        counts[0] < counts[1] && counts[1] < counts[2],
        "soft-token counts must strictly increase with the budget, got {counts:?}"
    );
    // Each budget is an upper bound, never exceeded.
    for (&budget, &count) in [70usize, 280, 1120].iter().zip(counts.iter()) {
        assert!(
            count <= budget,
            "budget {budget} produced {count} soft tokens, which exceeds the budget"
        );
    }
}

#[test]
fn placeholder_expansion_matches_feature_count_at_every_budget() {
    const IMAGE_TOKEN_ID: i32 = 258_880;
    let processor = Gemma4Processor::new(16, 280, 3);
    let image = synthetic_rgb(800, 600, 90);

    for budget in [70usize, 280, 1120] {
        let processed =
            processor.preprocess_with_budget(std::slice::from_ref(&image), Some(budget));

        // What the vision tower will emit, derived from the patch grid.
        let emitted: usize = processed
            .iter()
            .map(|input| soft_tokens_from_patch_grid(input, processor.pooling_kernel_size))
            .sum();
        // What the prompt will reserve.
        let placeholders = expanded_placeholder_count(&processed, IMAGE_TOKEN_ID);

        assert_eq!(
            placeholders, emitted,
            "budget {budget}: prompt reserves {placeholders} image tokens but the tower emits \
             {emitted} feature rows; the prompt would desync from the features"
        );
    }
}

#[test]
fn placeholder_expansion_matches_feature_count_for_multiple_images() {
    const IMAGE_TOKEN_ID: i32 = 258_880;
    let processor = Gemma4Processor::new(16, 280, 3);
    // Different aspect ratios so the per-image counts differ under one budget.
    let images = vec![
        synthetic_rgb(640, 480, 30),
        synthetic_rgb(480, 640, 60),
        synthetic_rgb(1024, 256, 90),
    ];

    for budget in [70usize, 560] {
        let processed = processor.preprocess_with_budget(&images, Some(budget));
        assert_eq!(processed.len(), 3);

        let emitted: usize = processed
            .iter()
            .map(|input| soft_tokens_from_patch_grid(input, processor.pooling_kernel_size))
            .sum();
        let placeholders = expanded_placeholder_count(&processed, IMAGE_TOKEN_ID);
        assert_eq!(placeholders, emitted, "budget {budget}: multi-image desync");
    }
}

#[test]
fn default_path_is_byte_identical_to_configured_budget() {
    let processor = Gemma4Processor::new(16, 280, 3);
    let images = vec![synthetic_rgb(640, 480, 120), synthetic_rgb(333, 777, 40)];

    let default_out = processor.preprocess(&images);
    let none_out = processor.preprocess_with_budget(&images, None);
    // Explicitly asking for the checkpoint's configured budget must land on the
    // same result as not asking at all.
    let explicit_out = processor.preprocess_with_budget(&images, Some(280));

    for i in 0..images.len() {
        assert_eq!(default_out[i].patch_grid, none_out[i].patch_grid);
        assert_eq!(
            default_out[i].num_soft_tokens, none_out[i].num_soft_tokens,
            "preprocess() and preprocess_with_budget(None) must agree"
        );
        assert_eq!(default_out[i].patch_grid, explicit_out[i].patch_grid);
        assert_eq!(
            default_out[i].num_soft_tokens,
            explicit_out[i].num_soft_tokens
        );

        // Same pixel tensor shape, and byte-identical contents.
        let lhs = default_out[i].pixel_values.as_ref().unwrap();
        let rhs = none_out[i].pixel_values.as_ref().unwrap();
        assert_eq!(mlxcel_core::array_shape(lhs), mlxcel_core::array_shape(rhs));
        mlxcel_core::eval(lhs);
        mlxcel_core::eval(rhs);
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(lhs),
            mlxcel_core::array_to_raw_bytes(rhs),
            "default path must be byte-identical to the pre-override behavior"
        );
    }
}

#[test]
fn validate_image_soft_tokens_accepts_every_ladder_rung() {
    for &budget in SUPPORTED_IMAGE_SOFT_TOKENS {
        assert_eq!(validate_image_soft_tokens(budget), Ok(budget));
    }
}

#[test]
fn validate_image_soft_tokens_rejects_off_ladder_values() {
    // Zero, an unbounded value, and a plausible-looking near-miss.
    for bad in [0usize, 1, 281, 2240, usize::MAX] {
        let err = validate_image_soft_tokens(bad)
            .expect_err("off-ladder budget must be rejected, not clamped");
        assert!(
            err.contains("must be one of"),
            "error should name the supported values, got: {err}"
        );
    }
}
