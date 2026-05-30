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
