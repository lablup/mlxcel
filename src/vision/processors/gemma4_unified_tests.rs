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

//! Unit tests for the Gemma 4 Unified image/audio processor.

use super::*;
use image::{DynamicImage, RgbImage};

fn read_i32_at(arr: &MlxArray, indices: &[i32]) -> i32 {
    let starts = indices.to_vec();
    let stops: Vec<i32> = indices.iter().map(|v| v + 1).collect();
    let scalar = mlxcel_core::slice(arr, &starts, &stops);
    mlxcel_core::item_i32(&scalar)
}

fn read_f32_at(arr: &MlxArray, indices: &[i32]) -> f32 {
    let starts = indices.to_vec();
    let stops: Vec<i32> = indices.iter().map(|v| v + 1).collect();
    let scalar = mlxcel_core::slice(arr, &starts, &stops);
    mlxcel_core::item_f32(&scalar)
}

fn solid_image(w: u32, h: u32, rgb: [u8; 3]) -> DynamicImage {
    let mut img = RgbImage::new(w, h);
    for px in img.pixels_mut() {
        *px = image::Rgb(rgb);
    }
    DynamicImage::ImageRgb8(img)
}

#[test]
fn patch_dim_and_padding() {
    // model_patch_size 48, num_soft_tokens 280.
    let proc = Gemma4UnifiedProcessor::new(48, 280, 640);
    assert_eq!(proc.patch_dim(), 48 * 48 * 3);

    // A 96x96 image with budget 6 produces a 2x2 patch grid (4 real patches,
    // factor == sqrt(1.5) → target stays 96x96) plus 2 padding rows. This
    // exercises the padding path deterministically.
    let small = Gemma4UnifiedProcessor::new(48, 6, 640);
    let img = solid_image(96, 96, [255, 0, 0]);
    let out = small.preprocess(std::slice::from_ref(&img));
    assert_eq!(out.len(), 1);
    let single = &out[0];
    assert_eq!(single.num_soft_tokens, 4);

    // patches: [num_soft_tokens, patch_dim]; positions: [num_soft_tokens, 2].
    assert_eq!(
        mlxcel_core::array_shape(&single.patches),
        vec![6, 48 * 48 * 3]
    );
    assert_eq!(mlxcel_core::array_shape(&single.positions), vec![6, 2]);

    // First patch position is (0, 0); first channel value is 255/255 = 1.0.
    assert_eq!(read_i32_at(&single.positions, &[0, 0]), 0);
    assert_eq!(read_i32_at(&single.positions, &[0, 1]), 0);
    assert!((read_f32_at(&single.patches, &[0, 0]) - 1.0).abs() < 1e-6);

    // A padding patch (index 4, beyond the 4 real patches) carries position -1
    // on both axes and zeros.
    assert_eq!(read_i32_at(&single.positions, &[4, 0]), -1);
    assert_eq!(read_i32_at(&single.positions, &[4, 1]), -1);
    assert_eq!(read_f32_at(&single.patches, &[4, 0]), 0.0);
}

#[test]
fn multi_patch_positions_are_grid_indexed() {
    // A 96x96 image with budget 4 lands on a 2x2 grid (factor == 1). The patch
    // loop is (py outer, px inner), so positions are row-major over (x, y).
    let proc = Gemma4UnifiedProcessor::new(48, 4, 640);
    let img = solid_image(96, 96, [10, 20, 30]);
    let out = proc.preprocess(std::slice::from_ref(&img));
    let single = &out[0];
    assert_eq!(single.num_soft_tokens, 4);
    // patch 0 = (x0, y0)
    assert_eq!(read_i32_at(&single.positions, &[0, 0]), 0);
    assert_eq!(read_i32_at(&single.positions, &[0, 1]), 0);
    // patch 1 = (x1, y0)
    assert_eq!(read_i32_at(&single.positions, &[1, 0]), 1);
    assert_eq!(read_i32_at(&single.positions, &[1, 1]), 0);
    // patch 2 = (x0, y1)
    assert_eq!(read_i32_at(&single.positions, &[2, 0]), 0);
    assert_eq!(read_i32_at(&single.positions, &[2, 1]), 1);
    // patch 3 = (x1, y1)
    assert_eq!(read_i32_at(&single.positions, &[3, 0]), 1);
    assert_eq!(read_i32_at(&single.positions, &[3, 1]), 1);
}

#[test]
fn resize_respects_soft_token_budget() {
    let proc = Gemma4UnifiedProcessor::new(48, 280, 640);
    // A very large image must not exceed num_soft_tokens patches.
    let img = solid_image(4096, 4096, [128, 128, 128]);
    let out = proc.preprocess(std::slice::from_ref(&img));
    assert!(out[0].num_soft_tokens <= 280);
    assert!(out[0].num_soft_tokens > 0);
}

#[test]
fn video_frames_use_per_frame_soft_token_budget() {
    // Default video budget is DEFAULT_VIDEO_SOFT_TOKENS_PER_FRAME (70), smaller
    // than the image budget (280). A large frame must not exceed it.
    let proc = Gemma4UnifiedProcessor::new(48, 280, 640);
    assert_eq!(
        proc.video_soft_tokens_per_frame,
        DEFAULT_VIDEO_SOFT_TOKENS_PER_FRAME
    );
    let big = solid_image(4096, 4096, [128, 128, 128]);
    let frames = proc.preprocess_video_frames(std::slice::from_ref(&big));
    assert_eq!(frames.len(), 1);
    // Padded to the per-frame budget (70), not the image budget (280).
    assert_eq!(
        mlxcel_core::array_shape(&frames[0].patches),
        vec![DEFAULT_VIDEO_SOFT_TOKENS_PER_FRAME as i32, 48 * 48 * 3]
    );
    assert_eq!(
        mlxcel_core::array_shape(&frames[0].positions),
        vec![DEFAULT_VIDEO_SOFT_TOKENS_PER_FRAME as i32, 2]
    );
    assert!(frames[0].num_soft_tokens <= DEFAULT_VIDEO_SOFT_TOKENS_PER_FRAME);
    assert!(frames[0].num_soft_tokens > 0);
}

#[test]
fn video_frames_patchify_each_frame_independently() {
    // A small explicit budget makes the patch count deterministic: a 96x96
    // frame on a 2x2 grid yields 4 real patches, padded to the budget (6).
    let mut proc = Gemma4UnifiedProcessor::new(48, 280, 640);
    proc.set_video_soft_tokens_per_frame(6);
    assert_eq!(proc.video_soft_tokens_per_frame, 6);

    let frame_a = solid_image(96, 96, [255, 0, 0]);
    let frame_b = solid_image(96, 96, [0, 255, 0]);
    let frames = proc.preprocess_video_frames(&[frame_a, frame_b]);
    assert_eq!(frames.len(), 2);
    for frame in &frames {
        assert_eq!(frame.num_soft_tokens, 4);
        assert_eq!(
            mlxcel_core::array_shape(&frame.patches),
            vec![6, 48 * 48 * 3]
        );
        assert_eq!(mlxcel_core::array_shape(&frame.positions), vec![6, 2]);
        // Padding row (index 4, beyond the 4 real patches) carries position -1.
        assert_eq!(read_i32_at(&frame.positions, &[4, 0]), -1);
        assert_eq!(read_i32_at(&frame.positions, &[4, 1]), -1);
    }
    // First channel of frame A's first patch is 255/255 = 1.0 (red).
    assert!((read_f32_at(&frames[0].patches, &[0, 0]) - 1.0).abs() < 1e-6);
}

#[test]
fn image_path_unchanged_by_video_budget() {
    // Setting the video budget must not affect the still-image path: a 96x96
    // image on budget 6 still yields 4 real patches padded to 6 (regression
    // guard for the shared patchify refactor).
    let mut proc = Gemma4UnifiedProcessor::new(48, 6, 640);
    proc.set_video_soft_tokens_per_frame(70);
    let img = solid_image(96, 96, [255, 0, 0]);
    let out = proc.preprocess(std::slice::from_ref(&img));
    assert_eq!(out[0].num_soft_tokens, 4);
    assert_eq!(
        mlxcel_core::array_shape(&out[0].patches),
        vec![6, 48 * 48 * 3]
    );
}

#[test]
fn audio_chunks_into_frames_with_mask() {
    let proc = Gemma4UnifiedProcessor::new(48, 280, 640);
    // 1.5 frames worth of samples → 2 frames; the second is zero-padded.
    let samples = vec![0.5f32; 640 + 320];
    let audio = proc.process_audio(&samples);
    assert_eq!(audio.num_frames, 2);
    assert_eq!(mlxcel_core::array_shape(&audio.features), vec![2, 640]);
    assert_eq!(mlxcel_core::array_shape(&audio.mask), vec![2]);
    assert_eq!(proc.audio_num_frames(samples.len()), 2);

    // Both frames are valid soft tokens: the trailing partial frame is
    // zero-padded to a full token and counts as a real audio soft token.
    // The placeholder count, mask length, and projected-feature count all
    // equal num_frames (2).
    let mask0 = mlxcel_core::item_bool(&mlxcel_core::slice(&audio.mask, &[0], &[1]));
    let mask1 = mlxcel_core::item_bool(&mlxcel_core::slice(&audio.mask, &[1], &[2]));
    assert!(mask0, "first full frame must be valid");
    assert!(
        mask1,
        "trailing partial frame is a valid soft token (zero-padded)"
    );

    // The padded tail of frame 1 is zero (samples ran out at index 320).
    assert_eq!(read_f32_at(&audio.features, &[1, 500]), 0.0);
    assert!((read_f32_at(&audio.features, &[1, 100]) - 0.5).abs() < 1e-6);
}

#[test]
fn audio_exact_multiple_all_valid() {
    let proc = Gemma4UnifiedProcessor::new(48, 280, 640);
    let samples = vec![1.0f32; 640 * 3];
    let audio = proc.process_audio(&samples);
    assert_eq!(audio.num_frames, 3);
    for f in 0..3 {
        let m = mlxcel_core::item_bool(&mlxcel_core::slice(&audio.mask, &[f], &[f + 1]));
        assert!(m, "frame {f} should be valid");
    }
}
