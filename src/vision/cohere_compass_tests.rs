// Copyright 2025-2026 Lablup Inc.
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

//! Prompt-side gates for the Compass VLM: the image grid its processor
//! geometry produces, and the `<|IMAGE_PAD|>` expansion that has to match it
//! token for token or the merge silently drops vision features.

use crate::multimodal::qwen_vl::{InsertedQwenVlmTokens, insert_qwen_vl_image_tokens};
use crate::vision::processors::qwen2_vl::Qwen2VLProcessor;

/// Vision token ids of the published checkpoint.
const VISION_START: i32 = 255028;
const VISION_END: i32 = 255029;
const IMAGE_PAD: i32 = 255031;

/// The processor the Compass loader builds: patch 16, temporal 2, merge 2,
/// mean/std 0.5, and the checkpoint's own resize bounds.
fn compass_processor() -> Qwen2VLProcessor {
    Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        .with_pixel_bounds(65536, 16_777_216)
}

/// A 448x448 image is already a multiple of `patch_size * merge_size = 32` and
/// sits inside the checkpoint's pixel bounds, so it survives `smart_resize`
/// untouched: grid `[1, 28, 28]`, merged to `14 x 14 = 196` image tokens.
#[test]
fn image_pad_expands_to_merged_grid() {
    let processor = compass_processor();
    let image = image::DynamicImage::new_rgb8(448, 448);
    let grid = processor.compute_grid_thw(std::slice::from_ref(&image));
    assert_eq!(grid, vec![(1, 28, 28)]);

    // What the chat template renders for one image content part.
    let mut prompt_tokens = vec![2, VISION_START, IMAGE_PAD, VISION_END, 42];
    let stats = insert_qwen_vl_image_tokens(&mut prompt_tokens, &grid, 2, VISION_START, IMAGE_PAD);

    assert_eq!(
        stats,
        Some(InsertedQwenVlmTokens {
            image_blocks: 1,
            video_blocks: 0,
            total_image_tokens: 196,
            total_video_tokens: 0,
        })
    );
    assert_eq!(prompt_tokens.len(), 5 + 195);
    assert_eq!(prompt_tokens[0], 2);
    assert_eq!(prompt_tokens[1], VISION_START);
    assert_eq!(
        prompt_tokens.iter().filter(|&&t| t == IMAGE_PAD).count(),
        196,
        "the single template placeholder must expand to t * (h/2) * (w/2) copies"
    );
    // BOS, VISION_START, 196 image pads (indices 2..=197), VISION_END, then the
    // one text token the fixture prompt carries.
    assert_eq!(prompt_tokens[198], VISION_END);
    assert_eq!(prompt_tokens[199], 42);
}

/// The checkpoint's `min_pixels` of 65536 is what pulls a small image up to a
/// usable grid. With the Qwen family default of 3136 it would stay tiny, and
/// the prompt would carry a different token count than the HF oracle.
#[test]
fn small_images_are_upscaled_to_the_checkpoint_min_pixels() {
    let processor = compass_processor();
    let image = image::DynamicImage::new_rgb8(64, 64);
    let grid = processor.compute_grid_thw(std::slice::from_ref(&image));
    let (_, h, w) = grid[0];
    assert!(
        (h * 16) as usize * (w * 16) as usize >= 65_536,
        "smart_resize ignored the checkpoint's min_pixels: got {h}x{w} patches"
    );

    let default_bounds = Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3]);
    assert_ne!(
        default_bounds.compute_grid_thw(std::slice::from_ref(&image)),
        grid,
        "the test would pass for the wrong reason if the family default already \
         produced this grid"
    );
}
