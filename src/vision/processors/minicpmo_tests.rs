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

use super::MiniCPMOProcessor;
use image::{DynamicImage, RgbImage};

#[test]
fn find_best_resize_respects_patch_alignment() {
    let processor = MiniCPMOProcessor::new(14, 448, 64);
    let (width, height) = processor.find_best_resize(1000, 400);

    assert_eq!(width % 14, 0);
    assert_eq!(height % 14, 0);
    assert!(width * height <= 448 * 448);
}

#[test]
fn find_best_resize_can_align_to_minicpmv46_downsample_factor() {
    // MiniCPM-V 4.6 runs a 2x2 VitMerger and a 2x2 Merger after patching, so
    // both image dimensions must be divisible by patch_size * 4.  The base
    // MiniCPM-O processor only aligns to patch_size; this constructor is the
    // upstream MiniCPM-V `_find_best_resize(... merge_factor=patch_size*4)`
    // equivalent.
    let processor = MiniCPMOProcessor::new_with_resize_multiples(14, 448, 64, 56, 56);
    let (width, height) = processor.find_best_resize(1000, 400);

    assert_eq!(width % 56, 0);
    assert_eq!(height % 56, 0);
    assert!(width >= 56);
    assert!(height >= 56);
}

#[test]
fn preprocess_outputs_hwc_tensor_and_spatial_shape() {
    let processor = MiniCPMOProcessor::new(14, 448, 64);
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(63, 31, image::Rgb([255, 0, 0])));

    let processed = processor.preprocess(&[image]);
    assert_eq!(processed.len(), 1);
    assert_eq!(processed[0].pixel_values_shape[0], 1);
    assert_eq!(processed[0].pixel_values_shape[3], 3);
    assert_eq!(
        processed[0].pixel_values.len(),
        (processed[0].pixel_values_shape[1]
            * processed[0].pixel_values_shape[2]
            * processed[0].pixel_values_shape[3]) as usize
    );
    assert!(processed[0].spatial_shape.0 >= 1);
    assert!(processed[0].spatial_shape.1 >= 1);
}
