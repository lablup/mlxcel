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

use super::{Phi4SigLipProcessor, get_image_size_for_max_num_patches};
use image::{DynamicImage, RgbImage};

#[test]
fn get_image_size_for_max_num_patches_preserves_aspect_ratio_bounds() {
    let (height, width) = get_image_size_for_max_num_patches(200, 400, 16, 256);
    assert_eq!(height % 16, 0);
    assert_eq!(width % 16, 0);
    assert!(height > 0);
    assert!(width > 0);
    assert!((height / 16) * (width / 16) <= 256);
}

#[test]
fn phi4_siglip_processor_clamps_small_images_to_minimum_patch_count() {
    let processor = Phi4SigLipProcessor::new(16, 256, 3600);
    let image = DynamicImage::ImageRgb8(RgbImage::new(64, 64));
    let processed = processor.preprocess(&[image]);

    assert_eq!(processed.len(), 1);
    assert_eq!(processed[0].spatial_shape, (16, 16));
    assert_eq!(
        mlxcel_core::array_shape(&processed[0].pixel_values),
        vec![1, 256, 768]
    );
}

#[test]
fn phi4_siglip_processor_clamps_large_images_to_maximum_patch_count() {
    let processor = Phi4SigLipProcessor::new(16, 256, 3600);
    let image = DynamicImage::ImageRgb8(RgbImage::new(4000, 1000));
    let processed = processor.preprocess(&[image]);

    assert_eq!(processed.len(), 1);
    let (height, width) = processed[0].spatial_shape;
    assert!(height > 0);
    assert!(width > 0);
    assert!((height * width) as usize <= 3600);
}
