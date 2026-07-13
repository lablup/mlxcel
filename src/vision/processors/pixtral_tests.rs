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

use super::*;
use crate::vision::processors::ImageProcessor;

fn pixtral() -> PixtralProcessor {
    // pixtral-12b-4bit: patch 16, spatial_merge 1, longest_edge 1024.
    PixtralProcessor::new(16, 1, 1024)
}

fn mistral3() -> PixtralProcessor {
    // mistral-small-3.1-24b-4bit: patch 14, spatial_merge 2, longest_edge 1540.
    PixtralProcessor::new(14, 2, 1540)
}

fn solid_image(w: u32, h: u32) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        w,
        h,
        image::Rgb([128, 64, 200]),
    ))
}

#[test]
fn factor_is_patch_times_merge() {
    assert_eq!(pixtral().factor(), 16);
    assert_eq!(mistral3().factor(), 28);
}

#[test]
fn small_image_is_never_upscaled() {
    // 32x48 is far below longest_edge; ratio <= 1 so no scaling, only the
    // round-up to whole patches applies.
    let (th, tw) = pixtral().target_size(32, 48);
    assert_eq!((th, tw), (32, 48));

    // A tiny Mistral3 image rounds up to the 28px factor, never to longest_edge.
    let (th, tw) = mistral3().target_size(20, 10);
    assert_eq!((th, tw), (28, 28));
}

#[test]
fn square_within_edge_keeps_aspect() {
    let (th, tw) = pixtral().target_size(512, 512);
    assert_eq!((th, tw), (512, 512));
    // Regression parity: a square image still produces a square patch grid, so
    // the only behavioral change vs. the old fixed-square path is the added
    // row-structure tokens, not a different feature count.
    assert_eq!(pixtral().token_grid(th, tw), (32, 32));
}

#[test]
fn wide_non_square_preserves_ratio() {
    // 1024x256 fits the longest edge exactly; the wide aspect ratio survives.
    let (th, tw) = pixtral().target_size(1024, 256);
    assert_eq!((th, tw), (1024, 256));
    assert_eq!(pixtral().token_grid(th, tw), (64, 16));
    assert_eq!(pixtral().patch_grid(th, tw), (64, 16));
}

#[test]
fn oversized_image_is_downscaled_by_longest_edge() {
    // 2048x1024 -> ratio 2.0 -> 1024x512, both multiples of 16.
    let (th, tw) = pixtral().target_size(2048, 1024);
    assert_eq!((th, tw), (1024, 512));
    assert_eq!(pixtral().token_grid(th, tw), (64, 32));
}

#[test]
fn dimensions_round_up_to_factor_and_stay_within_edge() {
    let procs = [pixtral(), mistral3()];
    let sizes = [
        (1000usize, 500usize),
        (777, 333),
        (1600, 900),
        (3000, 100),
        (1, 1),
        (1540, 1541),
    ];
    for p in &procs {
        let factor = p.factor();
        for &(h, w) in &sizes {
            let (th, tw) = p.target_size(h, w);
            // Whole multiples of the merged-patch factor.
            assert_eq!(th % factor, 0, "target_h {th} not multiple of {factor}");
            assert_eq!(tw % factor, 0, "target_w {tw} not multiple of {factor}");
            // Never exceeds the longest edge (both configs have longest_edge
            // itself a multiple of the factor).
            assert!(th <= p.longest_edge, "target_h {th} > {}", p.longest_edge);
            assert!(tw <= p.longest_edge, "target_w {tw} > {}", p.longest_edge);
            // Pre-merge patch grid divides evenly by the merge size so no row is
            // dropped by the connector's floor division.
            let (ph, pw) = p.patch_grid(th, tw);
            assert_eq!(ph % p.spatial_merge_size, 0);
            assert_eq!(pw % p.spatial_merge_size, 0);
            // token grid == patch grid / merge on both axes.
            let (tkh, tkw) = p.token_grid(th, tw);
            assert_eq!(tkh, ph / p.spatial_merge_size);
            assert_eq!(tkw, pw / p.spatial_merge_size);
        }
    }
}

#[test]
fn mistral3_partial_merged_row_is_not_dropped() {
    // A height that is a whole number of patches but NOT a whole number of
    // merged patches would lose a row if we rounded to patch_size alone. The
    // factor round-up (28) prevents that: patch grid stays even.
    let m = mistral3();
    let (th, tw) = m.target_size(15 * 14, 3 * 14); // 210 x 42
    let (ph, pw) = m.patch_grid(th, tw);
    assert_eq!(ph % 2, 0);
    assert_eq!(pw % 2, 0);
    let (tkh, tkw) = m.token_grid(th, tw);
    assert_eq!(tkh, ph / 2);
    assert_eq!(tkw, pw / 2);
}

#[test]
fn clip_normalization_constants() {
    let p = pixtral();
    assert_eq!(p.mean, PIXTRAL_IMAGE_MEAN);
    assert_eq!(p.std, PIXTRAL_IMAGE_STD);
}

#[test]
fn preprocess_one_reports_matching_geometry_and_shape() {
    let p = mistral3();
    let img = solid_image(1000, 600); // width x height
    let pre = p.preprocess_one(&img);

    let (th, tw) = p.target_size(600, 1000); // (h, w)
    assert_eq!(p.patch_grid(th, tw), (pre.patches_h, pre.patches_w));
    assert_eq!(p.token_grid(th, tw), (pre.tokens_h, pre.tokens_w));

    let shape = mlxcel_core::array_shape(&pre.pixel_values);
    assert_eq!(shape, vec![1, 3, th as i32, tw as i32]);
}

#[test]
fn trait_preprocess_single_image_matches_preprocess_one() {
    let p = pixtral();
    let img = solid_image(300, 700);
    let via_trait = ImageProcessor::preprocess(&p, std::slice::from_ref(&img));
    let shape = mlxcel_core::array_shape(&via_trait);
    let (th, tw) = p.target_size(700, 300);
    assert_eq!(shape, vec![1, 3, th as i32, tw as i32]);
}
