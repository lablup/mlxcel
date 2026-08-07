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

//! Location-token arithmetic. No checkpoint needed; the expected values are
//! computed by hand from `(bin + 0.5) * size / 1000`.

use super::*;

/// Ids pinned against `models/Florence-2-base-ft-bf16/added_tokens.json`:
/// 1000 location tokens occupying 50269..=51268 with no gaps.
#[test]
fn florence2_loc_tokens_are_contiguous() {
    assert_eq!(florence2_loc_token_id(0), Some(50269));
    assert_eq!(florence2_loc_token_id(1), Some(50270));
    assert_eq!(florence2_loc_token_id(999), Some(51268));
    // 1000 bins exactly: `<loc_1000>` is not a token.
    assert_eq!(florence2_loc_token_id(1000), None);
    assert_eq!(florence2_loc_token_id(u32::MAX), None);
    for n in 0..LOC_BINS as u32 {
        assert_eq!(
            florence2_loc_token_id(n),
            Some(FLORENCE2_LOC_TOKEN_BASE + n)
        );
    }
}

#[test]
fn dequantize_box_uses_bin_centers() {
    // 1000x1000 image: one pixel per bin, so bin n is centered at n + 0.5.
    let size = Florence2ImageSize::new(1000, 1000);
    let bbox = dequantize_box([0, 10, 500, 999], size);
    assert_eq!(bbox.to_array(), [0.5, 10.5, 500.5, 999.5]);
}

#[test]
fn dequantize_box_scales_axes_independently() {
    // A non-square image is the case a single shared scale would get wrong:
    // x must scale by 640/1000 and y by 480/1000.
    let size = Florence2ImageSize::new(640, 480);
    let bbox = dequantize_box([100, 100, 900, 900], size);
    assert!((bbox.xmin - 64.32).abs() < 1e-4, "got {}", bbox.xmin);
    assert!((bbox.ymin - 48.24).abs() < 1e-4, "got {}", bbox.ymin);
    assert!((bbox.xmax - 576.32).abs() < 1e-3, "got {}", bbox.xmax);
    assert!((bbox.ymax - 432.24).abs() < 1e-3, "got {}", bbox.ymax);
}

#[test]
fn dequantize_coordinates_alternates_x_and_y() {
    let size = Florence2ImageSize::new(1000, 2000);
    // (0, 0), (10, 10): x keeps the 1x scale, y doubles.
    let out = dequantize_coordinates(&[0, 0, 10, 10], size);
    assert_eq!(out, vec![0.5, 1.0, 10.5, 21.0]);
}

/// An odd-length run is the caller's problem, so the dequantizer simply drops
/// the unpaired tail rather than inventing a partner for it.
#[test]
fn dequantize_coordinates_drops_unpaired_tail() {
    let size = Florence2ImageSize::new(1000, 1000);
    assert_eq!(dequantize_coordinates(&[0, 0, 4], size), vec![0.5, 0.5]);
    assert!(dequantize_coordinates(&[], size).is_empty());
}

#[test]
fn quantize_is_the_inverse_of_dequantize() {
    let size = Florence2ImageSize::new(640, 480);
    for bins in [[0, 0, 999, 999], [52, 332, 932, 774], [1, 2, 3, 4]] {
        let bbox = dequantize_box(bins, size);
        assert_eq!(
            bbox.quantize(size),
            bins,
            "dequantize then quantize must return the original bins"
        );
    }
}

#[test]
fn quantize_clamps_out_of_range_pixels() {
    let size = Florence2ImageSize::new(100, 100);
    // Negative and past-the-edge coordinates land on the end bins rather than
    // producing a token id that does not exist.
    let bbox = Florence2BoundingBox {
        xmin: -50.0,
        ymin: -1.0,
        xmax: 1000.0,
        ymax: 100.0,
    };
    assert_eq!(bbox.quantize(size), [0, 0, 999, 999]);
}

#[test]
fn region_prompt_is_four_location_tokens() {
    let size = Florence2ImageSize::new(1000, 1000);
    let bbox = dequantize_box([52, 332, 932, 774], size);
    assert_eq!(
        bbox.to_region_prompt(size),
        "<loc_52><loc_332><loc_932><loc_774>"
    );
}

/// A degenerate size must not divide by zero on the way to a bin index.
#[test]
fn quantize_survives_a_zero_size() {
    let size = Florence2ImageSize::new(0, 0);
    let bbox = Florence2BoundingBox {
        xmin: 1.0,
        ymin: 2.0,
        xmax: 3.0,
        ymax: 4.0,
    };
    assert_eq!(bbox.quantize(size), [0, 0, 0, 0]);
}
