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

//! DINOv2's Pillow-compatible RGB8 bicubic preprocessing adapter.

use anyhow::Result;
use image::RgbImage;

use super::preprocess::{PillowResizeFilter, resize_pillow_rgb8};

pub(super) fn resize_pil_bicubic_rgb(
    image: &RgbImage,
    width: u32,
    height: u32,
) -> Result<RgbImage> {
    resize_pillow_rgb8(image, width, height, PillowResizeFilter::Bicubic)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pillow_bicubic_uint8_rounding() {
        let input = (0..3 * 4 * 3)
            .map(|value| (value * 7 + 11) as u8)
            .collect::<Vec<_>>();
        let input = RgbImage::from_raw(4, 3, input).unwrap();
        let output = resize_pil_bicubic_rgb(&input, 7, 5).unwrap();
        assert_eq!(
            output.as_raw(),
            &[
                3, 10, 17, 11, 18, 25, 24, 31, 38, 37, 44, 51, 49, 56, 63, 62, 69, 76, 70, 77, 84,
                35, 42, 49, 43, 50, 57, 56, 63, 70, 69, 76, 82, 81, 88, 95, 94, 101, 116, 102, 109,
                127, 93, 100, 107, 101, 108, 115, 114, 121, 128, 127, 134, 141, 139, 146, 153, 152,
                159, 166, 160, 167, 174, 151, 158, 165, 159, 166, 173, 172, 179, 186, 185, 192,
                210, 197, 204, 204, 210, 217, 109, 218, 225, 65, 183, 190, 197, 191, 198, 205, 204,
                211, 218, 217, 224, 249, 229, 236, 230, 242, 249, 65, 250, 255, 0,
            ]
        );
    }
}
