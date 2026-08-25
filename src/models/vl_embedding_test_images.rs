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

//! Deterministic synthetic images for the vision-language embedding gates.
//!
//! The cross-modal gate needs two pictures a caption can be ranked against,
//! and the repository's one image fixture is a flat orange square. Drawing the
//! pair here keeps the gate reproducible byte for byte, adds no binary
//! fixture, and lets a test pick its own aspect ratio, which is what exercises
//! the dynamic tiling. The two scenes are deliberately far apart in content:
//! a bar chart on white, and a beach with sky, sun, sea and sand.

/// A white canvas carrying five black bars of increasing height over a pair
/// of axes: a chart, not a photograph.
pub(crate) fn bar_chart(width: u32, height: u32) -> image::DynamicImage {
    let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([255, 255, 255]));
    let axis = image::Rgb([20, 20, 20]);
    let bar = image::Rgb([40, 70, 190]);

    let baseline = height * 6 / 7;
    let left = width / 8;
    for x in left..width {
        for thickness in 0..(height / 100).max(2) {
            let y = baseline + thickness;
            if y < height {
                canvas.put_pixel(x, y, axis);
            }
        }
    }
    for y in height / 10..=baseline {
        for thickness in 0..(width / 100).max(2) {
            let x = left + thickness;
            if x < width {
                canvas.put_pixel(x, y, axis);
            }
        }
    }

    let bars = 5u32;
    let span = width - left - width / 12;
    let slot = (span / bars).max(1);
    for index in 0..bars {
        let bar_width = (slot * 3 / 5).max(1);
        let x0 = left + slot * index + slot / 5;
        let bar_height = (baseline - height / 10) * (index + 2) / (bars + 2);
        for x in x0..(x0 + bar_width).min(width) {
            for y in (baseline - bar_height)..baseline {
                canvas.put_pixel(x, y, bar);
            }
        }
    }
    image::DynamicImage::ImageRgb8(canvas)
}

/// A beach: blue sky with a sun over a darker sea band and a sand band.
pub(crate) fn beach(width: u32, height: u32) -> image::DynamicImage {
    let mut canvas = image::RgbImage::new(width, height);
    let horizon = height * 5 / 9;
    let shore = height * 7 / 10;
    for y in 0..height {
        for x in 0..width {
            let pixel = if y < horizon {
                let shade = 40 + (y * 120 / horizon.max(1)) as u8;
                image::Rgb([shade / 2, shade, 235])
            } else if y < shore {
                image::Rgb([12, 74, 140])
            } else {
                let shade = 200 + ((height - y) * 40 / height.max(1)) as u8;
                image::Rgb([shade, shade - 30, 150])
            };
            canvas.put_pixel(x, y, pixel);
        }
    }

    let sun = (width.min(height) / 9).max(3) as i64;
    let (cx, cy) = ((width * 3 / 4) as i64, (height / 5) as i64);
    for dy in -sun..=sun {
        for dx in -sun..=sun {
            if dx * dx + dy * dy > sun * sun {
                continue;
            }
            let (x, y) = (cx + dx, cy + dy);
            if x >= 0 && y >= 0 && (x as u32) < width && (y as u32) < height {
                canvas.put_pixel(x as u32, y as u32, image::Rgb([255, 236, 130]));
            }
        }
    }
    image::DynamicImage::ImageRgb8(canvas)
}
