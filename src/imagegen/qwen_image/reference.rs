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

//! Edit-reference image handling with Pillow's integer arithmetic.
//!
//! Upstream opens every reference as RGBA and resizes it with
//! `Image.resize(size, LANCZOS)`, which for RGBA premultiplies the color by
//! alpha (`RGBa`), resamples all four bands, and divides back. The VAE then
//! encodes all four channels, and the conditioner composites the resized
//! RGBA onto white through its alpha (`Image.paste(rgba, mask=alpha)`).
//!
//! The arithmetic below is Pillow 13's: `DIV255` for the premultiply and the
//! paste blend, a truncating `255 * c / a` for the un-premultiply, and the
//! 22-bit fixed-point two-pass resampler with Pillow's `lanczos_filter`
//! (`sinc(x) * sinc(x / 3)` on `[-3, 3)`). The pass order follows the same
//! rule as [`crate::vision::dense::preprocess::resize_pillow_rgb8`].

use anyhow::{Result, ensure};
use image::{RgbImage, RgbaImage};

const PRECISION_BITS: i32 = 22;
const ROUNDING: i64 = 1 << (PRECISION_BITS - 1);

/// Pillow `DIV255`: `round(a / 255)` for `a` in `[0, 255 * 255]`.
fn div255(a: u32) -> u8 {
    let t = a + 128;
    (((t >> 8) + t) >> 8) as u8
}

/// `Image.new("RGB", size, "white").paste(rgba, mask=rgba.getchannel("A"))`.
#[must_use]
pub fn composite_on_white(image: &RgbaImage) -> RgbImage {
    let (w, h) = image.dimensions();
    let mut out = RgbImage::new(w, h);
    for (dst, src) in out.pixels_mut().zip(image.pixels()) {
        let a = u32::from(src.0[3]);
        for c in 0..3 {
            dst.0[c] = div255(255 * (255 - a) + u32::from(src.0[c]) * a);
        }
    }
    out
}

/// `RGBA` to `RGBa`.
fn premultiply(image: &RgbaImage) -> Vec<u8> {
    image
        .pixels()
        .flat_map(|p| {
            let a = u32::from(p.0[3]);
            [
                div255(u32::from(p.0[0]) * a),
                div255(u32::from(p.0[1]) * a),
                div255(u32::from(p.0[2]) * a),
                p.0[3],
            ]
        })
        .collect()
}

/// `RGBa` to `RGBA`.
fn unpremultiply(data: &[u8], width: u32, height: u32) -> Result<RgbaImage> {
    let pixels = data
        .chunks_exact(4)
        .flat_map(|p| {
            let a = u32::from(p[3]);
            let channel = |c: u8| {
                if a == 0 || a == 255 {
                    c
                } else {
                    (255 * u32::from(c) / a).min(255) as u8
                }
            };
            [channel(p[0]), channel(p[1]), channel(p[2]), p[3]]
        })
        .collect();
    RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow::anyhow!("RGBA buffer does not match {width}x{height}"))
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * std::f64::consts::PI;
        x.sin() / x
    }
}

/// Pillow `lanczos_filter`.
fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Pillow `precompute_coeffs` + `normalize_coeffs_8bpc` for Lanczos:
/// `(xmin, fixed-point weights)` per output position.
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 3.0 * filter_scale;
    let ss = 1.0 / filter_scale;
    (0..output)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(input as i64) as usize - xmin;
            let k: Vec<f64> = (0..xmax)
                .map(|x| lanczos((x as f64 + xmin as f64 - center + 0.5) * ss))
                .collect();
            let total: f64 = k.iter().sum();
            let fixed = k
                .iter()
                .map(|&v| {
                    let v = if total == 0.0 { v } else { v / total };
                    let scaled = v * f64::from(1 << PRECISION_BITS);
                    if v < 0.0 {
                        (-0.5 + scaled) as i64
                    } else {
                        (0.5 + scaled) as i64
                    }
                })
                .collect();
            (xmin, fixed)
        })
        .collect()
}

/// One resampling pass over a `[rows, cols, 4]` byte buffer along `axis`
/// (0 = rows, 1 = cols).
fn pass(data: &[u8], rows: usize, cols: usize, out_len: usize, axis: usize) -> Vec<u8> {
    let (in_len, other) = if axis == 0 {
        (rows, cols)
    } else {
        (cols, rows)
    };
    let table = coefficients(in_len, out_len);
    let (out_rows, out_cols) = if axis == 0 {
        (out_len, cols)
    } else {
        (rows, out_len)
    };
    let mut out = vec![0u8; out_rows * out_cols * 4];
    for (o, (xmin, weights)) in table.iter().enumerate() {
        for j in 0..other {
            for c in 0..4 {
                let mut sum = ROUNDING;
                for (t, w) in weights.iter().enumerate() {
                    let i = xmin + t;
                    let index = if axis == 0 {
                        (i * cols + j) * 4 + c
                    } else {
                        (j * cols + i) * 4 + c
                    };
                    sum += i64::from(data[index]) * w;
                }
                let index = if axis == 0 {
                    (o * cols + j) * 4 + c
                } else {
                    (j * out_cols + o) * 4 + c
                };
                out[index] = (sum >> PRECISION_BITS).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// `image.resize((width, height), Image.Resampling.LANCZOS)` of an RGBA
/// image, Pillow's premultiplied path included. The same size is a copy.
pub fn resize_lanczos_rgba(image: &RgbaImage, width: u32, height: u32) -> Result<RgbaImage> {
    ensure!(
        width > 0 && height > 0,
        "resize dimensions must be positive"
    );
    let (iw, ih) = image.dimensions();
    if (iw, ih) == (width, height) {
        return Ok(image.clone());
    }
    let (w, h, ow, oh) = (iw as usize, ih as usize, width as usize, height as usize);
    let data = premultiply(image);
    let height_reduction = ih as i64 - height as i64;
    let width_reduction = iw as i64 - width as i64;
    let horizontal_first = !(height_reduction > 0 && height_reduction > width_reduction * 2);
    let resized = match (ow != w, oh != h, horizontal_first) {
        (true, false, _) => pass(&data, h, w, ow, 1),
        (false, true, _) => pass(&data, h, w, oh, 0),
        (true, true, true) => pass(&pass(&data, h, w, ow, 1), h, ow, oh, 0),
        (true, true, false) => pass(&pass(&data, h, w, oh, 0), oh, w, ow, 1),
        (false, false, _) => data,
    };
    unpremultiply(&resized, width, height)
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
