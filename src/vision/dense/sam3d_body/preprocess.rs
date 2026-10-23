// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! OpenCV INTER_LINEAR affine crop, including quantized uint8 interpolation.
use super::{config::Config, layers::Array};
use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use mlxcel_core as mx;
use serde_json::{Map, Value};
#[derive(Clone, Debug)]
pub struct InputGeometry {
    pub bbox: [f64; 4],
    pub cam_int: [f32; 9],
    pub height: usize,
    pub width: usize,
}
impl InputGeometry {
    pub fn parse(image: &DynamicImage, options: &Map<String, Value>) -> Result<Self> {
        ensure!(
            options
                .keys()
                .all(|k| ["bbox", "cam_int", "task"].contains(&k.as_str())),
            "unsupported Body option"
        );
        let (width, height) = (image.width() as usize, image.height() as usize);
        ensure!(
            width > 0
                && height > 0
                && width
                    .checked_mul(height)
                    .is_some_and(|n| n <= 32 * 1024 * 1024),
            "Body input exceeds 32 Mi pixels or is empty"
        );
        let mut bbox = [0.0, 0.0, width as f64, height as f64];
        if let Some(v) = options.get("bbox") {
            bbox = parse_numbers::<4>(v, "bbox")?;
        }
        ensure!(
            bbox.iter().all(|v| v.is_finite()) && bbox[2] > bbox[0] && bbox[3] > bbox[1],
            "bbox must be finite and have positive area"
        );
        ensure!(
            bbox[2] > 0.0 && bbox[3] > 0.0 && bbox[0] < width as f64 && bbox[1] < height as f64,
            "bbox is fully outside the image"
        );
        // Bound integer affine coordinates before OpenCV-compatible fixed-point conversion.
        ensure!(
            bbox.iter().all(|v| v.abs() <= 1_000_000.0),
            "bbox exceeds supported affine coordinate range"
        );
        let f = ((width as f64).powi(2) + (height as f64).powi(2)).sqrt() as f32;
        let mut cam_int = [
            f,
            0.0,
            width as f32 / 2.0,
            0.0,
            f,
            height as f32 / 2.0,
            0.0,
            0.0,
            1.0,
        ];
        if let Some(v) = options.get("cam_int") {
            let flat = if v
                .as_array()
                .is_some_and(|a| a.len() == 3 && a.iter().all(Value::is_array))
            {
                let rows = v.as_array().context("cam_int rows")?;
                ensure!(
                    rows.iter()
                        .all(|r| r.as_array().is_some_and(|a| a.len() == 3)),
                    "cam_int must be 3 by 3"
                );
                Value::Array(
                    rows.iter()
                        .flat_map(|r| r.as_array().into_iter().flatten().cloned())
                        .collect(),
                )
            } else {
                v.clone()
            };
            cam_int = parse_numbers::<9>(&flat, "cam_int")?.map(|v| v as f32);
        }
        ensure!(
            cam_int.iter().all(|v| v.is_finite()) && cam_int[0] > 0.0 && cam_int[4] > 0.0,
            "camera intrinsics must be finite with positive focal lengths"
        );
        let m = cam_int.map(f64::from);
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
        ensure!(
            det.is_finite() && det != 0.0,
            "camera intrinsics must be invertible"
        );
        Ok(Self {
            bbox,
            cam_int,
            height,
            width,
        })
    }
    pub fn cliff(&self) -> [f32; 3] {
        let b = self.bbox.map(|v| v as f32);
        let f = self.cam_int[0];
        [
            ((b[0] + b[2]) / 2.0 - self.width as f32 / 2.0) / f,
            ((b[1] + b[3]) / 2.0 - self.height as f32 / 2.0) / f,
            (b[2] - b[0]) * 1.25 / f,
        ]
    }
}
fn parse_numbers<const N: usize>(v: &Value, name: &str) -> Result<[f64; N]> {
    let a = v
        .as_array()
        .with_context(|| format!("{name} must be an array"))?;
    ensure!(a.len() == N, "{name} must contain {N} numbers");
    let mut out = [0.0; N];
    for (dst, src) in out.iter_mut().zip(a) {
        *dst = src
            .as_f64()
            .with_context(|| format!("{name} must contain numbers"))?;
    }
    Ok(out)
}
/// Axis-aligned inverse of cv2.getAffineTransform on the source's F32 triangle.
pub fn inverse_affine(b: [f32; 4], size: [usize; 2]) -> [f64; 4] {
    let cx = (b[0] + b[2]) / 2.0;
    let cy = (b[1] + b[3]) / 2.0;
    let half = f64::from((b[2] - b[0]).max(b[3] - b[1]) * 1.2) / 2.0;
    let rx = (f64::from(cx) + half) as f32;
    let dy = (f64::from(cy) + half) as f32;
    let sx = (f64::from(rx) - f64::from(cx)) / (size[1] as f64 / 2.0);
    let sy = (f64::from(dy) - f64::from(cy)) / (size[0] as f64 / 2.0);
    [
        sx,
        sy,
        f64::from(cx) - sx * (size[1] as f64 / 2.0),
        f64::from(cy) - sy * (size[0] as f64 / 2.0),
    ]
}
pub fn crop(image: &DynamicImage, g: &InputGeometry, c: &Config) -> Result<Vec<u8>> {
    ensure!(
        c.image_size.iter().all(|&v| v > 0 && v <= 1024),
        "invalid affine output dimensions"
    );
    let [h, w] = c.image_size;
    let n = h
        .checked_mul(w)
        .and_then(|v| v.checked_mul(3))
        .context("crop size overflow")?;
    let rgb = image.to_rgb8();
    let [sx, sy, tx, ty] = inverse_affine(g.bbox.map(|v| v as f32), c.image_size);
    let mut out = vec![0u8; n];
    // OpenCV warpAffine uses AB_BITS=10, INTER_BITS=5 and ties-to-even cvRound.
    for y in 0..h {
        let yf = (((sy * y as f64 + ty) * 1024.0).round_ties_even() as i64 + 16) >> 5;
        let iy = yf >> 5;
        let fy = yf & 31;
        let base = (tx * 1024.0).round_ties_even() as i64 + 16;
        for x in 0..w {
            let xf = (base + (sx * x as f64 * 1024.0).round_ties_even() as i64) >> 5;
            let ix = xf >> 5;
            let fx = xf & 31;
            let weights = [
                (32 - fx) * (32 - fy),
                fx * (32 - fy),
                (32 - fx) * fy,
                fx * fy,
            ];
            for channel in 0..3 {
                let mut total = 0i64;
                for (i, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                    let (px, py) = (ix + dx, iy + dy);
                    if px >= 0 && py >= 0 && px < g.width as i64 && py < g.height as i64 {
                        total +=
                            i64::from(rgb.get_pixel(px as u32, py as u32)[channel]) * weights[i];
                    }
                }
                out[(y * w + x) * 3 + channel] = ((total + 512) >> 10).clamp(0, 255) as u8;
            }
        }
    }
    Ok(out)
}
pub fn prepare(image: &DynamicImage, g: &InputGeometry, c: &Config) -> Result<Array> {
    let bytes = crop(image, g, c)?;
    let values: Vec<f32> = bytes
        .into_iter()
        .enumerate()
        .map(|(i, v)| ((f32::from(v) / 255.0) - c.image_mean[i % 3]) / c.image_std[i % 3])
        .collect();
    Ok(mx::from_slice_f32(
        &values,
        &[1, c.image_size[0] as i32, c.image_size[1] as i32, 3],
    ))
}
#[cfg(test)]
#[path = "preprocess_tests.rs"]
mod tests;
