// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Conservative live-allocation admission for Objects preprocessing only.
use anyhow::{Result, ensure};

const MAX_PREPROCESSING_BYTES: usize = 1024 * 1024 * 1024;
fn product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or_else(|| anyhow::anyhow!("Objects preprocessing allocation overflow"))
}
fn add(total: &mut usize, bytes: usize) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| anyhow::anyhow!("Objects preprocessing allocation overflow"))?;
    Ok(())
}

/// Admit bounded source conversion buffers before extracting a mask or RGB.
pub(super) fn source_preflight(h: usize, w: usize) -> Result<()> {
    let bytes = product(&[h, w, 56])?;
    ensure!(
        bytes <= MAX_PREPROCESSING_BYTES,
        "Objects preprocessing source buffers exceed the {MAX_PREPROCESSING_BYTES}-byte allocation guard"
    );
    Ok(())
}

/// Bounds all graph-visible buffers conservatively, including buffers that may
/// be released earlier by MLX. Model weights/backend allocator scratch are not
/// preprocessing buffers and remain governed by their own admission policies.
pub(super) fn preflight(
    h: usize,
    w: usize,
    bounds: [i32; 4],
    size: usize,
    points: Option<[usize; 2]>,
) -> Result<usize> {
    ensure!(
        h > 0
            && w > 0
            && size > 0
            && h <= i32::MAX as usize
            && w <= i32::MAX as usize
            && size <= i32::MAX as usize,
        "invalid Objects preprocessing dimensions"
    );
    let [left, top, right, bottom] = bounds.map(i64::from);
    ensure!(
        right > left && bottom > top,
        "invalid Objects crop dimensions"
    );
    let crop_h = usize::try_from(bottom - top)?;
    let crop_w = usize::try_from(right - left)?;
    let pad_h = h
        .checked_add(usize::try_from((-top).max(0))?)
        .and_then(|n| n.checked_add(usize::try_from((bottom - h as i64).max(0)).ok()?))
        .ok_or_else(|| anyhow::anyhow!("Objects crop padding overflow"))?;
    let pad_w = w
        .checked_add(usize::try_from((-left).max(0))?)
        .and_then(|n| n.checked_add(usize::try_from((right - w as i64).max(0)).ok()?))
        .ok_or_else(|| anyhow::anyhow!("Objects crop padding overflow"))?;
    // Decoded RGBA32F, host/device RGB32F, mask host/device and conversion data.
    let mut bytes = product(&[h, w, 56])?;
    let mut views = vec![(3, false), (1, true)];
    if let Some([ph, pw]) = points {
        ensure!(
            ph > 0 && pw > 0 && ph <= i32::MAX as usize && pw <= i32::MAX as usize,
            "invalid Objects point-map dimensions"
        );
        // Source and F32 cast, nearest intermediate/output, host extraction,
        // normalized maps, subtraction/division and median temporary vectors.
        add(&mut bytes, product(&[ph, pw, 3, 4, 2])?)?;
        add(&mut bytes, product(&[h, pw, 3, 4])?)?;
        add(&mut bytes, product(&[h, w, 3, 4, 6])?)?;
        add(&mut bytes, product(&[h + w, 4, 2])?)?;
        views.push((3, true));
    }
    for (channels, nearest) in views {
        // crop() first pads the whole source, then slices it; count both.
        add(&mut bytes, product(&[pad_h, pad_w, channels, 4])?)?;
        add(&mut bytes, product(&[crop_h, crop_w, channels, 4])?)?;
        for side in [crop_h.max(crop_w), h.max(w)] {
            add(&mut bytes, product(&[side, side, channels, 4])?)?;
            if nearest {
                // Both take outputs plus host/device indices for both axes.
                add(&mut bytes, product(&[size, side, channels, 4])?)?;
                add(&mut bytes, product(&[size, size, channels, 4])?)?;
                add(&mut bytes, product(&[size, 4, 4])?)?;
            } else {
                // Exactly the tap count used by bilinear_weights_1d. Square
                // inputs/outputs make these sizes equal for either axis order.
                let scale = side as f32 / size as f32;
                let taps = if scale > 1.0 {
                    (2.0 * scale).ceil() as usize
                } else {
                    2
                };
                // Indices+weights, both axes, each retained on host and device.
                add(&mut bytes, product(&[size, taps, 4, 8])?)?;
                // Gather and weighted product coexist for each contraction.
                add(&mut bytes, product(&[size, taps, side, channels, 4, 2])?)?;
                add(&mut bytes, product(&[size, taps, size, channels, 4, 2])?)?;
                // Axis outputs and possible contiguous transpose copies.
                add(&mut bytes, product(&[size, side, channels, 4, 2])?)?;
                add(&mut bytes, product(&[size, size, channels, 4, 2])?)?;
            }
        }
    }
    ensure!(
        bytes <= MAX_PREPROCESSING_BYTES,
        "Objects preprocessing requires a conservative {bytes} live bytes, exceeding the {MAX_PREPROCESSING_BYTES}-byte allocation guard"
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn objects_preprocessing_budget_admits_published_geometry_and_rejects_extreme_aspect() {
        let bytes = preflight(448, 672, [99, 249, 183, 333], 518, Some([448, 672])).unwrap();
        assert_eq!(bytes, 167_447_840);
        assert!(source_preflight(2, 16_777_216).is_err());
        // Exactly 32Mi pixels, yet square padding would require petabytes.
        assert_eq!(2 * 16_777_216, 32 * 1024 * 1024);
        let error = preflight(
            2,
            16_777_216,
            [-1_677_721, -10_066_328, 18_454_937, 10_066_330],
            518,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("preprocessing"));
        assert!(preflight(usize::MAX, 2, [0, 0, 2, 2], 518, None).is_err());
    }
}
