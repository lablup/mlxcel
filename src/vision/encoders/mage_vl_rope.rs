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

//! Mage-ViT position and rotary helpers.
//!
//! Three details here look like bugs and are not: the checkpoint was trained
//! with them, so they are reproduced exactly (`VisionRotaryEmbedding`,
//! `rotate_half` and `MageVLVisionPretrainedModel.forward` in the hub's
//! `modeling_mage_vl.py`).
//!
//! 1. The inverse frequencies use `arange(size) / size` as the exponent, not
//!    the conventional `2i / d`, over a 4:6:6 split of `head_dim / 2`
//!    (8 / 12 / 12 for `head_dim = 64`).
//! 2. The per-row table is `concat(freqs, freqs)`, a plain concatenation, so
//!    under the interleaved rotation lanes `2i` and `2i+1` carry different
//!    frequencies.
//! 3. `rotate_half` is interleaved: `(x1, x2, x3, x4) -> (-x2, x1, -x4, x3)`.
//!    `mlxcel_core::fast_rope` offers split-half or traditional pairing with
//!    equal frequencies per pair, neither of which matches, so the rotation is
//!    built from primitives.
//!
//! Used by: `vision::encoders::mage_vl`.

use mlxcel_core::{MlxArray, UniquePtr};

/// Per-row `(t, h, w)` positions for a sequence of images or clips, in the
/// processor's merge-block row order.
///
/// The Qwen2-VL processor emits the rows of each frame block by block: for
/// every `merge x merge` cell (row-major over cells), the cell's patches in
/// row-major order. This mirrors `build_patch_positions` plus
/// `_convert_positions_to_block_layout` in the hub's
/// `video_processing_mage_vl.py`:
///
/// ```text
/// for t in 0..T: for hb in 0..h/m: for wb in 0..w/m: for dh in 0..m: for dw in 0..m:
///     (t, hb*m + dh, wb*m + dw)
/// ```
#[must_use]
pub fn positions_from_grid(grid_thw: &[(i32, i32, i32)], merge: usize) -> Vec<[i32; 3]> {
    let merge = merge.max(1) as i32;
    let total: usize = grid_thw
        .iter()
        .map(|&(t, h, w)| (t.max(0) * h.max(0) * w.max(0)) as usize)
        .sum();
    let mut out = Vec::with_capacity(total);
    for &(t, h, w) in grid_thw {
        for ti in 0..t {
            for hb in 0..h / merge {
                for wb in 0..w / merge {
                    for dh in 0..merge {
                        for dw in 0..merge {
                            out.push([ti, hb * merge + dh, wb * merge + dw]);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse frequencies for the `(t, h, w)` axes.
///
/// `half = head_dim / 2` splits 4:6:6 into sizes `4u / 6u / 6u` with
/// `u = half / 16`; each axis uses `base^(-i / size)` for `i < size`.
#[must_use]
pub fn inv_freqs(head_dim: usize, base: f32) -> [Vec<f32>; 3] {
    let unit = (head_dim / 2) / 16;
    let axis = |size: usize| -> Vec<f32> {
        (0..size)
            .map(|i| 1.0 / base.powf(i as f32 / size as f32))
            .collect()
    };
    [axis(4 * unit), axis(6 * unit), axis(6 * unit)]
}

/// Host-side rotary angle table `[L, head_dim]` for the given positions:
/// `concat(t*inv_t, h*inv_h, w*inv_w)` (width `head_dim / 2`), then that row
/// concatenated with itself.
#[must_use]
pub fn rotary_angles(positions: &[[i32; 3]], inv: &[Vec<f32>; 3]) -> Vec<f32> {
    let half: usize = inv.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(positions.len() * half * 2);
    let mut row = Vec::with_capacity(half);
    for pos in positions {
        row.clear();
        for (axis, freqs) in inv.iter().enumerate() {
            let p = pos[axis] as f32;
            row.extend(freqs.iter().map(|f| p * f));
        }
        out.extend_from_slice(&row);
        out.extend_from_slice(&row);
    }
    out
}

/// Interleaved `rotate_half` over the last axis:
/// `(x1, x2, x3, x4, ...) -> (-x2, x1, -x4, x3, ...)`.
///
/// Reshapes the last axis into `[D/2, 2]`, stacks `(-odd, even)` and flattens
/// back, so it works for any leading shape.
pub fn rotate_half_interleaved(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len();
    let last = shape[ndim - 1];
    let mut paired_shape = shape[..ndim - 1].to_vec();
    paired_shape.push(last / 2);
    paired_shape.push(2);
    let pairs = mlxcel_core::reshape(x, &paired_shape);

    let mut start = vec![0i32; ndim + 1];
    let mut stop = paired_shape.clone();
    stop[ndim] = 1;
    let even = mlxcel_core::slice(&pairs, &start, &stop);
    start[ndim] = 1;
    stop[ndim] = 2;
    let odd = mlxcel_core::slice(&pairs, &start, &stop);

    let neg_odd = mlxcel_core::negative(&odd);
    let rotated = mlxcel_core::concatenate(&neg_odd, &even, ndim as i32);
    mlxcel_core::reshape(&rotated, &shape)
}

/// `x * cos + rotate_half_interleaved(x) * sin`, computed in f32 and cast back
/// to the input dtype (upstream `apply_rotary_pos_emb`).
///
/// `cos` / `sin` are f32 and broadcast against `x` (`[1, 1, L, D]` for a
/// `[1, H, L, D]` tensor).
pub fn apply_rotary(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(x);
    let xf = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let rotated = rotate_half_interleaved(&xf);
    let out = mlxcel_core::add(
        &mlxcel_core::multiply(&xf, cos),
        &mlxcel_core::multiply(&rotated, sin),
    );
    mlxcel_core::astype(&out, dtype)
}

/// Attention windows for one image or clip: upstream splits a clip with more
/// than `frame_windows_size` frames into windows of that many frames (plus a
/// remainder) and attends within each window only. A still image (`t = 1`) is
/// a single window.
#[must_use]
pub fn attention_windows(grid_thw: &[(i32, i32, i32)], frame_windows_size: usize) -> Vec<usize> {
    let fixed = frame_windows_size as i32;
    let mut lengths = Vec::new();
    for &(t, h, w) in grid_thw {
        let frame = (h * w) as usize;
        if fixed > 0 && t > fixed {
            for _ in 0..t / fixed {
                lengths.push(fixed as usize * frame);
            }
            if t % fixed > 0 {
                lengths.push((t % fixed) as usize * frame);
            }
        } else {
            lengths.push(t as usize * frame);
        }
    }
    lengths
}
