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

//! Falcon-OCR positional machinery: 3-D rotary, temporal positions, spatial
//! positions, and the hybrid image/text attention mask.
//!
//! Falcon-OCR splits every head into two equal halves and rotates them with
//! different schemes (checkpoint `rope.py::apply_3d_rotary_emb`):
//!
//! - the **low** half carries a plain 1-D rotary over the *temporal* position,
//!   where a whole image block collapses onto a single temporal index;
//! - the **high** half carries a 2-D "golden" rotary whose per-head frequency
//!   table `freqs_cis_golden` (shape `[n_heads, head_dim/4, 2]`) ships in the
//!   checkpoint and is contracted against each token's `(h, w)` position inside
//!   its image.
//!
//! Both halves use the *interleaved* pair convention (`(x0,x1), (x2,x3), ...`),
//! not the half-split convention, because the reference builds complex numbers
//! with `view_as_complex(x.reshape(..., -1, 2))`.
//!
//! Text tokens get `(h, w) = (0, 0)`, which makes the golden rotation the
//! identity, so the same code path can run over the whole sequence. That is the
//! `mlx-vlm` simplification of the reference's NaN-masked scatter, and it is
//! exactly equivalent.

use mlxcel_core::{MlxArray, UniquePtr};

/// Non-parametric RMSNorm epsilon.
///
/// The reference calls `F.rms_norm(x, (dim,))` with `eps=None`, which PyTorch
/// resolves to `torch.finfo(x.dtype).eps`. The checkpoint is float32, so that
/// is exactly [`f32::EPSILON`]. Note this is *not* `norm_eps` (1e-5): that one
/// belongs to the single parametric `norm` at the end of the stack.
pub const NONPARAM_RMS_EPS: f32 = f32::EPSILON;

/// Token ids that mark image structure inside a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FalconOcrTokenIds {
    /// `<|image|>`, one per 16x16 patch.
    pub img_id: i32,
    /// `<|image_cls|>`, opens an image block.
    pub image_cls_token_id: i32,
    /// `<|end_of_image|>`, closes an image block.
    pub img_end_id: i32,
    /// `<|image_reg_1..4|>`, four register tokens after the CLS token.
    pub image_reg_token_ids: [i32; 4],
}

impl FalconOcrTokenIds {
    /// The five tokens that open an image block, in checkpoint order.
    pub fn block_prefix(&self) -> [i32; 5] {
        [
            self.image_cls_token_id,
            self.image_reg_token_ids[0],
            self.image_reg_token_ids[1],
            self.image_reg_token_ids[2],
            self.image_reg_token_ids[3],
        ]
    }

    /// True when the token must not advance the temporal position.
    ///
    /// Mirrors `processing_falcon_ocr.py::_get_image_token_masks`: the register
    /// tokens, the patch tokens and the closing token all share the temporal
    /// index of the CLS token, which *does* advance.
    fn holds_temporal_position(&self, token: i32) -> bool {
        token == self.img_id
            || token == self.img_end_id
            || self.image_reg_token_ids.contains(&token)
    }
}

/// Temporal rope positions for a prompt.
///
/// Mirrors `processing_falcon_ocr.py::get_pos_thw` for the unpadded,
/// single-sequence case: start a running counter at zero, advance it on every
/// token except the ones inside an image block that follow the CLS token.
pub fn temporal_positions(tokens: &[i32], ids: &FalconOcrTokenIds) -> Vec<i32> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut next = -1i32;
    for &token in tokens {
        if !ids.holds_temporal_position(token) {
            next += 1;
        }
        out.push(next.max(0));
    }
    out
}

/// The offset a decode step adds to the KV-cache position to recover the
/// absolute temporal position.
///
/// An image block consumes `1 + 4 + patches + 1` sequence slots but only one
/// temporal index, so the temporal axis runs *behind* the cache axis and the
/// delta is negative for any prompt that carries an image.
pub fn rope_delta(positions: &[i32]) -> i32 {
    match positions.last() {
        Some(&last) => last + 1 - positions.len() as i32,
        None => 0,
    }
}

/// Per-token `(h, w)` coordinates for the 2-D golden rotary.
///
/// Mirrors `processing_falcon_ocr.py::_compute_image_spatial_positions`: within
/// an image of `rows x cols` patches the coordinates are a centered linspace
/// scaled so the longer side spans a wider range,
/// `h in [-sqrt(rows/cols), +sqrt(rows/cols)]` and
/// `w in [-sqrt(cols/rows), +sqrt(cols/rows)]`, laid out row-major. Non-image
/// tokens get `(0, 0)`.
///
/// `grids` lists `(rows, cols)` per image in prompt order.
pub fn spatial_positions(
    tokens: &[i32],
    ids: &FalconOcrTokenIds,
    grids: &[(i32, i32)],
) -> Vec<f32> {
    let mut coords: Vec<(f32, f32)> = Vec::new();
    for &(rows, cols) in grids {
        let (rows, cols) = (rows.max(1), cols.max(1));
        let ylim = (rows as f32 / cols as f32).sqrt();
        let xlim = (cols as f32 / rows as f32).sqrt();
        for r in 0..rows {
            for c in 0..cols {
                coords.push((
                    linspace_at(-ylim, ylim, rows, r),
                    linspace_at(-xlim, xlim, cols, c),
                ));
            }
        }
    }

    let mut out = vec![0.0f32; tokens.len() * 2];
    let mut next = 0usize;
    for (i, &token) in tokens.iter().enumerate() {
        if token != ids.img_id {
            continue;
        }
        let (h, w) = coords.get(next).copied().unwrap_or((0.0, 0.0));
        out[i * 2] = h;
        out[i * 2 + 1] = w;
        next += 1;
    }
    out
}

/// `torch.linspace(start, stop, n)[i]`, including the `n == 1` degenerate case
/// where PyTorch returns just `start`.
fn linspace_at(start: f32, stop: f32, n: i32, i: i32) -> f32 {
    if n <= 1 {
        return start;
    }
    start + (stop - start) * (i as f32) / ((n - 1) as f32)
}

/// Additive prefill mask: causal over text, fully bidirectional inside each
/// image block.
///
/// Mirrors `attention.py::create_batch_attention_mask` reduced to the
/// single-sequence, unpadded case that mlxcel prefills: the document and
/// left-padding terms are constant there, so only `causal OR same-image`
/// survives. An image block spans `[cls, reg1..4, patches]`; the closing
/// `<|end_of_image|>` is *outside* the bidirectional region because the
/// reference computes membership as `cumsum(soi) - cumsum(eoi) > 0`.
///
/// Returns a `[1, 1, S, S]` f32 array with `0.0` where attention is allowed and
/// a large negative value where it is blocked.
pub fn build_hybrid_mask(tokens: &[i32], ids: &FalconOcrTokenIds) -> UniquePtr<MlxArray> {
    let n = tokens.len();
    // 1-indexed image-block id per token, 0 for tokens outside any block.
    let mut block_of = vec![0i32; n];
    let mut opened = 0i32;
    let mut closed = 0i32;
    for (i, &token) in tokens.iter().enumerate() {
        // The reference builds membership from *inclusive* cumulative sums, so
        // both counters must absorb the token at `i` before the test. That is
        // what puts `<|end_of_image|>` outside the bidirectional region while
        // `<|image_cls|>` stays inside it.
        if token == ids.image_cls_token_id {
            opened += 1;
        }
        if token == ids.img_end_id {
            closed += 1;
        }
        if opened - closed > 0 {
            block_of[i] = opened;
        }
    }

    let blocked = mask_blocked_value();
    let mut data = vec![blocked; n * n];
    for q in 0..n {
        let row = q * n;
        for (kv, slot) in data[row..row + n].iter_mut().enumerate() {
            let causal = q >= kv;
            let same_image = block_of[q] != 0 && block_of[q] == block_of[kv];
            if causal || same_image {
                *slot = 0.0;
            }
        }
    }
    mlxcel_core::from_slice_f32(&data, &[1, 1, n as i32, n as i32])
}

/// The additive value used for blocked positions.
///
/// `f32::MIN` rather than `-inf` so a downcast to bf16/f16 stays finite and a
/// fully-blocked row (which cannot occur here, but is cheap to stay safe about)
/// cannot turn the softmax into NaN.
fn mask_blocked_value() -> f32 {
    f32::MIN
}

/// Rotate the interleaved pairs of `x` by `cos`/`sin`.
///
/// `x` is `[.., D]` and `cos`/`sin` are broadcastable to `[.., D/2]`. Pair `j`
/// is `(x[2j], x[2j+1])`.
pub fn rotate_interleaved(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    let mut shape = mlxcel_core::array_shape(x);
    let dim = *shape
        .last()
        .expect("rotate_interleaved needs a non-empty shape");
    let half = dim / 2;

    let mut pair_shape = shape.clone();
    pair_shape.pop();
    pair_shape.push(half);
    pair_shape.push(2);
    let paired = mlxcel_core::reshape(x, &pair_shape);

    let mut half_shape = shape.clone();
    half_shape.pop();
    half_shape.push(half);
    let even = mlxcel_core::reshape(&mlxcel_core::slice_last_dim(&paired, 0, 1), &half_shape);
    let odd = mlxcel_core::reshape(&mlxcel_core::slice_last_dim(&paired, 1, 2), &half_shape);

    let out_even = mlxcel_core::subtract(
        &mlxcel_core::multiply(&even, cos),
        &mlxcel_core::multiply(&odd, sin),
    );
    let out_odd = mlxcel_core::add(
        &mlxcel_core::multiply(&even, sin),
        &mlxcel_core::multiply(&odd, cos),
    );

    let stacked = mlxcel_core::stack(
        &[
            out_even.as_ref().unwrap() as *const _,
            out_odd.as_ref().unwrap() as *const _,
        ],
        -1,
    );
    shape.pop();
    shape.push(dim);
    mlxcel_core::reshape(&stacked, &shape)
}

/// Inverse frequencies for the 1-D temporal rotary.
///
/// `precompute_freqs_cis(rope_dim, ...)` in `rope.py` builds
/// `1 / theta^(2j / rope_dim)` for `j < rope_dim/2`, where `rope_dim` is
/// `head_dim / 2` because only the low half of each head is rotated this way.
pub fn temporal_inv_freq(rope_dim: usize, theta: f32) -> Vec<f32> {
    (0..rope_dim / 2)
        .map(|j| 1.0 / theta.powf((2 * j) as f32 / rope_dim as f32))
        .collect()
}

/// `cos`/`sin` tables for the 1-D temporal rotary, shaped `[1, 1, L, F]` so
/// they broadcast across batch and heads of a `[B, H, L, D]` tensor.
pub fn temporal_cos_sin(
    positions: &[i32],
    inv_freq: &[f32],
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let l = positions.len();
    let f = inv_freq.len();
    let mut angles = Vec::with_capacity(l * f);
    for &p in positions {
        for &w in inv_freq {
            angles.push(p as f32 * w);
        }
    }
    let theta = mlxcel_core::from_slice_f32(&angles, &[1, 1, l as i32, f as i32]);
    (mlxcel_core::cos(&theta), mlxcel_core::sin(&theta))
}

/// `cos`/`sin` tables for the 2-D golden rotary, shaped `[B, H, L, F]`.
///
/// `pos_hw` is `[B, L, 2]` and `freqs_golden` is `[H, F, 2]`; the reference
/// contracts them as `einsum("bsp,hfp->bshf")`, which for `P == 2` is one
/// broadcast multiply-add per axis.
pub fn golden_cos_sin(
    freqs_golden: &MlxArray,
    pos_hw: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let fshape = mlxcel_core::array_shape(freqs_golden);
    let (heads, freqs) = (fshape[0], fshape[1]);
    let pshape = mlxcel_core::array_shape(pos_hw);
    let (batch, len) = (pshape[0], pshape[1]);

    let f32_dtype = mlxcel_core::dtype::FLOAT32;
    let freqs_f = mlxcel_core::astype(freqs_golden, f32_dtype);
    let pos_f = mlxcel_core::astype(pos_hw, f32_dtype);

    let fh = mlxcel_core::reshape(
        &mlxcel_core::slice_last_dim(&freqs_f, 0, 1),
        &[1, 1, heads, freqs],
    );
    let fw = mlxcel_core::reshape(
        &mlxcel_core::slice_last_dim(&freqs_f, 1, 2),
        &[1, 1, heads, freqs],
    );
    let ph = mlxcel_core::reshape(
        &mlxcel_core::slice_last_dim(&pos_f, 0, 1),
        &[batch, len, 1, 1],
    );
    let pw = mlxcel_core::reshape(
        &mlxcel_core::slice_last_dim(&pos_f, 1, 2),
        &[batch, len, 1, 1],
    );

    let theta = mlxcel_core::add(
        &mlxcel_core::multiply(&ph, &fh),
        &mlxcel_core::multiply(&pw, &fw),
    );
    // [B, L, H, F] -> [B, H, L, F]
    let theta = mlxcel_core::transpose_axes(&theta, &[0, 2, 1, 3]);
    (mlxcel_core::cos(&theta), mlxcel_core::sin(&theta))
}

/// Apply the full 3-D rotary to a `[B, H, L, D]` tensor.
///
/// The low half takes the temporal rotary, the high half the golden rotary
/// (skipped entirely when `golden` is `None`, which is the decode case where
/// every query is a text token and the golden rotation would be the identity).
pub fn apply_3d_rotary(
    x: &MlxArray,
    cos_1d: &MlxArray,
    sin_1d: &MlxArray,
    golden: Option<(&MlxArray, &MlxArray)>,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let dim = *shape
        .last()
        .expect("apply_3d_rotary needs a non-empty shape");
    let half = dim / 2;

    let low = mlxcel_core::slice_last_dim(x, 0, half);
    let high = mlxcel_core::slice_last_dim(x, half, dim);

    let low = rotate_interleaved(&low, cos_1d, sin_1d);
    let high = match golden {
        Some((cos_2d, sin_2d)) => rotate_interleaved(&high, cos_2d, sin_2d),
        None => high,
    };

    mlxcel_core::concatenate(&low, &high, -1)
}

#[cfg(test)]
#[path = "falcon_ocr_rope_tests.rs"]
mod tests;
