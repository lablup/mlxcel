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

//! DeepSeek-V4 proportional RoPE (`DeepseekV4RoPE` in the reference).
//!
//! Three properties distinguish it from the plain Yarn RoPE the other
//! DeepSeek families use, and each one is load-bearing:
//!
//! * **Full-head rotation with an `inf` prefix.** The reference calls
//!   `mx.fast.rope(x, head_dim, traditional=True, freqs=...)` over the WHOLE
//!   head (512 for attention, 128 for the indexer) and pads the frequency
//!   table with `inf` for the leading `(head_dim - qk_rope_head_dim) / 2`
//!   pairs, so those pairs rotate by `pos / inf == 0` and only the trailing
//!   `qk_rope_head_dim` lanes actually rotate. The same trick backs
//!   `compiled_proportional_rope` (Gemma 4), so `mx::fast::rope` handling an
//!   `inf` frequency is an established contract in this tree.
//! * **Inverse rotation for the attention output.** V4 un-rotates the
//!   attention output (`self.rope(out, offset, inverse=True)`); the reference
//!   negates the frequency table for the inverse table, which negates every
//!   angle. Skipping the inverse pass produces finite, plausible, wrong
//!   hidden states.
//! * **`freq_scale` for pooled positions.** The compressor's RoPE divides the
//!   frequency table by the compression ratio and divides the offset by the
//!   same ratio, so pooled token `i` at pool offset `p` lands on the angle of
//!   absolute position `p * ratio` under the unscaled table.
//!
//! The Yarn frequency correction matches the reference exactly (`factor`,
//! `original_max_position_embeddings`, `beta_fast` / `beta_slow` ramp). Tables
//! are precomputed per `(head_dim, inverse)` pair at load time; construction
//! declares which pairs a module will use so nothing is built lazily behind a
//! `RefCell`.

use mlxcel_core::{MlxArray, UniquePtr};

use super::RopeScalingV4;

/// One precomputed padded frequency table.
struct RopeTable {
    head_dim: i32,
    inverse: bool,
    freqs: UniquePtr<MlxArray>,
}

pub(crate) struct V4Rope {
    /// Rotated lane count (`qk_rope_head_dim`).
    dims: i32,
    /// Compression ratio for pooled positions, 1 for token positions.
    freq_scale: i32,
    tables: Vec<RopeTable>,
}

/// Compute the base (unpadded, unscaled) frequency table `1 / inv_freq` of
/// length `dims / 2`, applying the Yarn correction when configured.
///
/// Mirrors `DeepseekV4RoPE.__init__`. Pure over `f64` so the table is
/// unit-testable without an MLX context.
pub(crate) fn v4_rope_base_freqs(
    dims: i32,
    base: f32,
    scaling: Option<&RopeScalingV4>,
) -> Result<Vec<f32>, String> {
    if dims <= 0 || dims % 2 != 0 {
        return Err(format!(
            "DeepSeek-V4 RoPE dims must be a positive even number, got {dims}"
        ));
    }
    if !(base.is_finite() && base > 1.0) {
        return Err(format!("DeepSeek-V4 rope base must be > 1, got {base}"));
    }

    let half = (dims / 2) as usize;
    let dims_f = dims as f64;
    let base_f = base as f64;
    let mut inv_freq: Vec<f64> = (0..half)
        .map(|i| 1.0 / base_f.powf((2 * i) as f64 / dims_f))
        .collect();

    if let Some(sc) = scaling {
        let rope_type = sc.scaling_type.as_deref();
        match rope_type {
            None | Some("default") => {}
            Some("yarn") | Some("deepseek_yarn") => {
                let factor = sc
                    .factor
                    .ok_or("DeepSeek-V4 yarn rope_scaling requires `factor`")?;
                let orig = sc.original_max_position_embeddings.ok_or(
                    "DeepSeek-V4 yarn rope_scaling requires `original_max_position_embeddings`",
                )?;
                if !(factor.is_finite() && factor > 0.0) {
                    return Err(format!("DeepSeek-V4 yarn factor must be > 0, got {factor}"));
                }
                if orig == 0 {
                    return Err(
                        "DeepSeek-V4 yarn original_max_position_embeddings must be > 0".into(),
                    );
                }
                let beta_fast = sc.beta_fast.unwrap_or(32.0) as f64;
                let beta_slow = sc.beta_slow.unwrap_or(1.0) as f64;
                let orig_f = orig as f64;
                let correction_dim = |num_rotations: f64| -> f64 {
                    dims_f * (orig_f / (num_rotations * 2.0 * std::f64::consts::PI)).ln()
                        / (2.0 * base_f.ln())
                };
                let low = correction_dim(beta_fast).floor().max(0.0);
                let mut high = correction_dim(beta_slow).ceil().min(dims_f - 1.0);
                if low == high {
                    high += 0.001;
                }
                let factor_f = factor as f64;
                for (i, f) in inv_freq.iter_mut().enumerate() {
                    let ramp = (i as f64 - low) / (high - low);
                    let smooth = 1.0 - ramp.clamp(0.0, 1.0);
                    *f = *f / factor_f * (1.0 - smooth) + *f * smooth;
                }
            }
            Some(other) => {
                return Err(format!("Unsupported DeepSeek-V4 RoPE type: {other}"));
            }
        }
    }

    Ok(inv_freq.iter().map(|f| (1.0 / f) as f32).collect())
}

/// Build the padded table for one `(head_dim, inverse)` use site: divide by
/// `freq_scale`, negate for inverse, prefix `(head_dim - dims) / 2` pairs of
/// `inf` so the leading lanes do not rotate.
///
/// Pure so the inf-prefix / negate / scale composition is unit-testable.
pub(crate) fn v4_rope_padded_freqs(
    base_freqs: &[f32],
    dims: i32,
    head_dim: i32,
    freq_scale: i32,
    inverse: bool,
) -> Result<Vec<f32>, String> {
    if head_dim < dims || head_dim % 2 != 0 {
        return Err(format!(
            "DeepSeek-V4 RoPE head_dim {head_dim} must be even and >= rotated dims {dims}"
        ));
    }
    if freq_scale < 1 {
        return Err(format!(
            "DeepSeek-V4 RoPE freq_scale must be >= 1, got {freq_scale}"
        ));
    }
    let nope_pairs = ((head_dim - dims) / 2) as usize;
    let mut out = Vec::with_capacity(nope_pairs + base_freqs.len());
    out.resize(nope_pairs, f32::INFINITY);
    let scale = freq_scale as f32;
    for &f in base_freqs {
        let mut v = f / scale;
        if inverse {
            v = -v;
        }
        out.push(v);
    }
    Ok(out)
}

impl V4Rope {
    /// Build a rope with tables precomputed for every `(head_dim, inverse)`
    /// pair in `uses`.
    pub(crate) fn new(
        dims: i32,
        base: f32,
        scaling: Option<&RopeScalingV4>,
        freq_scale: i32,
        uses: &[(i32, bool)],
    ) -> Result<Self, String> {
        let base_freqs = v4_rope_base_freqs(dims, base, scaling)?;
        let mut tables = Vec::with_capacity(uses.len());
        for &(head_dim, inverse) in uses {
            let padded = v4_rope_padded_freqs(&base_freqs, dims, head_dim, freq_scale, inverse)?;
            tables.push(RopeTable {
                head_dim,
                inverse,
                freqs: mlxcel_core::from_slice_f32(&padded, &[padded.len() as i32]),
            });
        }
        Ok(Self {
            dims,
            freq_scale,
            tables,
        })
    }

    /// Apply the rotation to `x` (`[..., L, head_dim]`) at `offset`.
    ///
    /// The offset is divided by `freq_scale` exactly as the reference does;
    /// pooled callers pass a window-aligned base offset so the division is
    /// exact.
    pub(crate) fn apply(&self, x: &MlxArray, offset: i32, inverse: bool) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let head_dim = *shape.last().expect("V4Rope input must have a last axis");
        let table = self
            .tables
            .iter()
            .find(|t| t.head_dim == head_dim && t.inverse == inverse)
            .unwrap_or_else(|| {
                panic!(
                    "V4Rope table for (head_dim {head_dim}, inverse {inverse}) was not \
                     precomputed; declared uses cover a different shape (rotated dims {})",
                    self.dims
                )
            });
        let offset = if self.freq_scale != 1 {
            offset / self.freq_scale
        } else {
            offset
        };
        mlxcel_core::fast_rope_with_freqs(x, head_dim, true, 1.0, offset, &table.freqs)
    }
}
