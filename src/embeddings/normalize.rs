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

//! `--embd-normalize`: llama-server b10621's embedding normalization domain
//! (issue #1452).
//!
//! b10621 states the whole domain in one integer, and the integer is not an
//! enum: `-1` is no normalization, `0` is a max-absolute rescale into the int16
//! range, `2` is Euclidean, and **every other value is the p-norm with that
//! `p`**, which is why `1` (taxicab) needs no case of its own. mlxcel had one
//! behavior, L2, chosen by a per-checkpoint boolean.
//!
//! # The zero vector
//!
//! Upstream computes `norm = sum > 0 ? 1/sum : 0`, so a zero vector normalizes
//! to zeros rather than to NaN. That is reproduced here, and it is the reason
//! this module does not simply divide: dividing by a zero norm would put NaN
//! into a response body, and the embedding route's finite-value guard would
//! then answer 500 for an input that upstream serves.
//!
//! # Euclidean stays byte-identical
//!
//! [`EmbdNormalize::EUCLIDEAN`] delegates to [`super::pooling::normalize_l2`],
//! which is what every mlxcel embedding response has been normalized with. The
//! two formulas differ only for a vector whose norm is between zero and 1e-9
//! (upstream scales it to unit length, `normalize_l2` clamps the divisor), and
//! reusing the existing kernel means the default path produces exactly the
//! numbers it produced before this change. The new modes are new behavior; the
//! default is not.
//!
//! Upstream reference: `common_embd_normalize` in
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/common.cpp>

use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::pooling::normalize_l2;

/// The divisor upstream scales a max-absolute normalization by, so the result
/// spans the signed int16 range.
const INT16_SCALE: f32 = 32760.0;

/// One `--embd-normalize` value.
///
/// Held as the integer b10621 holds it as, because the domain is open above:
/// any `p > 2` is a valid p-norm and enumerating them is not possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmbdNormalize(i32);

impl EmbdNormalize {
    /// `-1`: leave the vector as the model produced it.
    pub const NONE: Self = Self(-1);
    /// `0`: divide by the largest absolute component over 32760, so the result
    /// spans the signed int16 range.
    pub const MAX_ABS_INT16: Self = Self(0);
    /// `1`: the taxicab (L1) norm, served by the p-norm branch.
    pub const TAXICAB: Self = Self(1);
    /// `2`: the Euclidean (L2) norm. b10621's default, and mlxcel's previous
    /// only behavior.
    pub const EUCLIDEAN: Self = Self(2);

    /// Accept one `--embd-normalize` value.
    ///
    /// Every integer at or above `-1` is in the domain. Below `-1` is not: it
    /// is neither a sentinel nor a usable `p`, and upstream would take the
    /// p-norm branch with a negative exponent and return values no client can
    /// interpret, so mlxcel refuses it instead of reproducing that.
    pub fn new(value: i32) -> Result<Self, String> {
        if value < -1 {
            return Err(format!(
                "--embd-normalize {value} is out of domain: pass -1 for no normalization, 0 for \
                 the max-absolute int16 rescale, 1 for taxicab, 2 for euclidean, or a value \
                 above 2 for that p-norm"
            ));
        }
        Ok(Self(value))
    }

    /// The integer b10621 states this as.
    #[must_use]
    pub fn value(self) -> i32 {
        self.0
    }

    /// Whether this leaves the vector untouched.
    #[must_use]
    pub fn is_none(self) -> bool {
        self == Self::NONE
    }

    /// The default a checkpoint's own `normalize` flag selects: Euclidean when
    /// the checkpoint normalizes, none when it does not.
    #[must_use]
    pub fn from_model_flag(normalize: bool) -> Self {
        if normalize {
            Self::EUCLIDEAN
        } else {
            Self::NONE
        }
    }
}

impl Default for EmbdNormalize {
    fn default() -> Self {
        Self::EUCLIDEAN
    }
}

impl std::fmt::Display for EmbdNormalize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Apply one `--embd-normalize` mode along the last axis.
///
/// Returns a fresh f32 array. See the module docs for the zero-vector rule and
/// for why Euclidean delegates rather than recomputing.
pub fn apply_embd_normalize(x: &MlxArray, kind: EmbdNormalize) -> UniquePtr<MlxArray> {
    if kind.is_none() {
        return mlxcel_core::astype(x, dtype::FLOAT32);
    }
    if kind == EmbdNormalize::EUCLIDEAN {
        return normalize_l2(x);
    }
    let x = mlxcel_core::astype(x, dtype::FLOAT32);
    // The divisor, one per row, keeping the axis so it broadcasts back.
    let divisor = if kind == EmbdNormalize::MAX_ABS_INT16 {
        let max_abs = mlxcel_core::max_axis(&mlxcel_core::abs(&x), -1, true);
        mlxcel_core::divide_scalar(&max_abs, INT16_SCALE)
    } else {
        // Every remaining value is a p-norm, taxicab (p = 1) included.
        mlxcel_core::linalg_norm_ord(&x, f64::from(kind.value()), -1, true)
    };
    scale_by_reciprocal(&x, &divisor)
}

/// `x * (divisor > 0 ? 1/divisor : 0)`, upstream's rule for a row whose
/// divisor is zero.
///
/// The reciprocal is taken of a floored divisor so the arithmetic never
/// produces an infinity that `where_cond` would then have to discard; the
/// selection is what zeroes the row, not the division.
fn scale_by_reciprocal(x: &MlxArray, divisor: &MlxArray) -> UniquePtr<MlxArray> {
    let zero = mlxcel_core::zeros_like(divisor);
    let one = mlxcel_core::ones_like(divisor);
    let positive = mlxcel_core::greater(divisor, &zero);
    // Substituting 1 where the divisor is zero keeps the reciprocal finite; the
    // outer selection is what actually zeroes those rows, so no infinity is
    // ever produced and then discarded.
    let safe = mlxcel_core::where_cond(&positive, divisor, &one);
    let scale = mlxcel_core::where_cond(&positive, &mlxcel_core::reciprocal(&safe), &zero);
    mlxcel_core::multiply(x, &scale)
}

#[cfg(test)]
#[path = "normalize_tests.rs"]
mod normalize_tests;
