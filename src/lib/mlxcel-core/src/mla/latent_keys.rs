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

//! A latent "K" slot wider than `kv_lora_rank`, and row selection on a
//! [`LatentWindow`] (issue #3163).
//!
//! The DSA families (`deepseek_v32`, `deepseek_v3.2`, `glm_moe_dsa`) cache the
//! indexer key beside the latent in one slot, `[kv_latent (kv_lora_rank) |
//! indexer key (index_head_dim)]`, 640 wide at the defaults. Stored quantized,
//! that row is affine group-64 like any other latent, and because both widths
//! are whole numbers of groups no group holds elements of both parts. Each
//! part is therefore a contiguous run of packed words, scales and biases, and
//! can be read on its own:
//!
//! * the indexer scores every cached key, so it dequantizes the indexer
//!   columns and nothing else ([`LatentWindow::dense_key_columns`]);
//! * the attention reads the latent columns, still packed
//!   ([`LatentWindow::narrow_keys`]);
//! * sparse decode gathers the selected rows from the packed triple and the
//!   rope stream ([`LatentWindow::gather_rows`]), so the window is never
//!   dequantized to be gathered from.
//!
//! Kept beside [`crate::mla::quant_latent`] rather than inside it so the DSA
//! column and row arithmetic stays in one small file.

use cxx::UniquePtr;

use crate::cache::KVCacheMode;
use crate::cache::latent_quant::LATENT_QUANT_GROUP_SIZE;
use crate::ffi::{self, MlxArray};
use crate::mla::quant_latent::{LatentWindow, dequantize_latent};

/// `model_type` values whose cached "K" slot is `[kv_latent | DSA indexer
/// key]`: `DeepSeekV32Model` under both spellings of its `config.json` value,
/// and GLM-5, which wraps that model wholesale. Every entry is also in
/// [`crate::mla::MLA_LATENT_CACHE_FAMILIES`].
///
/// Used by: [`caches_dsa_indexer_key`].
pub static DSA_INDEXER_KEY_FAMILIES: &[&str] = &["deepseek_v32", "deepseek_v3.2", "glm_moe_dsa"];

/// Whether `model_type` caches the DSA indexer key beside its MLA latent, so
/// its "K" slot is `kv_lora_rank + index_head_dim` wide. Exact, trimmed,
/// case-insensitive match, as [`crate::mla::caches_mla_latent_pair`].
///
/// Used by: `mlxcel::execution::kv_arch` (sizing) and
/// `mlxcel::cli::turbo_args` (the startup width check).
#[must_use]
pub fn caches_dsa_indexer_key(model_type: &str) -> bool {
    let needle = model_type.trim().to_ascii_lowercase();
    DSA_INDEXER_KEY_FAMILIES
        .iter()
        .any(|&family| needle == family)
}

/// Whether a "K" slot `stored_key_width` elements wide, whose first
/// `kv_lora_rank` elements are the latent, can be stored at `mode`'s bit
/// width.
///
/// The slot is never narrower than the latent. Under an affine latent mode
/// both the latent and the whole slot must be whole numbers of groups: the
/// first keeps every group on one side of the latent boundary, the second
/// leaves no partial group at the end. Returns the operator-facing reason
/// otherwise.
///
/// Used by: [`crate::mla::MlaLatentCache::supports_with_key_width`],
/// `mlxcel::execution::kv_arch` (sizing) and `mlxcel::cli::turbo_args` (the
/// startup decision on a DSA model).
pub fn check_stored_key_width(
    kv_lora_rank: usize,
    stored_key_width: usize,
    mode: KVCacheMode,
) -> Result<(), String> {
    let rank = kv_lora_rank;
    if stored_key_width < rank {
        return Err(format!(
            "mla: the cached key slot is {stored_key_width} wide, narrower than kv_lora_rank {rank}"
        ));
    }
    if mode.latent_affine_bits().is_none() {
        return Ok(());
    }
    let group = LATENT_QUANT_GROUP_SIZE as usize;
    if rank == 0 || !rank.is_multiple_of(group) || !stored_key_width.is_multiple_of(group) {
        return Err(format!(
            "mla: a {mode} latent cache groups the cached key slot in {group}s; kv_lora_rank \
             {rank} and the {stored_key_width}-wide slot (latent plus {} sidecar elements) must \
             both be multiples of that",
            stored_key_width - rank
        ));
    }
    Ok(())
}

/// `[.., from..to]` of a rank-4 array along its last axis.
fn last_axis(a: &MlxArray, from: i32, to: i32) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(a);
    ffi::slice(a, &[0, 0, 0, from], &[s[0], s[1], s[2], to])
}

/// Gather `[B, 1, K, 1]` sequence positions from a `[B, 1, S, W]` buffer.
fn take_rows(a: &MlxArray, positions: &MlxArray) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(a);
    let k = ffi::array_shape(positions)[2];
    let idx = ffi::broadcast_to(positions, &[s[0], s[1], k, s[3]]);
    ffi::take_along_axis(a, &idx, 2)
}

/// Elements `[from, to)` of a packed latent as its `(packed, scales, biases)`
/// slices. A group is `group * bits / 32` packed words and one scale and bias.
fn quantized_columns(
    packed: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    bits: i32,
    group: i32,
    from: i32,
    to: i32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    assert!(
        group > 0 && from % group == 0 && to % group == 0 && from <= to,
        "mla: key columns [{from}, {to}) of a quantized latent must fall on {group}-element \
         group boundaries"
    );
    let words_per_group = group * bits / 32;
    let (g0, g1) = (from / group, to / group);
    (
        last_axis(packed, g0 * words_per_group, g1 * words_per_group),
        last_axis(scales, g0, g1),
        last_axis(biases, g0, g1),
    )
}

impl LatentWindow {
    /// Width of the stored "K" slot in elements: `kv_lora_rank`, plus the
    /// sidecar a DSA family caches beside the latent.
    #[must_use]
    pub fn key_width(&self) -> i32 {
        match self {
            Self::Dense { ckv, .. } => ffi::array_shape(ckv).get(3).copied().unwrap_or(0),
            Self::Quantized {
                scales, group_size, ..
            } => ffi::array_shape(scales).get(3).copied().unwrap_or(0) * group_size,
        }
    }

    /// Elements `[from, to)` of the "K" slot as a dense `[B, 1, S, to - from]`
    /// array.
    ///
    /// On a quantized window only the groups that hold those elements are
    /// dequantized, and the result is cast to the rope stream's dtype, which
    /// is what [`Self::into_dense`] does for the whole slot: the two agree bit
    /// for bit on the columns they share. `from` and `to` must be group
    /// boundaries there.
    ///
    /// Used by: the DSA indexer's top-k selection (`models::deepseek_v32`),
    /// which reads the indexer-key columns and leaves the latent packed.
    #[must_use]
    pub fn dense_key_columns(&self, from: i32, to: i32) -> UniquePtr<MlxArray> {
        match self {
            Self::Dense { ckv, .. } => last_axis(ckv, from, to),
            Self::Quantized {
                ckv_q,
                scales,
                biases,
                kpe,
                bits,
                group_size,
            } => {
                let (packed, scales, biases) =
                    quantized_columns(ckv_q, scales, biases, *bits, *group_size, from, to);
                let dense = dequantize_latent(&packed, &scales, &biases, *bits, *group_size);
                ffi::astype(&dense, ffi::array_dtype(kpe))
            }
        }
    }

    /// The window restricted to elements `[from, to)` of its "K" slot, in the
    /// same storage form, with the rope stream unchanged.
    ///
    /// On a quantized window `from` and `to` must be group boundaries; a
    /// group is `group_size * bits / 32` packed words and one scale and bias,
    /// so the result is three last-axis slices and nothing is dequantized.
    ///
    /// Used by: the DSA attention (`models::deepseek_v32`), which drops the
    /// indexer-key columns before decode reads the latent.
    #[must_use]
    pub fn narrow_keys(self, from: i32, to: i32) -> Self {
        if from == 0 && to == self.key_width() {
            return self;
        }
        match self {
            Self::Dense { ckv, kpe } => Self::Dense {
                ckv: last_axis(&ckv, from, to),
                kpe,
            },
            Self::Quantized {
                ckv_q,
                scales,
                biases,
                kpe,
                bits,
                group_size,
            } => {
                let (ckv_q, scales, biases) =
                    quantized_columns(&ckv_q, &scales, &biases, bits, group_size, from, to);
                Self::Quantized {
                    ckv_q,
                    scales,
                    biases,
                    kpe,
                    bits,
                    group_size,
                }
            }
        }
    }

    /// The window's rows at `indices` along the sequence axis, in the same
    /// storage form.
    ///
    /// `indices` is the indexer's selection for one query, `[B, 1, 1, K]`.
    /// A quantized window gathers the packed latent, the group scales, the
    /// group biases and the rope stream with one `take_along_axis` each; a
    /// stored row is self-contained (its own scales and biases), so a gathered
    /// row dequantizes to exactly what the row dequantized to in place.
    ///
    /// Used by: DSA sparse decode (`models::deepseek_v32`), single-sequence
    /// and batched.
    #[must_use]
    pub fn gather_rows(&self, indices: &MlxArray) -> Self {
        let s = ffi::array_shape(indices);
        let (b, k) = (s[0], s[3]);
        let positions = ffi::reshape(
            &ffi::slice(indices, &[0, 0, 0, 0], &[b, 1, 1, k]),
            &[b, 1, k, 1],
        );
        match self {
            Self::Dense { ckv, kpe } => Self::Dense {
                ckv: take_rows(ckv, &positions),
                kpe: take_rows(kpe, &positions),
            },
            Self::Quantized {
                ckv_q,
                scales,
                biases,
                kpe,
                bits,
                group_size,
            } => Self::Quantized {
                ckv_q: take_rows(ckv_q, &positions),
                scales: take_rows(scales, &positions),
                biases: take_rows(biases, &positions),
                kpe: take_rows(kpe, &positions),
                bits: *bits,
                group_size: *group_size,
            },
        }
    }
}

#[cfg(test)]
#[path = "latent_keys_tests.rs"]
mod tests;
