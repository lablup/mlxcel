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
//! ## Split slot (issue #3179)
//!
//! Under `--mla-latent-quant int4` the latent columns are stored 4-bit and the
//! indexer key stays at [`SIDECAR_KEY_BITS`]: the #3163 overlap gate showed
//! the indexer's selection is the precision-sensitive part. The slot is then
//! two runs of whole groups packed at different widths, written one after the
//! other in the same three buffers:
//!
//! ```text
//!   packed  [ rank/64 groups x 8 words (4-bit) | index/64 groups x 16 words (8-bit) ]
//!   scales  [ rank/64                          | index/64                           ]
//!   biases  [ rank/64                          | index/64                           ]
//! ```
//!
//! No extra field records the boundary. A [`LatentWindow::Quantized`] carries
//! the latent width in `bits`, the sidecar width is fixed, and the packed and
//! scale widths then determine how many leading groups are latent
//! (`latent_group_count`). A window whose slot is one width (every non-DSA
//! family, and DSA at 8 bits) has no sidecar run and reads exactly as before.
//!
//! Kept beside [`crate::mla::quant_latent`] rather than inside it so the DSA
//! column and row arithmetic stays in one small file.

use cxx::UniquePtr;

use crate::cache::KVCacheMode;
use crate::cache::latent_quant::{LATENT_QUANT_GROUP_SIZE, QuantizedLatentRows};
use crate::concatenate;
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::quant_latent::{
    LatentWindow, dequantize_latent, quantize_latent_rows, quantize_latent_triple,
    quantized_latent_bytes_per_token,
};

/// Bit width a sidecar cached beside the latent (the DSA indexer key) is
/// stored at, whatever width the latent itself takes.
pub const SIDECAR_KEY_BITS: i32 = 8;

/// `model_type` values whose cached "K" slot is `[kv_latent | DSA indexer
/// key]`: `DeepSeekV32Model` under both spellings of its `config.json` value,
/// GLM-5, which wraps that model wholesale, and Hy4, which caches the key on
/// its `full` indexer layers only (its `shared` layers cache the latent
/// alone, #2354). Every entry is also in
/// [`crate::mla::MLA_LATENT_CACHE_FAMILIES`].
///
/// Used by: [`caches_dsa_indexer_key`].
pub static DSA_INDEXER_KEY_FAMILIES: &[&str] =
    &["deepseek_v32", "deepseek_v3.2", "glm_moe_dsa", "hy_v4"];

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

/// How many leading groups of a packed key slot are stored at `bits`; any
/// groups after them are a sidecar at [`SIDECAR_KEY_BITS`] (module docs,
/// "Split slot"). Equal to the slot's group count when it has one width.
fn latent_group_count(packed: &MlxArray, scales: &MlxArray, bits: i32, group: i32) -> i32 {
    let last = |a: &MlxArray| ffi::array_shape(a).last().copied().unwrap_or(0);
    let (words, groups) = (last(packed), last(scales));
    let (per_latent, per_sidecar) = (group * bits / 32, group * SIDECAR_KEY_BITS / 32);
    if per_latent == per_sidecar {
        return groups;
    }
    let latent = (groups * per_sidecar - words) / (per_sidecar - per_latent);
    assert!(
        (0..=groups).contains(&latent)
            && latent * per_latent + (groups - latent) * per_sidecar == words,
        "mla: a {words}-word packed key slot with {groups} groups is not a {bits}-bit latent \
         with a {SIDECAR_KEY_BITS}-bit sidecar"
    );
    latent
}

/// Whether a packed key slot stores a sidecar at a different width from its
/// latent, so it cannot be handed to a kernel as one `bits`-wide tensor.
pub(crate) fn has_split_sidecar(
    packed: &MlxArray,
    scales: &MlxArray,
    bits: i32,
    group: i32,
) -> bool {
    let groups = ffi::array_shape(scales).last().copied().unwrap_or(0);
    latent_group_count(packed, scales, bits, group) != groups
}

/// Elements `[from, to)` of a packed key slot as its `(packed, scales,
/// biases)` slices and the bit width they are stored at. A group is
/// `group * width / 32` packed words and one scale and bias. On a split slot
/// the range must lie in the latent run or in the sidecar run.
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
    i32,
) {
    assert!(
        group > 0 && from % group == 0 && to % group == 0 && from <= to,
        "mla: key columns [{from}, {to}) of a quantized latent must fall on {group}-element \
         group boundaries"
    );
    let latent = latent_group_count(packed, scales, bits, group);
    let (g0, g1) = (from / group, to / group);
    assert!(
        g1 <= latent || g0 >= latent,
        "mla: key columns [{from}, {to}) straddle the latent and its sidecar, which are packed \
         at different widths"
    );
    let (per_latent, per_sidecar) = (group * bits / 32, group * SIDECAR_KEY_BITS / 32);
    let word_at = |g: i32| {
        if g <= latent {
            g * per_latent
        } else {
            latent * per_latent + (g - latent) * per_sidecar
        }
    };
    let part_bits = if g0 >= latent && g1 > g0 {
        SIDECAR_KEY_BITS
    } else {
        bits
    };
    (
        last_axis(packed, word_at(g0), word_at(g1)),
        last_axis(scales, g0, g1),
        last_axis(biases, g0, g1),
        part_bits,
    )
}

/// Quantize one step's `[B, 1, L, width]` key-slot rows whose first
/// `kv_lora_rank` elements are the latent, stored at `bits`.
///
/// A slot with no sidecar, and any slot at [`SIDECAR_KEY_BITS`], is one call
/// of the latent quantizer over the whole row, as it always was. Otherwise
/// the latent and the sidecar go through that same quantizer separately, at
/// `bits` and at [`SIDECAR_KEY_BITS`], and are concatenated (module docs,
/// "Split slot").
///
/// Used by: [`crate::mla::MlaLatentCache::update_and_fetch`].
pub(crate) fn quantize_key_slot_rows(
    ckv: &MlxArray,
    kpe: UniquePtr<MlxArray>,
    kv_lora_rank: i32,
    bits: i32,
) -> QuantizedLatentRows {
    let width = ffi::array_shape(ckv).last().copied().unwrap_or(0);
    if bits == SIDECAR_KEY_BITS || width <= kv_lora_rank {
        return quantize_latent_rows(ckv, kpe, bits);
    }
    let kpe = ffi::astype(&kpe, dtype::FLOAT16);
    let (packed, scales, biases) = quantize_latent_triple(&last_axis(ckv, 0, kv_lora_rank), bits);
    let (side_packed, side_scales, side_biases) =
        quantize_latent_triple(&last_axis(ckv, kv_lora_rank, width), SIDECAR_KEY_BITS);
    QuantizedLatentRows {
        packed: concatenate(&packed, &side_packed, -1),
        scales: concatenate(&scales, &side_scales, -1),
        biases: concatenate(&biases, &side_biases, -1),
        kpe,
    }
}

/// Dequantize a whole packed key slot to `[B, 1, S, width]`, the latent run
/// at `bits` and any sidecar run at [`SIDECAR_KEY_BITS`].
///
/// Used by: [`LatentWindow::into_dense`].
pub(crate) fn dequantize_key_slot(
    packed: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    bits: i32,
    group: i32,
) -> UniquePtr<MlxArray> {
    let groups = ffi::array_shape(scales).last().copied().unwrap_or(0);
    let latent = latent_group_count(packed, scales, bits, group);
    if latent == groups {
        return dequantize_latent(packed, scales, biases, bits, group);
    }
    let part = |from: i32, to: i32| {
        let (p, s, b, part_bits) = quantized_columns(packed, scales, biases, bits, group, from, to);
        dequantize_latent(&p, &s, &b, part_bits, group)
    };
    concatenate(
        &part(0, latent * group),
        &part(latent * group, groups * group),
        -1,
    )
}

/// Bytes one token of a quantized key slot costs, per layer: a
/// `kv_lora_rank`-wide latent at `bits`, a `sidecar`-wide sidecar at
/// [`SIDECAR_KEY_BITS`], the group scales and biases of both, and the fp16
/// rope stream.
///
/// With no sidecar, or at `bits == 8`, this is
/// [`quantized_latent_bytes_per_token`] over the whole slot: 672 at 8 bits
/// and 416 at 4 bits for rank 512 and rope 64. A DSA slot (rank 512, indexer
/// key 128) costs 808 at 8 bits and 552 at 4 bits.
///
/// Used by: `mlxcel::execution::kv_arch` (sizing) and
/// [`crate::mla::MlaLatentCache::bytes_per_token`].
#[must_use]
pub const fn quantized_key_slot_bytes_per_token(
    kv_lora_rank: usize,
    sidecar: usize,
    qk_rope_head_dim: usize,
    bits: usize,
    group_size: usize,
) -> usize {
    if sidecar == 0 || bits == SIDECAR_KEY_BITS as usize {
        return quantized_latent_bytes_per_token(
            kv_lora_rank.saturating_add(sidecar),
            qk_rope_head_dim,
            bits,
            group_size,
        );
    }
    quantized_latent_bytes_per_token(kv_lora_rank, qk_rope_head_dim, bits, group_size)
        .saturating_add(quantized_latent_bytes_per_token(
            sidecar,
            0,
            SIDECAR_KEY_BITS as usize,
            group_size,
        ))
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
                let (packed, scales, biases, part_bits) =
                    quantized_columns(ckv_q, scales, biases, *bits, *group_size, from, to);
                let dense = dequantize_latent(&packed, &scales, &biases, part_bits, *group_size);
                ffi::astype(&dense, ffi::array_dtype(kpe))
            }
        }
    }

    /// The window restricted to elements `[from, to)` of its "K" slot, in the
    /// same storage form, with the rope stream unchanged.
    ///
    /// On a quantized window `from` and `to` must be group boundaries; a
    /// group is `group_size * bits / 32` packed words and one scale and bias,
    /// so the result is three last-axis slices and nothing is dequantized. On
    /// a split slot the range lies in the latent or in the sidecar, and the
    /// result's `bits` is that run's width.
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
                let (ckv_q, scales, biases, bits) =
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
    /// and batched, and the LSA sparse decode of
    /// `models::longcat_flash_sparse`.
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
