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

//! The buffers and scalars a model-state snapshot carries for one attention
//! cache, in every KV mode a snapshot can hold (issue #3250).
//!
//! A snapshot used to copy `keys` and `values` only, which is the whole state
//! of an FP16 cache and a fraction of a quantized one: `Int8` keeps its
//! per-token scales beside the int8 rows, `Turbo4Asym` keeps V in
//! `v_packed` / `v_norms` / `v_rescale` and has no `values` at all, and
//! symmetric `Turbo4` keeps K packed as well and has neither. The sidecar
//! fields are private to this crate, so the snapshot encoders in the `mlxcel`
//! crate read and write them through the pair of views here.
//!
//! The field set is the one [`DetachedKVCache`](super::DetachedKVCache)
//! moves, minus what only `Turbo4Delegated` uses (`cold_offset`,
//! `hot_threshold` and the FP16 fast-path flags): the delegated hot/cold
//! split and the 3-bit `Turbo3Asym` packing are not carried. A detached
//! handle takes the buffers out of the cache; these views leave them where
//! they are, because the cache a snapshot is taken from keeps decoding.
//!
//! One layout check, [`KVCacheSnapshotParts::validate`], serves both
//! directions. A snapshot refuses a cache whose buffers are not the layout
//! its mode writes, and a restore runs the same check before it assigns
//! anything, so a snapshot that fails leaves the cache untouched and the
//! caller prefills cold.

use cxx::UniquePtr;

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;

use super::{KVCache, KVCacheMode, RotatingKVCache, RotatingKVCacheSnapshotState, turbo};

/// Names of the sidecar buffers of a [`KVCache`], in the order a snapshot
/// stores them. `keys` and `values` are not in the list: every encoder
/// already names those two itself.
///
/// Used by: `models::kv_snapshot` (`push_kv_sidecars`, `restore_kv_sidecars`).
pub const KV_SIDECAR_FIELDS: [&str; 7] = [
    "key_scales",
    "val_scales",
    "v_packed",
    "v_norms",
    "v_rescale",
    "k_packed",
    "k_norms",
];

/// Names of the sidecar buffers of a `Turbo4Asym` [`RotatingKVCache`], in the
/// order a snapshot stores them.
///
/// Used by: `models::kv_snapshot` (`snapshot_rotating`, `restore_rotating`).
pub const ROTATING_SIDECAR_FIELDS: [&str; 3] = ["v_packed", "v_norms", "v_rescale"];

/// Whether a full-attention [`KVCache`] in `mode` can be carried by a
/// model-state snapshot.
///
/// `Turbo4Delegated` splits V into a packed cold body and an FP16 hot tail at
/// `cold_offset`, and `Turbo3Asym` packs V three bits to a coordinate; neither
/// layout is modelled here.
///
/// Used by: `models::kv_snapshot`, and the scheduler's check that a family's
/// KV mode table can donate snapshots at all.
pub const fn kv_mode_is_snapshotable(mode: KVCacheMode) -> bool {
    matches!(
        mode,
        KVCacheMode::Fp16 | KVCacheMode::Int8 | KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4
    )
}

/// Borrowed view of everything a snapshot of one [`KVCache`] has to carry.
///
/// Used by: `models::kv_snapshot` and the hybrid families' attention arms.
#[derive(Clone, Copy)]
pub struct KVCacheSnapshotParts<'a> {
    /// The mode the cache was built with.
    pub mode: KVCacheMode,
    /// Monotonic write position (the next RoPE position).
    pub offset: i32,
    /// Start of the live window. Non-zero only after a front trim, which only
    /// `Fp16` and `Int8` caches take.
    pub live_start: i32,
    /// Seed of the Turbo4 sign vectors. The packed rows cannot be decoded
    /// under another seed.
    pub turbo_seed: u32,
    pub keys: Option<&'a MlxArray>,
    pub values: Option<&'a MlxArray>,
    pub key_scales: Option<&'a MlxArray>,
    pub val_scales: Option<&'a MlxArray>,
    pub v_packed: Option<&'a MlxArray>,
    pub v_norms: Option<&'a MlxArray>,
    pub v_rescale: Option<&'a MlxArray>,
    pub k_packed: Option<&'a MlxArray>,
    pub k_norms: Option<&'a MlxArray>,
}

/// Owned counterpart of [`KVCacheSnapshotParts`], handed to
/// [`KVCache::restore_snapshot_parts`].
///
/// Used by: `models::kv_snapshot` and the hybrid families' attention arms.
#[derive(Default)]
pub struct KVCacheSnapshotOwned {
    pub mode: KVCacheMode,
    pub offset: i32,
    pub live_start: i32,
    pub turbo_seed: u32,
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub key_scales: Option<UniquePtr<MlxArray>>,
    pub val_scales: Option<UniquePtr<MlxArray>>,
    pub v_packed: Option<UniquePtr<MlxArray>>,
    pub v_norms: Option<UniquePtr<MlxArray>>,
    pub v_rescale: Option<UniquePtr<MlxArray>>,
    pub k_packed: Option<UniquePtr<MlxArray>>,
    pub k_norms: Option<UniquePtr<MlxArray>>,
}

/// What a present buffer has to look like.
#[derive(Clone, Copy)]
struct Expect {
    /// Required dtype, or `None` where the mode stores what it was given.
    dtype: Option<i32>,
    /// Required size of the last axis, or `None` for a head-dim axis.
    last: Option<i32>,
    /// Whether the last axis holds nibble-packed indices, whose width has to
    /// be half a power of two.
    packed: bool,
}

const ROWS: Expect = Expect {
    dtype: None,
    last: None,
    packed: false,
};
const INT8_ROWS: Expect = Expect {
    dtype: Some(dtype::INT8),
    last: None,
    packed: false,
};
const FP16_ROWS: Expect = Expect {
    dtype: Some(dtype::FLOAT16),
    last: None,
    packed: false,
};
const PER_TOKEN: Expect = Expect {
    dtype: Some(dtype::FLOAT16),
    last: Some(1),
    packed: false,
};
const PACKED: Expect = Expect {
    dtype: Some(dtype::UINT8),
    last: None,
    packed: true,
};

/// Check one named buffer and return its shape, or `None` when it is absent.
///
/// `want` is `None` for a buffer the mode must not hold.
fn check_buffer(
    name: &str,
    array: Option<&MlxArray>,
    want: Option<Expect>,
    mode: KVCacheMode,
) -> Result<Option<Vec<i32>>, String> {
    let (array, want) = match (array, want) {
        (None, None) => return Ok(None),
        (Some(_), None) => return Err(format!("a {mode:?} cache does not hold {name}")),
        (None, Some(_)) => return Err(format!("a {mode:?} cache needs {name}")),
        (Some(array), Some(want)) => (array, want),
    };
    let shape = ffi::array_shape(array);
    if shape.len() != 4 {
        return Err(format!("expected rank-4 {name}, got shape {shape:?}"));
    }
    if let Some(expected) = want.dtype {
        let actual = ffi::array_dtype(array);
        if actual != expected {
            return Err(format!(
                "{name} has dtype {actual}, a {mode:?} cache stores dtype {expected}"
            ));
        }
    }
    if let Some(last) = want.last
        && shape[3] != last
    {
        return Err(format!(
            "{name} has a last axis of {}, not {last}",
            shape[3]
        ));
    }
    if want.packed && !packed_width_is_valid(shape[3]) {
        return Err(format!(
            "{name} has a packed width of {}, which is not half a power-of-two head dim",
            shape[3]
        ));
    }
    Ok(Some(shape))
}

/// Whether `width` nibble-packed bytes describe a head dim the Turbo4 rotation
/// accepts. `TurboQuantParams::new` panics on anything else, so a restore has
/// to refuse it first.
fn packed_width_is_valid(width: i32) -> bool {
    width > 0 && width <= i32::MAX / 2 && u32::try_from(width * 2).is_ok_and(u32::is_power_of_two)
}

impl KVCacheSnapshotParts<'_> {
    /// Whether no buffer is present at all, which is a cache that stored
    /// nothing (a fresh one, a KV-shared layer, or one rewound to zero).
    pub fn holds_no_rows(&self) -> bool {
        self.keys.is_none()
            && self.values.is_none()
            && KV_SIDECAR_FIELDS
                .iter()
                .all(|name| self.sidecar(name).is_none())
    }

    /// The sidecar buffer called `name` in [`KV_SIDECAR_FIELDS`].
    pub fn sidecar(&self, name: &str) -> Option<&MlxArray> {
        match name {
            "key_scales" => self.key_scales,
            "val_scales" => self.val_scales,
            "v_packed" => self.v_packed,
            "v_norms" => self.v_norms,
            "v_rescale" => self.v_rescale,
            "k_packed" => self.k_packed,
            "k_norms" => self.k_norms,
            _ => None,
        }
    }

    /// Check that the buffers are the layout `mode` writes and that they
    /// cover the live window `[live_start, offset)`.
    ///
    /// Reads shapes and dtypes only, so the cost does not depend on how many
    /// tokens the cache holds.
    ///
    /// # Errors
    ///
    /// A mode a snapshot does not carry, an `Int8` cache holding a quantized
    /// MLA latent instead of per-head int8 rows, a buffer the mode needs and
    /// lacks or holds and should not, a rank, dtype or width the mode does
    /// not write, buffers that disagree on batch, heads or length, or a
    /// length shorter than the live window.
    pub fn validate(&self) -> Result<(), String> {
        let mode = self.mode;
        if !kv_mode_is_snapshotable(mode) {
            return Err(format!(
                "cache mode {mode:?} is not supported by model-state snapshots"
            ));
        }
        if self.offset < 0 || self.live_start < 0 || self.live_start > self.offset {
            return Err(format!(
                "live window [{}, {}) is not a valid range",
                self.live_start, self.offset
            ));
        }
        if self.live_start != 0 && !matches!(mode, KVCacheMode::Fp16 | KVCacheMode::Int8) {
            return Err(format!(
                "a {mode:?} cache cannot have a trimmed front (live_start {})",
                self.live_start
            ));
        }
        // An absorbed MLA latent under `Int8` reuses the four slots of a
        // per-head Int8 cache for a different layout (`cache/latent_quant.rs`:
        // uint32 packed latent rows in `keys`, FP16 `k_pe` in `values`, affine
        // group scales and biases in the two scale slots). It is not carried:
        // name it, rather than report its first mismatching buffer.
        if mode == KVCacheMode::Int8
            && self
                .keys
                .is_some_and(|keys| ffi::array_dtype(keys) == dtype::UINT32)
        {
            return Err(
                "an Int8 cache holding a quantized MLA latent is not supported by model-state snapshots"
                    .to_string(),
            );
        }
        // What each of the nine buffers must look like in this mode, in the
        // order: keys, values, then `KV_SIDECAR_FIELDS`.
        let wanted: [Option<Expect>; 9] = match mode {
            KVCacheMode::Fp16 => [
                Some(ROWS),
                Some(ROWS),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            KVCacheMode::Int8 => [
                Some(INT8_ROWS),
                Some(INT8_ROWS),
                Some(PER_TOKEN),
                Some(PER_TOKEN),
                None,
                None,
                None,
                None,
                None,
            ],
            KVCacheMode::Turbo4Asym => [
                Some(FP16_ROWS),
                None,
                None,
                None,
                Some(PACKED),
                Some(PER_TOKEN),
                Some(PER_TOKEN),
                None,
                None,
            ],
            KVCacheMode::Turbo4 => [
                None,
                None,
                None,
                None,
                Some(PACKED),
                Some(PER_TOKEN),
                Some(PER_TOKEN),
                Some(PACKED),
                Some(PER_TOKEN),
            ],
            KVCacheMode::Turbo4Delegated | KVCacheMode::Turbo3Asym => {
                unreachable!("refused above by kv_mode_is_snapshotable")
            }
        };
        let present: [(&str, Option<&MlxArray>); 9] = [
            ("keys", self.keys),
            ("values", self.values),
            ("key_scales", self.key_scales),
            ("val_scales", self.val_scales),
            ("v_packed", self.v_packed),
            ("v_norms", self.v_norms),
            ("v_rescale", self.v_rescale),
            ("k_packed", self.k_packed),
            ("k_norms", self.k_norms),
        ];
        let live = self.offset - self.live_start;
        let mut lead: Option<(&str, Vec<i32>)> = None;
        for ((name, array), want) in present.into_iter().zip(wanted) {
            let Some(shape) = check_buffer(name, array, want, mode)? else {
                continue;
            };
            if shape[2] < live {
                return Err(format!(
                    "{name} holds {} rows, fewer than the {live} live tokens",
                    shape[2]
                ));
            }
            if let Some((first, first_shape)) = &lead {
                if first_shape[..3] != shape[..3] {
                    return Err(format!(
                        "{first} {first_shape:?} and {name} {shape:?} disagree on batch, heads or length"
                    ));
                }
            } else {
                lead = Some((name, shape));
            }
        }
        if mode == KVCacheMode::Turbo4
            && let (Some(k_packed), Some(v_packed)) = (self.k_packed, self.v_packed)
        {
            let (k_width, v_width) = (ffi::array_shape(k_packed)[3], ffi::array_shape(v_packed)[3]);
            if k_width != v_width {
                return Err(format!(
                    "symmetric Turbo4 needs one head dim for K and V, got packed widths {k_width} and {v_width}"
                ));
            }
        }
        Ok(())
    }
}

impl KVCacheSnapshotOwned {
    /// Borrow this set as the view [`KVCacheSnapshotParts::validate`] checks.
    pub fn as_parts(&self) -> KVCacheSnapshotParts<'_> {
        KVCacheSnapshotParts {
            mode: self.mode,
            offset: self.offset,
            live_start: self.live_start,
            turbo_seed: self.turbo_seed,
            keys: self.keys.as_deref(),
            values: self.values.as_deref(),
            key_scales: self.key_scales.as_deref(),
            val_scales: self.val_scales.as_deref(),
            v_packed: self.v_packed.as_deref(),
            v_norms: self.v_norms.as_deref(),
            v_rescale: self.v_rescale.as_deref(),
            k_packed: self.k_packed.as_deref(),
            k_norms: self.k_norms.as_deref(),
        }
    }

    /// Set the sidecar buffer called `name` in [`KV_SIDECAR_FIELDS`]. A name
    /// outside the list is ignored.
    pub fn set_sidecar(&mut self, name: &str, array: Option<UniquePtr<MlxArray>>) {
        match name {
            "key_scales" => self.key_scales = array,
            "val_scales" => self.val_scales = array,
            "v_packed" => self.v_packed = array,
            "v_norms" => self.v_norms = array,
            "v_rescale" => self.v_rescale = array,
            "k_packed" => self.k_packed = array,
            "k_norms" => self.k_norms = array,
            _ => {}
        }
    }
}

/// Turbo4 parameters for the head dim a packed buffer was written with.
///
/// The parameters are a pure function of the head dim and the seed, so the
/// ones built here are the ones the lazy init in the update paths would
/// build. They are built at restore time, as `install_detached` does, because
/// a reader that dequantizes without appending first needs them.
fn turbo_params_for(packed: &MlxArray, seed: u32) -> turbo::TurboQuantParams {
    let width = ffi::array_shape(packed)[3];
    turbo::TurboQuantParams::new((width as u32) * 2, seed)
}

impl KVCache {
    /// Borrow everything a model-state snapshot of this cache has to carry.
    ///
    /// Used by: `models::kv_snapshot` and the hybrid families' attention arms
    /// (prompt-cache snapshot donation).
    pub fn snapshot_parts(&self) -> KVCacheSnapshotParts<'_> {
        KVCacheSnapshotParts {
            mode: self.mode,
            offset: self.offset,
            live_start: self.live_start,
            turbo_seed: self.turbo_seed,
            keys: self.keys.as_deref(),
            values: self.values.as_deref(),
            key_scales: self.key_scales.as_deref(),
            val_scales: self.val_scales.as_deref(),
            v_packed: self.v_packed.as_deref(),
            v_norms: self.v_norms.as_deref(),
            v_rescale: self.v_rescale.as_deref(),
            k_packed: self.k_packed.as_deref(),
            k_norms: self.k_norms.as_deref(),
        }
    }

    /// Install a snapshot's buffers and scalars into this empty cache.
    ///
    /// The cache must be dense, empty and configured for the snapshot's mode.
    /// Everything is validated first ([`KVCacheSnapshotParts::validate`]), so
    /// an `Err` leaves the cache exactly as it was. A set with no buffers and
    /// an offset of zero is a layer that stored nothing and leaves the cache
    /// fresh.
    ///
    /// The restored cache takes the snapshot's `turbo_seed`: packed rows only
    /// decode under the seed they were written with.
    ///
    /// # Errors
    ///
    /// A pool-backed or non-empty cache, a mode that differs from the cache's,
    /// or a layout [`KVCacheSnapshotParts::validate`] refuses.
    ///
    /// Used by: `models::kv_snapshot` and the hybrid families' attention arms
    /// (prompt-cache snapshot restore).
    pub fn restore_snapshot_parts(&mut self, parts: KVCacheSnapshotOwned) -> Result<(), String> {
        if self.paged_backing.is_some() {
            return Err("a pool-backed cache cannot take a model-state snapshot".to_string());
        }
        if !self.snapshot_parts().holds_no_rows() || self.offset != 0 {
            return Err("the target cache is not empty".to_string());
        }
        if parts.mode != self.mode {
            return Err(format!(
                "snapshot mode {:?} does not match configured cache mode {:?}",
                parts.mode, self.mode
            ));
        }
        let view = parts.as_parts();
        if view.holds_no_rows() {
            if parts.offset != 0 {
                return Err(format!(
                    "the snapshot declares {} tokens and holds no rows",
                    parts.offset
                ));
            }
            return Ok(());
        }
        view.validate()?;
        self.turbo_params = match self.mode {
            KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 => parts
                .v_packed
                .as_deref()
                .map(|packed| turbo_params_for(packed, parts.turbo_seed)),
            _ => None,
        };
        self.turbo3_params = None;
        self.keys = parts.keys;
        self.values = parts.values;
        self.key_scales = parts.key_scales;
        self.val_scales = parts.val_scales;
        self.v_packed = parts.v_packed;
        self.v_norms = parts.v_norms;
        self.v_rescale = parts.v_rescale;
        self.k_packed = parts.k_packed;
        self.k_norms = parts.k_norms;
        self.offset = parts.offset;
        self.live_start = parts.live_start;
        self.turbo_seed = parts.turbo_seed;
        self.cold_offset = 0;
        // The restored buffers are shared with the snapshot, so they are not
        // rows this cache's own last write produced.
        self.inplace_marker = None;
        Ok(())
    }
}

/// Borrowed view of everything a snapshot of one [`RotatingKVCache`] has to
/// carry: the ring scalars and the ring buffers.
///
/// Used by: `models::kv_snapshot` (`snapshot_rotating`).
#[derive(Clone, Copy)]
pub struct RotatingKVCacheSnapshotParts<'a> {
    pub state: RotatingKVCacheSnapshotState,
    pub keys: Option<&'a MlxArray>,
    pub values: Option<&'a MlxArray>,
    pub v_packed: Option<&'a MlxArray>,
    pub v_norms: Option<&'a MlxArray>,
    pub v_rescale: Option<&'a MlxArray>,
}

/// The ring buffers handed to [`RotatingKVCache::restore_snapshot_state`].
///
/// Used by: `models::kv_snapshot` (`restore_rotating`).
#[derive(Default)]
pub struct RotatingKVCacheSnapshotBuffers {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub v_packed: Option<UniquePtr<MlxArray>>,
    pub v_norms: Option<UniquePtr<MlxArray>>,
    pub v_rescale: Option<UniquePtr<MlxArray>>,
}

impl RotatingKVCacheSnapshotParts<'_> {
    /// The sidecar buffer called `name` in [`ROTATING_SIDECAR_FIELDS`].
    pub fn sidecar(&self, name: &str) -> Option<&MlxArray> {
        match name {
            "v_packed" => self.v_packed,
            "v_norms" => self.v_norms,
            "v_rescale" => self.v_rescale,
            _ => None,
        }
    }

    /// Whether the ring holds a buffer its storage mode does not write, or
    /// lacks one it does.
    ///
    /// # Errors
    ///
    /// The mismatch, named.
    pub fn check_layout(&self) -> Result<(), String> {
        let stored = RotatingKVCache::stored_mode(self.state.max_size, self.state.mode);
        check_ring_layout(
            stored,
            self.keys,
            self.values,
            [self.v_packed, self.v_norms, self.v_rescale],
        )
        .map(|_| ())
    }
}

impl RotatingKVCacheSnapshotBuffers {
    /// The buffers of a ring that stores plain rows: every mode but an
    /// aligned `Turbo4Asym`.
    pub fn fp16(keys: Option<UniquePtr<MlxArray>>, values: Option<UniquePtr<MlxArray>>) -> Self {
        Self {
            keys,
            values,
            ..Self::default()
        }
    }

    /// Set the sidecar buffer called `name` in [`ROTATING_SIDECAR_FIELDS`]. A
    /// name outside the list is ignored.
    pub fn set_sidecar(&mut self, name: &str, array: Option<UniquePtr<MlxArray>>) {
        match name {
            "v_packed" => self.v_packed = array,
            "v_norms" => self.v_norms = array,
            "v_rescale" => self.v_rescale = array,
            _ => {}
        }
    }
}

/// Check a ring's buffers against the mode it stores and return the physical
/// ring length, or `None` for a ring that holds nothing.
///
/// `stored` is [`RotatingKVCache::stored_mode`] of the ring, never its label:
/// a ring labelled `Int8`, `Turbo4`, `Turbo4Delegated` or `Turbo3Asym` stores
/// plain rows in `keys` and `values`, and only an aligned `Turbo4Asym` ring
/// keeps V in the three sidecars.
pub(super) fn check_ring_layout(
    stored: KVCacheMode,
    keys: Option<&MlxArray>,
    values: Option<&MlxArray>,
    sidecars: [Option<&MlxArray>; 3],
) -> Result<Option<i32>, String> {
    let [v_packed, v_norms, v_rescale] = sidecars;
    match stored {
        KVCacheMode::Fp16 => {
            if sidecars.iter().any(Option::is_some) {
                return Err("RotatingKVCache snapshot of plain rows carries packed V".into());
            }
            let (keys, values) = match (keys, values) {
                (None, None) => return Ok(None),
                (Some(keys), Some(values)) => (keys, values),
                _ => {
                    return Err(
                        "RotatingKVCache::restore_snapshot_state requires keys and values together"
                            .into(),
                    );
                }
            };
            let k_shape = ffi::array_shape(keys);
            if k_shape.len() != 4 {
                return Err(format!(
                    "RotatingKVCache::restore_snapshot_state expected rank-4 keys, got shape {k_shape:?}"
                ));
            }
            let v_shape = ffi::array_shape(values);
            if v_shape.len() != 4
                || k_shape[..3] != v_shape[..3]
                || ffi::array_dtype(keys) != ffi::array_dtype(values)
            {
                return Err("RotatingKVCache snapshot K/V shapes or dtypes disagree".into());
            }
            Ok(Some(k_shape[2]))
        }
        KVCacheMode::Turbo4Asym => {
            if values.is_some() {
                return Err("RotatingKVCache Turbo4Asym snapshot carries an FP16 V buffer".into());
            }
            if keys.is_none() && sidecars.iter().all(Option::is_none) {
                return Ok(None);
            }
            let mode = KVCacheMode::Turbo4Asym;
            let k_shape = check_buffer("keys", keys, Some(FP16_ROWS), mode)?
                .expect("a wanted buffer is present or an error");
            for (name, array, want) in [
                ("v_packed", v_packed, PACKED),
                ("v_norms", v_norms, PER_TOKEN),
                ("v_rescale", v_rescale, PER_TOKEN),
            ] {
                let shape = check_buffer(name, array, Some(want), mode)?
                    .expect("a wanted buffer is present or an error");
                if shape[..3] != k_shape[..3] {
                    return Err(format!(
                        "RotatingKVCache snapshot keys {k_shape:?} and {name} {shape:?} disagree on batch, heads or length"
                    ));
                }
            }
            Ok(Some(k_shape[2]))
        }
        other => Err(format!(
            "RotatingKVCache snapshots do not carry {other:?} storage"
        )),
    }
}

impl RotatingKVCache {
    /// Borrow everything a model-state snapshot of this ring has to carry.
    ///
    /// Used by: `models::kv_snapshot` (prompt-cache snapshot donation of the
    /// sliding-window layers).
    pub fn snapshot_parts(&self) -> RotatingKVCacheSnapshotParts<'_> {
        RotatingKVCacheSnapshotParts {
            state: self.snapshot_state(),
            keys: self.keys.as_deref(),
            values: self.values.as_deref(),
            v_packed: self.v_packed.as_deref(),
            v_norms: self.v_norms.as_deref(),
            v_rescale: self.v_rescale.as_deref(),
        }
    }

    /// Install the Turbo4 sidecars and parameters of a restored ring, or
    /// clear them for a ring that stores plain rows.
    pub(super) fn install_snapshot_sidecars(
        &mut self,
        buffers: &mut RotatingKVCacheSnapshotBuffers,
        seed: u32,
    ) {
        self.turbo_params = buffers
            .v_packed
            .as_deref()
            .map(|packed| turbo_params_for(packed, seed));
        self.v_packed = buffers.v_packed.take();
        self.v_norms = buffers.v_norms.take();
        self.v_rescale = buffers.v_rescale.take();
        self.key_scales = None;
        self.val_scales = None;
    }
}
