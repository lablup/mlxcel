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

use crate::generate::ModelStateSnapshot;
use crate::layers::KVCache;
use crate::{MlxArray, UniquePtr};

/// Inkling's per-layer KV cache plus four causal short-convolution states.
pub struct InklingLayerCache {
    pub kv: KVCache,
    pub conv: [Option<UniquePtr<MlxArray>>; 4],
    /// Each convolution's whole `[1, kernel - 1 + L, C]` input window from
    /// the last forward when it was a verify block or a speculative append
    /// (`crate::cache::speculative_append_active`) of `L` rows, so
    /// [`Self::keep_verified_prefix`] can rewind to any accepted prefix
    /// without replaying it; `None` after any other forward.
    pub block_tails: [Option<UniquePtr<MlxArray>>; 4],
}

impl InklingLayerCache {
    pub fn new() -> Self {
        Self {
            kv: KVCache::new(),
            conv: std::array::from_fn(|_| None),
            block_tails: std::array::from_fn(|_| None),
        }
    }

    /// Whether the last forward was a verify block whose rows can be dropped
    /// by [`Self::keep_verified_prefix`].
    pub fn holds_block_tails(&self) -> bool {
        self.conv
            .iter()
            .zip(&self.block_tails)
            .all(|(state, tail)| state.is_none() || tail.is_some())
            && self.block_tails.iter().any(Option::is_some)
    }

    /// Rows the last verify block appended, read off its recorded windows:
    /// each window holds the `kernel - 1` rows the convolution state keeps
    /// plus the block's own rows. `None` when no window was recorded or the
    /// windows disagree.
    pub fn verify_block_rows(&self) -> Option<i32> {
        if !self.holds_block_tails() {
            return None;
        }
        let mut rows = None;
        for (state, tail) in self.conv.iter().zip(&self.block_tails) {
            let (Some(state), Some(tail)) = (state, tail) else {
                continue;
            };
            let appended = crate::array_shape(tail)[1] - crate::array_shape(state)[1];
            if appended < 1 || rows.is_some_and(|rows| rows != appended) {
                return None;
            }
            rows = Some(appended);
        }
        rows
    }

    /// Keep the first `kept` of the `appended` rows the last verify block
    /// added: the KV tail is trimmed and each convolution state becomes the
    /// `kernel - 1` input rows before row `kept`, which is the state the
    /// single-token chain holds after those rows (issue #3280). `kept == 0`
    /// drops the whole block and restores the state from before it.
    /// Refused, with nothing changed, when the last forward recorded no tails.
    ///
    /// Used by: `InklingModel::keep_verified_prefix_for_sequence`,
    /// [`drop_newest_verify_rows`].
    pub fn keep_verified_prefix(&mut self, appended: i32, kept: i32) -> Result<(), String> {
        if !self.holds_block_tails() {
            return Err("the last Inkling forward was not a verify block".into());
        }
        if !(0..=appended).contains(&kept) {
            return Err(format!("cannot keep {kept} of {appended} verify rows"));
        }
        for tail in self.block_tails.iter().flatten() {
            let rows = crate::array_shape(tail)[1];
            if rows <= appended {
                return Err(format!("verify tail of {rows} rows for {appended} rows"));
            }
        }
        for (state, tail) in self.conv.iter_mut().zip(self.block_tails.iter_mut()) {
            if let Some(tail) = tail.take() {
                let keep = crate::array_shape(&tail)[1] - appended;
                let kept_state = crate::utils::slice_axis(&tail, 1, kept, kept + keep);
                *state = Some(crate::contiguous(&kept_state, false));
            }
        }
        self.kv.trim(appended - kept);
        Ok(())
    }

    pub fn snapshot_into(&self, snapshot: &mut ModelStateSnapshot, prefix: &str) {
        if let Some((keys, values)) = self.kv.visible_state() {
            snapshot.push_tensor(format!("{prefix}.keys"), &keys);
            snapshot.push_tensor(format!("{prefix}.values"), &values);
        }
        let offset = crate::from_slice_i32(&[self.kv.offset], &[1]);
        if let Some(offset) = offset.as_ref() {
            snapshot.push_tensor(format!("{prefix}.offset"), offset);
        }
        for (index, state) in self.conv.iter().enumerate() {
            if let Some(state) = state.as_ref().and_then(|value| value.as_ref()) {
                snapshot.push_tensor(format!("{prefix}.conv{index}"), state);
            }
        }
    }

    pub fn restore_from(
        &mut self,
        snapshot: &ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        let keys = snapshot.tensor(&format!("{prefix}.keys")).map(crate::copy);
        let values = snapshot
            .tensor(&format!("{prefix}.values"))
            .map(crate::copy);
        let offset = snapshot
            .tensor(&format!("{prefix}.offset"))
            .map(crate::item_i32)
            .unwrap_or(snapshot.token_len() as i32);
        self.kv.restore_fp16_live_window(keys, values, offset)?;
        for (index, state) in self.conv.iter_mut().enumerate() {
            *state = snapshot
                .tensor(&format!("{prefix}.conv{index}"))
                .map(crate::copy);
        }
        Ok(())
    }

    /// Flatten an exact KV + convolution-state snapshot into the opaque MTP
    /// capture representation. The scalar pair is `(absolute_offset, flags)`;
    /// bit 0 denotes KV and bits 1..=4 denote the four convolution states.
    pub fn capture_flat(&self, tensors: &mut Vec<UniquePtr<MlxArray>>, scalars: &mut Vec<i32>) {
        let mut flags = 0_i32;
        if let Some((keys, values)) = self.kv.visible_state() {
            flags |= 1;
            tensors.push(crate::copy(&keys));
            tensors.push(crate::copy(&values));
        }
        for (index, state) in self.conv.iter().enumerate() {
            if let Some(state) = state.as_ref().and_then(|value| value.as_ref()) {
                flags |= 1 << (index + 1);
                tensors.push(crate::copy(state));
            }
        }
        scalars.extend([self.kv.offset, flags]);
    }

    /// Restore one layer from [`Self::capture_flat`].
    pub fn restore_flat(
        &mut self,
        tensors: &mut impl Iterator<Item = UniquePtr<MlxArray>>,
        scalars: &[i32],
        scalar_index: &mut usize,
    ) -> Result<(), String> {
        let offset = *scalars
            .get(*scalar_index)
            .ok_or_else(|| "Inkling snapshot is missing a cache offset".to_string())?;
        let flags = *scalars
            .get(*scalar_index + 1)
            .ok_or_else(|| "Inkling snapshot is missing cache flags".to_string())?;
        *scalar_index += 2;
        let (keys, values) = if flags & 1 != 0 {
            (
                Some(
                    tensors
                        .next()
                        .ok_or_else(|| "Inkling snapshot is missing KV keys".to_string())?,
                ),
                Some(
                    tensors
                        .next()
                        .ok_or_else(|| "Inkling snapshot is missing KV values".to_string())?,
                ),
            )
        } else {
            (None, None)
        };
        self.kv.restore_fp16_live_window(keys, values, offset)?;
        for (index, state) in self.conv.iter_mut().enumerate() {
            *state = if flags & (1 << (index + 1)) != 0 {
                Some(tensors.next().ok_or_else(|| {
                    format!("Inkling snapshot is missing convolution state {index}")
                })?)
            } else {
                None
            };
        }
        Ok(())
    }
}

/// Drop the newest `n` rows of the last verify block from every layer of one
/// sequence, through [`InklingLayerCache::keep_verified_prefix`]; `n` may be
/// the whole block (a failed round or a width warm-up unwinds every append).
/// Every layer is checked first, so a refusal changes nothing: a layer
/// without recorded windows, layers that disagree on the block length, or
/// `n` past the block.
///
/// Used by: `InklingModel::rewind_decode_appends` (prompt-lookup verify
/// rounds, issue #3422).
pub fn drop_newest_verify_rows(caches: &mut [InklingLayerCache], n: i32) -> Result<(), String> {
    if n <= 0 {
        return Ok(());
    }
    let mut appended = None;
    for (layer, cache) in caches.iter().enumerate() {
        let rows = cache
            .verify_block_rows()
            .ok_or_else(|| format!("Inkling layer {layer} recorded no verify-block windows"))?;
        if let Some(expected) = appended
            && expected != rows
        {
            return Err(format!(
                "Inkling layer {layer} recorded a {rows}-row verify block, not {expected}"
            ));
        }
        appended = Some(rows);
    }
    let Some(appended) = appended else {
        return Err("an Inkling sequence without layers has no verify block".into());
    };
    if n > appended {
        return Err(format!(
            "cannot drop {n} rows of a {appended}-row Inkling verify block"
        ));
    }
    for cache in caches.iter_mut() {
        cache.keep_verified_prefix(appended, appended - n)?;
    }
    Ok(())
}

impl Default for InklingLayerCache {
    fn default() -> Self {
        Self::new()
    }
}
