use crate::generate::ModelStateSnapshot;
use crate::layers::KVCache;
use crate::{MlxArray, UniquePtr};

/// Inkling's per-layer KV cache plus four causal short-convolution states.
pub struct InklingLayerCache {
    pub kv: KVCache,
    pub conv: [Option<UniquePtr<MlxArray>>; 4],
}

impl InklingLayerCache {
    pub fn new() -> Self {
        Self {
            kv: KVCache::new(),
            conv: std::array::from_fn(|_| None),
        }
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

impl Default for InklingLayerCache {
    fn default() -> Self {
        Self::new()
    }
}
