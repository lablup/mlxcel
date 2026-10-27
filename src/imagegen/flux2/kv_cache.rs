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

//! Reference-image K/V cache for `FLUX.2-klein-9b-kv` edits.
//!
//! A port of `Flux2KVLayerCache` and `Flux2KVCache` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/transformer/kv_cache.py>:
//! step 0 runs the reference tokens through every block at a fixed timestep
//! and stores their rotated keys and values; later steps attend to the cached
//! entries instead of recomputing them.

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

/// How a transformer call treats the reference K/V cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheMode {
    /// Reference tokens are in the sequence: store their K/V per layer.
    Extract { num_ref_tokens: i32 },
    /// Reference tokens are absent: splice the stored K/V into attention.
    Cached,
}

/// Keys and values of the reference tokens at one layer, `[B, H, R, D]`.
#[derive(Default)]
pub struct KvLayerCache {
    entry: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
}

impl KvLayerCache {
    pub fn store(&mut self, k_ref: UniquePtr<MlxArray>, v_ref: UniquePtr<MlxArray>) {
        self.entry = Some((k_ref, v_ref));
    }

    pub fn get(&self) -> Result<(&MlxArray, &MlxArray)> {
        match &self.entry {
            Some((k, v)) => Ok((k, v)),
            None => bail!("FLUX.2 KV cache has not been populated"),
        }
    }

    #[must_use]
    pub fn is_populated(&self) -> bool {
        self.entry.is_some()
    }

    pub fn clear(&mut self) {
        self.entry = None;
    }
}

/// Per-layer reference K/V for the double- and single-stream stacks.
pub struct Flux2KvCache {
    pub double: Vec<KvLayerCache>,
    pub single: Vec<KvLayerCache>,
    pub num_ref_tokens: i32,
}

impl Flux2KvCache {
    #[must_use]
    pub fn new(num_double_layers: usize, num_single_layers: usize) -> Self {
        Self {
            double: (0..num_double_layers)
                .map(|_| KvLayerCache::default())
                .collect(),
            single: (0..num_single_layers)
                .map(|_| KvLayerCache::default())
                .collect(),
            num_ref_tokens: 0,
        }
    }

    /// Every stored array, for one evaluation after the extraction step.
    #[must_use]
    pub fn arrays(&self) -> Vec<&MlxArray> {
        self.double
            .iter()
            .chain(&self.single)
            .filter_map(|c| c.entry.as_ref())
            .flat_map(|(k, v)| [&**k, &**v])
            .collect()
    }

    /// Whether every layer holds an entry.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.double
            .iter()
            .chain(&self.single)
            .all(KvLayerCache::is_populated)
    }

    pub fn clear(&mut self) {
        for c in self.double.iter_mut().chain(self.single.iter_mut()) {
            c.clear();
        }
        self.num_ref_tokens = 0;
    }
}
