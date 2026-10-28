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

//! The reference KV cache of an edit, a port of `kv_cache.py` (mlx-vlm
//! `73d366ec`).
//!
//! With `causal_condition`, no prompt or reference token attends to the target
//! (the mask is block causal and the target block comes last), and those
//! tokens carry the zero-timestep modulation, so their keys and values are the
//! same at every denoising step. The first step runs the whole sequence and
//! stores each layer's prefix keys and values (`extract`); later steps feed
//! only the target latents and attend over the stored prefix (`cached`). One
//! cache belongs to one conditioning branch of one edit.

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr};

/// `(frames, height, width)` of one latent grid.
pub type ImageShape = (i32, i32, i32);

/// How a transformer call uses a [`KvCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMode {
    /// Run the full sequence and store the prefix keys and values.
    Extract,
    /// Run only the target latents against the stored prefix.
    Cached,
}

/// Keys and values `[B, heads, prefix_len, head_dim]` of one layer.
#[derive(Default)]
pub struct KvLayer {
    key: Option<UniquePtr<MlxArray>>,
    value: Option<UniquePtr<MlxArray>>,
}

impl KvLayer {
    /// Keep the first `prefix_len` positions, compacted (upstream
    /// `mx.contiguous`), so the full-sequence buffers can be released.
    pub fn store(&mut self, key: &MlxArray, value: &MlxArray, prefix_len: i32) {
        let prefix = |a: &MlxArray| {
            let s = mlxcel_core::array_shape(a);
            let sliced = mlxcel_core::slice(a, &[0, 0, 0, 0], &[s[0], s[1], prefix_len, s[3]]);
            mlxcel_core::contiguous(&sliced, false)
        };
        self.key = Some(prefix(key));
        self.value = Some(prefix(value));
    }

    pub fn get(&self) -> Result<(&MlxArray, &MlxArray)> {
        match (&self.key, &self.value) {
            (Some(k), Some(v)) => Ok((k, v)),
            _ => bail!("Qwen-Image KV cache has not been populated"),
        }
    }
}

/// Per-edit cache for the fixed prompt and reference prefix.
pub struct KvCache {
    pub layers: Vec<KvLayer>,
    pub prefix_len: i32,
    pub target_shape: Option<ImageShape>,
    /// `[B, target_len, in_channels]` and dtype of the target input.
    pub input_shape: Option<Vec<i32>>,
    pub dtype: Option<i32>,
    /// RoPE rows of the target tokens.
    pub cos: Option<UniquePtr<MlxArray>>,
    pub sin: Option<UniquePtr<MlxArray>>,
}

impl KvCache {
    #[must_use]
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| KvLayer::default()).collect(),
            prefix_len: 0,
            target_shape: None,
            input_shape: None,
            dtype: None,
            cos: None,
            sin: None,
        }
    }

    /// Every stored array, to evaluate with the first step.
    #[must_use]
    pub fn arrays(&self) -> Vec<&MlxArray> {
        let mut out: Vec<&MlxArray> = Vec::new();
        for layer in &self.layers {
            out.extend(layer.key.as_deref());
            out.extend(layer.value.as_deref());
        }
        out.extend(self.cos.as_deref());
        out.extend(self.sin.as_deref());
        out
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.key = None;
            layer.value = None;
        }
        self.prefix_len = 0;
        self.target_shape = None;
        self.input_shape = None;
        self.dtype = None;
        self.cos = None;
        self.sin = None;
    }
}
