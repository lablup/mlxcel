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

//! The drafter's layers: context attention, MLP, DFlare fusion and the
//! decoder layer, ported from upstream `dspark.py` (`DraftAttention`,
//! `DraftMLP`, `DraftLayer`, `Model.mix`). The dtype handling follows
//! upstream exactly: f32 sums and activations, BF16 residuals and outputs.

use cxx::UniquePtr;

use crate::cache::KVCache;
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedLinear};
use crate::weights::WeightMap;

use super::config::Inkling2DsparkConfig;

pub(super) fn f32(x: &MlxArray) -> UniquePtr<MlxArray> {
    ffi::astype(x, dtype::FLOAT32)
}

pub(super) fn norm_weight(weights: &WeightMap, name: &str, eps: f32) -> Result<RMSNorm, String> {
    let weight = weights
        .get(name)
        .ok_or_else(|| format!("Inkling 2 DSpark weight not found: {name}"))?;
    Ok(RMSNorm::new(ffi::copy(weight), eps))
}

pub(super) fn linear(
    weights: &WeightMap,
    prefix: &str,
    quant: (i32, i32),
) -> Result<UnifiedLinear, String> {
    UnifiedLinear::from_weights(weights, prefix, quant.0, quant.1)
}

/// `[B, L, heads * D]` to `[B, heads, L, D]`.
fn heads(x: &MlxArray, heads: i32, head_dim: i32) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let split = ffi::reshape(x, &[shape[0], shape[1], heads, head_dim]);
    ffi::transpose_axes(&split, &[0, 2, 1, 3])
}

/// Context K/V of one drafter layer. Only target context is ever written;
/// proposals never are. A local layer keeps the newest `window` keys, which
/// is what upstream reads back from its `RotatingKVCache`; a global layer
/// keeps every context row in a growing [`KVCache`].
pub struct DraftContextCache {
    /// Global layers.
    kv: KVCache,
    /// Local layers: the newest `window` rope'd keys and values.
    local: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    window: Option<i32>,
    /// Positions written so far, counting rows a local layer skipped.
    offset: i32,
}

impl DraftContextCache {
    pub fn new(window: Option<i32>) -> Self {
        Self {
            kv: KVCache::new(),
            local: None,
            window,
            offset: 0,
        }
    }

    /// The RoPE offset of the next context row.
    pub fn offset(&self) -> i32 {
        self.offset
    }

    fn append(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.offset += ffi::array_shape(&keys)[2];
        let Some(window) = self.window else {
            return self.kv.update_and_fetch(keys, values);
        };
        let (keys, values) = match self.local.take() {
            None => (keys, values),
            Some((old_keys, old_values)) => (
                crate::ops::concatenate(&old_keys, &keys, 2),
                crate::ops::concatenate(&old_values, &values, 2),
            ),
        };
        let shape = ffi::array_shape(&keys);
        let (keys, values) = if shape[2] > window {
            let start = [0, 0, shape[2] - window, 0];
            let stop = [shape[0], shape[1], shape[2], shape[3]];
            // Materialized so the window does not pin the longer buffer.
            (
                ffi::contiguous(&ffi::slice(&keys, &start, &stop), false),
                ffi::contiguous(&ffi::slice(&values, &start, &stop), false),
            )
        } else {
            (keys, values)
        };
        let fetched = (ffi::copy(&keys), ffi::copy(&values));
        self.local = Some((keys, values));
        fetched
    }

    fn skip(&mut self, rows: i32) {
        self.offset += rows;
    }
}

/// Upstream `DraftAttention`.
pub struct DraftAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    k_proj_ctx: UnifiedLinear,
    v_proj_ctx: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    k_norm_ctx: RMSNorm,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rope_theta: f32,
    pub(super) window: Option<i32>,
}

impl DraftAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Inkling2DsparkConfig,
        local: bool,
        quant: (i32, i32),
    ) -> Result<Self, String> {
        let eps = config.rms_norm_eps;
        let proj = |name: &str| linear(weights, &format!("{prefix}.{name}"), quant);
        let norm = |name: &str| norm_weight(weights, &format!("{prefix}.{name}.weight"), eps);
        Ok(Self {
            q_proj: proj("q_proj")?,
            k_proj: proj("k_proj")?,
            v_proj: proj("v_proj")?,
            k_proj_ctx: proj("k_proj_ctx")?,
            v_proj_ctx: proj("v_proj_ctx")?,
            o_proj: proj("o_proj")?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            k_norm_ctx: norm("k_norm_ctx")?,
            heads: config.num_attention_heads as i32,
            kv_heads: config.num_key_value_heads as i32,
            head_dim: config.head_dim as i32,
            rope_theta: config.rope_theta,
            window: local.then_some(config.sliding_window_size as i32),
        })
    }

    fn rope(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        ffi::fast_rope(x, self.head_dim, false, self.rope_theta, 1.0, offset)
    }

    /// Every block row sees the whole block and the same context window
    /// before its anchor; only `context` enters the persistent cache.
    pub(super) fn forward(
        &self,
        x: &MlxArray,
        context: &MlxArray,
        cache: &mut DraftContextCache,
    ) -> UniquePtr<MlxArray> {
        let ctx_shape = ffi::array_shape(context);
        let mut context_rows = ctx_shape[1];
        let windowed;
        let context = match self.window {
            Some(window) if context_rows > window => {
                let skip = context_rows - window;
                windowed = ffi::slice(
                    context,
                    &[0, skip, 0],
                    &[ctx_shape[0], context_rows, ctx_shape[2]],
                );
                cache.skip(skip);
                context_rows = window;
                windowed.as_ref().expect("windowed context")
            }
            _ => context,
        };
        let offset = cache.offset();
        let keys = self.k_norm_ctx.forward(&heads(
            &self.k_proj_ctx.forward(context),
            self.kv_heads,
            self.head_dim,
        ));
        let values = heads(
            &self.v_proj_ctx.forward(context),
            self.kv_heads,
            self.head_dim,
        );
        let (keys, values) = cache.append(self.rope(&keys, offset), values);
        let offset = offset + context_rows;

        let queries =
            self.q_norm
                .forward(&heads(&self.q_proj.forward(x), self.heads, self.head_dim));
        let block_keys = self.k_norm.forward(&heads(
            &self.k_proj.forward(x),
            self.kv_heads,
            self.head_dim,
        ));
        let block_values = heads(&self.v_proj.forward(x), self.kv_heads, self.head_dim);
        let keys = crate::ops::concatenate(&keys, &self.rope(&block_keys, offset), 2);
        let values = crate::ops::concatenate(&values, &block_values, 2);
        let scale = (self.head_dim as f32).powf(-0.5);
        // SAFETY: q, k and v are live for the call; a null mask means no mask.
        let out = unsafe {
            ffi::fast_scaled_dot_product_attention(
                &self.rope(&queries, offset),
                &keys,
                &values,
                scale,
                std::ptr::null(),
            )
        };
        let shape = ffi::array_shape(x);
        let out = ffi::reshape(
            &ffi::transpose_axes(&out, &[0, 2, 1, 3]),
            &[shape[0], shape[1], self.heads * self.head_dim],
        );
        self.o_proj.forward(&out)
    }
}

/// Upstream `DraftMLP`: the gate and up products are activated in f32.
pub struct DraftMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl DraftMlp {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = f32(&self.gate_proj.forward(x));
        let up = f32(&self.up_proj.forward(x));
        let hidden = ffi::astype(&ffi::multiply(&ffi::silu(&gate), &up), ffi::array_dtype(x));
        self.down_proj.forward(&hidden)
    }
}

/// Upstream `DraftLayer`: fused add/RMSNorm over the unrounded f32 sum, with
/// the residual stored in the activation dtype.
pub struct DraftLayer {
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pub(super) dflare_norm: RMSNorm,
    pub(super) self_attn: DraftAttention,
    mlp: DraftMlp,
}

impl DraftLayer {
    pub(super) fn from_weights(
        weights: &WeightMap,
        index: usize,
        config: &Inkling2DsparkConfig,
        quant: (i32, i32),
    ) -> Result<Self, String> {
        let prefix = format!("layers.{index}");
        let eps = config.rms_norm_eps;
        let norm = |name: &str| norm_weight(weights, &format!("{prefix}.{name}.weight"), eps);
        let mlp = |name: &str| linear(weights, &format!("{prefix}.mlp.{name}"), quant);
        Ok(Self {
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            dflare_norm: norm("dflare_norm")?,
            self_attn: DraftAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                config.local_layer_ids.contains(&index),
                quant,
            )?,
            mlp: DraftMlp {
                gate_proj: mlp("gate_proj")?,
                up_proj: mlp("up_proj")?,
                down_proj: mlp("down_proj")?,
            },
        })
    }

    /// `(output, residual)`, as upstream returns them.
    pub(super) fn forward(
        &self,
        x: &MlxArray,
        residual: Option<&MlxArray>,
        context: &MlxArray,
        cache: &mut DraftContextCache,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let dtype = ffi::array_dtype(x);
        let summed = match residual {
            None => ffi::copy(x),
            Some(residual) => ffi::add(&f32(x), &f32(residual)),
        };
        let residual = ffi::astype(&summed, dtype);
        let normalized = ffi::astype(&self.input_layernorm.forward(&summed), dtype);
        let attended = self.self_attn.forward(&normalized, context, cache);
        let dtype = ffi::array_dtype(&attended);
        let summed = ffi::add(&f32(&attended), &f32(&residual));
        let normalized = ffi::astype(&self.post_attention_layernorm.forward(&summed), dtype);
        (self.mlp.forward(&normalized), ffi::astype(&summed, dtype))
    }
}

/// DFlare fusion weights: `softmax(fuse_weight[row])`, rounded to the
/// hidden dtype as upstream does, then held in f32 for the weighted sum.
pub struct FuseWeights {
    /// `[rows][taps]` scalars, each a `[1]` f32 array.
    rows: Vec<Vec<UniquePtr<MlxArray>>>,
    taps: usize,
    hidden_size: i32,
}

impl FuseWeights {
    pub(super) fn new(
        fuse_weight: &MlxArray,
        hidden_dtype: i32,
        config: &Inkling2DsparkConfig,
    ) -> Result<Self, String> {
        let taps = config.target_layer_ids.len();
        let rows = config.num_hidden_layers + 1;
        let shape = ffi::array_shape(fuse_weight);
        if shape != [rows as i32, taps as i32] {
            return Err(format!(
                "Inkling 2 DSpark fuse_weight must be [{rows}, {taps}] (num_hidden_layers + 1, \
                 len(target_layer_ids)), got {shape:?}"
            ));
        }
        let alpha = f32(&ffi::astype(
            &ffi::softmax(&f32(fuse_weight), -1),
            hidden_dtype,
        ));
        let values = crate::utils::array_to_vec_f32(&alpha);
        let rows = values
            .chunks_exact(taps)
            .map(|row| {
                row.iter()
                    .map(|&value| ffi::from_slice_f32(&[value], &[1]))
                    .collect()
            })
            .collect();
        Ok(Self {
            rows,
            taps,
            hidden_size: config.hidden_size as i32,
        })
    }

    /// Upstream `Model.mix`: `hidden` is `[B, L, taps * hidden]`; `row`
    /// selects the drafter layer (the last row is the Markov head's).
    pub(super) fn mix(&self, hidden: &MlxArray, row: usize, norm: &RMSNorm) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(hidden);
        let features = f32(&ffi::reshape(
            hidden,
            &[shape[0], shape[1], self.taps as i32, self.hidden_size],
        ));
        let tap = |i: i32| {
            let picked = ffi::slice(
                &features,
                &[0, 0, i, 0],
                &[shape[0], shape[1], i + 1, self.hidden_size],
            );
            ffi::squeeze_axis(&picked, 2)
        };
        let weights = &self.rows[row];
        // Upstream starts from zeros; 0 + a == a, so the first term is exact.
        let mut mixed = ffi::multiply(&tap(0), &weights[0]);
        for (i, weight) in weights.iter().enumerate().skip(1) {
            mixed = ffi::add(&mixed, &ffi::multiply(&tap(i as i32), weight));
        }
        ffi::astype(&norm.forward(&mixed), ffi::array_dtype(hidden))
    }

    pub(super) fn last_row(&self) -> usize {
        self.rows.len() - 1
    }
}
