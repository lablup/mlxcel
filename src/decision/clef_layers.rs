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

//! The Clef head's building blocks: `nn.Linear`, `nn.gelu`,
//! `nn.MultiHeadAttention`, `EvidenceRoutingLayer` and `FieldDecoderLayer`
//! of mlx-vlm's `mlx_vlm/models/clef/clef.py`, each in the dtype of its
//! inputs with upstream's rounding points, and the tensor lookup that builds
//! them from the sanitized head weights.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::{MlxArray, UniquePtr};

use super::clef_head::{LAYER_NORM_EPS, scalar_like};

/// `nn.Linear`: `addmm(bias, x, W.T)`, one rounding in the activation dtype
/// as upstream (a matmul followed by an add rounds twice).
pub(super) struct Linear {
    pub(super) weight: UniquePtr<MlxArray>,
    pub(super) bias: Option<UniquePtr<MlxArray>>,
}

impl Linear {
    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let weight = mlxcel_core::transpose(&self.weight);
        match &self.bias {
            Some(bias) => mlxcel_core::addmm(bias, x, &weight, 1.0, 1.0),
            None => mlxcel_core::matmul(x, &weight),
        }
    }
}

/// `nn.gelu`: `x * (1 + erf(x / sqrt(2))) / 2` with every step in `x`'s
/// dtype, as upstream's compiled helper (the shared f32 GELU rounds once).
pub(super) fn gelu(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(x);
    let erf = mlxcel_core::erf(&mlxcel_core::divide(
        x,
        &scalar_like(std::f32::consts::SQRT_2, dtype),
    ));
    let lifted = mlxcel_core::add(&scalar_like(1.0, dtype), &erf);
    mlxcel_core::divide(&mlxcel_core::multiply(x, &lifted), &scalar_like(2.0, dtype))
}

pub(super) struct Weights<'a> {
    pub(super) map: &'a HashMap<String, UniquePtr<MlxArray>>,
}

impl Weights<'_> {
    pub(super) fn tensor(&self, key: &str) -> Result<UniquePtr<MlxArray>> {
        self.map
            .get(key)
            .map(|value| mlxcel_core::copy(value))
            .ok_or_else(|| anyhow!("clef head tensor `{key}` is missing from the checkpoint"))
    }

    pub(super) fn linear(&self, prefix: &str, bias: bool) -> Result<Linear> {
        let weight = self.tensor(&format!("{prefix}.weight"))?;
        let bias = if bias {
            Some(self.tensor(&format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    }

    pub(super) fn norm(&self, prefix: &str) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.tensor(&format!("{prefix}.weight"))?,
            Some(self.tensor(&format!("{prefix}.bias"))?),
            LAYER_NORM_EPS,
        ))
    }

    pub(super) fn attention(&self, prefix: &str, heads: usize) -> Result<Attention> {
        Ok(Attention {
            query: self.linear(&format!("{prefix}.query_proj"), true)?,
            key: self.linear(&format!("{prefix}.key_proj"), true)?,
            value: self.linear(&format!("{prefix}.value_proj"), true)?,
            out: self.linear(&format!("{prefix}.out_proj"), true)?,
            heads: heads as i32,
        })
    }
}

/// `nn.MultiHeadAttention(width, heads, bias=True)` without a mask.
pub(super) struct Attention {
    pub(super) query: Linear,
    pub(super) key: Linear,
    pub(super) value: Linear,
    pub(super) out: Linear,
    pub(super) heads: i32,
}

impl Attention {
    /// `queries` `[B, L, W]`, `memory` `[B, S, W]` (keys and values).
    pub(super) fn forward(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let split = |x: &MlxArray| {
            let shape = mlxcel_core::array_shape(x);
            let heads = mlxcel_core::reshape(x, &[shape[0], shape[1], self.heads, -1]);
            mlxcel_core::transpose_axes(&heads, &[0, 2, 1, 3])
        };
        let q = split(&self.query.forward(queries));
        let k = split(&self.key.forward(keys));
        let v = split(&self.value.forward(values));
        let head_dim = *mlxcel_core::array_shape(&q).last().unwrap_or(&1);
        let scale = (1.0 / head_dim as f32).sqrt();
        // SAFETY: q, k and v are live arrays borrowed for the call; a null
        // mask pointer is the documented "no mask" value.
        let output = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, scale, std::ptr::null())
        };
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let shape = mlxcel_core::array_shape(&output);
        let merged = mlxcel_core::reshape(&output, &[shape[0], shape[1], -1]);
        self.out.forward(&merged)
    }
}

/// `EvidenceRoutingLayer`: options attend to the projected memory.
pub(super) struct EvidenceRoutingLayer {
    pub(super) query_norm: LayerNorm,
    pub(super) memory_norm: LayerNorm,
    pub(super) attention: Attention,
    pub(super) feedforward_norm: LayerNorm,
    pub(super) feedforward: [Linear; 2],
}

impl EvidenceRoutingLayer {
    pub(super) fn forward(&self, queries: &MlxArray, memory: &MlxArray) -> UniquePtr<MlxArray> {
        let memory = self.memory_norm.forward(memory);
        let attended = self
            .attention
            .forward(&self.query_norm.forward(queries), &memory, &memory);
        let queries = mlxcel_core::add(queries, &attended);
        let hidden = gelu(&self.feedforward[0].forward(&self.feedforward_norm.forward(&queries)));
        mlxcel_core::add(&queries, &self.feedforward[1].forward(&hidden))
    }
}

/// `FieldDecoderLayer`: pre-norm self-attention, cross-attention, GELU FFN.
pub(super) struct FieldDecoderLayer {
    pub(super) self_attn: Attention,
    pub(super) multihead_attn: Attention,
    pub(super) linear1: Linear,
    pub(super) linear2: Linear,
    pub(super) norm1: LayerNorm,
    pub(super) norm2: LayerNorm,
    pub(super) norm3: LayerNorm,
}

impl FieldDecoderLayer {
    pub(super) fn forward(&self, x: &MlxArray, memory: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.norm1.forward(x);
        let x = mlxcel_core::add(x, &self.self_attn.forward(&h, &h, &h));
        let x = mlxcel_core::add(
            &x,
            &self
                .multihead_attn
                .forward(&self.norm2.forward(&x), memory, memory),
        );
        let hidden = gelu(&self.linear1.forward(&self.norm3.forward(&x)));
        mlxcel_core::add(&x, &self.linear2.forward(&hidden))
    }
}
