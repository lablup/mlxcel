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

//! LLaDA-Image prompt conditioning: the QueryFormer, the text projection and
//! the prompt template.
//!
//! A port of `conditioning.py` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llada_image/conditioning.py>.
//! The prompt's word embeddings feed a one-block QueryFormer whose 256
//! learned queries cross-attend them; the queries are appended to the
//! embeddings, the LLaDA-2 backbone encodes the whole sequence under
//! [`query_attention_mask`], and a six-block projection maps the hidden
//! states to the transformer's caption width.
//!
//! The arithmetic is upstream's, op for op: biased linears are one `addmm`
//! (a matmul plus a separate bias add rounds twice), `gelu_approx` and
//! `gelu` are their MLX formulas with dtype-matched constants, the
//! affine-free LayerNorms and weightless RMS norms are MLX's fast kernels.
//! These helpers are shared with [`super::sigvq`] and [`super::transformer`].

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::{QueryFormerConfig, TextProjectionConfig};

/// The prompt template (`format_prompt`). `None` is the unconditional
/// prompt "Generate an image.".
#[must_use]
pub fn format_prompt(prompt: Option<&str>) -> String {
    let instruction = match prompt {
        Some(p) => format!("Generate an image: {}", p.trim()),
        None => "Generate an image.".to_string(),
    };
    format!("<role>HUMAN</role> {instruction}\n<role>ASSISTANT</role>\n<IMAGE1>")
}

/// `query_attention_mask(T, Q)`: text attends text; the appended queries
/// attend everything. Row-major `[(T + Q)^2]` booleans.
#[must_use]
pub fn query_attention_mask_values(text_length: usize, query_length: usize) -> Vec<bool> {
    let n = text_length + query_length;
    (0..n)
        .flat_map(|q| (0..n).map(move |k| q >= text_length || k < text_length))
        .collect()
}

/// [`query_attention_mask_values`] as a `[1, 1, T + Q, T + Q]` bool array.
#[must_use]
pub fn query_attention_mask(text_length: usize, query_length: usize) -> UniquePtr<MlxArray> {
    let n = (text_length + query_length) as i32;
    bool_array(
        &query_attention_mask_values(text_length, query_length),
        &[1, 1, n, n],
    )
}

/// A bool array from host booleans.
#[must_use]
pub fn bool_array(values: &[bool], shape: &[i32]) -> UniquePtr<MlxArray> {
    let ints: Vec<i32> = values.iter().map(|&b| i32::from(b)).collect();
    mlxcel_core::astype(&mlxcel_core::from_slice_i32(&ints, shape), dtype::BOOL)
}

pub(crate) fn tensor(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .with_context(|| format!("LLaDA-Image weight not found: {key}"))
}

/// Require `key` to have `shape`.
pub(crate) fn shaped(weights: &WeightMap, key: &str, shape: &[i32]) -> Result<UniquePtr<MlxArray>> {
    let t = tensor(weights, key)?;
    let got = mlxcel_core::array_shape(&t);
    if got != shape {
        bail!("LLaDA-Image weight {key} has shape {got:?}, expected {shape:?}");
    }
    Ok(t)
}

fn scalar(value: f64, dt: i32) -> UniquePtr<MlxArray> {
    // A Python float scalar is a weak operand: it takes the array's dtype.
    mlxcel_core::full_f32(&[], value as f32, dt)
}

/// `nn.Linear(in, out, bias)`: `addmm(bias, x, W.T)` with a bias, `x @ W.T`
/// without.
pub struct Dense {
    weight_t: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl Dense {
    /// `<prefix>.weight` (`[out, in]`) and, when `bias`, `<prefix>.bias`.
    pub fn load(
        weights: &WeightMap,
        prefix: &str,
        in_dim: i32,
        out_dim: i32,
        bias: bool,
    ) -> Result<Self> {
        let weight = shaped(weights, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
        Self::from_parts(
            &weight,
            if bias {
                Some(shaped(weights, &format!("{prefix}.bias"), &[out_dim])?)
            } else {
                None
            },
        )
    }

    /// From an `[out, in]` weight and an optional `[out]` bias.
    pub fn from_parts(weight: &MlxArray, bias: Option<UniquePtr<MlxArray>>) -> Result<Self> {
        if mlxcel_core::array_shape(weight).len() != 2 {
            bail!("a dense weight must be 2-D");
        }
        Ok(Self {
            weight_t: mlxcel_core::transpose(weight),
            bias,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match &self.bias {
            Some(bias) => mlxcel_core::addmm(bias, x, &self.weight_t, 1.0, 1.0),
            None => mlxcel_core::matmul(x, &self.weight_t),
        }
    }
}

/// `nn.gelu_approx`: `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))`.
#[must_use]
pub fn gelu_approx(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let cubed = mlxcel_core::power(x, &scalar(3.0, dt));
    let inner = mlxcel_core::add(x, &mlxcel_core::multiply(&scalar(0.044715, dt), &cubed));
    let tanh = mlxcel_core::tanh(&mlxcel_core::multiply(
        &scalar((2.0 / std::f64::consts::PI).sqrt(), dt),
        &inner,
    ));
    let half_x = mlxcel_core::multiply(&scalar(0.5, dt), x);
    mlxcel_core::multiply(&half_x, &mlxcel_core::add(&scalar(1.0, dt), &tanh))
}

/// `nn.gelu`: `x * (1 + erf(x / sqrt(2))) / 2`.
#[must_use]
pub fn gelu(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let erf = mlxcel_core::erf(&mlxcel_core::divide(
        x,
        &scalar(std::f64::consts::SQRT_2, dt),
    ));
    let product = mlxcel_core::multiply(x, &mlxcel_core::add(&scalar(1.0, dt), &erf));
    mlxcel_core::divide(&product, &scalar(2.0, dt))
}

/// `nn.LayerNorm(dim, eps, affine=False)`.
#[must_use]
pub fn plain_layer_norm(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    // SAFETY: null weight and bias select the affine-free normalization.
    unsafe { mlxcel_core::fast_layer_norm(x, std::ptr::null(), std::ptr::null(), eps) }
}

/// `nn.LayerNorm(dim, eps)` with its learned weight and bias.
pub struct AffineLayerNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    eps: f32,
}

impl AffineLayerNorm {
    pub fn load(weights: &WeightMap, prefix: &str, dim: i32, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: shaped(weights, &format!("{prefix}.weight"), &[dim])?,
            bias: shaped(weights, &format!("{prefix}.bias"), &[dim])?,
            eps,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // SAFETY: both pointers come from arrays owned by `self`.
        unsafe { mlxcel_core::fast_layer_norm(x, &*self.weight, &*self.bias, self.eps) }
    }
}

/// `RMSNorm(eps)`: `mx.fast.rms_norm(x, None, eps)`.
#[must_use]
pub fn plain_rms_norm(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::fast_rms_norm_no_weight(x, eps)
}

/// Upstream `attention(q, k, v, heads, mask)`: split heads, SDPA at
/// `head_dim ** -0.5`, merge heads. Runs through
/// [`mlxcel_core::layers::attention`], which query-chunks a materialized
/// score matrix at large sizes.
#[must_use]
pub fn attention(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    heads: i32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(q);
    let (b, l, d) = (shape[0], shape[1], shape[2]);
    let head_dim = d / heads;
    let split = |t: &MlxArray| {
        mlxcel_core::transpose_axes(
            &mlxcel_core::reshape(t, &[b, -1, heads, head_dim]),
            &[0, 2, 1, 3],
        )
    };
    let scale = f64::from(head_dim).powf(-0.5) as f32;
    let out = mlxcel_core::layers::attention(&split(q), &split(k), &split(v), scale, mask, 0.0, 0);
    mlxcel_core::reshape(
        &mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]),
        &[b, l, d],
    )
}

/// `MLP`: `fc2(gelu_approx(fc1(x)))`.
pub struct Mlp {
    fc1: Dense,
    fc2: Dense,
}

impl Mlp {
    pub fn load(weights: &WeightMap, prefix: &str, hidden: i32, intermediate: i32) -> Result<Self> {
        Ok(Self {
            fc1: Dense::load(
                weights,
                &format!("{prefix}.fc1"),
                hidden,
                intermediate,
                true,
            )?,
            fc2: Dense::load(
                weights,
                &format!("{prefix}.fc2"),
                intermediate,
                hidden,
                true,
            )?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.fc2.forward(&gelu_approx(&self.fc1.forward(x)))
    }
}

struct QueryAttention {
    q: Dense,
    k: Dense,
    v: Dense,
    out_proj: Dense,
    heads: i32,
}

impl QueryAttention {
    fn load(weights: &WeightMap, prefix: &str, dim: i32, heads: i32) -> Result<Self> {
        let w = shaped(
            weights,
            &format!("{prefix}.in_proj_weight"),
            &[3 * dim, dim],
        )?;
        let b = shaped(weights, &format!("{prefix}.in_proj_bias"), &[3 * dim])?;
        let part = |i: i32| -> Result<Dense> {
            let weight = mlxcel_core::slice(&w, &[i * dim, 0], &[(i + 1) * dim, dim]);
            let bias = mlxcel_core::slice(&b, &[i * dim], &[(i + 1) * dim]);
            Dense::from_parts(&weight, Some(bias))
        };
        Ok(Self {
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out_proj: Dense::load(weights, &format!("{prefix}.out_proj"), dim, dim, true)?,
            heads,
        })
    }

    fn forward(
        &self,
        queries: &MlxArray,
        context: &MlxArray,
        mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let q = self.q.forward(queries);
        let k = self.k.forward(context);
        let v = self.v.forward(context);
        let shape = mlxcel_core::array_shape(mask);
        let mask = mlxcel_core::reshape(mask, &[shape[0], 1, 1, shape[1]]);
        self.out_proj
            .forward(&attention(&q, &k, &v, self.heads, Some(&mask)))
    }
}

struct QueryFormerBlock {
    cross_attn: QueryAttention,
    mlp: Mlp,
    eps: f32,
}

impl QueryFormerBlock {
    fn forward(
        &self,
        queries: &MlxArray,
        context: &MlxArray,
        mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        // Both residuals use normalized queries in the reference implementation.
        let queries = plain_layer_norm(queries, self.eps);
        let context = plain_layer_norm(context, self.eps);
        let queries =
            mlxcel_core::add(&queries, &self.cross_attn.forward(&queries, &context, mask));
        let queries = plain_layer_norm(&queries, self.eps);
        mlxcel_core::add(&queries, &self.mlp.forward(&queries))
    }
}

/// `QueryFormer`: learned queries that read the prompt embeddings.
pub struct QueryFormer {
    meta_queries: UniquePtr<MlxArray>,
    blocks: Vec<QueryFormerBlock>,
    config: QueryFormerConfig,
}

impl QueryFormer {
    pub fn from_weights(config: QueryFormerConfig, weights: &WeightMap) -> Result<Self> {
        let dim = config.hidden_size;
        let blocks = (0..config.num_hidden_layers)
            .map(|i| {
                let prefix = format!("query_blocks.{i}");
                Ok(QueryFormerBlock {
                    cross_attn: QueryAttention::load(
                        weights,
                        &format!("{prefix}.cross_attn"),
                        dim,
                        config.num_attention_heads,
                    )?,
                    mlp: Mlp::load(
                        weights,
                        &format!("{prefix}.mlp"),
                        dim,
                        config.intermediate_size,
                    )?,
                    eps: config.norm_eps,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            meta_queries: shaped(weights, "meta_queries", &[config.num_queries, dim])?,
            blocks,
            config,
        })
    }

    #[must_use]
    pub fn num_queries(&self) -> i32 {
        self.config.num_queries
    }

    /// `embeddings` `[B, T, D]`, `mask` `[B, T]` bool; returns `[B, Q, D]`.
    pub fn forward(&self, embeddings: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let b = mlxcel_core::array_shape(embeddings)[0];
        let mut queries = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&self.meta_queries, 0),
            &[b, self.config.num_queries, self.config.hidden_size],
        );
        for block in &self.blocks {
            queries = block.forward(&queries, embeddings, mask);
        }
        queries
    }
}

struct ProjectionBlock {
    q_proj: Dense,
    k_proj: Dense,
    v_proj: Dense,
    out_proj: Dense,
    mlp: Mlp,
    heads: i32,
    eps: f32,
}

impl ProjectionBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = plain_rms_norm(x, self.eps);
        let shape = mlxcel_core::array_shape(&h);
        let heads = [shape[0], shape[1], self.heads, -1];
        let head_norm = |t: &MlxArray| {
            mlxcel_core::reshape(
                &plain_rms_norm(&mlxcel_core::reshape(t, &heads), self.eps),
                &shape,
            )
        };
        let q = head_norm(&self.q_proj.forward(&h));
        let k = head_norm(&self.k_proj.forward(&h));
        let v = self.v_proj.forward(&h);
        let attended = self
            .out_proj
            .forward(&attention(&q, &k, &v, self.heads, None));
        let x = mlxcel_core::add(x, &attended);
        mlxcel_core::add(&x, &self.mlp.forward(&plain_rms_norm(&x, self.eps)))
    }
}

/// `TextProjection`: the backbone's hidden states to the caption width.
pub struct TextProjection {
    layers: Vec<ProjectionBlock>,
    projector: Dense,
}

impl TextProjection {
    pub fn from_weights(config: &TextProjectionConfig, weights: &WeightMap) -> Result<Self> {
        let dim = config.hidden_size;
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                let p = format!("layers.{i}");
                let dense = |name: &str| {
                    Dense::load(weights, &format!("{p}.self_attn.{name}"), dim, dim, true)
                };
                Ok(ProjectionBlock {
                    q_proj: dense("q_proj")?,
                    k_proj: dense("k_proj")?,
                    v_proj: dense("v_proj")?,
                    out_proj: dense("out_proj")?,
                    mlp: Mlp::load(weights, &format!("{p}.mlp"), dim, config.intermediate_size)?,
                    heads: config.num_attention_heads,
                    eps: config.norm_eps,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            projector: Dense::load(weights, "projector", dim, config.projection_dim, true)?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut x = mlxcel_core::copy(x);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        self.projector.forward(&x)
    }
}

#[cfg(test)]
#[path = "conditioning_tests.rs"]
pub(crate) mod tests;
