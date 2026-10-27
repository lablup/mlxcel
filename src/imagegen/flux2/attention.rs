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

//! FLUX.2 joint and parallel self-attention.
//!
//! Ports of `attention.py`, `parallel_self_attention.py` and
//! `attention_utils.py` in
//! <https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/flux2/transformer>.
//! Heads are `[B, H, S, D]`. q/k RMS norms and RoPE run in f32 and cast back;
//! the RoPE tables are `[S, D/2]` and rotate adjacent pairs.
//!
//! Used by: Flux2, Bonsai.

use anyhow::{Context, Result};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::embeddings::{Flux2Linear, LinearSource, split_last, swiglu};
use super::kv_cache::{KvCacheMode, KvLayerCache};

const QK_NORM_EPS: f32 = 1e-5;

/// `nn.RMSNorm(dim_head, eps=1e-5)` applied in f32.
struct QkNorm {
    weight: UniquePtr<MlxArray>,
}

impl QkNorm {
    fn load(weights: &WeightMap, key: &str) -> Result<Self> {
        let weight = weights
            .get(key)
            .map(|w| mlxcel_core::copy(w))
            .with_context(|| format!("FLUX.2 transformer weight not found: {key}"))?;
        Ok(Self { weight })
    }

    /// `norm(x.astype(f32)).astype(out_dtype)`.
    fn forward(&self, x: &MlxArray, out_dtype: i32) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::fast_rms_norm(
            &mlxcel_core::astype(x, dtype::FLOAT32),
            &self.weight,
            QK_NORM_EPS,
        );
        mlxcel_core::astype(&y, out_dtype)
    }
}

/// `[B, S, H*D]` -> `[B, H, S, D]`.
fn to_heads(x: &MlxArray, heads: i32, head_dim: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let x = mlxcel_core::reshape(x, &[shape[0], shape[1], heads, head_dim]);
    mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3])
}

/// `[B, H, S, D]` -> `[B, S, H*D]`.
fn from_heads(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let x = mlxcel_core::transpose_axes(x, &[0, 2, 1, 3]);
    mlxcel_core::reshape(&x, &[shape[0], shape[2], shape[1] * shape[3]])
}

/// Slice `[B, H, S, D]` along the sequence axis.
fn seq(x: &MlxArray, start: i32, stop: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    mlxcel_core::slice(x, &[0, 0, start, 0], &[s[0], s[1], stop, s[3]])
}

fn seq_len(x: &MlxArray) -> i32 {
    mlxcel_core::array_shape(x)[2]
}

fn cat_seq(parts: &[&MlxArray]) -> UniquePtr<MlxArray> {
    mlxcel_core::concatenate_many(parts, 2)
}

/// `AttentionUtils.apply_rope_bshd`: rotate adjacent pairs of `x` in f32 by
/// `cos`/`sin` (`[S, D/2]`) and cast back to `x`'s dtype.
pub fn apply_rope(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    let out_dtype = mlxcel_core::array_dtype(x);
    let shape = mlxcel_core::array_shape(x);
    let (b, h, s, d) = (shape[0], shape[1], shape[2], shape[3]);
    let xf = mlxcel_core::astype(x, dtype::FLOAT32);
    let cos_b = mlxcel_core::reshape(cos, &[1, 1, s, d / 2]);
    let sin_b = mlxcel_core::reshape(sin, &[1, 1, s, d / 2]);
    let pairs = mlxcel_core::reshape(&xf, &[b, h, s, d / 2, 2]);
    let pick = |i: i32| {
        let p = mlxcel_core::slice(&pairs, &[0, 0, 0, 0, i], &[b, h, s, d / 2, i + 1]);
        mlxcel_core::reshape(&p, &[b, h, s, d / 2])
    };
    let (real, imag) = (pick(0), pick(1));
    let out0 = mlxcel_core::add(
        &mlxcel_core::multiply(&real, &cos_b),
        &mlxcel_core::multiply(&mlxcel_core::negative(&imag), &sin_b),
    );
    let out1 = mlxcel_core::add(
        &mlxcel_core::multiply(&imag, &cos_b),
        &mlxcel_core::multiply(&real, &sin_b),
    );
    let stacked = mlxcel_core::stack_owned(&[out0, out1], -1);
    mlxcel_core::astype(&mlxcel_core::reshape(&stacked, &shape), out_dtype)
}

/// `scaled_dot_product_attention(q, k, v, scale=1/sqrt(D))`, no mask.
fn attend(q: &MlxArray, k: &MlxArray, v: &MlxArray) -> UniquePtr<MlxArray> {
    let d = *mlxcel_core::array_shape(q).last().unwrap_or(&1);
    let scale = 1.0_f32 / (d as f32).sqrt();
    // SAFETY: a null mask is the documented "no mask" argument.
    unsafe { mlxcel_core::fast_scaled_dot_product_attention(q, k, v, scale, std::ptr::null()) }
}

/// Attention over `[txt, (ref), img]` heads with the reference K/V cache mode
/// applied (`AttentionUtils.compute_attention` / `compute_kv_cache_attention`).
/// Returns `[B, S_q, H*D]`.
fn attention_with_cache(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    num_txt: i32,
    mode: Option<KvCacheMode>,
    cache: Option<&mut KvLayerCache>,
) -> Result<UniquePtr<MlxArray>> {
    let out = match (mode, cache) {
        (Some(KvCacheMode::Extract { num_ref_tokens }), cache) if num_ref_tokens > 0 => {
            let (ref_start, ref_end) = (num_txt, num_txt + num_ref_tokens);
            let k_ref = seq(k, ref_start, ref_end);
            let v_ref = seq(v, ref_start, ref_end);
            let q_txt_img = cat_seq(&[&seq(q, 0, ref_start), &seq(q, ref_end, seq_len(q))]);
            let txt_img = attend(&q_txt_img, k, v);
            let hidden_ref = attend(&seq(q, ref_start, ref_end), &k_ref, &v_ref);
            let total = seq_len(&txt_img);
            let out = cat_seq(&[
                &seq(&txt_img, 0, ref_start),
                &hidden_ref,
                &seq(&txt_img, ref_start, total),
            ]);
            if let Some(cache) = cache {
                cache.store(k_ref, v_ref);
            }
            out
        }
        (Some(KvCacheMode::Cached), Some(cache)) => {
            let (k_ref, v_ref) = cache.get()?;
            let n = seq_len(k);
            let k = cat_seq(&[&seq(k, 0, num_txt), k_ref, &seq(k, num_txt, n)]);
            let v = cat_seq(&[&seq(v, 0, num_txt), v_ref, &seq(v, num_txt, n)]);
            attend(q, &k, &v)
        }
        _ => attend(q, k, v),
    };
    Ok(from_heads(&out))
}

/// `Flux2Attention` with the added (text) projections: the double-stream
/// blocks' joint attention.
pub struct JointAttention {
    to_q: Box<dyn Flux2Linear>,
    to_k: Box<dyn Flux2Linear>,
    to_v: Box<dyn Flux2Linear>,
    norm_q: QkNorm,
    norm_k: QkNorm,
    to_out: Box<dyn Flux2Linear>,
    add_q_proj: Box<dyn Flux2Linear>,
    add_k_proj: Box<dyn Flux2Linear>,
    add_v_proj: Box<dyn Flux2Linear>,
    norm_added_q: QkNorm,
    norm_added_k: QkNorm,
    to_add_out: Box<dyn Flux2Linear>,
    heads: i32,
    head_dim: i32,
}

impl JointAttention {
    pub fn load(
        weights: &WeightMap,
        source: &dyn LinearSource,
        prefix: &str,
        heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let l = |name: &str| source.load(weights, &format!("{prefix}.{name}"));
        let n = |name: &str| QkNorm::load(weights, &format!("{prefix}.{name}.weight"));
        Ok(Self {
            to_q: l("to_q")?,
            to_k: l("to_k")?,
            to_v: l("to_v")?,
            norm_q: n("norm_q")?,
            norm_k: n("norm_k")?,
            to_out: l("to_out")?,
            add_q_proj: l("add_q_proj")?,
            add_k_proj: l("add_k_proj")?,
            add_v_proj: l("add_v_proj")?,
            norm_added_q: n("norm_added_q")?,
            norm_added_k: n("norm_added_k")?,
            to_add_out: l("to_add_out")?,
            heads,
            head_dim,
        })
    }

    /// `AttentionUtils.process_qkv`: project, split heads, f32 q/k norms cast
    /// back to the projection dtype.
    #[allow(clippy::too_many_arguments)]
    fn qkv(
        &self,
        x: &MlxArray,
        q: &dyn Flux2Linear,
        k: &dyn Flux2Linear,
        v: &dyn Flux2Linear,
        nq: &QkNorm,
        nk: &QkNorm,
    ) -> [UniquePtr<MlxArray>; 3] {
        let query = to_heads(&q.forward(x), self.heads, self.head_dim);
        let key = to_heads(&k.forward(x), self.heads, self.head_dim);
        let value = to_heads(&v.forward(x), self.heads, self.head_dim);
        let query = nq.forward(&query, mlxcel_core::array_dtype(&query));
        let key = nk.forward(&key, mlxcel_core::array_dtype(&key));
        [query, key, value]
    }

    /// Returns `(image_out, text_out)`.
    pub fn forward(
        &self,
        hidden: &MlxArray,
        encoder: &MlxArray,
        rope: (&MlxArray, &MlxArray),
        mode: Option<KvCacheMode>,
        cache: Option<&mut KvLayerCache>,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let [q, k, v] = self.qkv(
            hidden,
            &*self.to_q,
            &*self.to_k,
            &*self.to_v,
            &self.norm_q,
            &self.norm_k,
        );
        let [eq, ek, ev] = self.qkv(
            encoder,
            &*self.add_q_proj,
            &*self.add_k_proj,
            &*self.add_v_proj,
            &self.norm_added_q,
            &self.norm_added_k,
        );
        let q = cat_seq(&[&eq, &q]);
        let k = cat_seq(&[&ek, &k]);
        let v = cat_seq(&[&ev, &v]);
        let q = apply_rope(&q, rope.0, rope.1);
        let k = apply_rope(&k, rope.0, rope.1);
        let num_txt = mlxcel_core::array_shape(encoder)[1];
        let out = attention_with_cache(&q, &k, &v, num_txt, mode, cache)?;
        let s = mlxcel_core::array_shape(&out);
        let text = mlxcel_core::slice(&out, &[0, 0, 0], &[s[0], num_txt, s[2]]);
        let image = mlxcel_core::slice(&out, &[0, num_txt, 0], &s);
        Ok((self.to_out.forward(&image), self.to_add_out.forward(&text)))
    }
}

/// `Flux2ParallelSelfAttention`: one fused projection to q, k, v and the
/// SwiGLU MLP input, attention and MLP side by side, one output projection.
pub struct ParallelSelfAttention {
    to_qkv_mlp_proj: Box<dyn Flux2Linear>,
    norm_q: QkNorm,
    norm_k: QkNorm,
    to_out: Box<dyn Flux2Linear>,
    heads: i32,
    head_dim: i32,
    mlp_hidden_dim: i32,
}

impl ParallelSelfAttention {
    pub fn load(
        weights: &WeightMap,
        source: &dyn LinearSource,
        prefix: &str,
        heads: i32,
        head_dim: i32,
        mlp_hidden_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            to_qkv_mlp_proj: source.load(weights, &format!("{prefix}.to_qkv_mlp_proj"))?,
            norm_q: QkNorm::load(weights, &format!("{prefix}.norm_q.weight"))?,
            norm_k: QkNorm::load(weights, &format!("{prefix}.norm_k.weight"))?,
            to_out: source.load(weights, &format!("{prefix}.to_out"))?,
            heads,
            head_dim,
            mlp_hidden_dim,
        })
    }

    /// `hidden` is `[B, txt + (ref) + img, D]`. q and k are cast to
    /// `compute_dtype` after their norms (upstream's `ModelConfig.precision`).
    pub fn forward(
        &self,
        hidden: &MlxArray,
        rope: (&MlxArray, &MlxArray),
        num_txt: i32,
        compute_dtype: i32,
        mode: Option<KvCacheMode>,
        cache: Option<&mut KvLayerCache>,
    ) -> Result<UniquePtr<MlxArray>> {
        let inner = self.heads * self.head_dim;
        let proj = self.to_qkv_mlp_proj.forward(hidden);
        let parts = split_last(&proj, &[inner, inner, inner, self.mlp_hidden_dim * 2]);
        let q = to_heads(&parts[0], self.heads, self.head_dim);
        let k = to_heads(&parts[1], self.heads, self.head_dim);
        let v = to_heads(&parts[2], self.heads, self.head_dim);
        let q = self.norm_q.forward(&q, compute_dtype);
        let k = self.norm_k.forward(&k, compute_dtype);
        let q = apply_rope(&q, rope.0, rope.1);
        let k = apply_rope(&k, rope.0, rope.1);
        let attn = attention_with_cache(&q, &k, &v, num_txt, mode, cache)?;
        let mlp = swiglu(&parts[3]);
        Ok(self
            .to_out
            .forward(&mlxcel_core::concatenate(&attn, &mlp, -1)))
    }
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod tests;
