// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! PLaMo 2 hooks in the shared model graphs (epic #564, issue #3347).
//!
//! A child of `model`, like `model_recurrent`, so it reuses the private layer
//! plumbing. Every hook is a no-op (no emitted op) for a config without
//! [`Config::plamo2`], so the other families' graphs are unchanged.
//!
//! - Norm offsets: PLaMo 2 computes `(weight + offset) * rmsnorm(x)` with a
//!   different offset per site. The offset is added to each norm weight right
//!   where the graph takes it, so the shared four-norm (`GemmaFf`) placement
//!   then runs with raw weights.
//! - Q/K: a weightless per-head RMSNorm (eps fixed at 1e-6), then a learned
//!   `[heads, head_dim]` scale, before RoPE.
//! - The Mamba block: `x + post(mixer(pre(x)))`, then the shared MLP residual.

use super::super::plamo2_config::{
    FINAL_NORM_OFFSET, PRE_MIXER_NORM_OFFSET, PRE_MLP_NORM_OFFSET, QK_NORM_EPS,
    post_mixer_norm_offset, post_mlp_norm_offset,
};
use super::*;

/// `w + offset` over a `[hidden]` norm weight.
fn add_offset(b: &mut Builder, w: &Val, offset: f32) -> Val {
    let offset = ssm::splat(b, offset, &w.ty.shape);
    b.add(w, &offset)
}

/// Fold the PLaMo 2 offsets into a freshly taken layer's four norm weights.
pub(super) fn offset_layer_norms(b: &mut Builder, c: &Config, mut lw: LayerW) -> LayerW {
    if c.plamo2.is_none() {
        return lw;
    }
    for (slot, offset) in [
        (&mut lw.in_ln, PRE_MIXER_NORM_OFFSET),
        (&mut lw.post_ln, post_mixer_norm_offset()),
        (&mut lw.pre_ff_ln, PRE_MLP_NORM_OFFSET),
        (&mut lw.post_ff_ln, post_mlp_norm_offset()),
    ] {
        let w = slot.as_ref().expect("every PLaMo 2 block has four norms");
        *slot = Some(add_offset(b, w, offset));
    }
    lw
}

/// The final `model.norm` weight with its PLaMo 2 offset.
pub(super) fn offset_final_norm(b: &mut Builder, c: &Config, w: Val) -> Val {
    if c.plamo2.is_none() {
        return w;
    }
    add_offset(b, &w, FINAL_NORM_OFFSET)
}

/// Weightless RMSNorm over the last axis of `x [.., heads, d]`, then the
/// learned per-head scale `w [heads, d]`.
fn head_scaled(b: &mut Builder, x: &Val, w: &Val) -> Val {
    let shape = x.ty.shape.clone();
    let last = shape.len() - 1;
    let d = shape[last];
    let lead: Vec<usize> = (0..last).collect();
    let square = b.multiply(x, x);
    let zero = b.const_f32(0.0);
    let sum = b.reduce_add(&square, last, &zero);
    let width = ssm::splat(b, d as f32, &sum.ty.shape);
    let mean = b.divide(&sum, &width);
    let eps = ssm::splat(b, QK_NORM_EPS, &mean.ty.shape);
    let stabilized = b.add(&mean, &eps);
    let inverse = b.rsqrt(&stabilized);
    let inverse = b.broadcast(&inverse, &lead, shape.clone());
    let normed = b.multiply(x, &inverse);
    let scale = b.broadcast(w, &[last - 1, last], shape);
    b.multiply(&normed, &scale)
}

/// The PLaMo 2 q/k normalization: `q [.., n_q, d]`, `kk [.., n_kv, d]`.
pub(super) fn head_scaled_qk_norm(b: &mut Builder, lw: &LayerW, q: Val, kk: Val) -> (Val, Val) {
    let qw = lw
        .q_norm
        .as_ref()
        .expect("a PLaMo 2 attention layer has q_weight");
    let kw = lw
        .k_norm
        .as_ref()
        .expect("a PLaMo 2 attention layer has k_weight");
    (head_scaled(b, &q, qw), head_scaled(b, &kk, kw))
}

/// One PLaMo 2 Mamba block at the rank of `layout`: the mixer residual with
/// pre/post norms, then the shared pre/post-normed MLP residual.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_mamba_block(
    b: &mut Builder,
    c: &Config,
    k: &Consts,
    lw: &LayerW,
    li: usize,
    x: &Val,
    layout: &AttnLayout,
    state: &mut RecurrentLayers,
) -> Val {
    let hn = arch_norm(b, c, k, layout, x, &lw.in_ln, None);
    let mixed = recurrent::emit_recurrent_mixer(b, c, lw, li, &hn, layout, state);
    let mixed = post_attn_norm(b, c, k, layout, mixed, lw);
    let x1 = b.add(x, &mixed);
    let hn2 = mlp_pre_norm(b, c, k, layout, &x1, lw);
    let down = emit_ffn_body(b, c, k, layout, &hn2, lw, li);
    b.add(&x1, &down)
}
