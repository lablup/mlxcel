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

//! Shared Mixture-of-Experts FFN graph primitive (issue #500).
//!
//! A MoE layer replaces the dense SwiGLU MLP with a router that selects the top-k
//! of N experts. [`moe_block`] emits the whole block over the already-normed `[N, H]`
//! hidden (no pre-norm, no residual: the transformer layer owns those), and
//! `emitter/model.rs` calls it from one FFN dispatch site (`emit_ffn_body`) that
//! reaches prefill, ragged / batched decode, and single-token decode (which reshapes
//! `[H]` to `[1, H]`), exactly as the shared attention core (issue #494) does for
//! attention. The primitive is parameterized by [`MoeConfig`] so #501 can add MoE
//! families (Mixtral, Qwen2-MoE, and, once their attention lands, Qwen3-MoE /
//! DeepSeek) by setting the routing knobs and wiring `weight_specs`, without
//! touching the routing math here.
//!
//! # Routing math (softmax-before-top-k)
//!
//! Matching HF Mixtral / Qwen2-MoE / DeepSeek (`scoring_func = "softmax"`):
//!
//! 1. `router_logits = hn @ Wr^T` over the `[N, H]` post-attention-norm hidden.
//! 2. `probs = softmax(router_logits)` over ALL experts.
//! 3. Select the `top_k` largest probs (iterative argmax, each pick masked to
//!    `-inf` so the next argmax skips it; this reproduces `torch.topk`'s
//!    lower-index tie-break exactly).
//! 4. Renormalize the selected probs to sum to one when `norm_topk_prob`, then
//!    scale by `routed_scaling_factor`.
//! 5. Combine the selected experts' SwiGLU outputs by those weights, and add the
//!    shared-expert branch when the family has one.
//!
//! # Masked-dense dispatch
//!
//! Step 4 yields a dense `[N, E]` combine matrix that is zero for unselected
//! experts, and step 5 computes EVERY expert (batched over the expert axis of the
//! mlx-lm stacked `switch_mlp` weights) and weights each by its combine entry.
//! This is numerically identical to gathering only the `top_k` selected experts
//! (`0 * y == 0` and `x + 0 == x` exactly in IEEE-754), so the primitive is
//! token-exact; a sparse gather/scatter that computes only the selected experts is
//! a throughput follow-up (it does not change the result).

use super::builder::{Builder, Val};
use super::config::{Config, MoeConfig};
use super::model::{Consts, attn_softmax};

/// One MoE layer's shared-expert weight handles (Qwen2-MoE / DeepSeek), taken in
/// `emitter/model.rs` (which owns the arg schema) and consumed here.
pub(crate) struct MoeSharedW {
    /// Shared-expert gate projection `[Is, H]`.
    pub gate: Val,
    /// Shared-expert up projection `[Is, H]`.
    pub up: Val,
    /// Shared-expert down projection `[H, Is]`.
    pub down: Val,
    /// The sigmoid gate weight `[1, H]` (Qwen2-MoE `shared_expert_gate`); `None`
    /// for an ungated shared expert (DeepSeek).
    pub expert_gate: Option<Val>,
}

/// One MoE layer's routed + shared weight handles. The routed projections are the
/// mlx-lm stacked `switch_mlp` tensors (one `[E, out, in]` per projection), so the
/// whole expert bank is a single arg per projection and the compute is batched
/// over the leading expert axis.
pub(crate) struct MoeLayerW {
    /// Router (gate) projection `[E, H]`.
    pub router: Val,
    /// Stacked expert gate projection `[E, I, H]`.
    pub w_gate: Val,
    /// Stacked expert up projection `[E, I, H]`.
    pub w_up: Val,
    /// Stacked expert down projection `[E, H, I]`.
    pub w_down: Val,
    /// The shared-expert branch, when the family has one.
    pub shared: Option<MoeSharedW>,
}

/// SiLU(x) = x * sigmoid(x), elementwise over any-shape `x` (sigmoid via
/// `1 / (1 + e^-x)`), the SwiGLU activation the experts use.
fn silu(b: &mut Builder, k: &Consts, x: &Val) -> Val {
    let one_b = b.broadcast(&k.one, &[], x.ty.shape.clone());
    let neg = b.negate(x);
    let ex = b.exponential(&neg);
    let denom = b.add(&one_b, &ex);
    let sig = b.divide(&one_b, &denom);
    b.multiply(x, &sig)
}

/// sigmoid(x) = 1 / (1 + e^-x), elementwise, for the Qwen2-MoE shared-expert gate.
fn sigmoid(b: &mut Builder, k: &Consts, x: &Val) -> Val {
    let one_b = b.broadcast(&k.one, &[], x.ty.shape.clone());
    let neg = b.negate(x);
    let ex = b.exponential(&neg);
    let denom = b.add(&one_b, &ex);
    b.divide(&one_b, &denom)
}

/// The `[N, E]` top-k combine-weight matrix: for each row, the (renormalized,
/// scaled) routing probability of each selected expert, 0 for unselected experts.
/// See the module docs for the routing steps; the returned matrix drives the
/// masked-dense expert combine.
fn topk_combine(b: &mut Builder, moe: &MoeConfig, k: &Consts, logits: &Val, n: usize) -> Val {
    let e = moe.n_experts;
    let probs = attn_softmax(b, k, logits, 1); // [N, E], softmax over all experts

    // entry[n,e] = e, for the one-hot compare against the argmax index each pick.
    let iota_e = b.iota(e); // [E] i32
    let iota_ne = b.broadcast(&iota_e, &[1], vec![n, e]); // [N, E]
    let zeros = b.broadcast(&k.zero, &[], vec![n, e]);
    let neg_inf = b.broadcast(&k.neg_inf, &[], vec![n, e]);

    // iterative top-k: each pick is the per-row argmax of the remaining pool; mask
    // it to -inf so the next argmax skips it (torch.topk lower-index tie-break).
    let mut pool = probs;
    let mut onehots: Vec<Val> = Vec::with_capacity(moe.top_k);
    let mut vals: Vec<Val> = Vec::with_capacity(moe.top_k);
    for _ in 0..moe.top_k {
        let idx = b.argmax_batched(&pool); // [N] i32
        let idx_ne = b.broadcast(&idx, &[0], vec![n, e]); // [N, E]
        let oneh = b.compare("EQ", &iota_ne, &idx_ne, "SIGNED"); // [N, E] i1
        let picked = b.select(&oneh, &pool, &zeros); // [N, E] = pool at the pick, else 0
        let val = b.reduce_add(&picked, 1, &k.zero); // [N], the picked prob per row
        pool = b.select(&oneh, &neg_inf, &pool); // remove the pick from the pool
        onehots.push(oneh);
        vals.push(val);
    }

    // renormalize the selected probs to sum to one (norm_topk_prob).
    let weights: Vec<Val> = if moe.norm_topk_prob {
        let mut denom = vals[0].clone();
        for v in &vals[1..] {
            denom = b.add(&denom, v); // [N]
        }
        vals.iter().map(|v| b.divide(v, &denom)).collect()
    } else {
        vals.clone()
    };
    // optional routed scaling (1.0 for Mixtral / Qwen2-MoE, so usually no op).
    let weights: Vec<Val> = if moe.routed_scaling_factor != 1.0 {
        let s = b.const_f32(moe.routed_scaling_factor as f32);
        weights
            .iter()
            .map(|w| {
                let sb = b.broadcast(&s, &[], vec![n]);
                b.multiply(w, &sb)
            })
            .collect()
    } else {
        weights
    };

    // scatter the per-pick weights into a dense [N, E] combine matrix.
    let mut combine = b.broadcast(&k.zero, &[], vec![n, e]); // [N, E]
    for (oneh, w) in onehots.iter().zip(weights.iter()) {
        let wb = b.broadcast(w, &[0], vec![n, e]); // [N] -> [N, E]
        let term = b.select(oneh, &wb, &zeros);
        combine = b.add(&combine, &term);
    }
    combine
}

/// The routed experts' masked-dense combine: compute every expert's SwiGLU over
/// the shared normed hidden `hn` (batched over the expert axis of the stacked
/// weights), then combine by the `[N, E]` top-k weights, giving `[N, H]`.
#[allow(clippy::too_many_arguments)]
fn routed_experts(
    b: &mut Builder,
    moe: &MoeConfig,
    mw: &MoeLayerW,
    k: &Consts,
    hn: &Val,
    combine: &Val,
    n: usize,
    h: usize,
) -> Val {
    let e = moe.n_experts;
    let i = moe.intermediate;
    // broadcast hn[N,H] to [E,N,H] so every expert sees the same tokens.
    let hn_e = b.broadcast(hn, &[1, 2], vec![e, n, h]); // [E, N, H]
    // gate/up: [E,N,I] = sum_h hn_e[e,n,h] * w[e,i,h]  (batch e, contract h)
    let gate = b.dot_general(&hn_e, &mw.w_gate, &[0], &[0], &[2], &[2], vec![e, n, i]);
    let up = b.dot_general(&hn_e, &mw.w_up, &[0], &[0], &[2], &[2], vec![e, n, i]);
    let act = silu(b, k, &gate);
    let act = b.multiply(&act, &up); // [E, N, I]
    // down: [E,N,H] = sum_i act[e,n,i] * w_down[e,h,i]  (batch e, contract i)
    let down = b.dot_general(&act, &mw.w_down, &[0], &[0], &[2], &[2], vec![e, n, h]);
    // combine: out[N,H] = sum_e combine[n,e] * down[e,n,h]  (batch n, contract e)
    b.dot_general(combine, &down, &[0], &[1], &[1], &[0], vec![n, h])
}

/// The shared-expert branch output `[N, H]` (a SwiGLU of `shared.intermediate`),
/// gated by `sigmoid(hn @ Wg^T)` when the family gates it (Qwen2-MoE).
fn shared_expert(
    b: &mut Builder,
    k: &Consts,
    sh: &MoeSharedW,
    hn: &Val,
    n: usize,
    h: usize,
) -> Val {
    let sg = b.linear_seq(hn, &sh.gate); // [N, Is]
    let su = b.linear_seq(hn, &sh.up); // [N, Is]
    let act = silu(b, k, &sg);
    let act = b.multiply(&act, &su); // [N, Is]
    let down = b.linear_seq(&act, &sh.down); // [N, H]
    match &sh.expert_gate {
        Some(wg) => {
            let g = b.linear_seq(hn, wg); // [N, 1]
            let g = b.reshape(&g, vec![n]); // [N]
            let gs = sigmoid(b, k, &g); // [N]
            let gsb = b.broadcast(&gs, &[0], vec![n, h]); // [N, H]
            b.multiply(&down, &gsb)
        }
        None => down,
    }
}

/// The MoE block output `[N, H]` over the ALREADY-normed hidden `hn`: router
/// softmax + top-k combine of the routed experts, plus the shared-expert branch.
/// No pre-norm and no residual (the decoder layer owns those), matching an HF MoE
/// block's forward, so it is the unit the standalone execution probe compares.
pub(crate) fn moe_block(
    b: &mut Builder,
    c: &Config,
    moe: &MoeConfig,
    mw: &MoeLayerW,
    k: &Consts,
    hn: &Val,
    n: usize,
) -> Val {
    let h = c.hidden;
    let logits = b.linear_seq(hn, &mw.router); // [N, E]
    let combine = topk_combine(b, moe, k, &logits, n); // [N, E]
    let routed = routed_experts(b, moe, mw, k, hn, &combine, n, h); // [N, H]
    match &mw.shared {
        Some(sh) => {
            let s = shared_expert(b, k, sh, hn, n, h);
            b.add(&routed, &s)
        }
        None => routed,
    }
}
