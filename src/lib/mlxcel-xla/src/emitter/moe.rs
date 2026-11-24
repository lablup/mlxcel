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
use super::config::{Config, ExpertAct, MoeConfig, MoeScoring};
use super::model::{Consts, attn_softmax};

/// One MoE layer's shared-expert weight handles (Qwen2-MoE / DeepSeek), taken in
/// `emitter/model.rs` (which owns the arg schema) and consumed here.
pub(crate) struct MoeSharedW {
    /// Shared-expert gate projection `[Is, H]`.
    pub gate: Option<Val>,
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
    /// Selection-only score correction `[E]` for biased sigmoid routing.
    pub correction_bias: Option<Val>,
    /// Stacked expert gate projection `[E, I, H]`.
    pub w_gate: Option<Val>,
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

/// Select `count` distinct row-wise maxima from `[N, E]`, returning one `[N, E]`
/// one-hot per pick. [`Builder::argmax_batched`] resolves exact ties to the lower
/// index, matching MLX's stable expert and group ordering.
fn topk_onehots(
    b: &mut Builder,
    k: &Consts,
    selection: &Val,
    n: usize,
    e: usize,
    count: usize,
) -> Vec<Val> {
    let iota_e = b.iota(e);
    let iota_ne = b.broadcast(&iota_e, &[1], vec![n, e]);
    let neg_inf = b.broadcast(&k.neg_inf, &[], vec![n, e]);
    let mut pool = selection.clone();
    let mut onehots = Vec::with_capacity(count);
    for _ in 0..count {
        let idx = b.argmax_batched(&pool);
        let idx_ne = b.broadcast(&idx, &[0], vec![n, e]);
        let oneh = b.compare("EQ", &iota_ne, &idx_ne, "SIGNED");
        pool = b.select(&oneh, &neg_inf, &pool);
        onehots.push(oneh);
    }
    onehots
}

/// Gather one value per row from `scores` for each one-hot selection mask.
fn selected_values(
    b: &mut Builder,
    k: &Consts,
    scores: &Val,
    onehots: &[Val],
    n: usize,
    e: usize,
) -> Vec<Val> {
    let zeros = b.broadcast(&k.zero, &[], vec![n, e]);
    onehots
        .iter()
        .map(|oneh| {
            let picked = b.select(oneh, scores, &zeros);
            b.reduce_add(&picked, 1, &k.zero)
        })
        .collect()
}

/// Keep only experts in the `topk_group` groups. Each contiguous expert group is
/// scored by the sum of its top two corrected scores, exactly as Nemotron-H.
fn grouped_selection(
    b: &mut Builder,
    k: &Consts,
    corrected: &Val,
    n: usize,
    e: usize,
    n_group: usize,
    topk_group: usize,
) -> Val {
    if topk_group == n_group {
        return corrected.clone();
    }
    let per_group = e / n_group;
    let mut group_scores: Option<Val> = None;
    for group in 0..n_group {
        let start = group * per_group;
        let scores = b.slice(corrected, &[(0, n), (start, start + per_group)]);
        let picks = topk_onehots(b, k, &scores, n, per_group, 2);
        let vals = selected_values(b, k, &scores, &picks, n, per_group);
        let sum = b.add(&vals[0], &vals[1]);
        let sum = b.reshape(&sum, vec![n, 1]);
        group_scores = Some(match group_scores {
            Some(ref accumulated) => b.concatenate(accumulated, &sum, 1),
            None => sum,
        });
    }
    let group_scores = group_scores.expect("n_group is validated as positive");
    let chosen_groups = topk_onehots(b, k, &group_scores, n, n_group, topk_group);
    let neg_inf = b.broadcast(&k.neg_inf, &[], vec![n, e]);
    let mut masked = neg_inf;
    for chosen in chosen_groups {
        let mut expert_mask: Option<Val> = None;
        for group in 0..n_group {
            let selected = b.slice(&chosen, &[(0, n), (group, group + 1)]);
            let selected = b.reshape(&selected, vec![n]);
            let selected = b.broadcast(&selected, &[0], vec![n, per_group]);
            expert_mask = Some(match expert_mask {
                Some(ref accumulated) => b.concatenate(accumulated, &selected, 1),
                None => selected,
            });
        }
        masked = b.select(
            &expert_mask.expect("n_group is validated as positive"),
            corrected,
            &masked,
        );
    }
    masked
}

/// Nemotron-H's grouped sigmoid router. Corrected scores participate only in
/// selection; the returned combine matrix contains uncorrected sigmoid scores.
fn sigmoid_topk_combine(
    b: &mut Builder,
    moe: &MoeConfig,
    mw: &MoeLayerW,
    k: &Consts,
    logits: &Val,
    n: usize,
) -> Val {
    let MoeScoring::Sigmoid {
        correction_bias,
        n_group,
        topk_group,
        routed_scaling_factor,
        norm_topk_prob,
    } = moe.scoring
    else {
        unreachable!("sigmoid_topk_combine called for a softmax router")
    };
    let e = moe.n_experts;
    let scores = sigmoid(b, k, logits);
    let corrected = if correction_bias {
        let bias = mw
            .correction_bias
            .as_ref()
            .expect("correction-bias routing requires the correction-bias weight");
        let bias = b.broadcast(bias, &[1], vec![n, e]);
        b.add(&scores, &bias)
    } else {
        scores.clone()
    };
    let selection = grouped_selection(b, k, &corrected, n, e, n_group, topk_group);
    let onehots = topk_onehots(b, k, &selection, n, e, moe.top_k);
    let vals = selected_values(b, k, &scores, &onehots, n, e);
    let weights = if norm_topk_prob {
        let mut denom = vals[0].clone();
        for value in &vals[1..] {
            denom = b.add(&denom, value);
        }
        vals.iter()
            .map(|value| b.divide(value, &denom))
            .collect::<Vec<_>>()
    } else {
        vals
    };
    let weights = if routed_scaling_factor != 1.0 {
        let scale = b.const_f32(routed_scaling_factor);
        weights
            .iter()
            .map(|weight| {
                let scale = b.broadcast(&scale, &[], vec![n]);
                b.multiply(weight, &scale)
            })
            .collect::<Vec<_>>()
    } else {
        weights
    };
    let zeros = b.broadcast(&k.zero, &[], vec![n, e]);
    let mut combine = zeros.clone();
    for (oneh, weight) in onehots.iter().zip(&weights) {
        let weight = b.broadcast(weight, &[0], vec![n, e]);
        let term = b.select(oneh, &weight, &zeros);
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
    let act = match moe.expert_act {
        ExpertAct::SwiGlu => {
            let gate = b.dot_general(
                &hn_e,
                mw.w_gate
                    .as_ref()
                    .expect("SwiGLU experts require the gate projection"),
                &[0],
                &[0],
                &[2],
                &[2],
                vec![e, n, i],
            );
            let up = b.dot_general(&hn_e, &mw.w_up, &[0], &[0], &[2], &[2], vec![e, n, i]);
            let act = silu(b, k, &gate);
            b.multiply(&act, &up)
        }
        ExpertAct::Relu2 => {
            let up = b.dot_general(&hn_e, &mw.w_up, &[0], &[0], &[2], &[2], vec![e, n, i]);
            let zeros = b.broadcast(&k.zero, &[], vec![e, n, i]);
            let relu = b.maximum(&up, &zeros);
            b.multiply(&relu, &relu)
        }
    };
    // down: [E,N,H] = sum_i act[e,n,i] * w_down[e,h,i]  (batch e, contract i)
    let down = b.dot_general(&act, &mw.w_down, &[0], &[0], &[2], &[2], vec![e, n, h]);
    // combine: out[N,H] = sum_e combine[n,e] * down[e,n,h]  (batch n, contract e)
    b.dot_general(combine, &down, &[0], &[1], &[1], &[0], vec![n, h])
}

/// The shared-expert branch output `[N, H]` (a SwiGLU of `shared.intermediate`),
/// gated by `sigmoid(hn @ Wg^T)` when the family gates it (Qwen2-MoE).
fn shared_expert(
    b: &mut Builder,
    moe: &MoeConfig,
    k: &Consts,
    sh: &MoeSharedW,
    hn: &Val,
    n: usize,
    h: usize,
) -> Val {
    let act = match moe.expert_act {
        ExpertAct::SwiGlu => {
            let sg = b.linear_seq(
                hn,
                sh.gate
                    .as_ref()
                    .expect("SwiGLU shared experts require the gate projection"),
            );
            let su = b.linear_seq(hn, &sh.up); // [N, Is]
            let act = silu(b, k, &sg);
            b.multiply(&act, &su)
        }
        ExpertAct::Relu2 => {
            let su = b.linear_seq(hn, &sh.up); // [N, Is]
            let zeros = b.broadcast(&k.zero, &[], su.ty.shape.clone());
            let relu = b.maximum(&su, &zeros);
            b.multiply(&relu, &relu)
        }
    };
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
    let combine = match moe.scoring {
        MoeScoring::Softmax => topk_combine(b, moe, k, &logits, n),
        MoeScoring::Sigmoid { .. } => sigmoid_topk_combine(b, moe, mw, k, &logits, n),
    };
    let routed = routed_experts(b, moe, mw, k, hn, &combine, n, h); // [N, H]
    match &mw.shared {
        Some(sh) => {
            let s = shared_expert(b, moe, k, sh, hn, n, h);
            b.add(&routed, &s)
        }
        None => routed,
    }
}
