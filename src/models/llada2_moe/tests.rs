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

//! Unit and synthetic-parity tests for the LLaDA-2 MoE module:
//!
//! * pure host-side loop-schedule helpers (threshold decay, transfer-mask
//!   selection, progress guarantee, block count, EOS truncation),
//! * config-default resolution for the omitted-key cases,
//! * gate math on a tiny synthetic gate (group top-2 selection, bias affecting
//!   selection but not weights, `+ 1e-20` normalization, scaling factor),
//! * MoE-block parity against a naive per-expert host loop,
//! * a fixed-logits unmasking run that reproduces the hand-computed reveal
//!   order,
//! * an on-disk synthetic checkpoint loaded through the real detection +
//!   loading path and generated end to end.

use super::generate::{
    Llada2FinishReason, Llada2GenerateOptions, block_num_blocks, block_threshold, transfer_mask,
    truncate_at_eos,
};
use super::{
    GROUP_MASK_FILL, Llada2MoeModel, MoEBlock, MoEGate, ModelArgs, group_limited_mask,
    parse_eos_ids,
};
use mlxcel_core::weights::WeightMap;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

// -------------------------------------------------------------------------
// Host helpers shared by the device tests
// -------------------------------------------------------------------------

fn f32_array(data: &[f32], shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_f32(data, shape)
}

fn to_host_f32(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    let source = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&source);
    mlxcel_core::array_to_raw_bytes(&source)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

fn to_host_i32(array: &mlxcel_core::MlxArray) -> Vec<i32> {
    let source = mlxcel_core::astype(array, mlxcel_core::dtype::INT32);
    mlxcel_core::eval(&source);
    mlxcel_core::array_to_raw_bytes(&source)
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

fn sigmoid_host(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_host(x: f32) -> f32 {
    x * sigmoid_host(x)
}

/// `y[r] = sum_c w[r*cols + c] * x[c]` for a row-major `(rows, cols)` weight.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
        .collect()
}

/// Deterministic small weight fill in `[-0.5, 0.5)`, seeded per tensor.
fn fill(seed: u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let h = (i as u64)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            (h % 1000) as f32 / 1000.0 - 0.5
        })
        .collect()
}

/// Reference implementation of the LLaDA-2 gate math over host logits, matching
/// the issue spec: sigmoid scoring, additive bias for SELECTION only,
/// group-limited top-k (sum of top-2 per group), weights gathered from the
/// UNBIASED scores, `+ 1e-20` normalization, then routed scaling. Returns the
/// `(expert_index, weight)` pairs.
#[allow(clippy::too_many_arguments)]
fn gate_reference(
    logits: &[f32],
    bias: &[f32],
    n_group: usize,
    topk_group: usize,
    top_k: usize,
    scaling: f32,
    norm: bool,
) -> Vec<(usize, f32)> {
    let n = logits.len();
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid_host(l)).collect();
    let biased: Vec<f32> = scores.iter().zip(bias).map(|(&s, &b)| s + b).collect();
    let epg = n / n_group;

    // group_score = sum of the top-2 biased scores within each group.
    let mut group_score = vec![0f32; n_group];
    for (g, gs) in group_score.iter_mut().enumerate() {
        let mut vals: Vec<f32> = (0..epg).map(|i| biased[g * epg + i]).collect();
        vals.sort_by(|a, b| b.total_cmp(a));
        *gs = vals.iter().take(2).sum();
    }
    let mut group_order: Vec<usize> = (0..n_group).collect();
    group_order.sort_by(|&a, &b| group_score[b].total_cmp(&group_score[a]));
    let kept: std::collections::HashSet<usize> = group_order.into_iter().take(topk_group).collect();

    let masked: Vec<f32> = biased
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            if kept.contains(&(i / epg)) {
                b
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| masked[b].total_cmp(&masked[a]));
    let sel: Vec<usize> = idx.into_iter().take(top_k).collect();

    let mut w: Vec<f32> = sel.iter().map(|&e| scores[e]).collect();
    if norm && top_k > 1 {
        let sum: f32 = w.iter().sum::<f32>() + 1e-20;
        for x in &mut w {
            *x /= sum;
        }
    }
    for x in &mut w {
        *x *= scaling;
    }
    sel.into_iter().zip(w).collect()
}

// -------------------------------------------------------------------------
// Loop-schedule helpers
// -------------------------------------------------------------------------

#[test]
fn block_threshold_endpoints() {
    // t = 1 gives T; t = S gives min_threshold.
    assert!((block_threshold(1, 32, 0.95, 0.5) - 0.95).abs() < 1e-6);
    assert!((block_threshold(32, 32, 0.95, 0.5) - 0.5).abs() < 1e-6);
    // Constant threshold when min == T.
    assert!((block_threshold(16, 32, 0.9, 0.9) - 0.9).abs() < 1e-6);
    // Single-step block always uses T.
    assert!((block_threshold(1, 1, 0.8, 0.2) - 0.8).abs() < 1e-6);
}

#[test]
fn block_threshold_midpoint_is_linear() {
    // S = 5, t = 3 is exactly halfway: (t-1)/(S-1) = 2/4 = 0.5.
    let thr = block_threshold(3, 5, 1.0, 0.0);
    assert!((thr - 0.5).abs() < 1e-6);
}

#[test]
fn transfer_mask_all_above_threshold_reveals_all_active() {
    let active = [true, true, false, true];
    let conf = [0.99, 0.96, 0.10, 0.97];
    // Inactive position 2 stays false even with high threshold clearance.
    assert_eq!(
        transfer_mask(&active, &conf, 0.9),
        vec![true, true, false, true]
    );
}

#[test]
fn transfer_mask_none_above_forces_single_argmax() {
    // No active position clears 0.9, so exactly the single highest-confidence
    // active position is revealed (position 3 at 0.80).
    let active = [true, true, false, true];
    let conf = [0.70, 0.60, 0.99, 0.80];
    assert_eq!(
        transfer_mask(&active, &conf, 0.9),
        vec![false, false, false, true]
    );
}

#[test]
fn transfer_mask_tie_prefers_lower_index() {
    // Two active positions tie on confidence below threshold; the lower index
    // wins (matching argmax's first-maximum rule).
    let active = [true, true];
    let conf = [0.5, 0.5];
    assert_eq!(transfer_mask(&active, &conf, 0.9), vec![true, false]);
}

#[test]
fn transfer_mask_empty_when_no_active() {
    let active = [false, false, false];
    let conf = [0.99, 0.99, 0.99];
    assert_eq!(
        transfer_mask(&active, &conf, 0.9),
        vec![false, false, false]
    );
}

#[test]
fn prompt_tail_positions_are_never_revealed() {
    // First-generation-block prompt-tail guard: block positions holding prompt
    // tokens are not active (`canvas[s+j] != mask`), so they are never revealed
    // and never overwritten, even at max confidence, in either the
    // above-threshold or forced-single branch. Here positions 0,1 are prompt
    // tail, positions 2,3 are masked.
    let active = [false, false, true, true];
    // Both branches: high-confidence prompt tail must stay committed.
    let above = transfer_mask(&active, &[0.99, 0.99, 0.96, 0.95], 0.9);
    assert_eq!(above, vec![false, false, true, true]);
    // Forced-single branch (nothing clears the threshold): only a masked
    // position is forced, never a prompt-tail one.
    let forced = transfer_mask(&active, &[0.99, 0.99, 0.10, 0.20], 0.9);
    assert_eq!(forced, vec![false, false, false, true]);
}

#[test]
fn block_progress_guarantee_unmasks_within_b_iterations() {
    // A block of B masked positions whose confidences never clear the threshold
    // still fully unmasks within B iterations (one forced reveal per step).
    let block = 8usize;
    let conf = vec![0.1f32; block]; // never above threshold
    let mut revealed = vec![false; block];
    let mut iterations = 0;
    for _ in 0..block {
        let active: Vec<bool> = revealed.iter().map(|&r| !r).collect();
        if !active.iter().any(|&a| a) {
            break;
        }
        iterations += 1;
        let transfer = transfer_mask(&active, &conf, 0.9);
        let revealed_this_step = transfer.iter().filter(|&&t| t).count();
        assert_eq!(revealed_this_step, 1, "exactly one forced reveal per step");
        for (r, &t) in revealed.iter_mut().zip(&transfer) {
            *r = *r || t;
        }
    }
    assert!(revealed.iter().all(|&r| r), "block fully unmasked");
    assert!(iterations <= block, "unmasks within B iterations");
}

#[test]
fn fixed_logits_unmasking_reproduces_reveal_order() {
    // Fixed per-position confidences drive a deterministic reveal order under a
    // constant 0.9 threshold: step 1 reveals every position above 0.9, then
    // each subsequent step forces the single highest-confidence masked one.
    let conf = [0.95f32, 0.80, 0.99, 0.70];
    let steps = 4usize;
    let mut revealed = [false; 4];
    let mut reveal_order: Vec<Vec<usize>> = Vec::new();
    for t in 1..=steps {
        let active: Vec<bool> = revealed.iter().map(|&r| !r).collect();
        if !active.iter().any(|&a| a) {
            break;
        }
        let thr = block_threshold(t, steps, 0.9, 0.9);
        let transfer = transfer_mask(&active, &conf, thr);
        let this: Vec<usize> = transfer
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| t.then_some(i))
            .collect();
        for &i in &this {
            revealed[i] = true;
        }
        reveal_order.push(this);
    }
    assert_eq!(reveal_order, vec![vec![0, 2], vec![1], vec![3]]);
    assert!(revealed.iter().all(|&r| r));
}

#[test]
fn block_num_blocks_covers_prompt_and_generation() {
    // ceil((P + gen) / B).
    assert_eq!(block_num_blocks(3, 4, 2), 4); // ceil(7/2)
    assert_eq!(block_num_blocks(64, 32, 32), 3); // ceil(96/32)
    assert_eq!(block_num_blocks(0, 1, 32), 1); // at least one block
}

#[test]
fn truncate_at_eos_cuts_at_first_stop() {
    assert_eq!(truncate_at_eos(&[5, 6, 7, 2, 8], &[2]), vec![5, 6, 7]);
    assert_eq!(truncate_at_eos(&[5, 6, 7], &[2]), vec![5, 6, 7]);
    assert_eq!(truncate_at_eos(&[2, 5], &[2]), Vec::<i32>::new());
}

// -------------------------------------------------------------------------
// Config defaults
// -------------------------------------------------------------------------

fn parse_args(value: serde_json::Value) -> ModelArgs {
    serde_json::from_value(value).expect("config should parse")
}

#[test]
fn config_defaults_apply_for_omitted_keys() {
    // The real config omits eos_token_id, mask_token_id, and use_qk_norm.
    let args = parse_args(serde_json::json!({
        "model_type": "llada2_moe",
        "pad_token_id": 156892,
    }));
    assert!(args.use_qk_norm, "use_qk_norm defaults on");
    assert_eq!(args.mask_token_id(), 156895, "mask_token_id default");
    assert_eq!(args.pad_token_id(), 156892);
    // eos_token_id absent -> falls back to {pad_token_id}.
    assert_eq!(args.eos_token_ids(), vec![156892]);
    // Structural defaults from the spec table.
    assert_eq!(args.num_experts, 256);
    assert_eq!(args.num_experts_per_tok, 8);
    assert_eq!(args.n_group, 8);
    assert_eq!(args.topk_group, 4);
    assert_eq!(args.first_k_dense_replace, 1);
    assert!((args.routed_scaling_factor - 2.5).abs() < 1e-6);
    assert!(args.norm_topk_prob);
}

#[test]
fn config_derives_head_dim_and_rotary_dim() {
    // head_dim defaults to hidden_size / num_attention_heads; rotary_dim to
    // head_dim * partial_rotary_factor. Matches LLaDA2.0-mini (128 / 64).
    let args = parse_args(serde_json::json!({
        "model_type": "llada2_moe",
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "partial_rotary_factor": 0.5,
    }));
    assert_eq!(args.head_dim(), 128);
    assert_eq!(args.rotary_dim(), 64);
}

#[test]
fn config_pad_token_default_when_absent() {
    let args = parse_args(serde_json::json!({ "model_type": "llada2_moe" }));
    assert_eq!(args.pad_token_id(), 156892);
    assert_eq!(args.eos_token_ids(), vec![156892]);
}

#[test]
fn parse_eos_ids_accepts_scalar_array_and_null() {
    assert_eq!(parse_eos_ids(Some(&serde_json::json!(7))), vec![7]);
    assert_eq!(
        parse_eos_ids(Some(&serde_json::json!([1, 2, 3]))),
        vec![1, 2, 3]
    );
    assert_eq!(
        parse_eos_ids(Some(&serde_json::json!(null))),
        Vec::<i32>::new()
    );
    assert_eq!(parse_eos_ids(None), Vec::<i32>::new());
}

#[test]
fn config_explicit_eos_list_overrides_pad_fallback() {
    let args = parse_args(serde_json::json!({
        "model_type": "llada2_moe",
        "pad_token_id": 156892,
        "eos_token_id": [100, 200],
    }));
    assert_eq!(args.eos_token_ids(), vec![100, 200]);
}

// -------------------------------------------------------------------------
// Gate math (tiny synthetic gate)
// -------------------------------------------------------------------------

fn identity_f32(n: i32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut d = vec![0f32; (n * n) as usize];
    for i in 0..n {
        d[(i * n + i) as usize] = 1.0;
    }
    f32_array(&d, &[n, n])
}

/// Logits chosen to give clearly separated group scores (no ties).
const GATE_LOGITS: [f32; 8] = [2.0, -2.0, 1.5, -3.0, 0.8, 0.4, -1.5, -1.8];

#[test]
fn gate_selects_grouped_experts_bias_only_affects_selection() {
    // Identity gate weight -> logits == x. Bias on expert 5 flips the within-
    // group selection (5 over 4) without changing the gathered weights.
    let bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0];
    let gate = MoEGate {
        weight: identity_f32(8),
        expert_bias: f32_array(&bias, &[8]),
        top_k: 2,
        n_group: 4,
        topk_group: 2,
        routed_scaling_factor: 2.5,
        norm_topk_prob: true,
    };
    let x = f32_array(&GATE_LOGITS, &[1, 8]);
    let (indices, weights) = gate.forward(&x);
    let idx = to_host_i32(&indices);
    let wts = to_host_f32(&weights);

    let mut got: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
    for (&i, &w) in idx.iter().zip(&wts) {
        got.insert(i, w);
    }

    let reference = gate_reference(&GATE_LOGITS, &bias, 4, 2, 2, 2.5, true);
    let mut want: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
    for (i, w) in reference {
        want.insert(i as i32, w);
    }

    // Selected set is exactly {0, 5}: group masking drops experts 2/3/6/7
    // (non-kept groups) even though expert 2's score is high, and the bias
    // selects expert 5 over expert 4.
    let mut got_keys: Vec<i32> = got.keys().copied().collect();
    got_keys.sort();
    assert_eq!(got_keys, vec![0, 5], "grouped + biased selection");

    for (expert, want_w) in &want {
        let got_w = got
            .get(expert)
            .unwrap_or_else(|| panic!("missing expert {expert}"));
        assert!(
            (got_w - want_w).abs() < 1e-3,
            "expert {expert}: got {got_w}, want {want_w}"
        );
    }

    // The weight for expert 0 comes from the UNBIASED score, and normalized
    // weights sum to the routed scaling factor.
    let sum: f32 = wts.iter().sum();
    assert!((sum - 2.5).abs() < 1e-3, "weights sum to scaling factor");
}

#[test]
fn gate_group_mask_fills_non_kept_groups_below_all_biased() {
    // The fill constant orders masked experts strictly last even when the
    // real biased scores span a wide range including negatives.
    let biased = f32_array(&[0.9, -0.4, 0.8, 0.7, -0.9, 0.95, 0.1, 0.2], &[1, 8]);
    let masked = group_limited_mask(&biased, 4, 2, GROUP_MASK_FILL);
    let host = to_host_f32(&masked);
    // Groups (epg = 2) sums: g0 = 0.5, g1 = 1.5, g2 = 0.05, g3 = 0.3.
    // Keep g1 (experts 2,3) and g0 (experts 0,1); mask g2 (4,5) and g3 (6,7).
    assert!(host[4] <= GROUP_MASK_FILL / 2.0, "expert 4 masked");
    assert!(host[5] <= GROUP_MASK_FILL / 2.0, "expert 5 masked");
    assert!(host[6] <= GROUP_MASK_FILL / 2.0, "expert 6 masked");
    assert!(host[7] <= GROUP_MASK_FILL / 2.0, "expert 7 masked");
    // Kept experts keep their biased scores.
    for (i, &value) in host.iter().enumerate().take(4) {
        assert!(value > -1.0, "kept expert {i} retained: {value}");
    }
}

// -------------------------------------------------------------------------
// MoE-block parity against a naive per-expert host loop
// -------------------------------------------------------------------------

fn moe_test_args() -> ModelArgs {
    parse_args(serde_json::json!({
        "model_type": "llada2_moe",
        "hidden_size": 4,
        "moe_intermediate_size": 3,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "n_group": 4,
        "topk_group": 2,
        "num_shared_experts": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "rms_norm_eps": 1e-6,
    }))
}

#[test]
fn moe_block_matches_naive_per_expert_loop() {
    let hidden = 4usize;
    let moe = 3usize;
    let experts = 8usize;
    let prefix = "layer.mlp";

    // Gate weight row e = [logit_e, 0, 0, 0] so logits == the first input
    // channel scaled per expert; with x[0] = 1 the logits equal GATE_LOGITS.
    let mut gate_w = vec![0f32; experts * hidden];
    for (e, &l) in GATE_LOGITS.iter().enumerate() {
        gate_w[e * hidden] = l;
    }
    let bias = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0];

    // Per-expert projection weights, kept as raw host data for the reference.
    let mut expert_gate = Vec::new();
    let mut expert_up = Vec::new();
    let mut expert_down = Vec::new();
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.gate.weight"),
        f32_array(&gate_w, &[experts as i32, hidden as i32]),
    );
    weights.insert(
        format!("{prefix}.gate.expert_bias"),
        f32_array(&bias, &[experts as i32]),
    );
    for e in 0..experts {
        let g = fill(e as u64 + 1, moe * hidden);
        let u = fill(e as u64 + 101, moe * hidden);
        let d = fill(e as u64 + 201, hidden * moe);
        weights.insert(
            format!("{prefix}.experts.{e}.gate_proj.weight"),
            f32_array(&g, &[moe as i32, hidden as i32]),
        );
        weights.insert(
            format!("{prefix}.experts.{e}.up_proj.weight"),
            f32_array(&u, &[moe as i32, hidden as i32]),
        );
        weights.insert(
            format!("{prefix}.experts.{e}.down_proj.weight"),
            f32_array(&d, &[hidden as i32, moe as i32]),
        );
        expert_gate.push(g);
        expert_up.push(u);
        expert_down.push(d);
    }
    let shared_gate = fill(900, moe * hidden);
    let shared_up = fill(901, moe * hidden);
    let shared_down = fill(902, hidden * moe);
    weights.insert(
        format!("{prefix}.shared_experts.gate_proj.weight"),
        f32_array(&shared_gate, &[moe as i32, hidden as i32]),
    );
    weights.insert(
        format!("{prefix}.shared_experts.up_proj.weight"),
        f32_array(&shared_up, &[moe as i32, hidden as i32]),
    );
    weights.insert(
        format!("{prefix}.shared_experts.down_proj.weight"),
        f32_array(&shared_down, &[hidden as i32, moe as i32]),
    );

    let args = moe_test_args();
    let block = MoEBlock::from_weights(&weights, &args, prefix).expect("MoE block builds");

    // x[0] = 1 so gate logits == GATE_LOGITS.
    let x = [1.0f32, 0.5, -0.3, 0.2];
    let x_arr = f32_array(&x, &[1, 1, hidden as i32]);
    let device_out = to_host_f32(&block.forward(&x_arr));

    // Naive host reference: gate selection, then per-expert down(silu(gate) * up)
    // weighted-summed, plus the shared expert.
    let reference = gate_reference(&GATE_LOGITS, &bias, 4, 2, 2, 2.5, true);
    let mut expected = vec![0f32; hidden];
    let expert_forward = |g: &[f32], u: &[f32], d: &[f32]| -> Vec<f32> {
        let gate_v = matvec(g, moe, hidden, &x);
        let up_v = matvec(u, moe, hidden, &x);
        let act: Vec<f32> = gate_v
            .iter()
            .zip(&up_v)
            .map(|(&gv, &uv)| silu_host(gv) * uv)
            .collect();
        matvec(d, hidden, moe, &act)
    };
    for (e, w) in reference {
        let o = expert_forward(&expert_gate[e], &expert_up[e], &expert_down[e]);
        for (acc, ov) in expected.iter_mut().zip(&o) {
            *acc += w * ov;
        }
    }
    let shared_o = expert_forward(&shared_gate, &shared_up, &shared_down);
    for (acc, sv) in expected.iter_mut().zip(&shared_o) {
        *acc += sv;
    }

    for (i, (got, want)) in device_out.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "channel {i}: got {got}, want {want}"
        );
    }
}

// -------------------------------------------------------------------------
// On-disk synthetic checkpoint: detection + loading + generation
// -------------------------------------------------------------------------

fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("llada2_{tag}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write a real F32 safetensors file for the given `(name, shape, data)`
/// tensors, so the checkpoint loads through MLX's own safetensors reader.
fn write_safetensors_f32(path: &Path, tensors: &[(String, Vec<i64>, Vec<f32>)]) {
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for &v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header_str = serde_json::to_string(&header).unwrap();
    let header_bytes = header_str.as_bytes();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    f.write_all(header_bytes).unwrap();
    f.write_all(&data).unwrap();
}

/// Build a tiny on-disk `llada2_moe` checkpoint: 2 layers (layer 0 dense,
/// layer 1 MoE) with 8 experts, exercising every real weight key.
fn build_tiny_checkpoint(dir: &Path) {
    let hidden = 8usize;
    let heads = 2usize;
    let head_dim = 4usize;
    let kv_heads = 1usize;
    let inter = 6usize; // dense intermediate
    let moe = 3usize; // moe intermediate
    let experts = 8usize;
    let vocab = 16usize;
    let qkv_out = (heads + 2 * kv_heads) * head_dim; // 16
    let attn_out = heads * head_dim; // 8

    let config = serde_json::json!({
        "model_type": "llada2_moe",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "moe_intermediate_size": moe,
        "num_hidden_layers": 2,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "partial_rotary_factor": 0.5,
        "rope_theta": 600000.0,
        "use_qk_norm": true,
        "rms_norm_eps": 1e-6,
        "num_experts": experts,
        "num_experts_per_tok": 2,
        "n_group": 4,
        "topk_group": 2,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "pad_token_id": 1,
        "mask_token_id": 15,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut tensors: Vec<(String, Vec<i64>, Vec<f32>)> = Vec::new();
    let mut push = |name: String, shape: Vec<i64>, data: Vec<f32>| {
        let n: usize = shape.iter().product::<i64>() as usize;
        assert_eq!(n, data.len(), "shape/data mismatch for {name}");
        tensors.push((name, shape, data));
    };
    let ones = |n: usize| vec![1.0f32; n];

    push(
        "model.word_embeddings.weight".into(),
        vec![vocab as i64, hidden as i64],
        fill(10, vocab * hidden),
    );

    for layer in 0..2usize {
        let p = format!("model.layers.{layer}");
        push(
            format!("{p}.input_layernorm.weight"),
            vec![hidden as i64],
            ones(hidden),
        );
        push(
            format!("{p}.post_attention_layernorm.weight"),
            vec![hidden as i64],
            ones(hidden),
        );
        push(
            format!("{p}.attention.query_key_value.weight"),
            vec![qkv_out as i64, hidden as i64],
            fill(1000 + layer as u64, qkv_out * hidden),
        );
        push(
            format!("{p}.attention.query_layernorm.weight"),
            vec![head_dim as i64],
            ones(head_dim),
        );
        push(
            format!("{p}.attention.key_layernorm.weight"),
            vec![head_dim as i64],
            ones(head_dim),
        );
        push(
            format!("{p}.attention.dense.weight"),
            vec![hidden as i64, attn_out as i64],
            fill(2000 + layer as u64, hidden * attn_out),
        );

        if layer == 0 {
            // Dense SwiGLU MLP.
            push(
                format!("{p}.mlp.gate_proj.weight"),
                vec![inter as i64, hidden as i64],
                fill(3000, inter * hidden),
            );
            push(
                format!("{p}.mlp.up_proj.weight"),
                vec![inter as i64, hidden as i64],
                fill(3100, inter * hidden),
            );
            push(
                format!("{p}.mlp.down_proj.weight"),
                vec![hidden as i64, inter as i64],
                fill(3200, hidden * inter),
            );
        } else {
            // MoE block.
            let mut gate_w = vec![0f32; experts * hidden];
            for (e, &l) in GATE_LOGITS.iter().enumerate() {
                gate_w[e * hidden] = l;
            }
            push(
                format!("{p}.mlp.gate.weight"),
                vec![experts as i64, hidden as i64],
                gate_w,
            );
            push(
                format!("{p}.mlp.gate.expert_bias"),
                vec![experts as i64],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            );
            for e in 0..experts {
                push(
                    format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                    vec![moe as i64, hidden as i64],
                    fill(4000 + e as u64, moe * hidden),
                );
                push(
                    format!("{p}.mlp.experts.{e}.up_proj.weight"),
                    vec![moe as i64, hidden as i64],
                    fill(4100 + e as u64, moe * hidden),
                );
                push(
                    format!("{p}.mlp.experts.{e}.down_proj.weight"),
                    vec![hidden as i64, moe as i64],
                    fill(4200 + e as u64, hidden * moe),
                );
            }
            push(
                format!("{p}.mlp.shared_experts.gate_proj.weight"),
                vec![moe as i64, hidden as i64],
                fill(5000, moe * hidden),
            );
            push(
                format!("{p}.mlp.shared_experts.up_proj.weight"),
                vec![moe as i64, hidden as i64],
                fill(5100, moe * hidden),
            );
            push(
                format!("{p}.mlp.shared_experts.down_proj.weight"),
                vec![hidden as i64, moe as i64],
                fill(5200, hidden * moe),
            );
        }
    }

    push(
        "model.norm.weight".into(),
        vec![hidden as i64],
        ones(hidden),
    );
    push(
        "lm_head.weight".into(),
        vec![vocab as i64, hidden as i64],
        fill(6000, vocab * hidden),
    );

    write_safetensors_f32(&dir.join("model.safetensors"), &tensors);
}

#[test]
fn synthetic_checkpoint_detects_loads_and_generates() {
    let dir = unique_temp_dir("ckpt");
    build_tiny_checkpoint(&dir);

    // Real detection path.
    let model_type = crate::models::get_model_type(&dir).expect("detection succeeds");
    assert_eq!(model_type, crate::models::ModelType::Llada2Moe);

    // Real loading path (config parse + weight load + construction).
    let model = Llada2MoeModel::load(&dir).expect("model loads");
    assert_eq!(model.mask_token_id, 15);
    assert_eq!(model.eos_token_ids(), &[1]); // eos omitted -> pad fallback

    // End-to-end block-unmasking generation through the real engine.
    let options = Llada2GenerateOptions {
        max_new_tokens: 4,
        block_length: 2,
        steps: 2,
        // A stop id the greedy stream is unlikely to emit early, so the run
        // exercises the full 4-token length path.
        extra_eos_token_ids: vec![],
        ..Llada2GenerateOptions::default()
    };
    let prompt = [3i32, 4, 5];
    let mut emitted: Vec<i32> = Vec::new();
    let stats = model
        .generate_llada2_streaming(&prompt, &options, |id| {
            emitted.push(id);
            true
        })
        .expect("generation succeeds");

    assert_eq!(emitted.len(), stats.generated_tokens);
    assert!(stats.generated_tokens <= 4);
    assert!(stats.blocks >= 1);
    assert!(stats.denoising_steps >= 1);
    assert!(matches!(
        stats.finish_reason,
        Llada2FinishReason::Length | Llada2FinishReason::Stop
    ));
    // Every generated id is a valid vocab token.
    for &id in &emitted {
        assert!((0..16).contains(&id), "token {id} out of vocab range");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn synthetic_checkpoint_forward_logits_are_finite() {
    let dir = unique_temp_dir("logits");
    build_tiny_checkpoint(&dir);
    let model = Llada2MoeModel::load(&dir).expect("model loads");

    // Prefill a prompt, then a read-only block forward: the logits over the
    // loaded weights must be well shaped and finite.
    let mut caches = model.make_diffusion_caches();
    let prompt = [3i32, 4];
    let ids = mlxcel_core::from_slice_i32(&prompt, &[1, 2]);
    let _ = model.forward_append(&ids, &mut caches, 0);
    for c in &caches {
        c.eval_state();
    }
    let block = [5i32, 15];
    let block_ids = mlxcel_core::from_slice_i32(&block, &[1, 2]);
    let logits = model.forward_readonly_logits(&block_ids, &caches, 2);
    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(shape, vec![1, 2, 16]);
    let host = to_host_f32(&logits);
    assert!(host.iter().all(|v| v.is_finite()), "logits finite");

    std::fs::remove_dir_all(&dir).ok();
}
