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

//! Checkpoint-free Raon-1 mini tests: config rules, a tiny random model in
//! the checkpoint's own layout (MXFP8 experts included) through sanitize,
//! and prefill / chunked prefill / decode / left-padding agreement.

use super::{Raon1MiniConfig, Raon1MiniLayerCache, Raon1MiniModel};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

pub(super) const VOCAB: usize = 32;
pub(super) const HIDDEN: usize = 32;
const STREAMS: usize = 4;
pub(super) const RDA_HEADS: usize = 2;
pub(super) const RDA_DIM: usize = 32;
pub(super) const KERNEL: usize = 4;
const DECAY_RANK: usize = 8;
pub(super) const MLA_HEADS: usize = 2;
pub(super) const Q_LORA: usize = 16;
pub(super) const KV_LORA: usize = 16;
pub(super) const NOPE: usize = 8;
pub(super) const ROPE: usize = 4;
pub(super) const V_HEAD: usize = 8;
pub(super) const EXPERTS: usize = 4;
pub(super) const MOE_INTER: usize = 32;
const DENSE_INTER: usize = 48;

/// Layers 0, 1, 3 are RDA (1-based 1, 2, 4), layer 2 is MLA; layer 0 is dense.
pub(super) fn tiny_config() -> Raon1MiniConfig {
    let json = serde_json::json!({
        "model_type": "raon1_mini", "vocab_size": VOCAB, "hidden_size": HIDDEN,
        "intermediate_size": DENSE_INTER, "num_hidden_layers": 4, "num_attention_heads": MLA_HEADS,
        "head_dim": RDA_DIM, "num_experts": EXPERTS, "num_experts_per_token": 2,
        "moe_intermediate_size": MOE_INTER, "num_shared_experts": 1, "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA, "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE,
        "v_head_dim": V_HEAD, "rda_decay_rank": DECAY_RANK, "raon_mhc_streams": STREAMS,
        "eos_token_id": [31],
        "rda_config": {"num_heads": RDA_HEADS, "head_dim": RDA_DIM, "conv_kernel_size": KERNEL,
            "full_rank_gate": true, "gate_lower_bound": -5.0, "rda_layers": [1, 2, 4], "mla_layers": [3]},
        "quantization_config": {"quant_method": "mxfp8", "weight_block_size": [1, 32], "dequantize": false}
    });
    Raon1MiniConfig::from_json_str(&json.to_string()).expect("tiny config parses")
}

pub(super) fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

/// Valid E4M3 bytes with magnitudes in `[2^-3, 2^1.9]` (no NaN codes).
pub(super) fn e4m3_bytes(n: usize, seed: u32) -> Vec<u8> {
    noise(n, seed)
        .iter()
        .zip(noise(n, seed + 1))
        .map(|(a, b)| {
            let sign = if *a < 0.0 { 0x80 } else { 0 };
            let exp = 4 + ((a.abs() * 9.99) as u8 % 5);
            let man = ((b + 0.5) * 7.99) as u8;
            sign | (exp << 3) | man
        })
        .collect()
}

pub(super) struct Fixture {
    weights: WeightMap,
    seed: u32,
}

impl Fixture {
    pub(super) fn new(seed: u32) -> Self {
        Self {
            weights: WeightMap::new(),
            seed,
        }
    }

    pub(super) fn into_weights(self) -> WeightMap {
        self.weights
    }

    fn next(&mut self) -> u32 {
        self.seed += 7;
        self.seed
    }

    pub(super) fn f32(&mut self, key: &str, shape: &[usize], scale: f32, offset: f32) {
        let n = shape.iter().product();
        let seed = self.next();
        let data: Vec<f32> = noise(n, seed).iter().map(|v| v * scale + offset).collect();
        let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        self.weights
            .insert(key.to_string(), mlxcel_core::from_slice_f32(&data, &shape));
    }

    fn bytes(&mut self, key: &str, data: Vec<u8>, shape: &[i32]) {
        self.weights.insert(
            key.to_string(),
            mlxcel_core::from_bytes(&data, shape, dtype::UINT8),
        );
    }

    fn site(&mut self, p: &str) {
        let w = STREAMS * HIDDEN;
        self.f32(&format!("{p}.phi_pre"), &[STREAMS, w], 0.2, 0.0);
        self.f32(&format!("{p}.phi_post"), &[STREAMS, w], 0.2, 0.0);
        self.f32(&format!("{p}.res.phi_z"), &[9, w], 0.4, 0.0);
        for leaf in ["alpha_pre", "alpha_post", "res.alpha_z", "res.eta"] {
            self.f32(&format!("{p}.{leaf}"), &[], 0.4, 1.0);
        }
        self.f32(&format!("{p}.b_pre"), &[STREAMS], 0.5, 0.0);
        self.f32(&format!("{p}.b_post"), &[STREAMS], 0.5, 0.0);
        self.f32(&format!("{p}.res.b_z"), &[9], 0.5, 0.0);
    }

    fn rda(&mut self, p: &str, torch_conv_layout: bool) {
        let proj = RDA_HEADS * RDA_DIM;
        for leaf in ["q_proj", "k_proj", "v_proj", "g_proj"] {
            self.f32(&format!("{p}.{leaf}.weight"), &[proj, HIDDEN], 0.6, 0.0);
        }
        for leaf in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            let shape: &[usize] = if torch_conv_layout {
                &[proj, 1, KERNEL]
            } else {
                &[proj, KERNEL]
            };
            self.f32(&format!("{p}.{leaf}.weight"), shape, 1.0, 0.1);
        }
        self.f32(
            &format!("{p}.f_a_proj.weight"),
            &[DECAY_RANK, HIDDEN],
            0.5,
            0.0,
        );
        self.f32(
            &format!("{p}.f_b_proj.weight"),
            &[proj, DECAY_RANK],
            0.5,
            0.0,
        );
        self.f32(
            &format!("{p}.b_proj.weight"),
            &[RDA_HEADS, HIDDEN],
            0.5,
            0.0,
        );
        self.f32(
            &format!("{p}.rda_w_proj.weight"),
            &[RDA_HEADS, HIDDEN],
            0.5,
            0.0,
        );
        self.f32(&format!("{p}.A_log"), &[RDA_HEADS], 0.5, 0.0);
        self.f32(&format!("{p}.dt_bias"), &[proj], 0.5, 0.0);
        self.f32(&format!("{p}.o_norm.weight"), &[RDA_DIM], 0.2, 1.0);
        self.f32(&format!("{p}.o_proj.weight"), &[HIDDEN, proj], 0.3, 0.0);
    }

    pub(super) fn mla(&mut self, p: &str) {
        self.f32(&format!("{p}.q_a_proj.weight"), &[Q_LORA, HIDDEN], 0.5, 0.0);
        self.f32(&format!("{p}.q_a_layernorm.weight"), &[Q_LORA], 0.2, 1.0);
        self.f32(
            &format!("{p}.q_b_proj.weight"),
            &[MLA_HEADS * (NOPE + ROPE), Q_LORA],
            0.8,
            0.0,
        );
        self.f32(
            &format!("{p}.kv_a_proj_with_mqa.weight"),
            &[KV_LORA + ROPE, HIDDEN],
            0.8,
            0.0,
        );
        self.f32(&format!("{p}.kv_a_layernorm.weight"), &[KV_LORA], 0.2, 1.0);
        self.f32(
            &format!("{p}.kv_b_proj.weight"),
            &[MLA_HEADS * (NOPE + V_HEAD), KV_LORA],
            0.8,
            0.0,
        );
        self.f32(
            &format!("{p}.g_proj.weight"),
            &[MLA_HEADS * V_HEAD, HIDDEN],
            0.5,
            0.0,
        );
        self.f32(
            &format!("{p}.o_proj.weight"),
            &[HIDDEN, MLA_HEADS * V_HEAD],
            0.4,
            0.0,
        );
    }

    fn mlp(&mut self, p: &str, inter: usize) {
        self.f32(&format!("{p}.gate_proj.weight"), &[inter, HIDDEN], 0.5, 0.0);
        self.f32(&format!("{p}.up_proj.weight"), &[inter, HIDDEN], 0.5, 0.0);
        self.f32(&format!("{p}.down_proj.weight"), &[HIDDEN, inter], 0.4, 0.0);
    }

    pub(super) fn moe(&mut self, p: &str, steering: bool) {
        self.f32(&format!("{p}.gate.weight"), &[EXPERTS, HIDDEN], 1.0, 0.0);
        self.f32(
            &format!("{p}.gate.e_score_correction_bias"),
            &[EXPERTS],
            0.2,
            0.0,
        );
        for e in 0..EXPERTS {
            for (proj, rows, cols) in [
                ("gate_proj", MOE_INTER, HIDDEN),
                ("up_proj", MOE_INTER, HIDDEN),
                ("down_proj", HIDDEN, MOE_INTER),
            ] {
                let seed = self.next();
                let key = format!("{p}.experts.{e}.{proj}");
                self.bytes(
                    &format!("{key}.weight"),
                    e4m3_bytes(rows * cols, seed),
                    &[rows as i32, cols as i32],
                );
                let scales: Vec<u8> = (0..rows * cols / 32)
                    .map(|i| 121 + (i as u8 + seed as u8) % 3)
                    .collect();
                self.bytes(
                    &format!("{key}.weight_scale_inv"),
                    scales,
                    &[rows as i32, (cols / 32) as i32],
                );
            }
        }
        self.mlp(&format!("{p}.shared_experts"), MOE_INTER);
        if steering {
            self.f32(
                &format!("{p}.reasoning_effort_steering"),
                &[HIDDEN],
                1.0,
                0.0,
            );
        }
    }
}

/// The tiny model's tensors in the checkpoint's own (pre-sanitize) layout.
pub(super) fn raw_weights() -> WeightMap {
    let config = tiny_config();
    let mut f = Fixture::new(11);
    f.f32("model.embed_tokens.weight", &[VOCAB, HIDDEN], 1.0, 0.0);
    for layer in 0..config.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        f.f32(&format!("{p}.input_layernorm.weight"), &[HIDDEN], 0.2, 1.0);
        f.f32(
            &format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
            0.2,
            1.0,
        );
        f.site(&format!("{p}.self_attn_raon_mhc"));
        f.site(&format!("{p}.mlp_raon_mhc"));
        if config.is_rda_layer(layer) {
            f.rda(&format!("{p}.self_attn"), layer == 3);
        } else {
            f.mla(&format!("{p}.self_attn"));
        }
        if config.is_moe_layer(layer) {
            f.moe(&format!("{p}.moe"), layer == 2);
        } else {
            f.mlp(&format!("{p}.mlp"), DENSE_INTER);
        }
    }
    f.f32("model.norm.weight", &[HIDDEN], 0.2, 1.0);
    f.f32("lm_head.weight", &[VOCAB, HIDDEN], 1.0, 0.0);
    f.weights
}

pub(super) fn tiny_model() -> Raon1MiniModel {
    let config = tiny_config();
    let weights = Raon1MiniModel::sanitize_weights(raw_weights(), &config).expect("sanitize");
    Raon1MiniModel::from_weights(&weights, &config).expect("tiny model loads")
}

pub(super) fn values(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, dtype::FLOAT32);
    mlxcel_core::eval(&a);
    array_to_vec_f32(&a)
}

pub(super) fn ids(rows: &[&[i32]]) -> UniquePtr<MlxArray> {
    let flat: Vec<i32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    mlxcel_core::from_slice_i32(&flat, &[rows.len() as i32, rows[0].len() as i32])
}

/// Row `b`, positions `from..to` of `[B, L, V]` logits.
fn rows_of(logits: &MlxArray, b: usize, from: usize, to: usize) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    let v = shape[2] as usize;
    let all = values(logits);
    let base = b * shape[1] as usize * v;
    all[base + from * v..base + to * v].to_vec()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= tol, "{what}: max |diff| {worst} > {tol}");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{what}: non-finite logits"
    );
}

#[test]
fn full_prefill_equals_chunked_prefill_plus_decode_including_a_padded_row() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let a: Vec<i32> = (0..12).map(|i| (i * 7 + 3) % VOCAB as i32).collect();
    let tol = 2e-3;

    let mut caches = model.make_layer_caches();
    let full = model.forward_with_caches(&ids(&[&a]), &mut caches, None);
    let want = rows_of(&full, 0, 0, 12);

    // Chunked prefill (5 + 4 tokens) then three single-token decode steps.
    let mut caches = model.make_layer_caches();
    let mut got = Vec::new();
    for (from, to) in [(0, 5), (5, 9), (9, 10), (10, 11), (11, 12)] {
        let out = model.forward_with_caches(&ids(&[&a[from..to]]), &mut caches, None);
        got.extend(rows_of(&out, 0, 0, to - from));
    }
    assert_close(&got, &want, tol, "chunked prefill + decode");
    assert!(caches.iter().all(|c| c.offset() == 12));
    assert!(matches!(&caches[2], Raon1MiniLayerCache::Attn(_)));

    // A two-row batch: row 0 is `a[..9]` behind three pad tokens, row 1 is
    // a full twelve-token prompt. Prefill, then one decode step each.
    let b: Vec<i32> = (0..12).map(|i| (i * 5 + 1) % VOCAB as i32).collect();
    let pad = [0, 0, 0];
    let row0: Vec<i32> = pad.iter().chain(&a[..9]).copied().collect();
    let mut caches = model.make_layer_caches();
    let pre = model.forward_with_caches(&ids(&[&row0, &b]), &mut caches, Some(&[3, 0]));
    let step = model.forward_with_caches(&ids(&[&a[9..10], &b[..1]]), &mut caches, Some(&[3, 0]));
    let mut row0_got = rows_of(&pre, 0, 3, 12);
    row0_got.extend(rows_of(&step, 0, 0, 1));
    assert_close(&row0_got, &want[..10 * VOCAB], tol, "left-padded row");

    let mut solo = model.make_layer_caches();
    let mut b_want = rows_of(
        &model.forward_with_caches(&ids(&[&b]), &mut solo, None),
        0,
        0,
        12,
    );
    b_want.extend(rows_of(
        &model.forward_with_caches(&ids(&[&b[..1]]), &mut solo, None),
        0,
        0,
        1,
    ));
    let mut b_got = rows_of(&pre, 1, 0, 12);
    b_got.extend(rows_of(&step, 1, 0, 1));
    assert_close(&b_got, &b_want, tol, "unpadded neighbour row");
}

/// Native batched decode (stacked RDA states, per-row MLA latent caches)
/// equals decoding each sequence alone, across fresh, mixed and B=1
/// transitions; the MLA layer (index 2) is the only paged one.
#[test]
fn native_batched_decode_matches_per_sequence_decode() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::batched_decode_harness::assert_model_owned_multistep_with_paged_lengths_prepared_b1(
        tiny_model,
        &[0, 0, 3, 0],
    );
    crate::models::batched_decode_harness::assert_model_owned_fresh_and_mixed(tiny_model);
}

crate::models::last_row_logits_harness::last_row_case!(raon1_mini_last_row_logits, tiny_model);

/// The load error for the tiny checkpoint after the two edits:
/// `raw` edits the checkpoint's own layout, `sanitized` the loader's.
fn load_error(raw: impl FnOnce(&mut WeightMap), sanitized: impl FnOnce(&mut WeightMap)) -> String {
    let config = tiny_config();
    let mut weights = raw_weights();
    raw(&mut weights);
    let mut weights = Raon1MiniModel::sanitize_weights(weights, &config).expect("sanitize");
    sanitized(&mut weights);
    match Raon1MiniModel::from_weights(&weights, &config) {
        Ok(_) => panic!("a projection with the wrong input width must fail the load"),
        Err(err) => err,
    }
}

/// A projection whose input width disagrees with the config is a load error
/// naming the tensor (issue #3457), not an abort in the first matmul. Row
/// counts alone cannot catch it: the plane is otherwise self-consistent.
#[test]
fn a_projection_with_the_wrong_input_width_fails_the_load_by_name() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let key = "model.layers.0.self_attn.q_proj.weight";
    let err = load_error(
        |w| {
            let data = vec![0.1f32; RDA_HEADS * RDA_DIM * (HIDDEN + 8)];
            let shape = [(RDA_HEADS * RDA_DIM) as i32, (HIDDEN + 8) as i32];
            w.insert(key.into(), mlxcel_core::from_slice_f32(&data, &shape));
        },
        |_| {},
    );
    assert!(err.contains(key), "{err}");
    assert!(err.contains(&(HIDDEN + 8).to_string()), "{err}");

    // The routed experts' down projection, stacked and MXFP8-packed: one more
    // quantization group on the input axis than `moe_intermediate_size`.
    let prefix = "model.layers.1.moe.switch_mlp.down_proj";
    let err = load_error(
        |_| {},
        |w| {
            let weight = &w[&format!("{prefix}.weight")];
            let scales = &w[&format!("{prefix}.scales")];
            let (ws, ss) = (
                mlxcel_core::array_shape(weight),
                mlxcel_core::array_shape(scales),
            );
            let (wd, sd) = (
                mlxcel_core::array_dtype(weight),
                mlxcel_core::array_dtype(scales),
            );
            let words_per_group = ws[2] / ss[2];
            let wider = mlxcel_core::zeros(&[ws[0], ws[1], ws[2] + words_per_group], wd);
            let wider_scales = mlxcel_core::zeros(&[ss[0], ss[1], ss[2] + 1], sd);
            w.insert(format!("{prefix}.weight"), wider);
            w.insert(format!("{prefix}.scales"), wider_scales);
        },
    );
    assert!(err.contains(&format!("{prefix}.weight")), "{err}");
}
