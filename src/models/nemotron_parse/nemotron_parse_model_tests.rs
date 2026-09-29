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

//! Decode-path tests for the Nemotron-Parse port (issue #1369) on a tiny
//! synthetic model: encoder output geometry, head selection, the causal
//! cache (incremental decode equals teacher-forced prefill), and the greedy
//! loop's bounds.

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::checkpoint::canonicalize_keys;
use super::config::{NemotronParseConfig, NemotronParseTextConfig, NemotronParseVisionConfig};
use super::model::NemotronParseModel;
use crate::models::florence2::Florence2Quantization;

fn host(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    array_to_vec_f32(&a)
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
    }
}

// ---------------------------------------------------------------------------
// Tiny synthetic model
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn vals(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }
}

fn tiny_config() -> NemotronParseConfig {
    NemotronParseConfig {
        vision: NemotronParseVisionConfig {
            hidden_size: 8,
            num_heads: 2,
            mlp_dim: 16,
            num_layers: 1,
            patch_size: 2,
            pos_grid: 4,
            num_cls_tokens: 2,
            num_register_tokens: 2,
            summary_idxs: vec![0, 1],
            neck_dim: 8,
            neck_kernel_w: 4,
            image_size: (8, 8),
        },
        text: NemotronParseTextConfig {
            d_model: 8,
            decoder_layers: 2,
            decoder_attention_heads: 2,
            decoder_ffn_dim: 16,
            vocab_size: 20,
            scale_embedding: true,
        },
        decoder_start_token_id: 2,
        eos_token_id: 2,
        bos_token_id: 0,
        pad_token_id: 1,
        tie_word_embeddings: true,
        max_sequence_length: 64,
        quantization: Florence2Quantization::DENSE,
    }
}

fn put(w: &mut WeightMap, rng: &mut Rng, key: &str, shape: &[i32]) {
    let n: i32 = shape.iter().product();
    let v = rng.vals(n as usize, 0.8);
    w.insert(key.to_string(), mlxcel_core::from_slice_f32(&v, shape));
}

fn put_ln(w: &mut WeightMap, prefix: &str, d: i32) {
    w.insert(
        format!("{prefix}.weight"),
        mlxcel_core::from_slice_f32(&vec![1.0; d as usize], &[d]),
    );
    w.insert(
        format!("{prefix}.bias"),
        mlxcel_core::from_slice_f32(&vec![0.0; d as usize], &[d]),
    );
}

/// Canonical-layout weights (MLX conv layout) for [`tiny_config`].
fn tiny_weights() -> WeightMap {
    let mut rng = Rng(7);
    let r = &mut rng;
    let mut w = WeightMap::new();
    let v = "vision_tower";
    put(&mut w, r, &format!("{v}.patch_embed.weight"), &[8, 12]);
    put(&mut w, r, &format!("{v}.pos_embed"), &[1, 16, 8]);
    put(&mut w, r, &format!("{v}.cls_token"), &[4, 8]);
    let b = format!("{v}.blocks.0");
    put_ln(&mut w, &format!("{b}.norm1"), 8);
    put_ln(&mut w, &format!("{b}.norm2"), 8);
    for (name, o, i) in [
        ("attn.qkv", 24, 8),
        ("attn.proj", 8, 8),
        ("mlp.fc1", 16, 8),
        ("mlp.fc2", 8, 16),
    ] {
        put(&mut w, r, &format!("{b}.{name}.weight"), &[o, i]);
        put(&mut w, r, &format!("{b}.{name}.bias"), &[o]);
    }
    let n = format!("{v}.neck");
    put(&mut w, r, &format!("{n}.conv1.weight"), &[8, 1, 8]);
    put(&mut w, r, &format!("{n}.conv1.bias"), &[8]);
    put(&mut w, r, &format!("{n}.conv2.weight"), &[8, 1, 4, 8]);
    for i in 1..=3 {
        put_ln(&mut w, &format!("{n}.layer_norm{i}"), 8);
    }
    put(&mut w, r, &format!("{n}.sum_proj.weight"), &[8, 16]);
    put(&mut w, r, &format!("{n}.sum_proj.bias"), &[8]);

    let t = "language_model.model";
    put(&mut w, r, &format!("{t}.shared.weight"), &[20, 8]);
    let d = format!("{t}.decoder");
    put_ln(&mut w, &format!("{d}.layernorm_embedding"), 8);
    put_ln(&mut w, &format!("{d}.layer_norm"), 8);
    for l in 0..2 {
        let p = format!("{d}.layers.{l}");
        for attn in ["self_attn", "encoder_attn"] {
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                put(&mut w, r, &format!("{p}.{attn}.{proj}.weight"), &[8, 8]);
                put(&mut w, r, &format!("{p}.{attn}.{proj}.bias"), &[8]);
            }
        }
        for norm in [
            "self_attn_layer_norm",
            "encoder_attn_layer_norm",
            "final_layer_norm",
        ] {
            put_ln(&mut w, &format!("{p}.{norm}"), 8);
        }
        put(&mut w, r, &format!("{p}.fc1.weight"), &[16, 8]);
        put(&mut w, r, &format!("{p}.fc1.bias"), &[16]);
        put(&mut w, r, &format!("{p}.fc2.weight"), &[8, 16]);
        put(&mut w, r, &format!("{p}.fc2.bias"), &[8]);
    }
    w
}

fn tiny_model() -> NemotronParseModel {
    NemotronParseModel::from_weights(&canonicalize_keys(tiny_weights()), tiny_config()).unwrap()
}

fn tiny_pixels() -> UniquePtr<MlxArray> {
    let v = Rng(11).vals(3 * 8 * 8, 2.0);
    mlxcel_core::from_slice_f32(&v, &[1, 3, 8, 8])
}

#[test]
fn encoder_output_is_compressed_grid_plus_summary_row() {
    let model = tiny_model();
    let enc = model.encode_image(&tiny_pixels()).unwrap();
    // 4x4 patches, 4x horizontal compression -> 4 rows, plus 1 summary row.
    assert_eq!(mlxcel_core::array_shape(&enc), vec![1, 5, 8]);
    assert!(host(&enc).iter().all(|x| x.is_finite()));
}

#[test]
fn untied_head_is_required_when_embeddings_are_not_tied() {
    let mut cfg = tiny_config();
    cfg.tie_word_embeddings = false;
    let w = canonicalize_keys(tiny_weights());
    assert!(NemotronParseModel::from_weights(&w, cfg.clone()).is_err());
    let mut w = w;
    put(
        &mut w,
        &mut Rng(3),
        "language_model.lm_head.weight",
        &[20, 8],
    );
    assert!(NemotronParseModel::from_weights(&w, cfg.clone()).is_ok());
    // A head whose rows disagree with the vocabulary is refused.
    put(
        &mut w,
        &mut Rng(3),
        "language_model.lm_head.weight",
        &[21, 8],
    );
    assert!(NemotronParseModel::from_weights(&w, cfg).is_err());
}

#[test]
fn decoder_prenorm_layer_shape_and_cache_growth() {
    let model = tiny_model();
    let enc = model.encode_image(&tiny_pixels()).unwrap();
    let seq = [2, 5, 7, 9];

    // Teacher-forced prefill of all four tokens.
    let mut full = model.make_cache();
    let ids = mlxcel_core::from_slice_i32(&seq, &[1, 4]);
    let logits = model.decode(&ids, &enc, &mut full);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 4, 20]);
    assert_eq!(full.offset(), 4);
    let last_full = host(&mlxcel_core::slice(&logits, &[0, 3, 0], &[1, 4, 20]));

    // Prefill three, then one incremental step: same last-position logits.
    let mut inc = model.make_cache();
    let ids3 = mlxcel_core::from_slice_i32(&seq[..3], &[1, 3]);
    let _ = model.decode(&ids3, &enc, &mut inc);
    assert_eq!(inc.offset(), 3);
    let step = mlxcel_core::from_slice_i32(&seq[3..], &[1, 1]);
    let step_logits = model.decode(&step, &enc, &mut inc);
    assert_eq!(inc.offset(), 4);
    close(&host(&step_logits), &last_full, 1e-4);
}

#[test]
fn generate_respects_the_token_budget_and_seed_checks() {
    let model = tiny_model();
    let px = tiny_pixels();
    let out = model.generate(&px, &[2, 0, 5], 3, 1.0, None).unwrap();
    assert!(out.tokens.len() <= 3);
    assert!(out.tokens.iter().all(|t| (0..20).contains(t) && *t != 2));
    // Penalized decoding runs through the host argmax path.
    assert!(model.generate(&px, &[2, 0, 5], 3, 1.1, None).is_ok());
    // Out-of-vocabulary seed ids, empty seeds, and bad penalties are errors.
    assert!(model.generate(&px, &[2, 25], 3, 1.0, None).is_err());
    assert!(model.generate(&px, &[], 3, 1.0, None).is_err());
    assert!(model.generate(&px, &[2], 3, 0.0, None).is_err());
    // A cancelled run stops before decoding anything.
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let out = model.generate(&px, &[2], 3, 1.0, Some(&cancel)).unwrap();
    assert!(out.tokens.is_empty() && !out.hit_eos);
}
