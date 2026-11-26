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

//! Unit tests for the DFlash2 drafter: config contract, grouped convolution
//! block boundaries, the selector chain, the context window, and a tiny
//! random-weight drafter end to end.

use super::config::RaonDFlash2Config;
use super::conv::grouped_conv;
use super::drafter::RaonDFlash2Drafter;
use super::sanitize::{expected_shapes, sanitize_weights};
use super::selector::{Selection, chain_device, chain_host};
use crate::drafter::Drafter;
use crate::dtype;
use crate::ffi;
use crate::layers::{Embedding, UnifiedEmbedding};
use crate::weights::WeightMap;
use serde_json::json;

/// The published `KRAFTON/Raon-1-mini-dflash2` config (revision 777533d8),
/// reduced to the fields the loader reads plus the markers.
fn release_config() -> serde_json::Value {
    json!({
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "block_size": 8,
        "dflash_config": {
            "block_size": 8, "causal": false, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 259839, "selector_enabled": true, "selector_rank": 256,
            "selector_top_k": 16, "swa_window_size": 2048, "target_embedding_size": 2560,
            "target_feature_size": 51200, "target_layer_ids": [1, 6, 11, 16, 21],
            "target_streams_per_layer": 4, "use_aux_hidden_state": true, "use_swa": true
        },
        "head_dim": 128, "hidden_size": 2560, "intermediate_size": 8192,
        "num_attention_heads": 20, "num_hidden_layers": 5, "num_key_value_heads": 5,
        "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
        "vocab_size": 259840
    })
}

fn tiny_config_json() -> serde_json::Value {
    json!({
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3",
        "dflash_config": {
            "block_size": 4, "causal": false, "conv_group_size": 4, "conv_kernel_size": 2,
            "mask_token_id": 31, "selector_rank": 4, "selector_top_k": 3,
            "swa_window_size": 16, "target_feature_size": 2 * 2 * 8,
            "target_layer_ids": [0, 2], "target_streams_per_layer": 2
        },
        "head_dim": 4, "hidden_size": 8, "intermediate_size": 12,
        // Grouped like the release (20 query heads over 5 KV heads).
        "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2,
        "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 10000.0},
        "vocab_size": 32
    })
}

#[test]
fn release_config_loads() {
    let cfg = RaonDFlash2Config::from_json(&release_config()).expect("release config loads");
    assert_eq!(cfg.target_layer_ids, vec![1, 6, 11, 16, 21]);
    assert_eq!(cfg.target_streams_per_layer, 4);
    assert_eq!(cfg.target_feature_size, 51200);
    assert_eq!((cfg.block_size, cfg.window, cfg.conv_taps), (8, 2048, 2));
    assert_eq!(
        (cfg.selector_top_k, cfg.selector_rank, cfg.conv_groups()),
        (16, 256, 160)
    );
    assert_eq!(cfg.mask_token_id, 259839);
    assert!((cfg.rope_theta - 1.0e7).abs() < 1.0);
    assert!(RaonDFlash2Config::is_raon_dflash2_config(&release_config()));
    assert!(crate::drafter::dflash::is_dflash_drafter_config(
        &release_config()
    ));
}

#[test]
fn mismatched_target_layer_ids_is_a_named_load_error() {
    let mut c = release_config();
    c["dflash_config"]["target_layer_ids"] = json!([1, 6, 11, 16]);
    let err = RaonDFlash2Config::from_json(&c).expect_err("four taps cannot fill 51200 features");
    assert!(err.contains("target_layer_ids"), "{err}");

    let mut c = release_config();
    c["dflash_config"]["target_layer_ids"] = json!([1, 6, 6, 16, 21]);
    let err = RaonDFlash2Config::from_json(&c).expect_err("duplicate taps");
    assert!(err.contains("target_layer_ids"), "{err}");

    // A tap the target does not have, and a stream count it does not carry.
    let cfg = RaonDFlash2Config::from_json(&release_config()).expect("loads");
    let err = cfg
        .validate_target(20, Some(4), 2560, 259840)
        .expect_err("layer 21 of 20");
    assert!(err.contains("target_layer_ids"), "{err}");
    let err = cfg
        .validate_target(25, Some(1), 2560, 259840)
        .expect_err("one stream");
    assert!(err.contains("target_streams_per_layer"), "{err}");
    cfg.validate_target(25, Some(4), 2560, 259840)
        .expect("Raon-1 mini matches");
}

#[test]
fn config_rejects_bad_contract() {
    for (field, patch) in [
        ("causal", json!({"causal": true})),
        ("selector_enabled", json!({"selector_enabled": false})),
        ("block_size", json!({"block_size": 1})),
        ("block_size", json!({"block_size": 64})),
        ("mask_token_id", json!({"mask_token_id": 259840})),
        ("conv_group_size", json!({"conv_group_size": 3})),
        (
            "target_streams_per_layer",
            json!({"target_streams_per_layer": 0}),
        ),
    ] {
        let mut c = release_config();
        for (k, v) in patch.as_object().expect("object") {
            c["dflash_config"][k] = v.clone();
        }
        let err = RaonDFlash2Config::from_json(&c).expect_err(field);
        assert!(err.contains(field), "{field}: {err}");
    }
    let mut c = release_config();
    c["architectures"] = json!(["Qwen3ForCausalLM"]);
    assert!(RaonDFlash2Config::from_json(&c).is_err());
}

fn values(x: &ffi::MlxArray) -> Vec<f32> {
    let x = ffi::astype(x, dtype::FLOAT32);
    ffi::eval(&x);
    crate::utils::array_to_vec_f32(&x)
}

/// Host reference of the grouped convolution (the geometry spelled out
/// argument by argument, like `grouped_conv`).
#[allow(clippy::too_many_arguments)]
fn host_conv(
    h: &[f32],
    delta: &[f32],
    base: &[f32],
    n: usize,
    groups: usize,
    gsize: usize,
    taps: usize,
    block: usize,
) -> Vec<f32> {
    let width = groups * gsize;
    let mut out = vec![0.0f32; n * width];
    for t in 0..n {
        for tap in 0..taps {
            if t % block < tap {
                continue;
            }
            for g in 0..groups {
                for j in 0..gsize {
                    let c =
                        base[tap * width + g * gsize + j] + delta[(t * taps + tap) * groups + g];
                    out[t * width + g * gsize + j] += c * h[(t - tap) * width + g * gsize + j];
                }
            }
        }
    }
    out
}

#[test]
fn grouped_conv_never_mixes_positions_across_a_block_boundary() {
    let (n, groups, gsize, taps, block) = (8usize, 2usize, 3usize, 2usize, 4usize);
    let width = groups * gsize;
    let h: Vec<f32> = (0..n * width).map(|i| (i as f32 * 0.37).sin()).collect();
    let delta: Vec<f32> = (0..n * taps * groups)
        .map(|i| (i as f32 * 0.11).cos() * 0.5)
        .collect();
    let base: Vec<f32> = (0..taps * width).map(|i| 0.25 + i as f32 * 0.05).collect();
    let run = |h: &[f32]| {
        values(&grouped_conv(
            &ffi::from_slice_f32(h, &[n as i32, width as i32]),
            &ffi::from_slice_f32(&delta, &[n as i32, taps as i32, groups as i32]),
            &ffi::from_slice_f32(&base, &[taps as i32, width as i32]),
            block as i32,
            groups as i32,
            gsize as i32,
        ))
    };
    let out = run(&h);
    let expected = host_conv(&h, &delta, &base, n, groups, gsize, taps, block);
    for (a, b) in out.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
    // Perturb the last row of block 0: block 1 must not move.
    let mut perturbed = h.clone();
    for v in &mut perturbed[(block - 1) * width..block * width] {
        *v += 10.0;
    }
    let moved = run(&perturbed);
    assert_ne!(moved[(block - 1) * width], out[(block - 1) * width]);
    assert_eq!(&moved[block * width..], &out[block * width..]);
}

#[test]
fn selector_chain_matches_a_hand_computed_three_step_case() {
    // 3 steps, K = 3. scores[s][p][c].
    #[rustfmt::skip]
    let scores: Vec<f32> = vec![
        // step 0: only row 0 is read (every row holds the bonus); best c = 2.
        0.1, 0.5, 0.9,   9.0, 0.0, 0.0,   9.0, 0.0, 0.0,
        // step 1: row 2 is read; best c = 0.
        9.0, 0.0, 0.0,   0.0, 9.0, 0.0,   0.8, 0.3, 0.2,
        // step 2: row 0 is read; best c = 1 (row 1 would pick 2).
        0.0, 0.7, 0.6,   0.0, 0.0, 9.0,   9.0, 0.0, 0.0,
    ];
    let candidates: Vec<i32> = vec![10, 11, 12, 20, 21, 22, 30, 31, 32];
    assert_eq!(chain_host(&scores, &candidates, 3, 3), vec![12, 20, 31]);
    let selection = Selection {
        scores: ffi::from_slice_f32(&scores, &[3, 3, 3]),
        candidates: ffi::from_slice_i32(&candidates, &[3, 3]),
    };
    let device = chain_device(&selection);
    ffi::eval(&device);
    assert_eq!(
        crate::drafter::dflash::materialize_argmax_i32_vec(&device, 3),
        vec![12, 20, 31]
    );
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }
    fn array(&mut self, shape: &[i32], scale: f32) -> cxx::UniquePtr<ffi::MlxArray> {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.next(scale)).collect();
        ffi::from_slice_f32(&data, shape)
    }
}

fn tiny_weights(cfg: &RaonDFlash2Config, seed: u64) -> WeightMap {
    let mut rng = Lcg(seed);
    let mut w = WeightMap::new();
    for (key, shape) in expected_shapes(cfg) {
        let array = if key.ends_with("norm.weight") {
            ffi::ones(&shape, dtype::FLOAT32)
        } else {
            rng.array(&shape, 0.5)
        };
        w.insert(key, array);
    }
    w
}

fn tiny_drafter(seed: u64) -> (RaonDFlash2Drafter, RaonDFlash2Config) {
    let cfg = RaonDFlash2Config::from_json(&tiny_config_json()).expect("tiny config");
    let mut weights = tiny_weights(&cfg, seed);
    sanitize_weights(&mut weights, &cfg).expect("inventory complete");
    let mut drafter = RaonDFlash2Drafter::from_weights(&weights, cfg.clone()).expect("builds");
    let embed = Lcg(seed ^ 0x51).array(&[cfg.vocab_size as i32, cfg.hidden_size as i32], 0.5);
    drafter.model.embed_tokens = Some(UnifiedEmbedding::Regular(Embedding::new(embed)));
    (drafter, cfg)
}

fn features(cfg: &RaonDFlash2Config, rows: i32, seed: u64) -> cxx::UniquePtr<ffi::MlxArray> {
    Lcg(seed).array(&[1, rows, cfg.target_feature_size as i32], 1.0)
}

#[test]
fn sanitize_refuses_the_vendor_nvfp4_format_and_misshapen_tensors() {
    let cfg = RaonDFlash2Config::from_json(&tiny_config_json()).expect("tiny config");
    let mut w = tiny_weights(&cfg, 3);
    w.insert("w.norm.weight".into(), ffi::ones(&[8], dtype::FLOAT32));
    let err = sanitize_weights(&mut w, &cfg).expect_err("vendor format");
    assert!(err.contains("NVFP4"), "{err}");

    let mut w = tiny_weights(&cfg, 3);
    w.insert("fc.weight".into(), ffi::ones(&[8, 31], dtype::FLOAT32));
    let err = sanitize_weights(&mut w, &cfg).expect_err("fc width");
    assert!(err.contains("fc.weight"), "{err}");

    let mut w = tiny_weights(&cfg, 3);
    w.remove("layers.1.mlp_conv.base_kernel");
    let err = sanitize_weights(&mut w, &cfg).expect_err("missing tensor");
    assert!(err.contains("layers.1.mlp_conv.base_kernel"), "{err}");
}

#[test]
fn draft_block_is_deterministic_and_in_vocabulary() {
    let (mut a, cfg) = tiny_drafter(7);
    let (mut b, _) = tiny_drafter(7);
    let sampler = crate::generate::SamplingConfig::default();
    let feats = features(&cfg, 5, 11);
    let da = a.draft_block(3, Some(&feats), 4, &sampler).expect("drafts");
    let db = b.draft_block(3, Some(&feats), 4, &sampler).expect("drafts");
    assert_eq!(da, db);
    assert_eq!(da.len(), 3);
    assert!(da.iter().all(|&t| (0..cfg.vocab_size as i32).contains(&t)));
    assert_eq!(a.context_position(), 5);
    assert!(a.draft_block(3, None, 4, &sampler).is_err());
    assert!(a.draft_block(3, Some(&feats), 1, &sampler).is_err());
    let past_trained = a.model.config.block_size + 1;
    assert!(
        a.draft_block(3, Some(&feats), past_trained, &sampler)
            .is_err()
    );
}

#[test]
fn context_window_keeps_the_newest_positions_at_their_absolute_offsets() {
    let (mut drafter, cfg) = tiny_drafter(5);
    let sampler = crate::generate::SamplingConfig::default();
    let window = cfg.window as i32;
    // A prompt longer than the window: the oldest rows are skipped, the
    // offset still counts them.
    let long = features(&cfg, window + 6, 2);
    drafter
        .draft_block(1, Some(&long), 4, &sampler)
        .expect("drafts");
    assert_eq!(drafter.context_position(), window + 6);
    assert_eq!(drafter.context_len(), window);
    let next = features(&cfg, 3, 9);
    drafter
        .draft_block(2, Some(&next), 4, &sampler)
        .expect("drafts");
    assert_eq!(drafter.context_position(), window + 9);
    assert_eq!(drafter.context_len(), window);

    // The skipped rows cannot change the draft: a drafter that saw the
    // same rows in two appends (six, then the newest window) holds the same
    // window at the same positions and proposes the same block.
    let width = cfg.target_feature_size as i32;
    let (mut whole, _) = tiny_drafter(5);
    let a = whole
        .draft_block(4, Some(&long), 4, &sampler)
        .expect("drafts");
    let (mut split, _) = tiny_drafter(5);
    let head = ffi::slice(&long, &[0, 0, 0], &[1, 6, width]);
    let rest = ffi::slice(&long, &[0, 6, 0], &[1, window + 6, width]);
    split
        .draft_block(9, Some(&head), 4, &sampler)
        .expect("drafts");
    let b = split
        .draft_block(4, Some(&rest), 4, &sampler)
        .expect("drafts");
    assert_eq!(split.context_position(), whole.context_position());
    assert_eq!(a, b);
}

#[test]
fn features_of_the_wrong_width_are_a_draft_error_not_a_panic() {
    let (mut drafter, cfg) = tiny_drafter(3);
    let sampler = crate::generate::SamplingConfig::default();
    let narrow = ffi::ones(&[1, 4, cfg.target_feature_size as i32 - 1], dtype::FLOAT32);
    let err = drafter
        .draft_block(1, Some(&narrow), 4, &sampler)
        .expect_err("narrow features");
    assert!(err.to_string().contains("wide"), "{err}");
    let Err(err) = drafter.draft_block_array(1, Some(&narrow), 4, &sampler) else {
        panic!("narrow features must not draft");
    };
    assert!(err.to_string().contains("wide"), "{err}");
    assert_eq!(drafter.context_position(), 0, "nothing appended");
}

#[test]
fn skipping_the_uncaptured_rows_drafts_as_the_whole_prompt_does() {
    use crate::drafter::Drafter;
    let sampler = crate::generate::SamplingConfig::default();
    let (mut whole, cfg) = tiny_drafter(5);
    let window = cfg.window as i32;
    let width = cfg.target_feature_size as i32;
    let long = features(&cfg, window + 6, 2);
    let a = whole
        .draft_block(4, Some(&long), 4, &sampler)
        .expect("drafts");

    // The target captured only the newest window and reported the rest.
    let (mut tail, _) = tiny_drafter(5);
    assert_eq!(tail.context_window(), Some(cfg.window));
    tail.skip_context(6);
    let newest = ffi::slice(&long, &[0, 6, 0], &[1, window + 6, width]);
    let b = tail
        .draft_block(4, Some(&newest), 4, &sampler)
        .expect("drafts");
    assert_eq!(tail.context_position(), whole.context_position());
    assert_eq!(tail.context_len(), whole.context_len());
    assert_eq!(a, b);
}

#[test]
fn config_rejects_dimensions_that_overflow_an_i32() {
    let mut c = release_config();
    c["head_dim"] = json!(1 << 20);
    c["num_attention_heads"] = json!(1 << 20);
    let err = RaonDFlash2Config::from_json(&c).expect_err("q width overflows");
    assert!(err.contains("head_dim"), "{err}");
    let mut c = release_config();
    c["dflash_config"]["target_feature_size"] = json!(1u64 << 33);
    let err = RaonDFlash2Config::from_json(&c).expect_err("feature width overflows");
    assert!(err.contains("target_feature_size"), "{err}");
}
