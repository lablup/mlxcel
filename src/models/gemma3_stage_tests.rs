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

//! Split-boundary parity for [`Gemma3StageModel`] (issue #2654).
//!
//! A two-stage pipeline of the same weights must produce the logits the
//! single-process [`Gemma3Model`] produces, bit for bit. The stage model used
//! to build explicit causal and sliding masks for every prefill while the
//! dense forward leaves both `None` (maskless causal SDPA) inside the window,
//! so the two ran different attention kernels and the remote pipeline parity
//! tests failed on real checkpoints. The fixture is bf16 with a head dim the
//! fused SDPA kernels serve, so the masked and maskless paths are distinct
//! kernels here too, and the split puts stage 1 on a global layer, mid
//! sliding pattern.

use super::{Cache, Gemma3Model, Gemma3StageModel, ModelArgs};
use crate::distributed::pipeline::{LayerFilter, StageExecutionOutput};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const HIDDEN: i32 = 256;
const VOCAB: i32 = 64;
const INTERMEDIATE: i32 = 512;
const HEAD_DIM: i32 = 256;
const HEADS: i32 = 8;
const KV_HEADS: i32 = 1;
const LAYERS: usize = 4;
const WINDOW: i32 = 16;
/// The activation dtype the real checkpoint runs in on CUDA.
const DTYPE: i32 = mlxcel_core::dtype::FLOAT16;

fn tensor(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let data: Vec<f32> = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            unit * 0.2 - 0.1
        })
        .collect();
    let f32_arr = mlxcel_core::from_slice_f32(&data, shape);
    mlxcel_core::astype(&f32_arr, DTYPE)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    let f32_arr = mlxcel_core::from_slice_f32(&vec![1.0_f32; len], shape);
    mlxcel_core::astype(&f32_arr, DTYPE)
}

fn args() -> ModelArgs {
    serde_json::from_str(&format!(
        r#"{{
            "model_type": "gemma3_text",
            "hidden_size": {HIDDEN},
            "num_hidden_layers": {LAYERS},
            "intermediate_size": {INTERMEDIATE},
            "num_attention_heads": {HEADS},
            "num_key_value_heads": {KV_HEADS},
            "head_dim": {HEAD_DIM},
            "rms_norm_eps": 1e-6,
            "vocab_size": {VOCAB},
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "query_pre_attn_scalar": {HEAD_DIM}.0,
            "sliding_window": {WINDOW},
            "sliding_window_pattern": 2,
            "max_position_embeddings": 4096
        }}"#
    ))
    .expect("fixture config must parse")
}

fn weights() -> WeightMap {
    let mut w = WeightMap::new();
    w.insert(
        "model.embed_tokens.weight".into(),
        tensor(&[VOCAB, HIDDEN], 1),
    );
    w.insert("lm_head.weight".into(), tensor(&[VOCAB, HIDDEN], 2));
    w.insert("model.norm.weight".into(), ones(&[HIDDEN]));
    for layer in 0..LAYERS {
        let seed = 100 + layer as u64 * 10;
        let attn = format!("model.layers.{layer}.self_attn");
        for (name, shape, offset) in [
            ("q_proj", [HEADS * HEAD_DIM, HIDDEN], 0),
            ("k_proj", [KV_HEADS * HEAD_DIM, HIDDEN], 1),
            ("v_proj", [KV_HEADS * HEAD_DIM, HIDDEN], 2),
            ("o_proj", [HIDDEN, HEADS * HEAD_DIM], 3),
        ] {
            w.insert(
                format!("{attn}.{name}.weight"),
                tensor(&shape, seed + offset),
            );
        }
        w.insert(format!("{attn}.q_norm.weight"), ones(&[HEAD_DIM]));
        w.insert(format!("{attn}.k_norm.weight"), ones(&[HEAD_DIM]));

        let mlp = format!("model.layers.{layer}.mlp");
        for (name, shape, offset) in [
            ("gate_proj", [INTERMEDIATE, HIDDEN], 4),
            ("up_proj", [INTERMEDIATE, HIDDEN], 5),
            ("down_proj", [HIDDEN, INTERMEDIATE], 6),
        ] {
            w.insert(
                format!("{mlp}.{name}.weight"),
                tensor(&shape, seed + offset),
            );
        }
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            w.insert(
                format!("model.layers.{layer}.{norm}.weight"),
                ones(&[HIDDEN]),
            );
        }
    }
    w
}

fn stage(weights: &WeightMap, args: &ModelArgs, range: std::ops::Range<usize>) -> Gemma3StageModel {
    let filter = LayerFilter {
        has_embedding: range.start == 0,
        has_lm_head: range.end == LAYERS,
        layer_range: range,
    };
    Gemma3StageModel::from_filtered_weights(weights, args, &filter, 0)
        .expect("fixture stage must load")
}

fn two_stage_logits(
    stage0: &Gemma3StageModel,
    stage1: &Gemma3StageModel,
    ids: &MlxArray,
    caches0: &mut [Cache],
    caches1: &mut [Cache],
) -> UniquePtr<MlxArray> {
    let StageExecutionOutput::HiddenStates(hidden) = stage0
        .execute_from_token_ids(ids, caches0)
        .expect("stage 0 must run")
    else {
        panic!("stage 0 must hand hidden states downstream");
    };
    let StageExecutionOutput::Logits(logits) = stage1
        .execute_from_hidden_states(hidden, caches1)
        .expect("stage 1 must run")
    else {
        panic!("the last stage must produce logits");
    };
    logits
}

fn assert_bitwise_equal(got: &MlxArray, want: &MlxArray, what: &str) {
    assert_eq!(
        mlxcel_core::array_dtype(got),
        mlxcel_core::array_dtype(want),
        "{what}: dtype"
    );
    assert_eq!(
        mlxcel_core::array_shape(got),
        mlxcel_core::array_shape(want),
        "{what}: shape"
    );
    let got_bytes = mlxcel_core::array_to_raw_bytes(got);
    let want_bytes = mlxcel_core::array_to_raw_bytes(want);
    if got_bytes != want_bytes {
        let diff = mlxcel_core::abs(&mlxcel_core::subtract(
            &mlxcel_core::astype(got, mlxcel_core::dtype::FLOAT32),
            &mlxcel_core::astype(want, mlxcel_core::dtype::FLOAT32),
        ));
        let max = mlxcel_core::item_f32(&mlxcel_core::max_all(&diff));
        panic!("{what}: stage logits differ from the dense logits (max abs diff {max:e})");
    }
}

/// Prefills inside the window, each followed by one decode step, split mid
/// pattern.
#[test]
fn two_stage_split_matches_dense_prefill_and_decode_bitwise() {
    let args = args();
    let weights = weights();
    let dense = Gemma3Model::from_weights(&weights, &args).expect("fixture model must load");
    // Layer 0 is sliding and layer 1 global, so stage 1 opens on a global
    // layer and owns one layer of each kind.
    let stage0 = stage(&weights, &args, 0..1);
    let stage1 = stage(&weights, &args, 1..LAYERS);

    // The two-token prompt is the shape of the real-model parity tests and
    // the one where the masked and maskless SDPA kernels disagree.
    for ids in [&[3, 17][..], &[3, 17, 42, 5, 9, 60, 33, 1][..]] {
        let prompt = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
        let decode = mlxcel_core::from_slice_i32(&[11], &[1, 1]);

        let mut dense_caches = dense.make_caches();
        let dense_prefill = dense.forward_with_caches(&prompt, &mut dense_caches);
        let dense_decode = dense.forward_with_caches(&decode, &mut dense_caches);

        let mut caches0 = stage0.make_caches();
        let mut caches1 = stage1.make_caches();
        let stage_prefill = two_stage_logits(&stage0, &stage1, &prompt, &mut caches0, &mut caches1);
        let stage_decode = two_stage_logits(&stage0, &stage1, &decode, &mut caches0, &mut caches1);

        let len = ids.len();
        assert_bitwise_equal(
            &stage_prefill,
            &dense_prefill,
            &format!("{len}-token prefill"),
        );
        assert_bitwise_equal(&stage_decode, &dense_decode, &format!("decode after {len}"));
    }
}

/// A prefill longer than the sliding window takes the over-window branch of
/// the shared mask decision in both models.
#[test]
fn two_stage_split_matches_dense_over_window_prefill_bitwise() {
    let args = args();
    let weights = weights();
    let dense = Gemma3Model::from_weights(&weights, &args).expect("fixture model must load");
    let stage0 = stage(&weights, &args, 0..1);
    let stage1 = stage(&weights, &args, 1..LAYERS);

    let ids: Vec<i32> = (0..(WINDOW + 8)).map(|i| (i * 7 + 3) % VOCAB).collect();
    let prompt = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);

    let mut dense_caches = dense.make_caches();
    let dense_prefill = dense.forward_with_caches(&prompt, &mut dense_caches);

    let mut caches0 = stage0.make_caches();
    let mut caches1 = stage1.make_caches();
    let stage_prefill = two_stage_logits(&stage0, &stage1, &prompt, &mut caches0, &mut caches1);

    assert_bitwise_equal(&stage_prefill, &dense_prefill, "over-window prefill");
}
