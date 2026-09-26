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

//! Wrapper-level tests for the Nemotron H Nano Omni VLM.
//!
//! These tests exercise the multimodal projector and the
//! `pixel_shuffle` downsample without bringing up the full text
//! backbone, so they remain fast and platform-independent.

// `super` is the `nemotron_h_nano_omni_vl` module that includes this
// file via `#[path]`, so types defined there are reachable directly.
use super::{
    NemotronHNanoOmniProjector, NemotronHNanoOmniVlConfig, read_int32_vec,
    subsampling_output_lengths,
};
use mlxcel_core::weights::WeightMap;

const PROJ_IN_FEATURES: usize = 16;
const PROJ_HIDDEN: usize = 32;
const TEXT_HIDDEN: usize = 8;

fn build_projector_weights(prefix: &str) -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.layers.0.weight"),
        mlxcel_core::ones(&[PROJ_IN_FEATURES as i32], mlxcel_core::dtype::FLOAT32),
    );

    let fc1_data: Vec<f32> = (0..PROJ_HIDDEN * PROJ_IN_FEATURES)
        .map(|i| (i as f32) * 1e-3)
        .collect();
    weights.insert(
        format!("{prefix}.layers.1.weight"),
        mlxcel_core::from_slice_f32(&fc1_data, &[PROJ_HIDDEN as i32, PROJ_IN_FEATURES as i32]),
    );

    let fc2_data: Vec<f32> = (0..TEXT_HIDDEN * PROJ_HIDDEN)
        .map(|i| (i as f32) * 1e-3)
        .collect();
    weights.insert(
        format!("{prefix}.layers.3.weight"),
        mlxcel_core::from_slice_f32(&fc2_data, &[TEXT_HIDDEN as i32, PROJ_HIDDEN as i32]),
    );
    weights
}

#[test]
fn projector_maps_features_to_text_hidden_size() {
    let weights = build_projector_weights("mlp1");
    let projector =
        NemotronHNanoOmniProjector::from_weights(&weights, "mlp1", 64, 4).expect("build projector");

    let input = mlxcel_core::ones(
        &[1, 4, PROJ_IN_FEATURES as i32],
        mlxcel_core::dtype::FLOAT32,
    );
    let out = projector.forward(input.as_ref().unwrap());
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![1, 4, TEXT_HIDDEN as i32]
    );
}

/// `subsampling_output_lengths` should mirror upstream
/// `_get_subsampling_output_length` exactly. We exercise the helper here
/// because audio merge correctness depends on precise per-clip length
/// trimming; a one-off bug in the divider would silently skew the
/// number of audio tokens emitted into the LLM stream.
#[test]
fn subsampling_output_lengths_matches_python_reference() {
    let lengths = mlxcel_core::from_slice_i32(&[800, 400, 100, 0], &[4]);
    let kernel = 3;
    let stride = 2;
    let stages = 3; // log2(8) -- the released Nemotron Omni default.
    let out = subsampling_output_lengths(lengths.as_ref().unwrap(), kernel, stride, stages);
    let cpu = read_int32_vec(&out);
    // Python: floor((L - 1) / 2 + 1) per stage.
    // 800 → 400 → 200 → 100
    // 400 → 200 → 100 → 50
    // 100 → 50  → 25  → 13
    //   0 →  0  →  0  →  0  (padding does not produce more frames)
    // For clarity, emit the chain for the third element.
    assert_eq!(cpu, vec![100, 50, 13, 0]);
}

#[test]
fn vl_config_default_downsample_factor_round_trips() {
    let config = NemotronHNanoOmniVlConfig {
        vit_hidden_size: 1280,
        projector_hidden_size: 4096,
        text_hidden_size: 4096,
        downsample_ratio: 0.5,
        ps_version: "v1".to_string(),
        img_context_token_id: 100,
        image_start_token_id: 0,
        image_end_token_id: 0,
        sound_context_token_id: None,
        sound_start_token_id: 0,
        sound_end_token_id: 0,
        eos_token_ids: Vec::new(),
    };
    // Sanity that the config struct exposes everything the runtime wires.
    assert_eq!(config.downsample_ratio, 0.5);
    assert_eq!(config.ps_version, "v1");
    assert_eq!(config.img_context_token_id, 100);
    assert!(config.sound_context_token_id.is_none());
}

// ---------------------------------------------------------------------------
// Per-sequence state forwarding (#1979)
// ---------------------------------------------------------------------------

const TINY_VOCAB: i32 = 16;
const TINY_HIDDEN: i32 = 8;

/// Deterministic, position-varying weights so different KV histories give
/// visibly different logits (all-zero weights would make every slot agree).
fn ramp(shape: &[i32], scale: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * scale)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// One attention layer Nemotron-H: the smallest backbone whose output depends
/// on the per-sequence KV history, so a shared slot shows up in the logits.
fn tiny_attention_text_model() -> crate::models::NemotronHModel {
    use crate::models::nemotron_h::{BlockType, NemotronHConfig};

    let mut config: NemotronHConfig = serde_json::from_value(serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": TINY_VOCAB,
        "hidden_size": TINY_HIDDEN,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "mamba_num_heads": 1,
        "mamba_head_dim": 4,
        "ssm_state_size": 4,
        "conv_kernel": 4,
        "n_groups": 1,
        "hybrid_override_pattern": "*",
    }))
    .expect("tiny config must parse");
    config.post_init().expect("tiny config must pass post_init");

    let mut weights = WeightMap::new();
    weights.insert(
        "backbone.embeddings.weight".to_string(),
        ramp(&[TINY_VOCAB, TINY_HIDDEN], 0.1),
    );
    weights.insert(
        "lm_head.weight".to_string(),
        ramp(&[TINY_VOCAB, TINY_HIDDEN], 0.05),
    );
    weights.insert(
        "backbone.norm_f.weight".to_string(),
        mlxcel_core::ones(&[TINY_HIDDEN], mlxcel_core::dtype::FLOAT32),
    );
    weights.insert(
        "backbone.layers.0.norm.weight".to_string(),
        mlxcel_core::ones(&[TINY_HIDDEN], mlxcel_core::dtype::FLOAT32),
    );
    for (i, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
        weights.insert(
            format!("backbone.layers.0.mixer.{proj}.weight"),
            ramp(&[TINY_HIDDEN, TINY_HIDDEN], 0.03 * (i as f32 + 1.0)),
        );
    }
    crate::models::NemotronHModel::from_weights(config, weights, vec![BlockType::Attention])
        .expect("tiny attention-only Nemotron-H must load")
}

fn tiny_wrapper() -> super::NemotronHNanoOmniVlModel {
    use crate::vision::encoders::nemotron_h_nano_omni::NemotronHNanoOmniVisionModel;
    use crate::vision::encoders::nemotron_h_nano_omni::tests::{
        build_synthetic_weights, small_config,
    };
    use crate::vision::processors::nemotron_h_nano_omni::NemotronHNanoOmniImageProcessor;

    let vision_config = small_config();
    let vision_weights = build_synthetic_weights("vision_model.radio_model", &vision_config);
    let vision_tower = NemotronHNanoOmniVisionModel::from_weights(
        &vision_weights,
        "vision_model.radio_model",
        &vision_config,
        64,
        4,
    )
    .expect("synthetic vision tower must load");
    let projector =
        NemotronHNanoOmniProjector::from_weights(&build_projector_weights("mlp1"), "mlp1", 64, 4)
            .expect("build projector");
    let config = NemotronHNanoOmniVlConfig {
        vit_hidden_size: PROJ_IN_FEATURES,
        projector_hidden_size: PROJ_HIDDEN,
        text_hidden_size: TINY_HIDDEN as usize,
        downsample_ratio: 0.5,
        ps_version: "v1".to_string(),
        img_context_token_id: 3,
        image_start_token_id: 0,
        image_end_token_id: 0,
        sound_context_token_id: None,
        sound_start_token_id: 0,
        sound_end_token_id: 0,
        eos_token_ids: vec![1],
    };
    super::NemotronHNanoOmniVlModel::new(
        tiny_attention_text_model(),
        vision_tower,
        projector,
        NemotronHNanoOmniImageProcessor::new(Default::default()),
        config,
    )
}

fn host_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().expect("4-byte chunk")))
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    let max_diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "{what}: max |diff| {max_diff}");
}

/// Two sequence ids on one wrapper must not share recurrent/KV state. Before
/// #1979 the wrapper left the per-sequence trait methods at their defaults,
/// so every id ran on the text model's single fallback slot: the second
/// sequence's prefill attended over the first sequence's history.
#[test]
fn sequence_ids_keep_isolated_state_on_token_and_embedding_paths() {
    use crate::LanguageModel;
    use mlxcel_core::cache::{SequenceId, SequenceStateLayout};

    let model = tiny_wrapper();
    assert!(!model.supports_batching());
    assert!(!model.supports_padded_prefill());
    assert_eq!(
        model.sequence_state_layout(),
        SequenceStateLayout::model_owned(model.num_layers()),
        "wrapper must report the text model's model-owned layout"
    );
    assert!(model.supports_snapshot_reuse());

    let prompt = mlxcel_core::from_slice_i32(&[2, 5, 7, 11], &[1, 4]);
    let seq_a = SequenceId::from_raw(101);
    let seq_b = SequenceId::from_raw(202);
    model.prepare_sequence_state(seq_a);
    model.prepare_sequence_state(seq_b);
    let mut caches = model.make_caches();

    // Token path: the same prompt on a fresh slot must give the same logits
    // no matter what another sequence prefilled first.
    let logits_a = model.forward_with_sequence_id(&prompt, Some(seq_a), &mut caches, None);
    let logits_b = model.forward_with_sequence_id(&prompt, Some(seq_b), &mut caches, None);
    assert_close(
        &host_f32(&logits_a),
        &host_f32(&logits_b),
        "token prefill on seq B after seq A",
    );

    // Each slot advanced independently: one decode step on each agrees.
    let next = mlxcel_core::from_slice_i32(&[4], &[1, 1]);
    let step_a = model.forward_with_sequence_id(&next, Some(seq_a), &mut caches, None);
    let step_b = model.forward_with_sequence_id(&next, Some(seq_b), &mut caches, None);
    assert_close(&host_f32(&step_a), &host_f32(&step_b), "decode step");

    // Snapshots are taken per sequence id.
    let snapshot = model
        .snapshot_sequence_state(seq_a, 5)
        .expect("seq A must have snapshot-able state");
    assert_eq!(snapshot.token_len(), 5);

    // Embedding path (the server's multimodal prefill): a fresh sequence
    // prefilled with embeddings must not see seq A/B history, and the
    // last-logits variant must agree with the full forward's last row.
    let seq_c = SequenceId::from_raw(303);
    let seq_d = SequenceId::from_raw(404);
    model.prepare_sequence_state(seq_c);
    model.prepare_sequence_state(seq_d);
    let embeds = model
        .embed_tokens(&prompt)
        .expect("wrapper exposes token embeddings");
    let full_c = model.forward_with_embeddings_and_sequence_id(
        &prompt,
        Some(&embeds),
        Some(seq_c),
        &mut caches,
        None,
    );
    let last_d = model.forward_last_logits_with_embeddings_and_sequence_id(
        &prompt,
        Some(&embeds),
        Some(seq_d),
        &mut caches,
        None,
        3,
    );
    let full_c_last = mlxcel_core::generate::logits_at_position(&full_c, 3);
    assert_close(
        &host_f32(&full_c_last),
        &host_f32(&last_d),
        "embeddings prefill on seq D after seq C",
    );
    assert_close(
        &host_f32(&full_c),
        &host_f32(&logits_a),
        "embeddings prefill matches the token prefill of the same prompt",
    );

    for seq in [seq_a, seq_b, seq_c, seq_d] {
        model.release_sequence_state_by_id(seq);
    }
    assert!(
        model.snapshot_sequence_state(seq_a, 5).is_none(),
        "released sequence must drop its state"
    );
}
