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

//! Llama 3.2 Vision wrapper tests: projector loading, real-tile selection,
//! native text-only batching, and per-sequence image state (#2567).

use super::{MllamaVLModel, select_real_tile_states};
use crate::LanguageModel;
use crate::models::embedding_test_support::to_vec;
use crate::models::mllama::MllamaConfig;
use crate::vision::encoders::mllama::MllamaVisionModel;
use crate::vision::processors::mllama::MllamaImageProcessor;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::BatchedDecodeKind;
use mlxcel_core::layers::KVCache;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// The real checkpoint stores `multi_modal_projector` quantized with a float
/// `.bias` alongside the affine `.scales`/`.biases`. The loader must resolve
/// all four and report a quantized projector (issue #527 follow-up).
#[test]
fn projector_loads_quantized_with_bias() {
    let mut w = WeightMap::new();
    // group_size 64 / bits 4: in=64 -> packed_in 8, num_groups 1.
    w.insert(
        "multi_modal_projector.weight".to_string(),
        mlxcel_core::from_slice_u32(&[0u32; 8 * 8], &[8, 8]),
    );
    w.insert(
        "multi_modal_projector.scales".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
    );
    w.insert(
        "multi_modal_projector.biases".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
    );
    w.insert(
        "multi_modal_projector.bias".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 8], &[8]),
    );

    let projector =
        MllamaVLModel::load_projector(&w, 64, 4).expect("quantized projector must load");
    assert!(
        projector.is_quantized(),
        "the 4-bit projector must load quantized, not as a raw packed linear"
    );
}

// --- Real-tile row selection (issue #527 perf follow-up). ---

/// `[1, media, max_tiles, patches, dim]` filled with 0..N so every row is
/// identifiable by value.
fn arange_output(media: i32, max_tiles: i32, patches: i32, dim: i32) -> UniquePtr<MlxArray> {
    let n = (media * max_tiles * patches * dim) as usize;
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    mlxcel_core::from_slice_f32(&vals, &[1, media, max_tiles, patches, dim])
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    let diff = mlxcel_core::subtract(a, b);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&diff));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn filled(shape: &[i32], seed: usize) -> UniquePtr<MlxArray> {
    let count: i32 = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|index| ((index as usize * 131 + seed * 977 + 7) % 251) as f32 / 251.0 - 0.5)
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn put(weights: &mut WeightMap, key: &str, shape: &[i32], seed: usize) {
    weights.insert(key.to_string(), filled(shape, seed));
}

fn tiny_wrapper_config() -> MllamaConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "mllama",
        "image_token_index": 31,
        "vision_feature_layer": -1,
        "text_config": {
            "model_type": "mllama",
            "vocab_size": 32,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "cross_attention_layers": [1]
        },
        "vision_config": {
            "image_size": 2,
            "patch_size": 2,
            "num_channels": 3,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 0,
            "num_attention_heads": 1,
            "max_num_tiles": 1,
            "max_aspect_ratio_id": 1,
            "num_global_layers": 0,
            "norm_eps": 1e-5,
            "vision_output_dim": 4,
            "intermediate_layers_indices": [],
            "supported_aspect_ratios": [[1, 1]]
        }
    }))
    .expect("tiny Mllama wrapper config parses")
}

fn unused_vision_tower(config: &MllamaConfig) -> MllamaVisionModel {
    let mut weights = WeightMap::new();
    put(&mut weights, "vision.class_embedding", &[4], 1);
    put(
        &mut weights,
        "vision.patch_embedding.weight",
        &[4, 3, 2, 2],
        2,
    );
    put(
        &mut weights,
        "vision.gated_positional_embedding.embedding",
        &[2, 4],
        3,
    );
    put(
        &mut weights,
        "vision.gated_positional_embedding.gate",
        &[1],
        4,
    );
    put(
        &mut weights,
        "vision.gated_positional_embedding.tile_embedding.weight",
        &[2, 8],
        5,
    );
    for (name, seed) in [
        ("pre_tile_positional_embedding", 7),
        ("post_tile_positional_embedding", 11),
    ] {
        put(
            &mut weights,
            &format!("vision.{name}.embedding.weight"),
            &[2, 4],
            seed,
        );
        put(&mut weights, &format!("vision.{name}.gate"), &[1], seed + 1);
    }
    for (name, seed) in [("layernorm_pre", 13), ("layernorm_post", 17)] {
        put(&mut weights, &format!("vision.{name}.weight"), &[4], seed);
        put(&mut weights, &format!("vision.{name}.bias"), &[4], seed + 1);
    }
    MllamaVisionModel::from_weights(&weights, &config.vision_config, "vision")
        .expect("tiny zero-depth Mllama vision tower builds")
}

pub(crate) fn tiny_wrapper() -> MllamaVLModel {
    let config = tiny_wrapper_config();
    let vision_tower = unused_vision_tower(&config);
    let mut weights = WeightMap::new();
    put(&mut weights, "multi_modal_projector.weight", &[4, 4], 19);
    put(&mut weights, "multi_modal_projector.bias", &[4], 23);
    let projector =
        MllamaVLModel::load_projector(&weights, 0, 0).expect("tiny Mllama projector builds");
    MllamaVLModel::from_parts(
        crate::models::mllama::text::tests::tiny_batched_model(),
        vision_tower,
        projector,
        MllamaImageProcessor::new(2, 1),
        config,
        vec![2],
    )
}

fn prefill(model: &MllamaVLModel, prompt: &[i32]) -> Vec<KVCache> {
    let mut caches = model.make_caches();
    let input = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = model.forward(&input, &mut caches, None);
    mlxcel_core::eval(&logits);
    caches
}

fn independent_decode(
    model: &MllamaVLModel,
    prompts: &[&[i32]],
    next: &[i32],
) -> Vec<UniquePtr<MlxArray>> {
    prompts
        .iter()
        .zip(next)
        .map(|(prompt, &token)| {
            let mut caches = prefill(model, prompt);
            let input = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
            let logits = model.forward(&input, &mut caches, None);
            mlxcel_core::eval(&logits);
            logits
        })
        .collect()
}

fn assert_three_row_decode_matches_independent(model: &MllamaVLModel) {
    let prompts: [&[i32]; 3] = [&[1, 2], &[3, 4, 5, 6], &[7, 8, 9, 10, 11, 12]];
    let next = [13, 14, 15];
    let expected = independent_decode(model, &prompts, &next);
    let mut caches: Vec<Vec<KVCache>> = prompts
        .iter()
        .map(|prompt| prefill(model, prompt))
        .collect();
    let before: Vec<Vec<i32>> = caches
        .iter()
        .map(|layers| layers.iter().map(|cache| cache.offset).collect())
        .collect();
    let input = mlxcel_core::from_slice_i32(&next, &[3, 1]);
    let actual = {
        let mut rows: Vec<&mut [KVCache]> = caches.iter_mut().map(Vec::as_mut_slice).collect();
        model.forward_batched(&input, &mut rows, None)
    };
    mlxcel_core::eval(&actual);

    for (row, expected) in expected.iter().enumerate() {
        let actual_row = mlxcel_core::slice(&actual, &[row as i32, 0, 0], &[row as i32 + 1, 1, 32]);
        assert!(
            mlxcel_core::item_bool(&mlxcel_core::allclose(&actual_row, expected, 1e-4, 1e-4)),
            "Mllama wrapper row {row} differs from its independent forward"
        );
        for layer in [0, 2] {
            assert_eq!(caches[row][layer].offset, before[row][layer] + 1);
        }
        assert_eq!(caches[row][1].offset, before[row][1]);
    }
}

fn assert_single_row_exact(model: &MllamaVLModel) {
    let mut expected_caches = prefill(model, &[1, 3, 5]);
    let mut actual_caches = prefill(model, &[1, 3, 5]);
    let input = mlxcel_core::from_slice_i32(&[7], &[1, 1]);
    let expected = model.forward(&input, &mut expected_caches, None);
    let actual = {
        let mut rows = vec![actual_caches.as_mut_slice()];
        model.forward_batched(&input, &mut rows, None)
    };
    mlxcel_core::eval(&expected);
    mlxcel_core::eval(&actual);
    assert_eq!(max_abs_diff(&actual, &expected), 0.0);
}

#[test]
fn text_only_wrapper_uses_native_batch_and_preserves_b1_b0_contracts() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_wrapper();
    assert_eq!(model.batched_decode_kind(), BatchedDecodeKind::Native);
    assert!(!model.has_cross_attention_states());
    assert_three_row_decode_matches_independent(&model);
    assert_single_row_exact(&model);

    let input = mlxcel_core::from_slice_i32(&[], &[0, 1]);
    let mut rows: Vec<&mut [KVCache]> = Vec::new();
    let logits = model.forward_batched(&input, &mut rows, None);
    assert_eq!(mlxcel_core::array_shape(&logits), vec![0, 1, 1]);
}

#[test]
fn image_state_uses_nontrivial_cross_attention_and_historical_per_row_fallback() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_wrapper();
    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let mut text_caches = model.make_caches();
    let text_logits = model.forward(&prompt, &mut text_caches, None);

    let cross_states = filled(&[1, 3, 4], 701);
    model.set_cross_attention_states(cross_states);
    let mut image_caches = model.make_caches();
    let image_logits = model.forward(&prompt, &mut image_caches, None);
    mlxcel_core::eval(&text_logits);
    mlxcel_core::eval(&image_logits);
    assert!(
        max_abs_diff(&image_logits, &text_logits) > 1e-5,
        "nonzero cross-attention gates and image features must affect logits"
    );

    assert!(model.has_cross_attention_states());
    assert_three_row_decode_matches_independent(&model);
    assert_single_row_exact(&model);
}

/// Ragged multi-image selection keeps exactly the real-tile rows of every
/// image, media-major, in the reference's flatten order.
#[test]
fn select_keeps_ragged_real_tile_rows_media_major() {
    // 2 media, 2 tiles, 2 patches, dim 3; media 0 has 1 real tile, media 1
    // has 2. Row layout (patch rows of dim 3):
    //   media 0: tile 0 -> values 0..6,   tile 1 -> 6..12 (padding)
    //   media 1: tile 0 -> values 12..18, tile 1 -> 18..24
    let output = arange_output(2, 2, 2, 3);
    let selected =
        select_real_tile_states(&output, &[1, 2]).expect("sub-max ragged counts must select");
    assert_eq!(mlxcel_core::array_shape(&selected), vec![1, 6, 3]);

    let expected: Vec<f32> = (0..6)
        .map(|i| i as f32)
        .chain((12..24).map(|i| i as f32))
        .collect();
    let expected = mlxcel_core::from_slice_f32(&expected, &[1, 6, 3]);
    assert_eq!(
        max_abs_diff(&selected, &expected),
        0.0,
        "selection must keep media 0 tile 0 and media 1 tiles 0..2, in order"
    );
}

/// Full tile counts are a no-op: the caller must keep the legacy all-tiles
/// path so the states stay byte-identical to the pre-optimization output.
#[test]
fn select_declines_full_tile_counts() {
    let output = arange_output(2, 2, 2, 3);
    assert!(select_real_tile_states(&output, &[2, 2]).is_none());
}

/// Defensive fallbacks: count/media mismatch, zero counts, and
/// out-of-range counts all keep the legacy path instead of guessing.
#[test]
fn select_declines_invalid_tile_counts() {
    let output = arange_output(2, 2, 2, 3);
    assert!(select_real_tile_states(&output, &[1]).is_none());
    assert!(select_real_tile_states(&output, &[1, 2, 1]).is_none());
    assert!(select_real_tile_states(&output, &[0, 2]).is_none());
    assert!(select_real_tile_states(&output, &[1, 3]).is_none());
    assert!(select_real_tile_states(&output, &[]).is_none());
}

// --- Per-sequence image state (#2567). ---

const SEQ_PROMPT_A: [i32; 3] = [1, 2, 3];
const SEQ_PROMPT_B: [i32; 5] = [4, 5, 6, 7, 8];
const SEQ_NEXT_A: [i32; 4] = [9, 10, 11, 12];
const SEQ_NEXT_B: [i32; 4] = [13, 14, 15, 16];

pub(crate) fn image_states(seed: usize) -> UniquePtr<MlxArray> {
    filled(&[1, 3, 4], seed)
}

fn forward_id(
    model: &MllamaVLModel,
    tokens: &[i32],
    seq_id: SequenceId,
    caches: &mut [KVCache],
) -> Vec<f32> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    to_vec(&model.forward_with_sequence_id(&input, Some(seq_id), caches, None))
}

/// Prefill `prompt` and decode every token of `next` under `seq_id`, one call
/// per step. Returns the prefill logits followed by one row per decode step.
fn run_sequence(
    model: &MllamaVLModel,
    prompt: &[i32],
    next: &[i32],
    seq_id: SequenceId,
) -> Vec<Vec<f32>> {
    let mut caches = model.make_caches();
    let mut out = vec![forward_id(model, prompt, seq_id, &mut caches)];
    for &token in next {
        out.push(forward_id(model, &[token], seq_id, &mut caches));
    }
    out
}

/// The sequence served alone with image `seed`: prepare, bind, run, release.
fn run_alone_with_image(
    model: &MllamaVLModel,
    seed: usize,
    prompt: &[i32],
    next: &[i32],
    seq_id: SequenceId,
) -> Vec<Vec<f32>> {
    model.set_cross_attention_states(image_states(seed));
    model.bind_cross_attention_states_to_sequence(seq_id);
    let out = run_sequence(model, prompt, next, seq_id);
    model.release_sequence_state_by_id(seq_id);
    out
}

fn assert_steps_match(actual: &[Vec<f32>], expected: &[Vec<f32>], what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (step, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = crate::models::embedding_test_support::max_abs_diff(a, e);
        assert!(
            diff <= 1e-5,
            "{what}: step {step} differs from the sequence run alone (max abs diff {diff:e})"
        );
    }
}

/// Request 1 carries an image bound to sequence 1; request 2 is text-only on
/// sequence 2 and must not attend to request 1's image.
#[test]
fn text_only_sequence_after_image_sequence_stays_text_only() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_wrapper();
    let seq_image = SequenceId::from_raw(256711);
    let seq_text = SequenceId::from_raw(256712);

    // Reference first, on a model that has never seen an image.
    let text_alone = run_sequence(&model, &SEQ_PROMPT_B, &SEQ_NEXT_B[..2], seq_text);

    let with_image = run_alone_with_image(&model, 701, &SEQ_PROMPT_B, &SEQ_NEXT_B[..2], seq_image);
    assert!(
        crate::models::embedding_test_support::max_abs_diff(&with_image[0], &text_alone[0]) > 1e-5,
        "the image must change the logits for this gate to mean anything"
    );

    let text_after_image = run_sequence(&model, &SEQ_PROMPT_B, &SEQ_NEXT_B[..2], seq_text);
    assert_steps_match(
        &text_after_image,
        &text_alone,
        "text-only sequence 2 after image sequence 1",
    );
}

/// Images X and Y are prepared and bound to sequences 1 and 2 before either
/// prefills. Each sequence's prefill and decode steps, interleaved and as one
/// two-row batch, must equal that sequence served alone with its own image.
#[test]
fn image_states_bound_before_prefill_stay_with_their_sequences() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_wrapper();
    let alone_x = run_alone_with_image(
        &model,
        701,
        &SEQ_PROMPT_A,
        &SEQ_NEXT_A,
        SequenceId::from_raw(256721),
    );
    let alone_y = run_alone_with_image(
        &model,
        907,
        &SEQ_PROMPT_B,
        &SEQ_NEXT_B,
        SequenceId::from_raw(256722),
    );

    let seq_x = SequenceId::from_raw(256731);
    let seq_y = SequenceId::from_raw(256732);
    model.set_cross_attention_states(image_states(701));
    model.bind_cross_attention_states_to_sequence(seq_x);
    model.set_cross_attention_states(image_states(907));
    model.bind_cross_attention_states_to_sequence(seq_y);

    let mut caches_x = model.make_caches();
    let mut caches_y = model.make_caches();
    let mut got_x = vec![forward_id(&model, &SEQ_PROMPT_A, seq_x, &mut caches_x)];
    let mut got_y = vec![forward_id(&model, &SEQ_PROMPT_B, seq_y, &mut caches_y)];
    for step in 0..3 {
        got_x.push(forward_id(
            &model,
            &[SEQ_NEXT_A[step]],
            seq_x,
            &mut caches_x,
        ));
        got_y.push(forward_id(
            &model,
            &[SEQ_NEXT_B[step]],
            seq_y,
            &mut caches_y,
        ));
    }
    let batched = {
        let input = mlxcel_core::from_slice_i32(&[SEQ_NEXT_A[3], SEQ_NEXT_B[3]], &[2, 1]);
        let mut rows: Vec<&mut [KVCache]> = vec![caches_x.as_mut_slice(), caches_y.as_mut_slice()];
        model.forward_batched_with_ids(&input, Some(&[seq_x, seq_y]), &mut rows, None)
    };
    mlxcel_core::eval(&batched);
    got_x.push(to_vec(&mlxcel_core::slice(
        &batched,
        &[0, 0, 0],
        &[1, 1, 32],
    )));
    got_y.push(to_vec(&mlxcel_core::slice(
        &batched,
        &[1, 0, 0],
        &[2, 1, 32],
    )));

    assert_steps_match(&got_x, &alone_x, "sequence 1 with image X");
    assert_steps_match(&got_y, &alone_y, "sequence 2 with image Y");

    model.release_sequence_state_by_id(seq_x);
    model.release_sequence_state_by_id(seq_y);
}

/// A decode batch mixing one image row with two text-only rows: the text-only
/// rows batch natively and every row equals its own solo run.
#[test]
fn mixed_image_and_text_rows_batch_per_sequence() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_wrapper();
    let prompts: [&[i32]; 3] = [&SEQ_PROMPT_A, &SEQ_PROMPT_B, &[17, 18, 19, 20]];
    let next = [21, 22, 23];
    let seqs = [
        SequenceId::from_raw(256741),
        SequenceId::from_raw(256742),
        SequenceId::from_raw(256743),
    ];

    // Text-only references first, before any image state exists.
    let text_alone: Vec<Vec<Vec<f32>>> = (1..3)
        .map(|row| run_sequence(&model, prompts[row], &next[row..=row], seqs[row]))
        .collect();
    let image_alone = run_alone_with_image(
        &model,
        701,
        prompts[0],
        &next[..1],
        SequenceId::from_raw(256744),
    );

    model.set_cross_attention_states(image_states(701));
    model.bind_cross_attention_states_to_sequence(seqs[0]);
    let mut caches: Vec<Vec<KVCache>> = (0..3).map(|_| model.make_caches()).collect();
    let prefill: Vec<Vec<f32>> = (0..3)
        .map(|row| forward_id(&model, prompts[row], seqs[row], &mut caches[row]))
        .collect();
    let batched = {
        let input = mlxcel_core::from_slice_i32(&next, &[3, 1]);
        let mut rows: Vec<&mut [KVCache]> = caches.iter_mut().map(Vec::as_mut_slice).collect();
        model.forward_batched_with_ids(&input, Some(&seqs), &mut rows, None)
    };
    mlxcel_core::eval(&batched);
    let decode: Vec<Vec<f32>> = (0..3i32)
        .map(|row| {
            to_vec(&mlxcel_core::slice(
                &batched,
                &[row, 0, 0],
                &[row + 1, 1, 32],
            ))
        })
        .collect();

    assert_steps_match(
        &[prefill[0].clone(), decode[0].clone()],
        &image_alone,
        "image row in a mixed batch",
    );
    for row in 1..3 {
        assert_steps_match(
            &[prefill[row].clone(), decode[row].clone()],
            &text_alone[row - 1],
            "text-only row in a mixed batch",
        );
    }
    for seq in seqs {
        model.release_sequence_state_by_id(seq);
    }
}

/// Preemption round trip: the entry taken under the old id and installed
/// under the new id still drives that sequence, and release removes it.
#[test]
fn image_state_survives_preemption_and_release_drops_it() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_wrapper();
    let text_alone = run_sequence(
        &model,
        &SEQ_PROMPT_A,
        &SEQ_NEXT_A[..1],
        SequenceId::from_raw(256750),
    );
    let alone_x = run_alone_with_image(
        &model,
        701,
        &SEQ_PROMPT_A,
        &SEQ_NEXT_A[..1],
        SequenceId::from_raw(256751),
    );

    let old_id = SequenceId::from_raw(256752);
    let new_id = SequenceId::from_raw(256753);
    model.set_cross_attention_states(image_states(701));
    model.bind_cross_attention_states_to_sequence(old_id);
    assert!(model.has_image_state_for_sequence(old_id));
    assert!(
        !model.has_cross_attention_states(),
        "binding must drain the fallback slot"
    );

    let taken = model
        .take_image_state_for_sequence(old_id)
        .expect("bound entry can be taken");
    assert!(!model.has_image_state_for_sequence(old_id));
    assert!(model.take_image_state_for_sequence(old_id).is_none());
    model.release_sequence_state_by_id(old_id);

    model.install_image_state_for_sequence(new_id, taken);
    assert!(model.has_image_state_for_sequence(new_id));
    let after = run_sequence(&model, &SEQ_PROMPT_A, &SEQ_NEXT_A[..1], new_id);
    assert_steps_match(&after, &alone_x, "sequence re-prefilled under its new id");

    model.release_sequence_state_by_id(new_id);
    assert!(!model.has_image_state_for_sequence(new_id));
    let released = run_sequence(&model, &SEQ_PROMPT_A, &SEQ_NEXT_A[..1], new_id);
    assert_steps_match(&released, &text_alone, "released sequence runs text-only");
}

/// The raw-completion client (`mlxcel generate --image`) prepares the image
/// into the fallback slot and then drives the forward through its own
/// `SequenceId` without a scheduler bind. That sequence must attend to the
/// fallback image, as every forward did before #2567.
#[test]
fn unbound_sequence_reads_the_fallback_image_like_the_cli() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_wrapper();
    let text_alone = run_sequence(
        &model,
        &SEQ_PROMPT_A,
        &SEQ_NEXT_A[..2],
        SequenceId::from_raw(256760),
    );
    let alone_x = run_alone_with_image(
        &model,
        701,
        &SEQ_PROMPT_A,
        &SEQ_NEXT_A[..2],
        SequenceId::from_raw(256761),
    );
    assert!(
        crate::models::embedding_test_support::max_abs_diff(&alone_x[0], &text_alone[0]) > 1e-5,
        "the image must change the logits for this gate to mean anything"
    );

    let cli_seq = SequenceId::from_raw(256762);
    model.set_cross_attention_states(image_states(701));
    assert!(model.has_image_state_for_sequence(cli_seq));
    let cli = run_sequence(&model, &SEQ_PROMPT_A, &SEQ_NEXT_A[..2], cli_seq);
    assert_steps_match(&cli, &alone_x, "unbound CLI sequence with a fallback image");
    model.clear_cross_attention_states();
}

/// A text-only request bound at admission must keep running text-only even if
/// the fallback slot later holds an image that no admission bound (for
/// example a preparation whose admission then failed).
#[test]
fn bound_text_only_sequence_ignores_a_later_fallback_image() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_wrapper();
    let text_alone = run_sequence(
        &model,
        &SEQ_PROMPT_B,
        &SEQ_NEXT_B[..2],
        SequenceId::from_raw(256770),
    );

    let seq_text = SequenceId::from_raw(256771);
    model.bind_cross_attention_states_to_sequence(seq_text);
    let mut caches = model.make_caches();
    let mut got = vec![forward_id(&model, &SEQ_PROMPT_B, seq_text, &mut caches)];
    model.set_cross_attention_states(image_states(701));
    assert!(!model.has_image_state_for_sequence(seq_text));
    for &token in &SEQ_NEXT_B[..2] {
        got.push(forward_id(&model, &[token], seq_text, &mut caches));
    }
    assert_steps_match(&got, &text_alone, "bound text-only sequence");

    // A preemption round trip keeps it text-only under the new id.
    let new_id = SequenceId::from_raw(256772);
    let snapshot = model.take_image_state_for_sequence(seq_text);
    assert!(snapshot.is_none());
    model.release_sequence_state_by_id(seq_text);
    model.restore_image_state_for_sequence(new_id, snapshot);
    assert!(!model.has_image_state_for_sequence(new_id));
    model.release_sequence_state_by_id(new_id);
    model.clear_cross_attention_states();
}
