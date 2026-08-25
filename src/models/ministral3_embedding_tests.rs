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

//! Nemotron-3-Embed gates.
//!
//! Three properties carry this port. The Llama 4 attention scale is a
//! per-position multiplier the generator computes from its cache offset; the
//! embedder computes it at offset 0, and the two have to agree for the same
//! sequence length or every score is off by a factor that grows with position.
//! The prefill has to be bidirectional, which is what `is_causal: false` in the
//! config declares. And a right-padded batch has to reproduce the unpadded
//! single row.
//!
//! The last two tests load `nvidia/Nemotron-3-Embed-1B-BF16` and the
//! `mlx-community` 8-bit conversion, and soft-skip when they are not
//! downloaded.
//!
//! Every test that evaluates an MLX graph takes the shared `mlx_test_guard`;
//! see its doc comment for the two ways parallel MLX work breaks this suite.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;

use super::Ministral3EmbeddingModel;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel};
use crate::embeddings::pooling::PoolingMode;
use crate::models::embedding_test_support::{
    Rng, cosine, err_string, ids_array, local_checkpoint, max_abs_diff, mlx_test_guard, to_vec,
};
use crate::models::ministral3::{ModelArgs, RopeParameters, get_llama4_attn_scale};

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 32;

const NEMOTRON_EMBED_BF16: &str = "nvidia/Nemotron-3-Embed-1B-BF16";
const NEMOTRON_EMBED_8BIT: &str = "mlx-community/Nemotron-3-Embed-1B-BF16-8bit";
/// `hidden_size` of both published checkpoints.
const CHECKPOINT_DIM: usize = 2048;
/// `llama_4_scaling_beta` and `original_max_position_embeddings` of the
/// published `rope_parameters` block.
const CHECKPOINT_BETA: f32 = 0.1;
const CHECKPOINT_ORIGINAL_MAX: usize = 16384;

/// Cosine deviation allowed between the same text embedded alone and embedded
/// as a row of a longer, right-padded batch.
///
/// The issue asks for agreement within 1e-3, and the pooled vectors do agree to
/// that in cosine. They do not agree to 1e-3 in the largest single component,
/// and cannot on this hardware: bf16 attention picks its accumulation shape
/// from the batch geometry, so the vector moves when the batch does even with
/// no padding involved. Embedding one text as 2, 3, 4, 5 and 8 unpadded copies
/// of a single batch moved the largest component of this checkpoint by up to
/// 3.7e-3 while cosine stayed above 0.99997, and the already-merged
/// `Qwen/Qwen3-Embedding-0.6B` behaves the same way, so the component form of
/// that bound would gate the MLX CUDA backend rather than this port.
///
/// What the port owes is that the PADDING changes nothing, and that is gated at
/// zero tolerance by `padding_content_cannot_reach_a_real_position`: same batch
/// shape, same slot, same mask, different token ids underneath the mask.
const BATCH_COSINE_TOLERANCE: f32 = 1e-3;

fn tiny_args(num_layers: usize) -> ModelArgs {
    ModelArgs {
        model_type: "ministral3".to_string(),
        hidden_size: HIDDEN,
        num_hidden_layers: num_layers,
        intermediate_size: INTERMEDIATE,
        num_attention_heads: 2,
        rms_norm_eps: 1e-6,
        vocab_size: VOCAB,
        head_dim: Some(HEAD_DIM),
        max_position_embeddings: Some(128),
        num_key_value_heads: Some(1),
        rope_parameters: Some(RopeParameters {
            rope_theta: 1_000_000.0,
            llama_4_scaling_beta: CHECKPOINT_BETA,
            original_max_position_embeddings: 64,
        }),
        tie_word_embeddings: true,
        layer_types: None,
        sliding_window: None,
        quantization: None,
    }
}

/// Deterministic dense weights for `args`, in the sanitized key layout.
fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let mut rng = Rng::new(0x00C0_FFEE_1325);
    let mut w = WeightMap::new();
    let h = args.hidden_size as i32;
    let hd = args.head_dim() as i32;
    let q_out = args.num_attention_heads as i32 * hd;
    let kv_out = args.num_kv_heads() as i32 * hd;
    let inter = args.intermediate_size as i32;

    rng.insert(
        &mut w,
        "model.embed_tokens.weight",
        &[args.vocab_size as i32, h],
        0.5,
    );
    for i in 0..args.num_hidden_layers {
        let p = format!("model.layers.{i}");
        for (name, shape) in [
            ("self_attn.q_proj", [q_out, h]),
            ("self_attn.k_proj", [kv_out, h]),
            ("self_attn.v_proj", [kv_out, h]),
            ("self_attn.o_proj", [h, q_out]),
            ("mlp.gate_proj", [inter, h]),
            ("mlp.up_proj", [inter, h]),
            ("mlp.down_proj", [h, inter]),
        ] {
            rng.insert(&mut w, &format!("{p}.{name}.weight"), &shape, 0.2);
        }
        rng.insert(&mut w, &format!("{p}.input_layernorm.weight"), &[h], 0.1);
        rng.insert(
            &mut w,
            &format!("{p}.post_attention_layernorm.weight"),
            &[h],
            0.1,
        );
    }
    rng.insert(&mut w, "model.norm.weight", &[h], 0.1);
    w
}

fn build(args: &ModelArgs) -> Result<Ministral3EmbeddingModel> {
    Ministral3EmbeddingModel::from_weights(&tiny_weights(args), args, PoolingMode::Mean, true)
}

fn ramp(len: usize) -> Vec<i32> {
    (0..len).map(|i| (i % VOCAB) as i32).collect()
}

/// `[L, HIDDEN]` hidden states of one unpadded row.
fn hidden_states(model: &Ministral3EmbeddingModel, ids: &[i32]) -> Vec<f32> {
    let length = ids.len() as i32;
    let input = ids_array(ids, 1, length);
    let mask = ids_array(&vec![1; ids.len()], 1, length);
    to_vec(&model.forward_hidden(&input, &mask))
}

/// The pooled `[HIDDEN]` vector of one unpadded row.
fn embed_one(model: &Ministral3EmbeddingModel, ids: &[i32]) -> Vec<f32> {
    let length = ids.len() as i32;
    let input = ids_array(ids, 1, length);
    let mask = ids_array(&vec![1; ids.len()], 1, length);
    to_vec(
        &model
            .embed(&EmbeddingBatch {
                input_ids: &input,
                attention_mask: &mask,
                token_type_ids: None,
                images: None,
            })
            .unwrap()
            .embeddings,
    )
}

// Attention scale.

#[test]
fn attn_scale_at_offset_zero_matches_backbone() {
    // The generator anchors the scale on its full-attention cache offset,
    // which is 0 for a fresh prefill; the embedder always passes 0 because it
    // only ever prefills. For the same length the two must be the same vector,
    // element for element, or the embedder's scores drift from the
    // generator's on exactly the long inputs the scaling exists for.
    // Independent restatement of the documented schedule
    // `scale = 1 + beta * ln(1 + floor(pos / max_pos))`, so this compares the
    // shipped function against the formula rather than against itself.
    let reference = |length: i32, beta: f32, max_pos: usize| -> Vec<f32> {
        (0..length)
            .map(|pos| 1.0 + beta * (1.0 + (pos as f32 / max_pos as f32).floor()).ln())
            .collect()
    };

    for length in [1_i32, 7, 64, 300] {
        let scales = get_llama4_attn_scale(length, 0, CHECKPOINT_BETA, CHECKPOINT_ORIGINAL_MAX);
        assert_eq!(scales.len(), length as usize);
        assert_eq!(
            scales,
            reference(length, CHECKPOINT_BETA, CHECKPOINT_ORIGINAL_MAX),
            "length {length}"
        );
        // Below `original_max_position_embeddings` the schedule is exactly 1.0,
        // which is what makes the published 8192-capped inputs unscaled.
        assert!(
            scales.iter().all(|&s| (s - 1.0).abs() < 1e-6),
            "positions under original_max_position_embeddings must not be scaled"
        );
    }

    // The embedder passes offset 0 because it only ever prefills. That is the
    // same value the generator's fresh full-attention cache reports, so both
    // paths index position `i` as token `i`; a non-zero offset would shift the
    // whole schedule.
    let params = RopeParameters {
        rope_theta: 1_000_000.0,
        llama_4_scaling_beta: CHECKPOINT_BETA,
        original_max_position_embeddings: 64,
    };
    let at_zero = get_llama4_attn_scale(
        200,
        0,
        params.llama_4_scaling_beta,
        params.original_max_position_embeddings,
    );
    let shifted = get_llama4_attn_scale(
        200,
        64,
        params.llama_4_scaling_beta,
        params.original_max_position_embeddings,
    );
    assert_ne!(
        at_zero, shifted,
        "the offset has to matter, or this gate proves nothing"
    );

    // Past the original window the scale grows, and it does so per position:
    // a constant would be indistinguishable from no scaling at all.
    let small = get_llama4_attn_scale(200, 0, CHECKPOINT_BETA, 64);
    assert!((small[0] - 1.0).abs() < 1e-6);
    assert!(
        small[64] > small[63],
        "the scale must step up at the window"
    );
    assert!(small[199] > small[64], "the scale must keep growing");
}

// Forward pass.

#[test]
fn bidirectional_prefill_sees_future_tokens() {
    let _guard = mlx_test_guard();
    // `is_causal: false` is the whole difference from the Ministral 3
    // generator: changing the last token must move position 0.
    let args = tiny_args(2);
    let model = build(&args).unwrap();

    let mut ids = ramp(96);
    let baseline = hidden_states(&model, &ids);
    ids[95] = (ids[95] + 7) % VOCAB as i32;
    let perturbed = hidden_states(&model, &ids);

    let first = max_abs_diff(&baseline[..HIDDEN], &perturbed[..HIDDEN]);
    assert!(
        first > 1e-4,
        "position 0 did not react to the last token ({first}); the mask is still causal"
    );
}

#[test]
fn padding_invariance() {
    let _guard = mlx_test_guard();
    let args = tiny_args(2);
    let model = build(&args).unwrap();

    let short = ramp(8);
    let long: Vec<i32> = (0..12).map(|i| ((i * 5 + 3) % VOCAB) as i32).collect();
    let short_alone = embed_one(&model, &short);
    let long_alone = embed_one(&model, &long);

    let mut padded_ids = short.clone();
    padded_ids.resize(12, 11);
    padded_ids.extend_from_slice(&long);
    let mut padded_mask = vec![1; short.len()];
    padded_mask.resize(12, 0);
    padded_mask.extend(std::iter::repeat_n(1, 12));

    let batch_ids = ids_array(&padded_ids, 2, 12);
    let batch_mask = ids_array(&padded_mask, 2, 12);
    let batch = to_vec(
        &model
            .embed(&EmbeddingBatch {
                input_ids: &batch_ids,
                attention_mask: &batch_mask,
                token_type_ids: None,
                images: None,
            })
            .unwrap()
            .embeddings,
    );

    assert_eq!(batch.len(), 2 * HIDDEN);
    let padded_drift = max_abs_diff(&short_alone, &batch[..HIDDEN]);
    assert!(
        padded_drift < 1e-4,
        "the padded row drifted by {padded_drift}"
    );
    let unpadded_drift = max_abs_diff(&long_alone, &batch[HIDDEN..]);
    assert!(
        unpadded_drift < 1e-4,
        "the full-width row drifted by {unpadded_drift}"
    );
}

#[test]
fn images_are_rejected() {
    let _guard = mlx_test_guard();
    let args = tiny_args(1);
    let model = build(&args).unwrap();
    assert_eq!(model.default_pooling(), PoolingMode::Mean);
    assert_eq!(model.embedding_dim(), HIDDEN);

    let ids = ids_array(&[1, 2, 3], 1, 3);
    let mask = ids_array(&[1, 1, 1], 1, 3);
    let images = [crate::embeddings::model::ImageInput {
        image: image::DynamicImage::new_rgb8(2, 2),
    }];
    let err = err_string(model.embed(&EmbeddingBatch {
        input_ids: &ids,
        attention_mask: &mask,
        token_type_ids: None,
        images: Some(&images),
    }));
    assert!(err.contains("text-only"), "{err}");
}

// Real checkpoints.

/// The four texts the issue's gate uses. Index 3 repeats index 0 verbatim so a
/// right-padded batch can be checked for cosine 1.0 on duplicate rows.
fn gate_texts() -> Vec<String> {
    [
        "query: how do solar panels generate electricity",
        "passage: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect.",
        "passage: The recipe calls for two cups of flour and one egg.",
        "query: how do solar panels generate electricity",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Run the shared self-consistency gate on one checkpoint, returning the four
/// unit vectors so a caller can compare two conversions.
fn run_gate(repo: &str) -> Option<Vec<Vec<f32>>> {
    let dir = local_checkpoint(repo)?;
    let _runtime = crate::initialize_runtime();
    let loaded = crate::embeddings::load_embedding_model(&dir).expect("Nemotron-3-Embed loads");
    assert_eq!(
        loaded.model_type,
        crate::models::ModelType::Ministral3Embedding,
        "{repo}"
    );
    assert_eq!(loaded.limits.dim, CHECKPOINT_DIM, "{repo}");
    // sentence_bert_config.json says 32768, tokenizer_config.json 32768; the
    // hard cap wins.
    assert_eq!(loaded.limits.max_length, 8192, "{repo}");

    let engine = crate::embeddings::EmbeddingEngine::new(loaded, 16);
    let texts = gate_texts();
    let reply = engine
        .embed_texts(&texts, &crate::embeddings::EmbedOptions::default())
        .expect("Nemotron-3-Embed embeds");

    assert_eq!(reply.vectors.len(), 4);
    for vector in &reply.vectors {
        assert_eq!(vector.shape, vec![CHECKPOINT_DIM], "{repo}");
        assert!(
            vector.values.iter().all(|v| v.is_finite()),
            "{repo}: a non-finite component reached the caller"
        );
        let norm: f32 = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "{repo}: unit vector, got {norm}");
    }

    let duplicate = cosine(&reply.vectors[0].values, &reply.vectors[3].values);
    assert!(
        (duplicate - 1.0).abs() < 1e-6,
        "{repo}: identical inputs scored {duplicate}, not 1.0"
    );
    let related = cosine(&reply.vectors[0].values, &reply.vectors[1].values);
    let unrelated = cosine(&reply.vectors[0].values, &reply.vectors[2].values);
    assert!(
        related - unrelated >= 0.15,
        "{repo}: the solar passage must beat the recipe passage by 0.15: {related} vs {unrelated}"
    );
    assert!(
        unrelated < 0.5,
        "{repo}: unrelated sentences scored {unrelated}, which is not below 0.5"
    );

    // Batch invariance. See BATCH_COSINE_TOLERANCE for why this is a cosine
    // bound and where the exact padding gate lives.
    let solo = engine
        .embed_texts(
            &[texts[2].clone()],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("Nemotron-3-Embed embeds a single row");
    let drift = max_abs_diff(&solo.vectors[0].values, &reply.vectors[2].values);
    let batch_cosine = cosine(&solo.vectors[0].values, &reply.vectors[2].values);
    assert!(
        batch_cosine >= 1.0 - BATCH_COSINE_TOLERANCE,
        "{repo}: the passage drifted to cosine {batch_cosine} between the solo and the padded run"
    );

    // Two separate calls on the same input must be bit identical: the forward
    // pass carries no state between calls.
    let again = engine
        .embed_texts(
            &[texts[2].clone()],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("Nemotron-3-Embed embeds a single row twice");
    assert_eq!(
        max_abs_diff(&solo.vectors[0].values, &again.vectors[0].values),
        0.0,
        "{repo}: two identical calls produced different vectors"
    );

    eprintln!(
        "GATE {repo} duplicate={duplicate:.9} related={related:.6} unrelated={unrelated:.6} \
         margin={:.6} batch_cosine={batch_cosine:.9} batch_drift={drift:.3e}",
        related - unrelated
    );
    Some(reply.vectors.into_iter().map(|v| v.values).collect())
}

#[test]
fn nemotron_3_embed_bf16_ranks_the_related_passage_first() {
    let _guard = mlx_test_guard();
    run_gate(NEMOTRON_EMBED_BF16);
}

#[test]
fn nemotron_3_embed_8bit_agrees_with_the_bf16_original() {
    let _guard = mlx_test_guard();
    // The 8-bit conversion is a different weight layout end to end (packed
    // rows plus scales), so this is the one gate that would catch a quantized
    // path that loads and runs but reads the wrong channels.
    let (Some(bf16), Some(eight_bit)) =
        (run_gate(NEMOTRON_EMBED_BF16), run_gate(NEMOTRON_EMBED_8BIT))
    else {
        return;
    };
    for (i, (a, b)) in bf16.iter().zip(&eight_bit).enumerate() {
        let agreement = cosine(a, b);
        assert!(
            agreement >= 0.99,
            "input {i}: the 8-bit conversion scored {agreement} against the bf16 original"
        );
        eprintln!("GATE Nemotron-3-Embed quantization input={i} cosine={agreement:.6}");
    }
}

#[test]
fn padding_content_cannot_reach_a_real_position() {
    let _guard = mlx_test_guard();
    // The exact padding gate. Batch shape, slot and mask are held fixed and
    // only the token ids underneath the mask change, so nothing about the
    // kernel geometry moves and any difference is a genuine leak. Every real
    // position is fed only by masked attention, so the result has to be bit
    // identical. Both conversions are checked, since the quantized path reads
    // its weights through a different layout.
    for repo in [NEMOTRON_EMBED_BF16, NEMOTRON_EMBED_8BIT] {
        let Some(dir) = local_checkpoint(repo) else {
            continue;
        };
        let _runtime = crate::initialize_runtime();
        let loaded = crate::embeddings::load_embedding_model(&dir).expect("Nemotron-3-Embed loads");
        let dim = loaded.limits.dim;
        let width = 24_i32;
        let real = 10_i32;

        let run = |filler: i32| {
            let ids: Vec<i32> = (0..width)
                .map(|t| if t < real { 500 + t } else { filler })
                .collect();
            let mask: Vec<i32> = (0..width).map(|t| i32::from(t < real)).collect();
            let out = loaded
                .model
                .embed(&EmbeddingBatch {
                    input_ids: &ids_array(&ids, 1, width),
                    attention_mask: &ids_array(&mask, 1, width),
                    token_type_ids: None,
                    images: None,
                })
                .expect("Nemotron-3-Embed embeds the padded row");
            to_vec(&out.embeddings)
        };

        let with_pad_token = run(loaded.pad_token_id as i32);
        let with_other_token = run(777);
        let with_high_token = run(31337);
        assert_eq!(with_pad_token.len(), dim, "{repo}");
        assert_eq!(
            max_abs_diff(&with_pad_token, &with_other_token),
            0.0,
            "{repo}: the pad token id changed the vector"
        );
        assert_eq!(
            max_abs_diff(&with_other_token, &with_high_token),
            0.0,
            "{repo}: the masked tail's content changed the vector"
        );
        eprintln!("GATE {repo} pad_content_leak=0");
    }
}
