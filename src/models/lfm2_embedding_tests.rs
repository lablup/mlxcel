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

//! LFM2.5-Embedding gates.
//!
//! Four properties carry this port. The final norm root is spelled
//! `embedding_norm`, which the shared decoder-embedding sanitize does not
//! cover, so the family adds it and a missed prefix has to be a load error
//! rather than a silent fallback. CLS pooling has to read the
//! `<|startoftext|>` position and nothing else. The prefill has to be
//! bidirectional through both mixer kinds, attention and short conv. And
//! padding has to be invisible to every real position, which needs two
//! different mechanisms here: the attention mask removes padding from the key
//! axis, while the convolution, which has no key axis, has its input zeroed at
//! padding positions instead. Missing the second one is the bug
//! `padding_content_cannot_reach_a_real_position` caught, at cosine 0.94.
//!
//! The directional short convolution itself is gated in
//! `src/models/lfm2_tests.rs`, next to the mixer it changes.
//!
//! The last test loads the real `LiquidAI/LFM2.5-Embedding-350M` checkpoint and
//! soft-skips when it is not downloaded.
//!
//! Every test that evaluates an MLX graph takes the shared `mlx_test_guard`;
//! see its doc comment for the two ways parallel MLX work breaks this suite.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;

use super::Lfm2EmbeddingModel;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel};
use crate::embeddings::pooling::PoolingMode;
use crate::models::embedding_test_support::{
    Rng, cosine, err_string, ids_array, local_checkpoint, max_abs_diff, mlx_test_guard, temp_dir,
    to_vec,
};
use crate::models::lfm2::ModelArgs;

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const VOCAB: usize = 32;
/// Two conv layers around one attention layer, so both mixer kinds run.
const LAYER_TYPES: [&str; 3] = ["conv", "full_attention", "conv"];

const LFM2_EMBEDDING: &str = "LiquidAI/LFM2.5-Embedding-350M";
/// `hidden_size` of the published checkpoint.
const CHECKPOINT_DIM: usize = 1024;

/// Cosine deviation allowed between the same text embedded alone and embedded
/// as a row of a longer, right-padded batch.
///
/// The issue asks for agreement within 1e-3, and the pooled vectors do agree to
/// that in cosine. They do not agree to 1e-3 in the largest single component,
/// and cannot on this hardware: bf16 attention picks its accumulation shape
/// from the batch geometry, so the vector moves when the batch does even with
/// no padding involved. CLS pooling reads one position rather than averaging
/// many, so this family shows the effect more than the mean-pooled ones do; the
/// already-merged `Qwen/Qwen3-Embedding-0.6B` shows it too.
///
/// What the port owes is that the PADDING changes nothing, and that is gated at
/// zero tolerance by `padding_content_cannot_reach_a_real_position`: same batch
/// shape, same slot, same mask, different token ids underneath the mask. That
/// test is the one that caught the real bug in this port, a short-conv mixer
/// reading pad-token embeddings.
const BATCH_COSINE_TOLERANCE: f32 = 1e-3;

fn tiny_args() -> ModelArgs {
    ModelArgs {
        model_type: "lfm2".to_string(),
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_hidden_layers: LAYER_TYPES.len(),
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        conv_bias: false,
        conv_l_cache: 3,
        rope_theta: 1_000_000.0,
        conv_causal: false,
        full_attn_idxs: None,
        layer_types: Some(LAYER_TYPES.iter().map(|t| t.to_string()).collect()),
        intermediate_size: Some(INTERMEDIATE),
        moe_intermediate_size: None,
        num_experts: None,
        num_experts_per_tok: None,
        num_dense_layers: None,
        norm_topk_prob: None,
        use_expert_bias: None,
        routed_scaling_factor: 1.0,
        eos_token_id: Some(serde_json::json!(7)),
        quantization: None,
    }
}

/// Deterministic dense weights for `args`, in the checkpoint's own key layout
/// (`w1`/`w2`/`w3` feed-forward, `[hidden, 1, L_cache]` conv weight), so
/// `Lfm2Model::from_weights`'s own sanitize pass runs too.
fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let mut rng = Rng::new(0x00C0_FFEE_1325);
    let mut w = WeightMap::new();
    let h = args.hidden_size as i32;
    let hd = args.head_dim() as i32;
    let q_out = args.num_attention_heads as i32 * hd;
    let kv_out = args.num_key_value_heads as i32 * hd;
    let inter = args.intermediate_size.unwrap() as i32;
    let l_cache = args.conv_l_cache as i32;

    rng.insert(
        &mut w,
        "model.embed_tokens.weight",
        &[args.vocab_size as i32, h],
        0.5,
    );
    for i in 0..args.num_hidden_layers {
        let p = format!("model.layers.{i}");
        if args.is_attention_layer(i) {
            for (name, shape) in [
                ("q_proj", [q_out, h]),
                ("k_proj", [kv_out, h]),
                ("v_proj", [kv_out, h]),
                ("out_proj", [h, q_out]),
            ] {
                rng.insert(&mut w, &format!("{p}.self_attn.{name}.weight"), &shape, 0.2);
            }
            rng.insert(
                &mut w,
                &format!("{p}.self_attn.q_layernorm.weight"),
                &[hd],
                0.1,
            );
            rng.insert(
                &mut w,
                &format!("{p}.self_attn.k_layernorm.weight"),
                &[hd],
                0.1,
            );
        } else {
            rng.insert(
                &mut w,
                &format!("{p}.conv.in_proj.weight"),
                &[3 * h, h],
                0.2,
            );
            rng.insert(&mut w, &format!("{p}.conv.out_proj.weight"), &[h, h], 0.2);
            // PyTorch orientation; `Lfm2Model::from_weights` transposes it.
            rng.insert(
                &mut w,
                &format!("{p}.conv.conv.weight"),
                &[h, 1, l_cache],
                0.4,
            );
        }
        rng.insert(
            &mut w,
            &format!("{p}.feed_forward.w1.weight"),
            &[inter, h],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.feed_forward.w2.weight"),
            &[h, inter],
            0.2,
        );
        rng.insert(
            &mut w,
            &format!("{p}.feed_forward.w3.weight"),
            &[inter, h],
            0.2,
        );
        rng.insert(&mut w, &format!("{p}.operator_norm.weight"), &[h], 0.1);
        rng.insert(&mut w, &format!("{p}.ffn_norm.weight"), &[h], 0.1);
    }
    rng.insert(&mut w, "model.embedding_norm.weight", &[h], 0.1);
    w
}

fn build(pooling: PoolingMode) -> Result<Lfm2EmbeddingModel> {
    let args = tiny_args();
    Lfm2EmbeddingModel::from_weights(tiny_weights(&args), args, pooling, true)
}

fn ramp(len: usize) -> Vec<i32> {
    (0..len).map(|i| (i % VOCAB) as i32).collect()
}

/// `[L, HIDDEN]` hidden states of one unpadded row.
fn hidden_states(model: &Lfm2EmbeddingModel, ids: &[i32]) -> Vec<f32> {
    let length = ids.len() as i32;
    let input = ids_array(ids, 1, length);
    let mask = ids_array(&vec![1; ids.len()], 1, length);
    to_vec(&model.forward_hidden(&input, &mask))
}

/// The pooled `[HIDDEN]` vector of one unpadded row.
fn embed_one(model: &Lfm2EmbeddingModel, ids: &[i32]) -> Vec<f32> {
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

// Weight sanitize.

#[test]
fn embedding_norm_root_is_prefixed() {
    // LFM2's final norm is `embedding_norm.weight`, which matches none of the
    // shared helper's backbone roots (`embed_tokens.`, `layers.`, `norm.`), so
    // without this family's own pass the constructor would fail looking for
    // `model.embedding_norm.weight`.
    let mut rng = Rng::new(11);
    let mut weights = WeightMap::new();
    for key in [
        "embedding_norm.weight",
        "model.embedding_norm.weight",
        "layers.0.ffn_norm.weight",
    ] {
        rng.insert(&mut weights, key, &[2, 2], 0.1);
    }
    super::prefix_embedding_norm(&mut weights);
    assert!(weights.contains_key("model.embedding_norm.weight"));
    assert!(!weights.contains_key("embedding_norm.weight"));
    // Idempotent: an mlx conversion already carrying the prefix is untouched.
    let before = weights.len();
    super::prefix_embedding_norm(&mut weights);
    assert_eq!(weights.len(), before);
}

#[test]
fn rejects_late_interaction_layout() {
    // The LFM2 / LFM2.5 ColBERT checkpoints share this `model_type` and this
    // architecture but project every token through a `1_Dense` module. Loading
    // one here would return a single pooled vector and look like it worked.
    let dir = temp_dir("lfm2_colbert");
    std::fs::create_dir_all(dir.join("1_Dense")).unwrap();
    let err = err_string(super::reject_late_interaction(&dir, 0));
    assert!(err.contains("late-interaction"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();

    // The tensor-level signal catches the same layout when the folder is
    // spelled differently.
    let plain = temp_dir("lfm2_plain");
    assert!(super::reject_late_interaction(&plain, 0).is_ok());
    assert!(super::reject_late_interaction(&plain, 1).is_err());
    std::fs::remove_dir_all(plain).unwrap();
}

// Forward pass.

#[test]
fn cls_pooling_reads_bos_position() {
    let _guard = mlx_test_guard();
    // The tokenizer prepends `<|startoftext|>` and the checkpoint pools it.
    // Right padding puts it at index 0, so the pooled vector must equal the
    // hidden state there, and nothing else in the row may reach it.
    let model = build(PoolingMode::Cls).unwrap();
    let ids = ramp(10);
    let hidden = hidden_states(&model, &ids);
    let pooled = embed_one(&model, &ids);

    assert_eq!(pooled.len(), HIDDEN);
    let drift = max_abs_diff(&pooled, &hidden[..HIDDEN]);
    assert!(
        drift < 1e-5,
        "CLS pooling returned something other than position 0 ({drift})"
    );

    // Mean pooling over the same states is a different vector, so the gate
    // above is not passing by coincidence.
    let mean_model = build(PoolingMode::Mean).unwrap();
    let mean = embed_one(&mean_model, &ids);
    assert!(
        max_abs_diff(&pooled, &mean) > 1e-4,
        "CLS and mean pooling produced the same vector"
    );
}

#[test]
fn bidirectional_prefill_sees_future_tokens() {
    let _guard = mlx_test_guard();
    // Both mixer kinds have to look forward: the attention layers through the
    // padding-only mask, the conv layers through the split padding. Changing
    // the last of 96 tokens must move position 0.
    //
    // The bar is "moved at all" rather than a magnitude, and the control below
    // is what makes that meaningful: with the attention layer replaced by a
    // conv, token 95 is 92 positions outside the stack's `L_cache / 2` per
    // layer reach, so position 0 stays bit identical. Anything above zero is
    // therefore the mask doing its job, not noise.
    let model = build(PoolingMode::Cls).unwrap();

    let mut ids = ramp(96);
    let baseline = hidden_states(&model, &ids);
    ids[95] = (ids[95] + 7) % VOCAB as i32;
    let perturbed = hidden_states(&model, &ids);

    let first = max_abs_diff(&baseline[..HIDDEN], &perturbed[..HIDDEN]);
    assert!(
        first > 0.0,
        "position 0 did not react to the last token; the prefill is still causal"
    );

    // Control: an all-conv stack cannot reach position 0 from token 95.
    let mut conv_only_args = tiny_args();
    conv_only_args.layer_types = Some(vec!["conv".to_string(); LAYER_TYPES.len()]);
    let conv_only = Lfm2EmbeddingModel::from_weights(
        tiny_weights(&conv_only_args),
        conv_only_args,
        PoolingMode::Cls,
        true,
    )
    .unwrap();
    let conv_baseline = hidden_states(&conv_only, &ramp(96));
    let conv_perturbed = hidden_states(&conv_only, &ids);
    assert_eq!(
        max_abs_diff(&conv_baseline[..HIDDEN], &conv_perturbed[..HIDDEN]),
        0.0,
        "the all-conv control reached position 0 from token 95, so the gate above proves nothing"
    );

    // The conv's own look-ahead: position 1 is inside its reach, so perturbing
    // it must move position 0 even without an attention layer.
    let mut near = ramp(96);
    near[1] = (near[1] + 7) % VOCAB as i32;
    let near_perturbed = hidden_states(&conv_only, &near);
    assert!(
        max_abs_diff(&conv_baseline[..HIDDEN], &near_perturbed[..HIDDEN]) > 0.0,
        "position 0 did not react to position 1; the short conv is still causal"
    );
}

#[test]
fn padding_invariance() {
    let _guard = mlx_test_guard();
    let model = build(PoolingMode::Cls).unwrap();

    let short = ramp(8);
    let long: Vec<i32> = (0..12).map(|i| ((i * 5 + 3) % VOCAB) as i32).collect();
    let short_alone = embed_one(&model, &short);
    let long_alone = embed_one(&model, &long);

    let mut padded_ids = short.clone();
    padded_ids.resize(12, 0);
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
    // CLS pools position 0, which is `conv_L_cache / 2 * num_conv_layers`
    // positions away from any padding in this fixture, so the conv leak the
    // module doc describes cannot reach it at all.
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
    let model = build(PoolingMode::Cls).unwrap();
    assert_eq!(model.default_pooling(), PoolingMode::Cls);
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

// Real checkpoint.

#[test]
fn lfm2_embedding_checkpoint_ranks_the_related_passage_first() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(LFM2_EMBEDDING) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let loaded = crate::embeddings::load_embedding_model(&dir).expect("LFM2.5-Embedding loads");
    assert_eq!(loaded.model_type, crate::models::ModelType::Lfm2Embedding);
    assert_eq!(loaded.limits.dim, CHECKPOINT_DIM);
    // sentence_bert_config.json max_seq_length 512 wins over the hard cap.
    assert_eq!(loaded.limits.max_length, 512);

    let engine = crate::embeddings::EmbeddingEngine::new(loaded, 16);
    // Index 3 repeats index 0 verbatim: two byte-identical rows of one
    // right-padded batch must score cosine 1.0.
    let texts: Vec<String> = [
        "query: how do solar panels generate electricity",
        "document: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect.",
        "document: The recipe calls for two cups of flour and one egg.",
        "query: how do solar panels generate electricity",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let reply = engine
        .embed_texts(&texts, &crate::embeddings::EmbedOptions::default())
        .expect("LFM2.5-Embedding embeds");

    assert_eq!(reply.vectors.len(), 4);
    for vector in &reply.vectors {
        assert_eq!(vector.shape, vec![CHECKPOINT_DIM]);
        assert!(
            vector.values.iter().all(|v| v.is_finite()),
            "a non-finite component reached the caller"
        );
        let norm: f32 = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "unit vector, got {norm}");
    }

    let duplicate = cosine(&reply.vectors[0].values, &reply.vectors[3].values);
    assert!(
        (duplicate - 1.0).abs() < 1e-6,
        "identical inputs scored {duplicate}, not 1.0"
    );

    let related = cosine(&reply.vectors[0].values, &reply.vectors[1].values);
    let unrelated = cosine(&reply.vectors[0].values, &reply.vectors[2].values);
    assert!(
        related - unrelated >= 0.15,
        "the solar document must beat the recipe document by 0.15: {related} vs {unrelated}"
    );
    assert!(
        unrelated < 0.5,
        "unrelated sentences scored {unrelated}, which is not below 0.5"
    );

    // Batch invariance. See BATCH_COSINE_TOLERANCE for why this is a cosine
    // bound and where the exact padding gate lives.
    let solo = engine
        .embed_texts(
            &[texts[2].clone()],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("LFM2.5-Embedding embeds a single row");
    let drift = max_abs_diff(&solo.vectors[0].values, &reply.vectors[2].values);
    let batch_cosine = cosine(&solo.vectors[0].values, &reply.vectors[2].values);
    assert!(
        batch_cosine >= 1.0 - BATCH_COSINE_TOLERANCE,
        "the document drifted to cosine {batch_cosine} between the solo and the padded run"
    );

    // Two separate calls on the same input must be bit identical: the forward
    // pass carries no state between calls, conv state included.
    let again = engine
        .embed_texts(
            &[texts[2].clone()],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("LFM2.5-Embedding embeds a single row twice");
    assert_eq!(
        max_abs_diff(&solo.vectors[0].values, &again.vectors[0].values),
        0.0,
        "two identical calls produced different vectors"
    );

    // Bidirectionality on the real checkpoint: CLS pools position 0, so a
    // longer prompt and its prefix can only differ there if the prefill looks
    // forward.
    let long = "query: ".to_string() + &"the quick brown fox jumps over the lazy dog ".repeat(12);
    let words: Vec<&str> = long.split_whitespace().collect();
    let prefix = words[..words.len() / 3].join(" ");
    let both = engine
        .embed_texts(
            &[long.clone(), prefix],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("LFM2.5-Embedding embeds the long pair");
    let prefix_gap = cosine(&both.vectors[0].values, &both.vectors[1].values);
    assert!(
        prefix_gap < 0.999,
        "a truncated prefix produced the same CLS vector ({prefix_gap}); the prefill is causal"
    );

    eprintln!(
        "GATE LFM2.5-Embedding-350M duplicate={duplicate:.9} related={related:.6} \
         unrelated={unrelated:.6} margin={:.6} batch_cosine={batch_cosine:.9} \
         batch_drift={drift:.3e} prefix_cosine={prefix_gap:.6}",
        related - unrelated
    );
}

#[test]
fn padding_content_cannot_reach_a_real_position() {
    let _guard = mlx_test_guard();
    // The exact padding gate, and the one that caught the real bug in this
    // port. Batch shape, slot and mask are held fixed and only the token ids
    // underneath the mask change, so nothing about the kernel geometry moves
    // and any difference is a genuine leak.
    //
    // A convolution has no key axis for the attention mask to act on, so before
    // the short-conv input was zeroed at padding positions this gate failed at
    // cosine 0.94: the pad-token embeddings mixed into the real positions next
    // to the boundary and the six attention layers above spread that across the
    // whole row, including position 0 where CLS pools.
    let Some(dir) = local_checkpoint(LFM2_EMBEDDING) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let loaded = crate::embeddings::load_embedding_model(&dir).expect("LFM2.5-Embedding loads");
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
            .expect("LFM2.5-Embedding embeds the padded row");
        to_vec(&out.embeddings)
    };

    let with_pad_token = run(loaded.pad_token_id as i32);
    let with_other_token = run(777);
    let with_high_token = run(31337);
    assert_eq!(with_pad_token.len(), dim);
    assert_eq!(
        max_abs_diff(&with_pad_token, &with_other_token),
        0.0,
        "the pad token id changed the vector"
    );
    assert_eq!(
        max_abs_diff(&with_other_token, &with_high_token),
        0.0,
        "the masked tail's content changed the vector"
    );
    eprintln!("GATE LFM2.5-Embedding-350M pad_content_leak=0");
}
