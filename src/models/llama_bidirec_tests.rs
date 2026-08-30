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

//! Bidirectional Llama (LLM2Vec) gates.
//!
//! Three properties carry this port. The weight map has to survive the
//! published layout: bare backbone roots, an optional `language_model.`
//! wrapper, a head that is never applied and the `rotary_emb.inv_freq` buffer
//! this tree rebuilds itself. The prefill has to be genuinely bidirectional,
//! which is the whole point of the LLM2Vec conversion and the one property a
//! reused causal backbone silently gets wrong. And a right-padded batch has to
//! reproduce the unpadded single row, which is what mean pooling over a
//! padding-only mask buys.
//!
//! The last test loads the real `nvidia/llama-nemotron-embed-1b-v2` checkpoint
//! and soft-skips when it is not downloaded.
//!
//! Every test that evaluates an MLX graph takes the shared `mlx_test_guard`;
//! see its doc comment for the two ways parallel MLX work breaks this suite.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;

use super::LlamaBidirecModel;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel};
use crate::embeddings::pooling::PoolingMode;
use crate::models::embedding_test_support::{
    Rng, cosine, err_string, ids_array, local_checkpoint, max_abs_diff, mlx_test_guard, temp_dir,
    to_vec,
};
use crate::models::llama3::ModelArgs;

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 32;

const LLAMA_NEMOTRON_EMBED: &str = "nvidia/llama-nemotron-embed-1b-v2";
/// `hidden_size` of the published checkpoint.
const CHECKPOINT_DIM: usize = 2048;

/// Cosine deviation allowed between the same text embedded alone and embedded
/// as a row of a longer, right-padded batch.
///
/// The issue asks for agreement within 1e-3, and the pooled vectors do agree to
/// that in cosine. They do not agree to 1e-3 in the largest single component,
/// and cannot on this hardware: bf16 attention picks its accumulation shape
/// from the batch geometry, so the vector moves when the batch does even with
/// no padding involved. Embedding one text as 2, 3, 4, 5 and 8 unpadded copies
/// of a single batch moved the largest component by up to 3.7e-3 while cosine
/// stayed above 0.99997, and the already-merged `Qwen/Qwen3-Embedding-0.6B`
/// behaves the same way, so the component form of that bound would gate the MLX
/// CUDA backend rather than this port.
///
/// What the port owes is that the PADDING changes nothing, and that is gated at
/// zero tolerance by `padding_content_cannot_reach_a_real_position`: same batch
/// shape, same slot, same mask, different token ids underneath the mask.
const BATCH_COSINE_TOLERANCE: f32 = 1e-3;

fn tiny_args(num_layers: usize) -> ModelArgs {
    ModelArgs {
        model_type: "llama_bidirec".to_string(),
        max_position_embeddings: None,
        hidden_size: HIDDEN,
        num_hidden_layers: num_layers,
        intermediate_size: INTERMEDIATE,
        num_attention_heads: 2,
        rms_norm_eps: 1e-6,
        vocab_size: VOCAB,
        head_dim: Some(HEAD_DIM),
        num_key_value_heads: Some(1),
        attention_bias: false,
        mlp_bias: false,
        rope_theta: 500_000.0,
        rope_scaling: None,
        quantization: None,
        tie_word_embeddings: true,
        rope_traditional: false,
        checkpoint_label: None,
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

fn build(args: &ModelArgs) -> Result<LlamaBidirecModel> {
    LlamaBidirecModel::from_weights(&tiny_weights(args), args, PoolingMode::Mean, true)
}

fn ramp(len: usize) -> Vec<i32> {
    (0..len).map(|i| (i % VOCAB) as i32).collect()
}

/// `[L, HIDDEN]` hidden states of one unpadded row.
fn hidden_states(model: &LlamaBidirecModel, ids: &[i32]) -> Vec<f32> {
    let length = ids.len() as i32;
    let input = ids_array(ids, 1, length);
    let mask = ids_array(&vec![1; ids.len()], 1, length);
    to_vec(&model.forward_hidden(&input, &mask))
}

/// The pooled `[HIDDEN]` vector of one unpadded row.
fn embed_one(model: &LlamaBidirecModel, ids: &[i32]) -> Vec<f32> {
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
fn sanitize_adds_model_prefix_and_drops_head_and_inv_freq() {
    // The published export saves the inner `LlamaBidirectionalModel`: bare
    // roots, no `model.` prefix. A hand-merged LLM2Vec checkpoint can also
    // carry a `language_model.` wrapper, a tied `lm_head` and the derived
    // `rotary_emb.inv_freq` / `position_ids` buffers, none of which this path
    // reads.
    let mut rng = Rng::new(7);
    let mut weights = WeightMap::new();
    for key in [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "norm.weight",
        "lm_head.weight",
        "layers.0.self_attn.rotary_emb.inv_freq",
        "position_ids",
        "language_model.layers.1.mlp.up_proj.weight",
        "language_model.model.layers.2.mlp.up_proj.weight",
    ] {
        rng.insert(&mut weights, key, &[2, 2], 0.1);
    }

    let folded = super::sanitize_llama_bidirec_weights(&mut weights);
    assert_eq!(folded, 0, "no Dense module in this fixture");

    let mut keys: Vec<&str> = weights.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.2.mlp.up_proj.weight",
            "model.norm.weight",
        ]
    );

    // Idempotent: a second pass is a no-op, so a checkpoint already in the mlx
    // layout loads unchanged.
    let before: Vec<String> = {
        let mut k: Vec<String> = weights.keys().cloned().collect();
        k.sort();
        k
    };
    super::sanitize_llama_bidirec_weights(&mut weights);
    let mut after: Vec<String> = weights.keys().cloned().collect();
    after.sort();
    assert_eq!(before, after);
}

#[test]
fn rejects_adapter_only_directory() {
    // An LLM2Vec PEFT adapter carries no backbone tensors at all, so without
    // this check the load would die on the first `weight not found` lookup and
    // say nothing about the merge the user actually has to do.
    let dir = temp_dir("llm2vec_adapter");
    std::fs::write(dir.join("adapter_config.json"), "{}").unwrap();
    std::fs::write(dir.join("adapter_model.safetensors"), b"not a real shard").unwrap();
    let err = err_string(super::reject_adapter_only_directory(&dir));
    assert!(err.contains("PEFT adapter"), "{err}");
    assert!(err.contains("merge"), "{err}");

    // The same directory with a full shard next to the adapter is accepted:
    // a merged export that happens to keep the adapter around is loadable.
    std::fs::write(dir.join("model.safetensors"), b"not a real shard").unwrap();
    assert!(super::reject_adapter_only_directory(&dir).is_ok());
    std::fs::remove_dir_all(dir).unwrap();
}

// Forward pass.

#[test]
fn bidirectional_prefill_sees_future_tokens() {
    let _guard = mlx_test_guard();
    // The defining property: changing the LAST token must move the FIRST
    // token's hidden state. A causal backbone reused unchanged would leave it
    // bit-identical, load fine and only produce quietly wrong vectors.
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
    // A right-padded row of a mixed-length batch must reproduce the vector the
    // same text gets alone: the padding-only mask blocks the pad keys, and
    // RoPE positions start at 0 in both runs because the caches are fresh.
    let args = tiny_args(2);
    let model = build(&args).unwrap();

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

    // The pad token id must not matter either: only the mask decides.
    let mut other_pad = padded_ids.clone();
    for slot in other_pad.iter_mut().take(12).skip(short.len()) {
        *slot = (VOCAB - 1) as i32;
    }
    let other_ids = ids_array(&other_pad, 2, 12);
    let other = to_vec(
        &model
            .embed(&EmbeddingBatch {
                input_ids: &other_ids,
                attention_mask: &batch_mask,
                token_type_ids: None,
                images: None,
            })
            .unwrap()
            .embeddings,
    );
    let pad_id_drift = max_abs_diff(&batch[..HIDDEN], &other[..HIDDEN]);
    assert!(
        pad_id_drift < 1e-4,
        "changing the pad token id moved the vector by {pad_id_drift}"
    );
}

#[test]
fn images_are_rejected() {
    let _guard = mlx_test_guard();
    let args = tiny_args(1);
    let model = build(&args).unwrap();
    assert_eq!(model.default_pooling(), PoolingMode::Mean);
    assert_eq!(model.embedding_dim(), HIDDEN);
    assert_eq!(model.format_text("query: a", None), "query: a");

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
fn llama_nemotron_embed_checkpoint_ranks_the_related_passage_first() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(LLAMA_NEMOTRON_EMBED) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let loaded = crate::embeddings::load_embedding_model(&dir).expect("bidirectional Llama loads");
    assert_eq!(loaded.model_type, crate::models::ModelType::LlamaBidirec);
    assert_eq!(loaded.limits.dim, CHECKPOINT_DIM);
    // sentence_bert_config.json max_seq_length 8192 equals the hard cap.
    assert_eq!(loaded.limits.max_length, 8192);

    let engine = crate::embeddings::EmbeddingEngine::new(loaded, 16);
    // Index 3 repeats index 0 verbatim: two byte-identical rows of one
    // right-padded batch must score cosine 1.0. The three texts have very
    // different lengths, so the batch is genuinely padded.
    let texts: Vec<String> = [
        "query: how do solar panels generate electricity",
        "passage: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect.",
        "passage: The recipe calls for two cups of flour and one egg.",
        "query: how do solar panels generate electricity",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let reply = engine
        .embed_texts(&texts, &crate::embeddings::EmbedOptions::default())
        .expect("bidirectional Llama embeds");

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
        "the solar passage must beat the recipe passage by 0.15: {related} vs {unrelated}"
    );
    assert!(
        unrelated < 0.5,
        "unrelated sentences scored {unrelated}, which is not below 0.5"
    );

    // Batch invariance on the real checkpoint: the shortest input embedded
    // alone must reproduce its padded row. See BATCH_COSINE_TOLERANCE for why
    // this is a cosine bound and where the exact padding gate lives.
    let solo = engine
        .embed_texts(
            &[texts[2].clone()],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("bidirectional Llama embeds a single row");
    let drift = max_abs_diff(&solo.vectors[0].values, &reply.vectors[2].values);
    let batch_cosine = cosine(&solo.vectors[0].values, &reply.vectors[2].values);
    assert!(
        batch_cosine >= 1.0 - BATCH_COSINE_TOLERANCE,
        "the passage drifted to cosine {batch_cosine} between the solo and the padded run"
    );

    // The same text embedded twice through two separate calls must be bit
    // identical: the forward pass carries no state between calls.
    let again = engine
        .embed_texts(
            &[texts[2].clone()],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("bidirectional Llama embeds a single row twice");
    assert_eq!(
        max_abs_diff(&solo.vectors[0].values, &again.vectors[0].values),
        0.0,
        "two identical calls produced different vectors"
    );

    // Bidirectionality on the real checkpoint: a prompt of at least 64 tokens
    // and its 32-token prefix must differ at position 0, which only a
    // bidirectional prefill produces.
    let long = "query: ".to_string() + &"the quick brown fox jumps over the lazy dog ".repeat(12);
    let words: Vec<&str> = long.split_whitespace().collect();
    let prefix = words[..words.len() / 3].join(" ");
    let both = engine
        .embed_texts(
            &[long.clone(), prefix],
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("bidirectional Llama embeds the long pair");
    let prefix_gap = cosine(&both.vectors[0].values, &both.vectors[1].values);
    assert!(
        prefix_gap < 0.999,
        "a truncated prefix produced the same vector ({prefix_gap}); the extra tokens changed nothing"
    );

    eprintln!(
        "GATE llama-nemotron-embed-1b-v2 duplicate={duplicate:.9} related={related:.6} \
         unrelated={unrelated:.6} margin={:.6} batch_cosine={batch_cosine:.9} \
         batch_drift={drift:.3e} prefix_cosine={prefix_gap:.6}",
        related - unrelated
    );
}

#[test]
fn padding_content_cannot_reach_a_real_position() {
    let _guard = mlx_test_guard();
    // The exact padding gate. Batch shape, slot and mask are all held fixed and
    // only the token ids underneath the mask change, so nothing about the
    // kernel geometry moves and any difference is a genuine leak. Every real
    // position of a bidirectional Llama row is fed only by masked attention, so
    // the result has to be bit identical.
    let Some(dir) = local_checkpoint(LLAMA_NEMOTRON_EMBED) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let loaded = crate::embeddings::load_embedding_model(&dir).expect("bidirectional Llama loads");
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
            .expect("bidirectional Llama embeds the padded row");
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
    eprintln!("GATE llama-nemotron-embed-1b-v2 pad_content_leak=0");
}
