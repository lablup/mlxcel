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

//! EmbeddingGemma gates.
//!
//! The forward tests build a 16-wide synthetic Gemma 3 from deterministic
//! weights, so they run on any device without a checkpoint. Two properties
//! carry the port and neither is visible in a shape check: the layers must be
//! bidirectional (a later token has to change an earlier token's state), and
//! only the full-attention layers may see past the sliding band.
//!
//! The last test loads the real `mlx-community/embeddinggemma-300m-4bit`
//! conversion and soft-skips when it is not downloaded.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

use super::{Gemma3EmbeddingModel, resolve_sliding_window_pattern};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel};
use crate::embeddings::pooling::PoolingMode;
use crate::models::embedding_test_support::{
    Rng, cosine, err_string, ids_array, local_checkpoint, max_abs_diff, mlx_test_guard, temp_dir,
    to_vec, write_f32_safetensors,
};
use crate::models::gemma3::ModelArgs;

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 32;
const WINDOW: usize = 4;
/// Output width of the second synthetic `Dense`; deliberately different from
/// `HIDDEN` so a test can tell the projected width from the backbone width.
const DENSE_OUT: usize = 24;

const EMBEDDINGGEMMA: &str = "mlx-community/embeddinggemma-300m-4bit";

fn tiny_args(num_layers: usize, sliding_window_pattern: usize) -> ModelArgs {
    ModelArgs {
        model_type: "gemma3_text".to_string(),
        hidden_size: HIDDEN,
        num_hidden_layers: num_layers,
        intermediate_size: INTERMEDIATE,
        num_attention_heads: 2,
        head_dim: HEAD_DIM,
        rms_norm_eps: 1e-6,
        vocab_size: VOCAB,
        num_key_value_heads: 1,
        rope_theta: 1_000_000.0,
        rope_local_base_freq: 10_000.0,
        query_pre_attn_scalar: HEAD_DIM as f32,
        sliding_window: WINDOW,
        sliding_window_pattern,
        max_position_embeddings: 128,
        rope_scaling: None,
        quantization: None,
    }
}

/// One synthetic tensor: `(sanitized key, shape, values)`.
type WeightSpec = (String, Vec<i32>, Vec<f32>);

/// Deterministic weights for `args`, keyed the way the sanitizer leaves them.
///
/// Returned as raw values rather than MLX arrays so the same numbers can be
/// loaded in memory and written to a synthetic checkpoint directory, which is
/// what makes the two published layouts comparable bit for bit.
fn tiny_weight_specs(args: &ModelArgs, dense: bool) -> Vec<WeightSpec> {
    let mut rng = Rng::new(0x5EED_1329);
    let mut specs: Vec<WeightSpec> = Vec::new();
    let h = args.hidden_size as i32;
    let hd = args.head_dim as i32;
    let q_out = args.num_attention_heads as i32 * hd;
    let kv_out = args.num_key_value_heads as i32 * hd;
    let inter = args.intermediate_size as i32;

    let mut push = |rng: &mut Rng, key: String, shape: Vec<i32>, scale: f32| {
        let count: i32 = shape.iter().product();
        let values = rng.values(count as usize, scale);
        specs.push((key, shape, values));
    };

    push(
        &mut rng,
        "model.embed_tokens.weight".to_string(),
        vec![args.vocab_size as i32, h],
        0.5,
    );
    for i in 0..args.num_hidden_layers {
        let p = format!("model.layers.{i}");
        push(
            &mut rng,
            format!("{p}.self_attn.q_proj.weight"),
            vec![q_out, h],
            0.2,
        );
        push(
            &mut rng,
            format!("{p}.self_attn.k_proj.weight"),
            vec![kv_out, h],
            0.2,
        );
        push(
            &mut rng,
            format!("{p}.self_attn.v_proj.weight"),
            vec![kv_out, h],
            0.2,
        );
        push(
            &mut rng,
            format!("{p}.self_attn.o_proj.weight"),
            vec![h, q_out],
            0.2,
        );
        push(
            &mut rng,
            format!("{p}.self_attn.q_norm.weight"),
            vec![hd],
            0.1,
        );
        push(
            &mut rng,
            format!("{p}.self_attn.k_norm.weight"),
            vec![hd],
            0.1,
        );
        push(
            &mut rng,
            format!("{p}.mlp.gate_proj.weight"),
            vec![inter, h],
            0.2,
        );
        push(
            &mut rng,
            format!("{p}.mlp.up_proj.weight"),
            vec![inter, h],
            0.2,
        );
        push(
            &mut rng,
            format!("{p}.mlp.down_proj.weight"),
            vec![h, inter],
            0.2,
        );
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            push(&mut rng, format!("{p}.{norm}.weight"), vec![h], 0.1);
        }
    }
    push(&mut rng, "model.norm.weight".to_string(), vec![h], 0.1);
    if dense {
        push(&mut rng, "dense.0.weight".to_string(), vec![inter, h], 0.2);
        push(
            &mut rng,
            "dense.1.weight".to_string(),
            vec![DENSE_OUT as i32, inter],
            0.2,
        );
    }
    specs
}

/// The same weights as an in-memory map.
fn tiny_weights(args: &ModelArgs, dense: bool) -> WeightMap {
    tiny_weight_specs(args, dense)
        .into_iter()
        .map(|(key, shape, values)| (key, mlxcel_core::from_slice_f32(&values, &shape)))
        .collect()
}

fn build(args: &ModelArgs, weights: &WeightMap) -> Result<Gemma3EmbeddingModel> {
    Gemma3EmbeddingModel::from_weights(weights, args, PoolingMode::Mean, true)
}

/// `[L, HIDDEN]` hidden states of one unpadded row.
fn hidden_states(model: &Gemma3EmbeddingModel, ids: &[i32]) -> Vec<f32> {
    let length = ids.len() as i32;
    let input = ids_array(ids, 1, length);
    let mask = ids_array(&vec![1; ids.len()], 1, length);
    to_vec(&model.forward_hidden(&input, &mask))
}

/// The first token's hidden state.
fn first_token(states: &[f32]) -> &[f32] {
    &states[..HIDDEN]
}

fn ramp(len: usize) -> Vec<i32> {
    (0..len).map(|i| (i % VOCAB) as i32).collect()
}

// Config surface.

#[test]
fn layer_types_drives_the_sliding_pattern() {
    // transformers 4.57 spells the scalar `_sliding_window_pattern`, which the
    // Gemma 3 args type does not parse: without `layer_types` the family
    // default would silently apply.
    let config = json!({
        "_sliding_window_pattern": 6,
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention"
        ]
    });
    assert_eq!(resolve_sliding_window_pattern(&config, 99).unwrap(), 6);

    // Underscore-prefixed scalar, no layer_types.
    let scalar = json!({"_sliding_window_pattern": 4});
    assert_eq!(resolve_sliding_window_pattern(&scalar, 99).unwrap(), 4);

    // Neither: the caller's fallback (the ModelArgs default) stands.
    assert_eq!(resolve_sliding_window_pattern(&json!({}), 6).unwrap(), 6);

    // All sliding: the period is pushed past the last layer so no layer is
    // ever treated as full attention.
    let all_sliding = json!({"layer_types": ["sliding_attention", "sliding_attention"]});
    assert_eq!(resolve_sliding_window_pattern(&all_sliding, 6).unwrap(), 3);
}

#[test]
fn irregular_layer_types_are_a_load_error() {
    let config = json!({
        "layer_types": ["sliding_attention", "full_attention", "full_attention"]
    });
    let err = resolve_sliding_window_pattern(&config, 6)
        .unwrap_err()
        .to_string();
    assert!(err.contains("layer_types"), "{err}");
    assert!(err.contains("period-2"), "{err}");
}

// Forward pass.

#[test]
fn bidirectional_prefill_is_not_causal() {
    let _guard = mlx_test_guard();
    // One full-attention layer over 96 tokens: flipping the *last* token must
    // move the *first* token's hidden state. A causal mask makes this exactly
    // zero, which is the regression this guards.
    let args = tiny_args(1, 1);
    let weights = tiny_weights(&args, false);
    let model = build(&args, &weights).unwrap();

    let mut ids = ramp(96);
    let baseline = hidden_states(&model, &ids);
    ids[95] = (ids[95] + 7) % VOCAB as i32;
    let perturbed = hidden_states(&model, &ids);

    let moved = max_abs_diff(first_token(&baseline), first_token(&perturbed));
    assert!(
        moved > 1e-4,
        "the first token did not react to the last token ({moved}); the mask is still causal"
    );
}

#[test]
fn global_layers_use_padding_mask_and_sliding_layers_use_window() {
    let _guard = mlx_test_guard();
    // Same weights, same input, one layer each: the only difference is whether
    // layer 0 is full attention (pattern 1) or sliding (pattern 2). A token
    // 6 positions away is outside the window of 4.
    let global_args = tiny_args(1, 1);
    let sliding_args = tiny_args(1, 2);
    let weights = tiny_weights(&global_args, false);
    let global = build(&global_args, &weights).unwrap();
    let sliding = build(&sliding_args, &weights).unwrap();

    let mut ids = ramp(12);
    let global_before = hidden_states(&global, &ids);
    let sliding_before = hidden_states(&sliding, &ids);
    ids[6] = (ids[6] + 11) % VOCAB as i32;
    let global_after = hidden_states(&global, &ids);
    let sliding_after = hidden_states(&sliding, &ids);

    let global_moved = max_abs_diff(first_token(&global_before), first_token(&global_after));
    let sliding_moved = max_abs_diff(first_token(&sliding_before), first_token(&sliding_after));
    assert!(
        global_moved > 1e-4,
        "the full-attention layer ignored a key 6 positions away ({global_moved})"
    );
    assert!(
        sliding_moved < 1e-6,
        "the sliding layer attended a key 6 positions away with window {WINDOW} ({sliding_moved})"
    );
}

#[test]
fn padding_invariance() {
    let _guard = mlx_test_guard();
    // Row 0 of a right-padded two-row batch must equal the same text embedded
    // alone: padding keys are blocked, and mean pooling divides by the real
    // token count.
    let args = tiny_args(2, 2);
    let weights = tiny_weights(&args, true);
    let model = build(&args, &weights).unwrap();

    let short = ramp(8);
    let long: Vec<i32> = (0..12).map(|i| ((i * 5 + 3) % VOCAB) as i32).collect();

    let alone = ids_array(&short, 1, short.len() as i32);
    let alone_mask = ids_array(&vec![1; short.len()], 1, short.len() as i32);
    let alone_out = model
        .embed(&EmbeddingBatch {
            input_ids: &alone,
            attention_mask: &alone_mask,
            token_type_ids: None,
            images: None,
        })
        .unwrap();
    let alone_values = to_vec(&alone_out.embeddings);

    let mut padded_ids = short.clone();
    padded_ids.resize(12, 0);
    padded_ids.extend_from_slice(&long);
    let mut padded_mask = vec![1; short.len()];
    padded_mask.resize(12, 0);
    padded_mask.extend(std::iter::repeat_n(1, 12));

    let batch_ids = ids_array(&padded_ids, 2, 12);
    let batch_mask = ids_array(&padded_mask, 2, 12);
    let batch_out = model
        .embed(&EmbeddingBatch {
            input_ids: &batch_ids,
            attention_mask: &batch_mask,
            token_type_ids: None,
            images: None,
        })
        .unwrap();
    let batch_values = to_vec(&batch_out.embeddings);

    assert_eq!(alone_values.len(), DENSE_OUT);
    assert_eq!(batch_values.len(), 2 * DENSE_OUT);
    let row0 = &batch_values[..DENSE_OUT];
    assert!(
        cosine(&alone_values, row0) > 1.0 - 1e-4,
        "padded row 0 diverged from the unpadded run: cosine {}",
        cosine(&alone_values, row0)
    );
    assert!(max_abs_diff(&alone_values, row0) < 1e-3);
}

#[test]
fn dense_stack_sets_the_embedding_width() {
    let _guard = mlx_test_guard();
    let args = tiny_args(1, 1);
    let with_dense = build(&args, &tiny_weights(&args, true)).unwrap();
    let without_dense = build(&args, &tiny_weights(&args, false)).unwrap();
    assert_eq!(with_dense.embedding_dim(), DENSE_OUT);
    assert_eq!(without_dense.embedding_dim(), HIDDEN);
    assert_eq!(with_dense.default_pooling(), PoolingMode::Mean);
}

#[test]
fn sentence_transformers_subfolder_layout_loads_from_disk_and_matches_the_mlx_layout() {
    let _guard = mlx_test_guard();
    // The real `load(dir)` path over a synthetic checkpoint written in the
    // sentence-transformers layout: bare backbone roots, an unused lm_head,
    // and the two projections in `2_Dense/` and `3_Dense/` module folders.
    // The mlx conversion of the same numbers must embed identically, which is
    // the property `mlx-community/embeddinggemma-300m-4bit` alone cannot show
    // (it only ships one of the two layouts).
    let args = tiny_args(2, 2);
    let specs = tiny_weight_specs(&args, true);

    let dir = temp_dir("embeddinggemma_subfolders");
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&json!({
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextModel"],
            "use_bidirectional_attention": true,
            "hidden_size": HIDDEN,
            "num_hidden_layers": 2,
            "intermediate_size": INTERMEDIATE,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": HEAD_DIM,
            "rms_norm_eps": 1e-6,
            "vocab_size": VOCAB,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "query_pre_attn_scalar": HEAD_DIM,
            "sliding_window": WINDOW,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 128,
        }))
        .unwrap(),
    )
    .unwrap();

    let mut backbone: Vec<WeightSpec> = Vec::new();
    let mut up: Vec<WeightSpec> = Vec::new();
    let mut down: Vec<WeightSpec> = Vec::new();
    for (key, shape, values) in specs {
        match key.as_str() {
            "dense.0.weight" => up.push(("linear.weight".to_string(), shape, values)),
            "dense.1.weight" => down.push(("linear.weight".to_string(), shape, values)),
            _ => backbone.push((
                key.strip_prefix("model.").unwrap_or(&key).to_string(),
                shape,
                values,
            )),
        }
    }
    // An lm_head the embedder must drop rather than trip over.
    backbone.push((
        "lm_head.weight".to_string(),
        vec![VOCAB as i32, HIDDEN as i32],
        vec![0.0; VOCAB * HIDDEN],
    ));
    write_f32_safetensors(&dir.join("model.safetensors"), &backbone);
    for (folder, tensors) in [("2_Dense", &up), ("3_Dense", &down)] {
        std::fs::create_dir_all(dir.join(folder)).unwrap();
        write_f32_safetensors(&dir.join(folder).join("model.safetensors"), tensors);
    }

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let from_disk = Gemma3EmbeddingModel::load(&dir, &config).expect("subfolder layout loads");
    assert_eq!(from_disk.embedding_dim(), DENSE_OUT);

    let from_memory = build(&args, &tiny_weights(&args, true)).unwrap();
    let ids = ramp(12);
    let input = ids_array(&ids, 1, ids.len() as i32);
    let mask = ids_array(&vec![1; ids.len()], 1, ids.len() as i32);
    let batch = EmbeddingBatch {
        input_ids: &input,
        attention_mask: &mask,
        token_type_ids: None,
        images: None,
    };
    let disk_values = to_vec(&from_disk.embed(&batch).unwrap().embeddings);
    let memory_values = to_vec(&from_memory.embed(&batch).unwrap().embeddings);
    assert_eq!(
        max_abs_diff(&disk_values, &memory_values),
        0.0,
        "the two published layouts produced different embeddings"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn swapped_dense_projections_are_a_load_error() {
    let _guard = mlx_test_guard();
    let args = tiny_args(1, 1);
    let mut weights = tiny_weights(&args, true);
    let up = weights.remove("dense.0.weight").unwrap();
    let down = weights.remove("dense.1.weight").unwrap();
    weights.insert("dense.0.weight".to_string(), down);
    weights.insert("dense.1.weight".to_string(), up);

    let err = err_string(build(&args, &weights));
    assert!(err.contains("dense.0"), "{err}");
    assert!(err.contains("out of order"), "{err}");
}

#[test]
fn images_are_rejected() {
    let _guard = mlx_test_guard();
    let args = tiny_args(1, 1);
    let model = build(&args, &tiny_weights(&args, false)).unwrap();
    let ids = ids_array(&[1, 2, 3], 1, 3);
    let mask = ids_array(&[1, 1, 1], 1, 3);
    let image = crate::embeddings::model::ImageInput {
        image: image::DynamicImage::new_rgb8(2, 2),
    };
    let images = [image];
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
fn embeddinggemma_checkpoint_loads_and_ranks_the_matching_document() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(EMBEDDINGGEMMA) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let loaded = crate::embeddings::load_embedding_model(&dir).expect("EmbeddingGemma loads");
    assert_eq!(loaded.model_type, crate::models::ModelType::Gemma3Embedding);
    assert_eq!(loaded.limits.dim, 768, "768-wide after the Dense stack");
    assert_eq!(
        loaded.limits.max_length, 2048,
        "sentence_bert_config.json max_seq_length"
    );

    let engine = crate::embeddings::EmbeddingEngine::new(loaded, 16);
    // Index 4 repeats index 0 verbatim: two byte-identical rows of one
    // right-padded batch must produce cosine 1.0, which is the self-
    // consistency gate and also the tell for cross-row interference.
    let texts: Vec<String> = [
        "task: search result | query: Which planet is known as the Red Planet?",
        "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "title: none | text: Venus is the second planet from the Sun.",
        "title: none | text: Jupiter is the largest planet in our solar system.",
        "task: search result | query: Which planet is known as the Red Planet?",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let reply = engine
        .embed_texts(&texts, &crate::embeddings::EmbedOptions::default())
        .expect("EmbeddingGemma embeds");

    assert_eq!(reply.vectors.len(), 5);
    for vector in &reply.vectors {
        assert_eq!(vector.shape, vec![768]);
        let norm: f32 = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "unit vector, got {norm}");
    }
    let query = &reply.vectors[0].values;
    let duplicate = cosine(query, &reply.vectors[4].values);
    assert!(
        (duplicate - 1.0).abs() < 1e-6,
        "identical inputs scored {duplicate}, not 1.0"
    );
    let mars = cosine(query, &reply.vectors[1].values);
    let venus = cosine(query, &reply.vectors[2].values);
    let jupiter = cosine(query, &reply.vectors[3].values);
    assert!(
        mars > venus + 0.1 && mars > jupiter + 0.1,
        "Mars {mars}, Venus {venus}, Jupiter {jupiter}"
    );
    assert!(
        venus < 0.5 && jupiter < 0.5,
        "Venus {venus}, Jupiter {jupiter}"
    );

    // Matryoshka truncation: 256 trained components, re-normalized, ranking
    // unchanged.
    let truncated = engine
        .embed_texts(
            &texts[..4],
            &crate::embeddings::EmbedOptions {
                instruction: None,
                dimensions: Some(256),
                normalize: None,
            },
        )
        .expect("EmbeddingGemma embeds at 256 dimensions");
    for vector in &truncated.vectors {
        assert_eq!(vector.shape, vec![256]);
        let norm: f32 = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "unit vector, got {norm}");
    }
    let short_query = &truncated.vectors[0].values;
    let short_mars = cosine(short_query, &truncated.vectors[1].values);
    let short_venus = cosine(short_query, &truncated.vectors[2].values);
    let short_jupiter = cosine(short_query, &truncated.vectors[3].values);
    assert!(
        short_mars > short_venus && short_mars > short_jupiter,
        "ranking changed at 256 dimensions: Mars {short_mars}, Venus {short_venus}, Jupiter {short_jupiter}"
    );

    // Printed so a repeated run shows the spread rather than one pass/fail bit.
    eprintln!(
        "GATE embeddinggemma-300m-4bit duplicate={duplicate:.9} mars={mars:.6} venus={venus:.6} \
         jupiter={jupiter:.6} mars@256={short_mars:.6} venus@256={short_venus:.6} \
         jupiter@256={short_jupiter:.6}"
    );
}

#[test]
fn embeddinggemma_long_document_past_the_sliding_window_is_batch_invariant() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(EMBEDDINGGEMMA) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let loaded = crate::embeddings::load_embedding_model(&dir).expect("EmbeddingGemma loads");
    let engine = crate::embeddings::EmbeddingEngine::new(loaded, 4);

    // Well past the 512-token sliding window, so the windowed mask and the
    // full mask genuinely disagree on this input.
    let sentence = "Mars is the fourth planet from the Sun and the second smallest planet in the Solar System, with a thin carbon dioxide atmosphere and two small moons. ";
    let long = format!("title: none | text: {}", sentence.repeat(60));
    let short = "task: search result | query: Which planet is known as the Red Planet?".to_string();

    let alone = engine
        .embed_texts(
            std::slice::from_ref(&long),
            &crate::embeddings::EmbedOptions::default(),
        )
        .expect("long document embeds alone");
    let batched = engine
        .embed_texts(&[short, long], &crate::embeddings::EmbedOptions::default())
        .expect("long document embeds in a padded batch");

    assert!(
        alone.prompt_tokens > 512,
        "the long document is only {} tokens; it must exceed the sliding window",
        alone.prompt_tokens
    );
    let solo = &alone.vectors[0].values;
    let in_batch = &batched.vectors[1].values;
    let drift = max_abs_diff(solo, in_batch);
    assert!(
        drift < 1e-3,
        "the long document drifted by {drift} between the solo and the padded run"
    );
    assert!(
        solo.iter().all(|v| v.is_finite()),
        "the long document produced a non-finite component"
    );
    eprintln!(
        "GATE embeddinggemma-300m-4bit long_document tokens={} batch_drift={drift:.3e}",
        alone.prompt_tokens
    );
}
