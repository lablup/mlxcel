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

//! Unit tests for the SigLIP text tower plus the real-checkpoint gates of
//! issue #1341. The checkpoint gates soft-skip when
//! `google/siglip-base-patch16-224` is absent, following the convention of
//! `src/embeddings/real_checkpoint_tests.rs`. Fetch it with:
//!
//! ```sh
//! mlxcel download google/siglip-base-patch16-224
//! ```

use std::path::PathBuf;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::json;

use super::test_guard;
use super::{SigLipTextArgs, SigLipTextModel, sanitize_siglip_text_weights};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel};
use crate::embeddings::pooling::PoolingMode;
use crate::embeddings::tokenize::{EncodeOptions, encode_batch, strip_padding_and_truncation};
use crate::embeddings::{EmbedOptions, EmbeddingEngine, load_embedding_model};
use crate::models::{ModelType, get_model_type};
use crate::tokenizer::load_tokenizer;
use crate::vision::config::VisionHiddenActivation;
use crate::vision::encoders::siglip::VisionMlpActivation;

const SIGLIP_BASE: &str = "google/siglip-base-patch16-224";
/// `</s>`, which the SigLIP tokenizer appends and also pads with.
const EOS_ID: i32 = 1;

// ---------------------------------------------------------------- helpers

/// Deterministic linear congruential generator: the synthetic towers below
/// must be reproducible without pulling in a random-number crate.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }

    fn insert(
        &mut self,
        weights: &mut WeightMap,
        name: &str,
        shape: &[i32],
        scale: f32,
        offset: f32,
    ) {
        let count: usize = shape.iter().map(|&d| d as usize).product();
        let values: Vec<f32> = (0..count)
            .map(|_| self.next_f32() * scale + offset)
            .collect();
        weights.insert(
            name.to_string(),
            mlxcel_core::from_slice_f32(&values, shape),
        );
    }
}

/// A small but structurally complete text tower: `layers` blocks at
/// `hidden = 8`, `intermediate = 16`, two heads, `positions` learned
/// positions, a 32-entry vocabulary and an 8-wide projection head.
fn synthetic_args(positions: usize, layers: usize) -> SigLipTextArgs {
    SigLipTextArgs {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 16,
        num_attention_heads: 2,
        num_hidden_layers: layers,
        max_position_embeddings: positions,
        layer_norm_eps: 1e-6,
        projection_size: None,
        hidden_act: None,
    }
}

fn synthetic_weights(args: &SigLipTextArgs, seed: u64) -> WeightMap {
    let mut rng = Lcg(seed);
    let mut weights = WeightMap::new();
    let hidden = args.hidden_size as i32;
    let intermediate = args.intermediate_size as i32;
    rng.insert(
        &mut weights,
        "text_model.embeddings.token_embedding.weight",
        &[args.vocab_size as i32, hidden],
        0.5,
        0.0,
    );
    rng.insert(
        &mut weights,
        "text_model.embeddings.position_embedding.weight",
        &[args.max_position_embeddings as i32, hidden],
        0.2,
        0.0,
    );
    for index in 0..args.num_hidden_layers {
        let prefix = format!("text_model.encoder.layers.{index}");
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            let base = format!("{prefix}.self_attn.{projection}");
            rng.insert(
                &mut weights,
                &format!("{base}.weight"),
                &[hidden, hidden],
                0.3,
                0.0,
            );
            rng.insert(&mut weights, &format!("{base}.bias"), &[hidden], 0.1, 0.0);
        }
        for norm in ["layer_norm1", "layer_norm2"] {
            let base = format!("{prefix}.{norm}");
            rng.insert(&mut weights, &format!("{base}.weight"), &[hidden], 0.1, 1.0);
            rng.insert(&mut weights, &format!("{base}.bias"), &[hidden], 0.1, 0.0);
        }
        rng.insert(
            &mut weights,
            &format!("{prefix}.mlp.fc1.weight"),
            &[intermediate, hidden],
            0.3,
            0.0,
        );
        rng.insert(
            &mut weights,
            &format!("{prefix}.mlp.fc1.bias"),
            &[intermediate],
            0.1,
            0.0,
        );
        rng.insert(
            &mut weights,
            &format!("{prefix}.mlp.fc2.weight"),
            &[hidden, intermediate],
            0.3,
            0.0,
        );
        rng.insert(
            &mut weights,
            &format!("{prefix}.mlp.fc2.bias"),
            &[hidden],
            0.1,
            0.0,
        );
    }
    rng.insert(
        &mut weights,
        "text_model.final_layer_norm.weight",
        &[hidden],
        0.1,
        1.0,
    );
    rng.insert(
        &mut weights,
        "text_model.final_layer_norm.bias",
        &[hidden],
        0.1,
        0.0,
    );
    rng.insert(
        &mut weights,
        "text_model.head.weight",
        &[hidden, hidden],
        0.3,
        0.0,
    );
    rng.insert(&mut weights, "text_model.head.bias", &[hidden], 0.1, 0.0);
    weights
}

fn synthetic_tower(positions: usize, layers: usize) -> (SigLipTextModel, SigLipTextArgs) {
    let args = synthetic_args(positions, layers);
    let weights = synthetic_weights(&args, 0x5161_117E_7700_0001);
    let model = SigLipTextModel::from_weights(&weights, &args, 64, 4).unwrap();
    (model, args)
}

fn read(array: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(array);
    mlxcel_core::utils::array_to_vec_f32(array)
}

fn ids_array(ids: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32])
}

fn embed_ids(model: &SigLipTextModel, ids: &[i32]) -> Vec<f32> {
    let input_ids = ids_array(ids);
    let ones = vec![1i32; ids.len()];
    let mask = mlxcel_core::from_slice_i32(&ones, &[1, ids.len() as i32]);
    let output = model
        .embed(&EmbeddingBatch {
            input_ids: &input_ids,
            attention_mask: &mask,
            token_type_ids: None,
            images: None,
        })
        .unwrap();
    read(&output.embeddings)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm(a) * norm(b);
    if denom > 0.0 { dot / denom } else { 0.0 }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Locate a downloaded checkpoint: the mlxcel store, then the HuggingFace
/// cache, then `<repo>/models/<name>`. `None` skips the gate.
fn local_checkpoint(repo_id: &str) -> Option<PathBuf> {
    let candidates = [
        crate::downloader::model_dir(repo_id),
        crate::downloader::hf_cache_snapshot(repo_id, None),
        Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join(crate::downloader::repo_basename(repo_id)),
        ),
    ];
    let found = candidates
        .into_iter()
        .flatten()
        .find(|dir| dir.join("config.json").is_file());
    if found.is_none() {
        eprintln!(
            "skipping real-checkpoint gate: {repo_id} not present (mlxcel download {repo_id})"
        );
    }
    found
}

// ------------------------------------------------------------ unit tests

#[test]
fn text_config_defaults_match_the_reference_config() {
    // The base checkpoint declares only five keys; everything else has to
    // come from the reference SiglipTextConfig defaults.
    let args = SigLipTextArgs::from_config(&json!({
        "model_type": "siglip",
        "text_config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "model_type": "siglip_text_model",
            "num_attention_heads": 12,
            "vocab_size": 32000
        },
        "vision_config": { "model_type": "siglip_vision_model", "patch_size": 16 }
    }))
    .unwrap();
    assert_eq!(args, SigLipTextArgs::default());
    assert_eq!(args.vocab_size, 32_000);
    assert_eq!(args.hidden_size, 768);
    assert_eq!(args.intermediate_size, 3_072);
    assert_eq!(args.num_attention_heads, 12);
    assert_eq!(args.num_hidden_layers, 12);
    assert_eq!(args.max_position_embeddings, 64);
    assert!((args.layer_norm_eps - 1e-6).abs() <= 1e-12);
    assert_eq!(args.projection_size(), 768, "defaults to hidden_size");
    assert_eq!(
        args.activation(),
        VisionMlpActivation::PytorchTanh,
        "the text tower defaults to gelu_pytorch_tanh, not the vision exact-erf default"
    );
}

#[test]
fn text_config_overrides_are_read_including_projection_and_activation() {
    let args = SigLipTextArgs::from_config(&json!({
        "text_config": {
            "vocab_size": 250_000,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "max_position_embeddings": 16,
            "layer_norm_eps": 1e-5,
            "projection_size": 512,
            "hidden_act": "gelu"
        }
    }))
    .unwrap();
    assert_eq!(args.vocab_size, 250_000);
    assert_eq!(args.hidden_size, 1152);
    assert_eq!(args.num_hidden_layers, 27);
    assert_eq!(args.max_position_embeddings, 16);
    assert_eq!(args.projection_size(), 512);
    assert_eq!(args.activation(), VisionMlpActivation::Exact);
    assert_eq!(args.hidden_act, Some(VisionHiddenActivation::ExactGelu));

    // An explicit null falls back to the reference default, not to the
    // vision-side exact-erf default.
    let nulled =
        SigLipTextArgs::from_config(&json!({ "text_config": { "hidden_act": null } })).unwrap();
    assert_eq!(nulled.activation(), VisionMlpActivation::PytorchTanh);

    // A checkpoint that declares the text fields at the top level (no
    // `text_config` wrapper) is read as well.
    let flat =
        SigLipTextArgs::from_config(&json!({ "hidden_size": 64, "num_hidden_layers": 2 })).unwrap();
    assert_eq!(flat.hidden_size, 64);
    assert_eq!(flat.num_hidden_layers, 2);
}

#[test]
fn sanitize_drops_vision_and_logit_keys() {
    let _guard = test_guard::lock();
    let mut weights = WeightMap::new();
    let mut rng = Lcg(7);
    for key in [
        "text_model.embeddings.token_embedding.weight",
        "text_model.head.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.head.probe",
        "logit_scale",
        "logit_bias",
        "text_model.embeddings.position_ids",
    ] {
        rng.insert(&mut weights, key, &[2], 1.0, 0.0);
    }
    let kept = sanitize_siglip_text_weights(weights);
    let mut names: Vec<&String> = kept.keys().collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "text_model.embeddings.token_embedding.weight",
            "text_model.head.weight"
        ]
    );
}

#[test]
fn pooling_takes_the_last_position_and_not_cls_or_mean() {
    let _guard = test_guard::lock();
    let (model, args) = synthetic_tower(12, 2);
    let ids: Vec<i32> = vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, EOS_ID, EOS_ID];
    let pooled = embed_ids(&model, &ids);
    assert_eq!(pooled.len(), args.projection_size());

    // The returned hidden states are the pre-pooling `[B, L, D]` block; the
    // pooled vector must be `head(h[:, L - 1, :])`, so the alternatives an
    // implementation could plausibly have used must all disagree.
    let input_ids = ids_array(&ids);
    let hidden = read(&model.encode(&input_ids).unwrap());
    let width = args.hidden_size;
    let length = ids.len();
    let row = |index: usize| hidden[index * width..(index + 1) * width].to_vec();

    let project = |values: &[f32]| {
        let array = mlxcel_core::from_slice_f32(values, &[1, width as i32]);
        read(&model.head.forward(&array))
    };
    assert!(
        max_abs_diff(&pooled, &project(&row(length - 1))) <= 1e-5,
        "pooled vector is the projected last position"
    );
    assert!(
        max_abs_diff(&pooled, &project(&row(0))) > 1e-3,
        "pooling must not be cls"
    );
    let mean: Vec<f32> = (0..width)
        .map(|d| (0..length).map(|t| hidden[t * width + d]).sum::<f32>() / length as f32)
        .collect();
    assert!(
        max_abs_diff(&pooled, &project(&mean)) > 1e-3,
        "pooling must not be mean"
    );
}

#[test]
fn every_position_reaches_the_pooled_slot() {
    let _guard = test_guard::lock();
    let (model, _) = synthetic_tower(12, 2);
    let base: Vec<i32> = vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14, EOS_ID, EOS_ID];
    let reference = embed_ids(&model, &base);

    // Attention is unmasked and bidirectional, so a change at the first
    // position must move the vector pooled at the last position.
    let mut first_changed = base.clone();
    first_changed[0] = 21;
    assert!(max_abs_diff(&reference, &embed_ids(&model, &first_changed)) > 1e-4);

    // So must a change at position 9, the last non-EOS token.
    let mut ninth_changed = base.clone();
    ninth_changed[9] = 22;
    assert!(max_abs_diff(&reference, &embed_ids(&model, &ninth_changed)) > 1e-4);

    // Re-running the identical input reproduces the vector, which is what the
    // self-consistency gate below asserts on the real checkpoint.
    assert!(max_abs_diff(&reference, &embed_ids(&model, &base)) <= 1e-6);
}

#[test]
fn trait_surface_reports_fixed_width_padding_and_last_token_pooling() {
    let _guard = test_guard::lock();
    let (model, _) = synthetic_tower(12, 1);
    assert_eq!(model.pad_to_max_length(), Some(12));
    assert_eq!(model.default_pooling(), PoolingMode::LastToken);
    assert_eq!(model.embedding_dim(), 8);
    assert!(model.normalize());
    assert!(!model.multi_vector());
    assert!(!model.supports_images());
    assert!(!model.needs_token_type_ids());
    assert_eq!(
        model.format_text("a photo of a cat", Some("query")),
        "a photo of a cat"
    );
}

#[test]
fn encode_rejects_more_tokens_than_learned_positions() {
    let _guard = test_guard::lock();
    let (model, _) = synthetic_tower(12, 1);
    let too_long = vec![3i32; 13];
    let ids = ids_array(&too_long);
    // `MlxArray` has no `Debug`, so `unwrap_err` is not available here.
    let err = match model.encode(&ids) {
        Ok(_) => panic!("expected a learned-position overflow error"),
        Err(err) => err,
    };
    assert!(err.contains("12 learned positions"), "{err}");

    // A shorter batch is accepted: the pooled slot is simply the last column.
    assert!(model.encode(&ids_array(&[3, 4, EOS_ID])).is_ok());
}

#[test]
fn embed_rejects_image_inputs() {
    let _guard = test_guard::lock();
    let (model, _) = synthetic_tower(12, 1);
    let input_ids = ids_array(&[3, 4, EOS_ID]);
    let mask = mlxcel_core::from_slice_i32(&[1, 1, 1], &[1, 3]);
    let image = crate::embeddings::model::ImageInput {
        image: image::DynamicImage::new_rgb8(2, 2),
    };
    let images = [image];
    // `EmbeddingOutput` has no `Debug`, so `unwrap_err` is not available here.
    let err = match model.embed(&EmbeddingBatch {
        input_ids: &input_ids,
        attention_mask: &mask,
        token_type_ids: None,
        images: Some(&images),
    }) {
        Ok(_) => panic!("expected an image-input rejection"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("does not accept image inputs"), "{err}");
}

// ------------------------------------------------- real-checkpoint gates

#[test]
fn siglip_base_detects_pads_to_64_and_keeps_trailing_eos() {
    let _guard = test_guard::lock();
    let Some(dir) = local_checkpoint(SIGLIP_BASE) else {
        return;
    };
    assert_eq!(get_model_type(&dir).unwrap(), ModelType::SiglipText);

    let tokenizer = strip_padding_and_truncation(load_tokenizer(&dir).unwrap());
    let pad_id = crate::embeddings::limits::resolve_pad_token_id(&dir, &tokenizer);
    assert_eq!(
        pad_id, EOS_ID as u32,
        "pad_token is </s>, the same id as EOS"
    );

    let long = "a photo of a cat ".repeat(40);
    let batch = encode_batch(
        &tokenizer,
        &["a photo of a cat", long.as_str()],
        EncodeOptions {
            add_special_tokens: true,
            max_length: 64,
            with_token_type_ids: false,
        },
        pad_id,
        Some(64),
    )
    .unwrap();
    assert_eq!(
        batch.width, 64,
        "every row is padded to the 64 learned positions"
    );
    assert_eq!(batch.batch, 2);

    let short = &batch.input_ids[..64];
    let short_len = batch.token_counts[0];
    assert!(short_len < 64);
    assert_eq!(short[short_len - 1], EOS_ID, "the tokenizer appends </s>");
    assert!(
        short[short_len..].iter().all(|&id| id == EOS_ID),
        "padding uses the same </s> id, so position 63 always holds it"
    );

    let cut = &batch.input_ids[64..];
    assert_eq!(batch.token_counts[1], 64, "a 100-token input is cut to 64");
    assert_eq!(cut[63], EOS_ID, "truncation keeps the trailing </s>");
    assert_ne!(cut[62], EOS_ID, "the 63 slots before it are real tokens");
}

#[test]
fn siglip_base_text_tower_passes_the_embedding_gate() {
    let _guard = test_guard::lock();
    let Some(dir) = local_checkpoint(SIGLIP_BASE) else {
        return;
    };
    let loaded = load_embedding_model(&dir).unwrap();
    assert_eq!(loaded.model_type, ModelType::SiglipText);
    assert_eq!(
        loaded.limits.dim, 768,
        "projection_size defaults to hidden_size"
    );
    assert_eq!(loaded.limits.max_length, 64);
    assert!(!loaded.limits.multi_vector);
    assert_eq!(loaded.pad_token_id, EOS_ID as u32);
    assert_eq!(loaded.vocab_size, 32_000);
    assert_eq!(loaded.model.pad_to_max_length(), Some(64));

    let engine = EmbeddingEngine::new(loaded, 16);
    let prompts: Vec<String> = [
        "a photo of a cat",
        "a photo of a kitten",
        "a photo of a cat",
        "a diagram of a car engine",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let reply = engine
        .embed_texts(&prompts, &EmbedOptions::default())
        .unwrap();
    assert_eq!(reply.vectors.len(), 4);
    for vector in &reply.vectors {
        assert_eq!(vector.shape, vec![768]);
        let norm = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() <= 1e-5,
            "vectors are L2-normalized, got {norm}"
        );
    }
    assert!(
        reply.prompt_tokens < 4 * 64,
        "usage counts real tokens, not the 64 padded slots (got {})",
        reply.prompt_tokens
    );

    let cat = &reply.vectors[0].values;
    let kitten = &reply.vectors[1].values;
    let cat_again = &reply.vectors[2].values;
    let engine_diagram = &reply.vectors[3].values;

    let identical = cosine(cat, cat_again);
    assert!(
        (identical - 1.0).abs() <= 1e-6,
        "identical inputs must score 1.0, got {identical}"
    );
    let related = cosine(cat, kitten);
    let unrelated = cosine(cat, engine_diagram);
    assert!(
        related - unrelated >= 0.1,
        "cat/kitten {related} must beat cat/engine {unrelated} by at least 0.1"
    );
    // SigLIP's text tower is trained contrastively against images, never
    // against other texts, so its text-only cosines sit on a high anisotropic
    // floor: over six sentences spanning animals, machinery, finance, food
    // and physics, the fourteen unrelated pairs measured 0.519 to 0.725
    // (mean 0.653) while cat/kitten reached 0.966. An absolute "unrelated
    // below 0.5" threshold is therefore unreachable for this family and would
    // only encode a misunderstanding of its geometry; the margin above is
    // what discriminates. The bound below is a loose sanity ceiling that a
    // genuinely broken tower (one that collapses every input onto the padded
    // `</s>` row, say) would still blow through.
    assert!(
        unrelated < related - 0.1 && unrelated < 0.9,
        "unrelated sentences must stay clearly below the related pair, got {unrelated} against {related}"
    );

    // Absolute parity against an independent NumPy implementation of the
    // reference forward pass (tokenize, truncate to 63 + `</s>`, pad to 64,
    // token + position embedding, 12 unmasked pre-norm blocks, final
    // LayerNorm, `head`, L2 normalize) run over the same safetensors.
    //
    // This engine path reproduces the reference to 2.7e-8 on these twelve
    // components, identically across three repeated guarded runs. The bound
    // is nevertheless 2e-4, because the same vector computed through the
    // `mlxcel embed` process rather than through the test binary lands 4.5e-5
    // away from the reference at its worst component: MLX picks kernels per
    // process, and each path is bit-stable on its own but not identical to
    // the other. The bound has to clear that spread, and 2e-4 is roughly four
    // times it.
    const CAT_REFERENCE_PREFIX: [f32; 12] = [
        0.016_063_286,
        -0.000_183_518,
        -0.012_627_865,
        -0.017_693_21,
        -0.012_078_869,
        0.022_338_506,
        -0.026_179_017,
        0.020_287_77,
        -0.012_506_417,
        -0.030_153_267,
        -0.043_187_555,
        -0.014_645_457,
    ];
    let parity = max_abs_diff(&cat[..CAT_REFERENCE_PREFIX.len()], &CAT_REFERENCE_PREFIX);
    assert!(
        parity <= 2e-4,
        "drift {parity} from the NumPy reference embedding of \"a photo of a cat\""
    );

    // A padded batch and a single input take the same fixed-width path, so
    // the two must agree far tighter than the 1e-3 the gate allows.
    let single = engine
        .embed_texts(&[prompts[0].clone()], &EmbedOptions::default())
        .unwrap();
    assert_eq!(single.vectors.len(), 1);
    let drift = max_abs_diff(cat, &single.vectors[0].values);
    assert!(
        drift <= 1e-3,
        "batch and single-input drift {drift} exceeds 1e-3"
    );
    assert!(
        (cosine(cat, &single.vectors[0].values) - 1.0).abs() <= 1e-6,
        "batch and single-input vectors must be collinear"
    );

    // Printed so a repeated run can be checked for spread rather than for a
    // single pass: a concurrency-corrupted forward shows up as movement in
    // `identical`, not as an outright failure.
    eprintln!(
        "SIGLIP_GATE identical={identical:.9} related={related:.6} unrelated={unrelated:.6} \
         margin={:.6} batch_vs_single_drift={drift:.3e} numpy_parity={parity:.3e} \
         prompt_tokens={}",
        related - unrelated,
        reply.prompt_tokens
    );
}
