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

//! ColIdefics3 tests.
//!
//! A synthetic checkpoint directory is materialized on disk (config,
//! preprocessor config, a minimal `tokenizer.json` and one safetensors
//! shard) so every test runs the real [`ColIdefics3Model::load`] path,
//! including the `1_Dense` override and the tile-marker expansion, rather
//! than a hand-assembled struct.
//!
//! Every test that drives MLX takes
//! [`crate::models::embedding_test_support::mlx_test_guard`]: the trait is
//! single-thread by contract and libtest is not.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{ColIdefics3Model, IMAGE_DOCUMENT_PROMPT};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, ImageInput};
use crate::models::col_late_interaction::QUERY_AUGMENTATION_TOKENS;
use crate::models::embedding_test_support::{
    Rng, err_string, local_checkpoint, max_abs_diff, mlx_test_guard, temp_dir, to_vec,
    write_f32_safetensors,
};

/// Tiny geometry: 8 px images, 4 px patches and a pixel-shuffle factor of 2
/// give a 2x2 patch grid that compresses to exactly one image token.
const TEXT_HIDDEN: i32 = 16;
const TEXT_INTERMEDIATE: i32 = 32;
const TEXT_HEADS: i32 = 2;
const TEXT_KV_HEADS: i32 = 1;
const HEAD_DIM: i32 = 8;
const VOCAB: i32 = 64;
const VISION_HIDDEN: i32 = 8;
const VISION_INTERMEDIATE: i32 = 16;
const VISION_HEADS: i32 = 2;
const PATCH_SIZE: i32 = 4;
const IMAGE_SIZE: i32 = 8;
const SCALE_FACTOR: i32 = 2;
/// Deliberately not 128 so a test can tell the configured width from the
/// family default.
const EMBED_DIM: i32 = 4;
/// The `<image>` placeholder id inside the synthetic vocabulary.
const IMAGE_TOKEN_ID: i32 = 9;

const COLSMOLVLM_BASE: &str = "vidore/ColSmolVLM-Instruct-256M-base";

fn tiny_config(mask_non_image_embeddings: bool) -> Value {
    json!({
        "model_type": "idefics3",
        "architectures": ["ColIdefics3"],
        "image_token_id": IMAGE_TOKEN_ID,
        "scale_factor": SCALE_FACTOR,
        "embedding_dim": EMBED_DIM,
        "mask_non_image_embeddings": mask_non_image_embeddings,
        "tie_word_embeddings": false,
        "vocab_size": VOCAB,
        "text_config": {
            "model_type": "llama",
            "hidden_size": TEXT_HIDDEN,
            "intermediate_size": TEXT_INTERMEDIATE,
            "num_hidden_layers": 2,
            "num_attention_heads": TEXT_HEADS,
            "num_key_value_heads": TEXT_KV_HEADS,
            "head_dim": HEAD_DIM,
            "rms_norm_eps": 1e-5,
            "rope_theta": 100000.0,
            "vocab_size": VOCAB,
            "tie_word_embeddings": false,
            "max_position_embeddings": 512
        },
        "vision_config": {
            "model_type": "idefics3",
            "hidden_size": VISION_HIDDEN,
            "intermediate_size": VISION_INTERMEDIATE,
            "num_hidden_layers": 1,
            "num_attention_heads": VISION_HEADS,
            "patch_size": PATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "layer_norm_eps": 1e-6
        }
    })
}

/// One synthetic tensor: `(key, shape, values)`.
type WeightSpec = (String, Vec<i32>, Vec<f32>);

/// Deterministic weights keyed exactly the way the published ColIdefics3
/// checkpoint stores them (`model.text_model.*`, `model.vision_model.*`,
/// `model.connector.*`, root `linear.*`).
fn tiny_weight_specs() -> Vec<WeightSpec> {
    let mut rng = Rng::new(0xC0_1D_EF_13);
    let mut specs: Vec<WeightSpec> = Vec::new();
    let mut push = |rng: &mut Rng, key: String, shape: Vec<i32>, scale: f32| {
        let count: i32 = shape.iter().product();
        let values = rng.values(count as usize, scale);
        specs.push((key, shape, values));
    };

    // Llama text backbone.
    let q_out = TEXT_HEADS * HEAD_DIM;
    let kv_out = TEXT_KV_HEADS * HEAD_DIM;
    push(
        &mut rng,
        "model.text_model.embed_tokens.weight".to_string(),
        vec![VOCAB, TEXT_HIDDEN],
        0.5,
    );
    for layer in 0..2 {
        let p = format!("model.text_model.layers.{layer}");
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![q_out, TEXT_HIDDEN]),
            ("self_attn.k_proj.weight", vec![kv_out, TEXT_HIDDEN]),
            ("self_attn.v_proj.weight", vec![kv_out, TEXT_HIDDEN]),
            ("self_attn.o_proj.weight", vec![TEXT_HIDDEN, q_out]),
            ("mlp.gate_proj.weight", vec![TEXT_INTERMEDIATE, TEXT_HIDDEN]),
            ("mlp.up_proj.weight", vec![TEXT_INTERMEDIATE, TEXT_HIDDEN]),
            ("mlp.down_proj.weight", vec![TEXT_HIDDEN, TEXT_INTERMEDIATE]),
        ] {
            push(&mut rng, format!("{p}.{name}"), shape, 0.2);
        }
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            push(
                &mut rng,
                format!("{p}.{norm}.weight"),
                vec![TEXT_HIDDEN],
                0.1,
            );
        }
    }
    push(
        &mut rng,
        "model.text_model.norm.weight".to_string(),
        vec![TEXT_HIDDEN],
        0.1,
    );

    // SigLIP vision tower. The conv weight is stored MLX-side
    // (`[out, kH, kW, in]`), which the loader detects by shape.
    let vp = "model.vision_model";
    push(
        &mut rng,
        format!("{vp}.embeddings.patch_embedding.weight"),
        vec![VISION_HIDDEN, PATCH_SIZE, PATCH_SIZE, 3],
        0.2,
    );
    push(
        &mut rng,
        format!("{vp}.embeddings.patch_embedding.bias"),
        vec![VISION_HIDDEN],
        0.1,
    );
    let num_patches = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE);
    push(
        &mut rng,
        format!("{vp}.embeddings.position_embedding.weight"),
        vec![num_patches, VISION_HIDDEN],
        0.2,
    );
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        push(
            &mut rng,
            format!("{vp}.encoder.layers.0.self_attn.{proj}.weight"),
            vec![VISION_HIDDEN, VISION_HIDDEN],
            0.2,
        );
        push(
            &mut rng,
            format!("{vp}.encoder.layers.0.self_attn.{proj}.bias"),
            vec![VISION_HIDDEN],
            0.1,
        );
    }
    for norm in ["layer_norm1", "layer_norm2"] {
        for suffix in ["weight", "bias"] {
            push(
                &mut rng,
                format!("{vp}.encoder.layers.0.{norm}.{suffix}"),
                vec![VISION_HIDDEN],
                0.1,
            );
        }
    }
    push(
        &mut rng,
        format!("{vp}.encoder.layers.0.mlp.fc1.weight"),
        vec![VISION_INTERMEDIATE, VISION_HIDDEN],
        0.2,
    );
    push(
        &mut rng,
        format!("{vp}.encoder.layers.0.mlp.fc1.bias"),
        vec![VISION_INTERMEDIATE],
        0.1,
    );
    push(
        &mut rng,
        format!("{vp}.encoder.layers.0.mlp.fc2.weight"),
        vec![VISION_HIDDEN, VISION_INTERMEDIATE],
        0.2,
    );
    push(
        &mut rng,
        format!("{vp}.encoder.layers.0.mlp.fc2.bias"),
        vec![VISION_HIDDEN],
        0.1,
    );
    for suffix in ["weight", "bias"] {
        push(
            &mut rng,
            format!("{vp}.post_layernorm.{suffix}"),
            vec![VISION_HIDDEN],
            0.1,
        );
    }

    // Pixel-shuffle connector: 2x2 patches collapse into one token of
    // `VISION_HIDDEN * SCALE_FACTOR^2` channels.
    push(
        &mut rng,
        "model.connector.modality_projection.proj.weight".to_string(),
        vec![TEXT_HIDDEN, VISION_HIDDEN * SCALE_FACTOR * SCALE_FACTOR],
        0.2,
    );

    // The untrained root projection.
    push(
        &mut rng,
        "linear.weight".to_string(),
        vec![EMBED_DIM, TEXT_HIDDEN],
        0.4,
    );
    push(&mut rng, "linear.bias".to_string(), vec![EMBED_DIM], 0.2);

    specs
}

/// A `WordLevel` tokenizer whose vocabulary holds the exact marker strings
/// the tile expander asks for, so `load` and `expand_image_tokens` run
/// their real paths without a downloaded checkpoint.
fn write_tokenizer(dir: &Path) {
    let vocab = json!({
        "<unk>": 0,
        "<fake_token_around_image><global-img>": 1,
        "<fake_token_around_image>": 2,
        "\n": 3,
        "\n<fake_token_around_image><global-img>": 4,
        "hello": 5
    });
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    });
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string(&tokenizer).unwrap(),
    )
    .unwrap();
}

/// Materialize the synthetic checkpoint. `dense_override` also writes a
/// `1_Dense/model.safetensors` whose projection must win.
fn synthetic_checkpoint(name: &str, config: &Value, dense_override: bool) -> PathBuf {
    let dir = temp_dir(name);
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(config).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("preprocessor_config.json"),
        serde_json::to_string(&json!({
            "image_processor_type": "Idefics3ImageProcessor",
            "do_image_splitting": false,
            "size": {"longest_edge": IMAGE_SIZE * 4},
            "max_image_size": {"longest_edge": IMAGE_SIZE},
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        }))
        .unwrap(),
    )
    .unwrap();
    write_tokenizer(&dir);
    write_f32_safetensors(&dir.join("model.safetensors"), &tiny_weight_specs());
    if dense_override {
        std::fs::create_dir_all(dir.join("1_Dense")).unwrap();
        let count = (EMBED_DIM * TEXT_HIDDEN) as usize;
        write_f32_safetensors(
            &dir.join("1_Dense").join("model.safetensors"),
            &[
                (
                    "linear.weight".to_string(),
                    vec![EMBED_DIM, TEXT_HIDDEN],
                    vec![0.125; count],
                ),
                (
                    "linear.bias".to_string(),
                    vec![EMBED_DIM],
                    vec![0.0; EMBED_DIM as usize],
                ),
            ],
        );
    }
    dir
}

fn load_synthetic(name: &str, dense_override: bool) -> (PathBuf, ColIdefics3Model) {
    let config = tiny_config(false);
    let dir = synthetic_checkpoint(name, &config, dense_override);
    let model = ColIdefics3Model::load(&dir, &config).expect("the synthetic checkpoint loads");
    (dir, model)
}

/// Run one padded batch and read the `[B, L, EMBED_DIM]` result back.
fn embed_rows(
    model: &ColIdefics3Model,
    rows: &[Vec<i32>],
    images: Option<&[ImageInput]>,
) -> Vec<f32> {
    let width = rows.iter().map(Vec::len).max().unwrap();
    let mut ids = Vec::new();
    let mut mask = Vec::new();
    for row in rows {
        ids.extend(row.iter().copied());
        ids.extend(std::iter::repeat_n(0, width - row.len()));
        mask.extend(std::iter::repeat_n(1, row.len()));
        mask.extend(std::iter::repeat_n(0, width - row.len()));
    }
    let batch = rows.len() as i32;
    let ids = mlxcel_core::from_slice_i32(&ids, &[batch, width as i32]);
    let mask = mlxcel_core::from_slice_i32(&mask, &[batch, width as i32]);
    let out = model
        .embed(&EmbeddingBatch {
            input_ids: &ids,
            attention_mask: &mask,
            token_type_ids: None,
            images,
        })
        .expect("forward succeeds");
    assert_eq!(
        mlxcel_core::array_shape(&out.embeddings),
        vec![batch, width as i32, EMBED_DIM]
    );
    mlxcel_core::eval(&out.embeddings);
    to_vec(&out.embeddings)
}

fn row_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[test]
fn rejects_mask_non_image_embeddings_true() {
    let dir = temp_dir("colidefics3_masked");
    let config = tiny_config(true);
    let error = err_string(ColIdefics3Model::load(&dir, &config));
    assert!(
        error.contains("mask_non_image_embeddings"),
        "the error must name the rejected flag: {error}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn query_format_appends_ten_augmentation_tokens() {
    let _guard = mlx_test_guard();
    let (dir, model) = load_synthetic("colidefics3_format", false);

    let query = model.format_text("What was the total revenue in 2023?", None);
    assert!(query.starts_with("Query: What was the total revenue in 2023?"));
    assert_eq!(
        query.matches("<end_of_utterance>").count(),
        QUERY_AUGMENTATION_TOKENS
    );

    // An empty text is the engine's image path.
    assert_eq!(model.format_text("", None), IMAGE_DOCUMENT_PROMPT);
    // The instruction is not a query wrapper for this family.
    assert_eq!(
        model.format_text("abc", Some("ignored")),
        model.format_text("abc", None)
    );

    assert!(model.multi_vector());
    assert!(model.supports_images());
    assert_eq!(model.embedding_dim(), EMBED_DIM as usize);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn sanitize_prefers_1_dense_projection() {
    let _guard = mlx_test_guard();
    // The `1_Dense` weight is a constant 0.125 with a zero bias, so every
    // projected component is the same value and each normalized row is the
    // constant unit vector. The root projection is random and could not
    // produce that.
    let (dir, model) = load_synthetic("colidefics3_dense", true);
    let values = embed_rows(&model, &[vec![1, 2, 3, 4]], None);
    let expected = 1.0 / (EMBED_DIM as f32).sqrt();
    for (index, value) in values.iter().enumerate() {
        assert!(
            (value.abs() - expected).abs() < 1e-4,
            "component {index} is {value}, expected +/-{expected} from the 1_Dense projection"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn padding_rows_are_zero_and_real_rows_unit_norm() {
    let _guard = mlx_test_guard();
    let (dir, model) = load_synthetic("colidefics3_padding", false);

    let short = vec![7, 11, 13];
    let long = vec![7, 11, 13, 17, 19, 23];
    let padded = embed_rows(&model, &[long.clone(), short.clone()], None);
    let width = long.len();
    let dim = EMBED_DIM as usize;

    for row in 0..width {
        let slice = &padded[row * dim..(row + 1) * dim];
        assert!(
            (row_norm(slice) - 1.0).abs() < 1e-5,
            "row {row} of the full-length input has norm {}",
            row_norm(slice)
        );
    }
    let second = &padded[width * dim..];
    for row in 0..short.len() {
        let slice = &second[row * dim..(row + 1) * dim];
        assert!((row_norm(slice) - 1.0).abs() < 1e-5, "real row {row}");
    }
    for row in short.len()..width {
        let slice = &second[row * dim..(row + 1) * dim];
        assert!(
            slice.iter().all(|&v| v == 0.0),
            "padding row {row} is not zeroed: {slice:?}"
        );
    }

    // Self-consistency: the short input embedded alone matches its rows
    // inside the padded batch.
    let alone = embed_rows(&model, std::slice::from_ref(&short), None);
    let inside = &second[..short.len() * dim];
    assert!(
        max_abs_diff(&alone, inside) < 1e-3,
        "a padded row drifted from the single-input result by {}",
        max_abs_diff(&alone, inside)
    );

    // And the run is deterministic.
    let again = embed_rows(&model, std::slice::from_ref(&short), None);
    assert_eq!(alone, again, "identical inputs must give identical output");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn causal_prefill_is_causal() {
    let _guard = mlx_test_guard();
    let (dir, model) = load_synthetic("colidefics3_causal", false);

    let length = 96usize;
    let base: Vec<i32> = (0..length)
        .map(|i| (i % (VOCAB as usize - 1)) as i32 + 1)
        .collect();
    let mut changed = base.clone();
    let pivot = 90usize;
    changed[pivot] = (base[pivot] + 7) % VOCAB;
    assert_ne!(base[pivot], changed[pivot]);

    let a = embed_rows(&model, &[base], None);
    let b = embed_rows(&model, &[changed], None);
    let dim = EMBED_DIM as usize;

    let prefix = pivot * dim;
    assert!(
        max_abs_diff(&a[..prefix], &b[..prefix]) < 1e-4,
        "a token at position {pivot} changed an earlier vector, so attention is not causal"
    );
    assert!(
        max_abs_diff(&a[prefix..], &b[prefix..]) > 1e-4,
        "the changed token did not move its own vector, so the test proves nothing"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn image_placeholder_expands_to_one_run_per_tile() {
    let _guard = mlx_test_guard();
    let (dir, model) = load_synthetic("colidefics3_image", false);

    let image = ImageInput {
        image: image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            IMAGE_SIZE as u32,
            IMAGE_SIZE as u32,
            image::Rgb([120, 40, 200]),
        )),
    };
    // `<image>` sits between two ordinary tokens, the way the rendered
    // document prompt places it.
    let ids: Vec<u32> = vec![5, IMAGE_TOKEN_ID as u32, 6];
    let expanded = model
        .expand_image_tokens(&ids, std::slice::from_ref(&image))
        .expect("the placeholder expands");
    // Splitting is off and the image is one tile, so the block is
    // `<fake><global-img>` + one image token + `<fake>`.
    assert_eq!(expanded, vec![5, 1, IMAGE_TOKEN_ID as u32, 2, 6]);
    assert_eq!(
        expanded
            .iter()
            .filter(|&&t| t == IMAGE_TOKEN_ID as u32)
            .count(),
        1,
        "one tile compresses to exactly one image token at this geometry"
    );

    // The expanded row runs through the forward pass with the image merged
    // in, and every row is a unit vector.
    let rows: Vec<Vec<i32>> = vec![expanded.iter().map(|&t| t as i32).collect()];
    let values = embed_rows(&model, &rows, Some(std::slice::from_ref(&image)));
    let dim = EMBED_DIM as usize;
    for row in 0..rows[0].len() {
        let slice = &values[row * dim..(row + 1) * dim];
        assert!(
            (row_norm(slice) - 1.0).abs() < 1e-5,
            "image-prompt row {row} has norm {}",
            row_norm(slice)
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn text_only_embedding_ignores_the_absent_image_list() {
    let _guard = mlx_test_guard();
    let (dir, model) = load_synthetic("colidefics3_textonly", false);
    let with_none = embed_rows(&model, &[vec![3, 4, 5]], None);
    let with_empty = embed_rows(&model, &[vec![3, 4, 5]], Some(&[]));
    assert_eq!(with_none, with_empty);
    std::fs::remove_dir_all(&dir).unwrap();
}

// Real-checkpoint gates. Each soft-skips when the checkpoint is absent,
// the convention `src/embeddings/real_checkpoint_tests.rs` follows.

#[test]
fn real_colsmolvlm_base_loads_and_projects_to_128() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(COLSMOLVLM_BASE) else {
        return;
    };
    let config = crate::embeddings::loader::read_embedding_config(&dir).unwrap();
    let model = ColIdefics3Model::load(&dir, &config).expect("the base checkpoint loads");
    assert_eq!(model.embedding_dim(), 128);
    assert!(model.multi_vector());
    assert!(model.supports_images());
    // 512 px tiles, 16 px patches and a pixel-shuffle factor of 4 give 64
    // feature rows per tile, which is the checkpoint's own `image_seq_len`.
    assert_eq!(model.num_image_token, 64);
    assert_eq!(model.image_token_id, 49_190);

    let query = model.format_text("What was the total revenue in 2023?", None);
    assert!(query.starts_with("Query: "));
    assert_eq!(
        query.matches("<end_of_utterance>").count(),
        QUERY_AUGMENTATION_TOKENS
    );
}
