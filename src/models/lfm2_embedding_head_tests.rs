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

//! LFM2.5-Encoder and LFM2.5-ColBERT gates.
//!
//! The synthetic tests pin head selection: both ColBERT projection layouts
//! (`1_Dense/` folded to `dense.0.*`, and the mlx-community `dense.*`) make
//! the model multi-vector at the projection's width, a second Dense module or
//! a non-Identity `1_Dense/config.json` is a load error, and a padded batch
//! comes back `[B, L, D]` with unit real rows and zero padding rows.
//!
//! The real-checkpoint tests compare against `tests/fixtures/lfm2_bidirectional/`,
//! dumps of the vendored mlx-vlm `lfm2_encoder` / `lfm2_colbert` forward on the
//! same token ids, and soft-skip when a checkpoint is not downloaded.

use std::path::{Path, PathBuf};

use mlxcel_core::weights::WeightMap;
use serde_json::{Value, json};

use super::Lfm2EmbeddingModel;
use super::head::{find_projection, validate_projection};
use super::lfm2_embedding_tests::{HIDDEN, tiny_args, tiny_weights};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel};
use crate::embeddings::pooling::PoolingMode;
use crate::embeddings::{EmbedOptions, EmbeddingEngine, load_embedding_model};
use crate::models::embedding_test_support::{
    Rng, cosine, err_string, ids_array, local_checkpoint, mlx_test_guard, temp_dir, to_vec,
};

/// Width of the published ColBERT projection.
const PROJ_DIM: usize = 128;

/// Per-token cosine the bf16 and f32 checkpoints must reach against mlx-vlm.
const PARITY_COSINE: f32 = 0.999;

/// Per-token cosine floor for the 4-bit build against the bf16 reference:
/// quantization error, not a port defect, sets this one.
const QUANTIZED_COSINE: f32 = 0.95;

const ENCODER: &str = "LiquidAI/LFM2.5-Encoder-350M";
const COLBERT_HF: &str = "LiquidAI/LFM2.5-ColBERT-350M";
const COLBERT_BF16: &str = "mlx-community/LFM2.5-ColBERT-350M-bf16";
const COLBERT_4BIT: &str = "mlx-community/LFM2.5-ColBERT-350M-4bit";

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lfm2_bidirectional")
        .join(name);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&raw).unwrap()
}

fn ids_of(value: &Value) -> Vec<i32> {
    value["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}

fn rows_of(value: &Value) -> Vec<Vec<f32>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

/// Minimum and mean per-row cosine of `ours` (row-major, `width` wide)
/// against `reference`.
fn per_token_cosine(ours: &[f32], width: usize, reference: &[Vec<f32>]) -> (f32, f32) {
    assert_eq!(
        ours.len(),
        reference.len() * width,
        "token count differs from the reference"
    );
    let cosines: Vec<f32> = ours
        .chunks(width)
        .zip(reference)
        .map(|(a, b)| cosine(a, b))
        .collect();
    let min = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    let mean = cosines.iter().sum::<f32>() / cosines.len() as f32;
    (min, mean)
}

/// The tiny backbone plus a `[PROJ_DIM, HIDDEN]` projection at `key`.
fn colbert_weights(key: &str) -> WeightMap {
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    Rng::new(0x2366_00DE).insert(&mut weights, key, &[PROJ_DIM as i32, HIDDEN as i32], 0.3);
    weights
}

fn build_colbert(key: &str) -> Lfm2EmbeddingModel {
    let mut weights = colbert_weights(key);
    super::sanitize_family_weights(&mut weights);
    Lfm2EmbeddingModel::from_weights(weights, tiny_args(), PoolingMode::Cls, true).unwrap()
}

// Head selection.

#[test]
fn both_projection_layouts_load_multi_vector() {
    let _guard = mlx_test_guard();
    // mlx-community stores the projection as a bare `dense.weight`; the HF
    // export keeps it in `1_Dense/`, surfaced as `1_Dense.linear.weight` and
    // folded to `dense.0.weight`. Before this head existed the first loaded
    // as a single-vector CLS embedder with the projection unused.
    // An mlx-vlm re-export names the module `projection` (with or without
    // the `model.` its sanitize adds and then strips).
    for key in [
        "dense.weight",
        "1_Dense.linear.weight",
        "projection.weight",
        "model.projection.weight",
    ] {
        let model = build_colbert(key);
        assert!(
            model.multi_vector(),
            "{key} did not select the ColBERT head"
        );
        assert_eq!(model.embedding_dim(), PROJ_DIM, "{key}");
    }

    // Without a projection it stays the pooled embedder.
    let args = tiny_args();
    let plain = Lfm2EmbeddingModel::from_weights(tiny_weights(&args), args, PoolingMode::Cls, true)
        .unwrap();
    assert!(!plain.multi_vector());
    assert_eq!(plain.embedding_dim(), HIDDEN);
}

#[test]
fn quantized_projection_module_loads_multi_vector() {
    let _guard = mlx_test_guard();
    // An mlx-vlm re-export of a quantized build: `projection.{weight,scales,
    // biases}`, packed along the input axis, so the width check has to read
    // the input from the `scales` grouping. The tiny backbone is widened to 32
    // so one 4-bit group of 32 fits; it stays dense, which the per-linear
    // `scales` detection allows.
    let group_size = 32;
    let mut args = tiny_args();
    args.hidden_size = 32;
    args.quantization = Some(crate::models::lfm2::Quantization {
        group_size,
        bits: 4,
    });
    let mut weights = tiny_weights(&args);
    let dense = Rng::new(0x2366_0004).tensor(&[PROJ_DIM as i32, 32], 0.3);
    let quantized = mlxcel_core::quantize_weights(&dense, group_size, 4);
    weights.insert(
        "projection.weight".to_string(),
        mlxcel_core::quantized_weights_w(&quantized),
    );
    weights.insert(
        "projection.scales".to_string(),
        mlxcel_core::quantized_weights_scales(&quantized),
    );
    if mlxcel_core::quantized_weights_has_biases(&quantized) {
        weights.insert(
            "projection.biases".to_string(),
            mlxcel_core::quantized_weights_biases(&quantized),
        );
    }
    super::sanitize_family_weights(&mut weights);
    let model = Lfm2EmbeddingModel::from_weights(weights, args, PoolingMode::Cls, true).unwrap();
    assert!(model.multi_vector());
    assert_eq!(model.embedding_dim(), PROJ_DIM);

    let out = model
        .embed(&EmbeddingBatch {
            input_ids: &ids_array(&[1, 2, 3, 4], 1, 4),
            attention_mask: &ids_array(&[1, 1, 1, 1], 1, 4),
            token_type_ids: None,
            images: None,
        })
        .unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&out.embeddings),
        vec![1, 4, PROJ_DIM as i32]
    );
    for row in to_vec(&out.embeddings).chunks(PROJ_DIM) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "quantized head row norm {norm}");
    }
}

#[test]
fn padded_batch_returns_unit_rows_and_zero_padding() {
    let _guard = mlx_test_guard();
    let model = build_colbert("dense.weight");
    let (batch, length, real) = (2_usize, 9_usize, 5_usize);
    let ids: Vec<i32> = (0..(batch * length) as i32)
        .map(|i| (i * 7 + 3) % 32)
        .collect();
    let mask: Vec<i32> = (0..batch * length)
        .map(|i| i32::from(i >= length || i % length < real))
        .collect();
    let out = model
        .embed(&EmbeddingBatch {
            input_ids: &ids_array(&ids, batch as i32, length as i32),
            attention_mask: &ids_array(&mask, batch as i32, length as i32),
            token_type_ids: None,
            images: None,
        })
        .unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&out.embeddings),
        vec![batch as i32, length as i32, PROJ_DIM as i32]
    );
    let values = to_vec(&out.embeddings);
    for (index, row) in values.chunks(PROJ_DIM).enumerate() {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if mask[index] == 1 {
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "real row {index} has norm {norm}"
            );
        } else {
            assert_eq!(norm, 0.0, "padding row {index} is not zero");
        }
    }

    // The padded row equals the same row run alone: the conv input stays
    // zeroed at padding positions on this head too.
    let alone = model
        .embed(&EmbeddingBatch {
            input_ids: &ids_array(&ids[..real], 1, real as i32),
            attention_mask: &ids_array(&vec![1; real], 1, real as i32),
            token_type_ids: None,
            images: None,
        })
        .unwrap();
    let alone = to_vec(&alone.embeddings);
    for (t, (a, b)) in alone
        .chunks(PROJ_DIM)
        .zip(values.chunks(PROJ_DIM))
        .enumerate()
    {
        let c = cosine(a, b);
        assert!(c > 0.99999, "token {t} drifted to cosine {c} under padding");
    }
}

#[test]
fn more_than_one_projection_is_refused() {
    let mut rng = Rng::new(0x2366_0002);
    let mut stacked = WeightMap::new();
    rng.insert(&mut stacked, "dense.0.weight", &[4, 4], 0.1);
    rng.insert(&mut stacked, "dense.1.weight", &[4, 4], 0.1);
    assert!(err_string(find_projection(&stacked, 0)).contains("Dense"));
    assert!(find_projection(&WeightMap::new(), 2).is_err());

    let mut both = WeightMap::new();
    rng.insert(&mut both, "dense.0.weight", &[4, 4], 0.1);
    rng.insert(&mut both, "dense.weight", &[4, 4], 0.1);
    assert!(err_string(find_projection(&both, 1)).contains("exactly one"));
    let mut mixed = WeightMap::new();
    rng.insert(&mut mixed, "dense.weight", &[4, 4], 0.1);
    rng.insert(&mut mixed, "projection.weight", &[4, 4], 0.1);
    assert!(err_string(find_projection(&mixed, 0)).contains("exactly one"));

    assert_eq!(find_projection(&WeightMap::new(), 0).unwrap(), None);
}

/// The quantized `projection.*` triple of `quantized_projection_module_loads_multi_vector`,
/// over a 32-wide backbone.
fn quantized_projection_weights() -> (WeightMap, crate::models::lfm2::ModelArgs) {
    let group_size = 32;
    let mut args = tiny_args();
    args.hidden_size = 32;
    args.quantization = Some(crate::models::lfm2::Quantization {
        group_size,
        bits: 4,
    });
    let mut weights = tiny_weights(&args);
    let dense = Rng::new(0x2366_0005).tensor(&[PROJ_DIM as i32, 32], 0.3);
    let quantized = mlxcel_core::quantize_weights(&dense, group_size, 4);
    weights.insert(
        "projection.weight".to_string(),
        mlxcel_core::quantized_weights_w(&quantized),
    );
    weights.insert(
        "projection.scales".to_string(),
        mlxcel_core::quantized_weights_scales(&quantized),
    );
    weights.insert(
        "projection.biases".to_string(),
        mlxcel_core::quantized_weights_biases(&quantized),
    );
    (weights, args)
}

#[test]
fn malformed_projection_tensors_fail_the_load() {
    let _guard = mlx_test_guard();
    // Each of these used to load and then abort the process inside MLX at
    // the first embed request, where the shape error has no `Result` to
    // travel through.
    let load = |weights: WeightMap, args| {
        err_string(Lfm2EmbeddingModel::from_weights(
            weights,
            args,
            PoolingMode::Cls,
            true,
        ))
    };

    let mut short_bias = colbert_weights("dense.weight");
    Rng::new(0x2366_0006).insert(&mut short_bias, "dense.bias", &[PROJ_DIM as i32 - 1], 0.1);
    assert!(load(short_bias, tiny_args()).contains("dense.bias"));

    let mut no_rows = tiny_weights(&tiny_args());
    no_rows.insert(
        "dense.weight".to_string(),
        mlxcel_core::zeros(&[0, HIDDEN as i32], mlxcel_core::dtype::FLOAT32),
    );
    assert!(load(no_rows, tiny_args()).contains("must be positive"));

    let (mut rows, args) = quantized_projection_weights();
    let scales = mlxcel_core::array_shape(&rows["projection.scales"]);
    rows.insert(
        "projection.scales".to_string(),
        Rng::new(0x2366_0007).tensor(&[scales[0] - 1, scales[1]], 0.1),
    );
    rows.insert(
        "projection.biases".to_string(),
        Rng::new(0x2366_0008).tensor(&[scales[0] - 1, scales[1]], 0.1),
    );
    assert!(load(rows, args).contains("projection.scales"));

    let (mut biases, args) = quantized_projection_weights();
    biases.insert(
        "projection.biases".to_string(),
        Rng::new(0x2366_0009).tensor(&[scales[0], scales[1] + 1], 0.1),
    );
    assert!(load(biases, args).contains("projection.biases"));

    let (mut quant_bias, args) = quantized_projection_weights();
    Rng::new(0x2366_000A).insert(&mut quant_bias, "projection.bias", &[3], 0.1);
    assert!(load(quant_bias, args).contains("projection.bias"));

    // The well-formed triple still loads.
    let (weights, args) = quantized_projection_weights();
    assert!(Lfm2EmbeddingModel::from_weights(weights, args, PoolingMode::Cls, true).is_ok());
}

fn write_dense_config(dir: &Path, config: &Value) {
    std::fs::create_dir_all(dir.join("1_Dense")).unwrap();
    std::fs::write(
        dir.join("1_Dense/config.json"),
        serde_json::to_string(config).unwrap(),
    )
    .unwrap();
}

#[test]
fn dense_config_must_describe_a_plain_projection() {
    let mut rng = Rng::new(0x2366_0003);
    let mut weights = WeightMap::new();
    rng.insert(&mut weights, "dense.0.weight", &[8, 4], 0.1);
    let check = |dir: &Path, config: &Value| {
        validate_projection(dir, config, &weights, Some("dense.0"), 64)
    };
    let identity = json!({
        "in_features": 4, "out_features": 8, "bias": false,
        "activation_function": "torch.nn.modules.linear.Identity", "use_residual": false
    });

    let dir = temp_dir("lfm2_colbert_dense_ok");
    write_dense_config(&dir, &identity);
    check(&dir, &json!({})).unwrap();
    // mlx-community's `mlx.proj_dim` is cross-checked too.
    assert!(err_string(check(&dir, &json!({"mlx": {"proj_dim": 16}}))).contains("proj_dim"));

    for (field, value, needle) in [
        (
            "activation_function",
            json!("torch.nn.modules.activation.Tanh"),
            "Identity",
        ),
        ("use_residual", json!(true), "use_residual"),
        ("out_features", json!(16), "out_features"),
    ] {
        let mut config = identity.clone();
        config[field] = value;
        write_dense_config(&dir, &config);
        assert!(
            err_string(check(&dir, &json!({}))).contains(needle),
            "{field}"
        );
    }
    // A missing activation defaults to Tanh in sentence-transformers.
    let mut config = identity.clone();
    config
        .as_object_mut()
        .unwrap()
        .remove("activation_function");
    write_dense_config(&dir, &config);
    assert!(check(&dir, &json!({})).is_err());

    // A pooled single-vector Dense head lists a Pooling module; a plain
    // Transformer + Dense `modules.json` and a missing one both load.
    let modules = |kinds: &[&str]| {
        let list: Vec<Value> = kinds
            .iter()
            .enumerate()
            .map(|(idx, kind)| json!({"idx": idx, "name": idx.to_string(), "path": "", "type": kind}))
            .collect();
        std::fs::write(
            dir.join("modules.json"),
            serde_json::to_string(&list).unwrap(),
        )
        .unwrap();
    };
    write_dense_config(&dir, &identity);
    modules(&[
        "sentence_transformers.models.Transformer",
        "pylate.models.Dense.Dense",
    ]);
    check(&dir, &json!({})).unwrap();
    modules(&[
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Dense",
    ]);
    assert!(err_string(check(&dir, &json!({}))).contains("Pooling"));
    std::fs::remove_file(dir.join("modules.json")).unwrap();

    // A second Dense folder, and a folder with no projection tensor.
    write_dense_config(&dir, &identity);
    std::fs::create_dir_all(dir.join("2_Dense")).unwrap();
    assert!(err_string(check(&dir, &json!({}))).contains("Dense folders"));
    std::fs::remove_dir_all(dir.join("2_Dense")).unwrap();
    assert!(validate_projection(&dir, &json!({}), &WeightMap::new(), None, 64).is_err());
    std::fs::remove_dir_all(&dir).unwrap();

    // No folder and no tensor: the single-vector export.
    let plain = temp_dir("lfm2_colbert_plain");
    validate_projection(&plain, &json!({}), &WeightMap::new(), None, 64).unwrap();
    std::fs::remove_dir_all(plain).unwrap();
}

// Real checkpoints.

#[test]
fn lfm2_encoder_hidden_states_match_mlx_vlm() {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(ENCODER) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let reference = fixture("encoder_reference.json");
    let ids = ids_of(&reference);
    let expected = rows_of(&reference["hidden"]);

    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let model = Lfm2EmbeddingModel::load(&dir, &config).expect("LFM2.5-Encoder loads");
    assert!(!model.multi_vector());
    let length = ids.len() as i32;
    let hidden = to_vec(&mlxcel_core::astype(
        &model.forward_hidden(
            &ids_array(&ids, 1, length),
            &ids_array(&vec![1; ids.len()], 1, length),
        ),
        mlxcel_core::dtype::FLOAT32,
    ));
    let (min, mean) = per_token_cosine(&hidden, model.embedding_dim(), &expected);
    assert!(
        min >= PARITY_COSINE,
        "encoder hidden states: min per-token cosine {min} (mean {mean})"
    );

    // The served path: detection, tokenizer and CLS pooling.
    let loaded = load_embedding_model(&dir).expect("LFM2.5-Encoder loads as an embedder");
    assert_eq!(loaded.model_type, crate::models::ModelType::Lfm2Embedding);
    assert_eq!(loaded.limits.dim, 1024);
    let engine = EmbeddingEngine::new(loaded, 4);
    let text = reference["text"].as_str().unwrap().to_string();
    let reply = engine
        .embed_texts(&[text], &EmbedOptions::default())
        .expect("LFM2.5-Encoder embeds");
    assert_eq!(reply.vectors[0].shape, vec![1024]);
    let cls = cosine(&reply.vectors[0].values, &expected[0]);
    assert!(cls >= PARITY_COSINE, "CLS vector cosine {cls}");
    eprintln!("GATE LFM2.5-Encoder-350M min_cos={min:.6} mean_cos={mean:.6} cls_cos={cls:.6}");
}

/// Embed the three reference sentences through the served path (detection,
/// tokenizer, projection) and gate the per-token cosine against the reference
/// at `floor`, then the MaxSim ranking of the relevant document over the
/// unrelated one.
fn colbert_gate(repo: &str, floor: f32) {
    let _guard = mlx_test_guard();
    let Some(dir) = local_checkpoint(repo) else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let reference = fixture("colbert_reference.json");
    let sentences = &reference["sentences"];

    let loaded = load_embedding_model(&dir).expect("LFM2.5-ColBERT loads");
    assert_eq!(loaded.model_type, crate::models::ModelType::Lfm2Embedding);
    assert!(loaded.model.multi_vector(), "{repo} is not multi-vector");
    assert_eq!(loaded.limits.dim, PROJ_DIM);
    let engine = EmbeddingEngine::new(loaded, 4);

    let keys = ["query", "relevant", "unrelated"];
    let texts: Vec<String> = keys
        .iter()
        .map(|k| sentences[k]["text"].as_str().unwrap().to_string())
        .collect();
    let reply = engine
        .embed_texts(&texts, &EmbedOptions::default())
        .expect("LFM2.5-ColBERT embeds");

    let mut worst = f32::INFINITY;
    let mut means = Vec::new();
    for (key, vector) in keys.iter().zip(&reply.vectors) {
        let expected = rows_of(&sentences[key]["vectors"]);
        assert_eq!(
            vector.shape,
            vec![ids_of(&sentences[key]).len(), PROJ_DIM],
            "{key}"
        );
        let (min, mean) = per_token_cosine(&vector.values, PROJ_DIM, &expected);
        worst = worst.min(min);
        means.push(mean);
    }
    assert!(
        worst >= floor,
        "{repo}: min per-token cosine {worst} below {floor}"
    );

    let rows = |i: usize| -> Vec<&[f32]> { reply.vectors[i].rows().collect() };
    let relevant = crate::embeddings::maxsim(&rows(0), &rows(1));
    let unrelated = crate::embeddings::maxsim(&rows(0), &rows(2));
    assert!(
        relevant > unrelated,
        "{repo}: MaxSim ranked the unrelated document first ({relevant} vs {unrelated})"
    );
    eprintln!(
        "GATE {repo} min_cos={worst:.6} mean_cos={:?} maxsim_relevant={relevant:.4} \
         maxsim_unrelated={unrelated:.4}",
        means
    );
}

#[test]
fn lfm2_colbert_hf_layout_matches_mlx_vlm() {
    colbert_gate(COLBERT_HF, PARITY_COSINE);
}

#[test]
fn lfm2_colbert_bf16_matches_mlx_vlm() {
    colbert_gate(COLBERT_BF16, PARITY_COSINE);
}

#[test]
fn lfm2_colbert_4bit_loads_and_ranks() {
    colbert_gate(COLBERT_4BIT, QUANTIZED_COSINE);
}
