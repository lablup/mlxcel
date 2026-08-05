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

//! Unit tests for OpenELM's layer-wise scaling.
//!
//! Everything here is checkpoint-free. The config test parses the real
//! `mlx-community/OpenELM-1_1B-Instruct-4bit` field set, and the model tests
//! build tiny synthetic weight maps whose key names mirror the checkpoint.
//!
//! The tests that matter most are the per-layer ones. A loader that reads head
//! counts and FFN widths once and reuses them for every block still produces a
//! model that loads and generates on a *uniform* config; only a config whose
//! layers genuinely differ can tell the two apart, which is why the fixtures
//! here deliberately vary both across layers.

use super::{ModelArgs, OpenElmModel, make_divisible};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// Config surface.

/// The `mlx-community/OpenELM-1_1B-Instruct-4bit` config, field-for-field.
const OPENELM_1_1B_CONFIG: &str = r#"{
    "activation_fn_name": "swish",
    "architectures": ["OpenELMForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "ffn_dim_divisor": 256,
    "ffn_multipliers": [0.5, 0.63, 0.76, 0.89, 1.02, 1.15, 1.28, 1.41, 1.54, 1.67,
                        1.8, 1.93, 2.06, 2.19, 2.31, 2.44, 2.57, 2.7, 2.83, 2.96,
                        3.09, 3.22, 3.35, 3.48, 3.61, 3.74, 3.87, 4.0],
    "ffn_with_glu": true,
    "head_dim": 64,
    "initializer_range": 0.02,
    "max_context_length": 2048,
    "model_dim": 2048,
    "model_type": "openelm",
    "normalization_layer_name": "rms_norm",
    "normalize_qk_projections": true,
    "num_gqa_groups": 4,
    "num_kv_heads": [4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
                     7, 7, 7, 7, 8, 8, 8, 8],
    "num_query_heads": [16, 16, 16, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24,
                        24, 24, 24, 24, 28, 28, 28, 28, 28, 28, 32, 32, 32, 32],
    "num_transformer_layers": 28,
    "qkv_multipliers": [0.5, 1.0],
    "quantization": { "group_size": 64, "bits": 4 },
    "rope_freq_constant": 10000,
    "rope_max_length": 4096,
    "share_input_output_layers": true,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.39.3",
    "use_cache": true,
    "vocab_size": 32000
}"#;

fn openelm_1_1b() -> ModelArgs {
    serde_json::from_str(OPENELM_1_1B_CONFIG).expect("the real config must parse")
}

#[test]
fn the_real_config_parses_with_its_per_layer_lists() {
    let args = openelm_1_1b();
    assert_eq!(args.model_type, "openelm");
    assert_eq!(args.model_dim, 2048);
    assert_eq!(args.head_dim, 64);
    assert_eq!(args.num_transformer_layers, 28);
    assert_eq!(args.vocab_size, 32000);
    assert_eq!(args.ffn_dim_divisor, 256);

    // The three per-layer lists carry one entry per layer, not a scalar.
    assert_eq!(args.num_query_heads.len(), 28);
    assert_eq!(args.num_kv_heads.len(), 28);
    assert_eq!(args.ffn_multipliers.len(), 28);

    assert!(args.normalize_qk_projections);
    assert!(args.share_input_output_layers);
    assert_eq!(args.rope_freq_constant, 10_000.0);

    // Absent from the config, so the upstream default has to supply it.
    assert_eq!(args.rms_norm_eps, 1e-6);

    assert!(args.validate().is_ok());
}

#[test]
fn head_counts_and_ffn_widths_differ_across_layers() {
    // This is the whole point of the family. If any of these four assertions
    // could be satisfied by a single global number, the port would not need
    // per-layer construction at all.
    let args = openelm_1_1b();

    assert_eq!(args.num_query_heads[0], 16);
    assert_eq!(args.num_query_heads[27], 32);
    assert_eq!(args.num_kv_heads[0], 4);
    assert_eq!(args.num_kv_heads[27], 8);

    // GQA grouping changes with them: 16/4 = 4 at the bottom, 32/8 = 4 at the
    // top, but 20/5 and 28/7 in between, so no fixed group count works either.
    assert_eq!(args.num_query_heads[3] / args.num_kv_heads[3], 4);

    assert_eq!(args.intermediate_size(0), 1024);
    assert_eq!(args.intermediate_size(1), 1280);
    assert_eq!(args.intermediate_size(13), 4608);
    assert_eq!(args.intermediate_size(27), 8192);
}

#[test]
fn make_divisible_matches_the_upstream_helper() {
    // Values cross-checked against the Python original. The first three are the
    // ordinary round-to-nearest-multiple case.
    assert_eq!(make_divisible(1024.0, 256), 1024);
    assert_eq!(make_divisible(1100.0, 256), 1024);
    assert_eq!(make_divisible(13.0, 8), 16);

    // The floor is `divisor` itself, not zero: upstream's `min_value` defaults
    // to `divisor`, so a small width is lifted rather than rounded to nothing.
    assert_eq!(make_divisible(100.0, 256), 256);
    assert_eq!(make_divisible(1.0, 8), 8);

    // The "do not round down by more than 10%" correction. 600 rounds down to
    // 512, which is below 0.9 * 600 = 540, so one more divisor is added.
    // Dropping this branch silently narrows the FFN.
    assert_eq!(make_divisible(600.0, 256), 768);
    assert_eq!(make_divisible(300.0, 256), 512);
}

#[test]
fn a_per_layer_list_shorter_than_the_stack_is_rejected() {
    // Indexing these lists by layer id is the first thing construction does, so
    // a short list has to be a load-time error rather than a panic.
    let short = OPENELM_1_1B_CONFIG.replace(
        r#""num_transformer_layers": 28"#,
        r#""num_transformer_layers": 30"#,
    );
    let args: ModelArgs = serde_json::from_str(&short).expect("must parse");
    let err = args.validate().expect_err("a short list must be rejected");
    assert!(err.contains("num_query_heads"), "got: {err}");
}

#[test]
fn a_non_grouping_head_ratio_is_rejected() {
    let bad = OPENELM_1_1B_CONFIG.replace(
        r#""num_kv_heads": [4, 4, 4, 5,"#,
        r#""num_kv_heads": [5, 4, 4, 5,"#,
    );
    let args: ModelArgs = serde_json::from_str(&bad).expect("must parse");
    // 16 query heads against 5 KV heads is not a whole grouping.
    let err = args.validate().expect_err("must be rejected");
    assert!(err.contains("GQA grouping"), "got: {err}");
}

// Model construction and forward.

/// A tiny config whose layers genuinely differ, so a loader that reused layer
/// 0's geometry would build the wrong shapes for layer 1 and fail.
fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
        "model_type": "openelm",
        "vocab_size": 64,
        "model_dim": 32,
        "head_dim": 8,
        "num_transformer_layers": 3,
        "ffn_dim_divisor": 16,
        "num_query_heads": [2, 4, 4],
        "num_kv_heads": [1, 2, 4],
        "ffn_multipliers": [1.0, 2.0, 3.0],
        "normalize_qk_projections": true,
        "share_input_output_layers": true,
        "eos_token_id": 2
    }"#,
    )
    .expect("the tiny config must parse")
}

fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let dim = args.model_dim as i32;
    let vocab = args.vocab_size as i32;
    let head_dim = args.head_dim as i32;

    let mut w = WeightMap::new();
    w.insert(
        "transformer.token_embeddings.weight".into(),
        filled(&[vocab, dim]),
    );
    w.insert("transformer.norm.weight".into(), ones(&[dim]));

    for layer in 0..args.num_transformer_layers {
        let p = format!("transformer.layers.{layer}");
        let nq = args.num_query_heads[layer] as i32;
        let nkv = args.num_kv_heads[layer] as i32;
        let ff = args.intermediate_size(layer) as i32;

        w.insert(format!("{p}.attn_norm.weight"), ones(&[dim]));
        w.insert(format!("{p}.ffn_norm.weight"), ones(&[dim]));

        // Both widths move with the layer: the fused QKV output and the
        // out_proj input.
        w.insert(
            format!("{p}.attn.qkv_proj.weight"),
            filled(&[(nq + 2 * nkv) * head_dim, dim]),
        );
        w.insert(
            format!("{p}.attn.out_proj.weight"),
            filled(&[dim, nq * head_dim]),
        );
        w.insert(format!("{p}.attn.q_norm.weight"), ones(&[head_dim]));
        w.insert(format!("{p}.attn.k_norm.weight"), ones(&[head_dim]));

        w.insert(format!("{p}.ffn.proj_1.weight"), filled(&[2 * ff, dim]));
        w.insert(format!("{p}.ffn.proj_2.weight"), filled(&[dim, ff]));
    }
    w
}

#[test]
fn from_weights_builds_each_block_with_its_own_geometry() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = OpenElmModel::from_weights(&weights, &args).expect("must build");

    assert_eq!(model.num_layers(), 3);

    // Each attention block carries its own head counts rather than layer 0's.
    assert_eq!(model.layers[0].attn.num_heads, 2);
    assert_eq!(model.layers[0].attn.num_kv_heads, 1);
    assert_eq!(model.layers[1].attn.num_heads, 4);
    assert_eq!(model.layers[1].attn.num_kv_heads, 2);
    assert_eq!(model.layers[2].attn.num_kv_heads, 4);

    // And its own FFN width: 1.0/2.0/3.0 * 32, rounded to multiples of 16.
    assert_eq!(model.layers[0].ffn.intermediate_size, 32);
    assert_eq!(model.layers[1].ffn.intermediate_size, 64);
    assert_eq!(model.layers[2].ffn.intermediate_size, 96);

    // share_input_output_layers is true, so no separate output projection.
    assert!(model.lm_head.is_none());
}

#[test]
fn forward_produces_vocab_sized_logits_and_advances_the_cache() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = OpenElmModel::from_weights(&weights, &args).expect("must build");

    let mut caches = model.make_caches();
    assert_eq!(caches.len(), 3);

    let tokens = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);
    let logits = LanguageModel::forward(&model, &tokens, &mut caches, None);
    mlxcel_core::eval(&logits);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 4, args.vocab_size as i32]
    );
    assert_eq!(caches[0].offset, 4);

    let next = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
    let logits = LanguageModel::forward(&model, &next, &mut caches, None);
    mlxcel_core::eval(&logits);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 1, args.vocab_size as i32]
    );
    assert_eq!(caches[0].offset, 5);
}

#[test]
fn qk_norm_is_loaded_only_when_the_config_asks_for_it() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = OpenElmModel::from_weights(&weights, &args).expect("must build");
    assert!(model.layers[0].attn.q_norm.is_some());
    assert!(model.layers[0].attn.k_norm.is_some());

    let mut off = tiny_args();
    off.normalize_qk_projections = false;
    let model = OpenElmModel::from_weights(&weights, &off).expect("must build");
    assert!(
        model.layers[0].attn.q_norm.is_none(),
        "a config with normalize_qk_projections false must not load the norms"
    );
}

#[test]
fn a_missing_per_layer_tensor_is_an_error() {
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    // Drop a tensor from the LAST layer specifically: a loader that built every
    // block from layer 0's keys would never notice.
    weights.remove("transformer.layers.2.ffn.proj_2.weight");
    assert!(OpenElmModel::from_weights(&weights, &args).is_err());
}

#[test]
fn eos_ids_reach_the_built_model() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = OpenElmModel::from_weights(&weights, &args).expect("must build");
    assert_eq!(LanguageModel::eos_token_ids(&model), vec![2]);
}

// Fixtures.

fn filled(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state: u32 = 0x9E37_79B9;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}
