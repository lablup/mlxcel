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

//! Load-time guards against checkpoints whose config and tensors disagree.
//!
//! Each hostile case goes through the real loader and is paired with the
//! honest fixture loading through the same entry point.

use serde_json::{Value, json};

use super::Qwen4ExpModel;
use super::config::{Qwen4ExpConfig, ResolvedQwen4ExpConfig};
use super::parity_tests::{config, fixture, hashed, tiny_model, weights};
use super::ple::{NgramHasher, PleLayer};

/// One stored tensor replaced by one of another shape.
#[derive(Debug, Clone, Copy)]
struct ReshapedTensor {
    name: &'static str,
    shape: &'static [i32],
}

fn tiny_config(fixture: &Value) -> ResolvedQwen4ExpConfig {
    serde_json::from_value::<Qwen4ExpConfig>(fixture["config"].clone())
        .unwrap()
        .resolve()
        .unwrap()
}

/// Loads the tiny model with `case` applied and returns the load error.
#[track_caller]
fn load_error(fixture: &Value, case: ReshapedTensor) -> String {
    let mut weights = weights(fixture);
    assert!(
        weights.contains_key(case.name),
        "{case:?} names no fixture tensor"
    );
    weights.insert(
        case.name.to_string(),
        hashed(case.name, case.shape, 0.1, 0.0),
    );
    Qwen4ExpModel::from_weights(&weights, &tiny_config(fixture))
        .err()
        .unwrap_or_else(|| panic!("{case:?} must be refused at load"))
}

#[track_caller]
fn assert_blames(error: &str, case: ReshapedTensor) {
    let module = case.name.strip_suffix(".weight").unwrap_or(case.name);
    assert!(
        error.contains(module) && error.contains("the config implies"),
        "{case:?}: {error}"
    );
}

/// The router's width is the index range `gather_qmm` reads the stacked
/// expert planes with; a router wider than the stack, or a stack shorter than
/// `num_experts`, would read past the planes at the first forward pass.
#[test]
fn moe_tensors_that_disagree_with_the_config_are_refused() {
    let fixture = fixture("model");
    tiny_model(&fixture);
    for case in [
        ReshapedTensor {
            name: "model.layers.0.mlp.gate.weight",
            shape: &[16, 32],
        },
        ReshapedTensor {
            name: "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            shape: &[4, 16, 32],
        },
        ReshapedTensor {
            name: "model.layers.0.mlp.switch_mlp.down_proj.weight",
            shape: &[8, 32, 8],
        },
        ReshapedTensor {
            name: "model.layers.0.mlp.shared_expert.up_proj.weight",
            shape: &[24, 16],
        },
        ReshapedTensor {
            name: "model.layers.0.mlp.shared_expert_gate.weight",
            shape: &[2, 32],
        },
    ] {
        assert_blames(&load_error(&fixture, case), case);
    }
}

/// The heads must fit the table the checkpoint stores: one row short of the
/// derived size is refused, the exact size loads.
#[test]
fn ngram_prime_search_stops_at_the_stored_table() {
    let fixture = fixture("ple");
    let config = config(&fixture, None);
    let unbounded = NgramHasher::new(&config, 0, u64::MAX).unwrap();
    let rows = (unbounded.head_offsets.last().unwrap() + unbounded.head_vocab_sizes.last().unwrap())
        as u64;

    assert_eq!(NgramHasher::new(&config, 0, rows).unwrap(), unbounded);
    let error = NgramHasher::new(&config, 0, rows - 1).unwrap_err();
    assert!(error.contains("rows the checkpoint stores"), "{error}");
}

/// A base the stored table cannot hold is refused before any candidate past
/// the table is tested for primality; near 2^62 each test is a trial
/// division up to 2^31.
#[test]
fn ple_layer_refuses_an_ngram_base_the_stored_table_cannot_hold() {
    let fixture = fixture("ple");
    let weights = weights(&fixture);
    PleLayer::from_weights(&weights, &config(&fixture, None), "ple", 0)
        .expect("the fixture's own n-gram base loads");
    for base in [10_000u64, 1 << 62] {
        let config = config(&fixture, Some(&json!({ "ngram_vocab_size_base": base })));
        let error = PleLayer::from_weights(&weights, &config, "ple", 0)
            .err()
            .unwrap_or_else(|| panic!("ngram_vocab_size_base {base} must be refused"));
        assert!(
            error.contains("rows the checkpoint stores"),
            "base {base}: {error}"
        );
    }
}

/// A stored hash tensor is checked against the config by value, so its shape
/// is checked first: reading back an arbitrarily large tensor is not needed
/// to refuse it.
#[test]
fn stored_hash_tensors_of_the_wrong_shape_are_refused() {
    let fixture = fixture("ple");
    let config = config(&fixture, None);
    let hasher = NgramHasher::new(&config, 0, u64::MAX).unwrap();
    let multipliers = &hasher.multipliers;
    let stored = |shape: &[i32]| {
        let mut weights = weights(&fixture);
        weights.insert(
            "ple.ple_embedding.layer_multipliers".to_string(),
            mlxcel_core::from_slice_i64(multipliers, shape),
        );
        PleLayer::from_weights(&weights, &config, "ple", 0)
    };

    stored(&[multipliers.len() as i32]).expect("matching stored multipliers load");
    let error = stored(&[1, multipliers.len() as i32])
        .err()
        .expect("a 2-D multiplier tensor must be refused");
    assert!(error.contains("has shape [1, 3]"), "{error}");
}

/// Every projection, norm and gate vector is reshaped or multiplied by sizes
/// taken from the config; a stored shape that disagrees would make MLX throw
/// while the first forward pass is built.
#[test]
fn layer_tensors_that_disagree_with_the_config_are_refused() {
    let fixture = fixture("model");
    tiny_model(&fixture);
    for case in [
        ReshapedTensor {
            name: "model.embed_tokens.weight",
            shape: &[64, 16],
        },
        ReshapedTensor {
            name: "lm_head.weight",
            shape: &[48, 32],
        },
        ReshapedTensor {
            name: "model.layers.0.linear_attn.in_proj_qkv.weight",
            shape: &[96, 32],
        },
        ReshapedTensor {
            name: "model.layers.0.linear_attn.in_proj_z.weight",
            shape: &[32, 32],
        },
        ReshapedTensor {
            name: "model.layers.0.linear_attn.in_proj_b.weight",
            shape: &[2, 32],
        },
        ReshapedTensor {
            name: "model.layers.0.linear_attn.out_proj.weight",
            shape: &[32, 32],
        },
        ReshapedTensor {
            name: "model.layers.0.linear_attn.dt_bias",
            shape: &[8],
        },
        ReshapedTensor {
            name: "model.layers.0.linear_attn.A_log",
            shape: &[2, 2],
        },
        ReshapedTensor {
            name: "model.layers.0.linear_attn.norm.weight",
            shape: &[8],
        },
        ReshapedTensor {
            name: "model.layers.0.attn_hyper_connection.hc_norm.weight",
            shape: &[64],
        },
        ReshapedTensor {
            name: "model.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
            shape: &[64, 16],
        },
        ReshapedTensor {
            name: "model.layers.0.mlp_hyper_connection.block_inject_weight.weight",
            shape: &[2, 128],
        },
        ReshapedTensor {
            name: "model.hyper_connection_mixer.input_mix_weight_down.weight",
            shape: &[8, 128],
        },
        ReshapedTensor {
            name: "model.layers.1.ple.key_proj.weight",
            shape: &[64, 32],
        },
        ReshapedTensor {
            name: "model.layers.1.ple.value_proj.weight",
            shape: &[32, 16],
        },
        ReshapedTensor {
            name: "model.layers.1.ple.norm_query.weight",
            shape: &[64],
        },
        ReshapedTensor {
            name: "model.layers.3.self_attn.q_proj.weight",
            shape: &[128, 32],
        },
        ReshapedTensor {
            name: "model.layers.3.self_attn.k_proj.weight",
            shape: &[32, 32],
        },
        ReshapedTensor {
            name: "model.layers.3.self_attn.v_proj.weight",
            shape: &[64, 16],
        },
        ReshapedTensor {
            name: "model.layers.3.self_attn.o_proj.weight",
            shape: &[32, 64],
        },
        ReshapedTensor {
            name: "model.layers.3.self_attn.k_norm.weight",
            shape: &[16],
        },
        ReshapedTensor {
            name: "model.layers.3.self_attn.indexer.index_qk_proj.weight",
            shape: &[32, 32],
        },
        ReshapedTensor {
            name: "model.layers.3.self_attn.indexer.q_layernorm.weight",
            shape: &[8],
        },
    ] {
        assert_blames(&load_error(&fixture, case), case);
    }
}

#[test]
fn ple_shards_refuse_malformed_quantization_sidecars() {
    use super::ple::ShardedEmbedding;
    let fixture = fixture("ple");
    let config = config(
        &fixture,
        Some(&json!({
            "split_ngram_parts": 1,
            "quantization": {"group_size": 32, "bits": 4}
        })),
    );
    let prefix = "ple.ple_embedding.ngram_embedding";
    let module = format!("{prefix}.shards.0");
    let tensor = hashed("shard", &[2, 64], 0.1, 0.0);
    let packed = mlxcel_core::quantize_weights(&tensor, 32, 4);
    let mut weights = mlxcel_core::weights::WeightMap::new();
    weights.insert(
        format!("{module}.weight"),
        mlxcel_core::quantized_weights_w(&packed),
    );
    weights.insert(
        format!("{module}.scales"),
        mlxcel_core::quantized_weights_scales(&packed),
    );
    weights.insert(
        format!("{module}.biases"),
        mlxcel_core::quantized_weights_biases(&packed),
    );
    ShardedEmbedding::from_weights(&weights, &config, prefix, 2, 64)
        .expect("valid quantized shard");
    for suffix in ["scales", "biases"] {
        let key = format!("{module}.{suffix}");
        let original = weights.remove(&key).unwrap();
        weights.insert(
            key.clone(),
            mlxcel_core::zeros(&[1, 2], mlxcel_core::dtype::FLOAT32),
        );
        let error = ShardedEmbedding::from_weights(&weights, &config, prefix, 2, 64)
            .err()
            .expect("mismatched sidecar rows must fail at load");
        assert!(error.contains(&key), "{error}");
        weights.insert(key, original);
    }
}
