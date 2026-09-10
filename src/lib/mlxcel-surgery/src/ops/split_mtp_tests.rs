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

//! Tests for the `split-mtp` op on a synthetic 2-head, 2-expert checkpoint.

use std::collections::HashMap;
use std::path::Path;

use mlxcel_core::weights::WeightMap;
use serde_json::{Value, json};

use super::{DRAFTER_MODEL_TYPE, SplitMtpOptions, split_mtp, split_mtp_dir, write_safetensors};

const VOCAB: i32 = 16;
const HIDDEN: i32 = 64;
const HEADS: i32 = 2;
const QK_NOPE: i32 = 4;
const QK_ROPE: i32 = 2;
const V_HEAD: i32 = 6;
const KV_LORA_RANK: i32 = 16;
const EXPERTS: i32 = 2;
const MOE_INTER: i32 = 8;
const LAYERS: usize = 3;

fn source_config() -> Value {
    json!({
        "model_type": "glm4_moe_lite",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": MOE_INTER,
        "moe_intermediate_size": MOE_INTER,
        "num_hidden_layers": LAYERS,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "kv_lora_rank": KV_LORA_RANK,
        "q_lora_rank": null,
        "qk_rope_head_dim": QK_ROPE,
        "qk_nope_head_dim": QK_NOPE,
        "v_head_dim": V_HEAD,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    })
}

/// Every tensor the raw checkpoint stores under `model.layers.{LAYERS}.`,
/// with a `rotary_emb.inv_freq` the tool must drop, plus one decoder-layer
/// tensor the tool must ignore.
fn nextn_tensors() -> Vec<(String, Vec<i32>)> {
    let l = format!("model.layers.{LAYERS}");
    let q_head_dim = QK_NOPE + QK_ROPE;
    let mut out = vec![
        (format!("{l}.embed_tokens.weight"), vec![VOCAB, HIDDEN]),
        (format!("{l}.enorm.weight"), vec![HIDDEN]),
        (format!("{l}.hnorm.weight"), vec![HIDDEN]),
        (format!("{l}.eh_proj.weight"), vec![HIDDEN, 2 * HIDDEN]),
        (format!("{l}.shared_head.norm.weight"), vec![HIDDEN]),
        (format!("{l}.shared_head.head.weight"), vec![VOCAB, HIDDEN]),
        (format!("{l}.input_layernorm.weight"), vec![HIDDEN]),
        (format!("{l}.post_attention_layernorm.weight"), vec![HIDDEN]),
        (
            format!("{l}.self_attn.q_proj.weight"),
            vec![HEADS * q_head_dim, HIDDEN],
        ),
        (
            format!("{l}.self_attn.kv_a_proj_with_mqa.weight"),
            vec![KV_LORA_RANK + QK_ROPE, HIDDEN],
        ),
        (
            format!("{l}.self_attn.kv_a_layernorm.weight"),
            vec![KV_LORA_RANK],
        ),
        (
            format!("{l}.self_attn.kv_b_proj.weight"),
            vec![HEADS * (QK_NOPE + V_HEAD), KV_LORA_RANK],
        ),
        (
            format!("{l}.self_attn.o_proj.weight"),
            vec![HIDDEN, HEADS * V_HEAD],
        ),
        (
            format!("{l}.self_attn.rotary_emb.inv_freq"),
            vec![QK_ROPE / 2],
        ),
        (format!("{l}.mlp.gate.weight"), vec![EXPERTS, HIDDEN]),
        (
            format!("{l}.mlp.gate.e_score_correction_bias"),
            vec![EXPERTS],
        ),
        (
            format!("{l}.mlp.shared_experts.gate_proj.weight"),
            vec![MOE_INTER, HIDDEN],
        ),
        (
            format!("{l}.mlp.shared_experts.up_proj.weight"),
            vec![MOE_INTER, HIDDEN],
        ),
        (
            format!("{l}.mlp.shared_experts.down_proj.weight"),
            vec![HIDDEN, MOE_INTER],
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            vec![HIDDEN],
        ),
    ];
    for e in 0..EXPERTS {
        out.push((
            format!("{l}.mlp.experts.{e}.gate_proj.weight"),
            vec![MOE_INTER, HIDDEN],
        ));
        out.push((
            format!("{l}.mlp.experts.{e}.up_proj.weight"),
            vec![MOE_INTER, HIDDEN],
        ));
        out.push((
            format!("{l}.mlp.experts.{e}.down_proj.weight"),
            vec![HIDDEN, MOE_INTER],
        ));
    }
    out
}

/// A ramp so element mappings are observable, in float32 (the tool casts).
fn ramp(shape: &[i32], seed: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| seed + i as f32 * 0.01).collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn synthetic_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    for (i, (key, shape)) in nextn_tensors().into_iter().enumerate() {
        weights.insert(key, ramp(&shape, i as f32));
    }
    weights
}

fn shape_of(weights: &WeightMap, key: &str) -> Vec<i32> {
    mlxcel_core::array_shape(weights.get(key).unwrap_or_else(|| panic!("missing {key}")))
}

fn dtype_of(weights: &WeightMap, key: &str) -> i32 {
    mlxcel_core::array_dtype(weights.get(key).unwrap_or_else(|| panic!("missing {key}")))
}

fn read_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn split_flattens_nextn_layer() {
    let result = split_mtp(
        synthetic_weights(),
        &source_config(),
        &SplitMtpOptions::default(),
    )
    .expect("split");
    let w = &result.weights;
    assert_eq!(result.source_layer, LAYERS);
    assert_eq!(result.quantized_tensors, 0);

    // The six renamed roots.
    assert_eq!(
        shape_of(w, "model.embed_tokens.weight"),
        vec![VOCAB, HIDDEN]
    );
    assert_eq!(shape_of(w, "model.enorm.weight"), vec![HIDDEN]);
    assert_eq!(shape_of(w, "model.hnorm.weight"), vec![HIDDEN]);
    assert_eq!(
        shape_of(w, "model.eh_proj.weight"),
        vec![HIDDEN, 2 * HIDDEN]
    );
    assert_eq!(shape_of(w, "model.shared_head_norm.weight"), vec![HIDDEN]);
    assert_eq!(shape_of(w, "lm_head.weight"), vec![VOCAB, HIDDEN]);

    // MLA pair, decomposed the way the target loader decomposes it.
    assert_eq!(
        shape_of(w, "model.mtp_block.self_attn.embed_q.weight"),
        vec![HEADS, KV_LORA_RANK, QK_NOPE]
    );
    assert_eq!(
        shape_of(w, "model.mtp_block.self_attn.unembed_out.weight"),
        vec![HEADS, V_HEAD, KV_LORA_RANK]
    );
    assert!(!w.contains_key("model.mtp_block.self_attn.kv_b_proj.weight"));

    // Experts stacked, per-expert planes gone.
    assert_eq!(
        shape_of(w, "model.mtp_block.mlp.switch_mlp.gate_proj.weight"),
        vec![EXPERTS, MOE_INTER, HIDDEN]
    );
    assert_eq!(
        shape_of(w, "model.mtp_block.mlp.switch_mlp.down_proj.weight"),
        vec![EXPERTS, HIDDEN, MOE_INTER]
    );
    assert!(!w.contains_key("model.mtp_block.mlp.experts.0.gate_proj.weight"));

    // Stacking keeps expert order: plane e of the stack is expert e.
    let stacked = read_f32(
        w.get("model.mtp_block.mlp.switch_mlp.up_proj.weight")
            .unwrap(),
    );
    let per_plane = (MOE_INTER * HIDDEN) as usize;
    let source: Vec<(String, Vec<i32>)> = nextn_tensors();
    let seed_of = |key: &str| {
        source
            .iter()
            .position(|(k, _)| k == key)
            .expect("source key") as f32
    };
    let e1_seed = seed_of(&format!(
        "model.layers.{LAYERS}.mlp.experts.1.up_proj.weight"
    ));
    // bf16 keeps about three significant digits; the seed is an integer.
    assert!((stacked[per_plane] - e1_seed).abs() < 0.05);

    // dtype rules.
    assert_eq!(
        dtype_of(w, "model.mtp_block.mlp.gate.e_score_correction_bias"),
        mlxcel_core::dtype::FLOAT32
    );
    assert_eq!(
        dtype_of(w, "model.mtp_block.mlp.gate.weight"),
        mlxcel_core::dtype::BFLOAT16
    );
    assert_eq!(dtype_of(w, "lm_head.weight"), mlxcel_core::dtype::BFLOAT16);
    assert_eq!(
        dtype_of(w, "model.mtp_block.self_attn.embed_q.weight"),
        mlxcel_core::dtype::BFLOAT16
    );

    // Dropped and ignored keys.
    assert!(!w.keys().any(|k| k.contains("rotary_emb")));
    assert!(!w.keys().any(|k| k.starts_with("model.layers.")));
    for key in w.keys() {
        assert!(
            key.starts_with("model.mtp_block.")
                || key.starts_with("model.embed_tokens.")
                || key.starts_with("model.enorm.")
                || key.starts_with("model.hnorm.")
                || key.starts_with("model.eh_proj.")
                || key.starts_with("model.shared_head_norm.")
                || key.starts_with("lm_head."),
            "unexpected drafter key {key}"
        );
    }

    // Config.
    assert_eq!(result.config["model_type"], DRAFTER_MODEL_TYPE);
    assert_eq!(result.config["block_size"], 2);
    assert_eq!(result.config["tie_word_embeddings"], false);
    assert_eq!(result.config["text_config"]["model_type"], "glm4_moe_lite");
    assert_eq!(result.config["text_config"]["num_hidden_layers"], LAYERS);
    assert!(result.config["text_config"].get("quantization").is_none());
    assert!(result.config.get("quantization").is_none());
}

#[test]
fn split_refuses_checkpoint_without_nextn_tensors() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        ramp(&[HIDDEN], 0.0),
    );
    let err = split_mtp(weights, &source_config(), &SplitMtpOptions::default())
        .err()
        .expect("must refuse");
    let msg = err.to_string();
    assert!(msg.contains("converted without its MTP layer"), "{msg}");
    assert!(msg.contains("raw zai-org checkpoint"), "{msg}");
}

#[test]
fn split_refuses_a_quantized_nextn_layer() {
    // A conversion that kept the nextn layer with its experts already
    // stacked and packed reaches the bf16 cast, which would rewrite the
    // packed uint32 payload as floats without an error.
    let mut weights = synthetic_weights();
    let l = format!("model.layers.{LAYERS}");
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        for e in 0..EXPERTS {
            weights.remove(&format!("{l}.mlp.experts.{e}.{proj}.weight"));
        }
        weights.insert(
            format!("{l}.mlp.switch_mlp.{proj}.weight"),
            mlxcel_core::zeros(
                &[EXPERTS, MOE_INTER, HIDDEN / 8],
                mlxcel_core::dtype::UINT32,
            ),
        );
        weights.insert(
            format!("{l}.mlp.switch_mlp.{proj}.scales"),
            ramp(&[EXPERTS, MOE_INTER, HIDDEN / 64], 1.0),
        );
    }
    let err = split_mtp(weights, &source_config(), &SplitMtpOptions::default())
        .err()
        .expect("must refuse");
    let msg = err.to_string();
    assert!(msg.contains("quantized tensor(s)"), "{msg}");
    assert!(msg.contains("switch_mlp"), "{msg}");
    assert!(msg.contains("raw zai-org checkpoint"), "{msg}");
}

#[test]
fn split_refuses_a_foreign_family() {
    let mut config = source_config();
    config["model_type"] = json!("qwen3_moe");
    let err = split_mtp(synthetic_weights(), &config, &SplitMtpOptions::default())
        .err()
        .expect("must refuse");
    assert!(err.to_string().contains("qwen3_moe"), "{err}");
}

#[test]
fn split_quantizes_projections_but_not_the_router_or_norms() {
    let opts = SplitMtpOptions {
        block_size: Some(3),
        q_bits: Some(4),
        q_group_size: 64,
    };
    let result = split_mtp(synthetic_weights(), &source_config(), &opts).expect("split");
    let w = &result.weights;
    assert!(result.quantized_tensors > 0);

    // Quantized: last axis HIDDEN = 64 divides the group size.
    for key in [
        "model.embed_tokens",
        "lm_head",
        "model.eh_proj",
        "model.mtp_block.self_attn.q_proj",
        "model.mtp_block.self_attn.kv_a_proj_with_mqa",
        "model.mtp_block.mlp.switch_mlp.gate_proj",
        "model.mtp_block.mlp.shared_experts.up_proj",
    ] {
        assert!(w.contains_key(&format!("{key}.scales")), "{key} scales");
        assert!(w.contains_key(&format!("{key}.biases")), "{key} biases");
        assert_eq!(
            dtype_of(w, &format!("{key}.weight")),
            mlxcel_core::dtype::UINT32,
            "{key} packed"
        );
        assert_eq!(
            dtype_of(w, &format!("{key}.scales")),
            mlxcel_core::dtype::BFLOAT16,
            "{key} scales dtype"
        );
    }
    // The stacked plane keeps its leading expert axis: [E, out, in / 8].
    assert_eq!(
        shape_of(w, "model.mtp_block.mlp.switch_mlp.gate_proj.weight"),
        vec![EXPERTS, MOE_INTER, HIDDEN / 8]
    );

    // Never quantized: the router, every norm, the selection bias.
    for key in [
        "model.mtp_block.mlp.gate",
        "model.enorm",
        "model.hnorm",
        "model.shared_head_norm",
        "model.mtp_block.input_layernorm",
        "model.mtp_block.self_attn.kv_a_layernorm",
    ] {
        assert!(
            !w.contains_key(&format!("{key}.scales")),
            "{key} must stay dense"
        );
    }
    assert_eq!(
        dtype_of(w, "model.mtp_block.mlp.gate.weight"),
        mlxcel_core::dtype::BFLOAT16
    );
    // Not divisible by the group size: left dense rather than failing.
    assert!(!w.contains_key("model.mtp_block.self_attn.o_proj.scales"));
    assert!(!w.contains_key("model.mtp_block.self_attn.embed_q.scales"));

    assert_eq!(result.config["block_size"], 3);
    assert_eq!(result.config["quantization"]["bits"], 4);
    assert_eq!(result.config["quantization"]["group_size"], 64);
    assert_eq!(result.config["quantization"]["mode"], "affine");
    assert_eq!(result.config["quantization_config"]["bits"], 4);
}

/// Write the synthetic checkpoint as three f32 shards plus an index, the
/// shape the raw checkpoint has, so the directory driver's index filter and
/// writer are exercised end to end.
fn write_synthetic_checkpoint(dir: &Path) {
    let mut by_shard: Vec<WeightMap> = vec![WeightMap::new(), WeightMap::new(), WeightMap::new()];
    let mut weight_map: HashMap<String, String> = HashMap::new();
    for (i, (key, shape)) in nextn_tensors().into_iter().enumerate() {
        let shard = i % 3;
        let file = format!("model-0000{}-of-00003.safetensors", shard + 1);
        weight_map.insert(key.clone(), file);
        by_shard[shard].insert(key, ramp(&shape, i as f32));
    }
    for (shard, weights) in by_shard.into_iter().enumerate() {
        let file = format!("model-0000{}-of-00003.safetensors", shard + 1);
        write_safetensors(weights, &dir.join(file)).expect("write shard");
    }
    let index = json!({"metadata": {}, "weight_map": weight_map});
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(&source_config()).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    std::fs::write(dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
}

#[test]
fn split_dir_writes_a_loadable_drafter_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("raw");
    let out = tmp.path().join("drafter");
    std::fs::create_dir_all(&src).unwrap();
    write_synthetic_checkpoint(&src);

    let opts = SplitMtpOptions {
        block_size: None,
        q_bits: Some(4),
        q_group_size: 64,
    };
    let report = split_mtp_dir(&src, &out, &opts).expect("split dir");
    assert_eq!(report.source_layer, LAYERS);
    assert!(report.quantized_tensors > 0);
    assert!(report.bytes_written > 0);
    assert_eq!(
        report.copied_files,
        vec![
            "tokenizer.json".to_string(),
            "tokenizer_config.json".to_string(),
            "chat_template.jinja".to_string(),
        ]
    );
    assert!(report.missing_files.is_empty());

    // The written file round-trips through the core loader with the
    // drafter layout, the mlx metadata, and the packed dtypes intact.
    let loaded = mlxcel_core::weights::load_weights_from_dir(&out).expect("reload");
    assert_eq!(loaded.len(), report.tensors);
    assert_eq!(
        shape_of(&loaded, "model.mtp_block.self_attn.embed_q.weight"),
        vec![HEADS, KV_LORA_RANK, QK_NOPE]
    );
    assert_eq!(
        dtype_of(&loaded, "lm_head.weight"),
        mlxcel_core::dtype::UINT32
    );
    assert_eq!(
        dtype_of(&loaded, "model.mtp_block.mlp.gate.e_score_correction_bias"),
        mlxcel_core::dtype::FLOAT32
    );
    assert!(!loaded.keys().any(|k| k.starts_with("model.layers.")));

    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["model_type"], DRAFTER_MODEL_TYPE);
    assert_eq!(config["block_size"], 2);
    assert_eq!(config["quantization"]["bits"], 4);
    assert!(out.join("chat_template.jinja").is_file());
}

#[test]
fn split_dir_refuses_an_output_that_aliases_the_source() {
    // `--output <source>` passes the CLI overwrite guard on a sharded
    // checkpoint (no `model.safetensors`), and the writes would replace the
    // source config.json and truncate its tokenizer files to zero bytes.
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("raw");
    std::fs::create_dir_all(&src).unwrap();
    write_synthetic_checkpoint(&src);

    let err = split_mtp_dir(&src, &src, &SplitMtpOptions::default()).expect_err("must refuse");
    assert!(
        err.to_string().contains("is the source checkpoint"),
        "{err}"
    );

    // Nothing was written and nothing was truncated.
    assert!(!src.join("model.safetensors").exists());
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["model_type"], "glm4_moe_lite");
    assert_eq!(
        std::fs::read_to_string(src.join("chat_template.jinja")).unwrap(),
        "{{ messages }}"
    );

    // A nested path under the source is a different directory and is allowed.
    let nested = src.join("mtp");
    split_mtp_dir(&src, &nested, &SplitMtpOptions::default()).expect("nested output");
    assert!(nested.join("model.safetensors").is_file());
}

#[test]
fn split_dir_refuses_an_index_without_the_nextn_layer() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("converted");
    std::fs::create_dir_all(&src).unwrap();
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        ramp(&[HIDDEN], 0.0),
    );
    write_safetensors(weights, &src.join("model-00001-of-00001.safetensors")).unwrap();
    let index = json!({"metadata": {}, "weight_map": {
        "model.layers.0.input_layernorm.weight": "model-00001-of-00001.safetensors"
    }});
    std::fs::write(
        src.join("model.safetensors.index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();
    std::fs::write(
        src.join("config.json"),
        serde_json::to_string(&source_config()).unwrap(),
    )
    .unwrap();

    let err = split_mtp_dir(&src, &tmp.path().join("out"), &SplitMtpOptions::default())
        .expect_err("must refuse");
    assert!(
        err.to_string().contains("converted without its MTP layer"),
        "{err}"
    );
    assert!(!tmp.path().join("out").exists());
}

/// The recorded `block_size` becomes the served verify width, and this family
/// materializes one query row per verify position, so an unbounded value
/// produces a directory that loads fine and then wedges the scheduler tick.
/// A typo has to fail here, before any tensor work (issue #1326).
#[test]
fn split_refuses_an_oversized_block_size() {
    let opts = SplitMtpOptions {
        block_size: Some(20_000),
        ..SplitMtpOptions::default()
    };
    let err = split_mtp(synthetic_weights(), &source_config(), &opts)
        .err()
        .expect("must refuse");
    let msg = err.to_string();
    assert!(msg.contains("20000"), "{msg}");
    assert!(msg.contains("maximum"), "{msg}");
    // The message is a single sentence with no intentional alignment: a run
    // of two or more spaces means the `\` line continuations were dropped
    // and rustfmt joined the wrapped literal, leaving the wrap indentation
    // inside the rendered text (issue #1763).
    assert!(!msg.contains("  "), "{msg}");
}

/// The ceiling must not move the widths a caller can legitimately ask for.
#[test]
fn split_accepts_block_sizes_up_to_the_ceiling() {
    for requested in [2usize, 8, 16] {
        let opts = SplitMtpOptions {
            block_size: Some(requested),
            ..SplitMtpOptions::default()
        };
        let result =
            split_mtp(synthetic_weights(), &source_config(), &opts).expect("must be accepted");
        assert_eq!(result.config["block_size"], requested);
    }
}

/// MLX's affine quantize kernel only implements `{2, 3, 4, 5, 6, 8}`; a width
/// outside that set must be refused before any tensor work, naming the width
/// and the accepted set, rather than quantizing and failing later at load
/// (issue #1763).
///
/// Uses an empty `WeightMap` rather than [`synthetic_weights`]: it has no
/// `model.layers.*` tensor at all, so if the bits guard ran after the rename
/// pass (which needs at least one such tensor) this would instead fail with
/// "no model.layers... tensor", proving the guard runs before any tensor
/// work touches `weights` (PR #1778 review).
#[test]
fn split_refuses_an_unsupported_q_bits() {
    for bits in [1, 7, 9, 16, 32] {
        let opts = SplitMtpOptions {
            q_bits: Some(bits),
            ..SplitMtpOptions::default()
        };
        let err = split_mtp(WeightMap::new(), &source_config(), &opts)
            .err()
            .unwrap_or_else(|| panic!("--q-bits {bits} must be refused"));
        let msg = err.to_string();
        assert!(msg.contains(&format!("({bits})")), "{msg}");
        assert!(msg.contains("2, 3, 4, 5, 6, 8"), "{msg}");
    }
}

/// Every width the `--q-bits` help text advertises must actually be accepted.
#[test]
fn split_accepts_every_supported_affine_bit_width() {
    for bits in [2, 3, 4, 5, 6, 8] {
        let opts = SplitMtpOptions {
            q_bits: Some(bits),
            q_group_size: 64,
            ..SplitMtpOptions::default()
        };
        let result = split_mtp(synthetic_weights(), &source_config(), &opts)
            .unwrap_or_else(|e| panic!("--q-bits {bits} must be accepted: {e}"));
        assert_eq!(result.config["quantization"]["bits"], bits);
    }
}

/// MLX's affine quantize kernel only implements group sizes `{32, 64, 128}`;
/// a group size outside that set must be refused before any tensor work,
/// naming the value and the accepted set, the same way an unsupported
/// `--q-bits` is (PR #1778 review). Uses an empty `WeightMap` for the same
/// before-tensor-work reason as `split_refuses_an_unsupported_q_bits`.
#[test]
fn split_refuses_an_unsupported_q_group_size() {
    for group_size in [16, 256, 1] {
        let opts = SplitMtpOptions {
            q_bits: Some(4),
            q_group_size: group_size,
            ..SplitMtpOptions::default()
        };
        let err = split_mtp(WeightMap::new(), &source_config(), &opts)
            .err()
            .unwrap_or_else(|| panic!("--q-group-size {group_size} must be refused"));
        let msg = err.to_string();
        assert!(msg.contains(&format!("({group_size})")), "{msg}");
        assert!(msg.contains("32, 64, 128"), "{msg}");
    }
}

/// Every group size MLX's affine quantize kernel implements must be accepted
/// by the pre-flight guard (whether or not any tensor here is actually wide
/// enough to be quantized at that group size).
#[test]
fn split_accepts_every_supported_affine_group_size() {
    for group_size in [32, 64, 128] {
        let opts = SplitMtpOptions {
            q_bits: Some(4),
            q_group_size: group_size,
            ..SplitMtpOptions::default()
        };
        split_mtp(synthetic_weights(), &source_config(), &opts)
            .unwrap_or_else(|e| panic!("--q-group-size {group_size} must be accepted: {e}"));
    }
}
