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

//! Tests for the `split-mtp` op on a synthetic 2-head, 2-expert checkpoint.

use std::collections::HashMap;
use std::path::Path;

use mlxcel_core::weights::WeightMap;
use serde_json::{Value, json};

use super::output::{StagingDir, commit_staging_dir};
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
        force: false,
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
        force: false,
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

fn assert_no_staging_directory(parent: &Path, output_name: &str) {
    let prefix = format!(".{output_name}.split-mtp-");
    let staged: Vec<String> = std::fs::read_dir(parent)
        .expect("read parent")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with(&prefix))
        .collect();
    assert!(staged.is_empty(), "staging directories remain: {staged:?}");
}

#[test]
fn empty_existing_output_is_replaced_by_the_staged_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("drafter");
    std::fs::create_dir(&out).unwrap();
    let staging = StagingDir::create(&out).expect("staging");
    std::fs::write(staging.path().join("config.json"), b"new").unwrap();

    commit_staging_dir(staging, &out, false).expect("commit");

    assert_eq!(std::fs::read(out.join("config.json")).unwrap(), b"new");
    assert_no_staging_directory(tmp.path(), "drafter");
}

#[cfg(unix)]
#[test]
fn staging_directory_is_private() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("drafter");
    let staging = StagingDir::create(&out).expect("staging");

    let mode = std::fs::metadata(staging.path())
        .expect("metadata")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o700);
}

#[test]
fn dot_and_dotdot_outputs_stage_beside_the_resolved_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let dot_target = tmp.path().join("dot-target");
    let dotdot_child = tmp.path().join("dotdot-target").join("child");
    std::fs::create_dir(&dot_target).unwrap();
    std::fs::create_dir_all(&dotdot_child).unwrap();

    for output in [dot_target.join("."), dotdot_child.join("..")] {
        let staging = StagingDir::create(&output).expect("staging");
        assert_eq!(staging.path().parent(), Some(tmp.path()));
        std::fs::write(staging.path().join("config.json"), b"new").unwrap();
        commit_staging_dir(staging, &output, true).expect("commit");
        assert_eq!(std::fs::read(output.join("config.json")).unwrap(), b"new");
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_output_name_is_supported() {
    use std::os::unix::ffi::OsStringExt;

    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join(std::ffi::OsString::from_vec(vec![
        b'd', b'r', 0xff, b'f', b't',
    ]));
    let staging = StagingDir::create(&output).expect("staging");
    std::fs::write(staging.path().join("config.json"), b"new").unwrap();

    commit_staging_dir(staging, &output, false).expect("commit");

    assert_eq!(std::fs::read(output.join("config.json")).unwrap(), b"new");
}

#[cfg(unix)]
#[test]
fn commit_refuses_an_ancestor_symlink_retarget() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    let first = tmp.path().join("first");
    let second = tmp.path().join("second");
    let route = tmp.path().join("route");
    std::fs::create_dir(&first).unwrap();
    std::fs::create_dir(&second).unwrap();
    symlink(&first, &route).unwrap();
    let output = route.join("drafter");
    let staging = StagingDir::create(&output).expect("staging");
    std::fs::write(staging.path().join("config.json"), b"new").unwrap();
    std::fs::remove_file(&route).unwrap();
    symlink(&second, &route).unwrap();

    let err = commit_staging_dir(staging, &output, false).expect_err("must refuse drift");

    assert!(err.to_string().contains("changed while"), "{err}");
    assert!(!first.join("drafter").exists());
    assert!(!second.join("drafter").exists());
    assert_no_staging_directory(&first, "drafter");
}

#[test]
fn concurrent_non_empty_output_wins_without_interleaving() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("drafter");
    let staging = StagingDir::create(&out).expect("staging");
    std::fs::write(staging.path().join("config.json"), b"loser").unwrap();
    std::fs::create_dir(&out).unwrap();
    std::fs::write(out.join("config.json"), b"winner").unwrap();
    std::fs::write(out.join("model.safetensors"), b"winner weights").unwrap();

    let err = commit_staging_dir(staging, &out, false).expect_err("must refuse");

    assert!(err.to_string().contains("config.json"), "{err}");
    assert_eq!(std::fs::read(out.join("config.json")).unwrap(), b"winner");
    assert_eq!(
        std::fs::read(out.join("model.safetensors")).unwrap(),
        b"winner weights"
    );
    assert_no_staging_directory(tmp.path(), "drafter");
}

#[test]
fn split_dir_refuses_an_unrelated_non_empty_output_without_force() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("raw");
    let out = tmp.path().join("drafter");
    std::fs::create_dir(&src).unwrap();
    std::fs::create_dir(&out).unwrap();
    write_synthetic_checkpoint(&src);
    std::fs::write(out.join("README.md"), b"keep").unwrap();

    let err = split_mtp_dir(&src, &out, &SplitMtpOptions::default()).expect_err("must refuse");

    assert!(err.to_string().contains("README.md"), "{err}");
    assert_eq!(std::fs::read(out.join("README.md")).unwrap(), b"keep");
    assert!(!out.join("model.safetensors").exists());
    assert_no_staging_directory(tmp.path(), "drafter");
}

#[cfg(unix)]
#[test]
fn split_dir_refuses_present_and_dangling_output_symlinks_for_both_force_modes() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("raw");
    std::fs::create_dir(&src).unwrap();
    write_synthetic_checkpoint(&src);

    for force in [false, true] {
        let suffix = if force { "force" } else { "plain" };
        let tokenizer_out = tmp.path().join(format!("tokenizer-{suffix}"));
        let tokenizer_target = tmp.path().join(format!("tokenizer-target-{suffix}"));
        std::fs::create_dir(&tokenizer_out).unwrap();
        std::fs::write(&tokenizer_target, b"untouched").unwrap();
        symlink(&tokenizer_target, tokenizer_out.join("tokenizer.json")).unwrap();
        let opts = SplitMtpOptions {
            force,
            ..SplitMtpOptions::default()
        };
        let err = split_mtp_dir(&src, &tokenizer_out, &opts).expect_err("must refuse");
        assert!(err.to_string().contains("tokenizer.json"), "{err}");
        assert_eq!(std::fs::read(&tokenizer_target).unwrap(), b"untouched");
        assert_no_staging_directory(tmp.path(), &format!("tokenizer-{suffix}"));

        let config_out = tmp.path().join(format!("config-{suffix}"));
        let config_target = tmp.path().join(format!("config-target-{suffix}"));
        std::fs::create_dir(&config_out).unwrap();
        symlink(&config_target, config_out.join("config.json")).unwrap();
        let err = split_mtp_dir(&src, &config_out, &opts).expect_err("must refuse");
        assert!(err.to_string().contains("config.json"), "{err}");
        assert!(!config_target.exists(), "dangling target must stay absent");
        assert_no_staging_directory(tmp.path(), &format!("config-{suffix}"));
    }
}

#[test]
fn force_commit_replaces_drafter_files_and_removes_the_stale_index() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("raw");
    let out = tmp.path().join("drafter");
    std::fs::create_dir(&src).unwrap();
    std::fs::create_dir(&out).unwrap();
    write_synthetic_checkpoint(&src);
    std::fs::write(out.join("model.safetensors"), b"old").unwrap();
    std::fs::write(out.join("config.json"), b"old").unwrap();
    std::fs::write(out.join("model.safetensors.index.json"), b"stale").unwrap();
    let opts = SplitMtpOptions {
        force: true,
        ..SplitMtpOptions::default()
    };

    split_mtp_dir(&src, &out, &opts).expect("force replacement");

    assert!(!out.join("model.safetensors.index.json").exists());
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["model_type"], DRAFTER_MODEL_TYPE);
    assert_no_staging_directory(tmp.path(), "drafter");
}

#[test]
fn safetensors_writer_refuses_to_replace_an_existing_file() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.safetensors");
    std::fs::write(&path, b"keep").unwrap();

    let err = write_safetensors(synthetic_weights(), &path).expect_err("must refuse");

    assert!(err.to_string().contains("creating"), "{err}");
    assert_eq!(std::fs::read(path).unwrap(), b"keep");
}

#[test]
fn exclusive_safetensors_writer_matches_the_crate_serializer_byte_for_byte() {
    let tmp = tempfile::tempdir().unwrap();
    let exclusive_path = tmp.path().join("exclusive.safetensors");
    let reference_path = tmp.path().join("reference.safetensors");
    let mut weights = WeightMap::new();
    weights.insert("z-f32".to_string(), ramp(&[2, 3], 1.0));
    weights.insert("a-f32".to_string(), ramp(&[1, 4], 2.0));
    let bf16_source = ramp(&[2, 2], 3.0);
    weights.insert(
        "m-bf16".to_string(),
        mlxcel_core::astype(&bf16_source, mlxcel_core::dtype::BFLOAT16),
    );
    let u32_source = mlxcel_core::from_slice_i32(&[1, 2, 3], &[3]);
    weights.insert(
        "b-u32".to_string(),
        mlxcel_core::astype(&u32_source, mlxcel_core::dtype::UINT32),
    );

    write_safetensors(weights, &exclusive_path).expect("exclusive write");
    let exclusive = std::fs::read(&exclusive_path).expect("read exclusive");
    let parsed = safetensors::SafeTensors::deserialize(&exclusive).expect("parse exclusive");
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "mlx".to_string());
    safetensors::serialize_to_file(parsed.tensors(), Some(metadata), &reference_path)
        .expect("reference write");

    assert_eq!(
        exclusive,
        std::fs::read(reference_path).expect("read reference")
    );
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
fn split_dir_refuses_an_unresolved_dotdot_alias_without_touching_the_source() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("raw");
    std::fs::create_dir(&src).unwrap();
    write_synthetic_checkpoint(&src);
    let config_before = std::fs::read(src.join("config.json")).unwrap();
    let shard = src.join("model-00001-of-00003.safetensors");
    let weights_before = std::fs::read(&shard).unwrap();
    let mut entries_before: Vec<_> = std::fs::read_dir(&src)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    entries_before.sort();
    let output = src.join("new").join("..");

    let err = split_mtp_dir(&src, &output, &SplitMtpOptions::default()).expect_err("must refuse");

    assert!(err.to_string().contains("source checkpoint"), "{err}");
    assert_eq!(
        std::fs::read(src.join("config.json")).unwrap(),
        config_before
    );
    assert_eq!(std::fs::read(shard).unwrap(), weights_before);
    let mut entries_after: Vec<_> = std::fs::read_dir(&src)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    entries_after.sort();
    assert_eq!(entries_after, entries_before);
    assert!(!src.join("new").exists());
    assert_no_staging_directory(tmp.path(), "raw");
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
