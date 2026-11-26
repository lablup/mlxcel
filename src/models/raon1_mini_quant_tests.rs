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

//! Load-time refusals of malformed Raon-1 mini exports (#3437): scales that
//! disagree with their weight on any plane, malformed per-tensor
//! `global_scale` tensors, and tensor scales flagged on a dense linear.

use super::super::raon1_mini::Raon1MiniModel;
use super::super::raon1_mini_config::Raon1MiniConfig;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::{Value, json};

const RDA_Q: &str = "model.layers.1.self_attn.q_proj";
const MLA_Q_A: &str = "model.layers.3.self_attn.q_a_proj";

fn guard() -> crate::models::embedding_test_support::MlxTestGuard {
    crate::models::embedding_test_support::mlx_test_guard()
}

/// The tiny four-layer model (RDA layers 1, 2 and 4, MLA layer 3) declared
/// as the vendor MLX export, with `quantization` entries merged in.
fn mlx_config_json(entries: Value, global_scaled: &[&str]) -> Value {
    let mut quantization = json!({"group_size": 16, "bits": 4, "mode": "nvfp4"});
    for (path, entry) in entries.as_object().into_iter().flatten() {
        quantization[path] = entry.clone();
    }
    json!({
        "model_type": "raon1_mini", "vocab_size": 32, "hidden_size": 32,
        "intermediate_size": 48, "num_hidden_layers": 4, "num_attention_heads": 2,
        "head_dim": 32, "num_experts": 4, "num_experts_per_token": 2,
        "moe_intermediate_size": 16, "num_shared_experts": 1, "q_lora_rank": 16,
        "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
        "v_head_dim": 8, "rda_decay_rank": 8, "raon_mhc_streams": 4, "eos_token_id": [31],
        "rda_config": {"num_heads": 2, "head_dim": 32, "conv_kernel_size": 4,
            "full_rank_gate": true, "gate_lower_bound": -5.0, "rda_layers": [1, 2, 4],
            "mla_layers": [3]},
        "quantization": quantization,
        "raon_global_scaled_linears": global_scaled,
    })
}

fn mlx_config(global_scaled: &[&str]) -> Raon1MiniConfig {
    Raon1MiniConfig::from_json_str(&mlx_config_json(json!({}), global_scaled).to_string())
        .expect("tiny MLX export config")
}

fn u8s(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::zeros(shape, dtype::UINT8)
}

/// An NVFP4 g16 plane for a `[out, inp]` linear, with `rows` scale rows.
fn nvfp4_plane(weights: &mut WeightMap, prefix: &str, out: i32, inp: i32, rows: i32) {
    weights.insert(format!("{prefix}.weight"), u8s(&[out, inp / 2]));
    weights.insert(format!("{prefix}.scales"), u8s(&[rows, inp / 16]));
}

fn sanitize_err(weights: WeightMap, config: &Raon1MiniConfig) -> String {
    Raon1MiniModel::sanitize_weights(weights, config)
        .err()
        .expect("the malformed export must be refused at load")
}

/// RDA and MLA projections are built by their own modules, which check only
/// the weight's rows; a scales plane with fewer rows is refused by the
/// export sanitize before either reads it.
#[test]
fn rda_and_mla_planes_with_short_scale_rows_are_refused() {
    let _guard = guard();
    let config = mlx_config(&[]);
    for prefix in [RDA_Q, MLA_Q_A] {
        let mut weights = WeightMap::new();
        nvfp4_plane(&mut weights, prefix, 64, 32, 32);
        let err = sanitize_err(weights, &config);
        assert!(
            err.contains(&format!("{prefix}.scales")) && err.contains(&format!("{prefix}.weight")),
            "{err}"
        );
    }
}

/// A per-tensor `global_scale` on a listed linear must be one float element.
#[test]
fn malformed_per_tensor_global_scales_are_refused() {
    let _guard = guard();
    let config = mlx_config(&[RDA_Q]);
    let key = format!("{RDA_Q}.global_scale");
    let with_scale = |scale: UniquePtr<MlxArray>| {
        let mut weights = WeightMap::new();
        nvfp4_plane(&mut weights, RDA_Q, 64, 32, 64);
        weights.insert(key.clone(), scale);
        weights
    };
    for scale in [
        mlxcel_core::from_slice_f32(&[0.5, 0.5], &[2]),
        mlxcel_core::zeros(&[], dtype::INT32),
        mlxcel_core::zeros(&[1], dtype::UINT8),
    ] {
        let err = sanitize_err(with_scale(scale), &config);
        assert!(err.contains(&key) && err.contains("one float"), "{err}");
    }
    for scale in [
        mlxcel_core::from_slice_f32(&[0.5], &[]),
        mlxcel_core::from_slice_f32(&[0.5], &[1]),
        mlxcel_core::astype(&mlxcel_core::from_slice_f32(&[0.5], &[]), dtype::BFLOAT16),
    ] {
        let mut weights = with_scale(scale);
        assert!(
            super::sanitize_export(&mut weights, &config).is_ok(),
            "a scalar float global_scale loads"
        );
    }
}

/// `"global_scale": true` on a dense linear is refused by name: the loader
/// resolves a flagged scale for `gather_qmm` only, so the dense linear would
/// load with its scale dropped.
#[test]
fn global_scale_flag_on_a_dense_linear_is_refused() {
    let flagged = "model.layers.0.mlp.gate_proj";
    let entries = json!({
        flagged: {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": true},
        "model.layers.1.moe.switch_mlp.gate_proj":
            {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": true},
    });
    let err = Raon1MiniConfig::from_json_str(&mlx_config_json(entries, &[]).to_string())
        .expect_err("a dense global_scale flag must be refused");
    assert!(
        err.contains(flagged) && err.contains("raon_global_scaled_linears"),
        "{err}"
    );

    // The weight-side backstop: a resolved scale on a dense path (flagged
    // through `quantization_config`, or with no weight beside it).
    let _guard = guard();
    let config = mlx_config(&[]);
    let mut weights = WeightMap::new();
    let key = format!("{flagged}.native_global_scale");
    weights.insert(key.clone(), mlxcel_core::from_slice_f32(&[0.5], &[]));
    let err = sanitize_err(weights, &config);
    assert!(err.contains(&key), "{err}");
}
