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

use super::*;

#[test]
fn output_decode_rejects_non_finite_values() {
    let mut bytes = 1.0f32.to_ne_bytes().to_vec();
    bytes.extend_from_slice(&f32::INFINITY.to_ne_bytes());
    assert!(decode_output(&bytes).unwrap_err().contains("flat index 1"));
}

#[test]
fn compiler_identity_changes_when_same_path_and_version_bytes_change() {
    let path = std::env::temp_dir().join(format!("mlxcel-molmo2-compiler-{}", std::process::id()));
    std::fs::write(&path, b"first").unwrap();
    let first = sha256_file(&path).unwrap();
    std::fs::write(&path, b"second").unwrap();
    let second = sha256_file(&path).unwrap();
    std::fs::remove_file(path).ok();
    assert_ne!(first, second);
}

#[cfg(feature = "diagnostics")]
#[test]
fn layer18_row_probes_precede_the_first_failing_selected_stage() {
    let mut config = Molmo2VisionConfig::from_json_strs(
        &serde_json::json!({
            "model_type":"molmo2","image_patch_id":151938,
            "vit_config":{"hidden_size":8,"intermediate_size":16,"num_attention_heads":2,
                "head_dim":4,"num_hidden_layers":2,"image_default_input_size":[28,28],
                "image_patch_size":14,"image_num_pos":4,"layer_norm_eps":1e-6,
                "hidden_act":"gelu_pytorch_tanh"},
            "adapter_config":{"hidden_size":8,"intermediate_size":12,"text_hidden_size":10,
                "num_attention_heads":2,"head_dim":4,"vit_layers":[0,1],
                "pooling_attention_mask":true}
        })
        .to_string(),
        &serde_json::json!({"patch_size":14,"max_crops":1,"overlap_margins":[0,0],
            "pooling_size":[2,2],"size":{"height":28,"width":28}})
        .to_string(),
    )
    .unwrap();
    config.layers = 25;
    config.emitted_layers = 25;
    config.selected_layers = vec![24, 18];
    config.static_crops = 1;
    config.patches_per_crop = MOLMO2_VIT_PROBE_FLAT_ROW + 1;
    config.position_count = config.patches_per_crop;

    let names = diagnostic_stage_specs(&config, 1, 1)
        .into_iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    let first_probe = names
        .iter()
        .position(|name| name == "vit.probe.18.row.513.input")
        .expect("layer 18 input probe");
    let selected18 = names
        .iter()
        .position(|name| name == "vit.selected.18")
        .expect("selected layer 18");
    assert_eq!(
        &names[first_probe..selected18],
        &[
            "vit.probe.18.row.513.input",
            "vit.probe.18.row.513.attention_norm",
            "vit.probe.18.row.513.attention",
            "vit.probe.18.row.513.post_attention_residual",
            "vit.probe.18.row.513.ffn_norm",
            "vit.probe.18.row.513.mlp",
            "vit.probe.18.row.513.output",
        ],
        "all row-local producer boundaries must be compared before fail-fast reaches selected18"
    );
}
