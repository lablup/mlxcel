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

use super::*;

#[test]
fn defaults_are_the_published_flux2_vae() {
    let c = Flux2VaeConfig::default();
    assert_eq!(c.block_out_channels, [128, 256, 512, 512]);
    assert_eq!(
        (c.in_channels, c.out_channels, c.latent_channels),
        (3, 3, 32)
    );
    assert_eq!((c.layers_per_block, c.norm_num_groups), (2, 32));
    assert!(c.mid_block_add_attention);
    assert_eq!(c.batch_norm_eps, 1e-4);
    assert_eq!((c.scaling_factor, c.shift_factor), (1.0, 0.0));
    assert_eq!(c.spatial_scale(), 8);
    c.validate().unwrap();
}

#[test]
fn diffusers_config_json_parses_and_unknown_fields_are_ignored() {
    let c = Flux2VaeConfig::from_json(
        r#"{
            "_class_name": "AutoencoderKLFlux2",
            "act_fn": "silu",
            "batch_norm_eps": 0.0001,
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["DownEncoderBlock2D"],
            "latent_channels": 32,
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "patch_size": [2, 2],
            "use_post_quant_conv": true
        }"#,
    )
    .unwrap();
    assert_eq!(c, Flux2VaeConfig::default());
}

#[test]
fn a_flux1_style_config_overrides_the_layout() {
    let c = Flux2VaeConfig::from_json(
        r#"{"latent_channels": 16, "scaling_factor": 0.3611, "shift_factor": 0.1159}"#,
    )
    .unwrap();
    assert_eq!(c.latent_channels, 16);
    assert!((c.scaling_factor - 0.3611).abs() < 1e-6);
    assert!((c.shift_factor - 0.1159).abs() < 1e-6);
}

#[test]
fn invalid_configs_are_rejected_with_a_reason() {
    for (json, needle) in [
        (r#"{"block_out_channels": []}"#, "must not be empty"),
        (
            r#"{"block_out_channels": [100, 200]}"#,
            "divisible by norm_num_groups",
        ),
        (r#"{"norm_num_groups": 0}"#, "norm_num_groups"),
        (r#"{"latent_channels": 0}"#, "positive"),
        (r#"{"layers_per_block": 0}"#, "layers_per_block"),
        (r#"{"scaling_factor": 0.0}"#, "scaling_factor"),
        (
            r#"{"block_out_channels": "wide"}"#,
            "invalid FLUX.2 VAE config.json",
        ),
    ] {
        let err = Flux2VaeConfig::from_json(json).unwrap_err();
        assert!(format!("{err:#}").contains(needle), "{json}: {err:#}");
    }
}

#[test]
fn a_missing_config_file_yields_the_defaults() {
    let dir = tempfile::TempDir::new().unwrap();
    assert_eq!(
        Flux2VaeConfig::from_dir(dir.path()).unwrap(),
        Flux2VaeConfig::default()
    );
}

#[test]
fn from_dir_reads_config_json() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("config.json"), r#"{"latent_channels": 16}"#).unwrap();
    assert_eq!(
        Flux2VaeConfig::from_dir(dir.path())
            .unwrap()
            .latent_channels,
        16
    );
}

#[test]
fn spatial_scale_follows_the_number_of_down_blocks() {
    let c = Flux2VaeConfig {
        block_out_channels: vec![32, 64],
        ..Flux2VaeConfig::default()
    };
    assert_eq!(c.spatial_scale(), 2);
}
