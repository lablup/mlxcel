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

use serde_json::json;

use super::*;

fn real_vae_json() -> Value {
    json!({
        "_class_name": "AutoencoderKLQwenImage",
        "attn_scales": [],
        "base_dim": 96,
        "dim_mult": [1, 2, 4, 4],
        "dropout": 0.0,
        "input_channels": 4,
        "latents_mean": null,
        "latents_std": null,
        "scaling_factor": 8.0064,
        "shift_factor": 0.0,
        "num_res_blocks": 2,
        "temperal_downsample": [false, true, true],
        "z_dim": 16
    })
}

#[test]
fn the_vae_is_a_plain_four_channel_qwen_image_vae() {
    let vae = vae_config(&real_vae_json()).unwrap();
    assert_eq!((vae.in_channels, vae.out_channels), (4, 4));
    assert!(
        !vae.is_residual,
        "Ming's config has no is_residual; upstream defaults to plain"
    );
    assert_eq!(vae.scaling_factor, Some(8.0064));
    assert_eq!(vae.shift_factor, Some(0.0));
    assert!(vae.latents_mean.is_empty() && vae.latents_std.is_empty());
    assert_eq!(vae.spatial_scale(), DOWNSAMPLE as i32);
}

#[test]
fn an_explicit_residual_flag_or_a_zero_scale_is_honored_or_rejected() {
    let mut c = real_vae_json();
    c["is_residual"] = json!(true);
    assert!(vae_config(&c).unwrap().is_residual);
    let mut c = real_vae_json();
    c["scaling_factor"] = json!(0.0);
    assert!(vae_config(&c).is_err());
}

#[test]
fn the_bridge_reads_the_query_scale_and_selected_layers() {
    let bridge = BridgeConfig::from_json(&json!({
        "diffusion_c_input_dim": 2560,
        "img_gen_scales": [16],
        "selected_hidden_states_layers": [5, 12, 20],
        "diffusion_inner_dim": 3840
    }))
    .unwrap();
    assert_eq!(
        bridge,
        BridgeConfig {
            query_token_count: 256,
            cap_feat_dim: 2560,
            directvlm_dim: 3840,
            selected_layers: vec![5, 12, 20],
        }
    );
    assert!(BridgeConfig::from_json(&json!({"selected_hidden_states_layers": [-1]})).is_err());
    assert!(BridgeConfig::from_json(&json!({"img_gen_scales": [0]})).is_err());
}

fn llm_json() -> Value {
    json!({
        "hidden_size": 2048,
        "intermediate_size": 5120,
        "moe_intermediate_size": 512,
        "num_experts": 256,
        "num_shared_experts": 1,
        "num_attention_heads": 16,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 20,
        "num_key_value_heads": 4,
        "vocab_size": 157184,
        "routed_scaling_factor": 2.5,
        "n_group": 8,
        "topk_group": 4,
        "rope_theta": 600000.0,
        "partial_rotary_factor": 0.5,
        "use_expert_bias": true
    })
}

#[test]
fn the_backbone_maps_to_a_sigmoid_grouped_bailing_v2() {
    let args = bailing_args(&llm_json()).unwrap();
    assert_eq!(args.num_hidden_layers, 20);
    assert_eq!(args.num_experts(), 256);
    assert_eq!(args.first_k_dense_replace, 1);
    assert!(args.use_qk_norm);
    assert!(args.moe_router_enable_expert_bias);
    assert_eq!(args.score_function, "sigmoid");
    assert_eq!((args.n_group, args.topk_group), (8, 4));
    assert!((args.routed_scaling_factor - 2.5).abs() < 1e-6);
    let mut missing = llm_json();
    missing.as_object_mut().unwrap().remove("num_experts");
    assert!(bailing_args(&missing).is_err());
}

#[test]
fn the_schedule_shift_follows_upstream_scheduler_shift() {
    // 1024x1024: 64 * 64 patches.
    let small = scheduler_shift(4096);
    assert_eq!((small.max_image_seq_len, small.max_shift), (4096, 1.35));
    let s = scheduler_shift(1024);
    assert_eq!((s.max_image_seq_len, s.max_shift), (4096, 1.15));
    assert_eq!((s.base_shift, s.base_image_seq_len), (0.5, 256));
    assert!(s.shift_terminal.is_none());
    let big = scheduler_shift(8192);
    assert_eq!((big.max_image_seq_len, big.max_shift), (8192, 1.35));
}

#[test]
fn the_real_checkpoint_configs_parse() {
    let Some(root) = crate::imagegen::ming_image::parity_tests::model_dir("ming-image-0.1-design")
    else {
        return;
    };
    let config = MingImageConfig::from_model_dir(&root).unwrap();
    assert_eq!(config.bridge.query_token_count, 256);
    assert_eq!(config.dit.in_channels, 16);
    assert_eq!(config.connector.num_hidden_layers, 28);
    assert_eq!(
        config.tokens,
        BlockTokens {
            patch: 157_157,
            start: 157_158,
            end: 157_159
        }
    );
}
