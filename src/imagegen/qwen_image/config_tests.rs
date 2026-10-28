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

#[test]
fn image_dimensions_match_upstream_rounding() {
    // `(resolution, width, height) -> _image_dimensions(resolution, w / h)`,
    // computed by upstream Python; 1081600 / 1048576 puts the width exactly
    // on 32.5 multiples of 32, which Python rounds half to even (down).
    let cases = [
        ((1024, 1, 1), (1024, 1024)),
        ((1024, 640, 480), (1184, 896)),
        ((1024, 1920, 1080), (1376, 768)),
        ((1024, 1_081_600, 1_048_576), (1024, 1024)),
        ((512, 3, 1), (896, 288)),
        ((1024, 333, 1000), (576, 1760)),
        ((768, 1, 4), (384, 1536)),
    ];
    for ((res, w, h), expected) in cases {
        assert_eq!(
            image_dimensions(res, f64::from(w) / f64::from(h)),
            expected,
            "{res} {w}x{h}"
        );
    }
    assert_eq!(default_edit_size(640, 480), (1184, 896));
}

#[test]
fn dimensions_are_bounded_and_aligned() {
    validate_dimensions(1024, 1024).unwrap();
    validate_dimensions(256, 2048).unwrap();
    for (w, h) in [(240, 1024), (1024, 2064), (1000, 1024)] {
        assert!(validate_dimensions(w, h).is_err(), "{w}x{h}");
    }
}

#[test]
fn defaults_follow_the_task() {
    assert_eq!(defaults(ImageTask::Generate).steps, 30);
    assert_eq!(defaults(ImageTask::Edit).steps, 40);
    assert!((defaults(ImageTask::Edit).guidance - 1.0).abs() < f32::EPSILON);
}

#[test]
fn options_parse_and_reject_misuse() {
    let extra = |v: serde_json::Value| v.as_object().unwrap().clone();
    let o = QwenImageOptions::from_extra(&extra(json!({})), ImageTask::Edit).unwrap();
    assert_eq!(o.output_resolution, 1024);
    assert!(o.use_kv_cache);
    assert_eq!(o.negative_prompt, None);
    let o = QwenImageOptions::from_extra(
        &extra(
            json!({"output_resolution": 512, "use_kv_cache": false, "negative_prompt": "blurry"}),
        ),
        ImageTask::Edit,
    )
    .unwrap();
    assert_eq!(o.output_resolution, 512);
    assert!(!o.use_kv_cache);
    assert_eq!(o.negative_prompt.as_deref(), Some("blurry"));
    for bad in [
        json!({"output_resolution": 100}),
        json!({"output_resolution": "big"}),
        json!({"use_kv_cache": "no"}),
        json!({"negative_prompt": 3}),
    ] {
        assert!(
            QwenImageOptions::from_extra(&extra(bad.clone()), ImageTask::Edit).is_err(),
            "{bad}"
        );
    }
    assert!(
        QwenImageOptions::from_extra(&extra(json!({"use_kv_cache": true})), ImageTask::Generate)
            .is_err()
    );
}

#[test]
fn transformer_config_reads_the_checkpoint_json() {
    let c = TransformerConfig::from_json(&json!({
        "_class_name": "QwenImage21Transformer2DModel",
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "context_in_dim": 4096,
        "in_channels": 64,
        "num_attention_heads": 32,
        "num_layers": 32,
        "out_channels": 64,
        "patch_size": 1,
        "mlp_ratio": 3,
        "eps": 1e-06,
        "causal_condition": true
    }))
    .unwrap();
    assert_eq!(c, TransformerConfig::default());
    assert_eq!(c.inner_dim(), 4096);
    assert!(TransformerConfig::from_json(&json!({"patch_size": 2})).is_err());
    assert!(TransformerConfig::from_json(&json!({"axes_dims_rope": [16, 56, 50]})).is_err());
}

#[test]
fn vae_config_reads_residual_and_plain_layouts() {
    let mut mean = vec![0.0; 64];
    mean[3] = 1.3506;
    let c = QwenImageVaeConfig::from_json(&json!({
        "base_dim": 96, "decoder_base_dim": 144, "z_dim": 64,
        "dim_mult": [1, 2, 4, 8, 8], "num_res_blocks": 2,
        "temperal_downsample": [false, true, true, true],
        "in_channels": 4, "out_channels": 4, "is_residual": true,
        "latents_mean": mean, "latents_std": vec![2.0; 64]
    }))
    .unwrap();
    assert_eq!(c.decoder_base_dim, 144);
    assert_eq!(c.spatial_scale(), 16);
    assert!((c.latents_mean[3] - 1.3506).abs() < 1e-6);
    // Ming-Image: no decoder_base_dim, non-residual, scaling factors.
    let ming = QwenImageVaeConfig::from_json(&json!({
        "base_dim": 96, "z_dim": 16, "is_residual": false,
        "scaling_factor": 0.5, "shift_factor": 0.1
    }))
    .unwrap();
    assert_eq!(ming.decoder_base_dim, 96);
    assert!(!ming.is_residual);
    assert_eq!(ming.scaling_factor, Some(0.5));
    assert!(QwenImageVaeConfig::from_json(&json!({"temperal_downsample": [true]})).is_err());
    assert!(
        QwenImageVaeConfig::from_json(&json!({"z_dim": 4, "latents_std": [1.0, 0.0, 1.0, 1.0]}))
            .is_err()
    );
}

#[test]
fn scheduler_params_read_the_checkpoint_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("scheduler")).unwrap();
    std::fs::write(
        dir.path().join("scheduler/scheduler_config.json"),
        json!({"base_shift": 0.5, "max_shift": 0.9, "base_image_seq_len": 256,
               "max_image_seq_len": 8192, "num_train_timesteps": 1000, "shift_terminal": 0.02})
        .to_string(),
    )
    .unwrap();
    assert_eq!(
        scheduler_params(dir.path()).unwrap(),
        DynamicShift::default()
    );
    let missing = tempfile::tempdir().unwrap();
    assert_eq!(
        scheduler_params(missing.path()).unwrap(),
        DynamicShift::default()
    );
}
