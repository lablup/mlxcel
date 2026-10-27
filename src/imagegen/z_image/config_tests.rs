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

// Literals carry all nine significant digits of an f32 on purpose.
#![allow(clippy::excessive_precision)]

use super::*;
use serde_json::json;

/// `FlowMatchEulerDiscreteScheduler(num_inference_steps=n, shift=s).sigmas`
/// from upstream `mage_flow/scheduler.py` (mlx 0.32.4).
const SHIFT3_8: &[f32] = &[
    1.0,
    0.9545454382896423,
    0.8999999761581421,
    0.8333332538604736,
    0.7499999403953552,
    0.6428571343421936,
    0.5,
    0.30000001192092896,
    0.0,
];
const SHIFT6_8: &[f32] = &[
    1.0,
    0.9767441749572754,
    0.9473684430122375,
    0.9090908169746399,
    0.8571428656578064,
    0.782608687877655,
    0.6666666865348816,
    0.4615384638309479,
    0.0,
];
const SHIFT3_49: &[f32] = &[
    1.0,
    0.9931034445762634,
    0.9860138893127441,
    0.978723406791687,
    0.9712230563163757,
    0.9635035991668701,
    0.9555554986000061,
    0.9473683834075928,
    0.9389312267303467,
    0.9302325248718262,
    0.9212598204612732,
    0.9119999408721924,
    0.9024389982223511,
    0.8925620317459106,
    0.882352888584137,
    0.8717948198318481,
    0.8608696460723877,
    0.8495575189590454,
    0.8378379344940186,
    0.8256880640983582,
    0.8130841851234436,
    0.8000000715255737,
    0.7864077687263489,
    0.7722773551940918,
    0.7575757503509521,
    0.7422680258750916,
    0.7263157367706299,
    0.7096773982048035,
    0.692307710647583,
    0.6741572618484497,
    0.6551723480224609,
    0.6352941393852234,
    0.6144578456878662,
    0.5925925970077515,
    0.5696203112602234,
    0.5454545021057129,
    0.5199999809265137,
    0.4931507408618927,
    0.46478867530822754,
    0.43478256464004517,
    0.4029850959777832,
    0.3692307472229004,
    0.3333333134651184,
    0.2950820028781891,
    0.25423726439476013,
    0.21052631735801697,
    0.16363641619682312,
    0.11320749670267105,
    0.05882352963089943,
    0.0,
];
const SHIFT6_49: &[f32] = &[
    1.0,
    0.9965397715568542,
    0.9929577708244324,
    0.9892473220825195,
    0.985401451587677,
    0.9814126491546631,
    0.9772727489471436,
    0.9729729890823364,
    0.9685038328170776,
    0.9638554453849792,
    0.9590163230895996,
    0.9539748430252075,
    0.9487178921699524,
    0.943231463432312,
    0.9374999403953552,
    0.9315067529678345,
    0.9252336621284485,
    0.9186602830886841,
    0.9117647409439087,
    0.9045226573944092,
    0.8969072699546814,
    0.888888955116272,
    0.8804347515106201,
    0.8715084195137024,
    0.8620689511299133,
    0.8520709872245789,
    0.8414633870124817,
    0.8301886916160583,
    0.8181818127632141,
    0.8053690791130066,
    0.7916666269302368,
    0.7769784331321716,
    0.7611940503120422,
    0.7441860437393188,
    0.725806474685669,
    0.7058823704719543,
    0.6842105388641357,
    0.6605504751205444,
    0.6346153020858765,
    0.6060605645179749,
    0.5744681358337402,
    0.5393258333206177,
    0.5,
    0.4556962549686432,
    0.4054053723812103,
    0.3478260934352875,
    0.2812500596046448,
    0.2033897489309311,
    0.111111119389534,
    0.0,
];

fn config(shift: f32) -> ZImageConfig {
    ZImageConfig::new(ZImageTransformerConfig::default(), shift).unwrap()
}

fn assert_sigmas(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-6, "{label}[{i}]: {g} vs {w}");
    }
}

#[test]
fn schedules_match_upstream_static_shift() {
    // Generation runs `steps - 1` scheduler steps; editing runs `steps`.
    let turbo = config(3.0);
    let base = config(6.0);
    assert_sigmas(
        "turbo gen 9",
        turbo.generate_schedule(9).unwrap().sigmas(),
        SHIFT3_8,
    );
    assert_sigmas(
        "turbo edit 8",
        turbo.edit_schedule(8).unwrap().sigmas(),
        SHIFT3_8,
    );
    assert_sigmas(
        "turbo gen 50",
        turbo.generate_schedule(50).unwrap().sigmas(),
        SHIFT3_49,
    );
    assert_sigmas(
        "base gen 9",
        base.generate_schedule(9).unwrap().sigmas(),
        SHIFT6_8,
    );
    assert_sigmas(
        "base gen 50",
        base.generate_schedule(50).unwrap().sigmas(),
        SHIFT6_49,
    );
    assert_sigmas(
        "base edit 49",
        base.edit_schedule(49).unwrap().sigmas(),
        SHIFT6_49,
    );
    // The default Turbo generation (8 steps) runs 7 transformer forwards.
    assert_eq!(turbo.generate_schedule(8).unwrap().num_steps(), 7);
}

#[test]
fn img2img_start_index_matches_upstream_table() {
    // `_img2img_start_index(steps, strength)` evaluated in the upstream venv.
    let strengths = [0.1, 0.25, 0.3, 0.5, 0.6, 0.7, 0.75, 0.9, 1.0];
    let table: &[(u32, [usize; 9])] = &[
        (1, [0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (2, [1, 1, 1, 1, 0, 0, 0, 0, 0]),
        (4, [3, 3, 2, 2, 1, 1, 1, 0, 0]),
        (8, [7, 6, 5, 4, 3, 2, 2, 0, 0]),
        (9, [8, 6, 6, 4, 3, 2, 2, 0, 0]),
        (10, [9, 7, 7, 5, 4, 3, 2, 1, 0]),
        (50, [45, 37, 35, 25, 20, 15, 12, 5, 0]),
    ];
    for (steps, expected) in table {
        for (strength, want) in strengths.iter().zip(expected) {
            assert_eq!(
                img2img_start_index(*steps, *strength),
                *want,
                "steps={steps} strength={strength}"
            );
        }
    }
}

#[test]
fn variant_and_defaults_follow_the_scheduler_shift() {
    let turbo = config(3.0);
    assert_eq!(turbo.variant, Variant::Turbo);
    assert_eq!(
        (turbo.defaults().steps, turbo.defaults().guidance),
        (8, 0.0)
    );
    let base = config(6.0);
    assert_eq!(base.variant, Variant::Base);
    assert_eq!((base.defaults().steps, base.defaults().guidance), (50, 4.0));
    assert!(ZImageConfig::new(ZImageTransformerConfig::default(), 0.0).is_err());
}

#[test]
fn step_and_guidance_edges_are_rejected() {
    let turbo = config(3.0);
    let err = turbo.generate_schedule(1).unwrap_err().to_string();
    assert_eq!(err, "Z-Image steps must be at least 2, got 1");
    assert!(turbo.edit_schedule(1).is_ok());
    assert!(turbo.edit_schedule(0).is_err());
    assert!(turbo.check_guidance(0.0).is_ok());
    assert!(turbo.check_guidance(1.0).is_ok());
    assert!(
        turbo
            .check_guidance(1.5)
            .unwrap_err()
            .to_string()
            .contains("does not support classifier-free guidance")
    );
    assert!(config(6.0).check_guidance(7.5).is_ok());
}

#[test]
fn options_parse_and_range_check() {
    let opts = ZImageOptions::from_extra(&Map::new()).unwrap();
    assert_eq!(opts.cfg_truncation, 1.0);
    assert_eq!(opts.strength, 0.6);
    assert_eq!(opts.negative_prompt, None);
    let extra = |v: Value| v.as_object().unwrap().clone();
    let opts = ZImageOptions::from_extra(&extra(json!({
        "cfg_truncation": 0.5, "strength": 1.0, "negative_prompt": "blurry"
    })))
    .unwrap();
    assert_eq!(
        (
            opts.cfg_truncation,
            opts.strength,
            opts.negative_prompt.as_deref()
        ),
        (0.5, 1.0, Some("blurry"))
    );
    for (bad, message) in [
        (
            json!({"strength": 0.0}),
            "strength must be in (0, 1], got 0",
        ),
        (
            json!({"strength": 1.5}),
            "strength must be in (0, 1], got 1.5",
        ),
        (
            json!({"cfg_truncation": 1.5}),
            "cfg_truncation must be in [0, 1]",
        ),
        (
            json!({"cfg_truncation": -0.1}),
            "cfg_truncation must be in [0, 1]",
        ),
        (json!({"strength": "high"}), "strength must be a number"),
        (json!({"num_images": 2}), "use n"),
    ] {
        let err = ZImageOptions::from_extra(&extra(bad.clone()))
            .unwrap_err()
            .to_string();
        assert!(err.contains(message), "{bad}: {err}");
    }
}

#[test]
fn transformer_config_reads_diffusers_names() {
    let parsed = ZImageTransformerConfig::from_value(&json!({
        "_class_name": "ZImageTransformer2DModel",
        "all_f_patch_size": [1], "all_patch_size": [2], "axes_dims": [32, 48, 48],
        "cap_feat_dim": 2560, "dim": 3840, "in_channels": 16, "n_heads": 30,
        "n_kv_heads": 30, "n_layers": 30, "n_refiner_layers": 2, "norm_eps": 1e-05,
        "qk_norm": true, "rope_theta": 256.0, "t_scale": 1000.0
    }))
    .unwrap();
    assert_eq!(parsed, ZImageTransformerConfig::default());
    assert_eq!(parsed.head_dim(), 128);
    assert_eq!(parsed.adaln_dim(), 256);
    // The context refiner depth defaults to the noise refiner's.
    let small = ZImageTransformerConfig::from_value(&json!({
        "dim": 32, "n_heads": 2, "n_kv_heads": 2, "axes_dims": [4, 6, 6], "n_refiner_layers": 3
    }))
    .unwrap();
    assert_eq!(small.n_context_refiner_layers, 3);
    assert_eq!(small.adaln_dim(), 32);
    for bad in [
        json!({"dim": 32, "n_heads": 2, "n_kv_heads": 1, "axes_dims": [4, 6, 6]}),
        json!({"dim": 32, "n_heads": 2, "n_kv_heads": 2, "axes_dims": [4, 6, 8]}),
        json!({"dim": 30, "n_heads": 4, "n_kv_heads": 4}),
    ] {
        assert!(ZImageTransformerConfig::from_value(&bad).is_err(), "{bad}");
    }
}

#[test]
fn config_reads_a_checkpoint_directory() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("transformer")).unwrap();
    std::fs::create_dir_all(dir.path().join("scheduler")).unwrap();
    std::fs::write(dir.path().join("transformer/config.json"), "{}").unwrap();
    std::fs::write(
        dir.path().join("scheduler/scheduler_config.json"),
        r#"{"shift": 6.0, "use_dynamic_shifting": false}"#,
    )
    .unwrap();
    let cfg = ZImageConfig::from_model_dir(dir.path()).unwrap();
    assert_eq!(cfg.variant, Variant::Base);
    assert_eq!(cfg.scheduler_shift, 6.0);
    std::fs::remove_file(dir.path().join("scheduler/scheduler_config.json")).unwrap();
    let err = ZImageConfig::from_model_dir(dir.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains("Missing Z-Image config"), "{err}");
}
