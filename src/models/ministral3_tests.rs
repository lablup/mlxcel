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

//! YaRN rope parameters of Ministral 3.

use super::*;

/// `rope_parameters` exactly as every released Ministral 3 checkpoint ships
/// it (both `type` and `rope_type`).
const CHECKPOINT_ROPE: &str = r#"{
    "beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0,
    "llama_4_scaling_beta": 0.1, "mscale": 1.0, "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 16384, "rope_theta": 1000000.0,
    "rope_type": "yarn", "type": "yarn"
}"#;

#[test]
fn checkpoint_rope_parameters_select_yarn_with_mlx_lm_frequencies() {
    let params: RopeParameters = serde_json::from_str(CHECKPOINT_ROPE).unwrap();
    let (freqs, mscale) = params.yarn_freqs(128).expect("checkpoints declare yarn");
    assert_eq!(freqs.len(), 64);
    assert_eq!(mscale, 1.0);
    // mlx-lm 0.32.3 `initialize_rope(128, 1e6, False, <above>, 262144)._freqs`.
    let upstream = [
        (0, 1.0_f32),
        (10, 8.659643),
        (20, 74.98942),
        (30, 1447.8018),
        (40, 89974.61),
        (50, 779148.06),
        (63, 1.2893475e7),
    ];
    for (i, want) in upstream {
        let got = freqs[i];
        assert!(
            (got - want).abs() <= want * 1e-5,
            "freq {i}: got {got}, mlx-lm {want}"
        );
    }
    // The high-frequency end is the plain base^(2i/d) table; the low end is
    // stretched by the factor, which plain `rope_theta` rotation misses.
    let plain_63 = 1_000_000.0_f32.powf(126.0 / 128.0);
    assert!((freqs[63] / plain_63 - 16.0).abs() < 1e-3);
}

#[test]
fn non_yarn_rope_parameters_keep_the_plain_rotation() {
    let params: RopeParameters = serde_json::from_str(
        r#"{"rope_theta": 1000000.0, "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16384}"#,
    )
    .unwrap();
    assert!(params.yarn_freqs(128).is_none());
    let params = RopeParameters {
        rope_type: Some("default".into()),
        ..params
    };
    assert!(params.yarn_freqs(128).is_none());
}
