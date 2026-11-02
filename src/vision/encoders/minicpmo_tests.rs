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

use super::{current_patch_position_ids, get_2d_sincos_pos_embed};

#[test]
fn current_patch_position_ids_bucket_dynamic_grid() {
    let ids = current_patch_position_ids(70, 2, 3);
    assert_eq!(ids.len(), 6);
    assert_eq!(ids[0], 0);
    assert_eq!(ids[1], 23);
    assert_eq!(ids[2], 46);
    assert_eq!(ids[3], 35 * 70);
}

#[test]
fn get_2d_sincos_pos_embed_returns_expected_length() {
    let embed = get_2d_sincos_pos_embed(2, 3, 8);
    assert_eq!(embed.len(), 2 * 3 * 8);
}

#[test]
fn mlp_activation_follows_the_vision_config() {
    use super::mlp_activation_from_vision_config;
    use mlxcel_core::utils::GeluActivation;

    let tanh = serde_json::json!({"hidden_act": "gelu_pytorch_tanh"});
    assert_eq!(
        mlp_activation_from_vision_config(&tanh),
        Ok(GeluActivation::TanhF32Compiled)
    );
    // Omitted: the SigLIP class default, tanh (#2548).
    assert_eq!(
        mlp_activation_from_vision_config(&serde_json::json!({})),
        Ok(GeluActivation::TanhF32Compiled)
    );
    let erf = serde_json::json!({"hidden_act": "gelu"});
    assert_eq!(
        mlp_activation_from_vision_config(&erf),
        Ok(GeluActivation::Erf)
    );
    let silu = serde_json::json!({"hidden_act": "silu"});
    assert!(mlp_activation_from_vision_config(&silu).is_err());
}
