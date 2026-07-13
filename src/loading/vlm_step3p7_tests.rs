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

use super::normalize_key;

#[test]
fn strips_language_model_wrapper_to_model_prefix() {
    assert_eq!(
        normalize_key("model.language_model.layers.0.self_attn.q_proj.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        normalize_key("language_model.model.norm.weight"),
        "model.norm.weight"
    );
    assert_eq!(
        normalize_key("language_model.lm_head.weight"),
        "lm_head.weight"
    );
}

#[test]
fn maps_vision_prefixes_to_vision_model() {
    assert_eq!(
        normalize_key("model.vision_model.ln_pre.weight"),
        "vision_model.ln_pre.weight"
    );
    // Bare vision keys get the vision_model namespace.
    assert_eq!(normalize_key("conv1.weight"), "vision_model.conv1.weight");
    assert_eq!(
        normalize_key("positional_embedding"),
        "vision_model.positional_embedding"
    );
    assert_eq!(
        normalize_key("vit_downsampler1.weight"),
        "vision_model.vit_downsampler1.weight"
    );
}

#[test]
fn projector_and_lm_head_stay_top_level() {
    assert_eq!(
        normalize_key("model.vit_large_projector.weight"),
        "vit_large_projector.weight"
    );
    assert_eq!(normalize_key("lm_head.weight"), "lm_head.weight");
}

#[test]
fn resblocks_spelling_and_fused_qkv_naming_normalize() {
    assert_eq!(
        normalize_key("vision_model.transformer.resblocks.3.attn.in_proj_weight"),
        "vision_model.transformer.3.attn.in_proj.weight"
    );
    assert_eq!(
        normalize_key("vision_model.transformer.resblocks.3.attn.in_proj_bias"),
        "vision_model.transformer.3.attn.in_proj.bias"
    );
    // Bare transformer keys also land under vision_model.
    assert_eq!(
        normalize_key("transformer.resblocks.0.ln_1.weight"),
        "vision_model.transformer.0.ln_1.weight"
    );
}
