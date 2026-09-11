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

//! Tests for the GOT-OCR 2.0 loader's key canonicalizer and config shaping.

use super::*;
use serde_json::json;

/// Every distinct key shape in `stepfun-ai/GOT-OCR2_0`, taken from the
/// checkpoint's own `model.safetensors` header.
const ORIGINAL_KEYS: &[&str] = &[
    "model.vision_tower_high.patch_embed.proj.weight",
    "model.vision_tower_high.patch_embed.proj.bias",
    "model.vision_tower_high.pos_embed",
    "model.vision_tower_high.blocks.0.norm1.weight",
    "model.vision_tower_high.blocks.0.norm1.bias",
    "model.vision_tower_high.blocks.0.attn.qkv.weight",
    "model.vision_tower_high.blocks.0.attn.qkv.bias",
    "model.vision_tower_high.blocks.0.attn.proj.weight",
    "model.vision_tower_high.blocks.0.attn.rel_pos_h",
    "model.vision_tower_high.blocks.0.attn.rel_pos_w",
    "model.vision_tower_high.blocks.0.mlp.lin1.weight",
    "model.vision_tower_high.blocks.11.mlp.lin2.bias",
    "model.vision_tower_high.neck.0.weight",
    "model.vision_tower_high.neck.1.weight",
    "model.vision_tower_high.neck.1.bias",
    "model.vision_tower_high.neck.2.weight",
    "model.vision_tower_high.neck.3.weight",
    "model.vision_tower_high.neck.3.bias",
    "model.vision_tower_high.net_2.weight",
    "model.vision_tower_high.net_3.weight",
    "model.mm_projector_vary.weight",
    "model.mm_projector_vary.bias",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.bias",
    "model.layers.23.mlp.down_proj.weight",
    "lm_head.weight",
];

/// The same network as `mlx-community/GOT-OCR2_0-bf16` spells it. The neck's
/// four `nn.Sequential` indices became `conv1 / norm1 / conv2 / norm2`, the
/// prefixes changed, and there is no `lm_head`.
const CONVERTED_KEYS: &[&str] = &[
    "vision_tower.patch_embed.proj.weight",
    "vision_tower.patch_embed.proj.bias",
    "vision_tower.pos_embed",
    "vision_tower.blocks.0.norm1.weight",
    "vision_tower.blocks.0.norm1.bias",
    "vision_tower.blocks.0.attn.qkv.weight",
    "vision_tower.blocks.0.attn.qkv.bias",
    "vision_tower.blocks.0.attn.proj.weight",
    "vision_tower.blocks.0.attn.rel_pos_h",
    "vision_tower.blocks.0.attn.rel_pos_w",
    "vision_tower.blocks.0.mlp.lin1.weight",
    "vision_tower.blocks.11.mlp.lin2.bias",
    "vision_tower.conv1.weight",
    "vision_tower.norm1.weight",
    "vision_tower.norm1.bias",
    "vision_tower.conv2.weight",
    "vision_tower.norm2.weight",
    "vision_tower.norm2.bias",
    "vision_tower.net_2.weight",
    "vision_tower.net_3.weight",
    "multi_modal_projector.weight",
    "multi_modal_projector.bias",
    "language_model.model.embed_tokens.weight",
    "language_model.model.norm.weight",
    "language_model.model.layers.0.self_attn.q_proj.weight",
    "language_model.model.layers.0.self_attn.q_proj.bias",
    "language_model.model.layers.23.mlp.down_proj.weight",
];

fn weight_map(keys: &[&str]) -> WeightMap {
    let mut map = WeightMap::new();
    for key in keys {
        map.insert(
            (*key).to_string(),
            mlxcel_core::from_slice_f32(&[0.0], &[1]),
        );
    }
    map
}

fn sorted_keys(map: &WeightMap) -> Vec<String> {
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    keys
}

/// The two released layouts canonicalize onto the identical key set.
///
/// The original ships a tied `lm_head.weight` the conversion dropped, so the
/// comparison is made with the drop enabled, which is what the loader does when
/// `tie_word_embeddings` is set (it is, on every GOT checkpoint).
#[test]
fn canonicalize_got_keys_maps_both_layouts_to_one() {
    let original = canonicalize_got_keys(weight_map(ORIGINAL_KEYS), true);
    let converted = canonicalize_got_keys(weight_map(CONVERTED_KEYS), true);
    assert_eq!(sorted_keys(&original), sorted_keys(&converted));
    assert_eq!(original.len(), CONVERTED_KEYS.len());
}

/// The canonical names are the ones the loader's three consumers read:
/// `SamEncoder` at `vision_tower` (with the neck at its `neck.N` indices),
/// `UnifiedLinear` at `multi_modal_projector`, and the Llama backbone after
/// `strip_language_model_prefix` turns `language_model.model.*` into `model.*`.
#[test]
fn canonical_names_match_what_each_consumer_reads() {
    let map = canonicalize_got_keys(weight_map(ORIGINAL_KEYS), true);
    for key in [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.pos_embed",
        "vision_tower.blocks.0.attn.qkv.weight",
        "vision_tower.neck.0.weight",
        "vision_tower.neck.1.bias",
        "vision_tower.neck.2.weight",
        "vision_tower.neck.3.bias",
        "vision_tower.net_2.weight",
        "vision_tower.net_3.weight",
        "multi_modal_projector.weight",
        "multi_modal_projector.bias",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.bias",
    ] {
        assert!(map.contains_key(key), "missing {key}");
    }

    let text = strip_language_model_prefix(map);
    assert!(text.contains_key("model.embed_tokens.weight"));
    assert!(text.contains_key("model.layers.23.mlp.down_proj.weight"));
}

/// The converted layout's neck names are rewritten to the indices
/// `SamEncoder::from_weights` reads, and are gone afterwards.
#[test]
fn converted_neck_names_are_rewritten_to_sequential_indices() {
    let map = canonicalize_got_keys(weight_map(CONVERTED_KEYS), true);
    for key in [
        "vision_tower.conv1.weight",
        "vision_tower.conv2.weight",
        "vision_tower.norm2.weight",
    ] {
        assert!(!map.contains_key(key), "{key} should have been renamed");
    }
    assert!(map.contains_key("vision_tower.neck.0.weight"));
    assert!(map.contains_key("vision_tower.neck.2.weight"));
    assert!(map.contains_key("vision_tower.neck.3.weight"));
    // The per-block `norm1` / `norm2` are a different thing entirely and must
    // survive the neck rename untouched.
    assert!(map.contains_key("vision_tower.blocks.0.norm1.weight"));
    assert!(map.contains_key("vision_tower.blocks.0.norm1.bias"));
}

/// Running the canonicalizer twice equals running it once, on both layouts.
///
/// This is what makes it safe to apply on an already-converted checkpoint: the
/// second pass must not, for instance, re-prefix `language_model.model.*` into
/// `language_model.language_model.model.*`.
#[test]
fn sanitize_is_idempotent() {
    for keys in [ORIGINAL_KEYS, CONVERTED_KEYS] {
        let once = canonicalize_got_keys(weight_map(keys), true);
        let expected = sorted_keys(&once);
        let twice = canonicalize_got_keys(once, true);
        assert_eq!(sorted_keys(&twice), expected);
    }
}

/// The tied `lm_head` copy is dropped only when asked. An untied checkpoint
/// (none released, but the flag is read from the config rather than assumed)
/// keeps its head.
#[test]
fn tied_lm_head_is_dropped_only_when_requested() {
    let dropped = canonicalize_got_keys(weight_map(ORIGINAL_KEYS), true);
    assert!(!dropped.contains_key("language_model.lm_head.weight"));

    let kept = canonicalize_got_keys(weight_map(ORIGINAL_KEYS), false);
    assert!(kept.contains_key("language_model.lm_head.weight"));
}

/// The decoder args come from the flat root with `model_type` rewritten, and a
/// converted checkpoint's root `quantization` block survives into them.
#[test]
fn text_config_is_the_flat_root_relabelled_as_qwen2() {
    let full = json!({
        "model_type": "GOT",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "intermediate_size": 2816,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": 151860,
        "tie_word_embeddings": true,
        "image_token_len": 256,
        "vision_config": {},
        "quantization": {"group_size": 64, "bits": 4},
        "quantization_config": {"group_size": 64, "bits": 4},
    });
    let text = got_text_config(&full).expect("text config");
    assert_eq!(
        text.get("model_type").and_then(|v| v.as_str()),
        Some("qwen2")
    );
    assert!(text.get("vision_config").is_none());
    assert!(text.get("quantization_config").is_none());
    assert_eq!(
        text.get("quantization")
            .and_then(|q| q.get("bits"))
            .and_then(|b| b.as_i64()),
        Some(4)
    );

    let args: crate::models::llama3::ModelArgs =
        serde_json::from_value(text).expect("parse as llama-family args");
    assert_eq!(args.group_size(), 64);
    assert_eq!(args.bits(), 4);
    assert!(args.tie_word_embeddings);
}

/// `<|im_end|>` joins the stop set even though no config file names it.
///
/// Upstream stops on the separator through a `KeywordsStoppingCriteria`, not
/// through `eos_token_id`, so a run gated on the declared 151643 alone fills
/// `max_tokens` instead of terminating.
#[test]
fn stop_set_adds_the_turn_separator() {
    let full = json!({"eos_token_id": 151643});
    assert_eq!(got_eos_token_ids(&full, None), vec![151643, 151645]);

    // A list-valued field and a generation_config are both accepted, and the
    // separator is not duplicated when it is already declared.
    let full = json!({"eos_token_id": [151643, 151645]});
    assert_eq!(got_eos_token_ids(&full, None), vec![151643, 151645]);

    let full = json!({"eos_token_id": 151643});
    let generation = json!({"eos_token_id": 151643, "max_new_tokens": 2048});
    assert_eq!(
        got_eos_token_ids(&full, Some(&generation)),
        vec![151643, 151645]
    );

    // A config with no eos field at all still yields both stops.
    assert_eq!(got_eos_token_ids(&json!({}), None), vec![151643, 151645]);
}
