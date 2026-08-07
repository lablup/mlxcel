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

//! Unit tests for the Jina VLM vision tower and VL connector.

use super::{JinaVlmVisionConfig, JinaVlmVisionModel};
use mlxcel_core::MlxArray;
use mlxcel_core::weights::WeightMap;

#[test]
fn negative_vit_layers_resolve_against_the_post_norm_slot() {
    // The reference appends the `post_lnorm` output to `hidden_states`, so the
    // list is 28 long for a 27-layer tower and `[-4, -10]` selects layers 24 and
    // 18. Resolving against 27 instead would take 23 and 17 and silently shift
    // both halves of the connector's 2304-wide input.
    let config = JinaVlmVisionConfig::default();
    assert_eq!(config.hidden_state_slots(), 28);
    assert_eq!(config.resolved_vit_layers().unwrap(), vec![24, 18]);
}

#[test]
fn vit_layer_order_is_preserved_not_sorted() {
    let config = JinaVlmVisionConfig {
        vit_layers: vec![-10, -4],
        ..Default::default()
    };
    assert_eq!(config.resolved_vit_layers().unwrap(), vec![18, 24]);
}

#[test]
fn a_tower_without_post_norm_shifts_the_resolved_indices() {
    let config = JinaVlmVisionConfig {
        post_layer_norm: false,
        ..Default::default()
    };
    assert_eq!(config.hidden_state_slots(), 27);
    assert_eq!(config.resolved_vit_layers().unwrap(), vec![23, 17]);
}

#[test]
fn an_out_of_range_vit_layer_is_rejected_rather_than_clamped() {
    let config = JinaVlmVisionConfig {
        vit_layers: vec![-40],
        ..Default::default()
    };
    let err = config.resolved_vit_layers().unwrap_err();
    assert!(err.contains("outside"), "unexpected message: {err}");
}

#[test]
fn the_released_geometry_matches_the_checkpoint_tensor_shapes() {
    let config = JinaVlmVisionConfig::default();
    // 378 / 14 = 27 patches per side; 14*14*3 = 588 is `patch_embed.proj`'s
    // input width; 27 -> 14 pooled tokens per side is `tokens_per_image = 196`.
    assert_eq!(config.crop_patches(), 27);
    assert_eq!(config.patch_dim(), 588);
    assert_eq!(config.token_length(), (14, 14));
    // `pooling.q` / `pooling.kv` take the two concatenated ViT layers.
    assert_eq!(config.hidden_size * config.vit_layers.len() as i32, 2304);
    // `pooling.kv` emits K and V for 16 heads of 72.
    assert_eq!(2 * config.pooling_num_heads * config.pooling_head_dim, 2304);
}

// Synthetic tower + connector.
fn tiny_config() -> JinaVlmVisionConfig {
    JinaVlmVisionConfig {
        hidden_size: 4,
        num_hidden_layers: 3,
        num_attention_heads: 1,
        head_dim: 4,
        patch_size: 2,
        image_size: 4,
        num_channels: 3,
        intermediate_size: 6,
        layer_norm_eps: 1e-6,
        use_cls_token: false,
        post_layer_norm: false,
        // hidden_state_slots = 3 -> layers 2 and 1.
        vit_layers: vec![-1, -2],
        output_size: 8,
        pooling_num_heads: 1,
        pooling_head_dim: 4,
        connector_hidden_size: 6,
        pooling_h: 2,
        pooling_w: 2,
        group_size: 64,
        bits: 4,
    }
}

/// Synthetic weights at unit amplitude. Quarter-amplitude values look
/// harmless but make the connector's SwiGLU squash its (already small, because
/// the pooling query is a mean) input into the 1e-8 range, where a real change
/// in the output is indistinguishable from float noise.
fn deterministic(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.29 + seed).cos()).collect()
}

fn insert(wm: &mut WeightMap, key: &str, data: Vec<f32>, shape: &[i32]) {
    wm.insert(key.to_string(), mlxcel_core::from_slice_f32(&data, shape));
}

pub(crate) fn tiny_vision_weights(config: &JinaVlmVisionConfig) -> WeightMap {
    let mut wm = WeightMap::new();
    let hidden = config.hidden_size;
    let inner = config.num_attention_heads * config.head_dim;
    let patches = config.crop_patches() * config.crop_patches();

    insert(
        &mut wm,
        "vision_model.patch_embed.proj.weight",
        deterministic((hidden * config.patch_dim()) as usize, 0.1),
        &[hidden, config.patch_dim()],
    );
    insert(
        &mut wm,
        "vision_model.patch_embed.proj.bias",
        vec![0.0; hidden as usize],
        &[hidden],
    );
    insert(
        &mut wm,
        "vision_model.pos_embed",
        deterministic((patches * hidden) as usize, 0.2),
        &[patches, hidden],
    );

    for layer in 0..config.num_hidden_layers {
        let p = format!("vision_model.layers.{layer}");
        for norm in ["attn_norm", "ffn_norm"] {
            insert(
                &mut wm,
                &format!("{p}.{norm}.weight"),
                vec![1.0; hidden as usize],
                &[hidden],
            );
            insert(
                &mut wm,
                &format!("{p}.{norm}.bias"),
                vec![0.0; hidden as usize],
                &[hidden],
            );
        }
        insert(
            &mut wm,
            &format!("{p}.attn.qkv.weight"),
            deterministic((3 * inner * hidden) as usize, layer as f32 + 0.3),
            &[3 * inner, hidden],
        );
        insert(
            &mut wm,
            &format!("{p}.attn.qkv.bias"),
            vec![0.0; (3 * inner) as usize],
            &[3 * inner],
        );
        insert(
            &mut wm,
            &format!("{p}.attn.out.weight"),
            deterministic((hidden * inner) as usize, layer as f32 + 0.4),
            &[hidden, inner],
        );
        insert(
            &mut wm,
            &format!("{p}.attn.out.bias"),
            vec![0.0; hidden as usize],
            &[hidden],
        );
        insert(
            &mut wm,
            &format!("{p}.ffn.up.weight"),
            deterministic(
                (config.intermediate_size * hidden) as usize,
                layer as f32 + 0.5,
            ),
            &[config.intermediate_size, hidden],
        );
        insert(
            &mut wm,
            &format!("{p}.ffn.up.bias"),
            vec![0.0; config.intermediate_size as usize],
            &[config.intermediate_size],
        );
        insert(
            &mut wm,
            &format!("{p}.ffn.down.weight"),
            deterministic(
                (hidden * config.intermediate_size) as usize,
                layer as f32 + 0.6,
            ),
            &[hidden, config.intermediate_size],
        );
        insert(
            &mut wm,
            &format!("{p}.ffn.down.bias"),
            vec![0.0; hidden as usize],
            &[hidden],
        );
    }

    let concat = hidden * config.vit_layers.len() as i32;
    let pool_inner = config.pooling_num_heads * config.pooling_head_dim;
    insert(
        &mut wm,
        "vl_connector.pad_embed",
        deterministic((2 * concat) as usize, 0.9),
        &[2, concat],
    );
    insert(
        &mut wm,
        "vl_connector.pooling.q.weight",
        deterministic((pool_inner * concat) as usize, 0.11),
        &[pool_inner, concat],
    );
    insert(
        &mut wm,
        "vl_connector.pooling.q.bias",
        vec![0.0; pool_inner as usize],
        &[pool_inner],
    );
    insert(
        &mut wm,
        "vl_connector.pooling.kv.weight",
        deterministic((2 * pool_inner * concat) as usize, 0.12),
        &[2 * pool_inner, concat],
    );
    insert(
        &mut wm,
        "vl_connector.pooling.kv.bias",
        vec![0.0; (2 * pool_inner) as usize],
        &[2 * pool_inner],
    );
    insert(
        &mut wm,
        "vl_connector.pooling.out.weight",
        deterministic((hidden * pool_inner) as usize, 0.13),
        &[hidden, pool_inner],
    );
    insert(
        &mut wm,
        "vl_connector.pooling.out.bias",
        vec![0.0; hidden as usize],
        &[hidden],
    );
    insert(
        &mut wm,
        "vl_connector.projector.gate_up.weight",
        deterministic((2 * config.connector_hidden_size * hidden) as usize, 0.14),
        &[2 * config.connector_hidden_size, hidden],
    );
    insert(
        &mut wm,
        "vl_connector.projector.down.weight",
        deterministic(
            (config.output_size * config.connector_hidden_size) as usize,
            0.15,
        ),
        &[config.output_size, config.connector_hidden_size],
    );
    wm
}

pub(crate) fn build_tiny_vision_model() -> (JinaVlmVisionConfig, JinaVlmVisionModel) {
    let config = tiny_config();
    let weights = tiny_vision_weights(&config);
    let model =
        JinaVlmVisionModel::from_weights(&weights, "vision_model", "vl_connector", config.clone())
            .expect("tiny vision model builds");
    (config, model)
}

/// The rejection message, or a panic naming what was wrongly accepted.
/// `JinaVlmVisionModel` is not `Debug`, so `expect_err` is unavailable.
fn vision_error(result: Result<JinaVlmVisionModel, String>, what: &str) -> String {
    match result {
        Ok(_) => panic!("{what}"),
        Err(e) => e,
    }
}

#[test]
fn a_pos_embed_that_is_not_a_two_dimensional_table_is_rejected() {
    // `forward_features` broadcasts the table with
    // `reshape(pos_embed, [1, pe_shape[0], pe_shape[1]])`. A 1-D tensor is a
    // Rust panic on `pe_shape[1]`, and a 3-D one drops elements and aborts
    // inside MLX's reshape; both happen at the first image request, long after
    // the checkpoint appeared to load.
    let config = tiny_config();
    let flat = config.crop_patches() * config.crop_patches() * config.hidden_size;

    for shape in [
        vec![flat],
        vec![1, flat / config.hidden_size, config.hidden_size],
    ] {
        let mut weights = tiny_vision_weights(&config);
        insert(
            &mut weights,
            "vision_model.pos_embed",
            deterministic(flat as usize, 0.2),
            &shape,
        );
        let err = vision_error(
            JinaVlmVisionModel::from_weights(
                &weights,
                "vision_model",
                "vl_connector",
                config.clone(),
            ),
            "a mis-ranked pos_embed was accepted",
        );
        assert!(
            err.contains("pos_embed") && err.contains("2-D"),
            "shape {shape:?}: unexpected message: {err}"
        );
    }
}

#[test]
fn a_pad_embed_without_exactly_two_rows_is_rejected_but_absence_is_allowed() {
    let config = tiny_config();
    let concat = config.hidden_size * config.vit_layers.len() as i32;

    // One row: `slice(pad_embed, [1, 0], [2, feat])` is silently clamped by MLX
    // to a zero-row result and the following reshape aborts the process.
    // 1-D: a Rust panic on `pe_shape[1]`.
    for shape in [vec![1, concat], vec![2 * concat]] {
        let count: i32 = shape.iter().product();
        let mut weights = tiny_vision_weights(&config);
        insert(
            &mut weights,
            "vl_connector.pad_embed",
            deterministic(count as usize, 0.9),
            &shape,
        );
        let err = vision_error(
            JinaVlmVisionModel::from_weights(
                &weights,
                "vision_model",
                "vl_connector",
                config.clone(),
            ),
            "a mis-shaped pad_embed was accepted",
        );
        assert!(
            err.contains("pad_embed"),
            "shape {shape:?}: unexpected message: {err}"
        );
    }

    // A checkpoint that ships no pad embedding at all is still loadable; see the
    // field's doc comment for what that costs.
    let mut weights = tiny_vision_weights(&config);
    weights.remove("vl_connector.pad_embed");
    JinaVlmVisionModel::from_weights(&weights, "vision_model", "vl_connector", config)
        .expect("pad_embed is optional on purpose");
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn the_connector_emits_one_pooled_token_per_window_in_the_text_hidden_size() {
    let (config, model) = build_tiny_vision_model();
    let crops = 2i32;
    let patches = config.crop_patches() * config.crop_patches();
    let pixels: Vec<f32> = deterministic((crops * patches * config.patch_dim()) as usize, 0.5);
    let images = mlxcel_core::from_slice_f32(&pixels, &[1, crops, patches, config.patch_dim()]);
    let masks =
        mlxcel_core::from_slice_f32(&vec![1.0; (crops * patches) as usize], &[1, crops, patches]);

    let out = model.forward(&images, &masks);
    let (h, w) = config.token_length();
    assert_eq!(
        mlxcel_core::array_shape(&out),
        vec![1, crops, h * w, config.output_size]
    );
    assert!(to_vec_f32(&out).iter().all(|v| v.is_finite()));
}

#[test]
fn an_all_negative_one_crop_is_zeroed_before_the_connector() {
    // Upstream marks a fully padded crop with `-1` pixels and multiplies its
    // features by zero. The pad embedding is still added afterwards, so the
    // check is against the same crop run with a live mask, not against zero.
    let (config, model) = build_tiny_vision_model();
    let crops = 2i32;
    let patches = config.crop_patches() * config.crop_patches();
    let per_crop = (patches * config.patch_dim()) as usize;

    let mut pixels: Vec<f32> = deterministic(2 * per_crop, 0.5);
    let live = mlxcel_core::from_slice_f32(&pixels, &[1, crops, patches, config.patch_dim()]);
    pixels[per_crop..].iter_mut().for_each(|v| *v = -1.0);
    let padded = mlxcel_core::from_slice_f32(&pixels, &[1, crops, patches, config.patch_dim()]);

    let masks =
        mlxcel_core::from_slice_f32(&vec![1.0; (crops * patches) as usize], &[1, crops, patches]);

    let live_out = to_vec_f32(&model.forward(&live, &masks));
    let padded_out = to_vec_f32(&model.forward(&padded, &masks));

    let (h, w) = config.token_length();
    let per_crop_tokens = (h * w * config.output_size) as usize;
    // Crop 0 is untouched by crop 1's padding.
    for i in 0..per_crop_tokens {
        assert!(
            (live_out[i] - padded_out[i]).abs() < 1e-4,
            "the first crop changed when the second became padding"
        );
    }
    // Crop 1 collapsed to exactly zero: the projector has no biases, so a
    // zeroed feature block cannot produce anything else.
    assert!(
        padded_out[per_crop_tokens..].iter().all(|&v| v == 0.0),
        "the padded crop was not zeroed: {:?}",
        &padded_out[per_crop_tokens..]
    );
    assert!(
        live_out[per_crop_tokens..].iter().any(|&v| v.abs() > 1e-4),
        "the control crop was already zero, so the check proves nothing"
    );
}

#[test]
fn the_pad_embedding_only_fires_where_the_mask_says_padding() {
    let (config, model) = build_tiny_vision_model();
    let crops = 1i32;
    let patches = config.crop_patches() * config.crop_patches();
    let pixels: Vec<f32> = deterministic((crops * patches * config.patch_dim()) as usize, 0.5);
    let images = mlxcel_core::from_slice_f32(&pixels, &[1, crops, patches, config.patch_dim()]);

    let covered =
        mlxcel_core::from_slice_f32(&vec![1.0; (crops * patches) as usize], &[1, crops, patches]);
    let sentinel = mlxcel_core::from_slice_f32(
        &vec![-1.0; (crops * patches) as usize],
        &[1, crops, patches],
    );

    let a = to_vec_f32(&model.forward(&images, &covered));
    let b = to_vec_f32(&model.forward(&images, &sentinel));
    // A `-1` mask entry is `< 1` and `!= 0`, so it selects `pad_embed[1]`.
    assert!(
        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4),
        "the partial-pad embedding was never applied"
    );
}
