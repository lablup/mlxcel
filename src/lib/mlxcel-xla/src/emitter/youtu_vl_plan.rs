// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Static image-grid and window-attention contract for Youtu-VL.

use std::collections::HashSet;
use std::path::Path;

use serde_json::Value;

pub(crate) const YOUTU_VL_PATCH_BUCKETS: [usize; 3] = [16, 64, 256];
const MAX_IMAGES: usize = 4;
const DEFAULT_IMAGE_TOKEN_ID: i32 = 128_264;
const DEFAULT_VIDEO_TOKEN_ID: i32 = 128_265;
const DEFAULT_VISION_START_TOKEN_ID: i32 = 128_262;
const DEFAULT_VISION_END_TOKEN_ID: i32 = 128_263;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct YoutuVlWeightSpec {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct YoutuVlVisionConfig {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) channels: usize,
    pub(crate) patch_size: usize,
    pub(crate) spatial_merge_size: usize,
    pub(crate) window_size: usize,
    pub(crate) full_attention_layers: Vec<usize>,
    pub(crate) text_hidden: usize,
    pub(crate) layer_norm_eps: f32,
    pub(crate) max_patches_per_image: usize,
    pub(crate) resample: u8,
    pub(crate) image_token_id: i32,
    pub(crate) video_token_id: i32,
    pub(crate) vision_start_token_id: i32,
    pub(crate) vision_end_token_id: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct YoutuVlGridPlan {
    pub(crate) patch_bucket: usize,
    pub(crate) actual_patches: usize,
    pub(crate) merged_tokens_per_image: Vec<usize>,
    pub(crate) window_group_index: Vec<usize>,
    pub(crate) reverse_group_index: Vec<usize>,
    pub(crate) window_cu_seqlens: Vec<usize>,
    pub(crate) full_cu_seqlens: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct YoutuVlHostInputs {
    pub(crate) plan: YoutuVlGridPlan,
    pub(crate) patches: Vec<f32>,
    pub(crate) rope_freqs: Vec<f32>,
    pub(crate) window_attention_bias: Vec<f32>,
    pub(crate) full_attention_bias: Vec<f32>,
}

fn positive(object: &serde_json::Map<String, Value>, field: &str) -> Result<usize, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or_else(|| format!("Youtu-VL config field {field} must be a positive integer"))
}

impl YoutuVlVisionConfig {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&config_path)
            .map_err(|error| format!("{}: {error}", config_path.display()))?;
        let mut config = Self::from_json_str(&text)?;
        let processor_path = model_dir.join("preprocessor_config.json");
        let processor_text = std::fs::read_to_string(&processor_path)
            .map_err(|error| format!("{}: {error}", processor_path.display()))?;
        let processor: Value = serde_json::from_str(&processor_text)
            .map_err(|error| format!("parse {}: {error}", processor_path.display()))?;
        config.max_patches_per_image = processor
            .get("max_num_patches")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                "Youtu-VL preprocessor_config.json requires positive max_num_patches".to_string()
            })?;
        config.resample = processor
            .get("resample")
            .and_then(Value::as_u64)
            .and_then(|value| u8::try_from(value).ok())
            .ok_or_else(|| {
                "Youtu-VL preprocessor_config.json requires byte-sized resample".to_string()
            })?;
        if config.max_patches_per_image > YOUTU_VL_PATCH_BUCKETS[2] {
            return Err(format!(
                "Youtu-VL processor cap {} exceeds qualified XLA capacity {}",
                config.max_patches_per_image, YOUTU_VL_PATCH_BUCKETS[2]
            ));
        }
        Ok(config)
    }

    pub(crate) fn from_json_str(text: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| format!("parse config.json: {error}"))?;
        if root.get("model_type").and_then(Value::as_str) != Some("youtu_vl") {
            return Err("Youtu-VL IREE vision requires model_type=youtu_vl".to_string());
        }
        if root
            .get("quantization")
            .is_some_and(|value| !value.is_null())
        {
            return Err("Youtu-VL IREE vision currently requires BF16/F16/F32 weights".to_string());
        }
        let vision = root
            .get("vision_config")
            .and_then(Value::as_object)
            .ok_or_else(|| "Youtu-VL config.json vision_config must be an object".to_string())?;
        let depth = positive(vision, "num_hidden_layers")?;
        let hidden = positive(vision, "hidden_size")?;
        let heads = positive(vision, "num_attention_heads")?;
        if !hidden.is_multiple_of(heads) || !(hidden / heads).is_multiple_of(4) {
            return Err("Youtu-VL vision head dimension must be divisible by four".to_string());
        }
        let spatial_merge_size = vision
            .get("spatial_merge_size")
            .and_then(Value::as_u64)
            .map_or(2usize, |value| value as usize);
        if spatial_merge_size != 2 {
            return Err(format!(
                "Youtu-VL IREE vision qualifies spatial_merge_size=2, got {spatial_merge_size}"
            ));
        }
        let patch_size = positive(vision, "patch_size")?;
        let window_size = positive(vision, "window_size")?;
        if !window_size.is_multiple_of(patch_size * spatial_merge_size) {
            return Err(
                "Youtu-VL window_size must be divisible by patch_size * spatial_merge_size"
                    .to_string(),
            );
        }
        let full_attention_layers = vision
            .get("fullatt_block_indexes")
            .and_then(Value::as_array)
            .ok_or_else(|| "Youtu-VL fullatt_block_indexes must be an array".to_string())?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .filter(|&value| value < depth)
                    .ok_or_else(|| "invalid Youtu-VL full-attention layer".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if full_attention_layers.is_empty() {
            return Err("Youtu-VL requires at least one full-attention layer".to_string());
        }
        Ok(Self {
            depth,
            hidden,
            intermediate: positive(vision, "intermediate_size")?,
            heads,
            channels: positive(vision, "num_channels")?,
            patch_size,
            spatial_merge_size,
            window_size,
            full_attention_layers,
            text_hidden: positive(vision, "out_hidden_size")?,
            layer_norm_eps: vision
                .get("layer_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            max_patches_per_image: positive(vision, "num_patches")?,
            resample: 2,
            image_token_id: token_id(&root, "image_token_id", DEFAULT_IMAGE_TOKEN_ID)?,
            video_token_id: token_id(&root, "video_token_id", DEFAULT_VIDEO_TOKEN_ID)?,
            vision_start_token_id: token_id(
                &root,
                "vision_start_token_id",
                DEFAULT_VISION_START_TOKEN_ID,
            )?,
            vision_end_token_id: token_id(
                &root,
                "vision_end_token_id",
                DEFAULT_VISION_END_TOKEN_ID,
            )?,
        })
    }

    pub(crate) fn fingerprint(&self, bucket: usize) -> String {
        format!(
            "youtu-vl-vision-v1:bucket={bucket}:patch={}:merge={}:window={}:channels={}:hidden={}:intermediate={}:heads={}:depth={}:full={:?}:text={}:eps={}:max_patches={}:resample={}:image_token={}:video_token={}:vision_start={}:vision_end={}:dtype=f32",
            self.patch_size,
            self.spatial_merge_size,
            self.window_size,
            self.channels,
            self.hidden,
            self.intermediate,
            self.heads,
            self.depth,
            self.full_attention_layers,
            self.text_hidden,
            self.layer_norm_eps,
            self.max_patches_per_image,
            self.resample,
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
    }

    pub(crate) fn plan(&self, shapes: &[(i32, i32)]) -> Result<YoutuVlGridPlan, String> {
        if shapes.is_empty() || shapes.len() > MAX_IMAGES {
            return Err(format!(
                "Youtu-VL XLA requires 1..={MAX_IMAGES} images, got {}",
                shapes.len()
            ));
        }
        let merge = self.spatial_merge_size;
        let window = self.window_size / self.patch_size / merge;
        let merge_unit = merge * merge;
        let mut total_patches = 0usize;
        let mut merged_tokens_per_image = Vec::with_capacity(shapes.len());
        let mut window_group_index = Vec::new();
        let mut window_cu_seqlens = vec![0usize];
        let mut full_cu_seqlens = vec![0usize];
        let mut group_offset = 0usize;
        for (index, &(height, width)) in shapes.iter().enumerate() {
            let height = usize::try_from(height)
                .map_err(|_| format!("Youtu-VL image {index} has invalid height {height}"))?;
            let width = usize::try_from(width)
                .map_err(|_| format!("Youtu-VL image {index} has invalid width {width}"))?;
            if height == 0
                || width == 0
                || !height.is_multiple_of(merge)
                || !width.is_multiple_of(merge)
            {
                return Err(format!(
                    "Youtu-VL image {index} grid {height}x{width} must be positive and divisible by merge={merge}"
                ));
            }
            let patches = height
                .checked_mul(width)
                .ok_or_else(|| "Youtu-VL image patch count overflowed".to_string())?;
            if patches > self.max_patches_per_image {
                return Err(format!(
                    "Youtu-VL image {index} has {patches} patches, exceeding cap {}",
                    self.max_patches_per_image
                ));
            }
            total_patches = total_patches
                .checked_add(patches)
                .ok_or_else(|| "Youtu-VL packed patch count overflowed".to_string())?;
            let groups_h = height / merge;
            let groups_w = width / merge;
            let groups = groups_h * groups_w;
            merged_tokens_per_image.push(groups);
            full_cu_seqlens.push(total_patches);
            for window_h in (0..groups_h).step_by(window) {
                for window_w in (0..groups_w).step_by(window) {
                    let before = window_group_index.len();
                    for h in window_h..(window_h + window).min(groups_h) {
                        for w in window_w..(window_w + window).min(groups_w) {
                            window_group_index.push(group_offset + h * groups_w + w);
                        }
                    }
                    let group_count = window_group_index.len() - before;
                    window_cu_seqlens.push(
                        window_cu_seqlens
                            .last()
                            .copied()
                            .unwrap()
                            .checked_add(group_count * merge_unit)
                            .ok_or_else(|| "Youtu-VL window boundary overflowed".to_string())?,
                    );
                }
            }
            group_offset += groups;
        }
        let patch_bucket = YOUTU_VL_PATCH_BUCKETS
            .into_iter()
            .find(|&bucket| total_patches <= bucket)
            .ok_or_else(|| {
                format!(
                    "Youtu-VL packed grid has {total_patches} patches, exceeding qualified capacity {}",
                    YOUTU_VL_PATCH_BUCKETS[2]
                )
            })?;
        let mut reverse_group_index = vec![0usize; window_group_index.len()];
        for (window_position, &original_position) in window_group_index.iter().enumerate() {
            reverse_group_index[original_position] = window_position;
        }
        Ok(YoutuVlGridPlan {
            patch_bucket,
            actual_patches: total_patches,
            merged_tokens_per_image,
            window_group_index,
            reverse_group_index,
            window_cu_seqlens,
            full_cu_seqlens,
        })
    }

    pub(crate) fn weight_specs(&self) -> Vec<YoutuVlWeightSpec> {
        let patch_width = self.channels * self.patch_size * self.patch_size;
        let mut specs = vec![
            self.spec(
                "siglip2.vision_model.embeddings.patch_embedding.weight",
                [self.hidden, patch_width],
            ),
            self.spec(
                "siglip2.vision_model.embeddings.patch_embedding.bias",
                [self.hidden],
            ),
        ];
        for layer in 0..self.depth {
            let prefix = format!("siglip2.vision_model.encoder.layers.{layer}");
            for (suffix, shape) in [
                ("layer_norm1.weight", vec![self.hidden]),
                ("layer_norm1.bias", vec![self.hidden]),
                ("self_attn.q_proj.weight", vec![self.hidden, self.hidden]),
                ("self_attn.q_proj.bias", vec![self.hidden]),
                ("self_attn.k_proj.weight", vec![self.hidden, self.hidden]),
                ("self_attn.k_proj.bias", vec![self.hidden]),
                ("self_attn.v_proj.weight", vec![self.hidden, self.hidden]),
                ("self_attn.v_proj.bias", vec![self.hidden]),
                ("self_attn.out_proj.weight", vec![self.hidden, self.hidden]),
                ("self_attn.out_proj.bias", vec![self.hidden]),
                ("layer_norm2.weight", vec![self.hidden]),
                ("layer_norm2.bias", vec![self.hidden]),
                ("mlp.fc1.weight", vec![self.intermediate, self.hidden]),
                ("mlp.fc1.bias", vec![self.intermediate]),
                ("mlp.fc2.weight", vec![self.hidden, self.intermediate]),
                ("mlp.fc2.bias", vec![self.hidden]),
            ] {
                specs.push(YoutuVlWeightSpec {
                    name: format!("{prefix}.{suffix}"),
                    shape,
                });
            }
        }
        let merged = self.hidden * 4;
        specs.extend([
            self.spec("siglip2.vision_model.post_layernorm.weight", [self.hidden]),
            self.spec("siglip2.vision_model.post_layernorm.bias", [self.hidden]),
            self.spec("merger.ln_q.weight", [self.hidden]),
            self.spec("merger.mlp.0.weight", [merged, merged]),
            self.spec("merger.mlp.0.bias", [merged]),
            self.spec("merger.mlp.2.weight", [self.text_hidden, merged]),
            self.spec("merger.mlp.2.bias", [self.text_hidden]),
        ]);
        specs
    }

    fn spec<const N: usize>(&self, name: &str, shape: [usize; N]) -> YoutuVlWeightSpec {
        YoutuVlWeightSpec {
            name: name.to_string(),
            shape: shape.into(),
        }
    }
}

fn token_id(root: &Value, field: &str, default: i32) -> Result<i32, String> {
    match root.get(field) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_i64()
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| format!("Youtu-VL config field {field} must fit i32")),
    }
}

fn attention_bias(bucket: usize, boundaries: &[usize]) -> Vec<f32> {
    let mut bias = vec![f32::NEG_INFINITY; bucket * bucket];
    for segment in boundaries.windows(2) {
        for row in segment[0]..segment[1] {
            for column in segment[0]..segment[1] {
                bias[row * bucket + column] = 0.0;
            }
        }
    }
    for index in boundaries.last().copied().unwrap_or(0)..bucket {
        bias[index * bucket + index] = 0.0;
    }
    bias
}

pub(crate) fn prepare_youtu_vl_host_inputs(
    config: &YoutuVlVisionConfig,
    shapes: &[(i32, i32)],
    patch_rows: &[f32],
) -> Result<YoutuVlHostInputs, String> {
    let plan = config.plan(shapes)?;
    let patch_width = config.channels * config.patch_size * config.patch_size;
    if patch_rows.len() != plan.actual_patches * patch_width {
        return Err(format!(
            "Youtu-VL processor produced {} values, expected {}",
            patch_rows.len(),
            plan.actual_patches * patch_width
        ));
    }
    if patch_rows.iter().any(|value| !value.is_finite()) {
        return Err("Youtu-VL processor produced a non-finite patch value".to_string());
    }
    let merge_unit = config.spatial_merge_size * config.spatial_merge_size;
    let mut grouped_rows = Vec::with_capacity(patch_rows.len());
    let mut image_patch_offset = 0usize;
    for &(height, width) in shapes {
        let (height, width) = (height as usize, width as usize);
        for block_h in 0..height / config.spatial_merge_size {
            for block_w in 0..width / config.spatial_merge_size {
                for inner_h in 0..config.spatial_merge_size {
                    for inner_w in 0..config.spatial_merge_size {
                        let patch = image_patch_offset
                            + (block_h * config.spatial_merge_size + inner_h) * width
                            + block_w * config.spatial_merge_size
                            + inner_w;
                        let start = patch * patch_width;
                        grouped_rows.extend_from_slice(&patch_rows[start..start + patch_width]);
                    }
                }
            }
        }
        image_patch_offset += height * width;
    }
    let mut patches = Vec::with_capacity(plan.patch_bucket * patch_width);
    for &group in &plan.window_group_index {
        let start = group * merge_unit * patch_width;
        patches.extend_from_slice(&grouped_rows[start..start + merge_unit * patch_width]);
    }
    patches.resize(plan.patch_bucket * patch_width, 0.0);

    let head_dim = config.hidden / config.heads;
    let quarter = head_dim / 4;
    let inv = (0..quarter)
        .map(|index| 1.0 / 10_000.0f32.powf((2 * index) as f32 / (head_dim / 2) as f32))
        .collect::<Vec<_>>();
    let mut grouped_freqs = Vec::with_capacity(plan.actual_patches * head_dim / 2);
    for &(height, width) in shapes {
        let (height, width) = (height as usize, width as usize);
        for block_h in 0..height / 2 {
            for block_w in 0..width / 2 {
                for inner_h in 0..2 {
                    for inner_w in 0..2 {
                        let h = (block_h * 2 + inner_h) as f32;
                        let w = (block_w * 2 + inner_w) as f32;
                        grouped_freqs.extend(inv.iter().map(|frequency| h * frequency));
                        grouped_freqs.extend(inv.iter().map(|frequency| w * frequency));
                    }
                }
            }
        }
    }
    let mut rope_freqs = Vec::with_capacity(plan.patch_bucket * head_dim / 2);
    for &group in &plan.window_group_index {
        let start = group * merge_unit * head_dim / 2;
        rope_freqs.extend_from_slice(&grouped_freqs[start..start + merge_unit * head_dim / 2]);
    }
    rope_freqs.resize(plan.patch_bucket * head_dim / 2, 0.0);
    let window_attention_bias = attention_bias(plan.patch_bucket, &plan.window_cu_seqlens);
    let full_attention_bias = attention_bias(plan.patch_bucket, &plan.full_cu_seqlens);
    debug_assert_eq!(
        plan.window_group_index
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len(),
        plan.window_group_index.len()
    );
    Ok(YoutuVlHostInputs {
        plan,
        patches,
        rope_freqs,
        window_attention_bias,
        full_attention_bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> YoutuVlVisionConfig {
        YoutuVlVisionConfig {
            depth: 2,
            hidden: 16,
            intermediate: 32,
            heads: 4,
            channels: 3,
            patch_size: 2,
            spatial_merge_size: 2,
            window_size: 8,
            full_attention_layers: vec![1],
            text_hidden: 12,
            layer_norm_eps: 1e-6,
            max_patches_per_image: 256,
            resample: 2,
            image_token_id: DEFAULT_IMAGE_TOKEN_ID,
            video_token_id: DEFAULT_VIDEO_TOKEN_ID,
            vision_start_token_id: DEFAULT_VISION_START_TOKEN_ID,
            vision_end_token_id: DEFAULT_VISION_END_TOKEN_ID,
        }
    }

    #[test]
    fn plans_aspect_ratios_and_restores_window_group_order() {
        let plan = config().plan(&[(4, 8), (8, 4)]).unwrap();
        assert_eq!(plan.patch_bucket, 64);
        assert_eq!(plan.actual_patches, 64);
        assert_eq!(plan.merged_tokens_per_image, vec![8, 8]);
        assert_eq!(plan.full_cu_seqlens, vec![0, 32, 64]);
        assert_ne!(
            plan.window_group_index,
            (0..plan.window_group_index.len()).collect::<Vec<_>>()
        );
        let window_values = plan
            .window_group_index
            .iter()
            .map(|&original| format!("group-{original}"))
            .collect::<Vec<_>>();
        let restored = plan
            .reverse_group_index
            .iter()
            .map(|&window_position| window_values[window_position].as_str())
            .collect::<Vec<_>>();
        let expected = (0..16)
            .map(|original| format!("group-{original}"))
            .collect::<Vec<_>>();
        assert_eq!(
            restored,
            expected.iter().map(String::as_str).collect::<Vec<_>>()
        );
    }

    #[test]
    fn attention_bias_keeps_images_and_windows_isolated() {
        let width = config().channels * config().patch_size * config().patch_size;
        let inputs =
            prepare_youtu_vl_host_inputs(&config(), &[(4, 8), (8, 4)], &vec![0.0; 64 * width])
                .unwrap();
        let bucket = inputs.plan.patch_bucket;
        assert_eq!(inputs.full_attention_bias[0 * bucket + 31], 0.0);
        assert!(inputs.full_attention_bias[0 * bucket + 32].is_infinite());
        assert!(inputs.window_attention_bias[0 * bucket + 16].is_infinite());
        assert_eq!(
            inputs.window_attention_bias[(bucket - 1) * bucket + bucket - 1],
            0.0
        );
    }

    #[test]
    fn rejects_per_image_and_packed_bucket_overflow() {
        let mut capped = config();
        capped.max_patches_per_image = 15;
        assert!(
            capped
                .plan(&[(4, 4)])
                .unwrap_err()
                .contains("exceeding cap")
        );

        let error = config().plan(&[(16, 16), (4, 4)]).unwrap_err();
        assert!(error.contains("packed grid"));
        assert_eq!(config().plan(&[(16, 16)]).unwrap().patch_bucket, 256);
    }

    #[test]
    fn artifact_identity_binds_placeholder_ids() {
        let baseline = config();
        let mut changed = baseline.clone();
        changed.video_token_id += 1;
        assert_ne!(baseline.fingerprint(16), changed.fingerprint(16));
        assert!(baseline.fingerprint(16).contains("image_token=128264"));
    }

    #[test]
    fn host_patch_reorder_preserves_merge_group_membership() {
        let config = config();
        let width = config.channels * config.patch_size * config.patch_size;
        let mut rows = vec![0.0; 32 * width];
        for patch in 0..32 {
            rows[patch * width..(patch + 1) * width].fill(patch as f32);
        }
        let inputs = prepare_youtu_vl_host_inputs(&config, &[(4, 8)], &rows).unwrap();
        let observed = (0..inputs.plan.actual_patches)
            .map(|row| inputs.patches[row * width] as usize)
            .collect::<Vec<_>>();
        for (window_position, &group) in inputs.plan.window_group_index.iter().enumerate() {
            let group_rows = &observed[window_position * 4..window_position * 4 + 4];
            let group_h = group / 4;
            let group_w = group % 4;
            assert_eq!(
                group_rows,
                &[
                    group_h * 16 + group_w * 2,
                    group_h * 16 + group_w * 2 + 1,
                    group_h * 16 + group_w * 2 + 8,
                    group_h * 16 + group_w * 2 + 9,
                ]
            );
        }
    }
}
