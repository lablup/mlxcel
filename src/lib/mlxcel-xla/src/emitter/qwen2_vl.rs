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

//! Static-shape contract for the Qwen2-VL vision tower.
//!
//! The processor keeps the checkpoint's dynamic smart-resize policy, while the
//! compiled graph admits only this finite set of patch capacities. Inputs are
//! padded to the selected capacity and carry their actual patch count plus
//! packed-sequence boundaries. No graph is compiled for an arbitrary image
//! resolution.

use std::path::Path;

use serde_json::Value;

use super::builder::{Builder, Ty, Val};

/// Qualified flattened-patch capacities for the pinned Qwen2-VL image path.
///
/// Every capacity is divisible by `spatial_merge_size²` for the checkpoint's
/// 2x2 merger. Larger grids fail explicitly instead of triggering an unbounded
/// compile or falling back to the MLX vision encoder.
pub(crate) const QWEN2_VL_PATCH_BUCKETS: [usize; 3] = [16, 64, 256];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen2VlWeightSpec {
    pub(crate) name: String,
    /// Logical dequantized F32 shape consumed by StableHLO.
    pub(crate) shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen2VlConfig {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) patch_size: usize,
    pub(crate) temporal_patch_size: usize,
    pub(crate) spatial_merge_size: usize,
    pub(crate) channels: usize,
    pub(crate) text_hidden: usize,
    pub(crate) layer_norm_eps: f32,
    pub(crate) quantization: Option<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen2VlGridPlan {
    pub(crate) patch_bucket: usize,
    pub(crate) actual_patches: usize,
    pub(crate) packed_cu_seqlens: Vec<usize>,
    pub(crate) merged_tokens_per_image: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen2VlHostInputs {
    pub(crate) plan: Qwen2VlGridPlan,
    pub(crate) patches: Vec<f32>,
    pub(crate) vision_rope_freqs: Vec<f32>,
    pub(crate) packed_attention_bias: Vec<f32>,
}

struct Args {
    values: Vec<Val>,
    declarations: Vec<String>,
    cursor: usize,
}

impl Args {
    fn new(specs: &[Qwen2VlWeightSpec]) -> Self {
        let mut values = Vec::with_capacity(specs.len() + 3);
        let mut declarations = Vec::with_capacity(specs.len() + 3);
        for (index, spec) in specs.iter().enumerate() {
            let ty = Ty::f32(spec.shape.clone());
            declarations.push(format!(
                "%arg{index}: {} loc(\"{}\")",
                ty.render(),
                spec.name
            ));
            values.push(Builder::arg(index, ty));
        }
        Self {
            values,
            declarations,
            cursor: 0,
        }
    }

    fn take(&mut self) -> Val {
        let value = self.values[self.cursor].clone();
        self.cursor += 1;
        value
    }

    fn push_input(&mut self, ty: Ty, name: &str) -> Val {
        let index = self.values.len();
        self.declarations
            .push(format!("%arg{index}: {} loc(\"{name}\")", ty.render()));
        let value = Builder::arg(index, ty);
        self.values.push(value.clone());
        value
    }
}

fn positive_usize(object: &serde_json::Map<String, Value>, field: &str) -> Result<usize, String> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("config.json vision_config.{field} must be a positive integer"))?;
    let value = usize::try_from(value)
        .map_err(|_| format!("config.json vision_config.{field} does not fit usize"))?;
    if value == 0 {
        return Err(format!(
            "config.json vision_config.{field} must be greater than zero"
        ));
    }
    Ok(value)
}

impl Qwen2VlConfig {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Self::from_json_str(&text)
    }

    pub(crate) fn from_json_str(text: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| format!("parse config.json: {error}"))?;
        if root.get("model_type").and_then(Value::as_str) != Some("qwen2_vl") {
            return Err("Qwen2-VL IREE vision requires model_type=qwen2_vl".to_string());
        }
        let vision = root
            .get("vision_config")
            .and_then(Value::as_object)
            .ok_or_else(|| "config.json vision_config must be an object".to_string())?;
        let depth = positive_usize(vision, "depth")?;
        let hidden = positive_usize(vision, "embed_dim")?;
        let heads = positive_usize(vision, "num_heads")?;
        if hidden % heads != 0 {
            return Err(format!(
                "Qwen2-VL vision embed_dim={hidden} is not divisible by num_heads={heads}"
            ));
        }
        let mlp_ratio = vision
            .get("mlp_ratio")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                "config.json vision_config.mlp_ratio must be a positive number".to_string()
            })?;
        if !mlp_ratio.is_finite() || mlp_ratio <= 0.0 {
            return Err(
                "config.json vision_config.mlp_ratio must be finite and positive".to_string(),
            );
        }
        let intermediate_f64 = hidden as f64 * mlp_ratio;
        if intermediate_f64.fract() != 0.0 || intermediate_f64 > usize::MAX as f64 {
            return Err(format!(
                "Qwen2-VL vision intermediate size {hidden} * {mlp_ratio} is not an integer"
            ));
        }
        let intermediate = intermediate_f64 as usize;
        let patch_size = positive_usize(vision, "patch_size")?;
        let temporal_patch_size = positive_usize(vision, "temporal_patch_size")?;
        let spatial_merge_size = positive_usize(vision, "spatial_merge_size")?;
        let channels = vision
            .get("in_chans")
            .or_else(|| vision.get("in_channels"))
            .and_then(Value::as_u64)
            .map_or(Ok(3usize), |value| {
                usize::try_from(value).map_err(|_| {
                    "config.json vision_config.in_chans does not fit usize".to_string()
                })
            })?;
        if channels == 0 {
            return Err("config.json vision_config.in_chans must be positive".to_string());
        }
        let text_hidden = root
            .get("hidden_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| "config.json hidden_size must be a positive integer".to_string())
            .and_then(|value| {
                usize::try_from(value)
                    .map_err(|_| "config.json hidden_size does not fit usize".to_string())
            })?;
        if text_hidden == 0 {
            return Err("config.json hidden_size must be positive".to_string());
        }
        if spatial_merge_size != 2 {
            return Err(format!(
                "Qwen2-VL IREE vision currently qualifies spatial_merge_size=2, got {spatial_merge_size}"
            ));
        }
        if temporal_patch_size != 2 {
            return Err(format!(
                "Qwen2-VL IREE image vision currently qualifies temporal_patch_size=2, got {temporal_patch_size}"
            ));
        }
        let quantization = match root.get("quantization") {
            None => None,
            Some(value) => {
                let object = value
                    .as_object()
                    .ok_or_else(|| "config.json quantization must be an object".to_string())?;
                let bits = positive_usize(object, "bits")
                    .map_err(|error| error.replace("vision_config.", "quantization."))?;
                let group_size = positive_usize(object, "group_size")
                    .map_err(|error| error.replace("vision_config.", "quantization."))?;
                if !matches!(bits, 4 | 8) {
                    return Err(format!(
                        "Qwen2-VL IREE vision supports 4-bit or 8-bit affine weights, got {bits}"
                    ));
                }
                if hidden % group_size != 0 || intermediate % group_size != 0 {
                    return Err(format!(
                        "Qwen2-VL vision dimensions hidden={hidden}, intermediate={intermediate} must be divisible by quantization group_size={group_size}"
                    ));
                }
                Some((bits, group_size))
            }
        };
        Ok(Self {
            depth,
            hidden,
            intermediate,
            heads,
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            channels,
            text_hidden,
            layer_norm_eps: 1e-6,
            quantization,
        })
    }

    pub(crate) fn bucket_for_patches(&self, actual_patches: usize) -> Result<usize, String> {
        if actual_patches == 0 {
            return Err("Qwen2-VL image grid must contain at least one patch".to_string());
        }
        let merge_area = self
            .spatial_merge_size
            .checked_mul(self.spatial_merge_size)
            .ok_or_else(|| "Qwen2-VL merge area overflowed".to_string())?;
        if !actual_patches.is_multiple_of(merge_area) {
            return Err(format!(
                "Qwen2-VL image grid has {actual_patches} patches, which is not divisible by spatial_merge_size²={merge_area}"
            ));
        }
        QWEN2_VL_PATCH_BUCKETS
            .iter()
            .copied()
            .find(|&bucket| actual_patches <= bucket)
            .ok_or_else(|| {
                format!(
                    "Qwen2-VL image grid has {actual_patches} patches, exceeding qualified capacity {}",
                    QWEN2_VL_PATCH_BUCKETS[QWEN2_VL_PATCH_BUCKETS.len() - 1]
                )
            })
    }

    /// Validate each image grid before selecting one packed static bucket.
    ///
    /// Per-axis merge divisibility is stronger than checking only the total:
    /// `1x16` has a merge-area-divisible product but cannot form 2x2 groups.
    /// Temporal grids are rejected by the image-only issue scope.
    pub(crate) fn plan_image_grids(
        &self,
        grids: &[(i32, i32, i32)],
    ) -> Result<Qwen2VlGridPlan, String> {
        if grids.is_empty() {
            return Err("Qwen2-VL image execution requires at least one grid".to_string());
        }
        let mut actual_patches = 0usize;
        let mut packed_cu_seqlens = vec![0usize];
        let mut merged_tokens_per_image = Vec::with_capacity(grids.len());
        for (index, &(temporal, height, width)) in grids.iter().enumerate() {
            if temporal != 1 {
                return Err(format!(
                    "Qwen2-VL grid {index} has temporal size {temporal}; video/temporal grids are unsupported by the image-only XLA path"
                ));
            }
            let height = usize::try_from(height)
                .map_err(|_| format!("Qwen2-VL grid {index} has non-positive height {height}"))?;
            let width = usize::try_from(width)
                .map_err(|_| format!("Qwen2-VL grid {index} has non-positive width {width}"))?;
            if height == 0 || width == 0 {
                return Err(format!(
                    "Qwen2-VL grid {index} dimensions must be positive, got {height}x{width}"
                ));
            }
            if !height.is_multiple_of(self.spatial_merge_size)
                || !width.is_multiple_of(self.spatial_merge_size)
            {
                return Err(format!(
                    "Qwen2-VL grid {index} {height}x{width} must be divisible on each spatial axis by spatial_merge_size={}",
                    self.spatial_merge_size
                ));
            }
            let patches = height
                .checked_mul(width)
                .ok_or_else(|| format!("Qwen2-VL grid {index} patch count overflowed"))?;
            actual_patches = actual_patches
                .checked_add(patches)
                .ok_or_else(|| "Qwen2-VL packed patch count overflowed".to_string())?;
            packed_cu_seqlens.push(actual_patches);
            merged_tokens_per_image.push(self.merged_tokens(patches));
        }
        Ok(Qwen2VlGridPlan {
            patch_bucket: self.bucket_for_patches(actual_patches)?,
            actual_patches,
            packed_cu_seqlens,
            merged_tokens_per_image,
        })
    }

    #[must_use]
    pub(crate) fn merged_tokens(&self, actual_patches: usize) -> usize {
        actual_patches / (self.spatial_merge_size * self.spatial_merge_size)
    }

    #[must_use]
    pub(crate) fn fingerprint(&self, patch_bucket: usize) -> String {
        format!(
            "qwen2-vl-vision-v1:bucket={patch_bucket}:patch={}:temporal={}:merge={}:channels={}:hidden={}:intermediate={}:heads={}:depth={}:text={}:dtype=f32:source_quant={:?}",
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
            self.channels,
            self.hidden,
            self.intermediate,
            self.heads,
            self.depth,
            self.text_hidden,
            self.quantization,
        )
    }

    pub(crate) fn weight_specs(&self) -> Vec<Qwen2VlWeightSpec> {
        let patch_width =
            self.channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        let mut specs = vec![self.spec(
            "vision_tower.patch_embed.proj.weight",
            [self.hidden, patch_width],
        )];
        for layer in 0..self.depth {
            let prefix = format!("vision_tower.blocks.{layer}");
            for (suffix, shape) in [
                ("norm1.weight", vec![self.hidden]),
                ("norm1.bias", vec![self.hidden]),
                ("attn.qkv.weight", vec![self.hidden * 3, self.hidden]),
                ("attn.qkv.bias", vec![self.hidden * 3]),
                ("attn.proj.weight", vec![self.hidden, self.hidden]),
                ("attn.proj.bias", vec![self.hidden]),
                ("norm2.weight", vec![self.hidden]),
                ("norm2.bias", vec![self.hidden]),
                ("mlp.fc1.weight", vec![self.intermediate, self.hidden]),
                ("mlp.fc1.bias", vec![self.intermediate]),
                ("mlp.fc2.weight", vec![self.hidden, self.intermediate]),
                ("mlp.fc2.bias", vec![self.hidden]),
            ] {
                specs.push(Qwen2VlWeightSpec {
                    name: format!("{prefix}.{suffix}"),
                    shape,
                });
            }
        }
        let merge_width = self.hidden * self.spatial_merge_size * self.spatial_merge_size;
        specs.extend([
            self.spec("vision_tower.merger.ln_q.weight", [self.hidden]),
            self.spec("vision_tower.merger.ln_q.bias", [self.hidden]),
            self.spec(
                "vision_tower.merger.mlp.0.weight",
                [merge_width, merge_width],
            ),
            self.spec("vision_tower.merger.mlp.0.bias", [merge_width]),
            self.spec(
                "vision_tower.merger.mlp.2.weight",
                [self.text_hidden, merge_width],
            ),
            self.spec("vision_tower.merger.mlp.2.bias", [self.text_hidden]),
        ]);
        specs
    }

    fn spec<const N: usize>(&self, name: &str, shape: [usize; N]) -> Qwen2VlWeightSpec {
        Qwen2VlWeightSpec {
            name: name.to_string(),
            shape: shape.into(),
        }
    }
}

pub(crate) fn prepare_qwen2_vl_host_inputs(
    config: &Qwen2VlConfig,
    grids: &[(i32, i32, i32)],
    temporal_patch_rows: &[f32],
) -> Result<Qwen2VlHostInputs, String> {
    let plan = config.plan_image_grids(grids)?;
    let row_width = config
        .channels
        .checked_mul(config.patch_size)
        .and_then(|value| value.checked_mul(config.patch_size))
        .ok_or_else(|| "Qwen2-VL processor row width overflowed".to_string())?;
    let actual_values = plan
        .actual_patches
        .checked_mul(config.temporal_patch_size)
        .and_then(|value| value.checked_mul(row_width))
        .ok_or_else(|| "Qwen2-VL processor tensor size overflowed".to_string())?;
    if temporal_patch_rows.len() != actual_values {
        return Err(format!(
            "Qwen2-VL processor produced {} values, expected {actual_values} for {} patches x temporal {} x row width {row_width}",
            temporal_patch_rows.len(),
            plan.actual_patches,
            config.temporal_patch_size
        ));
    }
    if let Some((index, value)) = temporal_patch_rows
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "Qwen2-VL processor values contain non-finite value {value} at flat index {index}"
        ));
    }
    let patch_width = row_width
        .checked_mul(config.temporal_patch_size)
        .ok_or_else(|| "Qwen2-VL temporal patch width overflowed".to_string())?;
    let padded_values = plan
        .patch_bucket
        .checked_mul(patch_width)
        .ok_or_else(|| "Qwen2-VL padded patch tensor size overflowed".to_string())?;
    let mut patches = Vec::with_capacity(padded_values);
    // The shared processor emits the temporal rows contiguously for each
    // spatial patch. Flattening those adjacent rows is exactly the
    // `[patch, temporal*channels*height*width]` patch-projection contract.
    patches.extend_from_slice(temporal_patch_rows);
    patches.resize(padded_values, 0.0);

    let head_dim = config.hidden / config.heads;
    let rotary_dim = head_dim / 2;
    let frequency_width = rotary_dim;
    let axis_width = rotary_dim / 2;
    if head_dim % 4 != 0 {
        return Err(format!(
            "Qwen2-VL vision head_dim={head_dim} must be divisible by 4 for 2D RoPE"
        ));
    }
    let inv_freq = (0..axis_width)
        .map(|index| 1.0 / 10_000.0f32.powf((2 * index) as f32 / rotary_dim as f32))
        .collect::<Vec<_>>();
    let mut vision_rope_freqs = Vec::with_capacity(plan.patch_bucket * frequency_width);
    let merge = config.spatial_merge_size;
    for &(temporal, height, width) in grids {
        debug_assert_eq!(temporal, 1);
        let height = height as usize;
        let width = width as usize;
        for block_h in 0..height / merge {
            for block_w in 0..width / merge {
                for inner_h in 0..merge {
                    for inner_w in 0..merge {
                        let h = (block_h * merge + inner_h) as f32;
                        let w = (block_w * merge + inner_w) as f32;
                        vision_rope_freqs.extend(inv_freq.iter().map(|frequency| h * frequency));
                        vision_rope_freqs.extend(inv_freq.iter().map(|frequency| w * frequency));
                    }
                }
            }
        }
    }
    vision_rope_freqs.resize(plan.patch_bucket * frequency_width, 0.0);

    let bias_values = plan
        .patch_bucket
        .checked_mul(plan.patch_bucket)
        .ok_or_else(|| "Qwen2-VL packed attention bias size overflowed".to_string())?;
    let mut packed_attention_bias = vec![f32::NEG_INFINITY; bias_values];
    for segment in plan.packed_cu_seqlens.windows(2) {
        for row in segment[0]..segment[1] {
            for column in segment[0]..segment[1] {
                packed_attention_bias[row * plan.patch_bucket + column] = 0.0;
            }
        }
    }
    // Give every padded query one finite self key so its softmax cannot create
    // NaNs. Padded rows remain unreachable from every real media segment.
    for index in plan.actual_patches..plan.patch_bucket {
        packed_attention_bias[index * plan.patch_bucket + index] = 0.0;
    }
    Ok(Qwen2VlHostInputs {
        plan,
        patches,
        vision_rope_freqs,
        packed_attention_bias,
    })
}

fn bias_2d(builder: &mut Builder, value: &Val, bias: &Val) -> Val {
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    let bias = builder.broadcast(bias, &[1], vec![rows, width]);
    builder.add(value, &bias)
}

fn linear_2d(builder: &mut Builder, value: &Val, weight: &Val, bias: &Val) -> Val {
    let value = builder.linear_seq(value, weight);
    bias_2d(builder, &value, bias)
}

fn layer_norm(builder: &mut Builder, value: &Val, weight: &Val, bias: &Val, epsilon: f32) -> Val {
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    let zero = builder.const_f32(0.0);
    let width_scalar = builder.const_f32(width as f32);
    let width_rows = builder.broadcast(&width_scalar, &[], vec![rows]);
    let sum = builder.reduce_add(value, 1, &zero);
    let mean = builder.divide(&sum, &width_rows);
    let mean = builder.broadcast(&mean, &[0], vec![rows, width]);
    let centered = builder.subtract(value, &mean);
    let squared = builder.multiply(&centered, &centered);
    let squared_sum = builder.reduce_add(&squared, 1, &zero);
    let variance = builder.divide(&squared_sum, &width_rows);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], vec![rows]);
    let variance = builder.add(&variance, &epsilon);
    let inv_std = builder.rsqrt(&variance);
    let inv_std = builder.broadcast(&inv_std, &[0], vec![rows, width]);
    let normalized = builder.multiply(&centered, &inv_std);
    let weight = builder.broadcast(weight, &[1], vec![rows, width]);
    let bias = builder.broadcast(bias, &[1], vec![rows, width]);
    let normalized = builder.multiply(&normalized, &weight);
    builder.add(&normalized, &bias)
}

fn exact_gelu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = builder.const_f32(0.5);
    let half = builder.broadcast(&half, &[], shape.clone());
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape.clone());
    let inv_sqrt_two = builder.const_f32(std::f32::consts::FRAC_1_SQRT_2);
    let inv_sqrt_two = builder.broadcast(&inv_sqrt_two, &[], shape);
    let scaled = builder.multiply(value, &inv_sqrt_two);
    let erf = builder.erf(&scaled);
    let cdf = builder.add(&one, &erf);
    let half_value = builder.multiply(value, &half);
    builder.multiply(&half_value, &cdf)
}

fn tanh_gelu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = builder.const_f32(0.5);
    let half = builder.broadcast(&half, &[], shape.clone());
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape.clone());
    let coefficient = builder.const_f32(0.044_715);
    let coefficient = builder.broadcast(&coefficient, &[], shape.clone());
    let scale = builder.const_f32(0.797_884_6);
    let scale = builder.broadcast(&scale, &[], shape);
    let squared = builder.multiply(value, value);
    let cubed = builder.multiply(&squared, value);
    let nonlinear = builder.multiply(&coefficient, &cubed);
    let inner = builder.add(value, &nonlinear);
    let scaled = builder.multiply(&scale, &inner);
    let tanh = builder.tanh(&scaled);
    let cdf = builder.add(&one, &tanh);
    let half_value = builder.multiply(value, &half);
    builder.multiply(&half_value, &cdf)
}

fn rotate_half(builder: &mut Builder, value: &Val) -> Val {
    let tokens = value.ty.shape[0];
    let heads = value.ty.shape[1];
    let width = value.ty.shape[2];
    let half = width / 2;
    let first = builder.slice(value, &[(0, tokens), (0, heads), (0, half)]);
    let second = builder.slice(value, &[(0, tokens), (0, heads), (half, width)]);
    let second = builder.negate(&second);
    builder.concatenate(&second, &first, 2)
}

fn apply_vision_rope(builder: &mut Builder, value: &Val, freqs: &Val) -> Val {
    let tokens = value.ty.shape[0];
    let heads = value.ty.shape[1];
    let half = freqs.ty.shape[1];
    let cos = builder.cosine(freqs);
    let sin = builder.sine(freqs);
    let cos = builder.concatenate(&cos, &cos, 1);
    let sin = builder.concatenate(&sin, &sin, 1);
    let cos = builder.broadcast(&cos, &[0, 2], vec![tokens, heads, half * 2]);
    let sin = builder.broadcast(&sin, &[0, 2], vec![tokens, heads, half * 2]);
    let rotated = rotate_half(builder, value);
    let direct = builder.multiply(value, &cos);
    let rotated = builder.multiply(&rotated, &sin);
    builder.add(&direct, &rotated)
}

fn attention(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &Qwen2VlConfig,
    freqs: &Val,
    attention_bias: &Val,
) -> Val {
    let tokens = hidden.ty.shape[0];
    let head_dim = config.hidden / config.heads;
    let qkv = linear_2d(builder, hidden, &args.take(), &args.take());
    let qkv = builder.reshape(&qkv, vec![tokens, 3, config.heads, head_dim]);
    let q = builder.slice(
        &qkv,
        &[(0, tokens), (0, 1), (0, config.heads), (0, head_dim)],
    );
    let k = builder.slice(
        &qkv,
        &[(0, tokens), (1, 2), (0, config.heads), (0, head_dim)],
    );
    let v = builder.slice(
        &qkv,
        &[(0, tokens), (2, 3), (0, config.heads), (0, head_dim)],
    );
    let q = builder.reshape(&q, vec![tokens, config.heads, head_dim]);
    let k = builder.reshape(&k, vec![tokens, config.heads, head_dim]);
    let v = builder.reshape(&v, vec![tokens, config.heads, head_dim]);
    let q = apply_vision_rope(builder, &q, freqs);
    let k = apply_vision_rope(builder, &k, freqs);
    let q = builder.transpose(&q, &[1, 0, 2]);
    let k = builder.transpose(&k, &[1, 0, 2]);
    let v = builder.transpose(&v, &[1, 0, 2]);
    let scores = builder.dot_general(
        &q,
        &k,
        &[0],
        &[0],
        &[2],
        &[2],
        vec![config.heads, tokens, tokens],
    );
    let scale = builder.const_f32((head_dim as f32).powf(-0.5));
    let scale = builder.broadcast(&scale, &[], vec![config.heads, tokens, tokens]);
    let scores = builder.multiply(&scores, &scale);
    let attention_bias =
        builder.broadcast(attention_bias, &[1, 2], vec![config.heads, tokens, tokens]);
    let scores = builder.add(&scores, &attention_bias);
    let negative_infinity = builder.const_f32(f32::NEG_INFINITY);
    let maximum = builder.reduce_max(&scores, 2, &negative_infinity);
    let maximum = builder.broadcast(&maximum, &[0, 1], vec![config.heads, tokens, tokens]);
    let shifted = builder.subtract(&scores, &maximum);
    let exponentials = builder.exponential(&shifted);
    let zero = builder.const_f32(0.0);
    let denominator = builder.reduce_add(&exponentials, 2, &zero);
    let denominator = builder.broadcast(&denominator, &[0, 1], vec![config.heads, tokens, tokens]);
    let probabilities = builder.divide(&exponentials, &denominator);
    let context = builder.dot_general(
        &probabilities,
        &v,
        &[0],
        &[0],
        &[2],
        &[1],
        vec![config.heads, tokens, head_dim],
    );
    let context = builder.transpose(&context, &[1, 0, 2]);
    let context = builder.reshape(&context, vec![tokens, config.hidden]);
    linear_2d(builder, &context, &args.take(), &args.take())
}

fn encoder_layer(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &Qwen2VlConfig,
    freqs: &Val,
    attention_bias: &Val,
) -> Val {
    let norm1 = layer_norm(
        builder,
        hidden,
        &args.take(),
        &args.take(),
        config.layer_norm_eps,
    );
    let attention = attention(builder, &norm1, args, config, freqs, attention_bias);
    let residual = builder.add(hidden, &attention);
    let norm2 = layer_norm(
        builder,
        &residual,
        &args.take(),
        &args.take(),
        config.layer_norm_eps,
    );
    let fc1 = linear_2d(builder, &norm2, &args.take(), &args.take());
    let activated = tanh_gelu(builder, &fc1);
    let fc2 = linear_2d(builder, &activated, &args.take(), &args.take());
    builder.add(&residual, &fc2)
}

/// Emit one Qwen2-VL bucket. `patches` are already in the processor's
/// post-smart-resize, spatial-merge-grouped order. `vision_rope` and
/// `attention_bias` are host-built from `grid_thw`/packed boundaries, keeping
/// cross-image isolation explicit while preserving one finite static graph.
pub(crate) fn emit_qwen2_vl(config: &Qwen2VlConfig, patch_bucket: usize) -> String {
    assert!(
        QWEN2_VL_PATCH_BUCKETS.contains(&patch_bucket),
        "unqualified Qwen2-VL patch bucket"
    );
    let specs = config.weight_specs();
    let mut args = Args::new(&specs);
    let mut builder = Builder::new();
    let patch_weight = args.take();
    let patch_width =
        config.channels * config.temporal_patch_size * config.patch_size * config.patch_size;
    let head_dim = config.hidden / config.heads;
    let patches = args.push_input(Ty::f32(vec![patch_bucket, patch_width]), "patches.grouped");
    let freqs = args.push_input(
        Ty::f32(vec![patch_bucket, head_dim / 2]),
        "vision_rope.freqs",
    );
    let attention_bias = args.push_input(
        Ty::f32(vec![patch_bucket, patch_bucket]),
        "packed_attention.bias",
    );
    let mut hidden = builder.linear_seq(&patches, &patch_weight);
    for _ in 0..config.depth {
        hidden = encoder_layer(
            &mut builder,
            &hidden,
            &mut args,
            config,
            &freqs,
            &attention_bias,
        );
    }
    hidden = layer_norm(
        &mut builder,
        &hidden,
        &args.take(),
        &args.take(),
        config.layer_norm_eps,
    );
    let merge_width = config.hidden * config.spatial_merge_size * config.spatial_merge_size;
    let merged = builder.reshape(
        &hidden,
        vec![
            patch_bucket / (config.spatial_merge_size * config.spatial_merge_size),
            merge_width,
        ],
    );
    let merged = linear_2d(&mut builder, &merged, &args.take(), &args.take());
    let merged = exact_gelu(&mut builder, &merged);
    let projected = linear_2d(&mut builder, &merged, &args.take(), &args.take());
    assert_eq!(args.cursor, specs.len(), "Qwen2-VL weight schema drifted");
    format!(
        "module @qwen2_vl_vision {{\n  func.func public @main({signature}) -> {result_type} {{\n{body}    return {result} : {result_type}\n  }}\n}}\n",
        signature = args.declarations.join(", "),
        result_type = projected.ty.render(),
        body = builder.body(),
        result = projected.name,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Qwen2VlConfig {
        Qwen2VlConfig::from_json_str(
            r#"{
                "model_type": "qwen2_vl",
                "hidden_size": 1536,
                "vision_config": {
                    "depth": 2,
                    "embed_dim": 1280,
                    "mlp_ratio": 4,
                    "num_heads": 16,
                    "in_chans": 3,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn static_buckets_bound_qwen2_vl_grids_without_per_shape_compilation() {
        let config = config();
        assert_eq!(config.bucket_for_patches(16), Ok(16));
        assert_eq!(config.bucket_for_patches(20), Ok(64));
        assert_eq!(config.bucket_for_patches(256), Ok(256));
        assert!(
            config
                .bucket_for_patches(18)
                .unwrap_err()
                .contains("divisible")
        );
        assert!(
            config
                .bucket_for_patches(260)
                .unwrap_err()
                .contains("exceeding qualified capacity")
        );
        assert_eq!(config.merged_tokens(64), 16);
        assert!(config.fingerprint(64).contains("bucket=64"));
        assert!(
            config
                .plan_image_grids(&[(1, 1, 16)])
                .unwrap_err()
                .contains("each spatial axis")
        );
        assert!(
            config
                .plan_image_grids(&[(2, 4, 4)])
                .unwrap_err()
                .contains("video/temporal grids")
        );
        let packed = config.plan_image_grids(&[(1, 4, 4), (1, 4, 8)]).unwrap();
        assert_eq!(packed.patch_bucket, 64);
        assert_eq!(packed.actual_patches, 48);
        assert_eq!(packed.packed_cu_seqlens, vec![0, 16, 48]);
        assert_eq!(packed.merged_tokens_per_image, vec![4, 8]);
    }

    #[test]
    fn weight_schema_covers_patch_blocks_and_merger_in_checkpoint_order() {
        let config = config();
        let specs = config.weight_specs();
        assert_eq!(
            specs.first().unwrap(),
            &Qwen2VlWeightSpec {
                name: "vision_tower.patch_embed.proj.weight".to_string(),
                shape: vec![1280, 1176],
            }
        );
        assert_eq!(specs.len(), 1 + 2 * 12 + 6);
        assert_eq!(
            specs.last().unwrap(),
            &Qwen2VlWeightSpec {
                name: "vision_tower.merger.mlp.2.bias".to_string(),
                shape: vec![1536],
            }
        );
    }

    #[test]
    fn emitted_bucket_has_packed_bias_rope_and_merged_output_contract() {
        let config = config();
        let mlir = emit_qwen2_vl(&config, 16);
        assert!(mlir.contains("loc(\"patches.grouped\")"));
        assert!(mlir.contains("tensor<16x1176xf32>"));
        assert!(mlir.contains("loc(\"vision_rope.freqs\")"));
        assert!(mlir.contains("tensor<16x40xf32>"));
        assert!(mlir.contains("loc(\"packed_attention.bias\")"));
        assert!(mlir.contains("tensor<16x16xf32>"));
        assert!(mlir.contains("-> tensor<4x1536xf32>"));
        assert_eq!(mlir.matches("stablehlo.dot_general").count(), 15);
        assert_eq!(mlir.matches("chlo.erf").count(), 1);
    }

    #[test]
    fn host_inputs_preserve_packed_media_isolation_and_finite_padding_rows() {
        let config = config();
        let grids = [(1, 4, 4), (1, 4, 8)];
        let row_width = 3 * 14 * 14;
        let values = vec![0.25; (16 + 32) * 2 * row_width];
        let inputs = prepare_qwen2_vl_host_inputs(&config, &grids, &values).unwrap();
        assert_eq!(inputs.plan.patch_bucket, 64);
        assert_eq!(inputs.patches.len(), 64 * 1176);
        assert_eq!(inputs.vision_rope_freqs.len(), 64 * 40);
        assert_eq!(inputs.packed_attention_bias.len(), 64 * 64);
        assert_eq!(inputs.packed_attention_bias[15 * 64 + 15], 0.0);
        assert_eq!(
            inputs.packed_attention_bias[15 * 64 + 16],
            f32::NEG_INFINITY
        );
        assert_eq!(
            inputs.packed_attention_bias[16 * 64 + 15],
            f32::NEG_INFINITY
        );
        assert_eq!(inputs.packed_attention_bias[48 * 64 + 48], 0.0);
        assert_eq!(inputs.packed_attention_bias[48 * 64 + 0], f32::NEG_INFINITY);
        let mut non_finite = values;
        non_finite[7] = f32::NAN;
        assert!(
            prepare_qwen2_vl_host_inputs(&config, &grids, &non_finite)
                .unwrap_err()
                .contains("flat index 7")
        );
    }
}
