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

//! Strict Molmo2 vision/processor contract for the pinned XLA graph.

use std::path::Path;

use serde_json::Value;

const MAX_SUPPORTED_CROPS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Molmo2VisionWeightSpec {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Molmo2VisionConfig {
    pub(crate) crop_size: usize,
    pub(crate) patch_size: usize,
    pub(crate) patches_per_crop: usize,
    pub(crate) patch_dim: usize,
    pub(crate) max_crops: usize,
    pub(crate) static_crops: usize,
    pub(crate) overlap: [usize; 2],
    pub(crate) pool_h: usize,
    pub(crate) pool_w: usize,
    pub(crate) pool_size: usize,
    pub(crate) static_pool_groups: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) layers: usize,
    pub(crate) emitted_layers: usize,
    pub(crate) selected_layers: Vec<usize>,
    pub(crate) position_count: usize,
    pub(crate) layer_norm_eps: f32,
    pub(crate) pool_hidden: usize,
    pub(crate) pool_heads: usize,
    pub(crate) pool_head_dim: usize,
    pub(crate) projector_intermediate: usize,
    pub(crate) text_hidden: usize,
    pub(crate) pooling_attention_mask: bool,
    pub(crate) image_patch_id: i32,
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .get(name)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("Molmo2 config missing object `{name}`"))
}

fn usize_field(object: &serde_json::Map<String, Value>, name: &str) -> Result<usize, String> {
    object
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Molmo2 `{name}` must be a positive integer"))
}

fn usize_pair(object: &serde_json::Map<String, Value>, name: &str) -> Result<[usize; 2], String> {
    let values = object
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("Molmo2 `{name}` must contain two integers"))?;
    if values.len() != 2 {
        return Err(format!("Molmo2 `{name}` must contain two integers"));
    }
    let mut output = [0usize; 2];
    for (index, value) in values.iter().enumerate() {
        output[index] = value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("Molmo2 `{name}[{index}]` must be positive"))?;
    }
    Ok(output)
}

fn usize_pair_nonnegative(
    object: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<[usize; 2], String> {
    let values = object
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("Molmo2 `{name}` must contain two nonnegative integers"))?;
    if values.len() != 2 {
        return Err(format!(
            "Molmo2 `{name}` must contain two nonnegative integers"
        ));
    }
    let mut output = [0usize; 2];
    for (index, value) in values.iter().enumerate() {
        output[index] = value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("Molmo2 `{name}[{index}]` must be nonnegative"))?;
    }
    Ok(output)
}

fn maximum_pool_groups(max_crops: usize, crop_patches: usize, overlap: [usize; 2]) -> usize {
    let window = crop_patches - overlap[0] - overlap[1];
    let low = crop_patches.div_ceil(2).pow(2);
    let high = (1..=max_crops)
        .flat_map(|rows| (1..=max_crops).map(move |columns| (rows, columns)))
        .filter(|(rows, columns)| rows * columns <= max_crops)
        .map(|(rows, columns)| {
            let height = rows * window + overlap[0] + overlap[1];
            let width = columns * window + overlap[0] + overlap[1];
            height.div_ceil(2) * width.div_ceil(2)
        })
        .max()
        .unwrap_or(0);
    low + high
}

impl Molmo2VisionConfig {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let processor_path = model_dir.join("preprocessor_config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|error| format!("read {}: {error}", config_path.display()))?;
        let processor_text = std::fs::read_to_string(&processor_path)
            .map_err(|error| format!("read {}: {error}", processor_path.display()))?;
        Self::from_json_strs(&config_text, &processor_text)
    }

    pub(crate) fn from_json_strs(config: &str, processor: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(config).map_err(|error| format!("parse config.json: {error}"))?;
        if root.get("model_type").and_then(Value::as_str) != Some("molmo2") {
            return Err("Molmo2 XLA vision requires config.json model_type `molmo2`".to_string());
        }
        let vit = object(&root, "vit_config")?;
        let adapter = object(&root, "adapter_config")?;
        let processor: Value = serde_json::from_str(processor)
            .map_err(|error| format!("parse preprocessor_config.json: {error}"))?;
        let processor = processor
            .as_object()
            .ok_or_else(|| "Molmo2 preprocessor config must be an object".to_string())?;

        let input_size = usize_pair(vit, "image_default_input_size")?;
        if input_size[0] != input_size[1] {
            return Err("Molmo2 XLA requires square default image crops".to_string());
        }
        let crop_size = input_size[0];
        let patch_size = usize_field(vit, "image_patch_size")?;
        if !crop_size.is_multiple_of(patch_size) {
            return Err(format!(
                "Molmo2 crop size {crop_size} is not divisible by patch size {patch_size}"
            ));
        }
        let crop_patches = crop_size / patch_size;
        let patches_per_crop = crop_patches * crop_patches;
        let position_count = usize_field(vit, "image_num_pos")?;
        if position_count != patches_per_crop {
            return Err(format!(
                "Molmo2 XLA supports the pinned exact position table only: image_num_pos={position_count}, default grid has {patches_per_crop} patches"
            ));
        }
        let preprocessor_patch = usize_field(processor, "patch_size")?;
        let size = processor
            .get("size")
            .and_then(Value::as_object)
            .ok_or_else(|| "Molmo2 preprocessor missing object `size`".to_string())?;
        if preprocessor_patch != patch_size
            || usize_field(size, "height")? != crop_size
            || usize_field(size, "width")? != crop_size
        {
            return Err("Molmo2 processor and ViT crop/patch geometry disagree".to_string());
        }
        let max_crops = usize_field(processor, "max_crops")?;
        if max_crops > MAX_SUPPORTED_CROPS {
            return Err(format!(
                "Molmo2 XLA supports at most {MAX_SUPPORTED_CROPS} high-resolution crops, got {max_crops}"
            ));
        }
        let overlap = usize_pair_nonnegative(processor, "overlap_margins")?;
        if overlap[0] + overlap[1] >= crop_patches {
            return Err("Molmo2 overlap margins consume the complete crop".to_string());
        }
        let pooling = usize_pair(processor, "pooling_size")?;
        if pooling != [2, 2] {
            return Err(format!(
                "Molmo2 pinned XLA graph requires 2x2 pooling, got {pooling:?}"
            ));
        }

        let layers = usize_field(vit, "num_hidden_layers")?.min(25);
        let selected_raw = adapter
            .get("vit_layers")
            .and_then(Value::as_array)
            .ok_or_else(|| "Molmo2 adapter `vit_layers` must be an array".to_string())?;
        if selected_raw.is_empty() {
            return Err("Molmo2 adapter must select at least one ViT layer".to_string());
        }
        let selected_layers = selected_raw
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let raw = value
                    .as_i64()
                    .ok_or_else(|| format!("Molmo2 vit_layers[{index}] must be an integer"))?;
                let resolved = if raw < 0 { layers as i64 + raw } else { raw };
                usize::try_from(resolved)
                    .ok()
                    .filter(|layer| *layer < layers)
                    .ok_or_else(|| {
                        format!("Molmo2 vit_layers[{index}]={raw} resolves outside [0,{layers})")
                    })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let emitted_layers = selected_layers
            .iter()
            .copied()
            .max()
            .ok_or_else(|| "Molmo2 adapter vit_layers must not be empty".to_string())?
            + 1;
        let hidden = usize_field(vit, "hidden_size")?;
        let heads = usize_field(vit, "num_attention_heads")?;
        let kv_heads = vit
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(heads);
        let head_dim = usize_field(vit, "head_dim")?;
        if heads * head_dim != hidden || kv_heads != heads {
            return Err(
                "Molmo2 XLA requires ViT MHA with heads * head_dim = hidden_size".to_string(),
            );
        }
        let pool_hidden = usize_field(adapter, "hidden_size")?;
        let pool_heads = usize_field(adapter, "num_attention_heads")?;
        let pool_kv_heads = adapter
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(pool_heads);
        let pool_head_dim = usize_field(adapter, "head_dim")?;
        if pool_heads * pool_head_dim != pool_hidden || pool_kv_heads != pool_heads {
            return Err(
                "Molmo2 XLA requires pooling MHA with heads * head_dim = hidden_size".to_string(),
            );
        }
        let selected_width = hidden
            .checked_mul(selected_layers.len())
            .ok_or_else(|| "Molmo2 selected feature width overflowed".to_string())?;
        let pool_q_width = adapter
            .get("image_feature_dim")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(selected_width);
        if pool_q_width != selected_width {
            return Err(format!(
                "Molmo2 pooling input width {pool_q_width} disagrees with selected layer width {selected_width}"
            ));
        }

        let layer_norm_eps = vit
            .get("layer_norm_eps")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| "Molmo2 layer_norm_eps must be positive and finite".to_string())?
            as f32;
        let image_patch_id = root
            .get("image_patch_id")
            .and_then(Value::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| "Molmo2 image_patch_id must fit i32".to_string())?;

        Ok(Self {
            crop_size,
            patch_size,
            patches_per_crop,
            patch_dim: patch_size * patch_size * 3,
            max_crops,
            static_crops: max_crops + 1,
            overlap,
            pool_h: pooling[0],
            pool_w: pooling[1],
            pool_size: pooling[0] * pooling[1],
            static_pool_groups: maximum_pool_groups(max_crops, crop_patches, overlap),
            hidden,
            intermediate: usize_field(vit, "intermediate_size")?,
            heads,
            head_dim,
            layers,
            emitted_layers,
            selected_layers,
            position_count,
            layer_norm_eps,
            pool_hidden,
            pool_heads,
            pool_head_dim,
            projector_intermediate: usize_field(adapter, "intermediate_size")?,
            text_hidden: usize_field(adapter, "text_hidden_size")?,
            pooling_attention_mask: adapter
                .get("pooling_attention_mask")
                .and_then(Value::as_bool)
                .ok_or_else(|| {
                    "Molmo2 pooling_attention_mask must be an explicit boolean".to_string()
                })?,
            image_patch_id,
        })
    }

    #[must_use]
    pub(crate) fn selected_width(&self) -> usize {
        self.hidden * self.selected_layers.len()
    }

    pub(crate) fn valid_runtime_geometry(&self, crops: usize, grid: [usize; 4]) -> bool {
        if !(2..=self.static_crops).contains(&crops) {
            return false;
        }
        let crop_patches = self.crop_size / self.patch_size;
        let low = crop_patches.div_ceil(self.pool_h);
        if grid[0] != low || grid[1] != low {
            return false;
        }
        let high_crops = crops - 1;
        let window = crop_patches - self.overlap[0] - self.overlap[1];
        (1..=high_crops).any(|rows| {
            high_crops.is_multiple_of(rows) && {
                let columns = high_crops / rows;
                let height = rows * window + self.overlap[0] + self.overlap[1];
                let width = columns * window + self.overlap[0] + self.overlap[1];
                grid[2] == height.div_ceil(self.pool_h) && grid[3] == width.div_ceil(self.pool_w)
            }
        })
    }

    pub(crate) fn fingerprint(&self) -> String {
        format!(
            "molmo2-vision-v1;position=exact-default;selected={:?};pool-mask={};patch-id={};crops={};overlap={:?};patches={};pool-groups={};pool={}x{};hidden={};inter={};layers={};emitted={};heads={};head-dim={};pool-hidden={};pool-heads={};pool-head-dim={};projector-inter={};text-hidden={}",
            self.selected_layers,
            self.pooling_attention_mask,
            self.image_patch_id,
            self.static_crops,
            self.overlap,
            self.patches_per_crop,
            self.static_pool_groups,
            self.pool_h,
            self.pool_w,
            self.hidden,
            self.intermediate,
            self.layers,
            self.emitted_layers,
            self.heads,
            self.head_dim,
            self.pool_hidden,
            self.pool_heads,
            self.pool_head_dim,
            self.projector_intermediate,
            self.text_hidden
        )
    }

    fn spec(
        &self,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
    ) -> Molmo2VisionWeightSpec {
        Molmo2VisionWeightSpec {
            name: name.into(),
            shape: shape.into(),
        }
    }

    pub(crate) fn weight_specs(&self) -> Vec<Molmo2VisionWeightSpec> {
        let mut specs = vec![
            self.spec(
                "vision_tower.image_vit.patch_embedding.weight",
                [self.hidden, self.patch_dim],
            ),
            self.spec("vision_tower.image_vit.patch_embedding.bias", [self.hidden]),
            self.spec(
                "vision_tower.image_vit.positional_embedding",
                [self.position_count, self.hidden],
            ),
        ];
        for layer in 0..self.emitted_layers {
            let prefix = format!("vision_tower.image_vit.transformer.{layer}");
            for projection in ["wq", "wk", "wv", "wo"] {
                specs.push(self.spec(
                    format!("{prefix}.attention.{projection}.weight"),
                    [self.hidden, self.hidden],
                ));
                specs.push(self.spec(
                    format!("{prefix}.attention.{projection}.bias"),
                    [self.hidden],
                ));
            }
            for norm in ["attention_norm", "ffn_norm"] {
                specs.push(self.spec(format!("{prefix}.{norm}.weight"), [self.hidden]));
                specs.push(self.spec(format!("{prefix}.{norm}.bias"), [self.hidden]));
            }
            specs.push(self.spec(
                format!("{prefix}.feed_forward.w1.weight"),
                [self.intermediate, self.hidden],
            ));
            specs.push(self.spec(
                format!("{prefix}.feed_forward.w1.bias"),
                [self.intermediate],
            ));
            specs.push(self.spec(
                format!("{prefix}.feed_forward.w2.weight"),
                [self.hidden, self.intermediate],
            ));
            specs.push(self.spec(format!("{prefix}.feed_forward.w2.bias"), [self.hidden]));
        }
        for projection in ["wq", "wk", "wv"] {
            specs.push(self.spec(
                format!("vision_tower.image_pooling_2d.{projection}.weight"),
                [self.pool_hidden, self.selected_width()],
            ));
            specs.push(self.spec(
                format!("vision_tower.image_pooling_2d.{projection}.bias"),
                [self.pool_hidden],
            ));
        }
        specs.push(self.spec(
            "vision_tower.image_pooling_2d.wo.weight",
            [self.pool_hidden, self.pool_hidden],
        ));
        specs.push(self.spec("vision_tower.image_pooling_2d.wo.bias", [self.pool_hidden]));
        specs.push(self.spec(
            "vision_tower.image_projector.w1.weight",
            [self.projector_intermediate, self.pool_hidden],
        ));
        specs.push(self.spec(
            "vision_tower.image_projector.w2.weight",
            [self.text_hidden, self.projector_intermediate],
        ));
        specs.push(self.spec(
            "vision_tower.image_projector.w3.weight",
            [self.projector_intermediate, self.pool_hidden],
        ));
        specs
    }
}

#[cfg(test)]
#[path = "molmo2_config_tests.rs"]
mod tests;
