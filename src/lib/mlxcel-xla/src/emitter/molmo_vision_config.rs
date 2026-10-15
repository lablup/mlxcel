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

//! Static Molmo v1 native vision/pool/projector artifact contract.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::Value;

use crate::MOLMO_V1_MERGE_MODE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MolmoVisionWeightDType {
    Float32,
    Float16,
    Uint32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MolmoVisionWeightSpec {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: MolmoVisionWeightDType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MolmoVisionConfig {
    pub(crate) max_crops: usize,
    pub(crate) patches_per_crop: usize,
    pub(crate) patch_width: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) positions: usize,
    pub(crate) layers: usize,
    pub(crate) selected_layers: Vec<usize>,
    pub(crate) pool_h: usize,
    pub(crate) pool_w: usize,
    pub(crate) text_hidden: usize,
    pub(crate) projector_hidden: usize,
    pub(crate) group_size: usize,
    pub(crate) bits: usize,
    pub(crate) layer_norm_eps_bits: u32,
    pub(crate) text_architecture: String,
}

fn positive(value: Option<u64>, default: usize, name: &str) -> Result<usize, String> {
    let value = value.unwrap_or(default as u64);
    let value = usize::try_from(value).map_err(|_| format!("{name} does not fit usize"))?;
    if value == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(value)
}

fn checked_product(values: &[usize], name: &str) -> Result<usize, String> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| format!("{name} overflows usize"))
    })
}

impl MolmoVisionConfig {
    pub(crate) fn from_model_dir(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config = std::fs::read_to_string(&config_path)
            .map_err(|error| format!("{}: {error}", config_path.display()))?;
        let preprocessor_path = model_dir.join("preprocessor_config.json");
        let preprocessor = std::fs::read_to_string(&preprocessor_path)
            .map_err(|error| format!("{}: {error}", preprocessor_path.display()))?;
        Self::from_json_strs(&config, &preprocessor)
    }

    pub(crate) fn from_json_strs(config: &str, preprocessor: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(config).map_err(|error| format!("parse config.json: {error}"))?;
        if root.get("model_type").and_then(Value::as_str) != Some("molmo") {
            return Err("native Molmo vision requires model_type=molmo".to_string());
        }
        let vision = root.get("vision_config").unwrap_or(&Value::Null);
        let preprocess: Value = serde_json::from_str(preprocessor)
            .map_err(|error| format!("parse preprocessor_config.json: {error}"))?;

        let patch_size = positive(
            vision.get("image_patch_size").and_then(Value::as_u64),
            14,
            "vision_config.image_patch_size",
        )?;
        let input_size = vision
            .get("image_default_input_size")
            .and_then(Value::as_array)
            .map(|size| {
                let height = positive(
                    size.first().and_then(Value::as_u64),
                    336,
                    "image_default_input_size height",
                )?;
                let width = positive(
                    size.get(1).and_then(Value::as_u64),
                    336,
                    "image_default_input_size width",
                )?;
                Ok::<_, String>((height, width))
            })
            .transpose()?
            .unwrap_or((336, 336));
        if input_size.0 % patch_size != 0 || input_size.1 % patch_size != 0 {
            return Err("Molmo crop size must be divisible by image_patch_size".to_string());
        }
        let patch_h = input_size.0 / patch_size;
        let patch_w = input_size.1 / patch_size;
        let patches_per_crop = checked_product(&[patch_h, patch_w], "Molmo patches per crop")?;
        let patch_width = checked_product(&[patch_size, patch_size, 3], "Molmo patch width")?;
        let positions = patches_per_crop
            .checked_add(1)
            .ok_or_else(|| "Molmo vision position count overflows usize".to_string())?;
        let pool_h = positive(
            vision.get("image_pooling_h").and_then(Value::as_u64),
            2,
            "vision_config.image_pooling_h",
        )?;
        let pool_w = positive(
            vision.get("image_pooling_w").and_then(Value::as_u64),
            2,
            "vision_config.image_pooling_w",
        )?;
        if !patch_h.is_multiple_of(pool_h) || !patch_w.is_multiple_of(pool_w) {
            return Err("Molmo patch grid must tile evenly into pooling windows".to_string());
        }

        let layers = positive(
            vision.get("image_num_layers").and_then(Value::as_u64),
            23,
            "vision_config.image_num_layers",
        )?;
        let raw_layers = root
            .get("vit_layers")
            .or_else(|| vision.get("vit_layers"))
            .and_then(Value::as_array)
            .map(|layers| layers.iter().filter_map(Value::as_i64).collect::<Vec<_>>())
            .unwrap_or_else(|| vec![-2, -9]);
        if raw_layers.is_empty() {
            return Err("Molmo vit_layers must not be empty".to_string());
        }
        let layers_i64 =
            i64::try_from(layers).map_err(|_| "Molmo layer count does not fit i64".to_string())?;
        let mut selected_layers = Vec::with_capacity(raw_layers.len());
        let mut unique_layers = BTreeSet::new();
        for layer in raw_layers {
            let resolved = if layer < 0 { layers_i64 + layer } else { layer };
            if resolved < 0 || resolved >= layers_i64 {
                return Err(format!("Molmo vit_layers entry {layer} is out of range"));
            }
            let resolved = usize::try_from(resolved)
                .map_err(|_| "Molmo selected layer does not fit usize".to_string())?;
            if !unique_layers.insert(resolved) {
                return Err("Molmo vit_layers must be unique".to_string());
            }
            selected_layers.push(resolved);
        }

        let hidden = positive(
            vision.get("image_emb_dim").and_then(Value::as_u64),
            1024,
            "vision_config.image_emb_dim",
        )?;
        let heads = positive(
            vision.get("image_num_heads").and_then(Value::as_u64),
            16,
            "vision_config.image_num_heads",
        )?;
        let head_dim = positive(
            vision.get("image_head_dim").and_then(Value::as_u64),
            64,
            "vision_config.image_head_dim",
        )?;
        if heads.checked_mul(head_dim) != Some(hidden) {
            return Err(
                "Molmo image_num_heads * image_head_dim must equal image_emb_dim".to_string(),
            );
        }
        let bits = positive(
            root.get("quantization")
                .and_then(|quant| quant.get("bits"))
                .and_then(Value::as_u64),
            4,
            "quantization.bits",
        )?;
        if bits != 4 && bits != 8 {
            return Err(format!(
                "Molmo native vision supports 4/8-bit affine weights, got {bits}"
            ));
        }
        let group_size = positive(
            root.get("quantization")
                .and_then(|quant| quant.get("group_size"))
                .and_then(Value::as_u64),
            64,
            "quantization.group_size",
        )?;
        let text_hidden = positive(
            root.get("hidden_size").and_then(Value::as_u64),
            3584,
            "hidden_size",
        )?;
        let text_layers = positive(
            root.get("num_hidden_layers").and_then(Value::as_u64),
            28,
            "num_hidden_layers",
        )?;
        let text_heads = positive(
            root.get("num_attention_heads").and_then(Value::as_u64),
            28,
            "num_attention_heads",
        )?;
        let text_kv_heads = positive(
            root.get("num_key_value_heads").and_then(Value::as_u64),
            4,
            "num_key_value_heads",
        )?;
        if !text_hidden.is_multiple_of(text_heads) {
            return Err("Molmo hidden_size must be divisible by num_attention_heads".to_string());
        }
        if text_heads % text_kv_heads != 0 {
            return Err(
                "Molmo num_attention_heads must be divisible by num_key_value_heads".to_string(),
            );
        }
        let fused_intermediate = positive(
            root.get("intermediate_size").and_then(Value::as_u64),
            37_888,
            "intermediate_size",
        )?;
        if fused_intermediate % 2 != 0 {
            return Err("Molmo intermediate_size must contain equal gate/up halves".to_string());
        }
        if root.get("qkv_bias").and_then(Value::as_bool) == Some(false) {
            return Err("pinned Molmo v1 native path requires qkv_bias=true".to_string());
        }
        let rope_theta = root
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(1_000_000.0);
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err("Molmo rope_theta must be finite and positive".to_string());
        }
        let selected_width = checked_product(
            &[hidden, selected_layers.len()],
            "Molmo selected vision width",
        )?;
        let packed_group = 32usize
            .checked_div(bits)
            .ok_or_else(|| "Molmo quantization bits must divide 32".to_string())?;
        for (name, width) in [
            ("vision hidden", hidden),
            ("vision intermediate", 4096),
            ("selected vision width", selected_width),
            ("projector hidden", fused_intermediate / 2),
        ] {
            if !width.is_multiple_of(packed_group) {
                return Err(format!(
                    "Molmo {name}={width} must be divisible by packed group {packed_group}"
                ));
            }
            if !width.is_multiple_of(group_size) {
                return Err(format!(
                    "Molmo {name}={width} must be divisible by quantization group_size={group_size}"
                ));
            }
        }
        let max_high_res_crops = positive(
            preprocess.get("max_crops").and_then(Value::as_u64),
            12,
            "preprocessor_config.max_crops",
        )?;
        let layer_norm_eps = vision
            .get("image_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(1e-5);
        if !layer_norm_eps.is_finite() || layer_norm_eps <= 0.0 {
            return Err("vision_config.image_norm_eps must be finite and positive".to_string());
        }
        let max_crops = max_high_res_crops
            .checked_add(1)
            .ok_or_else(|| "Molmo max crop count overflows usize".to_string())?;
        let text_layer_norm_eps = root
            .get("layer_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(1e-6);
        if !text_layer_norm_eps.is_finite() || text_layer_norm_eps <= 0.0 {
            return Err("Molmo layer_norm_eps must be finite and positive".to_string());
        }
        let text_architecture = format!(
            "olmo=hidden:{text_hidden},layers:{text_layers},heads:{text_heads},kv_heads:{text_kv_heads},\
             fused_intermediate:{fused_intermediate},rope_theta:{:016x},rope:interleave,\
             qkv_bias:true,norm:rms:{:016x},tied:false",
            rope_theta.to_bits(),
            text_layer_norm_eps.to_bits(),
        );
        Ok(Self {
            max_crops,
            patches_per_crop,
            patch_width,
            hidden,
            intermediate: 4096,
            heads,
            head_dim,
            positions,
            layers,
            selected_layers,
            pool_h,
            pool_w,
            text_hidden,
            projector_hidden: fused_intermediate / 2,
            group_size,
            bits,
            layer_norm_eps_bits: (layer_norm_eps as f32).to_bits(),
            text_architecture,
        })
    }

    pub(crate) fn emitted_layers(&self) -> usize {
        self.selected_layers.iter().copied().max().unwrap_or(0) + 1
    }

    pub(crate) fn projected_rows_per_crop(&self) -> usize {
        (self.patches_per_crop / self.pool_h) / self.pool_w
    }

    pub(crate) fn fingerprint(&self) -> String {
        format!(
            "family=molmo-v1;crop={}x{}x{};max_crops={};vit={}x{}x{};layers={}/{};\
             selected={:?};pool={}x{};projector={}->{}->{};quant={}/{};merge={};eps={:08x};{}",
            self.patches_per_crop,
            self.patch_width,
            self.hidden,
            self.max_crops,
            self.hidden,
            self.heads,
            self.head_dim,
            self.emitted_layers(),
            self.layers,
            self.selected_layers,
            self.pool_h,
            self.pool_w,
            self.hidden,
            self.projector_hidden,
            self.text_hidden,
            self.bits,
            self.group_size,
            MOLMO_V1_MERGE_MODE,
            self.layer_norm_eps_bits,
            self.text_architecture,
        )
    }

    pub(crate) fn weight_specs(&self) -> Vec<MolmoVisionWeightSpec> {
        let mut specs = Vec::new();
        push_f32(
            &mut specs,
            "vision_tower.image_vit.class_embedding",
            &[self.hidden],
        );
        push_f32(
            &mut specs,
            "vision_tower.image_vit.patch_embedding.weight",
            &[self.hidden, self.patch_width],
        );
        push_f32(
            &mut specs,
            "vision_tower.image_vit.positional_embedding",
            &[self.positions, self.hidden],
        );
        push_f32(
            &mut specs,
            "vision_tower.image_vit.pre_ln.weight",
            &[self.hidden],
        );
        push_f32(
            &mut specs,
            "vision_tower.image_vit.pre_ln.bias",
            &[self.hidden],
        );
        for layer in 0..self.emitted_layers() {
            let prefix = format!("vision_tower.image_vit.transformer.resblocks.{layer}");
            push_f32(
                &mut specs,
                &format!("{prefix}.attention_norm.weight"),
                &[self.hidden],
            );
            push_f32(
                &mut specs,
                &format!("{prefix}.attention_norm.bias"),
                &[self.hidden],
            );
            for projection in ["wq", "wk", "wv", "wo"] {
                push_quant(
                    &mut specs,
                    &format!("{prefix}.attention.{projection}"),
                    self.hidden,
                    self.hidden,
                    self.bits,
                    self.group_size,
                );
                push_f32(
                    &mut specs,
                    &format!("{prefix}.attention.{projection}.bias"),
                    &[self.hidden],
                );
            }
            push_f32(
                &mut specs,
                &format!("{prefix}.ffn_norm.weight"),
                &[self.hidden],
            );
            push_f32(
                &mut specs,
                &format!("{prefix}.ffn_norm.bias"),
                &[self.hidden],
            );
            push_quant(
                &mut specs,
                &format!("{prefix}.feed_forward.w1"),
                self.intermediate,
                self.hidden,
                self.bits,
                self.group_size,
            );
            push_f32(
                &mut specs,
                &format!("{prefix}.feed_forward.w1.bias"),
                &[self.intermediate],
            );
            push_quant(
                &mut specs,
                &format!("{prefix}.feed_forward.w2"),
                self.hidden,
                self.intermediate,
                self.bits,
                self.group_size,
            );
            push_f32(
                &mut specs,
                &format!("{prefix}.feed_forward.w2.bias"),
                &[self.hidden],
            );
        }
        let selected_width = self
            .hidden
            .checked_mul(self.selected_layers.len())
            .expect("validated selected vision width");
        push_f32(&mut specs, "vision_tower.pad_embed", &[2, selected_width]);
        for projection in ["wq", "wk", "wv"] {
            push_quant(
                &mut specs,
                &format!("vision_tower.image_pooling_2d.{projection}"),
                self.hidden,
                selected_width,
                self.bits,
                self.group_size,
            );
            push_f32(
                &mut specs,
                &format!("vision_tower.image_pooling_2d.{projection}.bias"),
                &[self.hidden],
            );
        }
        push_quant(
            &mut specs,
            "vision_tower.image_pooling_2d.wo",
            self.hidden,
            self.hidden,
            self.bits,
            self.group_size,
        );
        push_f32(
            &mut specs,
            "vision_tower.image_pooling_2d.wo.bias",
            &[self.hidden],
        );
        for (name, output, input) in [
            ("w1", self.projector_hidden, self.hidden),
            ("w3", self.projector_hidden, self.hidden),
            ("w2", self.text_hidden, self.projector_hidden),
        ] {
            push_quant(
                &mut specs,
                &format!("vision_tower.image_projector.{name}"),
                output,
                input,
                self.bits,
                self.group_size,
            );
        }
        specs
    }
}

fn push_f32(specs: &mut Vec<MolmoVisionWeightSpec>, name: &str, shape: &[usize]) {
    specs.push(MolmoVisionWeightSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
        dtype: MolmoVisionWeightDType::Float32,
    });
}

fn push_quant(
    specs: &mut Vec<MolmoVisionWeightSpec>,
    prefix: &str,
    output: usize,
    input: usize,
    bits: usize,
    group_size: usize,
) {
    let packed_input = input / (32 / bits);
    let groups = input / group_size;
    specs.push(MolmoVisionWeightSpec {
        name: format!("{prefix}.weight"),
        shape: vec![output, packed_input],
        dtype: MolmoVisionWeightDType::Uint32,
    });
    for suffix in ["scales", "biases"] {
        specs.push(MolmoVisionWeightSpec {
            name: format!("{prefix}.{suffix}"),
            shape: vec![output, groups],
            dtype: MolmoVisionWeightDType::Float16,
        });
    }
}

#[cfg(test)]
#[path = "molmo_vision_config_tests.rs"]
mod tests;
