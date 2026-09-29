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

//! Nemotron-Parse `config.json` parsing.
//!
//! The checkpoint nests two sub-configs: `encoder` (a C-RADIO wrapper config
//! whose ViT geometry is implied by `args.model`, not spelled out) and
//! `decoder` (an mBART config; note the key is `decoder`, not `text_config`).
//! The top level carries the token ids and the decode-length bound.
//!
//! Reference: `hf_nemotron_parse_config.py` and `hf_nemotron_parse_modeling.py`
//! shipped with `nvidia/NVIDIA-Nemotron-Parse-2.0`.

use anyhow::{Result, anyhow};
use serde_json::Value;

use crate::models::florence2::Florence2Quantization;

/// C-RADIO ViT tower plus compression-neck geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronParseVisionConfig {
    /// ViT width (1280 for `vit_huge_patch16_224`).
    pub hidden_size: i32,
    /// Attention heads per block (16 for ViT-H).
    pub num_heads: i32,
    /// MLP hidden width (`hidden_size * mlp_ratio`, 5120 for ViT-H).
    pub mlp_dim: i32,
    /// Transformer depth (32 for ViT-H).
    pub num_layers: usize,
    /// Square patch side in pixels.
    pub patch_size: i32,
    /// Side of the learned positional grid (`max_resolution / patch_size`).
    pub pos_grid: i32,
    /// One CLS token per distillation teacher.
    pub num_cls_tokens: i32,
    /// Register tokens that pad the prefix to `register_multiple`.
    pub num_register_tokens: i32,
    /// CLS rows concatenated into the summary vector (teachers with
    /// `use_summary`).
    pub summary_idxs: Vec<i32>,
    /// Neck output width, equal to the decoder `d_model`.
    pub neck_dim: i32,
    /// Horizontal compression factor of the neck's `(1, k)` conv.
    pub neck_kernel_w: i32,
    /// Page size the processor pads to, `(height, width)`.
    pub image_size: (i32, i32),
}

impl NemotronParseVisionConfig {
    /// Prefix tokens (CLS + registers) in front of the patch sequence.
    pub fn num_prefix_tokens(&self) -> i32 {
        self.num_cls_tokens + self.num_register_tokens
    }

    /// Patch grid `(rows, cols)` for an `height x width` input.
    pub fn patch_grid(&self, height: i32, width: i32) -> (i32, i32) {
        (height / self.patch_size, width / self.patch_size)
    }

    /// Number of encoder states the decoder cross-attends to for an
    /// `height x width` page: the compressed patch grid plus one summary row.
    pub fn encoder_seq_len(&self, height: i32, width: i32) -> i32 {
        let (hp, wp) = self.patch_grid(height, width);
        hp * (wp / self.neck_kernel_w) + 1
    }
}

/// Pre-norm mBART decoder geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronParseTextConfig {
    pub d_model: i32,
    pub decoder_layers: usize,
    pub decoder_attention_heads: i32,
    pub decoder_ffn_dim: i32,
    pub vocab_size: i32,
    /// Multiply token embeddings by `sqrt(d_model)`.
    pub scale_embedding: bool,
}

/// Whole-model configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronParseConfig {
    pub vision: NemotronParseVisionConfig,
    pub text: NemotronParseTextConfig,
    pub decoder_start_token_id: i32,
    pub eos_token_id: i32,
    pub bos_token_id: i32,
    pub pad_token_id: i32,
    pub tie_word_embeddings: bool,
    /// Upper bound on decoder positions (seed plus generated tokens). The
    /// decoder has no positional table, so this is a length policy, not a
    /// memory-safety bound.
    pub max_sequence_length: i32,
    pub quantization: Florence2Quantization,
}

fn int_field(obj: Option<&Value>, key: &str) -> Result<Option<i64>> {
    match obj.and_then(|o| o.get(key)) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_i64()
            .map(Some)
            .ok_or_else(|| anyhow!("Nemotron-Parse config field {key} is not an integer: {v}")),
    }
}

fn i32_field(obj: Option<&Value>, key: &str, default: i32) -> Result<i32> {
    match int_field(obj, key)? {
        None => Ok(default),
        Some(v) => i32::try_from(v)
            .map_err(|_| anyhow!("Nemotron-Parse config field {key} = {v} does not fit in i32")),
    }
}

fn positive(value: i32, key: &str) -> Result<i32> {
    if value > 0 {
        Ok(value)
    } else {
        Err(anyhow!(
            "Nemotron-Parse config field {key} must be positive, got {value}"
        ))
    }
}

/// `(hidden_size, num_layers, num_heads)` of the timm ViT names the C-RADIO
/// wrapper config uses in `args.model`.
fn vit_geometry(model: &str) -> Result<(i32, usize, i32)> {
    let base = model.split("_patch").next().unwrap_or(model);
    match base {
        "vit_huge" => Ok((1280, 32, 16)),
        "vit_large" => Ok((1024, 24, 16)),
        "vit_base" => Ok((768, 12, 12)),
        _ => Err(anyhow!(
            "Nemotron-Parse encoder args.model {model:?} is not a supported C-RADIO ViT \
             (expected vit_huge/vit_large/vit_base _patch16)"
        )),
    }
}

impl NemotronParseVisionConfig {
    fn from_model_config(config: &Value) -> Result<Self> {
        let enc = config.get("encoder");
        let args = enc.and_then(|e| e.get("args"));
        let model_name = args
            .and_then(|a| a.get("model"))
            .and_then(Value::as_str)
            .unwrap_or("vit_huge_patch16_224");
        let (hidden_size, num_layers, num_heads) = vit_geometry(model_name)?;
        let patch_size = positive(i32_field(enc, "patch_size", 16)?, "encoder.patch_size")?;
        let max_resolution = positive(
            i32_field(enc, "max_resolution", 2048)?,
            "encoder.max_resolution",
        )?;
        if max_resolution % patch_size != 0 {
            return Err(anyhow!(
                "Nemotron-Parse encoder.max_resolution {max_resolution} is not a multiple of \
                 patch_size {patch_size}"
            ));
        }

        // One CLS token per teacher when `cls_token_per_teacher`, and the
        // summary is built from the CLS rows of the teachers that set
        // `use_summary` (clip, siglip, dino_v2 for C-RADIOv2-H; not sam).
        let teachers = args
            .and_then(|a| a.get("teachers"))
            .and_then(Value::as_array);
        let per_teacher = args
            .and_then(|a| a.get("cls_token_per_teacher"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let (num_cls_tokens, summary_idxs) = match (teachers, per_teacher) {
            (Some(list), true) if !list.is_empty() => {
                let idxs: Vec<i32> = list
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| {
                        t.get("use_summary")
                            .and_then(Value::as_bool)
                            .unwrap_or(true)
                    })
                    .map(|(i, _)| i as i32)
                    .collect();
                (list.len() as i32, idxs)
            }
            (_, false) => (1, vec![0]),
            _ => (4, vec![0, 1, 2]),
        };
        let register_multiple = i32_field(args, "register_multiple", 8)?;
        let num_register_tokens = if register_multiple > 0 {
            (register_multiple - num_cls_tokens % register_multiple) % register_multiple
        } else {
            0
        };

        let image_size = match config.get("image_size").and_then(Value::as_array) {
            Some(dims) if dims.len() == 2 => {
                let h = dims[0].as_i64().unwrap_or(2048) as i32;
                let w = dims[1].as_i64().unwrap_or(1664) as i32;
                (positive(h, "image_size[0]")?, positive(w, "image_size[1]")?)
            }
            _ => (2048, 1664),
        };

        Ok(Self {
            hidden_size,
            num_heads,
            mlp_dim: hidden_size * 4,
            num_layers,
            patch_size,
            pos_grid: max_resolution / patch_size,
            num_cls_tokens,
            num_register_tokens,
            summary_idxs,
            neck_dim: 1024,
            neck_kernel_w: 4,
            image_size,
        })
    }
}

impl NemotronParseConfig {
    /// Parse a Nemotron-Parse `config.json`.
    pub fn from_model_config(config: &Value) -> Result<Self> {
        let dec = config.get("decoder");
        if dec.is_none() {
            return Err(anyhow!(
                "Nemotron-Parse config.json has no `decoder` sub-config"
            ));
        }
        let top = Some(config);

        let d_model = positive(i32_field(dec, "d_model", 1024)?, "decoder.d_model")?;
        let heads = positive(
            i32_field(dec, "decoder_attention_heads", 16)?,
            "decoder.decoder_attention_heads",
        )?;
        if d_model % heads != 0 {
            return Err(anyhow!(
                "Nemotron-Parse decoder.d_model {d_model} is not divisible by \
                 decoder_attention_heads {heads}"
            ));
        }
        let decoder_layers = positive(
            i32_field(dec, "decoder_layers", 10)?,
            "decoder.decoder_layers",
        )? as usize;
        let vocab_size = positive(
            i32_field(dec, "vocab_size", i32_field(top, "vocab_size", 72256)?)?,
            "decoder.vocab_size",
        )?;
        let text = NemotronParseTextConfig {
            d_model,
            decoder_layers,
            decoder_attention_heads: heads,
            decoder_ffn_dim: positive(
                i32_field(dec, "decoder_ffn_dim", 4096)?,
                "decoder.decoder_ffn_dim",
            )?,
            vocab_size,
            scale_embedding: dec
                .and_then(|d| d.get("scale_embedding"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
        };
        if let Some(act) = dec
            .and_then(|d| d.get("activation_function"))
            .and_then(Value::as_str)
            && act != "gelu"
        {
            return Err(anyhow!(
                "Nemotron-Parse decoder.activation_function {act:?} is not supported (gelu only)"
            ));
        }

        let mut vision = NemotronParseVisionConfig::from_model_config(config)?;
        vision.neck_dim = d_model;

        // `decoder.decoder_start_token_id` is null on v2.0; the top-level
        // value (2, i.e. `</s>`) is the one generation uses.
        let decoder_start_token_id = match int_field(top, "decoder_start_token_id")? {
            Some(_) => i32_field(top, "decoder_start_token_id", 2)?,
            None => i32_field(dec, "decoder_start_token_id", 2)?,
        };
        let token_id = |key: &str, default: i32| -> Result<i32> {
            let v = i32_field(top, key, i32_field(dec, key, default)?)?;
            if !(0..vocab_size).contains(&v) {
                return Err(anyhow!(
                    "Nemotron-Parse {key} {v} is outside the vocabulary (0..{vocab_size})"
                ));
            }
            Ok(v)
        };
        let eos_token_id = token_id("eos_token_id", 2)?;
        if !(0..vocab_size).contains(&decoder_start_token_id) {
            return Err(anyhow!(
                "Nemotron-Parse decoder_start_token_id {decoder_start_token_id} is outside the \
                 vocabulary (0..{vocab_size})"
            ));
        }

        let max_sequence_length = positive(
            i32_field(top, "max_sequence_length", 9000)?,
            "max_sequence_length",
        )?;

        Ok(Self {
            vision,
            text,
            decoder_start_token_id,
            eos_token_id,
            bos_token_id: token_id("bos_token_id", 0)?,
            pad_token_id: token_id("pad_token_id", 1)?,
            tie_word_embeddings: config
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            max_sequence_length,
            quantization: Florence2Quantization::from_model_config(config)?,
        })
    }
}
