// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Exact released RF-DETR checkpoint schema validation.

use std::collections::HashSet;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::RfDetrConfig;

const MAX_PARAMETER_ELEMENTS: usize = 64 * 1024 * 1024;

pub fn validate(weights: &WeightMap, config: &RfDetrConfig) -> Result<i32, String> {
    let mut validator = Validator::new(weights)?;
    backbone(&mut validator, config)?;
    projector(&mut validator)?;
    transformer(&mut validator, config)?;
    heads(&mut validator, config)?;
    if config.segmentation {
        segmentation(&mut validator, config.seg_num_blocks)?;
    }
    validator.finish()
}

struct Validator<'a> {
    weights: &'a WeightMap,
    seen: HashSet<String>,
    dtype: i32,
}

impl<'a> Validator<'a> {
    fn new(weights: &'a WeightMap) -> Result<Self, String> {
        let dtype = weights
            .values()
            .next()
            .map(|value| mlxcel_core::array_dtype(value))
            .ok_or_else(|| "RF-DETR checkpoint contains no tensors".to_string())?;
        if dtype != mlxcel_core::dtype::FLOAT32 {
            return Err(format!(
                "released RF-DETR checkpoints must be F32, got dtype {dtype}"
            ));
        }
        let mut total = 0usize;
        for (name, value) in weights {
            let shape = mlxcel_core::array_shape(value);
            let count = shape.iter().try_fold(1usize, |count, &dimension| {
                let dimension = usize::try_from(dimension)
                    .map_err(|_| format!("RF-DETR weight {name} has invalid shape {shape:?}"))?;
                count
                    .checked_mul(dimension)
                    .ok_or_else(|| format!("RF-DETR weight {name} size overflow"))
            })?;
            total = total
                .checked_add(count)
                .ok_or_else(|| "RF-DETR parameter count overflow".to_string())?;
        }
        if total > MAX_PARAMETER_ELEMENTS {
            return Err(format!(
                "RF-DETR checkpoint has {total} parameter elements; limit is {MAX_PARAMETER_ELEMENTS}"
            ));
        }
        Ok(Self {
            weights,
            seen: HashSet::with_capacity(weights.len()),
            dtype,
        })
    }

    fn tensor(&mut self, key: impl Into<String>, shape: &[usize]) -> Result<(), String> {
        let key = key.into();
        let value = self
            .weights
            .get(&key)
            .ok_or_else(|| format!("RF-DETR weight not found: {key}"))?;
        let actual = mlxcel_core::array_shape(value);
        let expected: Vec<i32> = shape
            .iter()
            .map(|&value| {
                i32::try_from(value).map_err(|_| format!("RF-DETR shape overflow: {key}"))
            })
            .collect::<Result<_, _>>()?;
        if actual != expected || mlxcel_core::array_dtype(value) != self.dtype {
            return Err(format!(
                "RF-DETR weight {key} has shape {actual:?}/dtype {}; expected {expected:?}/dtype {}",
                mlxcel_core::array_dtype(value),
                self.dtype
            ));
        }
        self.seen.insert(key);
        Ok(())
    }

    fn linear(&mut self, prefix: &str, input: usize, output: usize) -> Result<(), String> {
        self.tensor(format!("{prefix}.weight"), &[output, input])?;
        self.tensor(format!("{prefix}.bias"), &[output])
    }

    fn norm(&mut self, prefix: &str, channels: usize) -> Result<(), String> {
        self.tensor(format!("{prefix}.weight"), &[channels])?;
        self.tensor(format!("{prefix}.bias"), &[channels])
    }

    fn mlp(&mut self, prefix: &str, dims: &[usize]) -> Result<(), String> {
        for index in 0..dims.len() - 1 {
            self.linear(
                &format!("{prefix}.layers.{index}"),
                dims[index],
                dims[index + 1],
            )?;
        }
        Ok(())
    }

    fn conv(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        groups: usize,
        bias: bool,
    ) -> Result<(), String> {
        self.tensor(
            format!("{prefix}.weight"),
            &[output, kernel, kernel, input / groups],
        )?;
        if bias {
            self.tensor(format!("{prefix}.bias"), &[output])?;
        }
        Ok(())
    }

    fn finish(self) -> Result<i32, String> {
        if self.seen.len() != self.weights.len() {
            let mut unexpected: Vec<_> = self
                .weights
                .keys()
                .filter(|key| !self.seen.contains(*key))
                .map(String::as_str)
                .collect();
            unexpected.sort_unstable();
            return Err(format!(
                "RF-DETR checkpoint has {} unexpected tensors: {}",
                unexpected.len(),
                unexpected
                    .into_iter()
                    .take(8)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        ensure_finite(self.weights)?;
        Ok(self.dtype)
    }
}

fn backbone(v: &mut Validator<'_>, config: &RfDetrConfig) -> Result<(), String> {
    v.tensor("backbone.embeddings.cls_token", &[1, 1, 384])?;
    let pos_side = config.positional_encoding_size.unwrap_or(37);
    v.tensor(
        "backbone.embeddings.position_embeddings",
        &[1, 1 + pos_side * pos_side, 384],
    )?;
    v.conv(
        "backbone.embeddings.patch_embeddings.projection",
        3,
        384,
        config.patch_size,
        1,
        true,
    )?;
    for index in 0..12 {
        let prefix = format!("backbone.encoder.layers.{index}");
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            v.linear(&format!("{prefix}.attention.{projection}"), 384, 384)?;
        }
        v.tensor(format!("{prefix}.layer_scale1"), &[384])?;
        v.tensor(format!("{prefix}.layer_scale2"), &[384])?;
        v.linear(&format!("{prefix}.mlp.fc1"), 384, 1536)?;
        v.linear(&format!("{prefix}.mlp.fc2"), 1536, 384)?;
        v.norm(&format!("{prefix}.norm1"), 384)?;
        v.norm(&format!("{prefix}.norm2"), 384)?;
    }
    v.norm("backbone.layernorm", 384)
}

fn projector(v: &mut Validator<'_>) -> Result<(), String> {
    let prefix = "projector.stages.0.0";
    conv_norm(v, &format!("{prefix}.cv1"), 1536, 256, 1)?;
    for index in 0..3 {
        conv_norm(v, &format!("{prefix}.m.{index}.cv1"), 128, 128, 3)?;
        conv_norm(v, &format!("{prefix}.m.{index}.cv2"), 128, 128, 3)?;
    }
    conv_norm(v, &format!("{prefix}.cv2"), 640, 256, 1)?;
    v.norm("projector.stages.0.1", 256)
}

fn conv_norm(
    v: &mut Validator<'_>,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
) -> Result<(), String> {
    v.conv(&format!("{prefix}.conv"), input, output, kernel, 1, false)?;
    v.norm(&format!("{prefix}.bn"), output)
}

fn transformer(v: &mut Validator<'_>, config: &RfDetrConfig) -> Result<(), String> {
    for group in 0..config.group_detr {
        v.linear(&format!("transformer.enc_output.{group}"), 256, 256)?;
        v.norm(&format!("transformer.enc_output_norm.{group}"), 256)?;
        v.linear(
            &format!("transformer.enc_out_class_embed.{group}"),
            256,
            config.logits_classes(),
        )?;
        v.mlp(
            &format!("transformer.enc_out_bbox_embed.{group}"),
            &[256, 256, 256, 4],
        )?;
    }
    for layer in 0..config.dec_layers {
        let prefix = format!("transformer.decoder.layers.{layer}");
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            v.linear(&format!("{prefix}.self_attn.{name}"), 256, 256)?;
        }
        v.linear(&format!("{prefix}.cross_attn.sampling_offsets"), 256, 64)?;
        v.linear(&format!("{prefix}.cross_attn.attention_weights"), 256, 32)?;
        v.linear(&format!("{prefix}.cross_attn.value_proj"), 256, 256)?;
        v.linear(&format!("{prefix}.cross_attn.output_proj"), 256, 256)?;
        v.linear(&format!("{prefix}.linear1"), 256, 2048)?;
        v.linear(&format!("{prefix}.linear2"), 2048, 256)?;
        for norm in ["norm1", "norm2", "norm3"] {
            v.norm(&format!("{prefix}.{norm}"), 256)?;
        }
    }
    v.norm("transformer.decoder.norm", 256)?;
    v.mlp("transformer.decoder.ref_point_head", &[512, 256, 256])
}

fn heads(v: &mut Validator<'_>, config: &RfDetrConfig) -> Result<(), String> {
    v.linear("class_embed", 256, config.logits_classes())?;
    v.mlp("bbox_embed", &[256, 256, 256, 4])?;
    v.tensor(
        "query_feat.weight",
        &[config.num_queries * config.group_detr, 256],
    )?;
    v.tensor(
        "refpoint_embed.weight",
        &[config.num_queries * config.group_detr, 4],
    )
}

fn segmentation(v: &mut Validator<'_>, blocks: usize) -> Result<(), String> {
    v.tensor("segmentation_head.bias", &[1])?;
    for index in 0..blocks {
        let prefix = format!("segmentation_head.blocks.{index}");
        v.conv(&format!("{prefix}.dwconv"), 256, 256, 3, 256, true)?;
        v.norm(&format!("{prefix}.norm"), 256)?;
        v.linear(&format!("{prefix}.pwconv1"), 256, 256)?;
    }
    v.conv(
        "segmentation_head.spatial_features_proj",
        256,
        256,
        1,
        1,
        true,
    )?;
    v.norm("segmentation_head.query_features_block.norm_in", 256)?;
    v.linear("segmentation_head.query_features_block.layers.0", 256, 1024)?;
    v.linear("segmentation_head.query_features_block.layers.2", 1024, 256)?;
    v.linear("segmentation_head.query_features_proj", 256, 256)
}

fn ensure_finite(weights: &WeightMap) -> Result<(), String> {
    let mut names: Vec<_> = weights.keys().collect();
    names.sort_unstable();
    let checks: Vec<UniquePtr<MlxArray>> = names
        .iter()
        .map(|name| mlxcel_core::all_all(&mlxcel_core::isfinite(&weights[*name])))
        .collect();
    if mlxcel_core::item_bool(&mlxcel_core::all_all(&mlxcel_core::stack_owned(&checks, 0))) {
        Ok(())
    } else {
        Err("RF-DETR checkpoint contains non-finite values".into())
    }
}
