// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Exact PP-DocLayoutV3 tensor schema validation.

use std::collections::HashSet;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{HgNetV2Config, PpDocLayoutV3Config};

const MAX_PARAMETER_ELEMENTS: usize = 256 * 1024 * 1024;

pub fn validate(weights: &WeightMap, config: &PpDocLayoutV3Config) -> Result<i32, String> {
    let mut validator = Validator::new(weights)?;
    backbone_schema(&mut validator, &config.backbone_config)?;
    model_schema(&mut validator, config)?;
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
            .ok_or_else(|| "PP-DocLayoutV3 checkpoint has no tensors".to_string())?;
        if !matches!(
            dtype,
            mlxcel_core::dtype::FLOAT16
                | mlxcel_core::dtype::BFLOAT16
                | mlxcel_core::dtype::FLOAT32
        ) {
            return Err(format!(
                "PP-DocLayoutV3 checkpoint tensors must be floating point, got dtype {dtype}"
            ));
        }
        validate_parameter_elements(weights, MAX_PARAMETER_ELEMENTS)?;
        Ok(Self {
            weights,
            seen: HashSet::with_capacity(weights.len()),
            dtype,
        })
    }

    fn tensor(&mut self, key: String, shape: &[usize]) -> Result<(), String> {
        let value = self
            .weights
            .get(&key)
            .ok_or_else(|| format!("PP-DocLayoutV3 weight not found: {key}"))?;
        let actual_dtype = mlxcel_core::array_dtype(value);
        if actual_dtype != self.dtype {
            return Err(format!(
                "PP-DocLayoutV3 weight {key} has dtype {actual_dtype}; expected checkpoint dtype {}",
                self.dtype
            ));
        }
        let expected: Vec<i32> = shape
            .iter()
            .map(|&value| {
                i32::try_from(value).map_err(|_| {
                    format!("PP-DocLayoutV3 expected shape for {key} exceeds the i32 backend limit")
                })
            })
            .collect::<Result<_, _>>()?;
        let actual = mlxcel_core::array_shape(value);
        if actual != expected {
            return Err(format!(
                "PP-DocLayoutV3 weight {key} has shape {actual:?}; expected {expected:?}"
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

    fn conv_norm(
        &mut self,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<(), String> {
        self.tensor(
            format!("{prefix}.conv.weight"),
            &[output, kernel, kernel, input / groups],
        )?;
        for name in ["weight", "bias", "running_mean", "running_var"] {
            self.tensor(format!("{prefix}.bn.{name}"), &[output])?;
        }
        Ok(())
    }

    fn mlp(&mut self, prefix: &str, dimensions: &[usize]) -> Result<(), String> {
        for index in 0..dimensions.len() - 1 {
            self.linear(
                &format!("{prefix}.layers.{index}"),
                dimensions[index],
                dimensions[index + 1],
            )?;
        }
        Ok(())
    }

    fn finish(self) -> Result<i32, String> {
        if self.seen.len() != self.weights.len() {
            let mut unexpected: Vec<&str> = self
                .weights
                .keys()
                .filter(|key| !self.seen.contains(*key))
                .map(String::as_str)
                .collect();
            unexpected.sort_unstable();
            return Err(format!(
                "PP-DocLayoutV3 checkpoint has {} unexpected tensor(s): {}",
                unexpected.len(),
                unexpected
                    .into_iter()
                    .take(8)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        ensure_weights_finite(self.weights)?;
        Ok(self.dtype)
    }
}

fn validate_parameter_elements(weights: &WeightMap, limit: usize) -> Result<(), String> {
    let mut elements = 0usize;
    for (name, value) in weights {
        let shape = mlxcel_core::array_shape(value);
        let count = shape.iter().try_fold(1usize, |count, &dimension| {
            let dimension = usize::try_from(dimension)
                .map_err(|_| format!("PP-DocLayoutV3 weight {name} has invalid shape {shape:?}"))?;
            count
                .checked_mul(dimension)
                .ok_or_else(|| format!("PP-DocLayoutV3 weight {name} element count overflow"))
        })?;
        elements = elements
            .checked_add(count)
            .ok_or_else(|| "PP-DocLayoutV3 parameter count overflow".to_string())?;
        if elements > limit {
            return Err(format!(
                "PP-DocLayoutV3 checkpoint has more than {limit} parameter elements"
            ));
        }
    }
    Ok(())
}

fn ensure_weights_finite(weights: &WeightMap) -> Result<(), String> {
    let mut entries: Vec<_> = weights.iter().collect();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let checks: Vec<UniquePtr<MlxArray>> = entries
        .iter()
        .map(|(_, value)| mlxcel_core::all_all(&mlxcel_core::isfinite(value)))
        .collect();
    let finite = mlxcel_core::all_all(&mlxcel_core::stack_owned(&checks, 0));
    if mlxcel_core::item_bool(&finite) {
        return Ok(());
    }
    for (name, value) in entries {
        if !mlxcel_core::item_bool(&mlxcel_core::all_all(&mlxcel_core::isfinite(value))) {
            return Err(format!(
                "PP-DocLayoutV3 weight {name} contains non-finite values"
            ));
        }
    }
    Err("PP-DocLayoutV3 checkpoint contains non-finite values".into())
}

fn backbone_schema(v: &mut Validator<'_>, config: &HgNetV2Config) -> Result<(), String> {
    let stem = &config.stem_channels;
    v.conv_norm("backbone.embedder.stem1", stem[0], stem[1], 3, 1)?;
    v.conv_norm("backbone.embedder.stem2a", stem[1], stem[1] / 2, 2, 1)?;
    v.conv_norm("backbone.embedder.stem2b", stem[1] / 2, stem[1], 2, 1)?;
    v.conv_norm("backbone.embedder.stem3", stem[1] * 2, stem[1], 3, 1)?;
    v.conv_norm("backbone.embedder.stem4", stem[1], stem[2], 1, 1)?;

    for stage in 0..4 {
        let prefix = format!("backbone.encoder.stages.{stage}");
        let input = config.stage_in_channels[stage];
        let middle = config.stage_mid_channels[stage];
        let output = config.stage_out_channels[stage];
        let kernel = config.stage_kernel_size[stage];
        if config.stage_downsample[stage] {
            v.conv_norm(&format!("{prefix}.downsample"), input, input, 3, input)?;
        }
        for block in 0..config.stage_num_blocks[stage] {
            let block_prefix = format!("{prefix}.blocks.{block}");
            let block_input = if block == 0 { input } else { output };
            for layer in 0..config.stage_numb_of_layers[stage] {
                let layer_prefix = format!("{block_prefix}.layers.{layer}");
                let layer_input = if layer == 0 { block_input } else { middle };
                if config.stage_light_block[stage] {
                    v.conv_norm(&format!("{layer_prefix}.conv1"), layer_input, middle, 1, 1)?;
                    v.conv_norm(
                        &format!("{layer_prefix}.conv2"),
                        middle,
                        middle,
                        kernel,
                        middle,
                    )?;
                } else {
                    v.conv_norm(&layer_prefix, layer_input, middle, kernel, 1)?;
                }
            }
            let joined = block_input + config.stage_numb_of_layers[stage] * middle;
            v.conv_norm(
                &format!("{block_prefix}.aggregation.0"),
                joined,
                output / 2,
                1,
                1,
            )?;
            v.conv_norm(
                &format!("{block_prefix}.aggregation.1"),
                output / 2,
                output,
                1,
                1,
            )?;
        }
    }
    Ok(())
}

fn csp_schema(
    v: &mut Validator<'_>,
    prefix: &str,
    input: usize,
    output: usize,
    expansion: f32,
) -> Result<(), String> {
    let hidden = (output as f32 * expansion) as usize;
    v.conv_norm(&format!("{prefix}.conv1"), input, hidden, 1, 1)?;
    v.conv_norm(&format!("{prefix}.conv2"), input, hidden, 1, 1)?;
    for index in 0..3 {
        v.conv_norm(
            &format!("{prefix}.bottlenecks.{index}.conv1"),
            hidden,
            hidden,
            3,
            1,
        )?;
        v.conv_norm(
            &format!("{prefix}.bottlenecks.{index}.conv2"),
            hidden,
            hidden,
            1,
            1,
        )?;
    }
    if hidden != output {
        v.conv_norm(&format!("{prefix}.conv3"), hidden, output, 1, 1)?;
    }
    Ok(())
}

fn attention_schema(v: &mut Validator<'_>, prefix: &str, dimension: usize) -> Result<(), String> {
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        v.linear(&format!("{prefix}.{name}"), dimension, dimension)?;
    }
    Ok(())
}

fn model_schema(v: &mut Validator<'_>, c: &PpDocLayoutV3Config) -> Result<(), String> {
    let d = c.d_model;
    for (index, &input) in c.encoder_in_channels.iter().enumerate() {
        v.conv_norm(
            &format!("encoder_input_proj.{index}"),
            input,
            c.encoder_hidden_dim,
            1,
            1,
        )?;
    }
    for layer in 0..c.encoder_layers {
        let prefix = format!("encoder.aifi.0.layers.{layer}");
        attention_schema(v, &format!("{prefix}.self_attn"), c.encoder_hidden_dim)?;
        v.norm(
            &format!("{prefix}.self_attn_layer_norm"),
            c.encoder_hidden_dim,
        )?;
        v.linear(
            &format!("{prefix}.fc1"),
            c.encoder_hidden_dim,
            c.encoder_ffn_dim,
        )?;
        v.linear(
            &format!("{prefix}.fc2"),
            c.encoder_ffn_dim,
            c.encoder_hidden_dim,
        )?;
        v.norm(&format!("{prefix}.final_layer_norm"), c.encoder_hidden_dim)?;
    }
    for index in 0..c.encoder_in_channels.len() - 1 {
        v.conv_norm(&format!("encoder.lateral_convs.{index}"), d, d, 1, 1)?;
        csp_schema(
            v,
            &format!("encoder.fpn_blocks.{index}"),
            2 * d,
            d,
            c.hidden_expansion,
        )?;
        v.conv_norm(&format!("encoder.downsample_convs.{index}"), d, d, 3, 1)?;
        csp_schema(
            v,
            &format!("encoder.pan_blocks.{index}"),
            2 * d,
            d,
            c.hidden_expansion,
        )?;
    }
    mask_schema(v, c)?;
    v.linear("enc_output.fc", d, d)?;
    v.norm("enc_output.ln", d)?;
    v.linear("enc_score_head", d, c.num_labels)?;
    v.mlp("enc_bbox_head", &[d, d, d, 4])?;
    for (index, &input) in c.decoder_in_channels.iter().enumerate() {
        v.conv_norm(&format!("decoder_input_proj.{index}"), input, d, 1, 1)?;
    }
    decoder_schema(v, c)?;
    Ok(())
}

fn mask_schema(v: &mut Validator<'_>, c: &PpDocLayoutV3Config) -> Result<(), String> {
    let feat = c.mask_feature_channels[0];
    let base = *c.feat_strides.iter().min().expect("validated strides");
    for (index, &stride) in c.feat_strides.iter().enumerate() {
        let ratio = stride / base;
        let count = 1usize.max((usize::BITS - ratio.leading_zeros() - 1) as usize);
        for layer in 0..count {
            let key_index = if stride == base { layer } else { layer * 2 };
            v.conv_norm(
                &format!("encoder.mask_feature_head.scale_heads.{index}.layers.{key_index}"),
                if layer == 0 { c.d_model } else { feat },
                feat,
                3,
                1,
            )?;
        }
    }
    v.conv_norm(
        "encoder.mask_feature_head.output_conv",
        feat,
        c.mask_feature_channels[1],
        3,
        1,
    )?;
    v.conv_norm(
        "encoder.encoder_mask_lateral",
        c.x4_feat_dim,
        c.mask_feature_channels[1],
        3,
        1,
    )?;
    v.conv_norm(
        "encoder.encoder_mask_output.base_conv",
        c.mask_feature_channels[1],
        c.mask_feature_channels[1],
        3,
        1,
    )?;
    v.tensor(
        "encoder.encoder_mask_output.conv.weight".into(),
        &[c.num_prototypes, 1, 1, c.mask_feature_channels[1]],
    )?;
    v.tensor(
        "encoder.encoder_mask_output.conv.bias".into(),
        &[c.num_prototypes],
    )
}

fn decoder_schema(v: &mut Validator<'_>, c: &PpDocLayoutV3Config) -> Result<(), String> {
    let d = c.d_model;
    let points = c.decoder_attention_heads * c.num_feature_levels * c.decoder_n_points;
    for layer in 0..c.decoder_layers {
        let prefix = format!("decoder.layers.{layer}");
        attention_schema(v, &format!("{prefix}.self_attn"), d)?;
        v.norm(&format!("{prefix}.self_attn_layer_norm"), d)?;
        v.linear(
            &format!("{prefix}.encoder_attn.sampling_offsets"),
            d,
            points * 2,
        )?;
        v.linear(
            &format!("{prefix}.encoder_attn.attention_weights"),
            d,
            points,
        )?;
        v.linear(&format!("{prefix}.encoder_attn.value_proj"), d, d)?;
        v.linear(&format!("{prefix}.encoder_attn.output_proj"), d, d)?;
        v.norm(&format!("{prefix}.encoder_attn_layer_norm"), d)?;
        v.linear(&format!("{prefix}.fc1"), d, c.decoder_ffn_dim)?;
        v.linear(&format!("{prefix}.fc2"), c.decoder_ffn_dim, d)?;
        v.norm(&format!("{prefix}.final_layer_norm"), d)?;
        v.linear(&format!("decoder_order_head.{layer}"), d, d)?;
    }
    v.mlp("decoder.query_pos_head", &[4, 2 * d, d])?;
    v.linear(
        "decoder_global_pointer.dense",
        d,
        2 * c.global_pointer_head_size,
    )?;
    v.norm("decoder_norm", d)?;
    v.mlp("mask_query_head", &[d, d, d, c.num_prototypes])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_finite_validation_rejects_nan() {
        let mut weights = WeightMap::new();
        weights.insert(
            "finite".into(),
            mlxcel_core::from_slice_f32(&[1.0, 2.0], &[2]),
        );
        assert!(ensure_weights_finite(&weights).is_ok());
        weights.insert(
            "nonfinite".into(),
            mlxcel_core::from_slice_f32(&[f32::NAN], &[1]),
        );
        let error = ensure_weights_finite(&weights).unwrap_err();
        assert!(error.contains("nonfinite"));
    }

    #[test]
    fn aggregate_parameter_limit_is_checked_before_evaluation() {
        let mut weights = WeightMap::new();
        weights.insert(
            "one".into(),
            mlxcel_core::from_slice_f32(&[0.0; 4], &[2, 2]),
        );
        weights.insert("two".into(), mlxcel_core::from_slice_f32(&[0.0; 3], &[3]));
        assert!(validate_parameter_elements(&weights, 7).is_ok());
        assert!(
            validate_parameter_elements(&weights, 6)
                .unwrap_err()
                .contains("more than 6")
        );
    }
}
