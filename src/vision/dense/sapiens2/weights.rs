// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Sapiens2 checkpoint loading and layout guards.

use anyhow::{Context, Result, ensure};
use mlxcel_core::layers::{Linear, RMSNorm};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::Sapiens2Config;

pub fn load_sanitized_weights(
    path: &std::path::Path,
    config: &Sapiens2Config,
) -> Result<WeightMap> {
    let config_path = path.join("config.json");
    let raw_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?,
    )
    .with_context(|| format!("parse {}", config_path.display()))?;
    let mut weights = load_weights_from_dir(path).map_err(anyhow::Error::msg)?;
    prefix_bare_backbone(&mut weights);
    merge_qkv(&mut weights, config)?;
    if crate::models::bf16_to_f16_at_load(false, Some(&raw_config)) {
        let _ = crate::models::convert_bf16_weights(&mut weights);
    }
    normalize_floating_dtype(&mut weights)?;
    Ok(weights)
}

fn normalize_floating_dtype(weights: &mut WeightMap) -> Result<()> {
    let target = weights
        .iter()
        .find(|(key, _)| key.ends_with("attn.wqkv.weight"))
        .map(|(_, value)| mlxcel_core::array_dtype(value))
        .ok_or_else(|| anyhow::anyhow!("Sapiens2 checkpoint has no attention projection"))?;
    let floating = |dtype| {
        matches!(
            dtype,
            mlxcel_core::dtype::FLOAT16
                | mlxcel_core::dtype::BFLOAT16
                | mlxcel_core::dtype::FLOAT32
        )
    };
    ensure!(
        floating(target),
        "quantized Sapiens2 attention weights are not supported"
    );
    let keys: Vec<String> = weights
        .iter()
        .filter(|(_, value)| {
            let dtype = mlxcel_core::array_dtype(value);
            floating(dtype) && dtype != target
        })
        .map(|(key, _)| key.clone())
        .collect();
    for key in keys {
        let converted = mlxcel_core::astype(&weights[&key], target);
        mlxcel_core::eval(&converted);
        weights.insert(key, converted);
    }
    Ok(())
}

fn prefix_bare_backbone(weights: &mut WeightMap) {
    let bare: Vec<String> = weights
        .keys()
        .filter(|key| {
            key.starts_with("blocks.")
                || key.starts_with("patch_embed.")
                || key.starts_with("rope_embed.")
                || matches!(key.as_str(), "cls_token" | "storage_tokens" | "ln1.weight")
        })
        .cloned()
        .collect();
    for key in bare {
        if let Some(value) = weights.remove(&key) {
            weights.insert(format!("backbone.{key}"), value);
        }
    }
}

fn merge_qkv(weights: &mut WeightMap, config: &Sapiens2Config) -> Result<()> {
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("backbone.blocks.{layer}.attn");
        for suffix in ["weight", "bias"] {
            let fused = format!("{prefix}.wqkv.{suffix}");
            if weights.contains_key(&fused) {
                continue;
            }
            let keys = ["q", "k", "v"].map(|name| format!("{prefix}.w{name}.{suffix}"));
            if !keys.iter().any(|key| weights.contains_key(key)) {
                continue;
            }
            ensure!(
                keys.iter().all(|key| weights.contains_key(key)),
                "incomplete Sapiens2 Q/K/V projection set at {prefix}.{suffix}"
            );
            let q = weights.remove(&keys[0]).expect("presence checked");
            let k = weights.remove(&keys[1]).expect("presence checked");
            let v = weights.remove(&keys[2]).expect("presence checked");
            let qk = mlxcel_core::concatenate(&q, &k, 0);
            weights.insert(fused, mlxcel_core::concatenate(&qk, &v, 0));
        }
    }
    Ok(())
}

pub fn array(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    let value = weights
        .get(key)
        .with_context(|| format!("Sapiens2 weight not found: {key}"))?;
    Ok(mlxcel_core::copy(value))
}

pub fn linear(weights: &WeightMap, prefix: &str) -> Result<Linear> {
    let weight = array(weights, &format!("{prefix}.weight"))?;
    let bias = weights
        .contains_key(&format!("{prefix}.bias"))
        .then(|| array(weights, &format!("{prefix}.bias")))
        .transpose()?;
    Ok(Linear::new(weight, bias))
}

pub fn rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm> {
    Ok(RMSNorm::new(
        array(weights, &format!("{prefix}.weight"))?,
        eps,
    ))
}

pub fn conv_weight(
    weights: &WeightMap,
    key: &str,
    out_channels: usize,
    kernel: usize,
    in_channels: usize,
) -> Result<UniquePtr<MlxArray>> {
    let weight = array(weights, key)?;
    let shape = mlxcel_core::array_shape(&weight);
    let target = [
        out_channels as i32,
        kernel as i32,
        kernel as i32,
        in_channels as i32,
    ];
    if shape == target {
        return Ok(weight);
    }
    let torch = [
        out_channels as i32,
        in_channels as i32,
        kernel as i32,
        kernel as i32,
    ];
    ensure!(
        shape == torch,
        "Sapiens2 convolution {key} has shape {shape:?}, expected {target:?} or PyTorch {torch:?}"
    );
    Ok(mlxcel_core::transpose_axes(&weight, &[0, 2, 3, 1]))
}

pub fn deconv_weight(
    weights: &WeightMap,
    key: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
) -> Result<UniquePtr<MlxArray>> {
    let weight = array(weights, key)?;
    let shape = mlxcel_core::array_shape(&weight);
    let target = [
        out_channels as i32,
        kernel as i32,
        kernel as i32,
        in_channels as i32,
    ];
    if shape == target {
        return Ok(weight);
    }
    let torch = [
        in_channels as i32,
        out_channels as i32,
        kernel as i32,
        kernel as i32,
    ];
    ensure!(
        shape == torch,
        "Sapiens2 transposed convolution {key} has shape {shape:?}, expected {target:?} or PyTorch {torch:?}"
    );
    Ok(mlxcel_core::transpose_axes(&weight, &[1, 2, 3, 0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::dense::sapiens2::config::Sapiens2Task;

    fn one_layer_config() -> Sapiens2Config {
        Sapiens2Config {
            task: Sapiens2Task::Backbone,
            hidden_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            kv_heads_per_layer: vec![1],
            intermediate_size: 4,
            image_size: (16, 16),
            patch_size: 16,
            num_channels: 3,
            num_register_tokens: 0,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
            normalize_backbone_outputs: true,
            head: None,
            num_labels: 0,
            keypoint_names: None,
            flip_pairs: Vec::new(),
        }
    }

    #[test]
    fn convolution_layout_guard_is_idempotent() {
        let mut weights = WeightMap::new();
        weights.insert(
            "head.weight".into(),
            mlxcel_core::from_slice_f32(&[0.0; 2 * 3 * 3 * 4], &[2, 3, 3, 4]),
        );
        let converted = conv_weight(&weights, "head.weight", 2, 3, 4).unwrap();
        assert_eq!(mlxcel_core::array_shape(&converted), vec![2, 3, 3, 4]);
    }

    #[test]
    fn deconvolution_converts_pytorch_layout() {
        let mut weights = WeightMap::new();
        weights.insert(
            "head.weight".into(),
            mlxcel_core::from_slice_f32(&[0.0; 4 * 2 * 3 * 3], &[4, 2, 3, 3]),
        );
        let converted = deconv_weight(&weights, "head.weight", 4, 2, 3).unwrap();
        assert_eq!(mlxcel_core::array_shape(&converted), vec![2, 3, 3, 4]);
    }

    #[test]
    fn floating_weights_follow_attention_dtype() {
        let mut weights = WeightMap::new();
        weights.insert(
            "backbone.blocks.0.attn.wqkv.weight".into(),
            mlxcel_core::astype(
                &mlxcel_core::from_slice_f32(&[1.0], &[1]),
                mlxcel_core::dtype::FLOAT16,
            ),
        );
        weights.insert(
            "backbone.blocks.0.ln1.weight".into(),
            mlxcel_core::from_slice_f32(&[1.0], &[1]),
        );
        normalize_floating_dtype(&mut weights).unwrap();
        assert_eq!(
            mlxcel_core::array_dtype(&weights["backbone.blocks.0.ln1.weight"]),
            mlxcel_core::dtype::FLOAT16
        );
    }

    #[test]
    fn legacy_qkv_planes_merge_in_query_key_value_order() {
        let mut weights = WeightMap::new();
        for (name, values) in [("q", [1.0, 2.0]), ("k", [3.0, 4.0]), ("v", [5.0, 6.0])] {
            weights.insert(
                format!("backbone.blocks.0.attn.w{name}.weight"),
                mlxcel_core::from_slice_f32(&values, &[2, 1]),
            );
        }
        merge_qkv(&mut weights, &one_layer_config()).unwrap();
        let fused = &weights["backbone.blocks.0.attn.wqkv.weight"];
        assert_eq!(mlxcel_core::array_shape(fused), [6, 1]);
        assert_eq!(
            mlxcel_core::utils::array_to_vec_f32(fused),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn bare_backbone_keys_gain_task_wrapper_prefix_once() {
        let mut weights = WeightMap::new();
        weights.insert(
            "blocks.0.ln1.weight".into(),
            mlxcel_core::from_slice_f32(&[1.0], &[1]),
        );
        prefix_bare_backbone(&mut weights);
        prefix_bare_backbone(&mut weights);
        assert!(weights.contains_key("backbone.blocks.0.ln1.weight"));
        assert_eq!(weights.len(), 1);
    }
}
