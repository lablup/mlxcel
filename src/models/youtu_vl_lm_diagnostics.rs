// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Diagnostics-only MLX/IREE seams for Youtu-VL's dense MLA decoder.

use mlxcel_core::{MlxArray, array_shape, array_to_raw_bytes, astype, dtype, eval};

use super::YoutuLanguageModel;

#[derive(Debug, Clone, PartialEq)]
pub struct YoutuMlaDiagnosticCapture {
    /// Final-position logits from a production MLX prefill.
    pub logits: Vec<f32>,
    /// Per-layer final-position compressed latent K rows.
    pub latent_kv: Vec<f32>,
    /// Per-layer final-position rotated positional K rows.
    pub rotary_kv: Vec<f32>,
    pub layers: usize,
    pub position: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct YoutuMlaParityReport {
    pub compared_logits: usize,
    pub compared_latent_kv: usize,
    pub compared_rotary_kv: usize,
    pub max_absolute: f32,
    pub max_relative: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct YoutuIreeMlaDiagnosticCapture {
    pub logits: Vec<f32>,
    pub kv: Vec<f32>,
    pub layers: usize,
    pub kv_width: usize,
}

#[cfg(feature = "xla-diagnostics")]
impl From<&mlxcel_xla::PreparedPrefillDiagnostics> for YoutuIreeMlaDiagnosticCapture {
    fn from(value: &mlxcel_xla::PreparedPrefillDiagnostics) -> Self {
        Self {
            logits: value.logits.clone(),
            kv: value.kv.clone(),
            layers: value.layers,
            kv_width: value.kv_width,
        }
    }
}

fn array_f32(array: &MlxArray) -> Vec<f32> {
    let array = astype(array, dtype::FLOAT32);
    eval(&array);
    array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect()
}

impl YoutuLanguageModel {
    /// Capture final-position logits and the native compressed MLA cache seam.
    ///
    /// A one-token prompt captures position 0; a longer prompt captures a
    /// nonzero position. `input_embeddings` may contain image-placeholder
    /// replacements produced by the ordinary Youtu-VL host path.
    pub fn capture_mla_diagnostics(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
    ) -> Result<YoutuMlaDiagnosticCapture, String> {
        let input_shape = array_shape(input_ids);
        if input_shape.len() != 2 || input_shape[0] != 1 || input_shape[1] <= 0 {
            return Err(format!(
                "Youtu-VL MLA diagnostics require input_ids [1, sequence], got {input_shape:?}"
            ));
        }
        let sequence = usize::try_from(input_shape[1])
            .map_err(|_| "Youtu-VL diagnostic sequence length does not fit usize".to_string())?;
        if let Some(embeddings) = input_embeddings {
            let shape = array_shape(embeddings);
            if shape != [1, input_shape[1], self.config.hidden_size as i32] {
                return Err(format!(
                    "Youtu-VL diagnostic embeddings must be [1, {sequence}, {}], got {shape:?}",
                    self.config.hidden_size
                ));
            }
        }
        let mut caches = self.make_caches_impl();
        let logits = self.forward_impl(input_ids, input_embeddings, &mut caches, None);
        let logits_shape = array_shape(&logits);
        let logits_values = array_f32(&logits);
        let vocab = self.config.vocab_size;
        if logits_shape != [1, input_shape[1], vocab as i32] {
            return Err(format!(
                "Youtu-VL diagnostic logits must be [1, {sequence}, {vocab}], got {logits_shape:?}"
            ));
        }
        let logits = logits_values[(sequence - 1) * vocab..sequence * vocab].to_vec();
        let mut latent_kv =
            Vec::with_capacity(self.config.num_hidden_layers * self.config.kv_lora_rank);
        let mut rotary_kv =
            Vec::with_capacity(self.config.num_hidden_layers * self.config.qk_rope_head_dim);
        for (layer, cache) in caches.iter().enumerate() {
            let keys = cache
                .keys
                .as_deref()
                .ok_or_else(|| format!("Youtu-VL diagnostic layer {layer} omitted latent K"))?;
            let values = cache
                .values
                .as_deref()
                .ok_or_else(|| format!("Youtu-VL diagnostic layer {layer} omitted rotary K"))?;
            let key_shape = array_shape(keys);
            let value_shape = array_shape(values);
            let valid_keys = key_shape.len() == 4
                && key_shape[0] == 1
                && key_shape[1] == 1
                && key_shape[2] >= input_shape[1]
                && key_shape[3] == self.config.kv_lora_rank as i32;
            let valid_values = value_shape.len() == 4
                && value_shape[0] == 1
                && value_shape[1] == 1
                && value_shape[2] >= input_shape[1]
                && value_shape[3] == self.config.qk_rope_head_dim as i32;
            if !valid_keys || !valid_values {
                return Err(format!(
                    "Youtu-VL diagnostic layer {layer} cache shape drifted: K={key_shape:?}, V={value_shape:?}"
                ));
            }
            let keys = array_f32(keys);
            let values = array_f32(values);
            let key_start = (sequence - 1) * self.config.kv_lora_rank;
            let value_start = (sequence - 1) * self.config.qk_rope_head_dim;
            latent_kv.extend_from_slice(&keys[key_start..key_start + self.config.kv_lora_rank]);
            rotary_kv.extend_from_slice(
                &values[value_start..value_start + self.config.qk_rope_head_dim],
            );
        }
        Ok(YoutuMlaDiagnosticCapture {
            logits,
            latent_kv,
            rotary_kv,
            layers: self.config.num_hidden_layers,
            position: sequence - 1,
            kv_lora_rank: self.config.kv_lora_rank,
            qk_rope_head_dim: self.config.qk_rope_head_dim,
        })
    }
}

fn compare_slice(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    maxima: &mut (f32, f32),
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} length differs: MLX={} IREE={}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (&mlx, &iree)) in actual.iter().zip(expected).enumerate() {
        if !mlx.is_finite() || !iree.is_finite() {
            return Err(format!(
                "{label} contains non-finite value at {index}: MLX={mlx}, IREE={iree}"
            ));
        }
        let absolute = (mlx - iree).abs();
        let relative = absolute / iree.abs().max(f32::MIN_POSITIVE);
        maxima.0 = maxima.0.max(absolute);
        maxima.1 = maxima.1.max(relative);
        if absolute > atol + rtol * iree.abs() {
            return Err(format!(
                "{label} differs at {index}: MLX={mlx}, IREE={iree}, absolute={absolute}, tolerance={}",
                atol + rtol * iree.abs()
            ));
        }
    }
    Ok(())
}

/// Compare MLX's native `(latent, rotary)` cache against the padded XLA MLA
/// cache layout `[latent, zero-rope]` for K and `[zero-latent, rotary]` for V.
pub fn compare_youtu_mla_diagnostics(
    mlx: &YoutuMlaDiagnosticCapture,
    iree: &YoutuIreeMlaDiagnosticCapture,
    atol: f32,
    rtol: f32,
) -> Result<YoutuMlaParityReport, String> {
    if atol < 0.0 || rtol < 0.0 || !atol.is_finite() || !rtol.is_finite() {
        return Err(format!(
            "Youtu-VL diagnostic tolerances must be finite and non-negative, got atol={atol}, rtol={rtol}"
        ));
    }
    if mlx.layers != iree.layers {
        return Err(format!(
            "Youtu-VL diagnostic layer count differs: MLX={} IREE={}",
            mlx.layers, iree.layers
        ));
    }
    let cache_width = mlx.kv_lora_rank + mlx.qk_rope_head_dim;
    if iree.kv_width != cache_width {
        return Err(format!(
            "Youtu-VL IREE diagnostic KV width must be latent+rotary={cache_width}, got {}",
            iree.kv_width
        ));
    }
    let expected_iree_kv = iree
        .layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(cache_width))
        .ok_or_else(|| "Youtu-VL diagnostic IREE KV length overflowed".to_string())?;
    if iree.kv.len() != expected_iree_kv {
        return Err(format!(
            "Youtu-VL diagnostic IREE KV length differs: got {}, expected {expected_iree_kv}",
            iree.kv.len()
        ));
    }
    let mut iree_latent = Vec::with_capacity(mlx.latent_kv.len());
    let mut iree_rotary = Vec::with_capacity(mlx.rotary_kv.len());
    for layer in 0..iree.layers {
        let layer_start = layer * 2 * cache_width;
        let key = &iree.kv[layer_start..layer_start + cache_width];
        let value = &iree.kv[layer_start + cache_width..layer_start + 2 * cache_width];
        if key[mlx.kv_lora_rank..]
            .iter()
            .chain(value[..mlx.kv_lora_rank].iter())
            .any(|value| *value != 0.0)
        {
            return Err(format!(
                "Youtu-VL IREE diagnostic layer {layer} has nonzero MLA cache padding"
            ));
        }
        iree_latent.extend_from_slice(&key[..mlx.kv_lora_rank]);
        iree_rotary.extend_from_slice(&value[mlx.kv_lora_rank..]);
    }
    let mut maxima = (0.0f32, 0.0f32);
    compare_slice(
        "Youtu-VL final-position logits",
        &mlx.logits,
        &iree.logits,
        atol,
        rtol,
        &mut maxima,
    )?;
    compare_slice(
        "Youtu-VL compressed latent K",
        &mlx.latent_kv,
        &iree_latent,
        atol,
        rtol,
        &mut maxima,
    )?;
    compare_slice(
        "Youtu-VL rotary K",
        &mlx.rotary_kv,
        &iree_rotary,
        atol,
        rtol,
        &mut maxima,
    )?;
    Ok(YoutuMlaParityReport {
        compared_logits: mlx.logits.len(),
        compared_latent_kv: mlx.latent_kv.len(),
        compared_rotary_kv: mlx.rotary_kv.len(),
        max_absolute: maxima.0,
        max_relative: maxima.1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_maps_padded_iree_mla_cache_to_native_mlx_seams() {
        let mlx = YoutuMlaDiagnosticCapture {
            logits: vec![0.25, -0.5],
            latent_kv: vec![1.0, 2.0, 3.0, 4.0],
            rotary_kv: vec![5.0, 6.0, 7.0, 8.0],
            layers: 2,
            position: 1,
            kv_lora_rank: 2,
            qk_rope_head_dim: 2,
        };
        let iree = YoutuIreeMlaDiagnosticCapture {
            logits: mlx.logits.clone(),
            kv: vec![
                1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0,
            ],
            layers: 2,
            kv_width: 4,
        };
        let report = compare_youtu_mla_diagnostics(&mlx, &iree, 0.0, 0.0).unwrap();
        assert_eq!(report.compared_logits, 2);
        assert_eq!(report.compared_latent_kv, 4);
        assert_eq!(report.compared_rotary_kv, 4);
    }
}
