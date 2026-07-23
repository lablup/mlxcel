//! Small f32 Gemma3n oracle kernels used to isolate graph math in unit tests.
//!
//! These are deliberately dependency-free and operate on flattened row-major
//! buffers.  They are not the production executor; they pin formulas and edge
//! semantics so StableHLO changes can be compared one sublayer at a time.

pub(crate) fn per_layer_embedding_lookup(
    token_ids: &[i32],
    table: &[f32],
    per_layer_vocab: usize,
    layers: usize,
    hidden_per_layer: usize,
) -> Result<Vec<f32>, String> {
    let row = layers
        .checked_mul(hidden_per_layer)
        .ok_or("Gemma3n PLE row width overflows")?;
    if table.len() != per_layer_vocab.saturating_mul(row) {
        return Err(format!(
            "Gemma3n PLE table has {} elements; expected {}",
            table.len(),
            per_layer_vocab.saturating_mul(row)
        ));
    }
    let mut out = vec![0.0; token_ids.len().saturating_mul(row)];
    for (position, &token) in token_ids.iter().enumerate() {
        // The PLE vocabulary is intentionally smaller than the ordinary token
        // vocabulary.  Negative and high ids map to an exact all-zero row.
        if let Ok(index) = usize::try_from(token)
            && index < per_layer_vocab
        {
            out[position * row..(position + 1) * row]
                .copy_from_slice(&table[index * row..(index + 1) * row]);
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn project_dense_ple(
    merged_embeddings: &[f32],
    token_ple: &[f32],
    projection: &[f32],
    projection_norm: &[f32],
    eps: f32,
    positions: usize,
    hidden: usize,
    layers: usize,
    hidden_per_layer: usize,
) -> Result<Vec<f32>, String> {
    let ple_width = layers
        .checked_mul(hidden_per_layer)
        .ok_or("Gemma3n projected PLE width overflows")?;
    if merged_embeddings.len() != positions.saturating_mul(hidden)
        || token_ple.len() != positions.saturating_mul(ple_width)
        || projection.len() != ple_width.saturating_mul(hidden)
        || projection_norm.len() != hidden_per_layer
    {
        return Err("Gemma3n dense PLE projection shape mismatch".into());
    }
    let mut projected = vec![0.0; token_ple.len()];
    let model_scale = (hidden as f32).sqrt().recip();
    for p in 0..positions {
        for o in 0..ple_width {
            let mut value = 0.0;
            for i in 0..hidden {
                value += merged_embeddings[p * hidden + i] * projection[o * hidden + i];
            }
            projected[p * ple_width + o] = value * model_scale;
        }
        for layer in 0..layers {
            let start = p * ple_width + layer * hidden_per_layer;
            let end = start + hidden_per_layer;
            let normalized = rms_weighted(&projected[start..end], projection_norm, eps)?;
            projected[start..end].copy_from_slice(&normalized);
        }
    }
    Ok(projected
        .into_iter()
        .zip(token_ple)
        .map(|(projected, token)| (projected + token) * std::f32::consts::FRAC_1_SQRT_2)
        .collect())
}

pub(crate) fn normalize_magnitude(plane: &mut [f32], target_rms: f32) {
    if plane.is_empty() {
        return;
    }
    let rms = (plane.iter().map(|v| v * v).sum::<f32>() / plane.len() as f32).sqrt();
    let scale = target_rms / rms.max(1e-6);
    for value in plane {
        *value *= scale;
    }
}

fn matvec(input: &[f32], weight: &[f32], output: usize) -> Result<Vec<f32>, String> {
    if output == 0 || weight.len() != output.saturating_mul(input.len()) {
        return Err("Gemma3n projection shape mismatch".into());
    }
    Ok(weight
        .chunks_exact(input.len())
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect())
}

fn rms_weighted(input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, String> {
    if input.is_empty() || input.len() != weight.len() {
        return Err("Gemma3n RMS norm shape mismatch".into());
    }
    let inverse = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32 + eps)
        .sqrt()
        .recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * inverse * weight)
        .collect())
}

pub(crate) fn laurel_residual(
    normalized: &[f32],
    linear_left: &[f32],
    linear_right: &[f32],
    post_norm: &[f32],
    rank: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    let low_rank = matvec(normalized, linear_left, rank)?;
    let restored = matvec(&low_rank, linear_right, normalized.len())?;
    let restored = rms_weighted(&restored, post_norm, eps)?;
    Ok(normalized
        .iter()
        .zip(restored)
        .map(|(residual, update)| residual + update)
        .collect())
}

pub(crate) fn altup_predict(
    planes: &[Vec<f32>],
    coefficients: &[f32],
) -> Result<Vec<Vec<f32>>, String> {
    let n = planes.len();
    let hidden = planes.first().map_or(0, Vec::len);
    if n < 2 || planes.iter().any(|plane| plane.len() != hidden) || coefficients.len() != n * n {
        return Err("Gemma3n AltUp predict shape mismatch".into());
    }
    let mut predicted = planes.to_vec();
    for target in 0..n {
        for source in 0..n {
            let coefficient = coefficients[source * n + target];
            for feature in 0..hidden {
                predicted[target][feature] += coefficient * planes[source][feature];
            }
        }
    }
    Ok(predicted)
}

pub(crate) fn altup_correct(
    predicted: &[Vec<f32>],
    activated: &[f32],
    active_prediction: &[f32],
    correction: &[f32],
) -> Result<Vec<Vec<f32>>, String> {
    let n = predicted.len();
    let hidden = activated.len();
    if predicted.iter().any(|plane| plane.len() != hidden)
        || active_prediction.len() != hidden
        || correction.len() != n
    {
        return Err("Gemma3n AltUp correct shape mismatch".into());
    }
    let innovation: Vec<f32> = activated
        .iter()
        .zip(active_prediction)
        .map(|(actual, predicted)| actual - predicted)
        .collect();
    Ok(predicted
        .iter()
        .zip(correction)
        .map(|(plane, &coefficient)| {
            plane
                .iter()
                .zip(&innovation)
                // Reference adds one to the learned correction coefficient.
                .map(|(value, delta)| value + (coefficient + 1.0) * delta)
                .collect()
        })
        .collect())
}

pub(crate) fn gelu_approx(value: f32) -> f32 {
    let inner =
        (2.0f32 / std::f32::consts::PI).sqrt() * (value + 0.044_715 * value * value * value);
    0.5 * value * (1.0 + inner.tanh())
}

fn erf_approx(value: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26 is sufficient for this independent f32 oracle.
    let sign = value.signum();
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn gelu_erf(value: f32) -> f32 {
    0.5 * value * (1.0 + erf_approx(value / std::f32::consts::SQRT_2))
}

pub(crate) fn sparse_gelu(values: &[f32], std_multiplier: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let cutoff = mean + std_multiplier * variance.sqrt();
    values
        .iter()
        .map(|&value| {
            let shifted = (value - cutoff).max(0.0);
            gelu_erf(shifted)
        })
        .collect()
}

pub(crate) fn geglu_approx(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    if gate.len() != up.len() {
        return Err("Gemma3n GeGLU inputs must have the same length".into());
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| gelu_approx(gate) * up)
        .collect())
}

pub(crate) fn ple_injection(
    corrected_active: &[f32],
    dense_ple: &[f32],
    gate_projection: &[f32],
    output_projection: &[f32],
    post_norm: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    let gate = matvec(corrected_active, gate_projection, dense_ple.len())?;
    let activated = geglu_approx(&gate, dense_ple)?;
    let projected = matvec(&activated, output_projection, corrected_active.len())?;
    rms_weighted(&projected, post_norm, eps)
}

pub(crate) fn finish_layer_planes(
    corrected: &[Vec<f32>],
    active_index: usize,
    ple_injection: &[f32],
) -> Result<Vec<Vec<f32>>, String> {
    let hidden = corrected.first().map_or(0, Vec::len);
    if active_index >= corrected.len()
        || hidden == 0
        || ple_injection.len() != hidden
        || corrected.iter().any(|plane| plane.len() != hidden)
    {
        return Err("Gemma3n layer-boundary plane shape mismatch".into());
    }
    Ok(corrected
        .iter()
        .enumerate()
        .map(|(index, plane)| {
            if index == active_index {
                plane.clone()
            } else {
                plane
                    .iter()
                    .zip(ple_injection)
                    .map(|(value, injection)| value + injection)
                    .collect()
            }
        })
        .collect())
}

pub(crate) fn collapse_altup_planes(
    active: &[f32],
    other_planes: &[Vec<f32>],
    unembed_projections: &[Vec<f32>],
) -> Result<Vec<f32>, String> {
    if active.is_empty()
        || other_planes.len() != unembed_projections.len()
        || other_planes.iter().any(|plane| plane.len() != active.len())
    {
        return Err("Gemma3n AltUp collapse shape mismatch".into());
    }
    let target_rms =
        (active.iter().map(|value| value * value).sum::<f32>() / active.len() as f32).sqrt();
    let mut normalized_planes = vec![active.to_vec()];
    for (plane, projection) in other_planes.iter().zip(unembed_projections) {
        let mut projected = matvec(plane, projection, active.len())?;
        normalize_magnitude(&mut projected, target_rms);
        normalized_planes.push(projected);
    }
    mean_altup_planes_bf16(&normalized_planes)
}

pub(crate) fn mean_altup_planes_bf16(planes: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    let hidden = planes.first().map_or(0, Vec::len);
    if hidden == 0 || planes.iter().any(|plane| plane.len() != hidden) {
        return Err("Gemma3n AltUp mean plane shape mismatch".into());
    }
    let mut sum = vec![0.0f32; hidden];
    for plane in planes {
        for (sum, value) in sum.iter_mut().zip(plane) {
            *sum += value;
        }
    }
    let reciprocal = crate::weights::round_bf16_f32((planes.len() as f32).recip());
    Ok(sum
        .into_iter()
        .map(|value| {
            let value = crate::weights::round_bf16_f32(value);
            crate::weights::round_bf16_f32(value * reciprocal)
        })
        .collect())
}

pub(crate) fn softcap_logits(logits: &[f32], cap: f32) -> Result<Vec<f32>, String> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err("Gemma3n logit softcap must be finite and positive".into());
    }
    Ok(logits
        .iter()
        .map(|value| cap * (value / cap).tanh())
        .collect())
}

pub(crate) fn attention_bias(
    capacity: usize,
    real_len: usize,
    window: Option<usize>,
) -> Result<Vec<f32>, String> {
    if !(1..=capacity).contains(&real_len) {
        return Err(format!("real_len {real_len} is outside 1..={capacity}"));
    }
    let mut bias = vec![-1e30; capacity.saturating_mul(capacity)];
    for query in 0..capacity {
        let first = window.map_or(0, |width| (query + 1).saturating_sub(width));
        for key in first..=query {
            // Padded rows keep a deterministic causal mask. Real rows cannot
            // see padded keys because key <= query < real_len.
            if query >= real_len || key < real_len {
                bias[query * capacity + key] = 0.0;
            }
        }
    }
    Ok(bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_layer_vocab_overflow_is_exact_zero_not_row_zero() {
        let table = vec![
            3.0, 4.0, // token 0
            5.0, 6.0, // token 1
        ];
        let got = per_layer_embedding_lookup(&[0, 2, -1, 1], &table, 2, 1, 2).unwrap();
        assert_eq!(got, vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0]);
        assert_eq!(got[2].to_bits(), 0);
        assert_eq!(got[3].to_bits(), 0);
    }

    #[test]
    fn altup_prediction_and_correction_are_plane_isolated() {
        let planes = vec![vec![1.0, 2.0], vec![10.0, 20.0]];
        let predicted = altup_predict(&planes, &[0.0, 0.5, 0.25, 0.0]).unwrap();
        assert_eq!(predicted, vec![vec![3.5, 7.0], vec![10.5, 21.0]]);
        let corrected =
            altup_correct(&predicted, &[12.5, 23.0], &predicted[1], &[0.0, 1.0]).unwrap();
        assert_eq!(corrected[0], vec![5.5, 9.0]);
        assert_eq!(corrected[1], vec![14.5, 25.0]);
    }

    #[test]
    fn laurel_low_rank_residual_matches_hand_computation() {
        let got =
            laurel_residual(&[3.0, 4.0], &[2.0, 0.0], &[1.0, 2.0], &[1.0, 1.0], 1, 0.0).unwrap();
        let rms = ((6.0f32.powi(2) + 12.0f32.powi(2)) / 2.0).sqrt();
        assert!((got[0] - (3.0 + 6.0 / rms)).abs() < 1e-6);
        assert!((got[1] - (4.0 + 12.0 / rms)).abs() < 1e-6);
    }

    #[test]
    fn dense_ple_projection_and_geglu_match_hand_computation() {
        let projected = project_dense_ple(
            &[2.0, 4.0],
            &[1.0, 3.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 1.0],
            0.0,
            1,
            2,
            1,
            2,
        )
        .unwrap();
        let projected_rms = ((1.0f32 + 4.0) / 2.0).sqrt();
        assert!((projected[0] - (1.0 / projected_rms + 1.0) / 2.0f32.sqrt()).abs() < 1e-6);
        assert!((projected[1] - (2.0 / projected_rms + 3.0) / 2.0f32.sqrt()).abs() < 1e-6);
        assert_eq!(geglu_approx(&[0.0], &[7.0]).unwrap(), vec![0.0]);
    }

    #[test]
    fn full_and_sliding_masks_keep_padding_explicit() {
        let full = attention_bias(4, 2, None).unwrap();
        let sliding = attention_bias(4, 2, Some(2)).unwrap();
        assert_eq!(&full[4..8], &[0.0, 0.0, -1e30, -1e30]);
        assert_eq!(&sliding[12..16], &[-1e30, -1e30, 0.0, 0.0]);
        assert_eq!(full[2], -1e30);
    }

    #[test]
    fn magnitude_and_sparse_activation_are_stable_at_zero() {
        let mut zero = vec![0.0; 4];
        normalize_magnitude(&mut zero, 2.0);
        assert_eq!(zero, vec![0.0; 4]);
        let sparse = sparse_gelu(&[0.0, 1.0, 2.0], 0.0);
        assert_eq!(sparse[0], 0.0);
        assert_eq!(sparse[1], 0.0);
        assert!((sparse[2] - gelu_erf(1.0)).abs() < 1e-6);
    }

    #[test]
    fn ple_injection_collapse_and_softcap_pin_late_layer_math() {
        let injection = ple_injection(
            &[1.0, 2.0],
            &[3.0],
            &[1.0, 0.0],
            &[1.0, 2.0],
            &[1.0, 1.0],
            0.0,
        )
        .unwrap();
        assert!((injection[0] - 2.5f32.sqrt().recip()).abs() < 1e-6);
        assert!((injection[1] - 2.0 * 2.5f32.sqrt().recip()).abs() < 1e-6);

        let next = finish_layer_planes(&[vec![2.0, 3.0], vec![5.0, 7.0]], 0, &injection).unwrap();
        assert_eq!(next[0], vec![2.0, 3.0]);
        assert!((next[1][0] - (5.0 + injection[0])).abs() < 1e-6);
        assert!((next[1][1] - (7.0 + injection[1])).abs() < 1e-6);

        let collapsed =
            collapse_altup_planes(&[3.0, 4.0], &[vec![0.0, 5.0]], &[vec![1.0, 0.0, 0.0, 1.0]])
                .unwrap();
        assert!((collapsed[0] - 1.5).abs() < 1e-6);
        assert!((collapsed[1] - 4.5).abs() < 1e-6);

        let capped = softcap_logits(&[-60.0, 0.0, 60.0], 30.0).unwrap();
        assert_eq!(capped[1].to_bits(), 0);
        assert!(capped[0] < -28.0 && capped[0] > -30.0);
        assert!(capped[2] > 28.0 && capped[2] < 30.0);
    }

    #[test]
    fn final_altup_mean_reduces_in_f32_before_bf16_rounding() {
        let planes = vec![
            vec![1.0],
            vec![0.00390625],
            vec![0.00390625],
            vec![0.00390625],
        ];
        let collapsed = mean_altup_planes_bf16(&planes).unwrap();
        assert_eq!(collapsed, vec![0.25390625]);

        let pairwise_rounded = planes.iter().skip(1).fold(planes[0][0], |sum, plane| {
            crate::weights::round_bf16_f32(sum + plane[0])
        });
        let pairwise_mean = crate::weights::round_bf16_f32(pairwise_rounded * 0.25);
        assert_eq!(pairwise_mean, 0.25);
        assert_ne!(collapsed[0], pairwise_mean);
    }
}
