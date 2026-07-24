use std::f64::consts::PI;

/// `scipy.signal.resample_poly(x, 1, down)` with its default Kaiser-5 FIR.
///
/// Phi-4 MM's pinned processor only requests integer downsampling. This
/// backend-neutral primitive is shared by the owned waveform boundary and the
/// legacy feature extractor so both paths stay bit-identical.
pub(crate) fn scipy_resample_poly_down(samples: &[f32], down: usize) -> Vec<f32> {
    if down <= 1 {
        return samples.to_vec();
    }

    let half_len = 10 * down;
    let filter_len = 2 * half_len + 1;
    let cutoff = 1.0 / down as f64;
    let beta = 5.0;
    let i0_beta = bessel_i0(beta);
    let mut filter = Vec::with_capacity(filter_len);
    for index in 0..filter_len {
        let offset = index as isize - half_len as isize;
        let phase = cutoff * offset as f64;
        let sinc = if offset == 0 {
            1.0
        } else {
            (PI * phase).sin() / (PI * phase)
        };
        let ratio = 2.0 * index as f64 / (filter_len - 1) as f64 - 1.0;
        let window = bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / i0_beta;
        filter.push(cutoff * sinc * window);
    }
    let scale: f64 = filter.iter().sum();
    for value in &mut filter {
        *value /= scale;
    }

    // SciPy prepends `down - half_len % down` zeros, then removes the first
    // `(half_len + pre_pad) / down` polyphase outputs to center sample zero.
    let pre_pad = down - half_len % down;
    let pre_remove = (half_len + pre_pad) / down;
    let output_len = samples.len().div_ceil(down);
    let padded_filter_len = pre_pad + filter.len();
    let mut output = Vec::with_capacity(output_len);
    for output_index in 0..output_len {
        let raw_index = (pre_remove + output_index) * down;
        let first_sample = raw_index.saturating_sub(padded_filter_len - 1);
        let last_sample = raw_index.min(samples.len().saturating_sub(1));
        let mut sum = 0.0f64;
        if first_sample <= last_sample {
            for (sample_index, sample) in samples
                .iter()
                .enumerate()
                .take(last_sample + 1)
                .skip(first_sample)
            {
                let padded_filter_index = raw_index - sample_index;
                if padded_filter_index >= pre_pad {
                    let filter_index = padded_filter_index - pre_pad;
                    if filter_index < filter.len() {
                        sum += *sample as f64 * filter[filter_index];
                    }
                }
            }
        }
        output.push(sum as f32);
    }
    output
}

fn bessel_i0(value: f64) -> f64 {
    let x = value.abs();
    if x < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / x;
        (x.exp() / x.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_polyphase_matches_pinned_scipy_reference() {
        // Expected prefixes were generated independently with
        // scipy.signal.resample_poly(x, 1, down), SciPy 1.15.2. These are the
        // pre-move Phi frontend goldens, not outputs captured from this helper.
        let samples: Vec<f32> = (0..64)
            .map(|index| (0.25 * (2.0 * PI * 440.0 * index as f64 / 48_000.0).sin()) as f32)
            .collect();
        for (down, expected) in [
            (
                2,
                [0.002_246_874_2, 0.028_348_744, 0.057_180_017, 0.084_596_574],
            ),
            (
                3,
                [0.003_921_834_3, 0.042_409_703, 0.084_868_06, 0.123_779_45],
            ),
        ] {
            let actual = scipy_resample_poly_down(&samples, down);
            assert_eq!(actual.len(), samples.len().div_ceil(down));
            for (index, expected) in expected.into_iter().enumerate() {
                assert!(
                    (actual[index] - expected).abs() < 2e-6,
                    "down={down} sample[{index}]={} expected {expected}",
                    actual[index]
                );
            }
        }
    }
}
