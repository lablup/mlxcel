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

//! Bounded host-side FFT primitives for audio feature extraction.
//!
//! The audio frontends all need real-input magnitude spectra, but their frame
//! sizes are not identical: Whisper uses a 400-point transform while Gemma 4,
//! Nemotron-Omni, and Gemma3n use power-of-two `n_fft` values. Zero-padding the
//! 400-point Whisper frame to 512 would change the retained bin frequencies, so
//! this helper computes the exact requested DFT grid with Bluestein's algorithm
//! for non-power-of-two lengths and the same radix-2 kernel for power-of-two
//! lengths.

use std::f64::consts::PI;

/// Largest FFT size accepted by the shared host audio helper.
///
/// Gemma 4's config guard permits a 65,536-sample frame and optional 2x FFT
/// overdrive, so 131,072 is the largest valid in-tree caller today. Larger
/// values are treated as malformed metadata and fail closed before allocating
/// transform scratch.
pub(crate) const MAX_REAL_FFT_LEN: usize = 1 << 17;
const MAX_REAL_FFT_BINS: usize = MAX_REAL_FFT_LEN / 2 + 1;

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    fn hypot(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// Compute real-input FFT magnitudes on the input's own DFT grid.
///
/// Returns exactly `num_bins` values for all valid in-tree callers. If the
/// caller asks for more positive-frequency bins than the input has, the extra
/// bins are zero-filled instead of panicking. Oversized requests return an empty
/// vector before allocating, so malformed metadata cannot drive an unbounded
/// preprocessing allocation.
pub(crate) fn real_fft_magnitude(input: &[f64], num_bins: usize) -> Vec<f64> {
    if num_bins == 0 {
        return Vec::new();
    }
    if num_bins > MAX_REAL_FFT_BINS {
        return Vec::new();
    }

    let n = input.len();
    let mut magnitudes = vec![0.0f64; num_bins];
    if n == 0 || n > MAX_REAL_FFT_LEN {
        return magnitudes;
    }

    let usable_bins = num_bins.min(n / 2 + 1);
    if n.is_power_of_two() {
        fill_radix2_real_magnitudes(input, usable_bins, &mut magnitudes);
    } else {
        fill_bluestein_real_magnitudes(input, usable_bins, &mut magnitudes);
    }
    magnitudes
}

fn fill_radix2_real_magnitudes(input: &[f64], usable_bins: usize, magnitudes: &mut [f64]) {
    let mut values: Vec<Complex> = input
        .iter()
        .map(|&sample| Complex::new(sample, 0.0))
        .collect();
    fft_radix2(&mut values, false);
    for (index, slot) in magnitudes.iter_mut().take(usable_bins).enumerate() {
        *slot = values[index].hypot();
    }
}

fn fill_bluestein_real_magnitudes(input: &[f64], usable_bins: usize, magnitudes: &mut [f64]) {
    let n = input.len();
    let convolution_len = (2 * n - 1).next_power_of_two();
    let mut signal = vec![Complex::default(); convolution_len];
    let mut chirp = vec![Complex::default(); convolution_len];

    for (index, &sample) in input.iter().enumerate() {
        let angle = chirp_angle(index, n);
        let forward_chirp = Complex::new(angle.cos(), -angle.sin());
        let inverse_chirp = Complex::new(angle.cos(), angle.sin());
        signal[index] = Complex::new(sample, 0.0).mul(forward_chirp);
        chirp[index] = inverse_chirp;
        if index != 0 {
            chirp[convolution_len - index] = inverse_chirp;
        }
    }

    fft_radix2(&mut signal, false);
    fft_radix2(&mut chirp, false);
    for (lhs, rhs) in signal.iter_mut().zip(chirp) {
        *lhs = lhs.mul(rhs);
    }
    fft_radix2(&mut signal, true);

    for (index, slot) in magnitudes.iter_mut().take(usable_bins).enumerate() {
        let angle = -chirp_angle(index, n);
        let correction = Complex::new(angle.cos(), angle.sin());
        *slot = signal[index].mul(correction).hypot();
    }
}

fn chirp_angle(index: usize, n: usize) -> f64 {
    PI * (index as f64) * (index as f64) / n as f64
}

fn fft_radix2(values: &mut [Complex], inverse: bool) {
    let n = values.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }

    let mut reversed = 0usize;
    for index in 1..n {
        let mut bit = n >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut width = 2usize;
    while width <= n {
        let half = width / 2;
        let angle = if inverse {
            2.0 * PI / width as f64
        } else {
            -2.0 * PI / width as f64
        };
        let step = Complex::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(width) {
            let mut twiddle = Complex::new(1.0, 0.0);
            for offset in 0..half {
                let left = start + offset;
                let right = left + half;
                let product = twiddle.mul(values[right]);
                let left_value = values[left];
                values[left] = Complex::new(left_value.re + product.re, left_value.im + product.im);
                values[right] =
                    Complex::new(left_value.re - product.re, left_value.im - product.im);
                twiddle = twiddle.mul(step);
            }
        }
        width <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for value in values {
            value.re *= scale;
            value.im *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft_magnitude(input: &[f64], num_bins: usize) -> Vec<f64> {
        let n = input.len();
        let mut magnitudes = vec![0.0; num_bins];
        if n == 0 {
            return magnitudes;
        }
        for (k, slot) in magnitudes
            .iter_mut()
            .enumerate()
            .take(num_bins.min(n / 2 + 1))
        {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (t, &sample) in input.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * t as f64 / n as f64;
                re += sample * angle.cos();
                im += sample * angle.sin();
            }
            *slot = re.hypot(im);
        }
        magnitudes
    }

    fn deterministic_frame(len: usize) -> Vec<f64> {
        (0..len)
            .map(|index| {
                let t = index as f64 / len as f64;
                0.3 * (2.0 * PI * 3.0 * t).sin()
                    + 0.2 * (2.0 * PI * 17.0 * t).cos()
                    + 0.05 * (2.0 * PI * 61.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn matches_dft_for_actual_audio_frame_sizes() {
        for len in [400, 512, 1024] {
            let input = deterministic_frame(len);
            let bins = len / 2 + 1;
            let expected = naive_dft_magnitude(&input, bins);
            let actual = real_fft_magnitude(&input, bins);
            assert_eq!(actual.len(), bins);
            for (bin, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 1e-8f64.max(expected.abs() * 1e-10);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "len={len} bin={bin} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn preserves_requested_cardinality_and_zero_fills_extra_bins() {
        let actual = real_fft_magnitude(&[1.0, -1.0, 1.0, -1.0], 5);
        assert_eq!(actual.len(), 5);
        assert_eq!(actual[3], 0.0);
        assert_eq!(actual[4], 0.0);
    }

    #[test]
    fn empty_input_returns_zero_magnitudes_with_requested_cardinality() {
        assert_eq!(real_fft_magnitude(&[], 3), vec![0.0, 0.0, 0.0]);
        assert!(real_fft_magnitude(&[], 0).is_empty());
    }

    #[test]
    fn oversized_input_fails_closed_without_transform_scratch() {
        let input = vec![1.0; MAX_REAL_FFT_LEN + 1];
        assert_eq!(real_fft_magnitude(&input, 3), vec![0.0, 0.0, 0.0]);
        assert!(real_fft_magnitude(&[1.0], MAX_REAL_FFT_BINS + 1).is_empty());
    }
}
