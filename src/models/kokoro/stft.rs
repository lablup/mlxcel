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

//! STFT / inverse-STFT for the iSTFTNet head, plus the NSF harmonic source.
//!
//! The STFT framing and overlap-add are done on the host (`Vec<f32>`) because
//! the transforms are tiny (`n_fft = 20`, `hop = 5`) and the windowing /
//! overlap-add logic is simpler and less error-prone in plain Rust than as an
//! MLX graph; the per-frame `rfft`/`irfft` are still MLX ops. The inverse uses
//! the PyTorch-reference convention: `audio = istft(magnitude * exp(j*phase))`
//! with a periodic Hann window and no phase unwrap.
//!
//! Window note: a periodic Hann window of length `N` is `0.5 - 0.5*cos(2*pi*n/N)`
//! (not `N-1` in the denominator), matching `torch.hann_window(N, periodic=True)`.

use mlxcel_core::{MlxArray, UniquePtr};

use super::ops;

/// Sample rate of Kokoro audio output.
pub(crate) const SAMPLE_RATE: u32 = 24_000;

const N_FFT: usize = 20;
const HOP: usize = 5;
const WIN: usize = 20;

/// Periodic Hann window of length `n`.
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflect-pad a signal by `pad` samples on each side (NumPy `mode='reflect'`:
/// the edge sample is not duplicated).
fn reflect_pad(y: &[f32], pad: usize) -> Vec<f32> {
    if y.is_empty() {
        return vec![0.0; 2 * pad];
    }
    let n = y.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        // reflect: index pad-i maps to source i+1
        let src = (pad - i).min(n - 1);
        out.push(y[src]);
    }
    out.extend_from_slice(y);
    for i in 0..pad {
        let src = n.saturating_sub(2 + i);
        out.push(y[src]);
    }
    out
}

/// Forward STFT of a host signal, returning `(magnitude, phase)` each as a
/// `(F, n_frames)` MLX array where `F = n_fft/2 + 1`.
///
/// `center = true`: the signal is reflect-padded by `n_fft/2` before framing.
pub(crate) fn stft(y: &[f32]) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let window = hann_periodic(WIN);
    let pad = N_FFT / 2;
    let yp = reflect_pad(y, pad);
    let n_frames = if yp.len() >= N_FFT {
        1 + (yp.len() - N_FFT) / HOP
    } else {
        0
    };
    let bins = N_FFT / 2 + 1;

    // Build the windowed frame matrix (n_frames, n_fft) on the host.
    let mut frames = vec![0.0_f32; n_frames * N_FFT];
    for f in 0..n_frames {
        let start = f * HOP;
        for k in 0..N_FFT {
            frames[f * N_FFT + k] = yp[start + k] * window[k];
        }
    }
    let frames_arr = mlxcel_core::from_slice_f32(&frames, &[n_frames as i32, N_FFT as i32]);
    // rfft along axis 1 -> complex (n_frames, bins).
    let spec = mlxcel_core::rfft(ops::r(&frames_arr), N_FFT as i32, 1);
    let re = mlxcel_core::real_part(ops::r(&spec));
    let im = mlxcel_core::imag_part(ops::r(&spec));
    // magnitude = sqrt(re^2 + im^2); phase = atan2(im, re).
    let mag = mlxcel_core::sqrt(ops::r(&ops::add(&ops::mul(&re, &re), &ops::mul(&im, &im))));
    let phase = mlxcel_core::arctan2(ops::r(&im), ops::r(&re));
    // Transpose to (bins, n_frames).
    let mag = ops::swap_axes(&mag, 0, 1);
    let phase = ops::swap_axes(&phase, 0, 1);
    let _ = bins;
    (mag, phase)
}

/// Inverse STFT from magnitude and phase `(F, n_frames)` MLX arrays, returning
/// the host waveform. Per frame: form the half-spectrum `mag * exp(j*phase)`,
/// take the real inverse DFT, window, overlap-add, divide by the window-power
/// sum, and trim `n_fft/2` from each end (the `center=true` padding).
///
/// The inverse rDFT is computed on the host with an explicit Hermitian formula
/// (bins `1..N/2` count twice). `n_fft` is tiny (20), so this is cheap and
/// matches NumPy's `irfft` normalization exactly, without depending on MLX's
/// complex64 byte layout for array construction.
pub(crate) fn istft(
    magnitude: &UniquePtr<MlxArray>,
    phase: &UniquePtr<MlxArray>,
) -> Result<Vec<f32>, String> {
    let window = hann_periodic(WIN);
    let shape = ops::shape(magnitude);
    let bins = shape[0] as usize; // n_fft/2 + 1
    let n_frames = shape[1] as usize;

    let mag = ops::to_vec_f32(magnitude)?; // (bins, n_frames) row-major
    let pha = ops::to_vec_f32(phase)?;

    // Inverse rDFT per frame on the host.
    let basis = irdft_basis(N_FFT, bins);
    let mut frames = vec![0.0_f32; n_frames * N_FFT];
    for f in 0..n_frames {
        // Gather this frame's re/im (column f of the (bins, n_frames) layout).
        for n in 0..N_FFT {
            let mut acc = 0.0_f32;
            for k in 0..bins {
                let m = mag[k * n_frames + f];
                let p = pha[k * n_frames + f];
                let re = m * p.cos();
                let im = m * p.sin();
                let (cw, sw) = basis[n * bins + k];
                // x[n] += weight_k * (re*cos - im*sin)
                acc += re * cw - im * sw;
            }
            frames[f * N_FFT + n] = acc;
        }
    }

    // Windowed overlap-add.
    let expected = N_FFT + HOP * n_frames.saturating_sub(1);
    let mut acc = vec![0.0_f32; expected];
    let mut wsum = vec![0.0_f32; expected];
    for f in 0..n_frames {
        let start = f * HOP;
        for k in 0..N_FFT {
            acc[start + k] += frames[f * N_FFT + k] * window[k];
            wsum[start + k] += window[k] * window[k];
        }
    }
    for i in 0..expected {
        if wsum[i] > 1e-8 {
            acc[i] /= wsum[i];
        }
    }
    // Trim center padding.
    let pad = N_FFT / 2;
    let end = expected.saturating_sub(pad);
    let trimmed = if end > pad {
        acc[pad..end].to_vec()
    } else {
        Vec::new()
    };
    Ok(trimmed)
}

/// Precompute the inverse real-DFT basis weights `(cos_w, sin_w)` for each
/// `(time index n, bin k)`, including the `1/N` normalization and the Hermitian
/// doubling of bins `1..N/2` (every bin except DC and Nyquist counts twice).
///
/// `x[n] = (1/N) * sum_k d_k * (re[k]*cos(2*pi*k*n/N) - im[k]*sin(2*pi*k*n/N))`
/// where `d_k = 1` for DC/Nyquist and `2` otherwise. The returned weights fold
/// `d_k / N` into the cos/sin terms.
fn irdft_basis(n_fft: usize, bins: usize) -> Vec<(f32, f32)> {
    let nyquist = n_fft.is_multiple_of(2) && bins == n_fft / 2 + 1;
    let mut basis = vec![(0.0_f32, 0.0_f32); n_fft * bins];
    let inv_n = 1.0 / n_fft as f32;
    for n in 0..n_fft {
        for k in 0..bins {
            let dk = if k == 0 || (nyquist && k == bins - 1) {
                1.0
            } else {
                2.0
            };
            let ang = 2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / (n_fft as f32);
            let w = dk * inv_n;
            basis[n * bins + k] = (w * ang.cos(), w * ang.sin());
        }
    }
    basis
}

/// NSF harmonic source.
///
/// Given an F0 curve upsampled to the audio rate, generate a harmonic sine
/// excitation (`harmonic_num + 1` partials), mix to a single source via the
/// learned `l_linear` + tanh, and return the host signal. Deterministic: the
/// initial phase is zero and no additive noise is injected, so the output is
/// reproducible (the upstream model seeds random phase, which only perturbs
/// phase, not the harmonic structure).
pub(crate) struct NsfSource {
    linear_w: UniquePtr<MlxArray>,
    linear_b: UniquePtr<MlxArray>,
    harmonic_num: usize,
}

impl NsfSource {
    pub(crate) fn new(
        linear_w: UniquePtr<MlxArray>,
        linear_b: UniquePtr<MlxArray>,
        harmonic_num: usize,
    ) -> Self {
        Self {
            linear_w,
            linear_b,
            harmonic_num,
        }
    }

    /// Produce the harmonic source signal for `f0` (host, audio rate).
    pub(crate) fn forward(&self, f0: &[f32]) -> Result<Vec<f32>, String> {
        let t = f0.len();
        let n_harm = self.harmonic_num + 1;
        let sr = SAMPLE_RATE as f32;
        // sine_waves[t, h] = sin( cumsum_t( ((h+1)*f0/sr) mod 1 ) * 2pi ) * 0.1 * uv
        let mut sines = vec![0.0_f32; t * n_harm];
        let mut phase_acc = vec![0.0_f32; n_harm];
        for ti in 0..t {
            let voiced = if f0[ti] > 10.0 { 1.0 } else { 0.0 };
            for h in 0..n_harm {
                let rad = ((h as f32 + 1.0) * f0[ti] / sr).rem_euclid(1.0);
                phase_acc[h] += rad;
                let s = (phase_acc[h] * 2.0 * std::f32::consts::PI).sin() * 0.1 * voiced;
                sines[ti * n_harm + h] = s;
            }
        }
        let sines_arr = mlxcel_core::from_slice_f32(&sines, &[t as i32, n_harm as i32]);
        // merged = tanh(linear(sines)) -> (T, 1)
        let merged = ops::tanh(&ops::linear(
            &sines_arr,
            &self.linear_w,
            Some(&self.linear_b),
        ));
        let v = ops::to_vec_f32(&merged)?;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_stream<R>(f: impl FnOnce() -> R) -> R {
        let stream = mlxcel_core::streams::new_thread_local_generation_stream();
        mlxcel_core::streams::install_thread_local_default_stream(stream.as_ref());
        f()
    }

    /// A forward STFT followed by an inverse STFT must reconstruct the input
    /// (within windowing edge effects) at unit scale. This pins the inverse-DFT
    /// normalization so an amplitude regression in the vocoder head is caught
    /// here rather than only in full synthesis.
    #[test]
    fn stft_istft_round_trip_is_unit_scale() {
        with_stream(|| {
            let n = 400usize;
            let signal: Vec<f32> = (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * 5.0 * i as f32 / n as f32).sin() * 0.3)
                .collect();
            let (mag, phase) = stft(&signal);
            let recon = istft(&mag, &phase).expect("istft");
            assert!(!recon.is_empty(), "reconstruction is non-empty");
            // Compare the stable interior (skip the first/last few hops where the
            // window normalization is incomplete).
            let cmp = 40;
            let len = recon.len().min(signal.len());
            let mut max_err = 0.0_f32;
            for i in cmp..len.saturating_sub(cmp) {
                max_err = max_err.max((recon[i] - signal[i]).abs());
            }
            assert!(
                max_err < 0.05,
                "STFT round-trip should reconstruct at unit scale, max interior error {max_err}"
            );
        });
    }
}
