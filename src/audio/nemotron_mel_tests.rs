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

use super::*;

#[test]
fn defaults_match_checkpoint_preprocessor() {
    let args = MelArgs::default();
    assert_eq!(args.win_length(), 400);
    assert_eq!(args.hop_length(), 160);
    assert_eq!(args.features, 128);
    assert_eq!(args.n_fft, 512);
    let parsed: MelArgs = serde_json::from_str(
        r#"{"dither":1e-05,"features":128,"frame_splicing":1,"log":true,"n_fft":512,
            "normalize":"NA","pad_to":0,"pad_value":0.0,"sample_rate":16000,
            "window":"hann","window_size":0.025,"window_stride":0.01}"#,
    )
    .unwrap();
    assert_eq!(parsed, args);
}

#[test]
fn frame_count_is_one_plus_floor_len_over_hop() {
    let args = MelArgs::default();
    for len in [300usize, 1600, 1759, 16_000, 29_017] {
        let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
        let (feats, frames) = log_mel_spectrogram(&x, &args).unwrap();
        assert_eq!(frames, 1 + len / 160, "len {len}");
        assert_eq!(frames, args.num_frames(len));
        assert_eq!(feats.len(), frames * 128);
        assert!(feats.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn silence_gives_log_guard_floor() {
    let args = MelArgs::default();
    let (feats, frames) = log_mel_spectrogram(&vec![0.0f32; 3200], &args).unwrap();
    assert_eq!(frames, 21);
    let floor = (2f32.powi(-24)).ln();
    assert!((floor - (-16.635_532)).abs() < 1e-5);
    assert!(
        feats.iter().all(|&v| v == floor),
        "silence must hit the guard floor"
    );
}

#[test]
fn preemphasis_first_sample_unchanged() {
    let x = [0.5f32, 1.0, -0.25, 0.0];
    let y = preemphasize(&x, 0.97);
    assert_eq!(y[0], 0.5);
    assert!((y[1] - (1.0 - 0.97 * 0.5)).abs() < 1e-7);
    assert!((y[2] - (-0.25 - 0.97)).abs() < 1e-7);
    assert!((y[3] - 0.97 * 0.25).abs() < 1e-7);
    assert_eq!(preemphasize(&x, 0.0), x.to_vec());
}

#[test]
fn window_is_symmetric_hann_centered_in_n_fft() {
    let w = centered_window("hann", 400, 512).unwrap();
    assert_eq!(w.len(), 512);
    assert!(w[..56].iter().all(|&v| v == 0.0));
    assert!(w[456..].iter().all(|&v| v == 0.0));
    // Symmetric: both ends of the 400-sample window are exactly zero and it
    // mirrors around its center (a periodic window would not).
    assert_eq!(w[56], 0.0);
    assert!(w[455].abs() < 1e-7);
    for i in 0..200 {
        assert!((w[56 + i] - w[455 - i]).abs() < 1e-6);
    }
    assert!(centered_window("kaiser", 400, 512).is_err());
}

#[test]
fn slaney_filterbank_rows_are_nonnegative_triangles() {
    let bank = slaney_filterbank(16_000, 512, 128);
    assert_eq!(bank.len(), 128 * 257);
    for mel in 0..128 {
        let row = &bank[mel * 257..(mel + 1) * 257];
        assert!(row.iter().all(|&v| v >= 0.0));
        assert!(row.iter().any(|&v| v > 0.0), "mel {mel} is empty");
    }
}

#[test]
fn tone_energy_lands_in_matching_mel_bin() {
    let args = MelArgs::default();
    let x: Vec<f32> = (0..8000)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin())
        .collect();
    let (feats, frames) = log_mel_spectrogram(&x, &args).unwrap();
    let mid = &feats[(frames / 2) * 128..(frames / 2 + 1) * 128];
    let peak = mid
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    // 1 kHz sits at Slaney mel 15; with 128 bins over 0..8 kHz that is bin ~41.
    assert!((38..=44).contains(&peak), "peak bin {peak}");
}

#[test]
fn per_feature_normalization_zero_mean() {
    let args = MelArgs {
        normalize: "per_feature".to_string(),
        ..MelArgs::default()
    };
    let x: Vec<f32> = (0..4000)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
        .collect();
    let (feats, frames) = log_mel_spectrogram(&x, &args).unwrap();
    for f in [0usize, 64, 127] {
        let mean: f32 = (0..frames).map(|t| feats[t * 128 + f]).sum::<f32>() / frames as f32;
        assert!(mean.abs() < 1e-4, "feature {f} mean {mean}");
    }
    let bad = MelArgs {
        normalize: "bogus".to_string(),
        ..MelArgs::default()
    };
    assert!(log_mel_spectrogram(&x, &bad).is_err());
}

#[test]
fn log_mel_array_shape() {
    let args = MelArgs::default();
    let arr = log_mel_array(&vec![0.1f32; 1600], &args).unwrap();
    assert_eq!(mlxcel_core::array_shape(&arr), vec![1, 11, 128]);
    assert!(log_mel_spectrogram(&[], &args).is_err());
}
