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

//! Mel spectrogram feature extraction for Gemma4 audio.
//!
//! Extracts log-mel spectrograms from raw audio waveforms using the USM
//! (Universal Speech Model) preprocessing pipeline. Pure DSP -- no neural
//! network weights involved.
//!
//! Ported from: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/audio_feature_extractor.py
//!
//! Used by: Gemma4 VLM (audio modality)

use std::f64::consts::PI;

/// Mel filter bank matrix: `[num_frequency_bins, num_mel_filters]`.
fn mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    min_frequency: f64,
    max_frequency: f64,
    sampling_rate: u32,
) -> Vec<f32> {
    fn hz_to_mel(freq: f64) -> f64 {
        2595.0 * (1.0 + freq / 700.0).log10()
    }
    fn mel_to_hz(mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
    }

    let mel_min = hz_to_mel(min_frequency);
    let mel_max = hz_to_mel(max_frequency);
    let mel_points: Vec<f64> = (0..=(num_mel_filters + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (num_mel_filters + 1) as f64)
        .collect();
    let freq_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let all_freqs: Vec<f64> = (0..num_frequency_bins)
        .map(|i| i as f64 * (sampling_rate as f64 / (2.0 * (num_frequency_bins as f64 - 1.0))))
        .collect();

    let mut bank = vec![0.0f32; num_frequency_bins * num_mel_filters];
    for i in 0..num_mel_filters {
        let lower = freq_points[i];
        let center = freq_points[i + 1];
        let upper = freq_points[i + 2];
        for j in 0..num_frequency_bins {
            let freq = all_freqs[j];
            let rising = (freq - lower) / (center - lower).max(1e-10);
            let falling = (upper - freq) / (upper - center).max(1e-10);
            bank[j * num_mel_filters + i] = rising.min(falling).max(0.0) as f32;
        }
    }
    bank
}

/// Audio feature extractor configuration.
///
/// Field names and defaults mirror the reference `Gemma4AudioFeatureExtractor`
/// (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/audio_feature_extractor.py),
/// so a `feature_extractor` block from a checkpoint's `processor_config.json`
/// deserializes directly. Any field absent from the block keeps the reference
/// default.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct AudioFeatureExtractorConfig {
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub padding_value: f32,
    pub frame_length_ms: f64,
    pub hop_length_ms: f64,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub preemphasis: f64,
    pub preemphasis_htk_flavor: bool,
    pub fft_overdrive: bool,
    pub input_scale_factor: f64,
    pub mel_floor: f64,
    pub per_bin_mean: Option<Vec<f64>>,
    pub per_bin_stddev: Option<Vec<f64>>,
}

impl Default for AudioFeatureExtractorConfig {
    fn default() -> Self {
        Self {
            feature_size: 128,
            sampling_rate: 16_000,
            padding_value: 0.0,
            frame_length_ms: 20.0,
            hop_length_ms: 10.0,
            min_frequency: 0.0,
            max_frequency: 8000.0,
            preemphasis: 0.0,
            preemphasis_htk_flavor: true,
            // The published Gemma 4 E-series checkpoints ship no
            // `feature_extractor` block, so this default IS the effective
            // config. The reference extractor defaults to `false`
            // (fft_length 512 for the 20 ms / 320-sample frame); `true`
            // (fft_length 1024) shifts every log-mel value by ~ln 2 and
            // changes the fine spectral structure, which garbles the
            // Conformer's perception (issue #782).
            fft_overdrive: false,
            input_scale_factor: 1.0,
            mel_floor: 1e-3,
            per_bin_mean: None,
            per_bin_stddev: None,
        }
    }
}

impl AudioFeatureExtractorConfig {
    /// Build the config from a checkpoint's `processor_config.json` value.
    ///
    /// Reads the optional `feature_extractor` block; missing file, missing
    /// block, or a malformed block all fall back to the reference defaults so
    /// existing checkpoints keep working.
    pub fn from_processor_config(processor_config: Option<&serde_json::Value>) -> Self {
        processor_config
            .and_then(|config| config.get("feature_extractor"))
            .and_then(|block| serde_json::from_value(block.clone()).ok())
            .unwrap_or_default()
    }
}

/// Audio feature extractor for Gemma4.
///
/// Converts raw waveform to log-mel spectrogram features.
pub struct AudioFeatureExtractor {
    feature_size: usize,
    sampling_rate: u32,
    padding_value: f32,
    preemphasis: f64,
    preemphasis_htk_flavor: bool,
    input_scale_factor: f64,
    mel_floor: f64,
    frame_length: usize,
    hop_length: usize,
    fft_length: usize,
    window: Vec<f32>,
    mel_filters: Vec<f32>, // [fft_length/2 + 1, feature_size]
    per_bin_mean: Option<Vec<f64>>,
    per_bin_stddev: Option<Vec<f64>>,
}

impl AudioFeatureExtractor {
    pub fn new(config: AudioFeatureExtractorConfig) -> Self {
        // A checkpoint's `processor_config.json` `feature_extractor` block is
        // semi-trusted input read at load. Guard the scalar fields so a
        // malformed or hostile block cannot drive a divide-by-zero in
        // `extract` (hop_length == 0), an unbounded mel/feature allocation
        // (feature_size), or a runaway `fft_length` doubling loop (frame_length).
        // Out-of-range values fall back to the reference defaults, mirroring the
        // per-bin-vector validation below.
        let defaults = AudioFeatureExtractorConfig::default();
        let mut config = config;
        if config.sampling_rate == 0 {
            config.sampling_rate = defaults.sampling_rate;
        }
        if !config.frame_length_ms.is_finite() || config.frame_length_ms <= 0.0 {
            config.frame_length_ms = defaults.frame_length_ms;
        }
        if !config.hop_length_ms.is_finite() || config.hop_length_ms <= 0.0 {
            config.hop_length_ms = defaults.hop_length_ms;
        }
        // 4096 mel bins is far above any real value (128) yet keeps the
        // `[fft_length/2+1, feature_size]` filterbank and the per-frame feature
        // buffer bounded.
        if config.feature_size == 0 || config.feature_size > 4096 {
            config.feature_size = defaults.feature_size;
        }

        let frame_length =
            (config.sampling_rate as f64 * config.frame_length_ms / 1000.0).round() as usize;
        let hop_length =
            (config.sampling_rate as f64 * config.hop_length_ms / 1000.0).round() as usize;

        // A tiny `*_ms` can still round to 0, and a huge one to an allocation
        // bomb; re-derive from the defaults if either lands out of range.
        let (frame_length, hop_length) = if (1..=65_536).contains(&frame_length) && hop_length >= 1
        {
            (frame_length, hop_length)
        } else {
            let fl = (defaults.sampling_rate as f64 * defaults.frame_length_ms / 1000.0).round()
                as usize;
            let hl =
                (defaults.sampling_rate as f64 * defaults.hop_length_ms / 1000.0).round() as usize;
            (fl, hl)
        };

        let mut fft_length = 1;
        while fft_length < frame_length {
            fft_length <<= 1;
        }
        if config.fft_overdrive {
            fft_length *= 2;
        }

        // Periodic Hann window: w(i) = 0.5 - 0.5*cos(2π·i/N)
        // Uses the periodic form (no +0.5 phase shift) to match the HuggingFace
        // Gemma 4 audio reference implementation.
        let window: Vec<f32> = (0..frame_length)
            .map(|i| {
                let arg = 2.0 * PI / frame_length as f64;
                (0.5 - 0.5 * (arg * i as f64).cos()) as f32
            })
            .collect();

        let mel_filters = mel_filter_bank(
            fft_length / 2 + 1,
            config.feature_size,
            config.min_frequency,
            config.max_frequency,
            config.sampling_rate,
        );

        // Per-bin normalization vectors must cover every mel bin; a
        // wrong-length vector from a malformed config block is ignored.
        let per_bin_mean = config
            .per_bin_mean
            .filter(|mean| mean.len() == config.feature_size);
        let per_bin_stddev = config
            .per_bin_stddev
            .filter(|stddev| stddev.len() == config.feature_size);

        Self {
            feature_size: config.feature_size,
            sampling_rate: config.sampling_rate,
            padding_value: config.padding_value,
            preemphasis: config.preemphasis,
            preemphasis_htk_flavor: config.preemphasis_htk_flavor,
            input_scale_factor: config.input_scale_factor,
            mel_floor: config.mel_floor,
            frame_length,
            hop_length,
            fft_length,
            window,
            mel_filters,
            per_bin_mean,
            per_bin_stddev,
        }
    }

    /// Extract log-mel spectrogram from raw waveform samples.
    ///
    /// Returns `(features, mask)` where:
    /// - features: `[T_frames, feature_size]` log-mel features
    /// - mask: `[T_frames]` boolean mask (true = padding/invalid)
    pub fn extract(&self, waveform: &[f32], max_length: Option<usize>) -> (Vec<f32>, Vec<bool>) {
        let max_len = max_length.unwrap_or(480_000);
        let effective_len = waveform.len().min(max_len);

        // Semicausal left-pad: prepend frame_length/2 zeros so the first frame
        // is centered at t=0, matching the HuggingFace Gemma 4 reference.
        let left_pad = self.frame_length / 2;

        // Pad to multiple of 128 if needed (computed on the signal length, not
        // including the left-pad which is always frame_length/2 zeros).
        let pad_multiple = 128;
        let padded_len = if !effective_len.is_multiple_of(pad_multiple) {
            ((effective_len / pad_multiple) + 1) * pad_multiple
        } else {
            effective_len
        };

        let total_len = left_pad + padded_len;
        let mut padded = vec![self.padding_value; total_len];
        padded[left_pad..left_pad + effective_len].copy_from_slice(&waveform[..effective_len]);

        // Attention mask: 1 = valid sample, 0 = left-pad or right-pad.
        // Left-pad region [0..left_pad] is structural zero-padding (mask = 0).
        // Real waveform region [left_pad..left_pad + effective_len] is mask = 1.
        // Right-pad region [left_pad + effective_len..total_len] is mask = 0.
        let mut attn_mask = vec![0i32; total_len];
        for m in attn_mask[left_pad..left_pad + effective_len].iter_mut() {
            *m = 1;
        }

        // Apply input scale
        if self.input_scale_factor != 1.0 {
            for s in &mut padded {
                *s *= self.input_scale_factor as f32;
            }
        }

        // Frame extraction. The reference unfolds frame_length + 1 samples per
        // window regardless of preemphasis (the trailing sample feeds the
        // preemphasis difference; without preemphasis it is dropped), so the
        // frame count is (total_len - frame_length - 1) / hop + 1. Unfolding
        // only frame_length samples would emit one extra frame whenever
        // total_len is a multiple of the hop.
        let frame_size_for_unfold = self.frame_length + 1;
        let num_frames = if total_len >= frame_size_for_unfold {
            (total_len - frame_size_for_unfold) / self.hop_length + 1
        } else {
            0
        };

        if num_frames == 0 {
            return (vec![0.0; self.feature_size], vec![true]);
        }

        let num_freq_bins = self.fft_length / 2 + 1;
        let mut features = vec![0.0f32; num_frames * self.feature_size];

        // FFT scratch buffer
        let mut fft_input = vec![0.0f64; self.fft_length];

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            let frame_data = &padded[start..start + frame_size_for_unfold];

            // Apply preemphasis
            let frame: Vec<f32> = if self.preemphasis > 0.0 {
                if self.preemphasis_htk_flavor {
                    let mut f = Vec::with_capacity(self.frame_length);
                    f.push(frame_data[0] * (1.0 - self.preemphasis as f32));
                    for i in 1..self.frame_length {
                        f.push(frame_data[i] - self.preemphasis as f32 * frame_data[i - 1]);
                    }
                    f
                } else {
                    (0..self.frame_length)
                        .map(|i| frame_data[i + 1] - self.preemphasis as f32 * frame_data[i])
                        .collect()
                }
            } else {
                frame_data[..self.frame_length].to_vec()
            };

            // Apply window
            for i in 0..self.frame_length {
                fft_input[i] = (frame[i] * self.window[i]) as f64;
            }
            for item in fft_input
                .iter_mut()
                .take(self.fft_length)
                .skip(self.frame_length)
            {
                *item = 0.0;
            }

            // Real FFT (using naive DFT for correctness; can optimize later)
            let magnitude = real_fft_magnitude(&fft_input, num_freq_bins);

            // Mel filterbank application
            for mel_idx in 0..self.feature_size {
                let mut mel_val = 0.0f64;
                for (freq_idx, &mag) in magnitude.iter().enumerate().take(num_freq_bins) {
                    mel_val +=
                        mag * self.mel_filters[freq_idx * self.feature_size + mel_idx] as f64;
                }
                // Additive log floor, log(mel + floor), matching the
                // reference. A hard clamp, log(max(mel, floor)), differs by
                // up to ln 2 on low-energy bins.
                let mut log_mel = (mel_val + self.mel_floor).ln();
                if let Some(mean) = &self.per_bin_mean {
                    log_mel -= mean[mel_idx];
                }
                if let Some(stddev) = &self.per_bin_stddev {
                    log_mel /= stddev[mel_idx];
                }
                features[frame_idx * self.feature_size + mel_idx] = log_mel as f32;
            }
        }

        // A frame is valid only when every sample in its analysis window
        // [i*hop, i*hop + frame_size_for_unfold - 1] is real audio; the
        // reference checks the window's LAST sample. Checking the first
        // sample instead would wrongly invalidate frame 0 (whose window
        // starts inside the structural left-pad) and wrongly validate tail
        // frames that straddle the right padding.
        let frame_mask: Vec<bool> = (0..num_frames)
            .map(|i| attn_mask[i * self.hop_length + frame_size_for_unfold - 1] == 0)
            .collect();

        // Zero out padded (invalid) frames, matching the reference's
        // `spec * mask` so the encoder receives identical input.
        for (frame_idx, &invalid) in frame_mask.iter().enumerate() {
            if invalid {
                features[frame_idx * self.feature_size..(frame_idx + 1) * self.feature_size]
                    .fill(0.0);
            }
        }

        (features, frame_mask)
    }

    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    pub fn feature_size(&self) -> usize {
        self.feature_size
    }
}

/// Compute magnitude of real-valued FFT using simple DFT.
/// Returns `[num_freq_bins]` magnitudes.
///
/// Shared with the Whisper log-mel front-end in [`super::whisper_mel`], which
/// squares the result to obtain the power spectrum.
pub(crate) fn real_fft_magnitude(input: &[f64], num_bins: usize) -> Vec<f64> {
    let n = input.len();
    let mut magnitudes = Vec::with_capacity(num_bins);
    for k in 0..num_bins {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, &sample) in input.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * t as f64 / n as f64;
            re += sample * angle.cos();
            im += sample * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Compute the number of audio soft tokens for a waveform of given duration.
///
/// `N = ceil(duration_ms / ms_per_token)`, capped at `max_tokens`.
pub fn compute_audio_num_tokens(
    num_samples: usize,
    sampling_rate: u32,
    ms_per_token: u32,
    max_tokens: usize,
) -> usize {
    let duration_ms = num_samples as f64 / sampling_rate as f64 * 1000.0;
    let num_tokens = (duration_ms / ms_per_token as f64).ceil() as usize;
    num_tokens.min(max_tokens)
}

/// Load raw audio samples from a WAV file.
///
/// Returns mono f32 samples at the file's native sample rate.
pub fn load_wav_file(path: &std::path::Path) -> Result<(Vec<f32>, u32), String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open audio file {}: {e}", path.display()))?;
    let reader = std::io::BufReader::new(file);

    // Simple WAV parser (supports PCM 16-bit and 32-bit float)
    parse_wav(reader)
}

/// Load raw audio samples from in-memory WAV bytes.
///
/// Returns mono f32 samples at the file's native sample rate.
pub fn load_wav_from_bytes(data: &[u8]) -> Result<(Vec<f32>, u32), String> {
    let reader = std::io::Cursor::new(data);
    parse_wav(reader)
}

fn parse_wav<R: std::io::Read>(mut reader: R) -> Result<(Vec<f32>, u32), String> {
    // Maximum audio data size: 500 MB (16-bit mono at 16kHz ~ 4.3 hours).
    // This prevents OOM from malformed WAV headers declaring absurd sizes.
    const MAX_DATA_SIZE: usize = 500 * 1024 * 1024;
    // Maximum number of chunk-scan iterations to prevent infinite loops
    // on malformed WAV files with crafted chunk headers.
    const MAX_CHUNK_SCAN_ITERATIONS: usize = 256;

    let mut header = [0u8; 44];
    reader
        .read_exact(&mut header)
        .map_err(|e| format!("Failed to read WAV header: {e}"))?;

    // Verify RIFF/WAVE header
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".to_string());
    }

    let num_channels = u16::from_le_bytes([header[22], header[23]]) as usize;
    if num_channels == 0 {
        return Err("Invalid WAV file: 0 channels".to_string());
    }
    let sample_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
    if sample_rate == 0 {
        return Err("Invalid WAV file: 0 sample rate".to_string());
    }
    let bits_per_sample = u16::from_le_bytes([header[34], header[35]]);
    let audio_format = u16::from_le_bytes([header[20], header[21]]);

    // Find data chunk (header[36..] might be "data" or we need to scan)
    let data_size = if &header[36..40] == b"data" {
        u32::from_le_bytes([header[40], header[41], header[42], header[43]]) as usize
    } else {
        // Scan for data chunk with bounded iteration count
        let mut buf = header[36..44].to_vec();
        let mut iterations = 0;
        loop {
            if buf.len() >= 8 && &buf[..4] == b"data" {
                break u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
            }
            iterations += 1;
            if iterations > MAX_CHUNK_SCAN_ITERATIONS {
                return Err(
                    "WAV file has too many chunks before data; possibly malformed".to_string(),
                );
            }
            let skip = if buf.len() >= 8 {
                let s = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
                if s == 0 {
                    return Err(
                        "WAV file contains zero-length chunk; possibly malformed".to_string()
                    );
                }
                s
            } else {
                return Err("WAV file chunk too small to contain size field".to_string());
            };
            if skip > MAX_DATA_SIZE {
                return Err(format!(
                    "WAV chunk skip size {skip} exceeds maximum allowed"
                ));
            }
            let mut skip_buf = vec![0u8; skip];
            reader
                .read_exact(&mut skip_buf)
                .map_err(|e| format!("WAV scan error: {e}"))?;
            buf = vec![0u8; 8];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("WAV scan error: {e}"))?;
        }
    };

    if data_size > MAX_DATA_SIZE {
        return Err(format!(
            "WAV data size {data_size} bytes exceeds maximum allowed ({MAX_DATA_SIZE} bytes)"
        ));
    }

    let mut data = vec![0u8; data_size];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("Failed to read WAV data: {e}"))?;

    let samples: Vec<f32> = match (audio_format, bits_per_sample) {
        (1, 16) => {
            // PCM 16-bit
            data.chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect()
        }
        (3, 32) => {
            // IEEE float 32-bit
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }
        _ => {
            return Err(format!(
                "Unsupported WAV format: audio_format={audio_format}, bits_per_sample={bits_per_sample}"
            ));
        }
    };

    // Convert to mono by averaging channels
    let mono = if num_channels > 1 {
        samples
            .chunks_exact(num_channels)
            .map(|ch| ch.iter().sum::<f32>() / num_channels as f32)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pure tone at `freq_hz` for `duration_s` seconds sampled at
    /// `sample_rate` Hz, with unit amplitude.
    fn generate_tone(freq_hz: f64, duration_s: f64, sample_rate: u32) -> Vec<f32> {
        let num_samples = (duration_s * sample_rate as f64).round() as usize;
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f64 / sample_rate as f64).sin() as f32)
            .collect()
    }

    /// Test that the semicausal left-pad produces the correct frame count.
    ///
    /// A 1-second 440 Hz tone at 16 kHz should produce exactly 99 frames with
    /// a 10 ms hop and 20 ms frame (frame_length=320, hop_length=160,
    /// left_pad=160). The reference unfolds frame_length + 1 = 321 samples
    /// per window:
    ///   total_len = 160 + 16000 = 16160
    ///   num_frames = (16160 - 321) / 160 + 1 = 99
    ///
    /// Unfolding only 320 samples would yield a 100th frame here, one more
    /// than the reference emits (issue #782).
    #[test]
    fn test_semicausal_left_pad_frame_count() {
        let tone = generate_tone(440.0, 1.0, 16_000);
        let extractor = AudioFeatureExtractor::new(AudioFeatureExtractorConfig::default());
        let (features, mask) = extractor.extract(&tone, None);
        let num_frames = features.len() / extractor.feature_size();
        assert_eq!(
            num_frames, 99,
            "expected 99 frames for 1s audio with left-pad, got {num_frames}"
        );
        assert_eq!(mask.len(), 99, "mask length must equal num_frames");
        assert!(
            mask.iter().all(|&invalid| !invalid),
            "no frame of an unpadded 1s clip may be marked invalid"
        );
    }

    /// The default config must match the reference `Gemma4AudioFeatureExtractor`
    /// defaults. The published Gemma 4 E-series checkpoints ship no
    /// `feature_extractor` block, so these defaults are the effective config;
    /// `fft_overdrive = true` (fft_length 1024) garbles the Conformer's
    /// perception (issue #782).
    #[test]
    fn test_default_config_matches_reference() {
        let config = AudioFeatureExtractorConfig::default();
        assert!(
            !config.fft_overdrive,
            "reference default is fft_overdrive=false"
        );
        let extractor = AudioFeatureExtractor::new(config);
        assert_eq!(
            extractor.fft_length, 512,
            "20ms/320-sample frame -> fft 512"
        );
        assert_eq!(extractor.frame_length, 320);
        assert_eq!(extractor.hop_length, 160);
        assert_eq!(extractor.feature_size, 128);
    }

    /// Golden log-mel check against the reference extractor.
    ///
    /// The probe values were produced by running the HF/mlx-vlm reference
    /// `Gemma4AudioFeatureExtractor` (default config) over this exact
    /// waveform: a 1 s f64 sine mixture cast to f32,
    ///   0.4*sin(2*pi*440*i/16000) + 0.2*sin(2*pi*1000*i/16000)
    ///   + 0.1*sin(2*pi*3200*i/16000).
    ///
    /// The 1e-3 tolerance is far below the smallest semantic divergence this
    /// guards against (the additive-vs-clamp mel floor differs by up to
    /// ~0.69 = ln 2 per bin; fft_overdrive shifts every bin by ~ln 2) while
    /// leaving room for f32 rounding and libm differences.
    #[test]
    fn test_golden_log_mel_matches_reference() {
        let n = 16_000;
        let waveform: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                (0.4 * (2.0 * PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * PI * 1000.0 * t).sin()
                    + 0.1 * (2.0 * PI * 3200.0 * t).sin()) as f32
            })
            .collect();

        let extractor = AudioFeatureExtractor::new(AudioFeatureExtractorConfig::default());
        let (features, mask) = extractor.extract(&waveform, None);
        let feature_size = extractor.feature_size();
        assert_eq!(mask.len(), 99);
        assert_eq!(features.len(), 99 * feature_size);

        // (frame, mel_bin) -> reference value. The -6.907755 = ln(1e-3)
        // entries probe the additive mel floor on near-zero-energy bins.
        let golden: &[(usize, usize, f32)] = &[
            (0, 0, -6.907755),
            (0, 25, 2.3889),
            (0, 64, -0.843163),
            (0, 127, -0.936063),
            (10, 3, -4.119002),
            (25, 25, 2.974617),
            (50, 40, -1.472469),
            (50, 90, -3.042887),
            (80, 10, -4.155638),
            (90, 70, -6.709886),
            (98, 0, -6.907755),
            (98, 127, -6.883512),
        ];
        for &(frame, bin, expected) in golden {
            let actual = features[frame * feature_size + bin];
            assert!(
                (actual - expected).abs() < 1e-3,
                "features[{frame}][{bin}] = {actual}, reference = {expected}"
            );
        }

        let mean: f64 = features.iter().map(|&v| v as f64).sum::<f64>() / features.len() as f64;
        assert!(
            (mean - (-3.93961)).abs() < 1e-3,
            "global feature mean {mean} deviates from reference -3.93961"
        );
    }

    /// A clip needing right-padding (15000 samples pads to 15104, a multiple
    /// of 128) must mark exactly the tail frame whose analysis window
    /// straddles the padding as invalid, zero that feature row, and keep
    /// frame 0 (whose window starts inside the structural left-pad) valid.
    /// Matches the reference, which checks the LAST sample of each window
    /// and multiplies the spectrogram by the mask.
    #[test]
    fn test_tail_padding_mask_and_zeroing() {
        let waveform = generate_tone(440.0, 0.9375, 16_000); // 15000 samples
        assert_eq!(waveform.len(), 15_000);
        let extractor = AudioFeatureExtractor::new(AudioFeatureExtractorConfig::default());
        let (features, mask) = extractor.extract(&waveform, None);
        let feature_size = extractor.feature_size();
        assert_eq!(mask.len(), 94, "(160 + 15104 - 321) / 160 + 1 = 94 frames");
        assert!(!mask[0], "frame 0 must be valid despite the left-pad");
        let invalid: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        assert_eq!(invalid, vec![93], "only the tail frame is invalid");
        assert!(
            features[93 * feature_size..94 * feature_size]
                .iter()
                .all(|&v| v == 0.0),
            "invalid frame's features must be zeroed"
        );
    }

    /// `from_processor_config` honors a checkpoint-shipped block and falls
    /// back to reference defaults otherwise; wrong-length per-bin vectors
    /// are rejected at construction.
    #[test]
    fn test_from_processor_config() {
        // No processor config / no block -> reference defaults.
        let config = AudioFeatureExtractorConfig::from_processor_config(None);
        assert!(!config.fft_overdrive);
        let empty = serde_json::json!({ "image_processor": {} });
        let config = AudioFeatureExtractorConfig::from_processor_config(Some(&empty));
        assert!(!config.fft_overdrive);
        assert_eq!(config.feature_size, 128);

        // Checkpoint-shipped block overrides only the fields it names.
        let with_block = serde_json::json!({
            "feature_extractor": {
                "feature_extractor_type": "Gemma4AudioFeatureExtractor",
                "fft_overdrive": true,
                "frame_length_ms": 32.0,
                "mel_floor": 1e-5,
                "per_bin_mean": [0.5],
            }
        });
        let config = AudioFeatureExtractorConfig::from_processor_config(Some(&with_block));
        assert!(config.fft_overdrive);
        assert_eq!(config.frame_length_ms, 32.0);
        assert_eq!(config.mel_floor, 1e-5);
        assert_eq!(config.hop_length_ms, 10.0, "unnamed fields keep defaults");
        let extractor = AudioFeatureExtractor::new(config);
        assert_eq!(extractor.fft_length, 1024, "512-sample frame + overdrive");
        assert!(
            extractor.per_bin_mean.is_none(),
            "length-1 per_bin_mean must be rejected for feature_size 128"
        );
    }

    /// Test that the periodic Hann window places the 440 Hz energy peak in the
    /// correct mel bin range (roughly bins 20-35 out of 128 for 440 Hz on a
    /// 0-8000 Hz mel scale at 16 kHz).
    ///
    /// The non-periodic window (with +0.5 phase shift) subtly distorts spectral
    /// magnitudes; the periodic form matches the HuggingFace reference.
    #[test]
    fn test_periodic_hann_window_mel_peak() {
        let tone = generate_tone(440.0, 1.0, 16_000);
        let extractor = AudioFeatureExtractor::new(AudioFeatureExtractorConfig::default());
        let (features, _mask) = extractor.extract(&tone, None);
        let num_frames = features.len() / extractor.feature_size();
        let feature_size = extractor.feature_size();

        // Average mel energy across all non-padding frames
        let mut mel_energy = vec![0.0f64; feature_size];
        for frame in 0..num_frames {
            for mel in 0..feature_size {
                mel_energy[mel] += features[frame * feature_size + mel] as f64;
            }
        }

        // Find the mel bin with maximum average energy
        let peak_bin = mel_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // 440 Hz maps to ~25th mel bin (out of 128) on a 0-8000 Hz scale.
        // Allow a range of ±10 bins to account for filterbank overlap.
        assert!(
            (15..=40).contains(&peak_bin),
            "440 Hz mel energy peak at bin {peak_bin}, expected in range 15-40"
        );
    }

    /// A malformed `feature_extractor` config block (which reaches this path now
    /// that the extractor honors `processor_config.json`) must not build an
    /// extractor that divides by zero, allocates unboundedly, or spins the
    /// `fft_length` loop. Each nonsensical scalar falls back to the reference
    /// default, and `extract` then runs to completion on a real waveform.
    #[test]
    fn malformed_config_scalars_fall_back_to_defaults() {
        let defaults = AudioFeatureExtractorConfig::default();
        let tone = generate_tone(440.0, 0.5, 16_000);

        // hop_length_ms == 0 would divide by zero in `extract`.
        for bad in [
            AudioFeatureExtractorConfig {
                hop_length_ms: 0.0,
                ..AudioFeatureExtractorConfig::default()
            },
            AudioFeatureExtractorConfig {
                frame_length_ms: 0.0,
                ..AudioFeatureExtractorConfig::default()
            },
            AudioFeatureExtractorConfig {
                frame_length_ms: f64::NAN,
                ..AudioFeatureExtractorConfig::default()
            },
            AudioFeatureExtractorConfig {
                sampling_rate: 0,
                ..AudioFeatureExtractorConfig::default()
            },
            AudioFeatureExtractorConfig {
                feature_size: 0,
                ..AudioFeatureExtractorConfig::default()
            },
            AudioFeatureExtractorConfig {
                feature_size: usize::MAX,
                ..AudioFeatureExtractorConfig::default()
            },
            AudioFeatureExtractorConfig {
                frame_length_ms: 1.0e12,
                ..AudioFeatureExtractorConfig::default()
            },
        ] {
            let extractor = AudioFeatureExtractor::new(bad);
            // feature_size never exceeds the sane cap; the default is restored
            // for the zero/oversized cases.
            assert!(extractor.feature_size() >= 1 && extractor.feature_size() <= 4096);
            let (features, mask) = extractor.extract(&tone, None);
            assert!(!features.is_empty());
            assert_eq!(features.len() % extractor.feature_size(), 0);
            assert!(!mask.is_empty());
        }

        // A well-formed default config is untouched by the guards.
        let ok = AudioFeatureExtractor::new(defaults);
        assert_eq!(ok.feature_size(), 128);
    }
}
