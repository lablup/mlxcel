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

//! Audio output encoding.
//!
//! Counterpart to the WAV reader in [`super::feature_extractor`]
//! (`load_wav_file` / `load_wav_from_bytes` / `parse_wav`). This module turns
//! `f32` PCM samples back into a self-contained byte stream so the server can
//! return synthesized audio over HTTP. The encoded layout is the canonical
//! 44-byte RIFF header followed by 16-bit little-endian PCM data, which the
//! reader decodes through its fast `header[36..40] == b"data"` path.

/// Number of bytes per encoded sample (16-bit PCM).
const BYTES_PER_SAMPLE: u32 = 2;
/// PCM bit depth produced by this encoder.
const BITS_PER_SAMPLE: u16 = 16;
/// WAV `audio_format` tag for integer PCM.
const WAV_FORMAT_PCM: u16 = 1;
/// Largest PCM payload a 32-bit RIFF `data` length can describe. The 44-byte
/// header is reserved so the total `RIFF` chunk size also fits in `u32`.
const MAX_DATA_BYTES: u32 = u32::MAX - 44;

/// Encode interleaved `f32` PCM samples as a canonical 16-bit little-endian
/// RIFF WAV byte stream.
///
/// `samples` are amplitudes nominally in `[-1.0, 1.0]`; values outside the
/// range are clamped to the representable `i16` window. For multi-channel
/// audio (`channels > 1`) the samples must already be interleaved
/// (`L, R, L, R, ...`). The scaling mirrors the reader's `i16 / 32768.0`
/// decode, so an encode followed by `load_wav_from_bytes` round-trips within
/// one quantization step.
///
/// Used by: server/routes/audio.rs (text-to-speech binary response)
#[must_use]
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    // A 32-bit RIFF `data` length caps a single WAV at roughly 4 GiB of PCM.
    // Clamp the sample count to what that field can represent so the advertised
    // length always matches the bytes actually written, even for pathological
    // inputs. A larger payload cannot be expressed in a standard RIFF WAV, so
    // truncating here keeps the output well-formed rather than corrupt.
    let max_samples = (MAX_DATA_BYTES / BYTES_PER_SAMPLE) as usize;
    let encoded = &samples[..samples.len().min(max_samples)];
    let data_len = (encoded.len() as u32).saturating_mul(BYTES_PER_SAMPLE);
    let byte_rate = sample_rate
        .saturating_mul(channels as u32)
        .saturating_mul(BYTES_PER_SAMPLE);
    let block_align = channels.saturating_mul(BITS_PER_SAMPLE / 8);
    // 44-byte header (RIFF + fmt + data descriptors) plus the PCM payload.
    let mut out = Vec::with_capacity(44 + data_len as usize);

    // RIFF chunk descriptor.
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32.saturating_add(data_len)).to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt sub-chunk.
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&WAV_FORMAT_PCM.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

    // data sub-chunk.
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &sample in encoded {
        // Mirror the reader's `i16 / 32768.0` decode: scale by 32768, then
        // clamp into the i16 range so a full-scale 1.0 saturates rather than
        // wrapping to a negative value. NaN maps to 0 and infinities saturate
        // because Rust's `f32 as i16` cast is saturating.
        let scaled = (sample * 32768.0).round();
        let clamped = scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        out.extend_from_slice(&clamped.to_le_bytes());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::load_wav_from_bytes;
    use std::f32::consts::PI;

    /// Generate a unit-amplitude tone for round-trip checks.
    fn tone(freq_hz: f32, num_samples: usize, sample_rate: u32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn encode_then_read_round_trips_mono_samples() {
        let sample_rate = 16_000;
        let original = tone(440.0, 1_600, sample_rate);

        let bytes = encode_wav_pcm16(&original, sample_rate, 1);
        let (decoded, decoded_rate) =
            load_wav_from_bytes(&bytes).expect("encoded WAV must read back");

        assert_eq!(decoded_rate, sample_rate);
        assert_eq!(decoded.len(), original.len());
        // 16-bit quantization error is bounded by 1/32768 ~= 3.05e-5.
        for (i, (&a, &b)) in original.iter().zip(decoded.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1.0e-4,
                "sample {i} diverged: encoded {a}, decoded {b}"
            );
        }
    }

    #[test]
    fn header_advertises_sample_rate_and_mono_channel() {
        let bytes = encode_wav_pcm16(&[0.0, 0.25, -0.25], 24_000, 1);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_eq!(rate, 24_000);
        let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
        assert_eq!(channels, 1);
        let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
        assert_eq!(bits, 16);
    }

    #[test]
    fn full_scale_values_saturate_without_wrapping() {
        // +1.0 would scale to 32768, which must clamp to i16::MAX (32767),
        // not wrap to a large negative value.
        let bytes = encode_wav_pcm16(&[1.0, -1.0], 8_000, 1);
        let (decoded, _) = load_wav_from_bytes(&bytes).expect("WAV must read back");
        assert!(
            decoded[0] > 0.99,
            "positive full scale wrapped: {}",
            decoded[0]
        );
        assert!(
            (decoded[1] + 1.0).abs() < 1.0e-4,
            "negative full scale diverged: {}",
            decoded[1]
        );
    }

    #[test]
    fn header_data_length_matches_payload() {
        let bytes = encode_wav_pcm16(&[0.1_f32; 100], 16_000, 1);
        let data_len = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize;
        assert_eq!(data_len, 200, "data chunk advertises the PCM byte count");
        assert_eq!(
            bytes.len(),
            44 + data_len,
            "payload length matches the header"
        );
        let riff_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        assert_eq!(riff_len, 36 + data_len, "RIFF size covers the body");
    }

    #[test]
    fn pathological_floats_saturate_without_panicking() {
        let bytes = encode_wav_pcm16(
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 5.0, -5.0],
            8_000,
            1,
        );
        let (decoded, _) = load_wav_from_bytes(&bytes).expect("pathological WAV must read back");
        assert!(
            decoded[0].abs() < 1.0e-4,
            "NaN should encode to silence, got {}",
            decoded[0]
        );
        assert!(
            decoded[1] > 0.99,
            "positive infinity should saturate high, got {}",
            decoded[1]
        );
        assert!(
            decoded[2] < -0.99,
            "negative infinity should saturate low, got {}",
            decoded[2]
        );
        assert!(
            decoded[3] > 0.99,
            "out-of-range positive should saturate high"
        );
        assert!(
            decoded[4] < -0.99,
            "out-of-range negative should saturate low"
        );
    }

    #[test]
    fn interleaved_stereo_round_trips_through_mono_downmix() {
        // The reader averages channels into mono; duplicating L==R means the
        // average equals the source so the duplicated channel still matches.
        let sample_rate = 16_000;
        let mono = tone(220.0, 512, sample_rate);
        let mut interleaved = Vec::with_capacity(mono.len() * 2);
        for &s in &mono {
            interleaved.push(s);
            interleaved.push(s);
        }

        let bytes = encode_wav_pcm16(&interleaved, sample_rate, 2);
        let (decoded, decoded_rate) =
            load_wav_from_bytes(&bytes).expect("stereo WAV must read back");

        assert_eq!(decoded_rate, sample_rate);
        assert_eq!(decoded.len(), mono.len());
        for (&a, &b) in mono.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1.0e-4, "downmix diverged: {a} vs {b}");
        }
    }
}
