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

//! Decoded-duration cap on decision audio.

use super::media::{MAX_DECISION_AUDIO_SECONDS, decode_audio_16k};
use crate::audio::encode_wav_pcm16;

fn wav(seconds: usize, rate: u32) -> Vec<u8> {
    let samples = vec![0.1_f32; seconds * rate as usize];
    encode_wav_pcm16(&samples, rate, 1)
}

#[test]
fn a_clip_within_the_cap_decodes_to_16k_samples() {
    let samples = decode_audio_16k(&wav(1, 16_000)).expect("one second decodes");
    assert_eq!(samples.len(), 16_000);
}

#[test]
fn a_clip_at_exactly_the_cap_is_accepted() {
    let samples = decode_audio_16k(&wav(MAX_DECISION_AUDIO_SECONDS, 8_000)).expect("at the cap");
    assert!(!samples.is_empty());
}

#[test]
fn a_clip_over_the_cap_is_rejected_with_a_clear_message() {
    let err = decode_audio_16k(&wav(MAX_DECISION_AUDIO_SECONDS + 1, 8_000)).unwrap_err();
    assert_eq!(
        err,
        "audio is 301 seconds long; decision audio is limited to 300 seconds"
    );
}

#[test]
fn the_cap_is_on_duration_not_sample_count() {
    // 100 s at 48 kHz is 4.8M samples, more than 300 s at 16 kHz would be.
    let samples = decode_audio_16k(&wav(100, 48_000)).expect("short high-rate clip decodes");
    assert_eq!(samples.len(), 100 * 16_000);
}

#[test]
fn a_clip_over_the_cap_is_refused_from_its_header_before_decoding() {
    use crate::audio::preprocessing::{AudioPreprocessError, decode_native_within};
    let clip = wav(MAX_DECISION_AUDIO_SECONDS + 1, 8_000);
    // The container's inspect step raises this limit from the declared frame
    // count; the decode loop is never entered.
    match decode_native_within(&clip, MAX_DECISION_AUDIO_SECONDS) {
        Err(AudioPreprocessError::Limit {
            limit,
            actual,
            maximum,
        }) => {
            assert_eq!(limit, "source duration samples");
            assert_eq!(actual, 301 * 8_000);
            assert_eq!(maximum, 300 * 8_000);
        }
        other => panic!("expected the container duration limit, got {other:?}"),
    }
    // Other callers keep the 24-hour compatibility ceiling.
    assert!(crate::audio::preprocessing::decode_wav_native_compat(&clip).is_ok());
    assert_eq!(
        decode_audio_16k(&clip).unwrap_err(),
        "audio is 301 seconds long; decision audio is limited to 300 seconds"
    );
}
