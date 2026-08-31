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

use crate::audio::{AudioClipBoundary, AudioSourceKind, AudioWaveformBatch, OwnedAudioWaveform};

use super::*;

fn waveform(samples: usize, ordinal: usize) -> OwnedAudioWaveform {
    OwnedAudioWaveform {
        samples: (0..samples)
            .map(|index| ((index * 17 + ordinal) % 97) as f32 / 97.0 - 0.5)
            .collect(),
        sample_rate: INKLING_SAMPLE_RATE,
        source_sample_rate: INKLING_SAMPLE_RATE,
        source_channels: 1,
        source_samples: samples,
        source_duration_micros: samples as u64 * 1_000_000 / INKLING_SAMPLE_RATE as u64,
        encoded_bytes: samples * 2,
        source: AudioSourceKind::CliFile,
        placeholder_ordinal: ordinal,
    }
}

#[test]
fn compacts_only_valid_rows_in_clip_order() {
    let clips = vec![waveform(1_600, 1), waveform(801, 2)];
    let batch = AudioWaveformBatch {
        family: "inkling",
        clips,
        boundaries: vec![
            AudioClipBoundary {
                clip_index: 0,
                placeholder_ordinal: 1,
                start_sample: 0,
                end_sample: 1_600,
            },
            AudioClipBoundary {
                clip_index: 1,
                placeholder_ordinal: 2,
                start_sample: 1_600,
                end_sample: 2_401,
            },
        ],
        total_source_samples: 2_401,
        total_samples: 2_401,
        total_source_duration_micros: 0,
        estimated_frames: 4,
        effective_audio_tokens: 4,
    };
    let prepared = prepare_inkling_dmel_input(&batch, &InklingProcessorConfig::default()).unwrap();
    assert_eq!(prepared.valid_frames, vec![2, 2]);
    assert_eq!(prepared.total_frames(), 4);
    assert_eq!(prepared.mlx_shape().unwrap(), [4, 80]);
    assert_eq!(prepared.input_ids.len(), 4 * 80);
    assert!(prepared.input_ids.iter().all(|id| (0..16).contains(id)));
}

#[test]
fn rejects_wrong_family_and_unresampled_clips() {
    let mut clip = waveform(800, 1);
    let mut batch = AudioWaveformBatch {
        family: "other",
        clips: vec![clip.clone()],
        boundaries: Vec::new(),
        total_source_samples: 800,
        total_samples: 800,
        total_source_duration_micros: 50_000,
        estimated_frames: 1,
        effective_audio_tokens: 1,
    };
    assert!(prepare_inkling_dmel_input(&batch, &InklingProcessorConfig::default()).is_err());
    clip.sample_rate = 8_000;
    batch.family = "inkling";
    batch.clips = vec![clip];
    assert!(prepare_inkling_dmel_input(&batch, &InklingProcessorConfig::default()).is_err());
}
