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

//! Host bridge from normalized waveforms to compact Inkling dMel token rows.

use crate::audio::AudioWaveformBatch;
use std::sync::atomic::AtomicBool;

use crate::audio::inkling_dmel::{INKLING_SAMPLE_RATE, InklingFeatureExtractor, quantize_dmel};
use crate::audio::inkling_processor::InklingProcessorConfig;

/// Owned valid dMel rows in clip order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingDmelInput {
    /// Row-major `[total_valid_frames, n_mel_bins]` bucket IDs.
    pub input_ids: Vec<i32>,
    /// Valid frame count for each input clip, in request order.
    pub valid_frames: Vec<usize>,
    pub n_mel_bins: usize,
    total_frames: usize,
}

impl InklingDmelInput {
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.total_frames
    }

    pub fn mlx_shape(&self) -> Result<[i32; 2], String> {
        Ok([
            i32::try_from(self.total_frames())
                .map_err(|_| "Inkling audio frame count exceeds the MLX i32 limit")?,
            i32::try_from(self.n_mel_bins)
                .map_err(|_| "Inkling audio channel count exceeds the MLX i32 limit")?,
        ])
    }
}

/// Extract, quantize, and compact one normalized waveform request.
///
/// The shared audio boundary owns decoding, mono downmixing, 16 kHz
/// resampling, and request limits. This function preserves clip order and
/// removes every right-padding row before the summed-channel tower runs.
pub fn prepare_inkling_dmel_input(
    waveforms: &AudioWaveformBatch,
    processor: &InklingProcessorConfig,
) -> Result<InklingDmelInput, String> {
    let cancelled = AtomicBool::new(false);
    prepare_inkling_dmel_input_cancellable(waveforms, processor, &cancelled)
}

/// Server variant that propagates cancellation into every frame transform.
pub fn prepare_inkling_dmel_input_cancellable(
    waveforms: &AudioWaveformBatch,
    processor: &InklingProcessorConfig,
    cancelled: &AtomicBool,
) -> Result<InklingDmelInput, String> {
    processor.validate()?;
    if waveforms.family != "inkling" {
        return Err(format!(
            "Inkling dMel preprocessing received a {} waveform batch",
            waveforms.family
        ));
    }
    if waveforms.clips.is_empty() {
        return Err("Inkling dMel preprocessing requires at least one clip".into());
    }
    for (index, clip) in waveforms.clips.iter().enumerate() {
        if clip.sample_rate != INKLING_SAMPLE_RATE {
            return Err(format!(
                "Inkling audio clip {index} must be resampled to {INKLING_SAMPLE_RATE} Hz, got {} Hz",
                clip.sample_rate
            ));
        }
        if clip.samples.is_empty() {
            return Err(format!("Inkling audio clip {index} is empty"));
        }
    }

    let extractor = InklingFeatureExtractor::new(processor.feature_extractor.clone())?;
    let clips: Vec<&[f32]> = waveforms
        .clips
        .iter()
        .map(|clip| clip.samples.as_slice())
        .collect();
    let features = extractor.extract_compact_batch_cancellable(
        &clips,
        crate::audio::AudioFamilyPolicy::inkling().max_frames_per_request,
        cancelled,
    )?;
    let quantized = quantize_dmel(
        &features.features,
        processor.num_dmel_bins,
        processor.dmel_min_value,
        processor.dmel_max_value,
    )?;

    let total_frames = features.transformed_frames;
    let id_capacity = total_frames
        .checked_mul(features.feature_size)
        .ok_or_else(|| "Inkling compact dMel allocation overflowed".to_string())?;
    let input_ids = quantized;
    if input_ids.len() != id_capacity {
        return Err(format!(
            "Inkling dMel mask retained {} IDs, expected {id_capacity}",
            input_ids.len()
        ));
    }
    let result = InklingDmelInput {
        input_ids,
        valid_frames: features.valid_frames,
        n_mel_bins: features.feature_size,
        total_frames,
    };
    result.mlx_shape()?;
    Ok(result)
}

#[cfg(test)]
#[path = "inkling_audio_tests.rs"]
mod tests;
