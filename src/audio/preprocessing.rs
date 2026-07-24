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

//! Owned, backend-neutral audio request preprocessing.
//!
//! The boundary deliberately ends at normalized mono waveforms. Model-specific
//! mel/Conformer execution remains in each family implementation, while CLI,
//! HTTP, MLX, and compiler backends can share byte validation, WAV decoding,
//! downmixing, resampling, clip order, and resource accounting.

use std::io::Read;
use std::mem::size_of;
use std::path::Path;

use thiserror::Error;

#[path = "preprocessing_policy.rs"]
mod policy;
#[path = "preprocessing_resample.rs"]
pub(crate) mod resampling;
#[path = "preprocessing_wav.rs"]
pub(crate) mod wav;

pub use policy::{
    AudioFamilyPolicy, AudioPlaceholderPolicy, AudioPolicySource, AudioResamplingPolicy,
};
use wav::{decode_wav, duration_micros, estimate_frames, inspect_wav, resample, resampled_shape};

/// Point at which cancellation was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPreprocessCheckpoint {
    Acquisition,
    Decode,
    Resample,
    Feature,
    Queue,
}

/// Cancellation probe shared by synchronous CLI and worker-thread callers.
pub trait AudioCancellation: Send + Sync {
    fn is_cancelled(&self, checkpoint: AudioPreprocessCheckpoint) -> bool;
}

impl AudioCancellation for std::sync::atomic::AtomicBool {
    fn is_cancelled(&self, _checkpoint: AudioPreprocessCheckpoint) -> bool {
        self.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Source class retained for observability without retaining paths or URLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSourceKind {
    CliFile,
    ServerInline,
    ServerDataUri,
    ServerFile,
    ServerUrl,
}

/// Encoded request clip before the owned waveform boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioEncodedClip {
    pub bytes: Vec<u8>,
    pub source: AudioSourceKind,
    /// One-based placeholder ordinal in request order.
    pub placeholder_ordinal: usize,
}

/// Borrowed encoded clip used by synchronous CLI/server preparation without
/// duplicating request buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioEncodedClipRef<'a> {
    pub bytes: &'a [u8],
    pub source: AudioSourceKind,
    pub placeholder_ordinal: usize,
}

/// Metadata and owned normalized mono samples for one request clip.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedAudioWaveform {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub source_sample_rate: u32,
    pub source_channels: u16,
    pub source_samples: usize,
    pub source_duration_micros: u64,
    pub encoded_bytes: usize,
    pub source: AudioSourceKind,
    pub placeholder_ordinal: usize,
}

/// Explicit flattened boundary for one clip in request order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioClipBoundary {
    pub clip_index: usize,
    pub placeholder_ordinal: usize,
    pub start_sample: usize,
    pub end_sample: usize,
}

/// Owned request batch passed to a family feature producer.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioWaveformBatch {
    pub family: &'static str,
    pub clips: Vec<OwnedAudioWaveform>,
    pub boundaries: Vec<AudioClipBoundary>,
    /// Input mono frames before family resampling.
    pub total_source_samples: usize,
    /// Normalized mono frames after family resampling.
    pub total_samples: usize,
    pub total_source_duration_micros: u64,
    pub estimated_frames: usize,
    pub effective_audio_tokens: usize,
}

/// Typed errors at the acquisition/decode/resample/placeholder boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AudioPreprocessError {
    #[error("audio preprocessing cancelled during {checkpoint:?} for clip {clip_index:?}")]
    Cancelled {
        checkpoint: AudioPreprocessCheckpoint,
        clip_index: Option<usize>,
    },
    #[error("audio clip {clip_index} is empty")]
    Empty { clip_index: usize },
    #[error("audio clip {clip_index} is corrupt: {reason}")]
    Corrupt { clip_index: usize, reason: String },
    #[error("audio clip {clip_index} contains a non-finite sample at frame {frame}")]
    NonFinite { clip_index: usize, frame: usize },
    #[error("audio {limit} limit exceeded: actual {actual}, maximum {maximum}")]
    Limit {
        limit: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("audio size calculation overflowed while computing {context}")]
    Overflow { context: &'static str },
    #[error("audio placeholder mapping is invalid: {reason}")]
    Placeholder { reason: String },
    #[error("audio source context failed for clip {clip_index}: {reason}")]
    Context { clip_index: usize, reason: String },
}

/// Read and preprocess a CLI file with the same family policy as server bytes.
pub fn preprocess_wav_file(
    path: &Path,
    policy: AudioFamilyPolicy,
    cancelled: &dyn AudioCancellation,
) -> Result<AudioWaveformBatch, AudioPreprocessError> {
    check_cancel(cancelled, AudioPreprocessCheckpoint::Acquisition, Some(0))?;
    let mut file = std::fs::File::open(path).map_err(|error| AudioPreprocessError::Context {
        clip_index: 0,
        reason: format!("failed to open {}: {error}", path.display()),
    })?;
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 64 * 1024];
    loop {
        check_cancel(cancelled, AudioPreprocessCheckpoint::Acquisition, Some(0))?;
        let read = file
            .read(&mut chunk)
            .map_err(|error| AudioPreprocessError::Context {
                clip_index: 0,
                reason: format!("failed to read {}: {error}", path.display()),
            })?;
        if read == 0 {
            break;
        }
        let next = bytes
            .len()
            .checked_add(read)
            .ok_or(AudioPreprocessError::Overflow {
                context: "CLI encoded audio bytes",
            })?;
        if next > policy.max_encoded_bytes_per_clip {
            return Err(AudioPreprocessError::Limit {
                limit: "encoded bytes per clip",
                actual: next,
                maximum: policy.max_encoded_bytes_per_clip,
            });
        }
        bytes.extend_from_slice(&chunk[..read]);
        check_cancel(cancelled, AudioPreprocessCheckpoint::Acquisition, Some(0))?;
    }
    if bytes.is_empty() {
        return Err(AudioPreprocessError::Empty { clip_index: 0 });
    }
    let clip = AudioEncodedClip {
        bytes,
        source: AudioSourceKind::CliFile,
        placeholder_ordinal: 1,
    };
    preprocess_wav_batch(&[clip], policy, cancelled)
}

/// Decode, normalize, and resample encoded clips without model/runtime state.
pub fn preprocess_wav_batch(
    encoded: &[AudioEncodedClip],
    policy: AudioFamilyPolicy,
    cancelled: &dyn AudioCancellation,
) -> Result<AudioWaveformBatch, AudioPreprocessError> {
    let borrowed: Vec<_> = encoded
        .iter()
        .map(|clip| AudioEncodedClipRef {
            bytes: &clip.bytes,
            source: clip.source,
            placeholder_ordinal: clip.placeholder_ordinal,
        })
        .collect();
    preprocess_wav_refs(&borrowed, policy, cancelled)
}

/// Preprocess borrowed request buffers without copying encoded payloads.
pub fn preprocess_wav_refs(
    encoded: &[AudioEncodedClipRef<'_>],
    policy: AudioFamilyPolicy,
    cancelled: &dyn AudioCancellation,
) -> Result<AudioWaveformBatch, AudioPreprocessError> {
    check_cancel(cancelled, AudioPreprocessCheckpoint::Acquisition, None)?;
    if encoded.is_empty() {
        return Err(AudioPreprocessError::Empty { clip_index: 0 });
    }
    if encoded.len() > policy.max_clips {
        return Err(AudioPreprocessError::Limit {
            limit: "clip count",
            actual: encoded.len(),
            maximum: policy.max_clips,
        });
    }

    // Inspect every RIFF header and enforce aggregate family limits before
    // allocating any decoded or resampled waveform. This prevents a request
    // from multiplying a valid per-clip maximum by `max_clips`.
    let mut preflight_encoded_bytes = 0usize;
    let mut preflight_source_samples = 0usize;
    let mut preflight_normalized_samples = 0usize;
    let mut preflight_duration_micros = 0u64;
    let mut preflight_frames = 0usize;
    let mut preflight_max_source_clip = 0usize;
    for (index, input) in encoded.iter().enumerate() {
        if input.placeholder_ordinal != index + 1 {
            return Err(AudioPreprocessError::Placeholder {
                reason: format!(
                    "clip {index} maps to ordinal {}, expected {}",
                    input.placeholder_ordinal,
                    index + 1
                ),
            });
        }
        if input.bytes.is_empty() {
            return Err(AudioPreprocessError::Empty { clip_index: index });
        }
        enforce_limit(
            "encoded bytes per clip",
            input.bytes.len(),
            policy.max_encoded_bytes_per_clip,
        )?;
        preflight_encoded_bytes = checked_add(
            preflight_encoded_bytes,
            input.bytes.len(),
            "request encoded bytes",
        )?;
        enforce_limit(
            "encoded bytes per request",
            preflight_encoded_bytes,
            policy.max_encoded_bytes_per_request,
        )?;
        let spec = inspect_wav(input.bytes, index, policy, cancelled)?;
        let source_duration = duration_micros(spec.frames, spec.sample_rate)?;
        preflight_source_samples = checked_add(
            preflight_source_samples,
            spec.frames,
            "request source samples",
        )?;
        enforce_limit(
            "source samples per request",
            preflight_source_samples,
            policy.max_source_samples_per_request,
        )?;
        preflight_duration_micros = preflight_duration_micros
            .checked_add(source_duration)
            .ok_or(AudioPreprocessError::Overflow {
                context: "request source duration",
            })?;
        if preflight_duration_micros > policy.max_source_duration_micros_per_request {
            return Err(AudioPreprocessError::Limit {
                limit: "source duration micros per request",
                actual: usize::try_from(preflight_duration_micros).unwrap_or(usize::MAX),
                maximum: usize::try_from(policy.max_source_duration_micros_per_request)
                    .unwrap_or(usize::MAX),
            });
        }
        let (normalized_samples, effective_rate) =
            resampled_shape(spec.frames, spec.sample_rate, policy)?;
        let per_clip_max = if policy.resampling == AudioResamplingPolicy::Phi4MmSpeechLib {
            (effective_rate as usize)
                .checked_mul(policy.max_duration_seconds)
                .ok_or(AudioPreprocessError::Overflow {
                    context: "Phi4MM effective sample limit",
                })?
        } else {
            policy.max_samples_per_clip
        };
        enforce_limit("duration samples", normalized_samples, per_clip_max)?;
        preflight_normalized_samples = checked_add(
            preflight_normalized_samples,
            normalized_samples,
            "request normalized samples",
        )?;
        enforce_limit(
            "normalized samples per request",
            preflight_normalized_samples,
            policy.max_normalized_samples_per_request,
        )?;
        let frames = estimate_frames(normalized_samples, effective_rate, policy);
        enforce_limit(
            "feature frames per clip",
            frames,
            policy.max_frames_per_clip,
        )?;
        preflight_frames = checked_add(preflight_frames, frames, "request feature frames")?;
        enforce_limit(
            "feature frames per request",
            preflight_frames,
            policy.max_frames_per_request,
        )?;
        preflight_max_source_clip = preflight_max_source_clip.max(spec.frames);
    }
    let waveform_result_bytes = preflight_normalized_samples
        .checked_mul(size_of::<f32>())
        .ok_or(AudioPreprocessError::Overflow {
            context: "request waveform result bytes",
        })?;
    enforce_limit(
        "waveform result bytes per request",
        waveform_result_bytes,
        policy.max_waveform_result_bytes_per_request,
    )?;
    let peak_source_bytes = preflight_max_source_clip
        .checked_mul(size_of::<f32>())
        .ok_or(AudioPreprocessError::Overflow {
            context: "peak source waveform bytes",
        })?;
    let waveform_working_bytes = peak_source_bytes.checked_add(waveform_result_bytes).ok_or(
        AudioPreprocessError::Overflow {
            context: "waveform working bytes",
        },
    )?;
    enforce_limit(
        "waveform working bytes per request",
        waveform_working_bytes,
        policy.max_waveform_working_bytes_per_request,
    )?;

    let mut clips = Vec::with_capacity(encoded.len());
    let mut boundaries = Vec::with_capacity(encoded.len());
    let mut total_source_samples = 0usize;
    let mut total_samples = 0usize;
    let mut total_duration = 0u64;
    let mut estimated_frames = 0usize;
    let mut effective_tokens = 0usize;
    for (index, input) in encoded.iter().enumerate() {
        let native = decode_wav(input.bytes, index, policy, cancelled)?;
        let duration_micros = duration_micros(native.frames, native.sample_rate)?;
        total_source_samples = total_source_samples.checked_add(native.frames).ok_or(
            AudioPreprocessError::Overflow {
                context: "request source samples",
            },
        )?;
        let (samples, effective_rate) = resample(
            &native.samples,
            native.sample_rate,
            policy,
            index,
            cancelled,
        )?;
        let max_samples = if policy.resampling == AudioResamplingPolicy::Phi4MmSpeechLib {
            (effective_rate as usize)
                .checked_mul(policy.max_duration_seconds)
                .ok_or(AudioPreprocessError::Overflow {
                    context: "Phi4MM effective sample limit",
                })?
        } else {
            policy.max_samples_per_clip
        };
        if samples.len() > max_samples {
            return Err(AudioPreprocessError::Limit {
                limit: "duration samples",
                actual: samples.len(),
                maximum: max_samples,
            });
        }
        let frames = estimate_frames(samples.len(), effective_rate, policy);
        if frames > policy.max_frames_per_clip {
            return Err(AudioPreprocessError::Limit {
                limit: "feature frames per clip",
                actual: frames,
                maximum: policy.max_frames_per_clip,
            });
        }
        let start = total_samples;
        total_samples =
            total_samples
                .checked_add(samples.len())
                .ok_or(AudioPreprocessError::Overflow {
                    context: "request samples",
                })?;
        total_duration =
            total_duration
                .checked_add(duration_micros)
                .ok_or(AudioPreprocessError::Overflow {
                    context: "source duration",
                })?;
        estimated_frames =
            estimated_frames
                .checked_add(frames)
                .ok_or(AudioPreprocessError::Overflow {
                    context: "feature frames",
                })?;
        effective_tokens = effective_tokens
            .checked_add(tokens_for_clip(frames, policy.placeholder))
            .ok_or(AudioPreprocessError::Overflow {
                context: "effective audio tokens",
            })?;
        boundaries.push(AudioClipBoundary {
            clip_index: index,
            placeholder_ordinal: input.placeholder_ordinal,
            start_sample: start,
            end_sample: total_samples,
        });
        clips.push(OwnedAudioWaveform {
            samples,
            sample_rate: effective_rate,
            source_sample_rate: native.sample_rate,
            source_channels: native.channels,
            source_samples: native.frames,
            source_duration_micros: duration_micros,
            encoded_bytes: input.bytes.len(),
            source: input.source,
            placeholder_ordinal: input.placeholder_ordinal,
        });
    }
    Ok(AudioWaveformBatch {
        family: policy.family,
        clips,
        boundaries,
        total_source_samples,
        total_samples,
        total_source_duration_micros: total_duration,
        estimated_frames,
        effective_audio_tokens: effective_tokens,
    })
}

fn checked_add(
    current: usize,
    value: usize,
    context: &'static str,
) -> Result<usize, AudioPreprocessError> {
    current
        .checked_add(value)
        .ok_or(AudioPreprocessError::Overflow { context })
}

fn enforce_limit(
    limit: &'static str,
    actual: usize,
    maximum: usize,
) -> Result<(), AudioPreprocessError> {
    if actual > maximum {
        Err(AudioPreprocessError::Limit {
            limit,
            actual,
            maximum,
        })
    } else {
        Ok(())
    }
}

/// Compatibility decoder used by legacy audio families.
///
/// This keeps one WAV parser/downmixer while preserving the historical native
/// sample-rate return type. New family integrations should use a loaded
/// [`AudioFamilyPolicy`] and [`preprocess_wav_batch`] instead.
pub(crate) fn decode_wav_native_compat(
    bytes: &[u8],
) -> Result<(Vec<f32>, u32), AudioPreprocessError> {
    struct NeverCancel;
    impl AudioCancellation for NeverCancel {
        fn is_cancelled(&self, _checkpoint: AudioPreprocessCheckpoint) -> bool {
            false
        }
    }
    let policy = AudioFamilyPolicy {
        family: "legacy-wav",
        target_sample_rate: 1,
        minimum_source_sample_rate: 1,
        maximum_source_sample_rate: u32::MAX,
        target_channels: 1,
        dtype: "f32",
        resampling: AudioResamplingPolicy::Native,
        max_duration_seconds: 24 * 60 * 60,
        max_samples_per_clip: usize::MAX,
        max_encoded_bytes_per_clip: 500 * 1024 * 1024,
        max_clips: 1,
        max_encoded_bytes_per_request: 500 * 1024 * 1024,
        max_source_samples_per_request: usize::MAX / size_of::<f32>(),
        max_normalized_samples_per_request: usize::MAX / size_of::<f32>(),
        max_source_duration_micros_per_request: u64::MAX,
        frame_length_samples: 1,
        frame_hop_samples: 1,
        max_frames_per_clip: usize::MAX,
        max_frames_per_request: usize::MAX,
        max_waveform_result_bytes_per_request: usize::MAX,
        max_waveform_working_bytes_per_request: usize::MAX,
        max_prepared_result_bytes_per_request: usize::MAX,
        placeholder: AudioPlaceholderPolicy::NumberedPerClip,
        source: AudioPolicySource::PinnedOfficialDefault("legacy-wav-native"),
    };
    if bytes.len() > policy.max_encoded_bytes_per_clip {
        return Err(AudioPreprocessError::Limit {
            limit: "encoded bytes per clip",
            actual: bytes.len(),
            maximum: policy.max_encoded_bytes_per_clip,
        });
    }
    let native = decode_wav(bytes, 0, policy, &NeverCancel)?;
    Ok((native.samples, native.sample_rate))
}

fn tokens_for_clip(frames: usize, placeholder: AudioPlaceholderPolicy) -> usize {
    match placeholder {
        AudioPlaceholderPolicy::NumberedPerClip => frames.div_ceil(8),
        AudioPlaceholderPolicy::FixedSoftTokensPerClip(tokens) => tokens,
    }
}

pub(super) fn check_cancel(
    cancelled: &dyn AudioCancellation,
    checkpoint: AudioPreprocessCheckpoint,
    clip_index: Option<usize>,
) -> Result<(), AudioPreprocessError> {
    if cancelled.is_cancelled(checkpoint) {
        Err(AudioPreprocessError::Cancelled {
            checkpoint,
            clip_index,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "preprocessing_tests.rs"]
mod tests;
