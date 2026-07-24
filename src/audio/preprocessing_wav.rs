use super::{
    AudioCancellation, AudioFamilyPolicy, AudioPreprocessCheckpoint, AudioPreprocessError,
    AudioResamplingPolicy, check_cancel,
};

const MAX_WAV_CHUNKS: usize = 256;

pub(super) struct NativeWaveform {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub frames: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WavSpec<'a> {
    pub data: &'a [u8],
    pub audio_format: u16,
    pub bits: u16,
    pub sample_rate: u32,
    pub channels: u16,
    pub frames: usize,
    pub bytes_per_sample: usize,
    pub frame_bytes: usize,
}

pub(super) fn inspect_wav<'a>(
    bytes: &'a [u8],
    clip_index: usize,
    policy: AudioFamilyPolicy,
    cancelled: &dyn AudioCancellation,
) -> Result<WavSpec<'a>, AudioPreprocessError> {
    check_cancel(
        cancelled,
        AudioPreprocessCheckpoint::Decode,
        Some(clip_index),
    )?;
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return corrupt(clip_index, "missing RIFF/WAVE header");
    }
    let mut offset = 12usize;
    let mut format = None;
    let mut data = None;
    for _ in 0..MAX_WAV_CHUNKS {
        if offset == bytes.len() {
            break;
        }
        let header_end = offset
            .checked_add(8)
            .ok_or(AudioPreprocessError::Overflow {
                context: "WAV chunk header",
            })?;
        if header_end > bytes.len() {
            return corrupt(clip_index, "truncated WAV chunk header");
        }
        let id = &bytes[offset..offset + 4];
        let declared =
            u32::from_le_bytes(bytes[offset + 4..header_end].try_into().unwrap_or([0; 4])) as usize;
        let start = header_end;
        let end = if declared == u32::MAX as usize && id == b"data" {
            bytes.len()
        } else {
            start
                .checked_add(declared)
                .ok_or(AudioPreprocessError::Overflow {
                    context: "WAV chunk length",
                })?
        };
        if end > bytes.len() {
            return corrupt(clip_index, "declared WAV chunk exceeds payload");
        }
        if id == b"fmt " {
            if declared < 16 {
                return corrupt(clip_index, "WAV fmt chunk is shorter than 16 bytes");
            }
            format = Some(&bytes[start..end]);
        } else if id == b"data" {
            data = Some(&bytes[start..end]);
            if format.is_some() {
                break;
            }
        }
        offset = end
            .checked_add(declared & 1)
            .ok_or(AudioPreprocessError::Overflow {
                context: "WAV chunk padding",
            })?;
        check_cancel(
            cancelled,
            AudioPreprocessCheckpoint::Decode,
            Some(clip_index),
        )?;
    }
    let fmt = format.ok_or_else(|| AudioPreprocessError::Corrupt {
        clip_index,
        reason: "missing WAV fmt chunk".to_string(),
    })?;
    let data = data.ok_or_else(|| AudioPreprocessError::Corrupt {
        clip_index,
        reason: "missing WAV data chunk".to_string(),
    })?;
    let audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
    let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
    let sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
    if channels == 0 || sample_rate == 0 {
        return corrupt(
            clip_index,
            "WAV channel count and sample rate must be non-zero",
        );
    }
    if sample_rate < policy.minimum_source_sample_rate {
        return corrupt(
            clip_index,
            &format!(
                "source sample rate {sample_rate} Hz is below the {} Hz family minimum",
                policy.minimum_source_sample_rate
            ),
        );
    }
    if sample_rate > policy.maximum_source_sample_rate {
        return Err(AudioPreprocessError::Limit {
            limit: "source sample rate",
            actual: sample_rate as usize,
            maximum: policy.maximum_source_sample_rate as usize,
        });
    }
    let bytes_per_sample = match (audio_format, bits) {
        (1, 16) => 2usize,
        (3, 32) => 4usize,
        _ => {
            return corrupt(
                clip_index,
                &format!("unsupported WAV format {audio_format}/{bits}-bit"),
            );
        }
    };
    let frame_bytes =
        bytes_per_sample
            .checked_mul(channels as usize)
            .ok_or(AudioPreprocessError::Overflow {
                context: "WAV frame size",
            })?;
    if data.is_empty() {
        return Err(AudioPreprocessError::Empty { clip_index });
    }
    if data.len() % frame_bytes != 0 {
        return corrupt(
            clip_index,
            "WAV data is not aligned to complete sample frames",
        );
    }
    let frames = data.len() / frame_bytes;
    let max_source_frames = (sample_rate as usize)
        .checked_mul(policy.max_duration_seconds)
        .ok_or(AudioPreprocessError::Overflow {
            context: "source sample limit",
        })?;
    if frames > max_source_frames {
        return Err(AudioPreprocessError::Limit {
            limit: "source duration samples",
            actual: frames,
            maximum: max_source_frames,
        });
    }
    Ok(WavSpec {
        data,
        audio_format,
        bits,
        sample_rate,
        channels,
        frames,
        bytes_per_sample,
        frame_bytes,
    })
}

pub(super) fn decode_wav(
    bytes: &[u8],
    clip_index: usize,
    policy: AudioFamilyPolicy,
    cancelled: &dyn AudioCancellation,
) -> Result<NativeWaveform, AudioPreprocessError> {
    let spec = inspect_wav(bytes, clip_index, policy, cancelled)?;
    let WavSpec {
        data,
        audio_format,
        bits,
        sample_rate,
        channels,
        frames,
        bytes_per_sample,
        frame_bytes,
    } = spec;
    let mut mono = Vec::with_capacity(frames);
    for (frame, chunk) in data.chunks_exact(frame_bytes).enumerate() {
        if frame % 4096 == 0 {
            check_cancel(
                cancelled,
                AudioPreprocessCheckpoint::Decode,
                Some(clip_index),
            )?;
        }
        let mut sum = 0.0f64;
        for channel in 0..channels as usize {
            let start = channel * bytes_per_sample;
            let sample = match (audio_format, bits) {
                (1, 16) => i16::from_le_bytes([chunk[start], chunk[start + 1]]) as f32 / 32768.0,
                (3, 32) => f32::from_le_bytes(chunk[start..start + 4].try_into().unwrap_or([0; 4])),
                _ => unreachable!(),
            };
            if !sample.is_finite() {
                return Err(AudioPreprocessError::NonFinite { clip_index, frame });
            }
            sum += sample as f64;
        }
        mono.push((sum / channels as f64).clamp(-1.0, 1.0) as f32);
    }
    Ok(NativeWaveform {
        samples: mono,
        sample_rate,
        channels,
        frames,
    })
}

pub(super) fn resample(
    samples: &[f32],
    source_rate: u32,
    policy: AudioFamilyPolicy,
    clip_index: usize,
    cancelled: &dyn AudioCancellation,
) -> Result<(Vec<f32>, u32), AudioPreprocessError> {
    check_cancel(
        cancelled,
        AudioPreprocessCheckpoint::Resample,
        Some(clip_index),
    )?;
    if policy.resampling == AudioResamplingPolicy::Native {
        return Ok((samples.to_vec(), source_rate));
    }
    if policy.resampling == AudioResamplingPolicy::Phi4MmSpeechLib {
        let (output, effective_rate) =
            phi4mm_resample_cancellable(samples, source_rate, clip_index, cancelled)?;
        check_cancel(
            cancelled,
            AudioPreprocessCheckpoint::Resample,
            Some(clip_index),
        )?;
        return Ok((output.into_owned(), effective_rate));
    }
    let target_rate = policy.target_sample_rate;
    if source_rate == target_rate {
        return Ok((samples.to_vec(), target_rate));
    }
    let numerator =
        samples
            .len()
            .checked_mul(target_rate as usize)
            .ok_or(AudioPreprocessError::Overflow {
                context: "resampled length",
            })?;
    let output_len = numerator.div_ceil(source_rate as usize);
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        if index % 4096 == 0 {
            check_cancel(
                cancelled,
                AudioPreprocessCheckpoint::Resample,
                Some(clip_index),
            )?;
        }
        let position =
            index
                .checked_mul(source_rate as usize)
                .ok_or(AudioPreprocessError::Overflow {
                    context: "resample position",
                })?;
        let left = position / target_rate as usize;
        let remainder = position % target_rate as usize;
        let a = samples[left.min(samples.len() - 1)];
        let b = samples[(left + 1).min(samples.len() - 1)];
        output.push(a + (b - a) * (remainder as f32 / target_rate as f32));
    }
    Ok((output, target_rate))
}

pub(crate) fn phi4mm_resample(
    samples: &[f32],
    source_rate: u32,
) -> Result<(std::borrow::Cow<'_, [f32]>, u32), String> {
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    phi4mm_resample_cancellable(samples, source_rate, 0, &cancelled)
        .map_err(|error| error.to_string())
}

fn phi4mm_resample_cancellable<'a>(
    samples: &'a [f32],
    source_rate: u32,
    clip_index: usize,
    cancelled: &dyn AudioCancellation,
) -> Result<(std::borrow::Cow<'a, [f32]>, u32), AudioPreprocessError> {
    if source_rate > 192_000 {
        return Err(AudioPreprocessError::Limit {
            limit: "source sample rate",
            actual: source_rate as usize,
            maximum: 192_000,
        });
    }
    let (down, effective_rate) = if source_rate == 8_000 || source_rate == 16_000 {
        (1, source_rate)
    } else if source_rate > 16_000 {
        ((source_rate / 16_000) as usize, 16_000)
    } else {
        ((source_rate / 8_000) as usize, 8_000)
    };
    if down <= 1 {
        Ok((std::borrow::Cow::Borrowed(samples), effective_rate))
    } else {
        Ok((
            std::borrow::Cow::Owned(super::resampling::scipy_resample_poly_down(
                samples, down, clip_index, cancelled,
            )?),
            effective_rate,
        ))
    }
}

pub(super) fn resampled_shape(
    source_samples: usize,
    source_rate: u32,
    policy: AudioFamilyPolicy,
) -> Result<(usize, u32), AudioPreprocessError> {
    if policy.resampling == AudioResamplingPolicy::Native {
        return Ok((source_samples, source_rate));
    }
    if policy.resampling == AudioResamplingPolicy::Phi4MmSpeechLib {
        if source_rate > policy.maximum_source_sample_rate {
            return Err(AudioPreprocessError::Limit {
                limit: "source sample rate",
                actual: source_rate as usize,
                maximum: policy.maximum_source_sample_rate as usize,
            });
        }
        let (down, effective_rate) = if source_rate == 8_000 || source_rate == 16_000 {
            (1usize, source_rate)
        } else if source_rate > 16_000 {
            ((source_rate / 16_000) as usize, 16_000)
        } else {
            ((source_rate / 8_000) as usize, 8_000)
        };
        let (output, _) = super::resampling::validate_polyphase_shape(source_samples, down.max(1))?;
        return Ok((output, effective_rate));
    }
    let numerator = source_samples
        .checked_mul(policy.target_sample_rate as usize)
        .ok_or(AudioPreprocessError::Overflow {
            context: "resampled length",
        })?;
    Ok((
        numerator.div_ceil(source_rate as usize),
        policy.target_sample_rate,
    ))
}

pub(super) fn estimate_frames(
    samples: usize,
    effective_rate: u32,
    policy: AudioFamilyPolicy,
) -> usize {
    if policy.family == "gemma3n" {
        let padded = samples.div_ceil(128) * 128;
        return padded
            .saturating_sub(policy.frame_length_samples)
            .checked_div(policy.frame_hop_samples)
            .unwrap_or(0)
            .saturating_add(usize::from(padded >= policy.frame_length_samples));
    }
    let (frame_length, hop) =
        if policy.resampling == AudioResamplingPolicy::Phi4MmSpeechLib && effective_rate == 8_000 {
            (200, 80)
        } else {
            (policy.frame_length_samples, policy.frame_hop_samples)
        };
    samples
        .saturating_sub(frame_length)
        .checked_div(hop)
        .unwrap_or(0)
        .saturating_add(usize::from(samples >= frame_length))
}

pub(super) fn duration_micros(
    frames: usize,
    sample_rate: u32,
) -> Result<u64, AudioPreprocessError> {
    let frames = u64::try_from(frames).map_err(|_| AudioPreprocessError::Overflow {
        context: "source frame count",
    })?;
    frames
        .checked_mul(1_000_000)
        .map(|value| value / sample_rate as u64)
        .ok_or(AudioPreprocessError::Overflow {
            context: "source duration",
        })
}

fn corrupt<T>(clip_index: usize, reason: &str) -> Result<T, AudioPreprocessError> {
    Err(AudioPreprocessError::Corrupt {
        clip_index,
        reason: reason.to_string(),
    })
}
