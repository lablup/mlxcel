use super::AudioPreprocessError;

const DEFAULT_MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;
const PHI4MM_MAX_ENCODED_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_CLIPS: usize = 16;
const MAX_SOURCE_SAMPLE_RATE: u32 = 192_000;
const MICROS_PER_SECOND: u64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPlaceholderPolicy {
    NumberedPerClip,
    FixedSoftTokensPerClip(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPolicySource {
    PinnedOfficialDefault(&'static str),
    ModelProcessorConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioResamplingPolicy {
    Native,
    Linear,
    Phi4MmSpeechLib,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFamilyPolicy {
    pub family: &'static str,
    pub target_sample_rate: u32,
    pub minimum_source_sample_rate: u32,
    pub maximum_source_sample_rate: u32,
    pub target_channels: u16,
    pub dtype: &'static str,
    pub resampling: AudioResamplingPolicy,
    pub max_duration_seconds: usize,
    pub max_samples_per_clip: usize,
    pub max_encoded_bytes_per_clip: usize,
    pub max_clips: usize,
    /// Aggregate encoded bytes retained by one request.
    pub max_encoded_bytes_per_request: usize,
    /// Aggregate native mono samples before resampling.
    pub max_source_samples_per_request: usize,
    /// Aggregate mono samples after family resampling.
    pub max_normalized_samples_per_request: usize,
    /// Aggregate source duration, independent of source rate.
    pub max_source_duration_micros_per_request: u64,
    pub frame_length_samples: usize,
    pub frame_hop_samples: usize,
    pub max_frames_per_clip: usize,
    pub max_frames_per_request: usize,
    /// Maximum retained normalized waveform storage.
    pub max_waveform_result_bytes_per_request: usize,
    /// Peak host waveform working storage, excluding encoded request bytes.
    pub max_waveform_working_bytes_per_request: usize,
    /// Maximum owned host payload returned by a future XLA feature producer.
    pub max_prepared_result_bytes_per_request: usize,
    pub placeholder: AudioPlaceholderPolicy,
    pub source: AudioPolicySource,
}

impl AudioFamilyPolicy {
    #[must_use]
    pub const fn phi4mm() -> Self {
        Self {
            family: "phi4mm",
            target_sample_rate: 16_000,
            minimum_source_sample_rate: 8_000,
            maximum_source_sample_rate: MAX_SOURCE_SAMPLE_RATE,
            target_channels: 1,
            dtype: "f32",
            resampling: AudioResamplingPolicy::Phi4MmSpeechLib,
            max_duration_seconds: crate::audio::phi4mm::MAX_AUDIO_DURATION_SECONDS,
            max_samples_per_clip: 16_000 * crate::audio::phi4mm::MAX_AUDIO_DURATION_SECONDS,
            // 30 minutes of stereo float32 PCM is about 230 MiB.
            max_encoded_bytes_per_clip: PHI4MM_MAX_ENCODED_BYTES,
            max_clips: DEFAULT_MAX_CLIPS,
            // A request may split the pinned 30-minute family maximum across
            // clips, but may not multiply it by `max_clips`.
            max_encoded_bytes_per_request: PHI4MM_MAX_ENCODED_BYTES,
            max_source_samples_per_request: MAX_SOURCE_SAMPLE_RATE as usize
                * crate::audio::phi4mm::MAX_AUDIO_DURATION_SECONDS,
            max_normalized_samples_per_request: 16_000
                * crate::audio::phi4mm::MAX_AUDIO_DURATION_SECONDS,
            max_source_duration_micros_per_request: crate::audio::phi4mm::MAX_AUDIO_DURATION_SECONDS
                as u64
                * MICROS_PER_SECOND,
            frame_length_samples: 400,
            frame_hop_samples: 160,
            // SpeechLib uses (samples - 400) / 160 + 1 at 16 kHz.
            max_frames_per_clip: 179_998,
            max_frames_per_request: 179_998,
            max_waveform_result_bytes_per_request: 128 * 1024 * 1024,
            max_waveform_working_bytes_per_request: 512 * 1024 * 1024,
            max_prepared_result_bytes_per_request: 256 * 1024 * 1024,
            placeholder: AudioPlaceholderPolicy::NumberedPerClip,
            source: AudioPolicySource::PinnedOfficialDefault(
                crate::audio::phi4mm::PHI4MM_TRANSFORMERS_REFERENCE_REVISION,
            ),
        }
    }

    #[must_use]
    pub const fn gemma3n() -> Self {
        Self {
            family: "gemma3n",
            target_sample_rate: crate::audio::gemma3n::GEMMA3N_SAMPLE_RATE,
            minimum_source_sample_rate: 1,
            maximum_source_sample_rate: MAX_SOURCE_SAMPLE_RATE,
            target_channels: 1,
            dtype: "f32",
            resampling: AudioResamplingPolicy::Linear,
            max_duration_seconds: 30,
            max_samples_per_clip: crate::audio::gemma3n::GEMMA3N_MAX_SAMPLES,
            max_encoded_bytes_per_clip: DEFAULT_MAX_ENCODED_BYTES,
            max_clips: DEFAULT_MAX_CLIPS,
            max_encoded_bytes_per_request: DEFAULT_MAX_ENCODED_BYTES,
            max_source_samples_per_request: MAX_SOURCE_SAMPLE_RATE as usize * 30,
            max_normalized_samples_per_request: crate::audio::gemma3n::GEMMA3N_MAX_SAMPLES,
            max_source_duration_micros_per_request: 30 * MICROS_PER_SECOND,
            frame_length_samples: 513,
            frame_hop_samples: 160,
            max_frames_per_clip: 2_997,
            max_frames_per_request: 2_997,
            max_waveform_result_bytes_per_request: 2 * 1024 * 1024,
            max_waveform_working_bytes_per_request: 32 * 1024 * 1024,
            max_prepared_result_bytes_per_request: 64 * 1024 * 1024,
            placeholder: AudioPlaceholderPolicy::FixedSoftTokensPerClip(
                crate::audio::gemma3n::GEMMA3N_AUDIO_SOFT_TOKENS,
            ),
            source: AudioPolicySource::PinnedOfficialDefault(
                crate::audio::gemma3n::GEMMA3N_TRANSFORMERS_REFERENCE_REVISION,
            ),
        }
    }

    pub fn from_phi4mm_configs(
        model: &serde_json::Value,
        processor: Option<&serde_json::Value>,
    ) -> Result<Self, AudioPreprocessError> {
        Self::from_configs(Self::phi4mm(), model, processor, None)
    }

    pub fn from_gemma3n_configs(
        model: &serde_json::Value,
        processor: Option<&serde_json::Value>,
        soft_tokens_per_clip: usize,
    ) -> Result<Self, AudioPreprocessError> {
        let mut policy = Self::gemma3n();
        policy.placeholder = AudioPlaceholderPolicy::FixedSoftTokensPerClip(soft_tokens_per_clip);
        Self::from_configs(policy, model, processor, Some(soft_tokens_per_clip))
    }

    fn from_configs(
        mut policy: Self,
        model: &serde_json::Value,
        processor: Option<&serde_json::Value>,
        soft_tokens: Option<usize>,
    ) -> Result<Self, AudioPreprocessError> {
        let roots = [processor, Some(model)];
        let official = policy;
        let official_max_duration = official.max_duration_seconds;
        let mut configured = false;
        if let Some(value) = first_u64(
            &roots,
            &[
                "/audio_processor/sampling_rate",
                "/audio_processor/sample_rate",
                "/feature_extractor/sampling_rate",
                "/sampling_rate",
            ],
        )? {
            configured = true;
            policy.target_sample_rate =
                u32::try_from(value).map_err(|_| config_error("sampling_rate overflows u32"))?;
        }
        if let Some(value) = first_u64(
            &roots,
            &[
                "/audio_processor/max_audio_samples",
                "/feature_extractor/max_length",
                "/max_audio_samples",
            ],
        )? {
            configured = true;
            policy.max_samples_per_clip =
                usize::try_from(value).map_err(|_| config_error("max audio samples overflow"))?;
        }
        if let Some(value) = first_u64(
            &roots,
            &[
                "/audio_processor/max_audio_duration",
                "/audio_processor/max_duration_seconds",
                "/feature_extractor/max_audio_duration",
                "/feature_extractor/max_duration_seconds",
                "/max_audio_duration",
                "/max_duration_seconds",
            ],
        )? {
            configured = true;
            let seconds = usize::try_from(value)
                .map_err(|_| config_error("max audio duration overflows usize"))?;
            if seconds == 0 || seconds > official_max_duration {
                return Err(config_error(
                    "max audio duration must be positive and no larger than the pinned family maximum",
                ));
            }
            let duration_samples = (policy.target_sample_rate as usize)
                .checked_mul(seconds)
                .ok_or(AudioPreprocessError::Overflow {
                    context: "loaded audio duration sample cap",
                })?;
            policy.max_samples_per_clip = policy.max_samples_per_clip.min(duration_samples);
        }
        if let Some(value) =
            first_u64(&roots, &["/mlxcel_audio_limits/max_encoded_bytes_per_clip"])?
        {
            configured = true;
            policy.max_encoded_bytes_per_clip =
                usize::try_from(value).map_err(|_| config_error("encoded byte cap overflow"))?;
        }
        if let Some(value) = first_u64(&roots, &["/mlxcel_audio_limits/max_clips"])? {
            configured = true;
            policy.max_clips =
                usize::try_from(value).map_err(|_| config_error("clip cap overflow"))?;
        }
        for (paths, field, name) in [
            (
                &["/mlxcel_audio_limits/max_encoded_bytes_per_request"][..],
                &mut policy.max_encoded_bytes_per_request,
                "request encoded byte cap",
            ),
            (
                &["/mlxcel_audio_limits/max_source_samples_per_request"][..],
                &mut policy.max_source_samples_per_request,
                "request source sample cap",
            ),
            (
                &["/mlxcel_audio_limits/max_normalized_samples_per_request"][..],
                &mut policy.max_normalized_samples_per_request,
                "request normalized sample cap",
            ),
            (
                &["/mlxcel_audio_limits/max_frames_per_request"][..],
                &mut policy.max_frames_per_request,
                "request feature frame cap",
            ),
            (
                &["/mlxcel_audio_limits/max_waveform_result_bytes_per_request"][..],
                &mut policy.max_waveform_result_bytes_per_request,
                "waveform result byte cap",
            ),
            (
                &["/mlxcel_audio_limits/max_waveform_working_bytes_per_request"][..],
                &mut policy.max_waveform_working_bytes_per_request,
                "waveform working byte cap",
            ),
            (
                &["/mlxcel_audio_limits/max_prepared_result_bytes_per_request"][..],
                &mut policy.max_prepared_result_bytes_per_request,
                "prepared result byte cap",
            ),
        ] {
            if let Some(value) = first_u64(&roots, paths)? {
                configured = true;
                *field = usize::try_from(value)
                    .map_err(|_| config_error(&format!("{name} overflows usize")))?;
            }
        }
        if let Some(value) = first_u64(
            &roots,
            &["/mlxcel_audio_limits/max_source_duration_micros_per_request"],
        )? {
            configured = true;
            policy.max_source_duration_micros_per_request = value;
        }
        if let Some(value) =
            first_u64(&roots, &["/mlxcel_audio_limits/maximum_source_sample_rate"])?
        {
            configured = true;
            policy.maximum_source_sample_rate = u32::try_from(value)
                .map_err(|_| config_error("maximum source sample rate overflows u32"))?;
        }
        if let Some(value) = first_u64(
            &roots,
            &[
                "/audio_processor/frame_length_samples",
                "/feature_extractor/frame_length",
            ],
        )? {
            configured = true;
            policy.frame_length_samples =
                usize::try_from(value).map_err(|_| config_error("frame length overflow"))?;
        }
        if let Some(value) = first_u64(
            &roots,
            &[
                "/audio_processor/frame_hop_samples",
                "/feature_extractor/hop_length",
            ],
        )? {
            configured = true;
            policy.frame_hop_samples =
                usize::try_from(value).map_err(|_| config_error("frame hop overflow"))?;
        }
        if let Some(channels) = first_u64(
            &roots,
            &["/audio_processor/channels", "/feature_extractor/channels"],
        )? {
            configured = true;
            policy.target_channels =
                u16::try_from(channels).map_err(|_| config_error("channel count overflows u16"))?;
        }
        if let Some(dtype) = first_str(
            &roots,
            &["/audio_processor/dtype", "/feature_extractor/dtype"],
        )? {
            configured = true;
            if dtype != policy.dtype {
                return Err(config_error("only f32 waveform dtype is supported"));
            }
        }
        if let Some(resampling) = first_str(&roots, &["/mlxcel_audio_limits/resampling_algorithm"])?
        {
            configured = true;
            let expected = match policy.resampling {
                AudioResamplingPolicy::Native => "native",
                AudioResamplingPolicy::Linear => "linear",
                AudioResamplingPolicy::Phi4MmSpeechLib => "scipy_polyphase_integer",
            };
            if resampling != expected {
                return Err(config_error(
                    "resampling algorithm does not match the pinned family frontend",
                ));
            }
        }
        if policy.target_sample_rate != 16_000
            || policy.target_channels != 1
            || policy.max_samples_per_clip == 0
            || policy.max_encoded_bytes_per_clip == 0
            || policy.max_clips == 0
            || policy.maximum_source_sample_rate < policy.minimum_source_sample_rate
            || policy.max_encoded_bytes_per_request == 0
            || policy.max_source_samples_per_request == 0
            || policy.max_normalized_samples_per_request == 0
            || policy.max_source_duration_micros_per_request == 0
            || policy.frame_length_samples == 0
            || policy.frame_hop_samples == 0
            || policy.max_frames_per_request == 0
            || policy.max_waveform_result_bytes_per_request == 0
            || policy.max_waveform_working_bytes_per_request == 0
            || policy.max_prepared_result_bytes_per_request == 0
            || soft_tokens == Some(0)
        {
            return Err(config_error(
                "loaded audio policy is incompatible with the pinned mono f32 16 kHz processor",
            ));
        }
        let official_max = policy
            .target_sample_rate
            .try_into()
            .ok()
            .and_then(|rate: usize| rate.checked_mul(official_max_duration))
            .ok_or(AudioPreprocessError::Overflow {
                context: "official family sample cap",
            })?;
        if policy.max_samples_per_clip > official_max {
            return Err(config_error(
                "loaded audio sample cap exceeds the pinned family duration",
            ));
        }
        let upward = [
            (
                policy.max_encoded_bytes_per_clip as u128,
                official.max_encoded_bytes_per_clip as u128,
                "per-clip encoded byte cap",
            ),
            (
                policy.max_clips as u128,
                official.max_clips as u128,
                "clip cap",
            ),
            (
                policy.maximum_source_sample_rate as u128,
                official.maximum_source_sample_rate as u128,
                "maximum source sample rate",
            ),
            (
                policy.max_encoded_bytes_per_request as u128,
                official.max_encoded_bytes_per_request as u128,
                "request encoded byte cap",
            ),
            (
                policy.max_source_samples_per_request as u128,
                official.max_source_samples_per_request as u128,
                "request source sample cap",
            ),
            (
                policy.max_normalized_samples_per_request as u128,
                official.max_normalized_samples_per_request as u128,
                "request normalized sample cap",
            ),
            (
                policy.max_source_duration_micros_per_request as u128,
                official.max_source_duration_micros_per_request as u128,
                "request source duration cap",
            ),
            (
                policy.max_frames_per_request as u128,
                official.max_frames_per_request as u128,
                "request feature frame cap",
            ),
            (
                policy.max_waveform_result_bytes_per_request as u128,
                official.max_waveform_result_bytes_per_request as u128,
                "waveform result byte cap",
            ),
            (
                policy.max_waveform_working_bytes_per_request as u128,
                official.max_waveform_working_bytes_per_request as u128,
                "waveform working byte cap",
            ),
            (
                policy.max_prepared_result_bytes_per_request as u128,
                official.max_prepared_result_bytes_per_request as u128,
                "prepared result byte cap",
            ),
        ];
        if let Some((_, _, name)) = upward
            .into_iter()
            .find(|(loaded, pinned, _)| loaded > pinned)
        {
            return Err(config_error(&format!(
                "{name} cannot exceed the pinned family maximum"
            )));
        }
        policy.max_duration_seconds = policy
            .max_samples_per_clip
            .div_ceil(policy.target_sample_rate as usize);
        policy.max_frames_per_clip = super::wav::estimate_frames(
            policy.max_samples_per_clip,
            policy.target_sample_rate,
            policy,
        );
        if configured {
            policy.source = AudioPolicySource::ModelProcessorConfig;
        }
        Ok(policy)
    }
}

fn first_u64(
    roots: &[Option<&serde_json::Value>],
    paths: &[&'static str],
) -> Result<Option<u64>, AudioPreprocessError> {
    let Some((path, value)) = first_value(roots, paths) else {
        return Ok(None);
    };
    value.as_u64().map(Some).ok_or_else(|| {
        config_error(&format!(
            "{path} must be a non-negative integer, got {}",
            json_type(value)
        ))
    })
}

fn first_str<'a>(
    roots: &'a [Option<&'a serde_json::Value>],
    paths: &[&'static str],
) -> Result<Option<&'a str>, AudioPreprocessError> {
    let Some((path, value)) = first_value(roots, paths) else {
        return Ok(None);
    };
    value.as_str().map(Some).ok_or_else(|| {
        config_error(&format!(
            "{path} must be a string, got {}",
            json_type(value)
        ))
    })
}

fn first_value<'a>(
    roots: &'a [Option<&'a serde_json::Value>],
    paths: &[&'static str],
) -> Option<(&'static str, &'a serde_json::Value)> {
    roots.iter().flatten().find_map(|root| {
        paths
            .iter()
            .find_map(|path| root.pointer(path).map(|value| (*path, value)))
    })
}

fn json_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number outside the accepted integer range",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn config_error(reason: &str) -> AudioPreprocessError {
    AudioPreprocessError::Context {
        clip_index: 0,
        reason: format!("invalid loaded audio processor config: {reason}"),
    }
}
