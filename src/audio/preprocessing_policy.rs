use super::AudioPreprocessError;

const DEFAULT_MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;
const PHI4MM_MAX_ENCODED_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_CLIPS: usize = 16;

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
    pub target_channels: u16,
    pub dtype: &'static str,
    pub resampling: AudioResamplingPolicy,
    pub max_duration_seconds: usize,
    pub max_samples_per_clip: usize,
    pub max_encoded_bytes_per_clip: usize,
    pub max_clips: usize,
    pub frame_length_samples: usize,
    pub frame_hop_samples: usize,
    pub max_frames_per_clip: usize,
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
            target_channels: 1,
            dtype: "f32",
            resampling: AudioResamplingPolicy::Phi4MmSpeechLib,
            max_duration_seconds: crate::audio::phi4mm::MAX_AUDIO_DURATION_SECONDS,
            max_samples_per_clip: 16_000 * crate::audio::phi4mm::MAX_AUDIO_DURATION_SECONDS,
            // 30 minutes of stereo float32 PCM is about 230 MiB.
            max_encoded_bytes_per_clip: PHI4MM_MAX_ENCODED_BYTES,
            max_clips: DEFAULT_MAX_CLIPS,
            frame_length_samples: 400,
            frame_hop_samples: 160,
            // SpeechLib uses (samples - 400) / 160 + 1 at 16 kHz.
            max_frames_per_clip: 179_998,
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
            target_channels: 1,
            dtype: "f32",
            resampling: AudioResamplingPolicy::Linear,
            max_duration_seconds: 30,
            max_samples_per_clip: crate::audio::gemma3n::GEMMA3N_MAX_SAMPLES,
            max_encoded_bytes_per_clip: DEFAULT_MAX_ENCODED_BYTES,
            max_clips: DEFAULT_MAX_CLIPS,
            frame_length_samples: 513,
            frame_hop_samples: 160,
            max_frames_per_clip: 2_997,
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
        let official_max_duration = policy.max_duration_seconds;
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
            || policy.frame_length_samples == 0
            || policy.frame_hop_samples == 0
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
