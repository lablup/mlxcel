use super::*;

struct NeverCancel;

impl AudioCancellation for NeverCancel {
    fn is_cancelled(&self, _checkpoint: AudioPreprocessCheckpoint) -> bool {
        false
    }
}

struct CancelAt(AudioPreprocessCheckpoint);

impl AudioCancellation for CancelAt {
    fn is_cancelled(&self, checkpoint: AudioPreprocessCheckpoint) -> bool {
        checkpoint == self.0
    }
}

fn wav_f32(sample_rate: u32, channels: u16, interleaved: &[f32]) -> Vec<u8> {
    let data_len = interleaved.len() * 4;
    let mut wav = Vec::with_capacity(44 + data_len);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36u32 + data_len as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&3u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * channels as u32 * 4).to_le_bytes());
    wav.extend_from_slice(&(channels * 4).to_le_bytes());
    wav.extend_from_slice(&32u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in interleaved {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

fn clip(bytes: Vec<u8>, ordinal: usize) -> AudioEncodedClip {
    AudioEncodedClip {
        bytes,
        source: AudioSourceKind::ServerInline,
        placeholder_ordinal: ordinal,
    }
}

#[test]
fn deterministic_resampling_downmix_and_normalization_have_golden_values() {
    let encoded = clip(wav_f32(8_000, 2, &[2.0, 0.0, -2.0, 0.0, 0.5, -0.5]), 1);
    let first = preprocess_wav_batch(
        std::slice::from_ref(&encoded),
        AudioFamilyPolicy::gemma3n(),
        &NeverCancel,
    )
    .unwrap();
    let second = preprocess_wav_batch(
        std::slice::from_ref(&encoded),
        AudioFamilyPolicy::gemma3n(),
        &NeverCancel,
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.clips[0].samples, vec![1.0, 0.0, -1.0, -0.5, 0.0, 0.0]);
    assert_eq!(first.clips[0].sample_rate, 16_000);
    assert_eq!(first.clips[0].source_channels, 2);
}

#[test]
fn cli_file_and_server_bytes_produce_identical_normalized_waveforms() {
    let bytes = wav_f32(8_000, 2, &[0.25, -0.25, 1.5, 0.5, -1.5, -0.5, 0.125, 0.375]);
    let path = std::env::temp_dir().join(format!(
        "mlxcel-audio-contract-{}.wav",
        uuid::Uuid::new_v4()
    ));
    std::fs::write(&path, &bytes).unwrap();

    let cli = preprocess_wav_file(&path, AudioFamilyPolicy::gemma3n(), &NeverCancel).unwrap();
    let mut server = preprocess_wav_refs(
        &[AudioEncodedClipRef {
            bytes: &bytes,
            source: AudioSourceKind::ServerInline,
            placeholder_ordinal: 1,
        }],
        AudioFamilyPolicy::gemma3n(),
        &NeverCancel,
    )
    .unwrap();
    std::fs::remove_file(path).unwrap();

    // Source class is observability-only; every decoded/resampled byte and
    // boundary/accounting field must otherwise be identical.
    server.clips[0].source = AudioSourceKind::CliFile;
    assert_eq!(cli, server);
}

#[test]
fn cli_file_acquisition_checks_cancellation_between_bounded_chunks() {
    struct CancelAfterFirstRead(std::sync::atomic::AtomicUsize);
    impl AudioCancellation for CancelAfterFirstRead {
        fn is_cancelled(&self, checkpoint: AudioPreprocessCheckpoint) -> bool {
            checkpoint == AudioPreprocessCheckpoint::Acquisition
                && self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel) >= 2
        }
    }

    let bytes = wav_f32(16_000, 1, &vec![0.0; 32 * 1024]);
    let path =
        std::env::temp_dir().join(format!("mlxcel-audio-cancel-{}.wav", uuid::Uuid::new_v4()));
    std::fs::write(&path, bytes).unwrap();
    let error = preprocess_wav_file(
        &path,
        AudioFamilyPolicy::gemma3n(),
        &CancelAfterFirstRead(std::sync::atomic::AtomicUsize::new(0)),
    )
    .unwrap_err();
    std::fs::remove_file(path).unwrap();

    assert!(matches!(
        error,
        AudioPreprocessError::Cancelled {
            checkpoint: AudioPreprocessCheckpoint::Acquisition,
            clip_index: Some(0),
        }
    ));
}

#[test]
fn batch_preserves_clip_boundaries_order_and_placeholder_ordinals() {
    let batch = preprocess_wav_batch(
        &[
            clip(wav_f32(16_000, 1, &[0.25, 0.5]), 1),
            clip(wav_f32(16_000, 1, &[-0.5]), 2),
        ],
        AudioFamilyPolicy::phi4mm(),
        &NeverCancel,
    )
    .unwrap();
    assert_eq!(batch.clips[0].samples, [0.25, 0.5]);
    assert_eq!(batch.clips[1].samples, [-0.5]);
    assert_eq!(
        batch.boundaries,
        [
            AudioClipBoundary {
                clip_index: 0,
                placeholder_ordinal: 1,
                start_sample: 0,
                end_sample: 2,
            },
            AudioClipBoundary {
                clip_index: 1,
                placeholder_ordinal: 2,
                start_sample: 2,
                end_sample: 3,
            },
        ]
    );
}

#[test]
fn aggregate_caps_reject_before_waveform_allocation_and_do_not_multiply_per_clip_max() {
    let clips = [
        clip(wav_f32(16_000, 1, &[0.0; 6]), 1),
        clip(wav_f32(16_000, 1, &[0.0; 6]), 2),
    ];
    let mut policy = AudioFamilyPolicy::gemma3n();
    policy.max_normalized_samples_per_request = 10;
    let error = preprocess_wav_batch(&clips, policy, &NeverCancel).unwrap_err();
    assert!(matches!(
        error,
        AudioPreprocessError::Limit {
            limit: "normalized samples per request",
            actual: 12,
            maximum: 10,
        }
    ));

    let mut duration_policy = AudioFamilyPolicy::gemma3n();
    duration_policy.max_source_duration_micros_per_request = 500;
    let error = preprocess_wav_batch(&clips, duration_policy, &NeverCancel).unwrap_err();
    assert!(matches!(
        error,
        AudioPreprocessError::Limit {
            limit: "source duration micros per request",
            ..
        }
    ));
}

#[test]
fn source_rate_and_phi_polyphase_work_are_bounded_and_inner_loop_is_cancellable() {
    let too_fast = preprocess_wav_batch(
        &[clip(wav_f32(192_001, 1, &[0.0; 8]), 1)],
        AudioFamilyPolicy::phi4mm(),
        &NeverCancel,
    )
    .unwrap_err();
    assert!(matches!(
        too_fast,
        AudioPreprocessError::Limit {
            limit: "source sample rate",
            ..
        }
    ));

    struct CancelDuringPolyphase(std::sync::atomic::AtomicUsize);
    impl AudioCancellation for CancelDuringPolyphase {
        fn is_cancelled(&self, checkpoint: AudioPreprocessCheckpoint) -> bool {
            checkpoint == AudioPreprocessCheckpoint::Resample
                && self.0.fetch_add(1, std::sync::atomic::Ordering::AcqRel) >= 2
        }
    }
    let error = preprocess_wav_batch(
        &[clip(wav_f32(48_000, 1, &[0.0; 48_000]), 1)],
        AudioFamilyPolicy::phi4mm(),
        &CancelDuringPolyphase(std::sync::atomic::AtomicUsize::new(0)),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        AudioPreprocessError::Cancelled {
            checkpoint: AudioPreprocessCheckpoint::Resample,
            ..
        }
    ));
}

#[test]
fn malformed_nonfinite_oversize_and_placeholder_fail_typed() {
    let corrupt = preprocess_wav_batch(
        &[clip(b"not wav".to_vec(), 1)],
        AudioFamilyPolicy::gemma3n(),
        &NeverCancel,
    )
    .unwrap_err();
    assert!(matches!(corrupt, AudioPreprocessError::Corrupt { .. }));

    let nonfinite = preprocess_wav_batch(
        &[clip(wav_f32(16_000, 1, &[f32::NAN]), 1)],
        AudioFamilyPolicy::gemma3n(),
        &NeverCancel,
    )
    .unwrap_err();
    assert!(matches!(nonfinite, AudioPreprocessError::NonFinite { .. }));

    let mut tiny_policy = AudioFamilyPolicy::gemma3n();
    tiny_policy.max_encoded_bytes_per_clip = 8;
    assert!(matches!(
        preprocess_wav_batch(
            &[clip(wav_f32(16_000, 1, &[0.0]), 1)],
            tiny_policy,
            &NeverCancel
        ),
        Err(AudioPreprocessError::Limit {
            limit: "encoded bytes per clip",
            ..
        })
    ));
    assert!(matches!(
        preprocess_wav_batch(
            &[clip(wav_f32(16_000, 1, &[0.0]), 2)],
            AudioFamilyPolicy::gemma3n(),
            &NeverCancel
        ),
        Err(AudioPreprocessError::Placeholder { .. })
    ));
}

#[test]
fn acquisition_decode_and_resample_are_independently_cancellable() {
    let encoded = clip(wav_f32(8_000, 1, &[0.0; 8_000]), 1);
    for checkpoint in [
        AudioPreprocessCheckpoint::Acquisition,
        AudioPreprocessCheckpoint::Decode,
        AudioPreprocessCheckpoint::Resample,
    ] {
        let error = preprocess_wav_batch(
            std::slice::from_ref(&encoded),
            AudioFamilyPolicy::gemma3n(),
            &CancelAt(checkpoint),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            AudioPreprocessError::Cancelled {
                checkpoint: actual,
                ..
            } if actual == checkpoint
        ));
    }
}

#[test]
fn family_policies_pin_waveform_feature_and_placeholder_contracts() {
    let phi = AudioFamilyPolicy::phi4mm();
    assert_eq!(
        (
            phi.target_sample_rate,
            phi.target_channels,
            phi.dtype,
            phi.max_duration_seconds,
            phi.placeholder,
        ),
        (
            16_000,
            1,
            "f32",
            1_800,
            AudioPlaceholderPolicy::NumberedPerClip
        )
    );
    let gemma = AudioFamilyPolicy::gemma3n();
    assert_eq!(gemma.max_duration_seconds, 30);
    assert_eq!(gemma.max_frames_per_clip, 2_997);
    assert_eq!(
        gemma.placeholder,
        AudioPlaceholderPolicy::FixedSoftTokensPerClip(188)
    );
    assert!(matches!(
        gemma.source,
        AudioPolicySource::PinnedOfficialDefault(
            crate::audio::gemma3n::GEMMA3N_TRANSFORMERS_REFERENCE_REVISION
        )
    ));
}

#[test]
fn model_processor_config_overrides_bounds_but_rejects_frontend_drift() {
    let model = serde_json::json!({
        "audio_soft_tokens_per_image": 188,
        "mlxcel_audio_limits": {
            "max_encoded_bytes_per_clip": 1024,
            "max_clips": 2
        }
    });
    let processor = serde_json::json!({
        "feature_extractor": {
            "sampling_rate": 16_000,
            "max_length": 160_000,
            "frame_length": 513,
            "hop_length": 160,
            "channels": 1
        }
    });
    let policy = AudioFamilyPolicy::from_gemma3n_configs(&model, Some(&processor), 188).unwrap();
    assert_eq!(policy.max_samples_per_clip, 160_000);
    assert_eq!(policy.max_duration_seconds, 10);
    assert_eq!(policy.max_encoded_bytes_per_clip, 1024);
    assert_eq!(policy.max_clips, 2);
    assert_eq!(policy.source, AudioPolicySource::ModelProcessorConfig);

    let incompatible = serde_json::json!({
        "feature_extractor": { "sampling_rate": 24_000 }
    });
    assert!(AudioFamilyPolicy::from_gemma3n_configs(&model, Some(&incompatible), 188).is_err());

    let duration_only = serde_json::json!({
        "feature_extractor": {
            "sampling_rate": 16_000,
            "max_audio_duration": 5
        }
    });
    let duration_policy =
        AudioFamilyPolicy::from_gemma3n_configs(&model, Some(&duration_only), 188).unwrap();
    assert_eq!(duration_policy.max_duration_seconds, 5);
    assert_eq!(duration_policy.max_samples_per_clip, 80_000);
}

#[test]
fn malformed_known_policy_keys_fail_closed_with_typed_startup_errors() {
    let model = serde_json::json!({});
    for processor in [
        serde_json::json!({
            "feature_extractor": { "sampling_rate": "16000" }
        }),
        serde_json::json!({
            "feature_extractor": { "max_length": -1 }
        }),
        serde_json::json!({
            "feature_extractor": { "max_audio_duration": "30" }
        }),
        serde_json::json!({
            "mlxcel_audio_limits": { "resampling_algorithm": 42 }
        }),
    ] {
        let error =
            AudioFamilyPolicy::from_gemma3n_configs(&model, Some(&processor), 188).unwrap_err();
        assert!(
            matches!(error, AudioPreprocessError::Context { clip_index: 0, .. }),
            "malformed loaded config must be a typed startup error: {error}"
        );
    }

    let beyond_pinned = serde_json::json!({
        "feature_extractor": {
            "sampling_rate": 16_000,
            "max_audio_duration": 31
        }
    });
    assert!(AudioFamilyPolicy::from_gemma3n_configs(&model, Some(&beyond_pinned), 188).is_err());

    let aggregate_beyond_pinned = serde_json::json!({
        "mlxcel_audio_limits": {
            "max_normalized_samples_per_request":
                AudioFamilyPolicy::gemma3n().max_normalized_samples_per_request + 1
        }
    });
    assert!(AudioFamilyPolicy::from_gemma3n_configs(&aggregate_beyond_pinned, None, 188).is_err());
}

#[test]
fn missing_processor_metadata_uses_revision_pinned_official_defaults() {
    let phi = AudioFamilyPolicy::from_phi4mm_configs(&serde_json::json!({}), None).unwrap();
    assert_eq!(phi, AudioFamilyPolicy::phi4mm());
    let gemma = AudioFamilyPolicy::from_gemma3n_configs(
        &serde_json::json!({}),
        None,
        crate::audio::gemma3n::GEMMA3N_AUDIO_SOFT_TOKENS,
    )
    .unwrap();
    assert_eq!(gemma, AudioFamilyPolicy::gemma3n());
}
