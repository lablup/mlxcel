# Audio input preprocessing

This document describes the shared request boundary for model input audio. It
does not cover the separate `/v1/audio` speech-to-text and text-to-speech API;
see [Audio API](audio-api.md) for that surface.

## Waveform and policy contract

CLI files and server-resolved WAV bytes use the same strict RIFF/WAVE decoder.
The decoder accepts PCM16 or float32, rejects empty, malformed, non-finite, or
out-of-policy input, averages channels to mono, and clamps samples to finite
`f32` values in `[-1, 1]`. Clip order and one-based placeholder ordinals are
preserved. The owned result records source rate, channels, sample count,
duration, encoded bytes, normalized sample count, feature-frame estimate, and
effective audio-token count separately.

The loaded checkpoint policy is derived from `config.json` plus
`processor_config.json` or `preprocessor_config.json`. Recognized keys are:

- sample rate: `audio_processor.sampling_rate`, `audio_processor.sample_rate`,
  `feature_extractor.sampling_rate`, or top-level `sampling_rate`;
- sample/duration limits: `audio_processor.max_audio_samples`,
  `feature_extractor.max_length`, top-level `max_audio_samples`, and the
  corresponding `max_audio_duration` or `max_duration_seconds` keys;
- frame shape: `audio_processor.frame_length_samples`,
  `audio_processor.frame_hop_samples`, `feature_extractor.frame_length`, and
  `feature_extractor.hop_length`;
- channel/dtype: `audio_processor.channels`, `audio_processor.dtype`,
  `feature_extractor.channels`, and `feature_extractor.dtype`;
- mlxcel bounds: `mlxcel_audio_limits.max_encoded_bytes_per_clip`,
  `mlxcel_audio_limits.max_clips`, and
  `mlxcel_audio_limits.resampling_algorithm`.

Known keys with the wrong JSON type, a negative integer, zero bound, unsupported
frontend shape, or a value beyond the pinned family maximum fail model startup.
Missing metadata uses immutable reference defaults:

| Family | Pinned defaults |
| --- | --- |
| Phi-4 Multimodal | Transformers revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`; mono float32; the published SpeechLib integer-ratio resampler; 30 minutes / 28,800,000 normalized samples; 256 MiB encoded per clip; 16 clips; numbered placeholders. |
| Gemma 3n | Transformers reference revision `181beb3ba4c47098ed8cbc97ee250d1d45ae0107`; mono float32 at 16 kHz with deterministic linear resampling; 30 seconds / 480,000 samples; 64 MiB encoded per clip; 16 clips; 188 soft tokens per clip. |

The Phi policy deliberately preserves the published processor's unusual
integer-ratio behavior. It is not replaced by the generic linear resampler.

## Bounds and cancellation

Server acquisition first enforces an absolute ceiling of 16 clips and 500 MiB
of encoded audio per clip and per request. Each later source is read against
the remaining request budget, so URL `Content-Length` and base64 character
preflights can reject before allocating beyond the aggregate ceiling. After the
model family is known, its stricter loaded encoded-byte, duration, sample,
frame, clip, and queued-memory limits apply before feature allocation.

Invalid audio is request-fatal rather than silently dropped into a text-only
request. Existing tolerant image resolution behavior is unchanged.

Before scheduler admission, production HTTP cancellation follows Axum handler
future lifetime: dropping the handler drops the acquisition future, partial
buffer, and remote response stream. The explicit chunk checkpoint probe is a
cooperative seam for a future XLA family admission path; it is not currently a
live disconnect token. The bounded host worker also has distinct queue,
decode, resample, and feature cancellation checkpoints and releases queued-byte
reservations on dequeue, rejection, disconnect, cancellation, and drop.

## Metrics and XLA status

`/metrics` exports cumulative source seconds, source samples, normalized
samples, feature frames, internal audio tokens, effective prepared-prefill
positions, preprocessing latency, cancellations, and current queued bytes.
`mlxcel_audio_preprocess_rejections_total` uses mutually exclusive `reason`
labels: `queue_full`, `memory_limit`, `worker_unavailable`, `overflow`,
`waveform`, `feature`, `feature_panic`, and `context_limit`. These internal
counters do not change public OpenAI usage-token accounting.

The OpenXLA/XLA backend still reports audio capability as unavailable and
rejects audio and video declarations before text fallback. The host waveform,
queue, cancellation, and observability foundation is present, but no
Phi-4 Multimodal or Gemma 3n XLA feature producer is implemented or wired yet.
