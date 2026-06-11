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

//! Unit tests for the model-free pieces of the DiffusionGemma serving loop:
//! the server -> engine options mapping, audio/video rejection, sampler
//! parsing, and finish-reason mapping.

use super::*;
use crate::models::diffusion_gemma::{DiffusionFinishReason, DiffusionSamplerKind};

fn defaults() -> DiffusionServeDefaults {
    DiffusionServeDefaults {
        sampler: DiffusionSamplerKind::ConfidenceThreshold,
        confidence_threshold: 0.75,
        max_denoising_steps: Some(24),
    }
}

#[test]
fn build_diffusion_options_maps_core_fields_and_serve_defaults() {
    let opts = build_diffusion_options(128, 0.7, &[], &defaults(), &[]);
    assert_eq!(opts.max_new_tokens, 128);
    assert_eq!(opts.temperature, 0.7);
    assert_eq!(opts.sampler, DiffusionSamplerKind::ConfidenceThreshold);
    assert_eq!(opts.confidence_threshold, 0.75);
    assert_eq!(opts.max_denoising_steps, Some(24));
    // Canvas knobs keep their engine defaults (not exposed at serve level).
    assert_eq!(opts.min_canvas_length, 64);
    assert_eq!(opts.max_canvas_length, None);
    assert!(!opts.full_canvas);
    assert_eq!(opts.prefill_chunk_size, 512);
}

#[test]
fn build_diffusion_options_unions_stop_tokens_and_config_eos_dedup() {
    // Request stop tokens come first, then config EOS, with duplicates removed
    // while preserving first-seen order.
    let opts = build_diffusion_options(64, 0.0, &[7, 50], &defaults(), &[1, 50, 106]);
    assert_eq!(opts.extra_eos_token_ids, vec![7, 50, 1, 106]);
}

#[test]
fn build_diffusion_options_defaults_apply_with_empty_eos() {
    let opts = build_diffusion_options(32, 0.0, &[], &DiffusionServeDefaults::default(), &[]);
    assert!(opts.extra_eos_token_ids.is_empty());
    assert_eq!(opts.sampler, DiffusionSamplerKind::EntropyBound);
    assert_eq!(opts.confidence_threshold, 0.9);
    assert_eq!(opts.max_denoising_steps, None);
}

#[test]
fn reject_audio_video_flags_unsupported_modalities() {
    assert_eq!(reject_audio_video(false, false), None);
    assert_eq!(
        reject_audio_video(true, false),
        Some(AUDIO_VIDEO_UNSUPPORTED_MSG)
    );
    assert_eq!(
        reject_audio_video(false, true),
        Some(AUDIO_VIDEO_UNSUPPORTED_MSG)
    );
    assert_eq!(
        reject_audio_video(true, true),
        Some(AUDIO_VIDEO_UNSUPPORTED_MSG)
    );
}

#[test]
fn parse_diffusion_sampler_accepts_known_samplers() {
    assert_eq!(
        parse_diffusion_sampler("entropy-bound"),
        Ok(DiffusionSamplerKind::EntropyBound)
    );
    assert_eq!(
        parse_diffusion_sampler("confidence-threshold"),
        Ok(DiffusionSamplerKind::ConfidenceThreshold)
    );
    assert!(parse_diffusion_sampler("nonsense").is_err());
}

#[test]
fn finish_reason_maps_to_server_strings() {
    assert_eq!(
        diffusion_finish_reason_str(DiffusionFinishReason::Length),
        "length"
    );
    assert_eq!(
        diffusion_finish_reason_str(DiffusionFinishReason::Stop),
        "stop"
    );
    // An aborted (client-cancelled) run has no distinct server reason string;
    // it maps to "stop".
    assert_eq!(
        diffusion_finish_reason_str(DiffusionFinishReason::Aborted),
        "stop"
    );
}

#[test]
fn diffusion_serve_defaults_default_is_entropy_bound() {
    let d = DiffusionServeDefaults::default();
    assert_eq!(d.sampler, DiffusionSamplerKind::EntropyBound);
    assert_eq!(d.confidence_threshold, 0.9);
    assert_eq!(d.max_denoising_steps, None);
}
