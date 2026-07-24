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

use std::fmt::Write;

use crate::server::batch::ObservabilitySnapshot;

pub(super) fn append_audio_preprocess_metrics(body: &mut String, snap: &ObservabilitySnapshot) {
    let _ = writeln!(
        body,
        "# HELP mlxcel_audio_source_seconds_total Source audio duration accepted by preprocessing\n\
         # TYPE mlxcel_audio_source_seconds_total counter\n\
         mlxcel_audio_source_seconds_total {:.6}\n\
         # HELP mlxcel_audio_source_samples_total Source mono frames decoded before resampling\n\
         # TYPE mlxcel_audio_source_samples_total counter\n\
         mlxcel_audio_source_samples_total {}\n\
         # HELP mlxcel_audio_normalized_samples_total Normalized mono frames after family resampling\n\
         # TYPE mlxcel_audio_normalized_samples_total counter\n\
         mlxcel_audio_normalized_samples_total {}\n\
         # HELP mlxcel_audio_feature_frames_total Family feature frames estimated\n\
         # TYPE mlxcel_audio_feature_frames_total counter\n\
         mlxcel_audio_feature_frames_total {}\n\
         # HELP mlxcel_audio_effective_tokens_total Internal audio soft-token positions\n\
         # TYPE mlxcel_audio_effective_tokens_total counter\n\
         mlxcel_audio_effective_tokens_total {}\n\
         # HELP mlxcel_audio_effective_prefill_tokens_total Final prepared-prefill sequence positions for audio requests\n\
         # TYPE mlxcel_audio_effective_prefill_tokens_total counter\n\
         mlxcel_audio_effective_prefill_tokens_total {}\n\
         # HELP mlxcel_audio_preprocess_latency_seconds_total Host audio preprocessing latency\n\
         # TYPE mlxcel_audio_preprocess_latency_seconds_total counter\n\
         mlxcel_audio_preprocess_latency_seconds_total {:.6}\n\
         # HELP mlxcel_audio_preprocess_rejections_total Audio preprocessing rejections\n\
         # TYPE mlxcel_audio_preprocess_rejections_total counter\n\
         mlxcel_audio_preprocess_rejections_total{{reason=\"queue_full\"}} {}\n\
         mlxcel_audio_preprocess_rejections_total{{reason=\"memory_limit\"}} {}\n\
         mlxcel_audio_preprocess_rejections_total{{reason=\"worker_unavailable\"}} {}\n\
         mlxcel_audio_preprocess_rejections_total{{reason=\"overflow\"}} {}\n\
         mlxcel_audio_preprocess_rejections_total{{reason=\"waveform\"}} {}\n\
         mlxcel_audio_preprocess_rejections_total{{reason=\"feature\"}} {}\n\
         mlxcel_audio_preprocess_rejections_total{{reason=\"feature_panic\"}} {}\n\
         mlxcel_audio_preprocess_rejections_total{{reason=\"context_limit\"}} {}\n\
         # HELP mlxcel_audio_preprocess_cancelled_total Cancelled audio preprocessing jobs\n\
         # TYPE mlxcel_audio_preprocess_cancelled_total counter\n\
         mlxcel_audio_preprocess_cancelled_total {}\n\
         # HELP mlxcel_audio_preprocess_queued_bytes Encoded audio bytes retained across queued, processing, and result-handoff states\n\
         # TYPE mlxcel_audio_preprocess_queued_bytes gauge\n\
         mlxcel_audio_preprocess_queued_bytes {}\n\
         # HELP mlxcel_audio_preprocess_inflight_host_bytes Host bytes reserved across queued, processing, and result handoff\n\
         # TYPE mlxcel_audio_preprocess_inflight_host_bytes gauge\n\
         mlxcel_audio_preprocess_inflight_host_bytes {}",
        snap.audio_source_duration_micros as f64 / 1_000_000.0,
        snap.audio_source_samples,
        snap.audio_normalized_samples,
        snap.audio_feature_frames,
        snap.audio_effective_tokens,
        snap.audio_effective_prefill_tokens,
        snap.audio_preprocess_latency_micros as f64 / 1_000_000.0,
        snap.audio_reject_queue_full,
        snap.audio_reject_memory_limit,
        snap.audio_reject_worker_unavailable,
        snap.audio_reject_overflow,
        snap.audio_reject_waveform,
        snap.audio_reject_feature,
        snap.audio_reject_feature_panic,
        snap.audio_reject_context_limit,
        snap.audio_preprocess_cancelled,
        snap.audio_preprocess_queued_bytes,
        snap.audio_preprocess_inflight_host_bytes,
    );
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::server::batch::BatchObservability;

    #[test]
    fn source_normalized_prefill_and_reasons_remain_distinct() {
        let observability = BatchObservability::new();
        observability
            .audio_source_samples
            .store(800, Ordering::Relaxed);
        observability
            .audio_normalized_samples
            .store(1_600, Ordering::Relaxed);
        observability
            .audio_effective_tokens
            .store(188, Ordering::Relaxed);
        observability
            .audio_effective_prefill_tokens
            .store(203, Ordering::Relaxed);
        observability
            .audio_preprocess_rejections
            .store(2, Ordering::Relaxed);
        observability
            .audio_reject_memory_limit
            .store(1, Ordering::Relaxed);
        observability
            .audio_reject_feature_panic
            .store(1, Ordering::Relaxed);

        let mut body = String::new();
        append_audio_preprocess_metrics(&mut body, &observability.snapshot());
        assert!(body.contains("mlxcel_audio_source_samples_total 800"));
        assert!(body.contains("mlxcel_audio_normalized_samples_total 1600"));
        assert!(body.contains("mlxcel_audio_effective_tokens_total 188"));
        assert!(body.contains("mlxcel_audio_effective_prefill_tokens_total 203"));
        assert!(
            body.contains("mlxcel_audio_preprocess_rejections_total{reason=\"memory_limit\"} 1")
        );
        assert!(
            body.contains("mlxcel_audio_preprocess_rejections_total{reason=\"feature_panic\"} 1")
        );
        assert!(body.contains("mlxcel_audio_preprocess_rejections_total{reason=\"queue_full\"} 0"));
        assert!(
            body.contains("mlxcel_audio_preprocess_rejections_total{reason=\"context_limit\"} 0")
        );
        assert!(
            !body.contains("reason=\"all\""),
            "reason labels must be exclusive so summing the family does not double-count"
        );
    }
}
