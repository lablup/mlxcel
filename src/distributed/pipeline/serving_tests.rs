// Copyright 2025-2026 Lablup Inc.
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

use super::*;
use crate::distributed::request_tracker::RequestId;
use std::time::Duration;

// ---------------------------------------------------------------------------
// StageRole tests
// ---------------------------------------------------------------------------

#[test]
fn stage_role_from_index_single() {
    assert_eq!(StageRole::from_index(0, 1), StageRole::SingleStage);
}

#[test]
fn stage_role_from_index_two_stages() {
    assert_eq!(StageRole::from_index(0, 2), StageRole::First);
    assert_eq!(StageRole::from_index(1, 2), StageRole::Last);
}

#[test]
fn stage_role_from_index_three_stages() {
    assert_eq!(StageRole::from_index(0, 3), StageRole::First);
    assert_eq!(StageRole::from_index(1, 3), StageRole::Middle);
    assert_eq!(StageRole::from_index(2, 3), StageRole::Last);
}

#[test]
fn stage_role_entry_point() {
    assert!(StageRole::First.is_entry_point());
    assert!(StageRole::SingleStage.is_entry_point());
    assert!(!StageRole::Middle.is_entry_point());
    assert!(!StageRole::Last.is_entry_point());
}

#[test]
fn stage_role_produces_tokens() {
    assert!(StageRole::Last.produces_tokens());
    assert!(StageRole::SingleStage.produces_tokens());
    assert!(!StageRole::First.produces_tokens());
    assert!(!StageRole::Middle.produces_tokens());
}

// ---------------------------------------------------------------------------
// PipelineServingConfig tests
// ---------------------------------------------------------------------------

#[test]
fn serving_config_valid() {
    let config = PipelineServingConfig::new(3, 1).unwrap();
    assert_eq!(config.role(), StageRole::Middle);
    assert!(config.is_pipeline_active());
    config.validate().unwrap();
}

#[test]
fn serving_config_single_stage() {
    let config = PipelineServingConfig::new(1, 0).unwrap();
    assert_eq!(config.role(), StageRole::SingleStage);
    assert!(!config.is_pipeline_active());
}

#[test]
fn serving_config_out_of_range() {
    let err = PipelineServingConfig::new(3, 5);
    assert!(err.is_err());
}

#[test]
fn serving_config_builders() {
    let config = PipelineServingConfig::new(2, 0)
        .unwrap()
        .with_timeout(Duration::from_secs(60))
        .with_max_in_flight(128)
        .with_micro_batch_size(4)
        .with_prefill_chunk_size(256);

    assert_eq!(config.stage_timeout, Duration::from_secs(60));
    assert_eq!(config.max_in_flight, 128);
    assert_eq!(config.micro_batch_size, 4);
    assert_eq!(config.prefill_chunk_size, 256);
}

// ---------------------------------------------------------------------------
// PipelineRequest tests
// ---------------------------------------------------------------------------

#[test]
fn pipeline_request_new() {
    let req = PipelineRequest::new(RequestId::new(), 1, vec![1, 2, 3, 4, 5], 100);
    assert_eq!(req.token_ids.len(), 5);
    assert_eq!(req.remaining_prefill_tokens(), 5);
    assert!(!req.is_prefill_complete());
    assert!(!req.is_decoding);
}

#[test]
fn pipeline_request_prefill_progress() {
    let mut req = PipelineRequest::new(RequestId::new(), 1, vec![1, 2, 3, 4, 5], 100);
    req.prefill_offset = 3;
    assert_eq!(req.remaining_prefill_tokens(), 2);
    assert!(!req.is_prefill_complete());

    req.prefill_offset = 5;
    assert_eq!(req.remaining_prefill_tokens(), 0);
    assert!(req.is_prefill_complete());
}

// ---------------------------------------------------------------------------
// PipelineResponse tests
// ---------------------------------------------------------------------------

#[test]
fn pipeline_response_success() {
    let resp = PipelineResponse::success(
        RequestId::new(),
        1,
        vec![10, 20, 30],
        true,
        Duration::from_millis(50),
    );
    assert!(!resp.is_error());
    assert!(resp.is_finished);
    assert_eq!(resp.generated_tokens.len(), 3);
}

#[test]
fn pipeline_response_error() {
    let resp = PipelineResponse::error(RequestId::new(), 1, "stage 2 timed out".to_string());
    assert!(resp.is_error());
    assert!(resp.is_finished);
    assert!(resp.generated_tokens.is_empty());
}

// ---------------------------------------------------------------------------
// StageHealth tests
// ---------------------------------------------------------------------------

#[test]
fn stage_health_usable() {
    assert!(StageHealth::Healthy.is_usable());
    assert!(StageHealth::Degraded.is_usable());
    assert!(!StageHealth::Failed.is_usable());
    assert!(!StageHealth::Unknown.is_usable());
}

// ---------------------------------------------------------------------------
// PipelineCoordinator tests
// ---------------------------------------------------------------------------

#[test]
fn coordinator_submit_and_stage_output() {
    let config = PipelineServingConfig::new(3, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1, 2, 3], 10);
    let req_id = req.request_id.clone();

    let mut rx = coord.submit_request(req).unwrap();
    assert_eq!(coord.in_flight_count(), 1);
    assert!(coord.can_accept());

    // Process through stages 0, 1, and final token from stage 2.
    coord.process_stage_output(&req_id, 0, None, false).unwrap();
    coord.process_stage_output(&req_id, 1, None, false).unwrap();
    coord
        .process_stage_output(&req_id, 2, Some(42), true)
        .unwrap();

    // Response should be delivered.
    let resp = rx.try_recv().unwrap();
    assert!(!resp.is_error());
    assert!(resp.is_finished);
    assert_eq!(resp.generated_tokens, vec![42]);
    assert_eq!(coord.in_flight_count(), 0);
}

#[test]
fn coordinator_capacity_limit() {
    let config = PipelineServingConfig::new(2, 0)
        .unwrap()
        .with_max_in_flight(2);
    let mut coord = PipelineCoordinator::new(config).unwrap();

    // Fill capacity.
    let _rx1 = coord
        .submit_request(PipelineRequest::new(RequestId::new(), 0, vec![1], 10))
        .unwrap();
    let _rx2 = coord
        .submit_request(PipelineRequest::new(RequestId::new(), 0, vec![2], 10))
        .unwrap();

    // Third should fail.
    let result = coord.submit_request(PipelineRequest::new(RequestId::new(), 0, vec![3], 10));
    assert!(result.is_err());
    assert!(!coord.can_accept());
}

#[test]
fn coordinator_stage_failure() {
    let config = PipelineServingConfig::new(3, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1], 10);
    let mut rx = coord.submit_request(req).unwrap();

    // Fail stage 0.
    let failed = coord.handle_stage_failure(0);
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].failed_stage, 0);

    // Response channel should have an error.
    let resp = rx.try_recv().unwrap();
    assert!(resp.is_error());

    assert_eq!(coord.stage_health(0), StageHealth::Failed);
    assert!(!coord.all_stages_usable());
}

#[test]
fn coordinator_health_update() {
    let config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    assert_eq!(coord.stage_health(0), StageHealth::Unknown);
    coord.update_stage_health(0, StageHealth::Healthy);
    assert_eq!(coord.stage_health(0), StageHealth::Healthy);

    coord.update_stage_health(1, StageHealth::Degraded);
    assert!(coord.all_stages_usable());

    coord.update_stage_health(1, StageHealth::Failed);
    assert!(!coord.all_stages_usable());
}

#[test]
fn coordinator_allocate_sequence_id() {
    let config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let id1 = coord.allocate_sequence_id();
    let id2 = coord.allocate_sequence_id();
    assert_eq!(id1 + 1, id2);
}

#[test]
fn coordinator_begin_drain_blocks_new_requests() {
    let config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1, 2, 3], 10);
    let _rx = coord.submit_request(req).unwrap();
    coord.begin_drain();

    assert!(coord.is_draining());
    assert!(!coord.drain_complete());
    assert!(!coord.can_accept());
    let err = coord
        .submit_request(PipelineRequest::new(RequestId::new(), 0, vec![9], 4))
        .unwrap_err();
    assert!(err.to_string().contains("draining"));
}

#[test]
fn coordinator_shutdown_fails_inflight_requests_and_completes_drain() {
    let config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1, 2, 3], 10);
    let mut rx = coord.submit_request(req).unwrap();

    let failed = coord.shutdown("pipeline shutdown");
    assert_eq!(failed.len(), 1);
    assert!(coord.is_draining());
    assert!(coord.drain_complete());
    assert_eq!(coord.in_flight_count(), 0);

    let response = rx.try_recv().expect("shutdown response");
    assert!(response.is_error());
    assert_eq!(response.error.as_deref(), Some("pipeline shutdown"));
}

#[test]
fn coordinator_multi_token_accumulation() {
    // Verify that tokens accumulate across multiple decode steps and the
    // response is only delivered when is_finished becomes true.
    let config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1, 2, 3], 10);
    let req_id = req.request_id.clone();
    let mut rx = coord.submit_request(req).unwrap();

    // First decode step: token from last stage, not finished.
    coord
        .process_stage_output(&req_id, 1, Some(10), false)
        .unwrap();
    // Should NOT be delivered yet.
    assert!(rx.try_recv().is_err());
    assert_eq!(coord.in_flight_count(), 1);

    // Second decode step.
    coord
        .process_stage_output(&req_id, 1, Some(20), false)
        .unwrap();
    assert!(rx.try_recv().is_err());

    // Third decode step: finished.
    coord
        .process_stage_output(&req_id, 1, Some(30), true)
        .unwrap();

    let resp = rx.try_recv().unwrap();
    assert!(!resp.is_error());
    assert!(resp.is_finished);
    assert_eq!(resp.generated_tokens, vec![10, 20, 30]);
    assert_eq!(coord.in_flight_count(), 0);
}

#[test]
fn coordinator_duplicate_request_id_rejected() {
    let config = PipelineServingConfig::new(2, 0).unwrap();
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let rid = RequestId::new();
    let req1 = PipelineRequest::new(rid.clone(), 0, vec![1], 10);
    let req2 = PipelineRequest::new(rid, 0, vec![2], 10);

    let _rx1 = coord.submit_request(req1).unwrap();
    let result = coord.submit_request(req2);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("duplicate request ID")
    );
}

#[test]
fn coordinator_enforce_timeouts() {
    let config = PipelineServingConfig::new(2, 0)
        .unwrap()
        .with_timeout(Duration::from_millis(1));
    let mut coord = PipelineCoordinator::new(config).unwrap();

    let req = PipelineRequest::new(RequestId::new(), 0, vec![1], 10);
    let mut rx = coord.submit_request(req).unwrap();

    // Wait for the timeout to expire.
    std::thread::sleep(Duration::from_millis(5));

    let timed_out = coord.enforce_timeouts();
    assert_eq!(timed_out.len(), 1);
    assert!(timed_out[0].reason.contains("timed out"));

    let resp = rx.try_recv().unwrap();
    assert!(resp.is_error());
    assert_eq!(coord.in_flight_count(), 0);
}

// ---------------------------------------------------------------------------
// ChunkedPrefillPipeline tests
// ---------------------------------------------------------------------------

#[test]
fn chunked_prefill_basic() {
    let mut prefill = ChunkedPrefillPipeline::new(3);
    let req = PipelineRequest::new(RequestId::new(), 1, vec![1, 2, 3, 4, 5, 6, 7, 8], 100);

    let (start, end) = prefill.begin_prefill(&req);
    assert_eq!((start, end), (0, 3));
    assert!(prefill.is_prefilling(&req.request_id));
    assert_eq!(prefill.progress(&req.request_id), Some((0, 8)));

    // Advance: processed 3 tokens.
    let next = prefill.advance_prefill(&req.request_id, 3);
    assert_eq!(next, Some((3, 6)));
    assert_eq!(prefill.progress(&req.request_id), Some((3, 8)));

    // Advance: processed 3 more tokens.
    let next = prefill.advance_prefill(&req.request_id, 3);
    assert_eq!(next, Some((6, 8)));

    // Advance: processed final 2 tokens.
    let next = prefill.advance_prefill(&req.request_id, 2);
    assert_eq!(next, None); // Prefill complete.
    assert!(!prefill.is_prefilling(&req.request_id));
}

#[test]
fn chunked_prefill_disabled() {
    let mut prefill = ChunkedPrefillPipeline::new(0); // 0 = disabled
    let req = PipelineRequest::new(RequestId::new(), 1, vec![1, 2, 3, 4, 5], 100);

    let (start, end) = prefill.begin_prefill(&req);
    assert_eq!((start, end), (0, 5)); // Full prompt in one chunk.

    let next = prefill.advance_prefill(&req.request_id, 5);
    assert_eq!(next, None); // Done.
}

#[test]
fn chunked_prefill_cancel() {
    let mut prefill = ChunkedPrefillPipeline::new(2);
    let req = PipelineRequest::new(RequestId::new(), 1, vec![1, 2, 3, 4], 100);

    let _ = prefill.begin_prefill(&req);
    assert_eq!(prefill.active_sessions(), 1);

    prefill.cancel_prefill(&req.request_id);
    assert_eq!(prefill.active_sessions(), 0);
    assert!(!prefill.is_prefilling(&req.request_id));
}

// ---------------------------------------------------------------------------
// API compatibility helper tests
// ---------------------------------------------------------------------------

#[test]
fn detect_pipeline_config_single_stage() {
    let config = detect_pipeline_config(1, 0, 30, 512);
    assert!(config.is_none());
}

#[test]
fn detect_pipeline_config_multi_stage() {
    let config = detect_pipeline_config(3, 1, 30, 512);
    assert!(config.is_some());
    let config = config.unwrap();
    assert_eq!(config.num_stages, 3);
    assert_eq!(config.stage_index, 1);
    assert_eq!(config.prefill_chunk_size, 512);
}

#[test]
fn should_use_pipeline_none() {
    assert!(!should_use_pipeline(None));
}

#[test]
fn should_use_pipeline_active() {
    let config = PipelineServingConfig::new(3, 0).unwrap();
    assert!(should_use_pipeline(Some(&config)));
}

#[test]
fn should_use_pipeline_single() {
    let config = PipelineServingConfig::new(1, 0).unwrap();
    assert!(!should_use_pipeline(Some(&config)));
}

#[test]
fn to_schedule_config() {
    let config = PipelineServingConfig::new(3, 0)
        .unwrap()
        .with_micro_batch_size(4);
    let sched = to_pipeline_schedule_config(&config).unwrap();
    assert_eq!(sched.num_stages, 3);
    assert_eq!(sched.micro_batch_size, 4);
}
