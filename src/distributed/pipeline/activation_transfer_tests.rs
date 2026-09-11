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

use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::distributed::mock_transport::{MockRouter, MockTransport, MockTransportConfig};
use crate::distributed::request_tracker::RequestId;
use crate::distributed::tensor_protocol::TensorDtype;

/// Helper: create a small dummy wire-format tensor (4 float32 elements).
fn dummy_tensor_bytes() -> Vec<u8> {
    let data: Vec<u8> = vec![0u8; 16]; // 4 x float32
    let shape = [2, 2];
    ActivationMessage::serialize_activation(TensorDtype::Float32, &shape.map(|s| s as u64), &data)
        .expect("serialize dummy tensor")
}

fn dummy_forward_msg(stage: u32, mb: u32) -> ActivationMessage {
    ActivationMessage::forward(
        RequestId::new(),
        mb,
        stage,
        3, // 3-stage pipeline
        dummy_tensor_bytes(),
        None,
        None,
        4,
    )
}

fn dummy_reverse_msg(stage: u32, mb: u32) -> ActivationMessage {
    ActivationMessage::reverse(RequestId::new(), mb, stage, 3, dummy_tensor_bytes(), 4)
}

// --- ActivationMessage ---

#[test]
fn test_forward_message_fields() {
    let msg = dummy_forward_msg(0, 1);
    assert!(!msg.is_reverse_path);
    assert_eq!(msg.stage_index, 0);
    assert_eq!(msg.micro_batch_id, 1);
    assert_eq!(msg.num_stages, 3);
    assert_eq!(msg.seq_len, 4);
    assert!(msg.payload_size() > 0);
    assert!(msg.attention_mask.is_none());
    assert!(msg.position_ids.is_none());
}

#[test]
fn test_reverse_message_fields() {
    let msg = dummy_reverse_msg(2, 0);
    assert!(msg.is_reverse_path);
    assert_eq!(msg.stage_index, 2);
    assert!(msg.attention_mask.is_none());
    assert!(msg.position_ids.is_none());
}

#[test]
fn test_serialize_deserialize_activation() {
    let data: Vec<u8> = vec![0u8; 8]; // 2 x float32
    let shape = [2u64];
    let wire = ActivationMessage::serialize_activation(TensorDtype::Float32, &shape, &data)
        .expect("serialize");
    let tensor = ActivationMessage::deserialize_activation(&wire).expect("deserialize");
    assert_eq!(tensor.dtype, TensorDtype::Float32);
    assert_eq!(tensor.shape, vec![2u64]);
    assert_eq!(tensor.data.len(), 8);
}

#[test]
fn test_message_display() {
    let msg = dummy_forward_msg(1, 3);
    let s = format!("{msg}");
    assert!(s.contains("forward"));
    assert!(s.contains("stage=1"));
    assert!(s.contains("mb=3"));
}

#[test]
fn test_payload_size_with_mask() {
    let tensor = dummy_tensor_bytes();
    let mask = vec![1u8; 10];
    let pos = vec![2u8; 20];
    let msg = ActivationMessage {
        request_id: RequestId::new(),
        micro_batch_id: 0,
        stage_index: 0,
        num_stages: 2,
        tensor_data: tensor.clone(),
        attention_mask: Some(mask.clone()),
        position_ids: Some(pos.clone()),
        is_reverse_path: false,
        seq_len: 1,
        timestamp_ns: 0,
    };
    assert_eq!(msg.payload_size(), tensor.len() + mask.len() + pos.len());
}

// --- Validation ---

#[test]
fn test_validate_good_forward() {
    let msg = dummy_forward_msg(0, 0);
    validate_activation(&msg).expect("should be valid");
}

#[test]
fn test_validate_good_reverse() {
    let msg = dummy_reverse_msg(2, 0);
    validate_activation(&msg).expect("should be valid");
}

#[test]
fn test_validate_too_few_stages() {
    let mut msg = dummy_forward_msg(0, 0);
    msg.num_stages = 1;
    assert!(validate_activation(&msg).is_err());
}

#[test]
fn test_validate_stage_out_of_range() {
    let mut msg = dummy_forward_msg(0, 0);
    msg.stage_index = 5;
    assert!(validate_activation(&msg).is_err());
}

#[test]
fn test_validate_empty_tensor() {
    let mut msg = dummy_forward_msg(0, 0);
    msg.tensor_data.clear();
    assert!(validate_activation(&msg).is_err());
}

#[test]
fn test_validate_forward_from_last_stage() {
    let msg = dummy_forward_msg(2, 0);
    // stage_index = num_stages - 1 on forward path
    assert!(validate_activation(&msg).is_err());
}

#[test]
fn test_validate_reverse_from_stage_zero() {
    let msg = dummy_reverse_msg(0, 0);
    // dummy_reverse_msg already sets is_reverse_path=true and stage_index=0
    assert!(validate_activation(&msg).is_err());
}

// --- Channel ---

#[tokio::test]
async fn test_channel_send_recv() {
    let (tx, mut rx) = activation_channel("test", ChannelConfig::default());
    let msg = dummy_forward_msg(0, 0);
    tx.send(msg.clone()).await.expect("send");

    let received = rx.recv().await.expect("recv").expect("not closed");
    assert_eq!(received.stage_index, 0);
    assert!(!received.is_reverse_path);
}

#[tokio::test]
async fn test_channel_backpressure() {
    let config = ChannelConfig {
        capacity: 2,
        send_timeout: Some(Duration::from_millis(50)),
        recv_timeout: None,
    };
    let (tx, _rx) = activation_channel("bp-test", config);

    // Fill the channel
    tx.send(dummy_forward_msg(0, 0)).await.expect("send 1");
    tx.send(dummy_forward_msg(0, 1)).await.expect("send 2");

    // Third send should time out (channel full, no receiver draining)
    let result = tx.send(dummy_forward_msg(0, 2)).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("timed out"));
}

#[tokio::test]
async fn test_channel_try_send_full() {
    let config = ChannelConfig {
        capacity: 1,
        ..Default::default()
    };
    let (tx, _rx) = activation_channel("try-test", config);
    tx.try_send(dummy_forward_msg(0, 0)).expect("try_send 1");

    let result = tx.try_send(dummy_forward_msg(0, 1));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("full"));
}

#[tokio::test]
async fn test_channel_closed_detection() {
    let config = ChannelConfig::default();
    let (tx, rx) = activation_channel("close-test", config);
    assert!(!tx.is_closed());
    drop(rx);
    assert!(tx.is_closed());

    let result = tx.send(dummy_forward_msg(0, 0)).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("closed"));
}

#[tokio::test]
async fn test_channel_recv_timeout() {
    let config = ChannelConfig {
        capacity: 4,
        recv_timeout: Some(Duration::from_millis(30)),
        send_timeout: None,
    };
    let (_tx, mut rx) = activation_channel("timeout-test", config);

    let result = rx.recv().await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("timed out"));
}

#[tokio::test]
async fn test_channel_recv_returns_none_on_drop() {
    let config = ChannelConfig::default();
    let (tx, mut rx) = activation_channel("drop-test", config);
    drop(tx);
    let result = rx.recv().await.expect("no timeout error");
    assert!(result.is_none());
}

#[test]
fn test_queued_count() {
    let config = ChannelConfig {
        capacity: 4,
        ..Default::default()
    };
    let (tx, _rx) = activation_channel("queue-test", config);
    assert_eq!(tx.queued(), 0);
    tx.try_send(dummy_forward_msg(0, 0)).unwrap();
    assert_eq!(tx.queued(), 1);
    tx.try_send(dummy_forward_msg(0, 1)).unwrap();
    assert_eq!(tx.queued(), 2);
}

// --- PipelineChannel ---

#[tokio::test]
async fn test_pipeline_channel_forward() {
    let ch = PipelineChannel::new(0, 1, &ChannelConfig::default());
    let (left, mut right) = ch.split();

    let msg = dummy_forward_msg(0, 0);
    left.send_forward.send(msg).await.expect("send forward");

    // right.recv_reverse is actually forward_rx for the right endpoint
    let received = right.recv_reverse.recv().await.expect("recv").expect("msg");
    assert_eq!(received.stage_index, 0);
}

#[tokio::test]
async fn test_pipeline_channel_reverse() {
    let ch = PipelineChannel::new(0, 1, &ChannelConfig::default());
    let (mut left, right) = ch.split();

    let msg = dummy_reverse_msg(1, 0);
    right.send_forward.send(msg).await.expect("send reverse");

    let received = left.recv_reverse.recv().await.expect("recv").expect("msg");
    assert_eq!(received.stage_index, 1);
    assert!(received.is_reverse_path);
}

// --- StageLink ---

#[tokio::test]
async fn test_stage_link_forward_path() {
    let link = StageLink::new(0, 1, &ChannelConfig::default());
    let StageLink {
        forward_tx,
        mut forward_rx,
        ..
    } = link;

    let msg = dummy_forward_msg(0, 0);
    forward_tx.send(msg).await.expect("send");

    let received = forward_rx.recv().await.expect("recv").expect("msg");
    assert_eq!(received.stage_index, 0);
    assert_eq!(received.micro_batch_id, 0);
}

#[tokio::test]
async fn test_stage_link_reverse_path() {
    let link = StageLink::new(0, 1, &ChannelConfig::default());
    let StageLink {
        reverse_tx,
        mut reverse_rx,
        ..
    } = link;

    let msg = dummy_reverse_msg(1, 0);
    reverse_tx.send(msg).await.expect("send");

    let received = reverse_rx.recv().await.expect("recv").expect("msg");
    assert_eq!(received.stage_index, 1);
    assert!(received.is_reverse_path);
}

// --- build_pipeline_links ---

#[test]
fn test_build_pipeline_links_3_stages() {
    let links = build_pipeline_links(3, &ChannelConfig::default()).expect("build");
    assert_eq!(links.len(), 2);
    assert_eq!(links[0].upstream_stage, 0);
    assert_eq!(links[0].downstream_stage, 1);
    assert_eq!(links[1].upstream_stage, 1);
    assert_eq!(links[1].downstream_stage, 2);
}

#[test]
fn test_build_pipeline_links_min_stages() {
    let links = build_pipeline_links(2, &ChannelConfig::default()).expect("build");
    assert_eq!(links.len(), 1);
}

#[test]
fn test_build_pipeline_links_too_few() {
    let result = build_pipeline_links(1, &ChannelConfig::default());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("at least 2"));
}

// --- Latency ---

#[test]
fn test_activation_latency() {
    let msg = dummy_forward_msg(0, 0);
    let latency = activation_latency(&msg);
    // Should be very small since we just created the message.
    assert!(latency < Duration::from_secs(1));
}

// --- End-to-end multi-stage pipeline ---

#[tokio::test]
async fn test_end_to_end_3_stage_pipeline() {
    // Build a 3-stage pipeline: stage 0 -> stage 1 -> stage 2
    let links = build_pipeline_links(3, &ChannelConfig::default()).expect("build links");

    // Destructure into individual components for each link.
    // We use channels directly rather than split() for this test.
    let mut link_parts: Vec<_> = links
        .into_iter()
        .map(|link| {
            (
                link.upstream_stage,
                link.downstream_stage,
                link.forward_tx,
                link.forward_rx,
                link.reverse_tx,
                link.reverse_rx,
            )
        })
        .collect();

    // Stage 0 sends forward to stage 1 via link 0.
    let msg0 = dummy_forward_msg(0, 0);
    link_parts[0].2.send(msg0).await.expect("stage0 fwd send");

    // Stage 1 receives from link 0.
    let received1 = link_parts[0].3.recv().await.expect("recv").expect("msg");
    assert_eq!(received1.stage_index, 0);

    // Stage 1 processes and forwards to stage 2 via link 1.
    let msg1 = dummy_forward_msg(1, 0);
    link_parts[1].2.send(msg1).await.expect("stage1 fwd send");

    // Stage 2 receives from link 1.
    let received2 = link_parts[1].3.recv().await.expect("recv").expect("msg");
    assert_eq!(received2.stage_index, 1);

    // Stage 2 sends reverse (logits) back to stage 1 via link 1.
    let rev2 = dummy_reverse_msg(2, 0);
    link_parts[1].4.send(rev2).await.expect("stage2 rev send");

    // Stage 1 receives reverse from link 1.
    let rev_recv1 = link_parts[1].5.recv().await.expect("recv").expect("msg");
    assert_eq!(rev_recv1.stage_index, 2);
    assert!(rev_recv1.is_reverse_path);

    // Stage 1 forwards reverse to stage 0 via link 0.
    let rev1 = dummy_reverse_msg(1, 0);
    link_parts[0].4.send(rev1).await.expect("stage1 rev send");

    // Stage 0 receives the final reverse result.
    let rev_recv0 = link_parts[0].5.recv().await.expect("recv").expect("msg");
    assert_eq!(rev_recv0.stage_index, 1);
    assert!(rev_recv0.is_reverse_path);
}

// --- Concurrent micro-batch overlap ---

#[tokio::test]
async fn test_concurrent_micro_batches() {
    let config = ChannelConfig {
        capacity: 8,
        ..Default::default()
    };
    let (tx, mut rx) = activation_channel("overlap-test", config);

    // Send multiple micro-batches concurrently.
    let tx_clone = tx.clone();
    let send_handle = tokio::spawn(async move {
        for mb in 0..4u32 {
            tx_clone.send(dummy_forward_msg(0, mb)).await.expect("send");
        }
    });

    send_handle.await.expect("send task");

    // Receive all.
    let mut received_mbs = Vec::new();
    for _ in 0..4 {
        let msg = rx.recv().await.expect("recv").expect("msg");
        received_mbs.push(msg.micro_batch_id);
    }
    // Should receive in order (FIFO channel).
    assert_eq!(received_mbs, vec![0, 1, 2, 3]);
}

#[tokio::test]
async fn test_transport_stage_link_round_trip_and_lifecycle() {
    let router = MockRouter::new();
    let upstream_transport = Arc::new(
        MockTransport::new(
            "stage-0:9000".to_string(),
            router.clone(),
            MockTransportConfig::default(),
        )
        .await,
    );
    let downstream_transport = Arc::new(
        MockTransport::new(
            "stage-1:9001".to_string(),
            router,
            MockTransportConfig::default(),
        )
        .await,
    );

    let upstream_state = Arc::new(Mutex::new(StageLifecycleState {
        health: StageHealth::Healthy,
        ..Default::default()
    }));
    let downstream_state = Arc::new(Mutex::new(StageLifecycleState {
        health: StageHealth::Healthy,
        ..Default::default()
    }));
    downstream_state
        .lock()
        .expect("state lock")
        .mark_request_started();

    install_stage_control_service(upstream_transport.clone(), upstream_state.clone())
        .await
        .expect("install upstream control");
    install_stage_control_service(downstream_transport.clone(), downstream_state.clone())
        .await
        .expect("install downstream control");

    let link = TransportStageLink::new(0, 1, upstream_transport, downstream_transport)
        .expect("build transport link");

    link.upstream
        .send_activation(dummy_forward_msg(0, 7))
        .await
        .expect("send forward");
    let (from, received) = link
        .downstream
        .recv_activation()
        .await
        .expect("recv forward");
    assert_eq!(from, "stage-0:9000");
    assert_eq!(received.micro_batch_id, 7);

    link.downstream
        .send_activation(dummy_reverse_msg(1, 7))
        .await
        .expect("send reverse");
    let (from, received) = link.upstream.recv_activation().await.expect("recv reverse");
    assert_eq!(from, "stage-1:9001");
    assert!(received.is_reverse_path);

    let health = link.upstream.probe_health().await.expect("probe health");
    assert_eq!(health.health, StageHealth::Healthy);
    assert_eq!(health.in_flight_requests, 1);

    let drained = link.upstream.begin_drain().await.expect("begin drain");
    assert!(drained.draining);

    let barrier = link.upstream.barrier(42).await.expect("barrier");
    assert_eq!(barrier.last_barrier_id, Some(42));

    let cancelled = link
        .upstream
        .cancel_request(&RequestId::from_string("req-42".to_string()).unwrap())
        .await
        .expect("cancel");
    assert_eq!(cancelled.in_flight_requests, 0);

    let shutdown = link.upstream.shutdown_peer().await.expect("shutdown");
    assert!(shutdown.shutdown);
    assert_eq!(shutdown.health, StageHealth::Failed);
}
