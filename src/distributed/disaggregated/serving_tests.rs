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

use std::net::SocketAddr;
use std::time::Duration;

use super::*;

// ── ServingMode tests ────────────────────────────────────────────────

#[test]
fn serving_mode_from_node_role_prefill() {
    assert_eq!(
        ServingMode::from_node_role(NodeRole::Prefill),
        ServingMode::PrefillOnly
    );
}

#[test]
fn serving_mode_from_node_role_decode() {
    assert_eq!(
        ServingMode::from_node_role(NodeRole::Decode),
        ServingMode::DecodeOnly
    );
}

#[test]
fn serving_mode_from_node_role_hybrid() {
    assert_eq!(
        ServingMode::from_node_role(NodeRole::Hybrid),
        ServingMode::Hybrid
    );
}

#[test]
fn serving_mode_from_node_role_pipeline_defaults_hybrid() {
    assert_eq!(
        ServingMode::from_node_role(NodeRole::PipelineStage),
        ServingMode::Hybrid
    );
}

#[test]
fn serving_mode_runs_inference() {
    assert!(ServingMode::Hybrid.runs_inference());
    assert!(ServingMode::PrefillOnly.runs_inference());
    assert!(ServingMode::DecodeOnly.runs_inference());
    assert!(!ServingMode::Router.runs_inference());
}

#[test]
fn serving_mode_needs_model() {
    assert!(ServingMode::Hybrid.needs_model());
    assert!(ServingMode::PrefillOnly.needs_model());
    assert!(ServingMode::DecodeOnly.needs_model());
    assert!(!ServingMode::Router.needs_model());
}

#[test]
fn serving_mode_display() {
    assert_eq!(ServingMode::Hybrid.to_string(), "hybrid");
    assert_eq!(ServingMode::PrefillOnly.to_string(), "prefill-only");
    assert_eq!(ServingMode::DecodeOnly.to_string(), "decode-only");
    assert_eq!(ServingMode::Router.to_string(), "router");
}

// ── DisaggregatedServingConfig tests ─────────────────────────────────

#[test]
fn config_from_cli_no_flags_returns_none() {
    let result = DisaggregatedServingConfig::from_cli(None, vec![], vec![]);
    assert!(result.unwrap().is_none());
}

#[test]
fn config_from_cli_hybrid_mode() {
    let result = DisaggregatedServingConfig::from_cli(Some("hybrid"), vec![], vec![]);
    let config = result.unwrap().unwrap();
    assert_eq!(config.mode, ServingMode::Hybrid);
}

#[test]
fn config_from_cli_prefill_requires_decode_peers() {
    let result = DisaggregatedServingConfig::from_cli(Some("prefill"), vec![], vec![]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("decode-peers"));
}

#[test]
fn config_from_cli_decode_requires_prefill_peers() {
    let result = DisaggregatedServingConfig::from_cli(Some("decode"), vec![], vec![]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("prefill-peers"));
}

#[test]
fn config_from_cli_prefill_with_peers() {
    let decode_peer: SocketAddr = "127.0.0.1:9001".parse().unwrap();
    let result = DisaggregatedServingConfig::from_cli(Some("prefill"), vec![], vec![decode_peer]);
    let config = result.unwrap().unwrap();
    assert_eq!(config.mode, ServingMode::PrefillOnly);
    assert_eq!(config.decode_peers.len(), 1);
}

#[test]
fn config_from_cli_decode_with_peers() {
    let prefill_peer: SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let result = DisaggregatedServingConfig::from_cli(Some("decode"), vec![prefill_peer], vec![]);
    let config = result.unwrap().unwrap();
    assert_eq!(config.mode, ServingMode::DecodeOnly);
    assert_eq!(config.prefill_peers.len(), 1);
}

#[test]
fn config_from_cli_invalid_role() {
    let result = DisaggregatedServingConfig::from_cli(Some("invalid_role"), vec![], vec![]);
    assert!(result.is_err());
}

#[test]
fn config_from_cli_peers_without_role_defaults_hybrid() {
    let prefill_peer: SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let result = DisaggregatedServingConfig::from_cli(None, vec![prefill_peer], vec![]);
    let config = result.unwrap().unwrap();
    assert_eq!(config.mode, ServingMode::Hybrid);
}

// ── DisaggregatedMetrics tests ───────────────────────────────────────

#[test]
fn metrics_initial_values() {
    let metrics = DisaggregatedMetrics::new();
    let snap = metrics.snapshot();
    assert_eq!(snap.prefill_prompts_total, 0);
    assert_eq!(snap.decode_tokens_total, 0);
    assert_eq!(snap.cache_transfers_total, 0);
    assert_eq!(snap.prefill_tokens_per_sec, 0.0);
}

#[test]
fn metrics_record_prefill() {
    let metrics = DisaggregatedMetrics::new();
    metrics.record_prefill(1024, Duration::from_millis(100));
    let snap = metrics.snapshot();
    assert_eq!(snap.prefill_prompts_total, 1);
    assert_eq!(snap.prefill_tokens_total, 1024);
    assert!(snap.prefill_tokens_per_sec > 0.0);
}

#[test]
fn metrics_record_decode_tokens() {
    let metrics = DisaggregatedMetrics::new();
    metrics.record_decode_tokens(50, Duration::from_millis(500));
    let snap = metrics.snapshot();
    assert_eq!(snap.decode_tokens_total, 50);
    assert!(snap.decode_tokens_per_sec > 0.0);
}

#[test]
fn metrics_record_cache_transfer() {
    let metrics = DisaggregatedMetrics::new();
    metrics.record_cache_transfer(Duration::from_millis(15), 1024 * 1024);
    let snap = metrics.snapshot();
    assert_eq!(snap.cache_transfers_total, 1);
    assert!(snap.cache_transfer_avg_latency_ms > 0.0);
    assert_eq!(snap.cache_transfer_bytes_total, 1024 * 1024);
}

#[test]
fn metrics_record_cache_transfer_failure() {
    let metrics = DisaggregatedMetrics::new();
    metrics.record_cache_transfer_failure();
    let snap = metrics.snapshot();
    assert_eq!(snap.cache_transfer_failures, 1);
}

#[test]
fn metrics_update_queue_depths() {
    let metrics = DisaggregatedMetrics::new();
    metrics.update_queue_depths(5, 10, 3);
    let snap = metrics.snapshot();
    assert_eq!(snap.prefill_queue_depth, 5);
    assert_eq!(snap.decode_queue_depth, 10);
    assert_eq!(snap.transfer_queue_depth, 3);
}

#[test]
fn metrics_stream_bridged() {
    let metrics = DisaggregatedMetrics::new();
    metrics.record_stream_bridged();
    metrics.record_stream_bridge_failure();
    let snap = metrics.snapshot();
    assert_eq!(snap.streams_bridged_total, 1);
    assert_eq!(snap.stream_bridge_failures, 1);
}

// ── HybridModeGuard tests ────────────────────────────────────────────

#[test]
fn guard_hybrid_is_local() {
    let guard = HybridModeGuard::new(ServingMode::Hybrid);
    assert!(guard.is_local());
    assert!(guard.should_prefill());
    assert!(guard.should_decode());
    assert!(!guard.should_route());
}

#[test]
fn guard_prefill_only() {
    let guard = HybridModeGuard::new(ServingMode::PrefillOnly);
    assert!(!guard.is_local());
    assert!(guard.should_prefill());
    assert!(!guard.should_decode());
    assert!(!guard.should_route());
}

#[test]
fn guard_decode_only() {
    let guard = HybridModeGuard::new(ServingMode::DecodeOnly);
    assert!(!guard.is_local());
    assert!(!guard.should_prefill());
    assert!(guard.should_decode());
    assert!(!guard.should_route());
}

#[test]
fn guard_router() {
    let guard = HybridModeGuard::new(ServingMode::Router);
    assert!(!guard.is_local());
    assert!(!guard.should_prefill());
    assert!(!guard.should_decode());
    assert!(guard.should_route());
}

// ── DisaggregatedServer tests ────────────────────────────────────────

#[test]
fn server_hybrid_mode() {
    let config = DisaggregatedServingConfig::default();
    let server = DisaggregatedServer::new(config);
    assert_eq!(server.mode(), ServingMode::Hybrid);
    assert!(server.should_handle_locally());
    assert!(server.is_ready());
}

#[test]
fn server_prefill_mode_ready_with_decode_peers() {
    let config = DisaggregatedServingConfig {
        mode: ServingMode::PrefillOnly,
        decode_peers: vec!["127.0.0.1:9001".parse().unwrap()],
        ..Default::default()
    };
    let server = DisaggregatedServer::new(config);
    assert_eq!(server.mode(), ServingMode::PrefillOnly);
    assert!(!server.should_handle_locally());
    assert!(server.is_ready());
}

#[test]
fn server_prefill_mode_not_ready_without_peers() {
    let config = DisaggregatedServingConfig {
        mode: ServingMode::PrefillOnly,
        decode_peers: vec![],
        ..Default::default()
    };
    let server = DisaggregatedServer::new(config);
    assert!(!server.is_ready());
}

#[test]
fn server_decode_mode_ready_with_prefill_peers() {
    let config = DisaggregatedServingConfig {
        mode: ServingMode::DecodeOnly,
        prefill_peers: vec!["127.0.0.1:9000".parse().unwrap()],
        ..Default::default()
    };
    let server = DisaggregatedServer::new(config);
    assert!(server.is_ready());
}

#[test]
fn server_router_mode_requires_both_peers() {
    let config = DisaggregatedServingConfig {
        mode: ServingMode::Router,
        prefill_peers: vec!["127.0.0.1:9000".parse().unwrap()],
        decode_peers: vec![],
        ..Default::default()
    };
    let server = DisaggregatedServer::new(config);
    assert!(!server.is_ready());
}

#[test]
fn server_router_mode_ready_with_both_peers() {
    let config = DisaggregatedServingConfig {
        mode: ServingMode::Router,
        prefill_peers: vec!["127.0.0.1:9000".parse().unwrap()],
        decode_peers: vec!["127.0.0.1:9001".parse().unwrap()],
        ..Default::default()
    };
    let server = DisaggregatedServer::new(config);
    assert!(server.is_ready());
}

#[test]
fn server_uptime_increases() {
    let server = DisaggregatedServer::new(DisaggregatedServingConfig::default());
    let t1 = server.uptime();
    std::thread::sleep(Duration::from_millis(5));
    let t2 = server.uptime();
    assert!(t2 > t1);
}

#[test]
fn server_debug_format() {
    let server = DisaggregatedServer::new(DisaggregatedServingConfig::default());
    let debug = format!("{server:?}");
    assert!(debug.contains("DisaggregatedServer"));
    assert!(debug.contains("Hybrid"));
}
