// Copyright 2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0. See LICENSE for details.
use super::*;
use crate::server::{ChatTemplateProcessor, ModelProvider};
use crate::tokenizer::MlxcelTokenizer;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

fn state(config: ServerConfig) -> AppState {
    let (tx, _rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(tx));
    let metrics = provider.batch_metrics().clone();
    AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("runtime-test-no-model"),
        metrics,
    )
}

fn snapshot(config: &ServerConfig, state: Option<&AppState>) -> RuntimeSnapshot {
    runtime_snapshot(
        "srv_20260912_a".into(),
        "mdl_lR1nHQwFUguxLqHbEzH2DJLdZYiDFJ0S3FzxIzY5MUU".into(),
        8,
        43,
        config,
        state,
    )
}

#[test]
fn disabled_projection_matches_whole_canonical_fixture() {
    let config = ServerConfig {
        n_parallel: 4,
        context_size: 4096,
        default_temperature: 0.5,
        enable_slots_endpoint: false,
        ..Default::default()
    };
    let result = serde_json::to_value(snapshot(&config, None)).unwrap();
    let mut expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/webui/scenarios/runtime.disabled.json"
    ))
    .unwrap();
    expected.as_object_mut().unwrap().remove("$schemaName");
    assert_eq!(result, expected);
}

#[test]
fn slots_mirror_native_context_without_text_or_double_counting() {
    let config = ServerConfig {
        n_parallel: 2,
        context_size: 4096,
        context_size_total: 8192,
        kv_unified: true,
        enable_slots_endpoint: true,
        enable_metrics_endpoint: true,
        ..Default::default()
    };
    let mut state = state(config.clone());
    state.slots = Arc::new(crate::server::slots_state::SlotRegistry::new(2, true));
    let handle = state.slots.begin(
        "secret prompt",
        serde_json::json!({"stop":"secret params"}),
        Some(20),
    );
    handle.on_prefill(100, 20);
    handle.on_token("secret generated");
    let result = snapshot(&config, Some(&state));
    assert_eq!(result.slots.request_context_tokens, Some(4096));
    assert_eq!(result.slots.shared_pool_context_tokens, Some(8192));
    assert_eq!(result.slots.items[0].prompt_tokens, Some(101));
    assert_eq!(result.slots.items[0].cached_prompt_tokens, Some(20));
    assert!(result.slots.items[0].processing);
    assert_eq!(result.slots.items[1].prompt_tokens, None);
    let encoded = serde_json::to_string(&result).unwrap();
    assert!(!encoded.contains("secret"));
    let native = state
        .slots
        .slots_json(state.effective_context_size(), false, false);
    assert_eq!(
        native[0]["n_prompt_tokens"],
        result.slots.items[0].prompt_tokens.unwrap()
    );
    assert!(result.measurements["kv_cache_bytes"].value.is_none());
    assert!(result.slots.effective_parallelism.is_none());
    drop(handle);
    assert!(!snapshot(&config, Some(&state)).slots.items[0].processing);
}

#[test]
fn nonbatch_geometry_and_unknown_denominator_are_not_guessed() {
    assert_eq!(nonzero(0), None);
    let config = ServerConfig {
        n_parallel: 4,
        context_size: 4096,
        context_size_total: 16384,
        enable_slots_endpoint: true,
        ..Default::default()
    };
    let state = state(config.clone());
    state
        .batch_metrics
        .publish_runtime_context_geometry(16384, Some(16384));
    let result = snapshot(&config, Some(&state));
    assert_eq!(result.slots.request_context_tokens, Some(16384));
    assert_eq!(result.slots.effective_parallelism, Some(1));
    assert_eq!(result.slots.shared_pool_context_tokens, None);
}

#[test]
fn counters_are_opt_in_and_reset_is_visible_without_mutating_scrape_baseline() {
    let config = ServerConfig {
        enable_metrics_endpoint: true,
        ..Default::default()
    };
    let state = state(config.clone());
    state
        .metrics
        .completion_tokens_total
        .store(42, Ordering::Relaxed);
    state.batch_metrics.queue_depth.store(2, Ordering::Relaxed);
    let first = snapshot(&config, Some(&state));
    assert_eq!(
        first.measurements["completion_tokens_total"].value,
        Some(42.0)
    );
    assert_eq!(first.measurements["queued_requests"].value, Some(2.0));
    state
        .metrics
        .completion_tokens_total
        .store(0, Ordering::Relaxed);
    let reset = snapshot(&config, Some(&state));
    assert_eq!(
        reset.measurements["completion_tokens_total"].value,
        Some(0.0)
    );
    assert_eq!(state.llama_scrape.lock().unwrap().predicted_tokens, 0);
    let disabled = snapshot(&ServerConfig::default(), Some(&state));
    assert!(
        disabled.measurements["completion_tokens_total"]
            .value
            .is_none()
    );
    assert!(
        disabled.measurements["completion_tokens_total"]
            .measured_at
            .is_none()
    );
    let unloaded = snapshot(&config, None);
    assert!(unloaded.measurements["active_requests"].value.is_none());
    assert!(!unloaded.slots.available);
}

#[test]
fn ready_projection_matches_whole_canonical_fixture() {
    let config = ServerConfig {
        n_parallel: 4,
        context_size: 4096,
        context_size_total: 16384,
        kv_unified: true,
        default_temperature: 0.5,
        enable_metrics_endpoint: true,
        enable_slots_endpoint: true,
        ..Default::default()
    };
    let state = state(config.clone());
    let mut result = serde_json::to_value(snapshot(&config, Some(&state))).unwrap();
    for metric in result["measurements"].as_object_mut().unwrap().values_mut() {
        if !metric["measured_at"].is_null() {
            metric["measured_at"] = "2026-09-12T03:04:05Z".into();
        }
    }
    result["slots"]["measured_at"] = "2026-09-12T03:04:05Z".into();
    let mut expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/webui/scenarios/runtime.ready-shared.json"
    ))
    .unwrap();
    expected.as_object_mut().unwrap().remove("$schemaName");
    assert_eq!(result, expected);
}

#[test]
fn single_stream_does_not_invent_idle_active_or_decode_counters() {
    let config = ServerConfig {
        enable_metrics_endpoint: true,
        ..Default::default()
    };
    let (tx, _rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests_with_admission(
        tx, true, 16,
    ));
    let metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config.clone(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("runtime-test-no-model"),
        metrics,
    );
    let snapshot = snapshot(&config, Some(&state));
    for key in [
        "active_requests",
        "decode_tokens_total",
        "decode_time_us_total",
    ] {
        assert!(snapshot.measurements[key].value.is_none());
        assert!(snapshot.measurements[key].reason.is_some());
    }
    assert_eq!(snapshot.measurements["queued_requests"].value, Some(0.0));
}
