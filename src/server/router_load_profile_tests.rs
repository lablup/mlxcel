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
use crate::server::webui::load_profile::UiLoadProfile;

fn pool(root: &Path) -> Arc<RouterPool> {
    std::fs::create_dir(root.join("alpha")).expect("model dir");
    std::fs::write(root.join("alpha/config.json"), r#"{"model_type":"llama"}"#).expect("config");
    let presets =
        super::super::router_presets::parse_preset_text("[alpha]\nctx-size=4096\nparallel=2\n")
            .expect("preset");
    Arc::new(
        RouterPool::new(
            RouterSources {
                models_dir: Some(root.to_path_buf()),
                cache: None,
                presets,
            },
            ServerStartupConfig::default(),
            Default::default(),
            Default::default(),
            1,
            false,
        )
        .expect("pool"),
    )
}

#[test]
fn profile_resolution_preserves_template_and_model_identity() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pool = pool(dir.path());
    let entry = pool.get("alpha").expect("entry");
    assert!(pool.next_load_cli_overrides().is_empty());
    let revision = entry.lifecycle.revision();
    let profile = UiLoadProfile {
        ctx_size: Some(8192),
        n_parallel: Some(4),
        kv_cache_mode: Some("fp16".into()),
    };
    let config = pool
        .resolve_entry_load_profile(&entry, &profile)
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(
        (
            config.context_size_total,
            config.n_parallel,
            config.context_size
        ),
        (8192, 4, 2048)
    );
    assert_eq!(config.model_alias, entry.config.model_alias);
    assert_eq!(config.model_aliases, entry.config.model_aliases);
    assert_eq!(
        (entry.config.context_size_total, entry.config.n_parallel),
        (4096, 2)
    );
    assert_eq!(entry.lifecycle.revision(), revision);
    assert!(!entry.is_running());
    let inherited = pool
        .resolve_entry_load_profile(&entry, &UiLoadProfile::default())
        .unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(
        (inherited.context_size_total, inherited.n_parallel),
        (4096, 2)
    );
}

#[tokio::test]
async fn profile_fingerprint_replay_conflict_and_failed_load_leave_template_intact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pool = pool(dir.path());
    let entry = pool.get("alpha").expect("entry");
    let revision = entry.lifecycle.revision();
    let profile = UiLoadProfile {
        ctx_size: Some(8192),
        ..Default::default()
    };
    let accepted = pool
        .submit_model_action_with_profile(
            &entry.ui_model_id,
            RouterModelAction::Load,
            revision,
            "profile-key-0001",
            None,
            Some(profile.clone()),
        )
        .expect("accepted");
    let replay = pool
        .submit_model_action_with_profile(
            &entry.ui_model_id,
            RouterModelAction::Load,
            revision,
            "profile-key-0001",
            None,
            Some(profile),
        )
        .expect("replay");
    assert!(replay.idempotent_replay);
    assert_eq!(replay.operation_id, accepted.operation_id);
    assert!(
        pool.submit_model_action_with_profile(
            &entry.ui_model_id,
            RouterModelAction::Load,
            revision,
            "profile-key-0001",
            None,
            Some(UiLoadProfile {
                ctx_size: Some(16384),
                ..Default::default()
            })
        )
        .is_err()
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let operation = pool
                .lifecycle
                .get_operation(&accepted.operation_id)
                .expect("operation");
            if operation.state == OperationState::Failed {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("tokenizerless fake must fail before any GPU worker starts");
    assert_eq!(entry.config.context_size_total, 4096);
    assert_eq!(
        pool.config_for_visible_model_id(&entry.ui_model_id)
            .expect("visible")
            .context_size_total,
        4096
    );
}

#[tokio::test]
async fn invalid_profile_is_rejected_before_eviction_or_lifecycle_mutation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pool = pool(dir.path());
    let entry = pool.get("alpha").expect("entry");
    let revision = entry.lifecycle.revision();
    let result = pool.submit_model_action_with_profile(
        &entry.ui_model_id,
        RouterModelAction::Load,
        revision,
        "invalid-profile-0001",
        Some(&entry.ui_model_id),
        Some(UiLoadProfile {
            ctx_size: Some(1),
            ..Default::default()
        }),
    );
    assert!(matches!(result, Err(RouterPoolError::OperationRejected(_))));
    assert_eq!(entry.lifecycle.revision(), revision);
    assert!(!entry.reserves_capacity());
}

#[tokio::test]
async fn active_profile_config_is_observed_but_cannot_hot_mutate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let pool = pool(dir.path());
    let entry = pool.get("alpha").expect("entry");
    let profile = UiLoadProfile {
        ctx_size: Some(8192),
        ..Default::default()
    };
    let config = pool
        .resolve_entry_load_profile(&entry, &profile)
        .unwrap_or_else(|e| panic!("{}", e.message));
    let (tx, _rx) = std::sync::mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(tx));
    let state = AppState::new(
        provider.clone(),
        config,
        ChatTemplateProcessor::with_template("ok".into()),
        crate::tokenizer::MlxcelTokenizer::stub(),
        entry.path.clone(),
        provider.batch_metrics().clone(),
    );
    entry.state.lock().expect("state").app = Some(LoadedApp {
        state,
        router: axum::Router::new(),
    });
    entry.lifecycle.mark_loading();
    entry.lifecycle.mark_ready();
    assert_eq!(
        pool.config_for_visible_model_id(&entry.ui_model_id)
            .expect("active")
            .context_size_total,
        8192
    );
    let attempted = pool.submit_model_action_with_profile(
        &entry.ui_model_id,
        RouterModelAction::Load,
        entry.lifecycle.revision(),
        "no-hot-profile-0001",
        None,
        Some(UiLoadProfile {
            ctx_size: Some(16384),
            ..Default::default()
        }),
    );
    assert!(attempted.is_err());
    assert_eq!(
        pool.config_for_visible_model_id(&entry.ui_model_id)
            .expect("active")
            .context_size_total,
        8192
    );
    assert_eq!(entry.config.context_size_total, 4096);
    provider.shutdown_worker();
    assert!(
        provider
            .worker_exit_observer()
            .wait_timeout(std::time::Duration::from_secs(2))
    );
}
