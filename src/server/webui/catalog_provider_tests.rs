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

use super::super::catalog_metadata::apply_single_model_runtime_fields;
use super::*;
use crate::server::router_models::RouterCatalogProviderCapabilities;

fn provider() -> Option<RouterCatalogProviderCapabilities> {
    Some(RouterCatalogProviderCapabilities {
        image_input: false,
        audio_input: false,
    })
}
fn ready_count(entry: &CatalogEntry, task: TaskKind) -> usize {
    entry
        .capabilities
        .iter()
        .filter(|cap| cap.task == task && cap.phase == "provider_ready" && cap.available)
        .count()
}

#[test]
fn text_provider_projection_preserves_preload_and_resets_every_nonready_state() {
    let root = temp_dir("text-provider-projection");
    let path = write_model(&root, "text", "qwen3");
    let mut entry = catalog_entry(model("text", path, RouterModelSource::ModelsDir));
    // Isolate provider projection from the build host's backend support flag.
    entry.supported = true;
    let preload = serde_json::to_value(
        entry
            .capabilities
            .iter()
            .filter(|cap| cap.phase == "pre_load")
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for _ in 0..3 {
        apply_single_model_runtime_fields(
            &mut entry,
            lifecycle_with(ModelLifecycleState::Ready, false, 0),
            provider(),
        );
        assert_eq!(ready_count(&entry, TaskKind::Chat), 1);
        assert_eq!(ready_count(&entry, TaskKind::Completion), 1);
        assert_eq!(
            serde_json::to_value(
                entry
                    .capabilities
                    .iter()
                    .filter(|cap| cap.phase == "pre_load")
                    .collect::<Vec<_>>()
            )
            .unwrap(),
            preload
        );
    }
    for state in [
        ModelLifecycleState::Unloaded,
        ModelLifecycleState::Loading,
        ModelLifecycleState::Draining,
        ModelLifecycleState::Unloading,
        ModelLifecycleState::Failed,
    ] {
        apply_single_model_runtime_fields(&mut entry, lifecycle_with(state, false, 0), provider());
        assert_eq!(ready_count(&entry, TaskKind::Chat), 0, "{state:?}");
        assert_eq!(ready_count(&entry, TaskKind::Completion), 0, "{state:?}");
        apply_single_model_runtime_fields(
            &mut entry,
            lifecycle_with(ModelLifecycleState::Ready, false, 0),
            provider(),
        );
    }
    apply_single_model_runtime_fields(
        &mut entry,
        lifecycle_with(ModelLifecycleState::Ready, false, 0),
        None,
    );
    assert_eq!(ready_count(&entry, TaskKind::Chat), 0);
    entry.supported = false;
    apply_single_model_runtime_fields(
        &mut entry,
        lifecycle_with(ModelLifecycleState::Ready, false, 0),
        provider(),
    );
    assert_eq!(ready_count(&entry, TaskKind::Chat), 0);
    assert_eq!(ready_count(&entry, TaskKind::Completion), 0);
}

#[test]
fn provider_presence_does_not_invent_generation_tasks_for_nontext_models() {
    let root = temp_dir("nontext-provider-projection");
    let path = write_model(&root, "text", "qwen3");
    let mut entry = catalog_entry(model("text", path, RouterModelSource::ModelsDir));
    entry.supported = true;
    entry.metadata.output_tasks = vec![TaskKind::Embedding];
    apply_single_model_runtime_fields(
        &mut entry,
        lifecycle_with(ModelLifecycleState::Ready, false, 0),
        provider(),
    );
    assert_eq!(ready_count(&entry, TaskKind::Chat), 0);
    assert_eq!(ready_count(&entry, TaskKind::Completion), 0);
}

#[test]
#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "cuda"))]
fn cached_text_capabilities_follow_ready_unload_ready_without_epoch_rescan() {
    let cache = CatalogProjectionCache::new();
    let root = temp_dir("text-provider-cache");
    let path = write_model(&root, "text", "qwen3");
    let mut snapshot = model("text", path.clone(), RouterModelSource::ModelsDir);
    let id = snapshot.ui_model_id.clone();
    let first = get_catalog_entry_with_cache(&cache, vec![snapshot.clone()], &id).unwrap();
    assert!(first.supported);
    assert_eq!(ready_count(&first, TaskKind::Chat), 0);
    // A removed fake shard proves subsequent runtime projection uses the same cached metadata.
    std::fs::remove_file(path.join("model.safetensors")).unwrap();
    cache.reset_heavy_metadata_probe_count();
    for (revision, state) in [
        (2, ModelLifecycleState::Ready),
        (3, ModelLifecycleState::Unloaded),
        (4, ModelLifecycleState::Ready),
        (5, ModelLifecycleState::Failed),
    ] {
        snapshot.revision = revision;
        snapshot.lifecycle = lifecycle_with(state, false, 0);
        snapshot.provider_capabilities = provider();
        let projected = get_catalog_entry_with_cache(&cache, vec![snapshot.clone()], &id).unwrap();
        assert_eq!(projected.identity.revision, revision);
        for task in [TaskKind::Chat, TaskKind::Completion] {
            assert_eq!(
                ready_count(&projected, task),
                usize::from(state == ModelLifecycleState::Ready)
            );
        }
        assert_eq!(cache.heavy_metadata_probe_count(), 0);
    }
}
