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

use crate::server::router_models::RouterCatalogProviderCapabilities;

use super::*;

#[test]
fn catalog_cache_uses_epoch_and_projects_fresh_provider_lifecycle() {
    let cache = CatalogProjectionCache::new();
    let root = temp_dir("epoch-cache");
    let path = write_model(&root, "vision", "qwen2_vl");
    let id = stable_model_identity("models_dir", 1, "test-root", "vision").0;
    let first = get_catalog_entry_with_cache(
        &cache,
        vec![model("vision", path.clone(), RouterModelSource::ModelsDir)],
        &id,
    )
    .unwrap();
    assert!(first.complete);
    assert!(
        first.capabilities.iter().any(|capability| {
            capability.task == TaskKind::VisionInput && !capability.available
        })
    );

    std::fs::remove_file(path.join("model.safetensors")).expect("remove shard");
    cache.reset_heavy_metadata_probe_count();
    let mut same_epoch = model_with_lifecycle(
        "vision",
        path.clone(),
        RouterModelSource::ModelsDir,
        lifecycle_with(ModelLifecycleState::Ready, true, 3),
        9,
    );
    same_epoch.provider_capabilities = Some(RouterCatalogProviderCapabilities {
        image_input: true,
        audio_input: false,
    });
    let cached = get_catalog_entry_with_cache(&cache, vec![same_epoch], &id).unwrap();
    assert!(cached.complete, "same epoch must reuse cached metadata");
    assert_eq!(cached.identity.revision, 9);
    assert_eq!(cached.lifecycle.active_requests, 3);
    assert_eq!(cache.heavy_metadata_probe_count(), 0);
    assert!(
        cached
            .capabilities
            .iter()
            .any(|capability| { capability.task == TaskKind::VisionInput && capability.available })
    );

    let unprovided = get_catalog_entry_with_cache(
        &cache,
        vec![model_with_lifecycle(
            "vision",
            path.clone(),
            RouterModelSource::ModelsDir,
            lifecycle(),
            10,
        )],
        &id,
    )
    .unwrap();
    assert_eq!(unprovided.identity.revision, 10);
    assert_eq!(cache.heavy_metadata_probe_count(), 0);
    assert!(
        unprovided.capabilities.iter().any(|capability| {
            capability.task == TaskKind::VisionInput && !capability.available
        })
    );

    let mut next_epoch = model("vision", path, RouterModelSource::ModelsDir);
    next_epoch.catalog_epoch = 2;
    let refreshed = get_catalog_entry_with_cache(&cache, vec![next_epoch], &id).unwrap();
    assert!(!refreshed.complete);
    assert!(
        refreshed
            .metadata
            .support
            .complete_reason
            .as_deref()
            .unwrap()
            .contains("no non-empty SafeTensors")
    );
}
