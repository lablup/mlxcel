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

use super::{Capability, CatalogEntry, TaskKind};
use crate::server::router_lifecycle::ModelLifecycleState;
use crate::server::router_models::RouterCatalogProviderCapabilities;

// Used by: router catalog projection and single-model catalog projection.
pub(super) fn apply_provider_confirmed_capabilities(
    entry: &mut CatalogEntry,
    provider: Option<RouterCatalogProviderCapabilities>,
) {
    reset_provider_ready_capabilities(entry);
    let Some(provider) = provider else { return };
    for capability in &mut entry.capabilities {
        if capability.task == TaskKind::VisionInput {
            capability.available = provider.image_input && entry.supported;
            capability.reason = if capability.available {
                None
            } else if provider.image_input {
                entry.metadata.support.reason.clone()
            } else {
                Some("loaded provider does not advertise image input".to_string())
            };
        }
        if capability.task == TaskKind::AudioTranscription && provider.audio_input {
            capability.available = entry.supported;
            capability.reason = if entry.supported {
                None
            } else {
                entry.metadata.support.reason.clone()
            };
        }
    }
    // Provider facts are emitted by the existing loaded generation provider, not
    // inferred from a family name or merely from an observed lifecycle label.
    if entry.lifecycle.state == ModelLifecycleState::Ready && entry.supported {
        for task in [TaskKind::Chat, TaskKind::Completion] {
            if entry.metadata.output_tasks.contains(&task) {
                entry.capabilities.push(Capability {
                    task,
                    phase: "provider_ready".to_string(),
                    available: true,
                    reason: None,
                });
            }
        }
    }
}

fn reset_provider_ready_capabilities(entry: &mut CatalogEntry) {
    // Text provider facts are dynamic. Preserve independent pre-load metadata
    // and rebuild only these runtime entries, avoiding duplicates and stale positives.
    entry.capabilities.retain(|capability| {
        capability.phase != "provider_ready"
            || !matches!(capability.task, TaskKind::Chat | TaskKind::Completion)
    });
    for capability in &mut entry.capabilities {
        if capability.phase == "provider_ready" {
            capability.available = false;
            capability.reason = entry
                .metadata
                .support
                .reason
                .clone()
                .or_else(|| Some("provider readiness is confirmed after load".to_string()));
        }
    }
}
