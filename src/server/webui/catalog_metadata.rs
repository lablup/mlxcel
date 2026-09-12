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

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::models::detect_model_type_with_probes;
use crate::models::registry::{
    ArchitectureFamily, BackendStatus, Modality, OutputKind, Runtime, build_architecture_registry,
};
use crate::server::AppState;
use crate::server::router_lifecycle::{DownloadState, LifecycleSnapshot, ModelLifecycleState};
use crate::server::router_models::{
    RouterCatalogModel, RouterCatalogProviderCapabilities, RouterModelSource,
};

use super::catalog_detection::BoundedCatalogDetectionProbes;
use super::catalog_fs::{
    completeness, content_fingerprint, disk_size, format_for, read_json_bounded,
};
use super::catalog_metadata_config::{
    declared_architectures_from_config, declared_model_type_from_config,
};

use super::catalog_types::{
    Capability, CatalogEntry, CatalogMetadata, CatalogMetadataUnknownReasons, CatalogSourceKind,
    MAX_CONFIG_BYTES, ModelIdentity, RemovalStatus, SupportStatus, TaskKind,
};

pub(super) fn catalog_entry(model: RouterCatalogModel) -> CatalogEntry {
    let metadata = metadata_for(&model.path);
    let complete = metadata.support.complete;
    let supported = metadata.support.architecturally_supported
        && metadata.support.runnable_on_backend
        && complete;
    let removal = removal_status(model.source, &model.lifecycle);
    let mut entry = CatalogEntry {
        identity: ModelIdentity {
            id: model.ui_model_id,
            inference_id: model.name.clone(),
            display_name: display_name(&model.name),
            source: source_kind(model.source),
            source_key_hash: model.source_key_hash,
            generation: model.generation,
            revision: model.revision,
            content_fingerprint: content_fingerprint(&model.path),
        },
        capabilities: capabilities_for(&metadata, supported),
        lifecycle: model.lifecycle,
        complete,
        supported,
        removable: removal.eligible,
        metadata,
        removal,
    };
    apply_provider_confirmed_capabilities(&mut entry, model.provider_capabilities);
    entry
}

pub(super) fn catalog_content_signature(path: &Path) -> Option<String> {
    content_fingerprint(path)
}

pub(super) fn apply_runtime_fields(entry: &mut CatalogEntry, model: &RouterCatalogModel) {
    entry.identity.revision = model.revision;
    entry.identity.generation = model.generation;
    entry.lifecycle = model.lifecycle.clone();
    let removal = removal_status(model.source, &model.lifecycle);
    entry.removable = removal.eligible;
    entry.removal = removal;
    apply_provider_confirmed_capabilities(entry, model.provider_capabilities);
}

pub(super) fn apply_single_model_runtime_fields(
    entry: &mut CatalogEntry,
    lifecycle: LifecycleSnapshot,
    provider_capabilities: Option<RouterCatalogProviderCapabilities>,
) {
    entry.lifecycle = lifecycle;
    apply_provider_confirmed_capabilities(entry, provider_capabilities);
}

#[allow(dead_code)]
pub fn single_model_entry_from_state(state: &AppState) -> CatalogEntry {
    single_model_entry_with_provider(
        state.model_path.clone(),
        state.display_model_id().to_string(),
        single_model_lifecycle_from_state(state),
        Some(single_model_provider_capabilities_from_state(state)),
    )
}

pub(super) fn single_model_lifecycle_from_state(state: &AppState) -> LifecycleSnapshot {
    LifecycleSnapshot {
        state: if state.model_provider.is_loaded() {
            ModelLifecycleState::Ready
        } else if state.model_provider.is_chat_unavailable() {
            ModelLifecycleState::Failed
        } else {
            ModelLifecycleState::Loading
        },
        download: DownloadState::Complete,
        busy: false,
        active_requests: 0,
        draining_requests: 0,
        worker_exit_observed: state.model_provider.worker_exit_observer().observed(),
        last_error: state
            .model_provider
            .is_chat_unavailable()
            .then(|| "single-model provider is unavailable".to_string()),
    }
}

pub(super) fn single_model_provider_capabilities_from_state(
    state: &AppState,
) -> RouterCatalogProviderCapabilities {
    RouterCatalogProviderCapabilities {
        image_input: state.media_support.image,
        audio_input: state.media_support.audio,
    }
}

#[allow(dead_code)]
pub fn single_model_entry(
    model_path: PathBuf,
    inference_id: String,
    lifecycle: LifecycleSnapshot,
) -> CatalogEntry {
    single_model_entry_with_provider(model_path, inference_id, lifecycle, None)
}

pub(super) fn single_model_entry_with_provider(
    model_path: PathBuf,
    inference_id: String,
    lifecycle: LifecycleSnapshot,
    provider_capabilities: Option<RouterCatalogProviderCapabilities>,
) -> CatalogEntry {
    let (id, source_key_hash) = crate::server::router_lifecycle::stable_model_identity(
        "single_model",
        3,
        "single_model:redacted-path",
        &inference_id,
    );
    let metadata = metadata_for(&model_path);
    let complete = metadata.support.complete;
    let supported = metadata.support.architecturally_supported
        && metadata.support.runnable_on_backend
        && complete;
    let mut entry = CatalogEntry {
        identity: ModelIdentity {
            id,
            inference_id: inference_id.clone(),
            display_name: display_name(&inference_id),
            source: CatalogSourceKind::SingleModel,
            source_key_hash,
            generation: 1,
            revision: 1,
            content_fingerprint: content_fingerprint(&model_path),
        },
        capabilities: capabilities_for(&metadata, supported),
        lifecycle,
        complete,
        supported,
        removable: false,
        metadata,
        removal: RemovalStatus {
            eligible: false,
            reason: Some("single-model servers are read-only from the catalog".to_string()),
            instructions: Some(
                "Restart without -m and use model-free WebUI mode to switch or delete models."
                    .to_string(),
            ),
        },
    };
    apply_provider_confirmed_capabilities(&mut entry, provider_capabilities);
    entry
}

fn metadata_for(path: &Path) -> CatalogMetadata {
    let mut reasons = CatalogMetadataUnknownReasons::default();
    let (config, config_error) = read_json_bounded(&path.join("config.json"), MAX_CONFIG_BYTES);
    let declared_model_type =
        declared_model_type_from_config(config.as_ref(), config_error.as_deref(), &mut reasons);
    let declared_architectures =
        declared_architectures_from_config(config.as_ref(), config_error.as_deref(), &mut reasons);
    let detected_model_type = match (config.as_ref(), declared_model_type.as_ref()) {
        (Some(config), Some(_)) => {
            detect_model_type_with_probes(path, config, &BoundedCatalogDetectionProbes)
                .map(|model_type| model_type.registry_id().to_string())
                .map_err(|err| sanitize_detection_error(path, &err.to_string()))
        }
        (Some(_), None) => Err(reasons
            .model_type
            .clone()
            .unwrap_or_else(|| "model_type is unknown".to_string())),
        (None, _) => Err(config_error
            .clone()
            .unwrap_or_else(|| "config.json is unavailable in bounded metadata".to_string())),
    };
    let family = detected_model_type
        .as_ref()
        .ok()
        .and_then(|model_type| family_for_registry_id(model_type));
    if family.is_none() {
        reasons.architecture = Some(
            detected_model_type
                .as_ref()
                .err()
                .cloned()
                .or_else(|| reasons.model_type.clone())
                .unwrap_or_else(|| "model_type is unknown".to_string()),
        );
    }
    let complete = completeness(path);
    let (disk_bytes, disk_reason) = disk_size(path);
    reasons.disk_bytes = disk_reason;
    let quantization = config.as_ref().and_then(quantization_from_config);
    if quantization.is_none() {
        reasons.quantization = Some("quantization is not declared in bounded metadata".to_string());
    }
    let (format, format_reason) = format_for(path);
    if format.is_none() {
        reasons.format = Some(
            format_reason.unwrap_or_else(|| "no SafeTensors metadata file was found".to_string()),
        );
    }
    reasons.parameter_count =
        Some("parameter count is not measured during metadata-only catalog scans".to_string());
    reasons.memory_estimate_bytes =
        Some("memory estimate requires backend/provider measurement after load".to_string());
    let runnable = family.as_ref().is_some_and(runnable_on_backend);
    let support = SupportStatus {
        architecturally_supported: family.is_some(),
        runnable_on_backend: runnable,
        complete: complete.ok,
        reason: support_reason(family.is_some(), runnable, &complete.reason),
        architecturally_supported_reason: family
            .is_none()
            .then(|| reasons.architecture.clone())
            .flatten(),
        runnable_on_backend_reason: (!runnable).then(|| {
            "current backend does not advertise this architecture as runnable".to_string()
        }),
        complete_reason: (!complete.ok).then(|| complete.reason.clone()),
        tested_checkpoint: false,
        tested_checkpoint_reason: Some(
            "No catalog-specific tested-checkpoint evidence is available.".to_string(),
        ),
    };
    let (input_tasks, output_tasks) = family
        .as_ref()
        .map(tasks_for_family)
        .unwrap_or_else(|| (Vec::new(), Vec::new()));
    CatalogMetadata {
        architecture: family.as_ref().map(|family| family.id.to_string()),
        declared_architectures,
        input_tasks,
        output_tasks,
        quantization,
        format,
        parameter_count: None,
        disk_bytes,
        memory_estimate_bytes: None,
        support,
        model_type: declared_model_type,
        unknown_reasons: reasons,
    }
}

fn sanitize_detection_error(path: &Path, message: &str) -> String {
    let path_text = path.display().to_string();
    if path_text.is_empty() {
        message.to_string()
    } else {
        message.replace(&path_text, "model directory")
    }
}

fn family_for_registry_id(registry_id: &str) -> Option<ArchitectureFamily> {
    build_architecture_registry(env!("CARGO_PKG_VERSION"))
        .families
        .into_iter()
        .find(|family| family.id == registry_id)
}

fn runnable_on_backend(family: &ArchitectureFamily) -> bool {
    let status = if cfg!(all(target_os = "macos", feature = "metal")) {
        family.backends.metal
    } else if cfg!(feature = "cuda") {
        family.backends.cuda
    } else {
        BackendStatus::Unsupported
    };
    matches!(status, BackendStatus::Supported | BackendStatus::Partial)
}

fn tasks_for_family(family: &ArchitectureFamily) -> (Vec<TaskKind>, Vec<TaskKind>) {
    let mut input = Vec::new();
    let mut output = Vec::new();
    for runtime in family.runtimes {
        match runtime {
            Runtime::Generate | Runtime::Serve => push_unique(&mut output, TaskKind::Chat),
            Runtime::Embed => push_unique(&mut output, TaskKind::Embedding),
            Runtime::Rerank => push_unique(&mut output, TaskKind::Rerank),
            Runtime::Asr => push_unique(&mut output, TaskKind::AudioTranscription),
            Runtime::Tts => push_unique(&mut output, TaskKind::AudioSpeech),
            Runtime::Detect => push_unique(&mut input, TaskKind::VisionInput),
        }
    }
    for modality in family.modalities_in {
        match modality {
            Modality::Text => push_unique(&mut input, TaskKind::Chat),
            Modality::Image | Modality::Video => push_unique(&mut input, TaskKind::VisionInput),
            Modality::Audio => push_unique(&mut input, TaskKind::AudioTranscription),
        }
    }
    if family.output == OutputKind::Tokens {
        push_unique(&mut output, TaskKind::Completion);
    }
    (input, output)
}

fn push_unique<T: PartialEq>(items: &mut Vec<T>, item: T) {
    if !items.contains(&item) {
        items.push(item);
    }
}

fn capabilities_for(metadata: &CatalogMetadata, supported: bool) -> Vec<Capability> {
    metadata
        .output_tasks
        .iter()
        .chain(
            metadata
                .input_tasks
                .iter()
                .filter(|task| **task == TaskKind::VisionInput),
        )
        .copied()
        .map(|task| Capability {
            task,
            phase: if task == TaskKind::VisionInput {
                "provider_ready"
            } else {
                "pre_load"
            }
            .to_string(),
            available: supported && task != TaskKind::VisionInput,
            reason: if supported && task != TaskKind::VisionInput {
                None
            } else {
                metadata
                    .support
                    .reason
                    .clone()
                    .or_else(|| Some("provider readiness is confirmed after load".to_string()))
            },
        })
        .collect()
}

fn apply_provider_confirmed_capabilities(
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
}

fn reset_provider_ready_capabilities(entry: &mut CatalogEntry) {
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

fn quantization_from_config(config: &Value) -> Option<String> {
    config
        .get("quantization")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            config
                .get("quantization_config")?
                .get("bits")?
                .as_u64()
                .map(|bits| format!("{bits}bit"))
        })
}

fn support_reason(architectural: bool, runnable: bool, complete_reason: &str) -> Option<String> {
    if !architectural {
        Some("architecture is not supported by the mlxcel registry".to_string())
    } else if !runnable {
        Some("architecture is not runnable on the current backend".to_string())
    } else if !complete_reason.is_empty() {
        Some(complete_reason.to_string())
    } else {
        None
    }
}

fn removal_status(source: RouterModelSource, lifecycle: &LifecycleSnapshot) -> RemovalStatus {
    if source != RouterModelSource::Cache {
        return RemovalStatus {
            eligible: false,
            reason: Some("only managed cache entries can be removed".to_string()),
            instructions: Some(
                "Use model-free mode with the managed cache to delete downloaded snapshots."
                    .to_string(),
            ),
        };
    }
    if lifecycle.busy {
        return RemovalStatus {
            eligible: false,
            reason: Some("model is busy".to_string()),
            instructions: Some(
                "Unload or wait for the model operation to finish before deleting the cache entry."
                    .to_string(),
            ),
        };
    }
    RemovalStatus {
        eligible: true,
        reason: None,
        instructions: None,
    }
}

fn display_name(name: &str) -> String {
    name.rsplit('/')
        .next()
        .unwrap_or(name)
        .replace(['-', '_'], " ")
}

fn source_kind(source: RouterModelSource) -> CatalogSourceKind {
    match source {
        RouterModelSource::Cache => CatalogSourceKind::Cache,
        RouterModelSource::ModelsDir => CatalogSourceKind::ModelsDir,
        RouterModelSource::Preset => CatalogSourceKind::Preset,
    }
}
