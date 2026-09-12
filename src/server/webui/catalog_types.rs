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

use serde::{Deserialize, Serialize};

use crate::server::router_lifecycle::LifecycleSnapshot;

pub(super) const DEFAULT_LIMIT: usize = 50;
pub(super) const MAX_LIMIT: usize = 200;
pub(super) const MAX_INVENTORY: usize = 1_000;
pub(super) const MAX_CONFIG_BYTES: u64 = 256 * 1024;
pub(super) const MAX_INDEX_BYTES: u64 = 512 * 1024;
pub(super) const MAX_DISK_FILES: usize = 4_096;
pub(super) const MAX_DISK_DEPTH: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Chat,
    Completion,
    Embedding,
    Rerank,
    AudioTranscription,
    AudioSpeech,
    VisionInput,
    ImageGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogSourceKind {
    Cache,
    ModelsDir,
    Preset,
    SingleModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub task: TaskKind,
    pub phase: String,
    pub available: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub id: String,
    pub inference_id: String,
    pub display_name: String,
    pub source: CatalogSourceKind,
    pub source_key_hash: String,
    pub generation: u64,
    pub revision: u64,
    pub content_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemovalStatus {
    pub eligible: bool,
    pub reason: Option<String>,
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportStatus {
    pub architecturally_supported: bool,
    pub runnable_on_backend: bool,
    pub complete: bool,
    pub reason: Option<String>,
    pub architecturally_supported_reason: Option<String>,
    pub runnable_on_backend_reason: Option<String>,
    pub complete_reason: Option<String>,
    pub tested_checkpoint: bool,
    pub tested_checkpoint_reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogMetadataUnknownReasons {
    pub architecture: Option<String>,
    pub model_type: Option<String>,
    pub quantization: Option<String>,
    pub format: Option<String>,
    pub parameter_count: Option<String>,
    pub disk_bytes: Option<String>,
    pub memory_estimate_bytes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogMetadata {
    pub architecture: Option<String>,
    pub input_tasks: Vec<TaskKind>,
    pub output_tasks: Vec<TaskKind>,
    pub quantization: Option<String>,
    pub format: Option<String>,
    pub parameter_count: Option<u64>,
    pub disk_bytes: Option<u64>,
    pub memory_estimate_bytes: Option<u64>,
    pub support: SupportStatus,
    pub model_type: Option<String>,
    pub unknown_reasons: CatalogMetadataUnknownReasons,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub identity: ModelIdentity,
    pub capabilities: Vec<Capability>,
    pub lifecycle: LifecycleSnapshot,
    pub complete: bool,
    pub supported: bool,
    pub removable: bool,
    pub metadata: CatalogMetadata,
    pub removal: RemovalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pagination {
    pub limit: usize,
    pub next_cursor: Option<String>,
    pub total_known: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogListResponse {
    pub schema_version: String,
    pub items: Vec<CatalogEntry>,
    pub pagination: Pagination,
    pub server_instance_id: String,
    pub snapshot_sequence: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct CatalogQuery {
    pub limit: Option<usize>,
    pub cursor: Option<String>,
    pub q: Option<String>,
    pub source: Option<String>,
    pub task: Option<String>,
    pub lifecycle: Option<String>,
    pub support: Option<String>,
    pub completeness: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogError {
    InvalidField {
        field: &'static str,
        message: String,
    },
    NotFound,
}
