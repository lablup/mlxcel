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

//! Validation of the cuda-oxide fused residual-add + RMSNorm artifact against
//! its manifests and ABI schemas (issue #2833).
//!
//! The artifact is the one `spike/cuda-oxide-fused-norm` produces (#2832): one
//! PTX module with five entry points, one generated manifest and one
//! hand-written ABI schema per entry. Nothing here touches a GPU. The output,
//! [`ValidatedArtifact`], is the only thing the adapter in
//! [`crate::cuda_oxide_fused_norm`] will install, so every claim the launch
//! relies on is checked first: the PTX bytes are the ones the manifests pin,
//! the target and ABI version are the ones this adapter was written for, and
//! the kernel parameter order is read from the manifest and cross-checked
//! against the schema instead of being assumed.
//!
//! The JIT module name carries the payload digest, the target, the ABI version
//! and the compiler revision. MLX's in-process module cache is keyed on that name and
//! is consulted before the payload is, so a changed artifact gets a new name
//! and can never be served a stale module. The module is created without the
//! PTX disk cache, so no entry on disk can stand in for the payload either.

use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

pub use crate::cuda_oxide_abi::{ARG_COUNT, ArgRole, Lane};
use crate::cuda_oxide_abi::{Argument, argument_roles};

/// The manifest format this adapter reads.
pub const MANIFEST_ID: &str = "mlxcel.cuda-oxide-fused-norm.manifest/v1";
/// The ABI schema format this adapter reads.
pub const SCHEMA_ID: &str = "mlxcel.cuda-oxide-fused-norm.abi/v1";
/// The kernel ABI version the argument packing implements.
pub const ABI_VERSION: u64 = 1;
/// The device architecture the committed artifact targets.
pub const TARGET_ARCH: &str = "sm_121";
/// [`TARGET_ARCH`] as a CUDA compute capability.
pub const TARGET_CAPABILITY: (u32, u32) = (12, 1);
/// The PTX module, under `artifacts/`.
pub const PTX_FILE: &str = "fused_add_rms_norm.sm_121.ptx";
/// The module's entry points; one manifest and one schema each.
pub const ENTRIES: [&str; 5] = [
    "fused_add_rms_norm_bf16",
    "fused_add_rms_norm_f16",
    "fused_add_rms_norm_f32",
    "fused_add_rms_norm_f16_wf32",
    "fused_add_rms_norm_bf16_wf32",
];

const GRID_RULE: &str = "rows";
/// The block width rule the adapter launches with: the one `fused_norm.cpp`
/// computes for the native CUDA kernel, MLX's own `rms_norm` geometry (#3510).
/// An artifact built for the earlier `fused_norm_threads(dim)` rule sums the
/// squares in another order at this width, so it is refused.
const BLOCK_RULE: &str = "fused_norm_cuda_block(dim, n_reads)";

/// One entry point of the module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntrySpec {
    pub symbol: String,
    pub activation: Lane,
    pub weight: Lane,
    /// The role of each kernel parameter, in parameter order.
    pub roles: [ArgRole; ARG_COUNT],
}

/// An artifact whose manifests, schemas and bytes agree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedArtifact {
    pub ptx: Vec<u8>,
    /// SHA-256 of [`Self::ptx`], lowercase hex.
    pub sha256: String,
    /// The cuda-oxide compiler revision that produced the PTX.
    pub compiler_revision: String,
    pub entries: Vec<EntrySpec>,
    /// JIT module name and PTX disk-cache key.
    pub module_name: String,
}

/// The raw documents of one entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryDocuments {
    pub entry: String,
    pub manifest: String,
    pub schema: String,
}

/// Why an artifact was refused. A refused artifact is never installed, so the
/// existing CUDA kernel keeps serving every call.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ArtifactError {
    #[error("cuda-oxide artifact file {path} cannot be read: {source}")]
    Missing {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("cuda-oxide {document} is not valid JSON for this adapter: {source}")]
    Json {
        document: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("cuda-oxide PTX hashes to {actual}, but {document} pins {pinned}")]
    Digest {
        document: String,
        pinned: String,
        actual: String,
    },
    #[error("cuda-oxide manifest for {entry} is not usable: {detail}")]
    Manifest { entry: String, detail: String },
    #[error("cuda-oxide ABI schema for {entry} disagrees with its manifest: {detail}")]
    Schema { entry: String, detail: String },
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Launch {
    grid_dim_rule: String,
    block_dim_rule: String,
    /// Elements per vector the entry was compiled for.
    n_reads: u32,
    shared_mem_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct Source {
    revision: String,
}

#[derive(Debug, Deserialize)]
struct Target {
    device_arch: String,
}

#[derive(Debug, Deserialize)]
struct Container {
    target: String,
    payload_kind: String,
    payload_bytes: u64,
    payload_sha256: String,
}

#[derive(Debug, Deserialize)]
struct PtxArtifact {
    sha256: String,
    bytes: u64,
    format: String,
}

#[derive(Debug, Deserialize)]
struct Artifacts {
    ptx: PtxArtifact,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    manifest: String,
    abi_version: u64,
    entry_symbol: String,
    module_entries: Vec<String>,
    source: Source,
    target: Target,
    container: Container,
    artifacts: Artifacts,
    arguments: Vec<Argument>,
    launch: Launch,
}

#[derive(Debug, Deserialize)]
struct Schema {
    schema: String,
    abi_version: u64,
    entry_symbol: String,
    activation_dtype: String,
    weight_dtype: String,
    source_revision: Source,
    arguments: Vec<Argument>,
    launch: Launch,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The JIT module name for an artifact: every field that identifies the
/// compiled code, so MLX's module cache cannot return another artifact's module.
#[must_use]
pub fn module_name(sha256: &str, compiler_revision: &str) -> String {
    let revision = &compiler_revision[..compiler_revision.len().min(12)];
    let digest = &sha256[..sha256.len().min(32)];
    format!("mlxcel_oxide_fused_norm_abi{ABI_VERSION}_{TARGET_ARCH}_{revision}_{digest}")
}

/// Read an artifact from a directory laid out like the spike package:
/// `artifacts/<ptx>`, `artifacts/<entry>.sm_121.manifest.json` and
/// `abi/<entry>.schema.json`.
pub fn read_dir(dir: &Path) -> Result<(Vec<u8>, Vec<EntryDocuments>), ArtifactError> {
    let missing = |path: &Path, source| ArtifactError::Missing {
        path: path.display().to_string(),
        source,
    };
    let ptx_path = dir.join("artifacts").join(PTX_FILE);
    let ptx = std::fs::read(&ptx_path).map_err(|e| missing(&ptx_path, e))?;
    let mut documents = Vec::with_capacity(ENTRIES.len());
    for entry in ENTRIES {
        let manifest_path = dir
            .join("artifacts")
            .join(format!("{entry}.sm_121.manifest.json"));
        let schema_path = dir.join("abi").join(format!("{entry}.schema.json"));
        documents.push(EntryDocuments {
            entry: entry.to_string(),
            manifest: std::fs::read_to_string(&manifest_path)
                .map_err(|e| missing(&manifest_path, e))?,
            schema: std::fs::read_to_string(&schema_path).map_err(|e| missing(&schema_path, e))?,
        });
    }
    Ok((ptx, documents))
}

/// Validate a PTX module against the manifest and schema of each entry.
pub fn validate(
    ptx: &[u8],
    documents: &[EntryDocuments],
) -> Result<ValidatedArtifact, ArtifactError> {
    let sha256 = sha256_hex(ptx);
    let ptx_text = std::str::from_utf8(ptx).unwrap_or("");
    let mut entries: Vec<EntrySpec> = Vec::with_capacity(documents.len());
    let mut compiler_revision: Option<String> = None;

    for expected in ENTRIES {
        let bad = |detail: String| ArtifactError::Manifest {
            entry: expected.to_string(),
            detail,
        };
        let docs = documents
            .iter()
            .find(|d| d.entry == expected)
            .ok_or_else(|| bad("no manifest was supplied for this entry".to_string()))?;
        let manifest: Manifest =
            serde_json::from_str(&docs.manifest).map_err(|source| ArtifactError::Json {
                document: format!("manifest for {expected}"),
                source,
            })?;
        let schema: Schema =
            serde_json::from_str(&docs.schema).map_err(|source| ArtifactError::Json {
                document: format!("ABI schema for {expected}"),
                source,
            })?;

        if manifest.manifest != MANIFEST_ID {
            return Err(bad(format!("format is {:?}", manifest.manifest)));
        }
        if manifest.abi_version != ABI_VERSION {
            return Err(bad(format!(
                "ABI version {} is not the supported {ABI_VERSION}",
                manifest.abi_version
            )));
        }
        if manifest.entry_symbol != expected {
            return Err(bad(format!(
                "it describes entry {:?}",
                manifest.entry_symbol
            )));
        }
        let mut module_entries = manifest.module_entries.clone();
        module_entries.sort();
        let mut known = ENTRIES.map(str::to_string).to_vec();
        known.sort();
        if module_entries != known {
            return Err(bad(format!(
                "module entries are {:?}",
                manifest.module_entries
            )));
        }
        if manifest.target.device_arch != TARGET_ARCH || manifest.container.target != TARGET_ARCH {
            return Err(bad(format!(
                "target is {:?} (container {:?}), not {TARGET_ARCH}",
                manifest.target.device_arch, manifest.container.target
            )));
        }
        if manifest.container.payload_kind != "ptx" || manifest.artifacts.ptx.format != "PTX" {
            return Err(bad(format!(
                "payload is {:?} / {:?}, not PTX",
                manifest.container.payload_kind, manifest.artifacts.ptx.format
            )));
        }
        for pinned in [
            &manifest.artifacts.ptx.sha256,
            &manifest.container.payload_sha256,
        ] {
            if *pinned != sha256 {
                return Err(ArtifactError::Digest {
                    document: format!("the manifest for {expected}"),
                    pinned: pinned.clone(),
                    actual: sha256.clone(),
                });
            }
        }
        let len = ptx.len() as u64;
        if manifest.artifacts.ptx.bytes != len || manifest.container.payload_bytes != len {
            return Err(bad(format!(
                "it pins {} PTX bytes, the payload has {len}",
                manifest.artifacts.ptx.bytes
            )));
        }
        let revision = &manifest.source.revision;
        if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(bad(format!("compiler revision is {revision:?}")));
        }
        match &compiler_revision {
            None => compiler_revision = Some(revision.clone()),
            Some(first) if first != revision => {
                return Err(bad(format!(
                    "compiler revision {revision} differs from {first} in another entry"
                )));
            }
            Some(_) => {}
        }
        if manifest.launch.grid_dim_rule != GRID_RULE
            || manifest.launch.block_dim_rule != BLOCK_RULE
            || manifest.launch.shared_mem_bytes != 0
        {
            return Err(bad(format!(
                "launch rule is grid {:?}, block {:?}, {} shared bytes; the adapter launches grid {GRID_RULE:?}, block {BLOCK_RULE:?}, 0",
                manifest.launch.grid_dim_rule,
                manifest.launch.block_dim_rule,
                manifest.launch.shared_mem_bytes
            )));
        }
        if !ptx_text.contains(&format!(".target {TARGET_ARCH}"))
            || !ptx_text.contains(&format!(".entry {expected}("))
        {
            return Err(bad(format!(
                "the PTX has no `.target {TARGET_ARCH}` module with entry {expected}"
            )));
        }

        let differs = |detail: String| ArtifactError::Schema {
            entry: expected.to_string(),
            detail,
        };
        if schema.schema != SCHEMA_ID {
            return Err(differs(format!("format is {:?}", schema.schema)));
        }
        if schema.abi_version != manifest.abi_version || schema.entry_symbol != expected {
            return Err(differs(format!(
                "it is ABI version {} for entry {:?}",
                schema.abi_version, schema.entry_symbol
            )));
        }
        if schema.source_revision.revision != *revision {
            return Err(differs(format!(
                "compiler revision {}",
                schema.source_revision.revision
            )));
        }
        if schema.launch != manifest.launch {
            return Err(differs("launch rule".to_string()));
        }
        if schema.arguments != manifest.arguments {
            return Err(differs("argument list".to_string()));
        }
        let (Some(activation), Some(weight)) = (
            Lane::parse(&schema.activation_dtype),
            Lane::parse(&schema.weight_dtype),
        ) else {
            return Err(differs(format!(
                "dtypes {:?} / {:?} are not f32, f16 or bf16",
                schema.activation_dtype, schema.weight_dtype
            )));
        };
        let n_reads = activation.n_reads_under(weight);
        if manifest.launch.n_reads != n_reads {
            return Err(bad(format!(
                "launch rule has n_reads {} for {:?} under a {:?} weight; the adapter launches fused_norm_cuda_block(dim, {n_reads})",
                manifest.launch.n_reads, activation, weight
            )));
        }
        if entries
            .iter()
            .any(|e| e.activation == activation && e.weight == weight)
        {
            return Err(bad(
                "another entry already serves this dtype pair".to_string()
            ));
        }
        let roles = argument_roles(&manifest.arguments, activation, weight).map_err(&bad)?;
        entries.push(EntrySpec {
            symbol: expected.to_string(),
            activation,
            weight,
            roles,
        });
    }

    let compiler_revision = compiler_revision.unwrap_or_default();
    Ok(ValidatedArtifact {
        module_name: module_name(&sha256, &compiler_revision),
        ptx: ptx.to_vec(),
        sha256,
        compiler_revision,
        entries,
    })
}
