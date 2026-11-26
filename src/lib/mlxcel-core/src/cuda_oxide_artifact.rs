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
//! and the compiler revision. MLX's PTX disk cache is keyed on that name and is
//! consulted before the payload is, so a changed artifact gets a new name and
//! can never be served a stale entry.

use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

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
/// Number of kernel parameters under ABI version 1.
pub const ARG_COUNT: usize = 13;

const GRID_RULE: &str = "rows";
const BLOCK_RULE: &str = "fused_norm_threads(dim)";

/// Element type of an activation or weight buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lane {
    F32,
    F16,
    Bf16,
}

impl Lane {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }

    /// How the kernel sees one element: 16-bit lanes travel as bit patterns.
    fn pointee(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 | Self::Bf16 => "u16",
        }
    }

    /// The MLX dtype of a buffer with this element type.
    #[must_use]
    pub fn mlx_dtype(self) -> i32 {
        match self {
            Self::F32 => crate::dtype::FLOAT32,
            Self::F16 => crate::dtype::FLOAT16,
            Self::Bf16 => crate::dtype::BFLOAT16,
        }
    }

    /// Wire code of `mlxcel_oxide_fused_norm_install` (`fused_norm_oxide.cpp`).
    pub(crate) fn code(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 2,
        }
    }
}

/// What one kernel parameter carries. The discriminants are the wire format of
/// `mlxcel_oxide_fused_norm_install` and mirror `Role` in
/// `fused_norm_oxide.cpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArgRole {
    XPtr = 0,
    XLen = 1,
    ResidualPtr = 2,
    ResidualLen = 3,
    WeightPtr = 4,
    WeightLen = 5,
    Eps = 6,
    WeightBias = 7,
    Dim = 8,
    NormedPtr = 9,
    NormedLen = 10,
    NewResidualPtr = 11,
    NewResidualLen = 12,
}

/// Which buffer a pointer or length parameter belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Buffer {
    Activation,
    Weight,
}

/// What the ABI requires of a parameter with a given role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    Pointer {
        buffer: Buffer,
        write: bool,
        length: ArgRole,
    },
    Length,
    Scalar(&'static str),
}

impl ArgRole {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "x_ptr" => Self::XPtr,
            "x_len" => Self::XLen,
            "residual_ptr" => Self::ResidualPtr,
            "residual_len" => Self::ResidualLen,
            "weight_ptr" => Self::WeightPtr,
            "weight_len" => Self::WeightLen,
            "eps" => Self::Eps,
            "weight_bias" => Self::WeightBias,
            "dim" => Self::Dim,
            "normed_ptr" => Self::NormedPtr,
            "normed_len" => Self::NormedLen,
            "new_residual_ptr" => Self::NewResidualPtr,
            "new_residual_len" => Self::NewResidualLen,
            _ => return None,
        })
    }

    fn expect(self) -> Expect {
        let pointer = |buffer, write, length| Expect::Pointer {
            buffer,
            write,
            length,
        };
        match self {
            Self::XPtr => pointer(Buffer::Activation, false, Self::XLen),
            Self::ResidualPtr => pointer(Buffer::Activation, false, Self::ResidualLen),
            Self::WeightPtr => pointer(Buffer::Weight, false, Self::WeightLen),
            Self::NormedPtr => pointer(Buffer::Activation, true, Self::NormedLen),
            Self::NewResidualPtr => pointer(Buffer::Activation, true, Self::NewResidualLen),
            Self::XLen
            | Self::ResidualLen
            | Self::WeightLen
            | Self::NormedLen
            | Self::NewResidualLen => Expect::Length,
            Self::Eps | Self::WeightBias => Expect::Scalar("f32"),
            Self::Dim => Expect::Scalar("u32"),
        }
    }
}

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
struct Argument {
    ordinal: usize,
    name: String,
    cuda_param_type: String,
    width_bits: u32,
    kind: String,
    #[serde(default)]
    pointee: Option<String>,
    #[serde(default)]
    access: Option<String>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    length_argument: Option<usize>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Launch {
    grid_dim_rule: String,
    block_dim_rule: String,
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
/// compiled code, so the PTX disk cache cannot return another artifact's entry.
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

/// Map the manifest's argument list to roles, refusing anything the packing in
/// `fused_norm_oxide.cpp` would get wrong: a missing or repeated parameter, a
/// width other than the one the role is packed with, a buffer of another
/// element type, or a pointer whose length parameter is not its own.
fn argument_roles(
    arguments: &[Argument],
    activation: Lane,
    weight: Lane,
) -> Result<[ArgRole; ARG_COUNT], String> {
    if arguments.len() != ARG_COUNT {
        return Err(format!(
            "{} kernel parameters, the adapter packs {ARG_COUNT}",
            arguments.len()
        ));
    }
    let mut roles = Vec::with_capacity(ARG_COUNT);
    for (index, arg) in arguments.iter().enumerate() {
        if arg.ordinal != index {
            return Err(format!(
                "parameter {index} has ordinal {} (not in order)",
                arg.ordinal
            ));
        }
        let role = ArgRole::from_name(&arg.name)
            .ok_or_else(|| format!("unknown parameter {:?}", arg.name))?;
        if roles.contains(&role) {
            return Err(format!("parameter {:?} appears twice", arg.name));
        }
        roles.push(role);
    }
    for (arg, role) in arguments.iter().zip(&roles) {
        let wrong = |what: &str| format!("parameter {:?} {what}", arg.name);
        let is = |ty: &str, bits: u32, kind: &str| {
            arg.cuda_param_type == ty && arg.width_bits == bits && arg.kind == kind
        };
        match role.expect() {
            Expect::Pointer {
                buffer,
                write,
                length,
            } => {
                let lane = match buffer {
                    Buffer::Activation => activation,
                    Buffer::Weight => weight,
                };
                if !is("u64", 64, "device_pointer") {
                    return Err(wrong("is not a 64-bit device pointer"));
                }
                if arg.pointee.as_deref() != Some(lane.pointee()) {
                    return Err(wrong("points at another element type"));
                }
                if arg.access.as_deref() != Some(if write { "write" } else { "read" }) {
                    return Err(wrong("has the wrong access"));
                }
                if arg.length_argument.and_then(|i| roles.get(i)) != Some(&length) {
                    return Err(wrong("is not paired with its own length"));
                }
            }
            Expect::Length => {
                if !is("u64", 64, "length") || arg.unit.as_deref() != Some("elements") {
                    return Err(wrong("is not a 64-bit element count"));
                }
            }
            Expect::Scalar(ty) => {
                if !is(ty, 32, "value") {
                    return Err(wrong("has the wrong scalar type"));
                }
            }
        }
    }
    roles
        .try_into()
        .map_err(|_| "parameter count changed during validation".to_string())
}
