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

//! IREE backend identity shared by the operator micro-oracle probes and the
//! production auxiliary artifacts (issue #3211).
//!
//! Both record the same compiler version string and runtime archive digest in
//! their operator numeric contracts, so a probe report and a production
//! manifest built by one toolchain name the same backend. Issue #3212's CUDA
//! micro-oracles reuse these helpers as well.

use std::path::Path;
use std::process::Command;

use safetensors::Dtype;
use sha2::{Digest, Sha256};

use crate::aux_manifest::AuxiliaryArtifactContract;
use crate::emitter::Precision;
use crate::operator_numeric_contract::{
    CheckpointDType, CheckpointSourceObserver, CheckpointWeights, DeclaredOperator,
    ProductionBackend, production_operator_contract,
};

/// Whitespace-normalized `iree-compile --version` output.
pub(crate) fn compiler_version(compiler: &Path) -> Result<String, String> {
    let output = Command::new(compiler)
        .arg("--version")
        .output()
        .map_err(|error| format!("run {} --version: {error}", compiler.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} --version failed: {}",
            compiler.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let version = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if version.is_empty() {
        return Err(format!(
            "{} --version returned no identity",
            compiler.display()
        ));
    }
    Ok(version)
}

/// Digest of the IREE runtime archive linked into this binary.
///
/// `build.rs` hashes the `libiree_runtime_unified.a` it links (dist, CUDA,
/// ROCm, or macOS source build) into `MLXCEL_XLA_IREE_RUNTIME_SHA256`. A build
/// that could not locate the archive falls back to reading the dist archive
/// next to `compiler`, the layout the CPU probes have always used.
pub(crate) fn runtime_archive_identity(compiler: &Path) -> Result<String, String> {
    if let Some(digest) = option_env!("MLXCEL_XLA_IREE_RUNTIME_SHA256") {
        return Ok(format!("libiree-runtime-sha256-{digest}"));
    }
    let dist = option_env!("MLXCEL_XLA_IREE_DIST")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("IREE_DIST").map(std::path::PathBuf::from))
        .or_else(|| compiler.parent().and_then(Path::parent).map(Path::to_owned))
        .ok_or_else(|| format!("derive IREE dist root from {}", compiler.display()))?;
    let runtime = dist.join("lib/libiree_runtime_unified.a");
    let bytes =
        std::fs::read(&runtime).map_err(|error| format!("read {}: {error}", runtime.display()))?;
    let digest = Sha256::digest(&bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(format!("libiree-runtime-sha256-{hex}"))
}

/// The compiler, runtime, and device identity one artifact is built and run with.
pub(crate) fn production_backend(
    compiler: &Path,
    device: &str,
) -> Result<ProductionBackend, String> {
    Ok(ProductionBackend::new(
        compiler_version(compiler)?,
        runtime_archive_identity(compiler)?,
        device,
    ))
}

/// The floating checkpoint class of a safetensors dtype, if it is one.
pub(crate) fn checkpoint_float_dtype(dtype: Dtype) -> Option<CheckpointDType> {
    match dtype {
        Dtype::F16 => Some(CheckpointDType::F16),
        Dtype::BF16 => Some(CheckpointDType::Bf16),
        Dtype::F32 => Some(CheckpointDType::F32),
        _ => None,
    }
}

/// Record one floating safetensors tensor; other dtypes are classified by the
/// caller (affine U32) or rejected by its decoder.
pub(crate) fn observe_float_tensor(
    observer: &mut CheckpointSourceObserver,
    dtype: Dtype,
) -> Result<(), String> {
    match checkpoint_float_dtype(dtype) {
        Some(dtype) => observer.observe_float(dtype),
        None => Ok(()),
    }
}

/// The numeric inputs of one production family's operator contract.
pub(crate) struct ProductionNumerics<'a> {
    /// Operation id prefix, for example `qwen2-5-vl-vision`.
    pub(crate) family: &'a str,
    pub(crate) operators: &'a [DeclaredOperator],
    /// The precision value the emitter received for this graph.
    pub(crate) precision: Precision,
    pub(crate) weights: CheckpointWeights,
}

/// Build a v3 auxiliary artifact contract that carries the family's operator
/// numeric contract for `device` and the toolchain at `compiler`.
pub(crate) fn production_artifact_contract(
    entry_name: impl Into<String>,
    config_identity: impl Into<String>,
    generation_identity: impl Into<String>,
    compiler: &Path,
    device: &str,
    numerics: &ProductionNumerics<'_>,
) -> Result<AuxiliaryArtifactContract, String> {
    let backend = production_backend(compiler, device)?;
    let operators = production_operator_contract(
        numerics.family,
        numerics.operators,
        numerics.precision,
        numerics.weights,
        &backend,
    )?;
    AuxiliaryArtifactContract::new_with_operator_numeric_contract(
        entry_name,
        config_identity,
        generation_identity,
        operators,
    )
}
