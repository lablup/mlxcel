// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Persisted identity contract for generic auxiliary VMFBs.
//!
//! The VMFB hash alone is insufficient: a valid bytecode module can still have
//! the wrong config, argument order, or compiler target. Loading therefore
//! compares the actual VMFB, canonical config, ordered resident-weight schema,
//! entry point, and VMFB generation identity against a persisted manifest.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::aux::AuxiliaryWeight;

const SCHEMA: &str = "mlxcel-xla-aux-artifact-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AuxiliaryArtifactContract {
    pub(crate) entry_name: String,
    pub(crate) config_identity: String,
    /// Canonical compiler identity, target flags, and source-MLIR digest.
    pub(crate) generation_identity: String,
}

impl AuxiliaryArtifactContract {
    pub(crate) fn new(
        entry_name: impl Into<String>,
        config_identity: impl Into<String>,
        generation_identity: impl Into<String>,
    ) -> Result<Self, String> {
        let contract = Self {
            entry_name: entry_name.into(),
            config_identity: config_identity.into(),
            generation_identity: generation_identity.into(),
        };
        if contract.entry_name.is_empty()
            || contract.config_identity.is_empty()
            || contract.generation_identity.is_empty()
        {
            return Err(
                "auxiliary entry, config identity, and generation identity must be non-empty"
                    .to_string(),
            );
        }
        Ok(contract)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn weight_schema(weights: &[AuxiliaryWeight]) -> Result<String, String> {
    if weights.is_empty() {
        return Err("auxiliary module requires resident weights".to_string());
    }
    let mut schema = String::new();
    for (index, weight) in weights.iter().enumerate() {
        if weight.name.is_empty() {
            return Err(format!("auxiliary weight {index} name must be non-empty"));
        }
        use std::fmt::Write;
        write!(
            schema,
            "{index}:{}:{:?}:{:?}\n",
            weight.name, weight.dtype, weight.shape
        )
        .expect("writing to String cannot fail");
    }
    Ok(schema)
}

pub(crate) fn auxiliary_manifest_path(vmfb: &Path) -> PathBuf {
    let mut name = vmfb
        .file_name()
        .map_or_else(|| "module.vmfb".into(), |name| name.to_os_string());
    name.push(".aux.json");
    vmfb.with_file_name(name)
}

fn identity_fields(
    vmfb: &Path,
    contract: &AuxiliaryArtifactContract,
    weights: &[AuxiliaryWeight],
) -> Result<[String; 5], String> {
    let vmfb_bytes =
        std::fs::read(vmfb).map_err(|error| format!("read {}: {error}", vmfb.display()))?;
    let schema = weight_schema(weights)?;
    Ok([
        sha256_hex(contract.entry_name.as_bytes()),
        sha256_hex(contract.config_identity.as_bytes()),
        sha256_hex(schema.as_bytes()),
        sha256_hex(contract.generation_identity.as_bytes()),
        sha256_hex(&vmfb_bytes),
    ])
}

fn artifact_digest(fields: &[String; 5]) -> String {
    let mut bytes = SCHEMA.as_bytes().to_vec();
    for field in fields {
        bytes.push(0);
        bytes.extend_from_slice(field.as_bytes());
    }
    sha256_hex(&bytes)
}

pub(crate) fn write_auxiliary_manifest(
    vmfb: &Path,
    contract: &AuxiliaryArtifactContract,
    weights: &[AuxiliaryWeight],
) -> Result<PathBuf, String> {
    let fields = identity_fields(vmfb, contract, weights)?;
    let value = serde_json::json!({
        "schema": SCHEMA,
        "entry_sha256": fields[0],
        "config_sha256": fields[1],
        "weight_schema_sha256": fields[2],
        "generation_sha256": fields[3],
        "vmfb_sha256": fields[4],
        "artifact_sha256": artifact_digest(&fields),
    });
    let path = auxiliary_manifest_path(vmfb);
    let bytes = serde_json::to_vec_pretty(&value)
        .map_err(|error| format!("serialize {}: {error}", path.display()))?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
        .as_nanos();
    let mut temporary_name = path
        .file_name()
        .map_or_else(|| "module.vmfb.aux.json".into(), |name| name.to_os_string());
    temporary_name.push(format!(".{}.{}.tmp", std::process::id(), nonce));
    let temporary = path.with_file_name(temporary_name);
    std::fs::write(&temporary, bytes)
        .map_err(|error| format!("write {}: {error}", temporary.display()))?;
    if let Err(error) = std::fs::rename(&temporary, &path) {
        std::fs::remove_file(&temporary).ok();
        return Err(format!(
            "atomically install {} as {}: {error}",
            temporary.display(),
            path.display()
        ));
    }
    Ok(path)
}

pub(crate) fn verify_auxiliary_manifest(
    vmfb: &Path,
    contract: &AuxiliaryArtifactContract,
    weights: &[AuxiliaryWeight],
) -> Result<u64, String> {
    let path = auxiliary_manifest_path(vmfb);
    let bytes =
        std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("{} must contain a JSON object", path.display()))?;
    let fields = identity_fields(vmfb, contract, weights)?;
    let expected = [
        ("schema", SCHEMA.to_string()),
        ("entry_sha256", fields[0].clone()),
        ("config_sha256", fields[1].clone()),
        ("weight_schema_sha256", fields[2].clone()),
        ("generation_sha256", fields[3].clone()),
        ("vmfb_sha256", fields[4].clone()),
        ("artifact_sha256", artifact_digest(&fields)),
    ];
    for (name, expected_value) in &expected {
        let actual = object.get(*name).and_then(serde_json::Value::as_str);
        if actual != Some(expected_value.as_str()) {
            return Err(format!(
                "auxiliary artifact identity mismatch for {name} in {}",
                path.display()
            ));
        }
    }
    if object.len() != expected.len() {
        return Err(format!(
            "{} contains unknown identity fields",
            path.display()
        ));
    }
    let digest = artifact_digest(&fields);
    let fingerprint = u64::from_str_radix(&digest[..16], 16)
        .map_err(|error| format!("invalid artifact digest: {error}"))?;
    Ok(fingerprint.max(1))
}

/// Reuse a qualified auxiliary VMFB or rebuild and publish one exactly once.
///
/// `compile` must write a complete VMFB to the supplied temporary sibling.
/// The compiler never sees the final cache name, so another loader cannot
/// observe partially-written bytecode. The final VMFB and manifest are each
/// installed by atomic rename after stale pairs have been removed.
pub(crate) fn ensure_qualified_auxiliary_artifact<F>(
    vmfb: &Path,
    contract: &AuxiliaryArtifactContract,
    weights: &[AuxiliaryWeight],
    compile: F,
) -> Result<(), String>
where
    F: FnOnce(&Path) -> Result<(), String>,
{
    let manifest = auxiliary_manifest_path(vmfb);
    if vmfb.is_file()
        && manifest.is_file()
        && verify_auxiliary_manifest(vmfb, contract, weights).is_ok()
    {
        return Ok(());
    }

    remove_file_if_present(vmfb)?;
    remove_file_if_present(&manifest)?;
    let temporary = temporary_sibling(vmfb, "compile");
    remove_file_if_present(&temporary)?;
    if let Err(error) = compile(&temporary) {
        std::fs::remove_file(&temporary).ok();
        return Err(error);
    }
    if !temporary.is_file() {
        return Err(format!(
            "auxiliary compiler did not produce {}",
            temporary.display()
        ));
    }
    if let Err(error) = std::fs::rename(&temporary, vmfb) {
        std::fs::remove_file(&temporary).ok();
        return Err(format!(
            "atomically install {} as {}: {error}",
            temporary.display(),
            vmfb.display()
        ));
    }
    if let Err(error) = write_auxiliary_manifest(vmfb, contract, weights) {
        // A VMFB without its matching identity is never a reusable cache
        // member. Remove it so the next load cannot mistake it for qualified.
        std::fs::remove_file(vmfb).ok();
        std::fs::remove_file(&manifest).ok();
        return Err(error);
    }
    Ok(())
}

fn remove_file_if_present(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove stale {}: {error}", path.display())),
    }
}

fn temporary_sibling(path: &Path, purpose: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let mut name = path
        .file_name()
        .map_or_else(|| "module.vmfb".into(), |name| name.to_os_string());
    name.push(format!(".{purpose}.{}.{}.tmp", std::process::id(), nonce));
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aux::{AuxiliaryWeight, AuxiliaryWeightDType};

    fn temp_vmfb(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "mlxcel-xla-aux-manifest-{tag}-{}-{nonce}.vmfb",
            std::process::id(),
        ))
    }

    fn weights() -> Vec<AuxiliaryWeight> {
        vec![AuxiliaryWeight {
            name: "weight".to_string(),
            bytes: 1.0f32.to_ne_bytes().to_vec(),
            dtype: AuxiliaryWeightDType::Float32,
            shape: vec![1],
        }]
    }

    #[test]
    fn identity_mismatches_fail_closed() {
        let vmfb = temp_vmfb("mismatch");
        std::fs::write(&vmfb, b"vmfb-a").unwrap();
        let contract = AuxiliaryArtifactContract::new(
            "aux.main",
            "config=image:384",
            "compiler=v1;flags=cpu;mlir=abc",
        )
        .unwrap();
        let resident_weights = weights();
        let manifest = write_auxiliary_manifest(&vmfb, &contract, &resident_weights).unwrap();
        assert!(verify_auxiliary_manifest(&vmfb, &contract, &resident_weights).is_ok());

        let wrong_config = AuxiliaryArtifactContract::new(
            "aux.main",
            "config=image:224",
            &contract.generation_identity,
        )
        .unwrap();
        assert!(
            verify_auxiliary_manifest(&vmfb, &wrong_config, &resident_weights)
                .unwrap_err()
                .contains("config_sha256")
        );
        let wrong_generation =
            AuxiliaryArtifactContract::new("aux.main", &contract.config_identity, "compiler=v2")
                .unwrap();
        assert!(
            verify_auxiliary_manifest(&vmfb, &wrong_generation, &resident_weights)
                .unwrap_err()
                .contains("generation_sha256")
        );
        let mut wrong_weights = weights();
        wrong_weights[0].shape = vec![1, 1];
        assert!(
            verify_auxiliary_manifest(&vmfb, &contract, &wrong_weights)
                .unwrap_err()
                .contains("weight_schema_sha256")
        );
        std::fs::write(&vmfb, b"vmfb-b").unwrap();
        assert!(
            verify_auxiliary_manifest(&vmfb, &contract, &resident_weights)
                .unwrap_err()
                .contains("vmfb_sha256")
        );
        std::fs::remove_file(vmfb).ok();
        std::fs::remove_file(manifest).ok();
    }

    #[test]
    fn cold_cache_compiles_once_then_reuses_and_rebuilds_stale_pair_once() {
        let vmfb = temp_vmfb("single-compile");
        let resident_weights = weights();
        let contract =
            AuxiliaryArtifactContract::new("aux.main", "config=v1", "compiler=v1").unwrap();
        let mut compile_count = 0usize;
        ensure_qualified_auxiliary_artifact(&vmfb, &contract, &resident_weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-v1")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();
        assert_eq!(compile_count, 1);
        assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-v1");

        ensure_qualified_auxiliary_artifact(&vmfb, &contract, &resident_weights, |_| {
            compile_count += 1;
            Err("qualified cache must not compile".to_string())
        })
        .unwrap();
        assert_eq!(compile_count, 1);

        let changed =
            AuxiliaryArtifactContract::new("aux.main", "config=v2", "compiler=v1").unwrap();
        ensure_qualified_auxiliary_artifact(&vmfb, &changed, &resident_weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-v2")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();
        assert_eq!(compile_count, 2);
        assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-v2");
        verify_auxiliary_manifest(&vmfb, &changed, &resident_weights).unwrap();

        std::fs::remove_file(auxiliary_manifest_path(&vmfb)).ok();
        std::fs::remove_file(vmfb).ok();
    }
}
