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
//! Families may additionally opt into a versioned dtype/materialization
//! identity. That identity is necessary for, but does not by itself establish,
//! bounded operator-oracle qualification.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use sha2::{Digest, Sha256};

use crate::aux::AuxiliaryWeight;
use crate::numeric_dtype_contract::NumericDTypeContract;

const SCHEMA_V1: &str = "mlxcel-xla-aux-artifact-v1";
const SCHEMA_NUMERIC_DTYPE_V2: &str = "mlxcel-xla-aux-artifact-v2-numeric-dtype";
const CACHE_LOCK_TIMEOUT: Duration = Duration::from_secs(30);
const CACHE_LOCK_STALE_AFTER: Duration = Duration::from_secs(30 * 60);
const CACHE_LOCK_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AuxiliaryArtifactContract {
    pub(crate) entry_name: String,
    pub(crate) config_identity: String,
    /// Canonical compiler identity, target flags, and source-MLIR digest.
    pub(crate) generation_identity: String,
    /// A necessary identity subset, not proof that an operator oracle passed.
    numeric_dtype_contract: Option<NumericDTypeContract>,
}

impl AuxiliaryArtifactContract {
    /// Preserve the pre-numeric v1 identity for existing qualified artifacts.
    ///
    /// New family activation must use a versioned numeric contract instead.
    pub(crate) fn new_legacy_unqualified(
        entry_name: impl Into<String>,
        config_identity: impl Into<String>,
        generation_identity: impl Into<String>,
    ) -> Result<Self, String> {
        Self::build(entry_name, config_identity, generation_identity, None)
    }

    /// Opt into the dtype-aware v2 manifest.
    #[allow(dead_code)] // Adopted explicitly by each family with its complete oracle contract.
    pub(crate) fn new_with_numeric_dtype_contract(
        entry_name: impl Into<String>,
        config_identity: impl Into<String>,
        generation_identity: impl Into<String>,
        numeric_dtype_contract: NumericDTypeContract,
    ) -> Result<Self, String> {
        Self::build(
            entry_name,
            config_identity,
            generation_identity,
            Some(numeric_dtype_contract),
        )
    }

    fn build(
        entry_name: impl Into<String>,
        config_identity: impl Into<String>,
        generation_identity: impl Into<String>,
        numeric_dtype_contract: Option<NumericDTypeContract>,
    ) -> Result<Self, String> {
        let contract = Self {
            entry_name: entry_name.into(),
            config_identity: config_identity.into(),
            generation_identity: generation_identity.into(),
            numeric_dtype_contract,
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

    fn manifest_schema(&self) -> &'static str {
        if self.numeric_dtype_contract.is_some() {
            SCHEMA_NUMERIC_DTYPE_V2
        } else {
            SCHEMA_V1
        }
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
        writeln!(
            schema,
            "{index}:{}:{:?}:{:?}",
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

struct IdentityFields {
    entry: String,
    config: String,
    numeric_dtype_contract: Option<String>,
    weight_schema: String,
    generation: String,
    vmfb: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedManifest {
    schema: String,
    entry_sha256: String,
    config_sha256: String,
    numeric_dtype_contract_sha256: Option<String>,
    weight_schema_sha256: String,
    generation_sha256: String,
    vmfb_sha256: String,
    artifact_sha256: String,
}

fn read_persisted_manifest(path: &Path) -> Result<PersistedManifest, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let persisted: PersistedManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    // `Option<String>` accepts both a missing field and an explicit JSON null.
    // Preserve the exact schema shape by requiring a present field to be text.
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let numeric_field_present = value
        .as_object()
        .is_some_and(|object| object.contains_key("numeric_dtype_contract_sha256"));
    if numeric_field_present != persisted.numeric_dtype_contract_sha256.is_some() {
        return Err(format!(
            "parse {}: numeric_dtype_contract_sha256 must be a string when present",
            path.display()
        ));
    }
    Ok(persisted)
}

fn identity_fields(
    vmfb: &Path,
    contract: &AuxiliaryArtifactContract,
    weights: &[AuxiliaryWeight],
) -> Result<IdentityFields, String> {
    let vmfb_bytes =
        std::fs::read(vmfb).map_err(|error| format!("read {}: {error}", vmfb.display()))?;
    let schema = weight_schema(weights)?;
    Ok(IdentityFields {
        entry: sha256_hex(contract.entry_name.as_bytes()),
        config: sha256_hex(contract.config_identity.as_bytes()),
        numeric_dtype_contract: contract
            .numeric_dtype_contract
            .map(NumericDTypeContract::canonical_identity)
            .map(|identity| sha256_hex(identity.as_bytes())),
        weight_schema: sha256_hex(schema.as_bytes()),
        generation: sha256_hex(contract.generation_identity.as_bytes()),
        vmfb: sha256_hex(&vmfb_bytes),
    })
}

fn artifact_digest(schema: &str, fields: &IdentityFields) -> String {
    let mut bytes = schema.as_bytes().to_vec();
    let ordered = [
        Some(&fields.entry),
        Some(&fields.config),
        fields.numeric_dtype_contract.as_ref(),
        Some(&fields.weight_schema),
        Some(&fields.generation),
        Some(&fields.vmfb),
    ];
    for field in ordered.into_iter().flatten() {
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
    let schema = contract.manifest_schema();
    let mut object = serde_json::Map::new();
    object.insert("schema".to_string(), schema.into());
    object.insert("entry_sha256".to_string(), fields.entry.clone().into());
    object.insert("config_sha256".to_string(), fields.config.clone().into());
    if let Some(numeric_dtype_contract) = &fields.numeric_dtype_contract {
        object.insert(
            "numeric_dtype_contract_sha256".to_string(),
            numeric_dtype_contract.clone().into(),
        );
    }
    object.insert(
        "weight_schema_sha256".to_string(),
        fields.weight_schema.clone().into(),
    );
    object.insert(
        "generation_sha256".to_string(),
        fields.generation.clone().into(),
    );
    object.insert("vmfb_sha256".to_string(), fields.vmfb.clone().into());
    object.insert(
        "artifact_sha256".to_string(),
        artifact_digest(schema, &fields).into(),
    );
    let value = serde_json::Value::Object(object);
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
    let persisted = read_persisted_manifest(&path)?;
    let fields = identity_fields(vmfb, contract, weights)?;
    let schema = contract.manifest_schema();
    let artifact = artifact_digest(schema, &fields);
    if persisted.schema != schema {
        return Err(format!(
            "auxiliary artifact identity mismatch for schema in {}",
            path.display()
        ));
    }
    if persisted.numeric_dtype_contract_sha256.as_deref()
        != fields.numeric_dtype_contract.as_deref()
    {
        return Err(format!(
            "auxiliary artifact identity mismatch for numeric_dtype_contract_sha256 in {}",
            path.display()
        ));
    }
    let expected = [
        (
            "entry_sha256",
            persisted.entry_sha256.as_str(),
            fields.entry.as_str(),
        ),
        (
            "config_sha256",
            persisted.config_sha256.as_str(),
            fields.config.as_str(),
        ),
        (
            "weight_schema_sha256",
            persisted.weight_schema_sha256.as_str(),
            fields.weight_schema.as_str(),
        ),
        (
            "generation_sha256",
            persisted.generation_sha256.as_str(),
            fields.generation.as_str(),
        ),
        (
            "vmfb_sha256",
            persisted.vmfb_sha256.as_str(),
            fields.vmfb.as_str(),
        ),
        (
            "artifact_sha256",
            persisted.artifact_sha256.as_str(),
            artifact.as_str(),
        ),
    ];
    for (name, actual, expected_value) in expected {
        if actual != expected_value {
            return Err(format!(
                "auxiliary artifact identity mismatch for {name} in {}",
                path.display()
            ));
        }
    }
    let fingerprint = u64::from_str_radix(&artifact[..16], 16)
        .map_err(|error| format!("invalid artifact digest: {error}"))?;
    Ok(fingerprint.max(1))
}

fn manifest_schema_version(schema: &str) -> Option<u32> {
    let version = schema.strip_prefix("mlxcel-xla-aux-artifact-v")?;
    let digits = version
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digits == 0 {
        None
    } else {
        version[..digits].parse().ok()
    }
}

fn reject_manifest_downgrade(
    manifest: &Path,
    contract: &AuxiliaryArtifactContract,
) -> Result<(), String> {
    let persisted = read_persisted_manifest(manifest).map_err(|error| {
        format!("refusing to replace an unclassified auxiliary artifact manifest: {error}")
    })?;
    let actual_schema = persisted.schema.as_str();
    let expected_schema = contract.manifest_schema();
    if actual_schema == expected_schema {
        return Ok(());
    }
    let actual_version = manifest_schema_version(actual_schema);
    let expected_version = manifest_schema_version(expected_schema);
    if actual_version
        .zip(expected_version)
        .is_some_and(|(actual_version, expected_version)| actual_version < expected_version)
    {
        return Ok(());
    }
    Err(format!(
        "refusing to replace auxiliary artifact manifest {} from schema {actual_schema} \
         with {expected_schema}",
        manifest.display()
    ))
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
    let _lock = acquire_auxiliary_cache_lock(vmfb)?;
    let manifest = auxiliary_manifest_path(vmfb);
    if vmfb.is_file() && manifest.is_file() {
        if verify_auxiliary_manifest(vmfb, contract, weights).is_ok() {
            return Ok(());
        }
        reject_manifest_downgrade(&manifest, contract)?;
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
    if let Err(error) = verify_auxiliary_manifest(vmfb, contract, weights) {
        std::fs::remove_file(vmfb).ok();
        std::fs::remove_file(&manifest).ok();
        return Err(format!(
            "verify newly published auxiliary artifact {}: {error}",
            vmfb.display()
        ));
    }
    Ok(())
}

struct AuxiliaryCacheLock {
    _file: File,
    path: PathBuf,
    token: String,
}

impl Drop for AuxiliaryCacheLock {
    fn drop(&mut self) {
        // A stale owner may have had its lock replaced. Only remove the marker
        // if it is still the exact one this guard created.
        if std::fs::read_to_string(&self.path)
            .ok()
            .is_some_and(|contents| contents == self.token)
        {
            std::fs::remove_file(&self.path).ok();
        }
    }
}

fn acquire_auxiliary_cache_lock(vmfb: &Path) -> Result<AuxiliaryCacheLock, String> {
    acquire_auxiliary_cache_lock_with_policy(
        vmfb,
        CACHE_LOCK_TIMEOUT,
        CACHE_LOCK_STALE_AFTER,
        CACHE_LOCK_POLL_INTERVAL,
    )
}

fn acquire_auxiliary_cache_lock_with_policy(
    vmfb: &Path,
    timeout: Duration,
    stale_after: Duration,
    poll_interval: Duration,
) -> Result<AuxiliaryCacheLock, String> {
    let path = cache_lock_path(vmfb);
    let started = Instant::now();
    loop {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                let token = format!("pid={};nonce={}", std::process::id(), unique_nonce());
                if let Err(error) = file.write_all(token.as_bytes()) {
                    std::fs::remove_file(&path).ok();
                    return Err(format!("initialize cache lock {}: {error}", path.display()));
                }
                return Ok(AuxiliaryCacheLock {
                    _file: file,
                    path,
                    token,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if cache_lock_is_stale(&path, stale_after) {
                    remove_file_if_present(&path)?;
                    continue;
                }
                let elapsed = started.elapsed();
                if elapsed >= timeout {
                    return Err(format!(
                        "timed out after {:.3}s waiting for auxiliary cache lock {}",
                        elapsed.as_secs_f64(),
                        path.display()
                    ));
                }
                std::thread::sleep(poll_interval.min(timeout.saturating_sub(elapsed)));
            }
            Err(error) => {
                return Err(format!("create cache lock {}: {error}", path.display()));
            }
        }
    }
}

fn cache_lock_is_stale(path: &Path, stale_after: Duration) -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(token) = std::fs::read_to_string(path) {
            if let Some(pid) = cache_lock_owner_pid(&token) {
                match Path::new("/proc").join(pid.to_string()).try_exists() {
                    Ok(false) => return true,
                    Ok(true) => {}
                    Err(_) => {}
                }
            }
        }
    }
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age >= stale_after)
}

#[cfg(target_os = "linux")]
fn cache_lock_owner_pid(token: &str) -> Option<u32> {
    let mut fields = token.split(';');
    let pid = fields.next()?.strip_prefix("pid=")?.parse().ok()?;
    fields
        .next()?
        .strip_prefix("nonce=")?
        .parse::<u128>()
        .ok()?;
    fields.next().is_none().then_some(pid)
}

fn cache_lock_path(vmfb: &Path) -> PathBuf {
    let mut name = vmfb
        .file_name()
        .map_or_else(|| "module.vmfb".into(), |name| name.to_os_string());
    name.push(".lock");
    vmfb.with_file_name(name)
}

fn remove_file_if_present(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove stale {}: {error}", path.display())),
    }
}

fn temporary_sibling(path: &Path, purpose: &str) -> PathBuf {
    let mut name = path
        .file_name()
        .map_or_else(|| "module.vmfb".into(), |name| name.to_os_string());
    name.push(format!(
        ".{purpose}.{}.{}.tmp",
        std::process::id(),
        unique_nonce()
    ));
    path.with_file_name(name)
}

fn unique_nonce() -> u128 {
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::aux::{AuxiliaryWeight, AuxiliaryWeightDType};
    use crate::numeric_dtype_contract::{NumericDType, NumericDTypeContract, WeightExecution};

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

    fn numeric_dtype_contract() -> NumericDTypeContract {
        NumericDTypeContract::new(
            WeightExecution::HostAffineDequantized,
            NumericDType::F32,
            NumericDType::F32,
            NumericDType::F32,
            NumericDType::F32,
            NumericDType::F32,
        )
    }

    #[test]
    fn identity_mismatches_fail_closed() {
        let vmfb = temp_vmfb("mismatch");
        std::fs::write(&vmfb, b"vmfb-a").unwrap();
        let contract = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=image:384",
            "compiler=v1;flags=cpu;mlir=abc",
        )
        .unwrap();
        let resident_weights = weights();
        let manifest = write_auxiliary_manifest(&vmfb, &contract, &resident_weights).unwrap();
        assert!(verify_auxiliary_manifest(&vmfb, &contract, &resident_weights).is_ok());

        let wrong_config = AuxiliaryArtifactContract::new_legacy_unqualified(
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
        let wrong_generation = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            &contract.config_identity,
            "compiler=v2",
        )
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
    fn numeric_dtype_contract_mismatch_fails_closed() {
        let vmfb = temp_vmfb("numeric-mismatch");
        std::fs::write(&vmfb, b"vmfb-numeric").unwrap();
        let contract = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            numeric_dtype_contract(),
        )
        .unwrap();
        let resident_weights = weights();
        let manifest = write_auxiliary_manifest(&vmfb, &contract, &resident_weights).unwrap();
        assert!(verify_auxiliary_manifest(&vmfb, &contract, &resident_weights).is_ok());

        let changed_numeric = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            NumericDTypeContract {
                contraction_input: NumericDType::Bf16,
                ..numeric_dtype_contract()
            },
        )
        .unwrap();
        assert!(
            verify_auxiliary_manifest(&vmfb, &changed_numeric, &resident_weights)
                .unwrap_err()
                .contains("numeric_dtype_contract_sha256")
        );

        let v1_contract = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=v1",
            "compiler=v1",
        )
        .unwrap();
        assert!(
            verify_auxiliary_manifest(&vmfb, &v1_contract, &resident_weights)
                .unwrap_err()
                .contains("schema")
        );

        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("numeric_dtype_contract_sha256");
        std::fs::write(&manifest, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(
            verify_auxiliary_manifest(&vmfb, &contract, &resident_weights)
                .unwrap_err()
                .contains("numeric_dtype_contract_sha256")
        );

        write_auxiliary_manifest(&vmfb, &contract, &resident_weights).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unexpected".to_string(), "field".into());
        std::fs::write(&manifest, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(
            verify_auxiliary_manifest(&vmfb, &contract, &resident_weights)
                .unwrap_err()
                .contains("unknown field")
        );

        std::fs::remove_file(vmfb).ok();
        std::fs::remove_file(manifest).ok();
    }

    #[test]
    fn v1_artifact_digest_remains_legacy_compatible() {
        let vmfb = temp_vmfb("v1-digest");
        std::fs::write(&vmfb, b"legacy-vmfb").unwrap();
        let contract = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=v1",
            "compiler=v1",
        )
        .unwrap();
        let resident_weights = weights();
        let fields = identity_fields(&vmfb, &contract, &resident_weights).unwrap();
        assert!(fields.numeric_dtype_contract.is_none());

        let mut legacy_bytes = SCHEMA_V1.as_bytes().to_vec();
        for field in [
            &fields.entry,
            &fields.config,
            &fields.weight_schema,
            &fields.generation,
            &fields.vmfb,
        ] {
            legacy_bytes.push(0);
            legacy_bytes.extend_from_slice(field.as_bytes());
        }
        assert_eq!(
            artifact_digest(SCHEMA_V1, &fields),
            sha256_hex(&legacy_bytes)
        );

        let manifest = write_auxiliary_manifest(&vmfb, &contract, &resident_weights).unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
        assert_eq!(value["schema"], SCHEMA_V1);
        assert!(value.get("numeric_dtype_contract_sha256").is_none());
        assert_eq!(value.as_object().unwrap().len(), 7);

        let mut null_numeric = value;
        null_numeric.as_object_mut().unwrap().insert(
            "numeric_dtype_contract_sha256".to_string(),
            serde_json::Value::Null,
        );
        std::fs::write(&manifest, serde_json::to_vec_pretty(&null_numeric).unwrap()).unwrap();
        assert!(
            verify_auxiliary_manifest(&vmfb, &contract, &resident_weights)
                .unwrap_err()
                .contains("must be a string")
        );

        std::fs::remove_file(vmfb).ok();
        std::fs::remove_file(manifest).ok();
    }

    #[test]
    fn numeric_dtype_contract_changes_fingerprint_for_same_artifact() {
        let vmfb = temp_vmfb("numeric-fingerprint");
        std::fs::write(&vmfb, b"same-vmfb").unwrap();
        let resident_weights = weights();
        let contract = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            numeric_dtype_contract(),
        )
        .unwrap();
        let manifest = write_auxiliary_manifest(&vmfb, &contract, &resident_weights).unwrap();
        let baseline = verify_auxiliary_manifest(&vmfb, &contract, &resident_weights).unwrap();

        let changed = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            NumericDTypeContract {
                reduction_compute: NumericDType::Bf16,
                ..numeric_dtype_contract()
            },
        )
        .unwrap();
        write_auxiliary_manifest(&vmfb, &changed, &resident_weights).unwrap();
        let changed_fingerprint =
            verify_auxiliary_manifest(&vmfb, &changed, &resident_weights).unwrap();
        assert_ne!(baseline, changed_fingerprint);

        std::fs::remove_file(vmfb).ok();
        std::fs::remove_file(manifest).ok();
    }

    #[test]
    fn cold_cache_compiles_once_then_reuses_and_rebuilds_stale_pair_once() {
        let vmfb = temp_vmfb("single-compile");
        let resident_weights = weights();
        let contract = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=v1",
            "compiler=v1",
        )
        .unwrap();
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

        let changed = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=v2",
            "compiler=v1",
        )
        .unwrap();
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

    #[test]
    fn numeric_dtype_contract_change_rebuilds_once_then_reuses() {
        let vmfb = temp_vmfb("numeric-rebuild");
        let resident_weights = weights();
        let contract = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            numeric_dtype_contract(),
        )
        .unwrap();
        let mut compile_count = 0usize;
        ensure_qualified_auxiliary_artifact(&vmfb, &contract, &resident_weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-f32")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();
        assert_eq!(compile_count, 1);

        let changed = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            NumericDTypeContract {
                activation_carrier: NumericDType::Bf16,
                ..numeric_dtype_contract()
            },
        )
        .unwrap();
        ensure_qualified_auxiliary_artifact(&vmfb, &changed, &resident_weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-bf16")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();
        assert_eq!(compile_count, 2);
        assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-bf16");

        ensure_qualified_auxiliary_artifact(&vmfb, &changed, &resident_weights, |_| {
            compile_count += 1;
            Err("qualified numeric cache must not compile".to_string())
        })
        .unwrap();
        assert_eq!(compile_count, 2);

        std::fs::remove_file(auxiliary_manifest_path(&vmfb)).ok();
        std::fs::remove_file(vmfb).ok();
    }

    #[test]
    fn v1_cache_migrates_to_numeric_dtype_contract_once() {
        let vmfb = temp_vmfb("numeric-migrate");
        let resident_weights = weights();
        let v1_contract = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=v1",
            "compiler=v1",
        )
        .unwrap();
        let dtype_contract = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            numeric_dtype_contract(),
        )
        .unwrap();
        let mut compile_count = 0usize;

        ensure_qualified_auxiliary_artifact(&vmfb, &v1_contract, &resident_weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-v1")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();
        assert_eq!(compile_count, 1);

        ensure_qualified_auxiliary_artifact(
            &vmfb,
            &dtype_contract,
            &resident_weights,
            |temporary| {
                compile_count += 1;
                std::fs::write(temporary, b"vmfb-v2")
                    .map_err(|error| format!("write test VMFB: {error}"))
            },
        )
        .unwrap();
        assert_eq!(compile_count, 2);
        assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-v2");

        ensure_qualified_auxiliary_artifact(&vmfb, &dtype_contract, &resident_weights, |_| {
            compile_count += 1;
            Err("migrated numeric cache must not compile".to_string())
        })
        .unwrap();
        assert_eq!(compile_count, 2);

        std::fs::remove_file(auxiliary_manifest_path(&vmfb)).ok();
        std::fs::remove_file(vmfb).ok();
    }

    #[test]
    fn numeric_dtype_cache_refuses_legacy_downgrade() {
        let vmfb = temp_vmfb("numeric-downgrade");
        let resident_weights = weights();
        let dtype_contract = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            numeric_dtype_contract(),
        )
        .unwrap();
        let mut compile_count = 0usize;
        ensure_qualified_auxiliary_artifact(
            &vmfb,
            &dtype_contract,
            &resident_weights,
            |temporary| {
                compile_count += 1;
                std::fs::write(temporary, b"vmfb-v2")
                    .map_err(|error| format!("write test VMFB: {error}"))
            },
        )
        .unwrap();
        let manifest = auxiliary_manifest_path(&vmfb);
        let manifest_before = std::fs::read(&manifest).unwrap();

        let v1_contract = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=v1",
            "compiler=v1",
        )
        .unwrap();
        let error =
            ensure_qualified_auxiliary_artifact(&vmfb, &v1_contract, &resident_weights, |_| {
                compile_count += 1;
                Err("downgrade must not compile".to_string())
            })
            .unwrap_err();
        assert!(error.contains("refusing to replace"));
        assert_eq!(compile_count, 1);
        assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-v2");
        assert_eq!(std::fs::read(&manifest).unwrap(), manifest_before);
        assert!(verify_auxiliary_manifest(&vmfb, &dtype_contract, &resident_weights).is_ok());

        std::fs::remove_file(manifest).ok();
        std::fs::remove_file(vmfb).ok();
    }

    #[test]
    fn unclassified_manifest_cannot_be_replaced_by_legacy_contract() {
        let vmfb = temp_vmfb("numeric-unclassified");
        std::fs::write(&vmfb, b"vmfb-v2").unwrap();
        let resident_weights = weights();
        let dtype_contract = AuxiliaryArtifactContract::new_with_numeric_dtype_contract(
            "aux.main",
            "config=v1",
            "compiler=v1",
            numeric_dtype_contract(),
        )
        .unwrap();
        let manifest = write_auxiliary_manifest(&vmfb, &dtype_contract, &resident_weights).unwrap();
        let original = std::fs::read(&manifest).unwrap();
        let original_text = String::from_utf8(original.clone()).unwrap();
        let duplicate_schema = format!(
            "{{\"schema\":\"{SCHEMA_V1}\",{}",
            original_text.strip_prefix('{').unwrap()
        )
        .into_bytes();

        let mut unknown_schema: serde_json::Value = serde_json::from_slice(&original).unwrap();
        unknown_schema["schema"] = "mlxcel-xla-aux-artifact-v2-unknown".into();
        let unknown_schema = serde_json::to_vec(&unknown_schema).unwrap();

        let mut overflow_schema: serde_json::Value = serde_json::from_slice(&original).unwrap();
        overflow_schema["schema"] = "mlxcel-xla-aux-artifact-v999999999999999999999".into();
        let overflow_schema = serde_json::to_vec(&overflow_schema).unwrap();

        let mut missing_schema: serde_json::Value = serde_json::from_slice(&original).unwrap();
        missing_schema.as_object_mut().unwrap().remove("schema");
        let missing_schema = serde_json::to_vec(&missing_schema).unwrap();

        let v1_contract = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=v1",
            "compiler=v1",
        )
        .unwrap();
        for (case, bytes) in [
            ("malformed", b"{".to_vec()),
            ("duplicate schema", duplicate_schema),
            ("unknown schema", unknown_schema),
            ("overflow schema", overflow_schema),
            ("missing schema", missing_schema),
        ] {
            std::fs::write(&manifest, &bytes).unwrap();
            let mut compile_called = false;
            let error =
                ensure_qualified_auxiliary_artifact(&vmfb, &v1_contract, &resident_weights, |_| {
                    compile_called = true;
                    Ok(())
                })
                .unwrap_err();
            assert!(!compile_called, "{case}");
            assert!(error.contains("refusing to"), "{case}: {error}");
            assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-v2", "{case}");
            assert_eq!(std::fs::read(&manifest).unwrap(), bytes, "{case}");
        }

        std::fs::remove_file(manifest).ok();
        std::fs::remove_file(vmfb).ok();
    }

    #[test]
    fn concurrent_loaders_publish_one_qualified_artifact() {
        let vmfb = Arc::new(temp_vmfb("concurrent"));
        let contract = AuxiliaryArtifactContract::new_legacy_unqualified(
            "aux.main",
            "config=v1",
            "compiler=v1",
        )
        .unwrap();
        let compile_count = Arc::new(AtomicUsize::new(0));
        let ready = Arc::new(Barrier::new(8));
        let threads = (0..8)
            .map(|_| {
                let vmfb = Arc::clone(&vmfb);
                let contract = contract.clone();
                let compile_count = Arc::clone(&compile_count);
                let ready = Arc::clone(&ready);
                std::thread::spawn(move || {
                    ready.wait();
                    ensure_qualified_auxiliary_artifact(
                        vmfb.as_ref(),
                        &contract,
                        &weights(),
                        |temporary| {
                            compile_count.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(Duration::from_millis(25));
                            std::fs::write(temporary, b"one-vmfb")
                                .map_err(|error| format!("write test VMFB: {error}"))
                        },
                    )
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap().unwrap();
        }
        assert_eq!(compile_count.load(Ordering::SeqCst), 1);
        verify_auxiliary_manifest(vmfb.as_ref(), &contract, &weights()).unwrap();

        std::fs::remove_file(auxiliary_manifest_path(vmfb.as_ref())).ok();
        std::fs::remove_file(vmfb.as_ref()).ok();
        std::fs::remove_file(cache_lock_path(vmfb.as_ref())).ok();
    }

    #[test]
    fn cache_lock_has_bounded_timeout_and_recovers_stale_marker() {
        let vmfb = temp_vmfb("lock-policy");
        let held = acquire_auxiliary_cache_lock_with_policy(
            &vmfb,
            Duration::from_secs(1),
            Duration::from_secs(1),
            Duration::from_millis(1),
        )
        .unwrap();
        let error = acquire_auxiliary_cache_lock_with_policy(
            &vmfb,
            Duration::from_millis(20),
            Duration::from_secs(1),
            Duration::from_millis(1),
        )
        .err()
        .expect("a live lock must time out");
        assert!(error.contains("timed out"));
        drop(held);

        let lock = cache_lock_path(&vmfb);
        #[cfg(target_os = "linux")]
        for token in [b"".as_slice(), b"damaged".as_slice(), &[0xff]] {
            std::fs::write(&lock, token).unwrap();
            assert!(
                !cache_lock_is_stale(&lock, Duration::from_secs(1)),
                "a fresh incomplete owner token must remain live"
            );
            std::thread::sleep(Duration::from_millis(5));
            let recovered = acquire_auxiliary_cache_lock_with_policy(
                &vmfb,
                Duration::from_millis(50),
                Duration::from_millis(1),
                Duration::from_millis(1),
            )
            .unwrap();
            drop(recovered);
            assert!(!lock.exists());
        }
        std::fs::write(
            &lock,
            format!("pid={};nonce={}", std::process::id(), unique_nonce()),
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let recovered = acquire_auxiliary_cache_lock_with_policy(
            &vmfb,
            Duration::from_millis(50),
            Duration::from_millis(1),
            Duration::from_millis(1),
        )
        .unwrap();
        drop(recovered);
        assert!(!lock.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_lock_immediately_recovers_an_impossible_dead_pid() {
        let vmfb = temp_vmfb("dead-owner");
        let lock = cache_lock_path(&vmfb);
        let impossible_pid = u32::MAX;
        assert!(
            !Path::new("/proc").join(impossible_pid.to_string()).exists(),
            "test PID unexpectedly exists"
        );
        std::fs::write(&lock, format!("pid={impossible_pid};nonce=1")).unwrap();

        let recovered = acquire_auxiliary_cache_lock_with_policy(
            &vmfb,
            Duration::from_millis(50),
            Duration::from_secs(60 * 60),
            Duration::from_millis(1),
        )
        .unwrap();
        drop(recovered);
        assert!(!lock.exists());
    }
}
