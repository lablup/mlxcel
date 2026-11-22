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

//! Cache-rejection tests for production operator contracts (issue #3211).
//!
//! `IreeAuxiliaryModule::load` runs `verify_auxiliary_manifest` before it
//! creates a session, and `ensure_qualified_auxiliary_artifact` recompiles when
//! that check fails. Each test mutates one field class of the contract a
//! production family builds and asserts both halves.

use std::path::PathBuf;

use crate::aux_manifest::{
    AuxiliaryArtifactContract, auxiliary_manifest_path, ensure_qualified_auxiliary_artifact,
    qualify_auxiliary_artifact, schema_scoped_vmfb_path, verify_auxiliary_manifest,
};
use crate::auxiliary::{AuxiliaryWeight, AuxiliaryWeightDType, AuxiliaryWeightStorage};
use crate::emitter::Precision;
use crate::numeric_dtype_contract::NumericDType;

use super::families::qwen2_vl_operators;
use super::production::{
    AffineExecution, CheckpointSourceObserver, CheckpointWeights, ProductionBackend,
    production_operator_contract,
};
use super::{AssociationPolicy, CheckpointDType, OperatorNumericContractSet};

fn temp_vmfb(tag: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "mlxcel-xla-production-contract-{tag}-{}-{nonce}.vmfb",
        std::process::id(),
    ))
}

fn weights() -> Vec<AuxiliaryWeight> {
    vec![AuxiliaryWeight {
        name: "weight".to_string(),
        storage: AuxiliaryWeightStorage::Float32(vec![1.0]),
        dtype: AuxiliaryWeightDType::Float32,
        shape: vec![1],
    }]
}

fn backend(compiler: &str, runtime: &str, device: &str) -> ProductionBackend {
    ProductionBackend::new(compiler, runtime, device)
}

fn baseline_backend() -> ProductionBackend {
    backend(
        "IREE compiler version 3.12.0rc20260721",
        "libiree-runtime-sha256-aaaa",
        "cuda",
    )
}

fn float_weights(dtype: CheckpointDType) -> CheckpointWeights {
    let mut observer = CheckpointSourceObserver::default();
    observer.observe_float(dtype).unwrap();
    observer.resolve(AffineExecution::Host).unwrap()
}

fn operator_set(
    precision: Precision,
    weights: CheckpointWeights,
    backend: &ProductionBackend,
) -> OperatorNumericContractSet {
    production_operator_contract(
        "qwen2-5-vl-vision",
        &qwen2_vl_operators(true),
        precision,
        weights,
        backend,
    )
    .unwrap()
}

fn artifact(operators: OperatorNumericContractSet) -> AuxiliaryArtifactContract {
    AuxiliaryArtifactContract::new_with_operator_numeric_contract(
        "vision.main",
        "config=qwen2-5-vl",
        "compiler=iree;mlir_sha256=0",
        operators,
    )
    .unwrap()
}

/// Publish `baseline`, then assert `changed` fails verification (the check
/// `IreeAuxiliaryModule::load` runs) and recompiles exactly once.
fn assert_rejected_then_recompiled(
    name: &str,
    baseline: &AuxiliaryArtifactContract,
    changed: &AuxiliaryArtifactContract,
) {
    let vmfb = temp_vmfb(name);
    let resident = weights();
    let mut compiles = 0usize;
    let mut compile = |output: &std::path::Path| {
        compiles += 1;
        std::fs::write(output, format!("vmfb-{compiles}")).map_err(|error| error.to_string())
    };
    ensure_qualified_auxiliary_artifact(&vmfb, baseline, &resident, &mut compile).unwrap();
    assert!(verify_auxiliary_manifest(&vmfb, baseline, &resident).is_ok());
    let error = verify_auxiliary_manifest(&vmfb, changed, &resident).unwrap_err();
    assert!(
        error.contains("operator_numeric_contract_sha256"),
        "{name}: unexpected rejection {error}"
    );
    ensure_qualified_auxiliary_artifact(&vmfb, changed, &resident, &mut compile).unwrap();
    assert_eq!(compiles, 2, "{name} must recompile exactly once");
    assert!(verify_auxiliary_manifest(&vmfb, changed, &resident).is_ok());
    assert!(verify_auxiliary_manifest(&vmfb, baseline, &resident).is_err());
    std::fs::remove_file(auxiliary_manifest_path(&vmfb)).ok();
    std::fs::remove_file(vmfb).ok();
}

#[test]
fn every_contract_field_class_rejects_a_stale_cached_artifact() {
    let base_backend = baseline_backend();
    let base_weights = float_weights(CheckpointDType::F16);
    let baseline_set = operator_set(Precision::F16, base_weights, &base_backend);
    let baseline = artifact(baseline_set.clone());

    let mut accumulator = baseline_set.clone();
    accumulator.operators[0].accumulator = NumericDType::Bf16;
    let mut association = baseline_set.clone();
    let matmul = association
        .operators
        .iter_mut()
        .find(|operator| operator.association == AssociationPolicy::BackendAlgorithm)
        .unwrap();
    matmul.association = AssociationPolicy::PairwiseTree;

    let variants = [
        (
            "operator-dtype",
            operator_set(
                Precision::F16,
                float_weights(CheckpointDType::Bf16),
                &base_backend,
            ),
        ),
        ("accumulator", accumulator),
        ("association", association),
        (
            "precision",
            operator_set(Precision::F32, base_weights, &base_backend),
        ),
        (
            "compiler",
            operator_set(
                Precision::F16,
                base_weights,
                &backend(
                    "IREE compiler version 3.13.0",
                    "libiree-runtime-sha256-aaaa",
                    "cuda",
                ),
            ),
        ),
        (
            "runtime",
            operator_set(
                Precision::F16,
                base_weights,
                &backend(
                    "IREE compiler version 3.12.0rc20260721",
                    "libiree-runtime-sha256-bbbb",
                    "cuda",
                ),
            ),
        ),
        (
            "device",
            operator_set(
                Precision::F16,
                base_weights,
                &backend(
                    "IREE compiler version 3.12.0rc20260721",
                    "libiree-runtime-sha256-aaaa",
                    "local-task",
                ),
            ),
        ),
    ];
    for (name, changed) in variants {
        assert_ne!(
            changed.canonical_identity(),
            baseline_set.canonical_identity(),
            "{name} must change the operator contract identity"
        );
        assert_rejected_then_recompiled(name, &baseline, &artifact(changed));
    }
}

#[test]
fn legacy_v1_manifest_is_replaced_by_one_recompile_and_then_reused() {
    let vmfb = temp_vmfb("v1-upgrade");
    let resident = weights();
    let legacy = AuxiliaryArtifactContract::new_legacy_unqualified(
        "vision.main",
        "config=qwen2-5-vl",
        "compiler=iree;mlir_sha256=0",
    )
    .unwrap();
    let production = artifact(operator_set(
        Precision::F16,
        float_weights(CheckpointDType::F16),
        &baseline_backend(),
    ));
    let mut compiles = 0usize;
    let mut compile = |output: &std::path::Path| {
        compiles += 1;
        std::fs::write(output, format!("vmfb-{compiles}")).map_err(|error| error.to_string())
    };
    ensure_qualified_auxiliary_artifact(&vmfb, &legacy, &resident, &mut compile).unwrap();
    let manifest = std::fs::read_to_string(auxiliary_manifest_path(&vmfb)).unwrap();
    assert!(manifest.contains("mlxcel-xla-aux-artifact-v1"));
    assert!(verify_auxiliary_manifest(&vmfb, &production, &resident).is_err());

    ensure_qualified_auxiliary_artifact(&vmfb, &production, &resident, &mut compile).unwrap();
    ensure_qualified_auxiliary_artifact(&vmfb, &production, &resident, &mut compile).unwrap();
    assert_eq!(compiles, 2, "the v1 cache is replaced once, then reused");
    let manifest = std::fs::read_to_string(auxiliary_manifest_path(&vmfb)).unwrap();
    assert!(manifest.contains("mlxcel-xla-aux-artifact-v3-operator-numeric"));
    assert!(manifest.contains("operator_numeric_contract_sha256"));
    assert!(verify_auxiliary_manifest(&vmfb, &production, &resident).is_ok());
    std::fs::remove_file(auxiliary_manifest_path(&vmfb)).ok();
    std::fs::remove_file(vmfb).ok();
}

#[test]
fn observer_resolves_float_and_affine_sources_and_fails_closed() {
    let mut observer = CheckpointSourceObserver::default();
    assert!(observer.resolve(AffineExecution::Host).is_err());
    observer.observe_float(CheckpointDType::Bf16).unwrap();
    observer.observe_float(CheckpointDType::Bf16).unwrap();
    observer.observe_float(CheckpointDType::F32).unwrap();
    assert_eq!(
        observer.resolve(AffineExecution::Host).unwrap(),
        CheckpointWeights::Float(CheckpointDType::Bf16)
    );
    observer.observe_affine(4, 64, NumericDType::F16).unwrap();
    observer.observe_affine(4, 64, NumericDType::F16).unwrap();
    let CheckpointWeights::Affine {
        float,
        dequantization,
        execution,
    } = observer.resolve(AffineExecution::InGraph).unwrap()
    else {
        panic!("affine observation must resolve to affine weights");
    };
    assert_eq!(float, CheckpointDType::Bf16);
    assert_eq!(dequantization.bits, 4);
    assert_eq!(dequantization.group_size, 64);
    assert_eq!(execution, AffineExecution::InGraph);
    assert!(observer.observe_affine(8, 64, NumericDType::F16).is_err());
    assert!(observer.observe_float(CheckpointDType::AffineU4).is_err());

    let mut tie = CheckpointSourceObserver::default();
    tie.observe_float(CheckpointDType::F16).unwrap();
    tie.observe_float(CheckpointDType::F32).unwrap();
    assert_eq!(
        tie.resolve(AffineExecution::Host).unwrap(),
        CheckpointWeights::Float(CheckpointDType::F32)
    );
}

#[test]
fn backend_identity_stays_unobserved_and_sanitizes_compiler_separators() {
    let set = operator_set(
        Precision::F16,
        float_weights(CheckpointDType::F16),
        &backend("IREE; 3.12", "libiree-runtime-sha256-aaaa", "cuda"),
    );
    let identity = set.canonical_identity();
    assert!(identity.contains("compiler=IREE, 3.12"));
    assert!(identity.contains("algorithm=unobserved-iree-dispatch-selection"));
    assert!(identity.contains("patch=libiree-runtime-sha256-aaaa"));
    assert!(identity.contains("target=cuda"));
}

#[test]
fn v3_artifacts_use_a_schema_scoped_path_and_leave_v1_caches_alone() {
    let cache = temp_vmfb("schema-scope").with_extension("");
    std::fs::create_dir_all(&cache).unwrap();
    let vmfb = cache.join("vision-256.vmfb");
    let resident = weights();
    let legacy = AuxiliaryArtifactContract::new_legacy_unqualified(
        "vision.main",
        "config=qwen2-5-vl",
        "compiler=iree;mlir_sha256=0",
    )
    .unwrap();
    let production = artifact(operator_set(
        Precision::F16,
        float_weights(CheckpointDType::F16),
        &baseline_backend(),
    ));
    let mut compiles = 0usize;
    let mut compile = |output: &std::path::Path| {
        compiles += 1;
        std::fs::write(output, format!("vmfb-{compiles}")).map_err(|error| error.to_string())
    };

    // An older binary published a v1 artifact at the shared fixed location.
    let legacy_path = qualify_auxiliary_artifact(&vmfb, &legacy, &resident, &mut compile).unwrap();
    assert_eq!(legacy_path, vmfb);
    let legacy_manifest = std::fs::read(auxiliary_manifest_path(&vmfb)).unwrap();

    // The v3 build qualifies its own entry and leaves the v1 pair untouched.
    let scoped = qualify_auxiliary_artifact(&vmfb, &production, &resident, &mut compile).unwrap();
    assert_eq!(scoped, schema_scoped_vmfb_path(&vmfb, &production));
    assert_ne!(scoped, vmfb);
    assert!(scoped.starts_with(cache.join("mlxcel-xla-aux-artifact-v3-operator-numeric")));
    assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-1");
    assert_eq!(
        std::fs::read(auxiliary_manifest_path(&vmfb)).unwrap(),
        legacy_manifest
    );
    assert!(verify_auxiliary_manifest(&vmfb, &legacy, &resident).is_ok());
    assert!(verify_auxiliary_manifest(&scoped, &production, &resident).is_ok());

    // Alternating old and new binaries reuse their own entries without a
    // downgrade refusal or another compile.
    qualify_auxiliary_artifact(&vmfb, &legacy, &resident, &mut compile).unwrap();
    qualify_auxiliary_artifact(&vmfb, &production, &resident, &mut compile).unwrap();
    assert_eq!(compiles, 2);
    std::fs::remove_dir_all(cache).ok();
}
