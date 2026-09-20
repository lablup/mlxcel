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

//! Unit tests for the hardware-dependent draft block width policy (#1797).
//!
//! Every test here runs without a GPU: the policy is pure over
//! `(kind, compute capability, target quantization)`, so each arm of the
//! table is reachable on any host, including the Metal and CPU-only builders
//! that can never observe a compute capability at all.

use super::{
    BlockSizeSource, TargetQuantization, classify_quantization, measured_default_block_size,
    peek_target_quantization, resolve_measured_block_size,
};
use mlxcel_core::drafter::DrafterKind;

/// The flat DFlash constant, duplicated here rather than imported so a
/// change to it shows up as a failure in these tests too.
const FLAT_DFLASH: u32 = 16;

/// Every quantization classification, so a new arm cannot be added without
/// the grid tests below covering it.
const ALL_QUANTIZATIONS: [TargetQuantization; 4] = [
    TargetQuantization::Affine,
    TargetQuantization::Nvfp4,
    TargetQuantization::Other,
    TargetQuantization::Unknown,
];

/// The drafter kinds a user or the auto-detect can produce. `DrafterKind` is
/// `#[non_exhaustive]`, so this is the known set rather than an exhaustive
/// one.
const ALL_KINDS: [DrafterKind; 3] = [
    DrafterKind::Dflash,
    DrafterKind::Mtp,
    DrafterKind::InternalMtp,
];

// ── The policy table, per arm ────────────────────────────────────────────────

#[test]
fn no_cuda_has_no_measured_default_for_any_kind_or_quantization() {
    // Metal, CPU-only builds, and a CUDA build with no visible device all
    // report `None`. This is the arm that keeps every non-CUDA host on
    // exactly the behavior it had before #1797.
    for kind in ALL_KINDS {
        for quantization in ALL_QUANTIZATIONS {
            assert_eq!(
                measured_default_block_size(kind, None, quantization),
                None,
                "{kind:?} / {quantization:?} must have no measured default without CUDA"
            );
        }
    }
}

#[test]
fn pre_ampere_cuda_has_no_measured_default() {
    // A V100 is compute capability 7.0. It has no NVFP4 path at all
    // (`supports_fp_qmv` rejects compute capability 9 and below) and no
    // sweep was run on it, so it keeps the flat default.
    for quantization in ALL_QUANTIZATIONS {
        assert_eq!(
            measured_default_block_size(DrafterKind::Dflash, Some((7, 0)), quantization),
            None,
            "{quantization:?} on sm_70 must have no measured default"
        );
    }
}

#[test]
fn gb10_with_an_affine_target_resolves_to_the_measured_width() {
    // The one seeded entry: GB10 (sm_121) with an affine-quantized target on
    // a DFlash drafter.
    assert_eq!(
        measured_default_block_size(
            DrafterKind::Dflash,
            Some((12, 1)),
            TargetQuantization::Affine
        ),
        Some(4)
    );
}

#[test]
fn gb10_with_a_non_affine_target_takes_the_flat_fallback() {
    // NVFP4 routes to `fp_qmv` rather than `qmv`, so the affine crossover
    // does not transfer to it and no entry was seeded from the NVFP4 sweep.
    // `Other` and `Unknown` are unmeasured by construction.
    for quantization in [
        TargetQuantization::Nvfp4,
        TargetQuantization::Other,
        TargetQuantization::Unknown,
    ] {
        assert_eq!(
            measured_default_block_size(DrafterKind::Dflash, Some((12, 1)), quantization),
            None,
            "{quantization:?} on GB10 must take the flat fallback"
        );
    }
}

#[test]
fn mtp_kinds_keep_their_own_default_on_every_device() {
    // `DEFAULT_MTP_BLOCK_SIZE` has separate upstream provenance and was not
    // swept here, so no device may narrow it. `InternalMtp` is auto-detected
    // and has no override surface of its own.
    for kind in [DrafterKind::Mtp, DrafterKind::InternalMtp] {
        for capability in [None, Some((7, 0)), Some((9, 0)), Some((12, 1))] {
            for quantization in ALL_QUANTIZATIONS {
                assert_eq!(
                    measured_default_block_size(kind, capability, quantization),
                    None,
                    "{kind:?} on {capability:?} / {quantization:?} must be untouched"
                );
            }
        }
    }
}

#[test]
fn measured_defaults_never_disable_speculation() {
    // `round_loop.rs` breaks out of the round loop at `bs <= 1`, so a policy
    // entry of 0 or 1 would silently turn speculative decoding off instead of
    // narrowing it. Swept over the whole reachable input grid rather than
    // asserted on the one entry, so a future entry cannot regress it.
    for kind in ALL_KINDS {
        for major in 0..=13u32 {
            for minor in 0..=9u32 {
                for quantization in ALL_QUANTIZATIONS {
                    if let Some(width) =
                        measured_default_block_size(kind, Some((major, minor)), quantization)
                    {
                        assert!(
                            width >= 2,
                            "{kind:?} on sm_{major}{minor} / {quantization:?} resolved to \
                             {width}, which disables speculation"
                        );
                    }
                }
            }
        }
    }
}

// ── The fallback wrapper and its reason ──────────────────────────────────────

#[test]
fn resolve_measured_block_size_reports_the_measured_source() {
    let resolved = resolve_measured_block_size(
        DrafterKind::Dflash,
        Some((12, 1)),
        TargetQuantization::Affine,
        FLAT_DFLASH,
    );
    assert_eq!(resolved.width, 4);
    assert_eq!(
        resolved.source,
        BlockSizeSource::MeasuredHardwareDefault {
            capability: Some((12, 1)),
            quantization: TargetQuantization::Affine,
        }
    );
    let line = resolved.describe();
    assert!(
        line.starts_with("4 ("),
        "startup log line must lead with the width: {line}"
    );
    // The log has to name what the policy matched on, not only that a default
    // applied: an operator reading it must be able to tell an unmeasured
    // device from an unreadable target config.
    assert!(line.contains("sm_121"), "{line}");
    assert!(line.contains("affine"), "{line}");
}

#[test]
fn resolve_measured_block_size_reports_the_fallback_source() {
    let resolved = resolve_measured_block_size(
        DrafterKind::Dflash,
        None,
        TargetQuantization::Affine,
        FLAT_DFLASH,
    );
    assert_eq!(resolved.width, FLAT_DFLASH);
    assert_eq!(
        resolved.source,
        BlockSizeSource::KindDefault {
            capability: None,
            quantization: TargetQuantization::Affine,
        }
    );
    assert!(
        resolved.describe().contains("no CUDA device"),
        "{}",
        resolved.describe()
    );
}

#[test]
fn an_unreadable_target_config_is_distinguishable_from_an_unmeasured_device() {
    // Both take the flat fallback, but they are different situations and the
    // startup log is the only place an operator can tell them apart.
    let unreadable = resolve_measured_block_size(
        DrafterKind::Dflash,
        Some((12, 1)),
        TargetQuantization::Unknown,
        FLAT_DFLASH,
    );
    let unmeasured_device = resolve_measured_block_size(
        DrafterKind::Dflash,
        Some((9, 0)),
        TargetQuantization::Affine,
        FLAT_DFLASH,
    );
    assert_eq!(unreadable.width, FLAT_DFLASH);
    assert_eq!(unmeasured_device.width, FLAT_DFLASH);
    assert_ne!(unreadable.describe(), unmeasured_device.describe());
    assert!(
        unreadable.describe().contains("config.json"),
        "{}",
        unreadable.describe()
    );
    assert!(
        unmeasured_device.describe().contains("sm_90"),
        "{}",
        unmeasured_device.describe()
    );
}

#[test]
fn every_block_size_source_has_a_distinct_reason() {
    // The startup log is the only way an operator can tell a measured
    // default from a flag they forgot they exported, so the reasons must not
    // collide.
    let keys = (Some((12, 1)), TargetQuantization::Affine);
    let reasons = [
        BlockSizeSource::Override.reason(),
        BlockSizeSource::DrafterCheckpoint("the drafter checkpoint's own width").reason(),
        BlockSizeSource::MeasuredHardwareDefault {
            capability: keys.0,
            quantization: keys.1,
        }
        .reason(),
        BlockSizeSource::KindDefault {
            capability: keys.0,
            quantization: keys.1,
        }
        .reason(),
    ];
    for (i, a) in reasons.iter().enumerate() {
        assert!(!a.is_empty());
        for b in &reasons[i + 1..] {
            assert_ne!(a, b, "two sources share the reason {a:?}");
        }
    }
}

// ── Quantization classification ──────────────────────────────────────────────

#[test]
fn an_explicit_affine_mode_classifies_as_affine() {
    // `models/mlx/qwen3.5-4b-4bit`, the pairing target #1782 measured.
    let config = serde_json::json!({
        "model_type": "qwen3_5",
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    });
    assert_eq!(classify_quantization(&config), TargetQuantization::Affine);
}

#[test]
fn a_quantization_block_without_a_mode_classifies_as_affine() {
    // MLX's classic quantized exports predate the `mode` field and are
    // affine; `infer_quantization_mode` reaches the same answer from the
    // weights, which are not loaded at policy-resolution time.
    let config = serde_json::json!({
        "model_type": "llama",
        "quantization": {"group_size": 64, "bits": 4}
    });
    assert_eq!(classify_quantization(&config), TargetQuantization::Affine);
}

#[test]
fn a_compressed_tensors_nvfp4_export_classifies_as_nvfp4() {
    // `models/mlx/laguna-xs-2.1-nvfp4` declares no root `quantization` at
    // all; the only declaration is the compressed-tensors
    // `nvfp4-pack-quantized` format under `quantization_config`. Reading only
    // the root `quantization` key would call this unquantized.
    let config = serde_json::json!({
        "model_type": "laguna",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "weights": {"num_bits": 4, "group_size": 16}
                }
            }
        }
    });
    assert_eq!(classify_quantization(&config), TargetQuantization::Nvfp4);
}

#[test]
fn an_mlx_native_nvfp4_mode_classifies_as_nvfp4() {
    let config = serde_json::json!({
        "quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    });
    assert_eq!(classify_quantization(&config), TargetQuantization::Nvfp4);
}

#[test]
fn block_float_modes_other_than_nvfp4_classify_as_other() {
    for mode in ["mxfp4", "mxfp8"] {
        let config =
            serde_json::json!({"quantization": {"group_size": 32, "bits": 4, "mode": mode}});
        assert_eq!(
            classify_quantization(&config),
            TargetQuantization::Other,
            "{mode} must not be mistaken for a path the policy has an entry for"
        );
    }
    let packed = serde_json::json!({
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"format": "mxfp4-pack-quantized"}}
        }
    });
    assert_eq!(classify_quantization(&packed), TargetQuantization::Other);
}

#[test]
fn a_vendor_fp8_quantization_config_classifies_as_other() {
    // A block-scaled fp8 release declares `quantization_config` with no
    // per-group format. Readable, but not one of the two kernel paths.
    let config = serde_json::json!({
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
    });
    assert_eq!(classify_quantization(&config), TargetQuantization::Other);
}

#[test]
fn a_nested_text_config_declaration_is_found() {
    // VLM checkpoints nest the text tower's quantization under
    // `text_config`; the root carries none.
    let config = serde_json::json!({
        "model_type": "qwen3_5_vl",
        "text_config": {"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}
    });
    assert_eq!(classify_quantization(&config), TargetQuantization::Affine);
}

#[test]
fn an_unquantized_checkpoint_classifies_as_other_not_unknown() {
    // Understood, just not quantized. `Unknown` is reserved for a config
    // that could not be read, so the two stay tellable apart in a log.
    let config = serde_json::json!({"model_type": "llama", "hidden_size": 4096});
    assert_eq!(classify_quantization(&config), TargetQuantization::Other);
}

#[test]
fn a_case_variant_mode_is_still_recognised() {
    let config = serde_json::json!({"quantization": {"bits": 4, "mode": "Affine"}});
    assert_eq!(classify_quantization(&config), TargetQuantization::Affine);
}

// ── The on-disk peek ─────────────────────────────────────────────────────────

#[test]
fn peeking_a_missing_checkpoint_is_unknown_not_a_failure() {
    let missing = std::path::Path::new("/nonexistent/mlxcel-1797-target");
    assert_eq!(
        peek_target_quantization(missing),
        TargetQuantization::Unknown
    );
}

#[test]
fn peeking_an_unparseable_config_is_unknown() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), "{ not json").expect("write config.json");
    assert_eq!(
        peek_target_quantization(dir.path()),
        TargetQuantization::Unknown
    );
}

#[test]
fn peeking_reads_the_declaration_from_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3_5", "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}"#,
    )
    .expect("write config.json");
    assert_eq!(
        peek_target_quantization(dir.path()),
        TargetQuantization::Affine
    );
}
