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

//! Config-plumbing coverage for the DeepSeek-OCR family (issue #975).
//!
//! Everything here runs on JSON alone, with no checkpoint on disk and no
//! forward pass. That is deliberate: the defect these guard is a silent one at
//! the deserialization boundary, and a test that reached a quantized kernel
//! with a mismatched `group_size` would abort the whole test binary with
//! SIGABRT rather than fail, because `gather_qmm` crosses the cxx bridge as
//! `UniquePtr<MlxArray>` rather than `Result`.

use super::{deepseekocr_language_args, unlimited_ocr_sliding_window};
use serde_json::{Value, json};

/// The `family` labels the three loaders in this module pass to
/// [`deepseekocr_language_args`]: `load_deepseekocr_vlm`,
/// `load_unlimited_ocr_vlm` and `load_deepseekocr_2_vlm`, in that order.
///
/// Issue #975 asks for the inheritance to be verified at each of the three
/// sites rather than only the first. The three now funnel through one helper,
/// so every assertion below is driven once per label to keep that explicit: if
/// a future edit gives one loader its own inline copy of the fixup again, the
/// label it no longer passes here is the thing that goes stale.
const FAMILIES: [&str; 3] = ["DeepSeek-OCR", "Unlimited-OCR", "DeepSeek-OCR 2"];

/// A `language_config` in the shape the DeepSeek-OCR checkpoints ship: the
/// structural fields the decoder needs, and no quantization declaration of its
/// own, which is exactly why the loaders inherit the root block.
fn language_config() -> Value {
    json!({
        "vocab_size": 129280,
        "hidden_size": 1280,
        "intermediate_size": 6848,
        "num_hidden_layers": 12,
        "num_attention_heads": 10,
        "num_key_value_heads": 10,
        "max_position_embeddings": 8192,
        "n_routed_experts": 64,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 896,
        "n_shared_experts": 2,
        "first_k_dense_replace": 1,
    })
}

fn full_config(root_quantization: Option<Value>, lc_quantization: Option<Value>) -> Value {
    let mut lc = language_config();
    if let Some(q) = lc_quantization {
        lc.as_object_mut().unwrap().insert("quantization".into(), q);
    }
    let mut full = json!({ "model_type": "deepseekocr", "language_config": lc });
    if let Some(q) = root_quantization {
        full.as_object_mut()
            .unwrap()
            .insert("quantization".into(), q);
    }
    full
}

/// The regression this issue is about. The loaders copy the root
/// `quantization` object into `language_config`; before #975
/// `deepseek::ModelArgs` had no field to receive it, so serde discarded the
/// block and the accessors returned the hardcoded 64 / 4 for every checkpoint.
#[test]
fn inherits_the_root_quantization_block_into_the_text_args() {
    let full = full_config(
        Some(json!({"group_size": 32, "bits": 8, "mode": "affine"})),
        None,
    );

    for family in FAMILIES {
        let args = deepseekocr_language_args(&full, family)
            .unwrap_or_else(|e| panic!("{family} language_config must parse: {e}"));
        assert_eq!(args.group_size(), 32, "{family} group_size");
        assert_eq!(args.bits(), 8, "{family} bits");
        assert_eq!(args.quantization_mode().unwrap(), "affine", "{family} mode");
    }
}

/// The `!obj.contains_key("quantization")` guard in the helper: a sub-config
/// that declares its own block keeps it, and the root block does not overwrite
/// it.
#[test]
fn a_declared_language_config_block_beats_the_inherited_root_block() {
    let full = full_config(
        Some(json!({"group_size": 64, "bits": 4})),
        Some(json!({"group_size": 32, "bits": 8})),
    );

    for family in FAMILIES {
        let args = deepseekocr_language_args(&full, family)
            .unwrap_or_else(|e| panic!("{family} language_config must parse: {e}"));
        assert_eq!(args.group_size(), 32, "{family} group_size");
        assert_eq!(args.bits(), 8, "{family} bits");
    }
}

/// The two in-tree checkpoints, `models/deepseek-ocr-4bit` and
/// `models/deepseek-ocr-2-4bit`, both declare exactly `{"group_size": 64,
/// "bits": 4, "mode": "affine"}` at the root, which is also what the accessors
/// fall back to. This pins that the fix changes nothing for them: the resolved
/// pair is the same one they loaded with before, so their OCR output is
/// unchanged by construction.
#[test]
fn the_in_tree_checkpoint_declaration_resolves_to_the_previous_fallback() {
    let full = full_config(
        Some(json!({"group_size": 64, "bits": 4, "mode": "affine"})),
        None,
    );

    for family in FAMILIES {
        let args = deepseekocr_language_args(&full, family)
            .unwrap_or_else(|e| panic!("{family} language_config must parse: {e}"));
        assert_eq!(args.group_size(), 64, "{family} group_size");
        assert_eq!(args.bits(), 4, "{family} bits");
    }
}

/// No declaration anywhere still resolves to the family defaults rather than
/// failing, which is what a bf16 checkpoint relies on.
#[test]
fn an_undeclared_quantization_block_keeps_the_family_defaults() {
    let full = full_config(None, None);

    for family in FAMILIES {
        let args = deepseekocr_language_args(&full, family)
            .unwrap_or_else(|e| panic!("{family} language_config must parse: {e}"));
        assert_eq!(args.group_size(), 64, "{family} group_size");
        assert_eq!(args.bits(), 4, "{family} bits");
        assert_eq!(args.quantization_mode().unwrap(), "affine", "{family} mode");
    }
}

/// The sub-config omits `model_type`, so the helper supplies it. Without this
/// the required field would fail deserialization.
#[test]
fn supplies_the_missing_model_type_for_the_sub_config() {
    let full = full_config(None, None);

    for family in FAMILIES {
        let args = deepseekocr_language_args(&full, family)
            .unwrap_or_else(|e| panic!("{family} language_config must parse: {e}"));
        assert_eq!(args.model_type, "deepseek", "{family} model_type");
    }
}

/// A partially declared block resolves per key rather than failing to
/// deserialize, because `QuantizationArgs` keeps all three fields optional.
/// An 8-bit export that names only `bits` still loads, and the unnamed
/// `group_size` falls back.
#[test]
fn a_partial_quantization_block_resolves_per_key() {
    let full = full_config(Some(json!({"bits": 8})), None);

    for family in FAMILIES {
        let args = deepseekocr_language_args(&full, family)
            .unwrap_or_else(|e| panic!("{family} language_config must parse: {e}"));
        assert_eq!(args.group_size(), 64, "{family} group_size");
        assert_eq!(args.bits(), 8, "{family} bits");
    }
}

/// A mode MLX cannot parse is rejected at the point it is read rather than
/// stored and handed to a kernel (#973). The inherited block is the route that
/// carries it here.
#[test]
fn an_unparseable_inherited_mode_is_rejected() {
    let full = full_config(
        Some(json!({"group_size": 64, "bits": 4, "mode": "Affine"})),
        None,
    );

    for family in FAMILIES {
        let args = deepseekocr_language_args(&full, family)
            .unwrap_or_else(|e| panic!("{family} language_config must parse: {e}"));
        let err = args
            .quantization_mode()
            .expect_err("a misspelled mode must not resolve");
        assert!(
            err.contains("quantization.mode"),
            "{family} error should name the field: {err}"
        );
    }
}

/// A missing `language_config` is still an error, and the message still names
/// the loader that asked.
#[test]
fn a_missing_language_config_names_the_family() {
    let full = json!({ "model_type": "deepseekocr" });

    for family in FAMILIES {
        let err = deepseekocr_language_args(&full, family)
            .expect_err("a config without language_config must not parse")
            .to_string();
        assert!(err.contains(family), "error should name {family}: {err}");
    }
}

/// The Unlimited-OCR ring-cache window reads through the untouched
/// `full_config`, so pulling the quantization fixup into a shared helper did
/// not change which key it comes from.
#[test]
fn unlimited_ocr_window_reads_the_declared_sliding_window() {
    let mut full = full_config(None, None);
    full["language_config"]["sliding_window_size"] = json!(512);
    assert_eq!(unlimited_ocr_sliding_window(&full), 512);
}

/// Undeclared falls back, and a non-positive declaration is clamped rather than
/// producing a degenerate cache.
#[test]
fn unlimited_ocr_window_falls_back_and_clamps() {
    let full = full_config(None, None);
    assert_eq!(
        unlimited_ocr_sliding_window(&full),
        super::DEFAULT_SLIDING_WINDOW
    );

    let mut zero = full_config(None, None);
    zero["language_config"]["sliding_window_size"] = json!(0);
    assert_eq!(unlimited_ocr_sliding_window(&zero), 1);
}
