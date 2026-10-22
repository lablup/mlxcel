// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::vision::dense::{DenseInput, DenseOutput, DensePredictor};
use serde_json::{Map, Value};

struct Stub;

impl DensePredictor for Stub {
    fn family(&self) -> &'static str {
        "sapiens2"
    }

    fn tasks(&self) -> &[&'static str] {
        &["depth"]
    }

    fn predict(&self, _: &DenseInput, _: &Map<String, Value>) -> Result<DenseOutput> {
        Ok(DenseOutput::default())
    }
}

fn load_stub(_: &Path) -> Result<Box<dyn DensePredictor>> {
    Ok(Box::new(Stub))
}

#[test]
fn resolves_known_model_type_from_config() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"sapiens2"}"#,
    )
    .unwrap();
    let families = [DenseFamily {
        model_type: "sapiens2",
        supports_prompts: false,
        load: load_stub,
    }];
    assert_eq!(
        resolve_family(dir.path(), &families).unwrap().model_type,
        "sapiens2"
    );
    assert_eq!(
        load_dense_model_from(dir.path(), &families)
            .unwrap()
            .family(),
        "sapiens2"
    );
}

#[test]
fn recognized_but_unported_family_is_named() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(dir.path().join("config.json"), r#"{"model_type":"moge3"}"#).unwrap();
    let error = resolve_family(dir.path(), &[]).unwrap_err().to_string();
    assert!(
        error.contains("moge3 is not supported by this build"),
        "{error}"
    );
}

#[test]
fn both_sam3_model_types_resolve_to_the_promptable_loader() {
    for model_type in ["sam3_video", "sam3"] {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            format!(r#"{{"model_type":"{model_type}"}}"#),
        )
        .unwrap();
        let family = resolve_family(dir.path(), FAMILIES).unwrap();
        assert_eq!(family.model_type, model_type);
        assert!(family.supports_prompts);
        assert!(std::ptr::fn_addr_eq(
            family.load,
            crate::vision::dense::sam3::load as FamilyLoader
        ));
    }
}
