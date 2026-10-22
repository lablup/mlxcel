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
