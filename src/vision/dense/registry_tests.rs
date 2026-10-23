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
        supports_sidecars: false,
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
fn recognized_dense_family_is_loadable() {
    let family = FAMILIES
        .iter()
        .find(|family| family.model_type == "moge3")
        .unwrap();
    assert!(!family.supports_prompts);
    assert!(std::ptr::fn_addr_eq(
        family.load,
        crate::vision::dense::moge3::load as FamilyLoader
    ));
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

#[test]
fn both_sam31_model_types_resolve_to_the_promptable_loader() {
    for model_type in ["sam3.1_video", "sam3.1"] {
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
            crate::vision::dense::sam3_1::load as FamilyLoader
        ));
    }
}

#[test]
fn sam3d_objects_registers_typed_sidecars_and_mesh_output() {
    let family = FAMILIES
        .iter()
        .find(|f| f.model_type == "sam3d_objects")
        .unwrap();
    assert!(family.supports_sidecars);
    assert!(!family.supports_prompts);
    let arch = crate::models::registry::build_architecture_registry("test")
        .families
        .into_iter()
        .find(|f| f.id == "sam3d_objects")
        .unwrap();
    assert_eq!(arch.output, crate::models::registry::OutputKind::Mesh);
}

#[test]
fn sam3d_body_registers_colorless_mesh_predictor() {
    let family = super::FAMILIES
        .iter()
        .find(|f| f.model_type == "sam3d_body")
        .expect("Body family");
    assert!(!family.supports_prompts && !family.supports_sidecars);
    assert!(super::is_dense_model_type("sam3d_body"));
}

#[test]
fn dense_config_reader_rejects_oversized_metadata_before_parsing() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let file = std::fs::File::create(dir.path().join("config.json"))?;
    file.set_len(8 * 1024 * 1024 + 1)?;
    let error = checkpoint_model_type(dir.path()).expect_err("oversized config");
    assert!(error.to_string().contains("8 MiB"), "{error:#}");
    Ok(())
}
#[test]
fn dense_config_reader_preserves_case_normalization_and_parse_errors() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("config.json");
    std::fs::write(
        &path,
        r#"{"model_type":"SAM3D_BODY","unrelated":{"value":17}}"#,
    )?;
    assert_eq!(checkpoint_model_type(dir.path())?, "sam3d_body");
    std::fs::write(&path, b"not json")?;
    let error = checkpoint_model_type(dir.path()).expect_err("invalid JSON");
    assert!(error.to_string().contains("parse"), "{error:#}");
    Ok(())
}
