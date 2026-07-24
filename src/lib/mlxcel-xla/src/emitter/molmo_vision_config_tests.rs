use super::*;

fn config() -> MolmoVisionConfig {
    MolmoVisionConfig::from_json_strs(
        r#"{
            "model_type":"molmo",
            "hidden_size":3584,
            "intermediate_size":37888,
            "quantization":{"bits":4,"group_size":64},
            "vision_config":{}
        }"#,
        r#"{"max_crops":12}"#,
    )
    .unwrap()
}

#[test]
fn pinned_defaults_bind_sparse_add_artifact_identity() {
    let config = config();
    assert_eq!(config.max_crops, 13);
    assert_eq!(config.patches_per_crop, 576);
    assert_eq!(config.patch_width, 588);
    assert_eq!(config.selected_layers, vec![21, 14]);
    assert_eq!(config.emitted_layers(), 22);
    assert_eq!(config.projected_rows_per_crop(), 144);
    assert!(config.fingerprint().contains(MOLMO_V1_MERGE_MODE));
}

#[test]
fn schema_keeps_packed_quantized_weights_resident() {
    let specs = config().weight_specs();
    assert_eq!(specs.len(), 647);
    assert_eq!(
        specs[1],
        MolmoVisionWeightSpec {
            name: "vision_tower.image_vit.patch_embedding.weight".to_string(),
            shape: vec![1024, 588],
            dtype: MolmoVisionWeightDType::Float32,
        }
    );
    assert!(specs.iter().any(|spec| {
        spec.name == "vision_tower.image_projector.w2.weight"
            && spec.shape == [3584, 2368]
            && spec.dtype == MolmoVisionWeightDType::Uint32
    }));
}

#[test]
fn rejects_duplicate_or_out_of_range_selected_layers() {
    let duplicate = r#"{
        "model_type":"molmo",
        "vision_config":{"image_num_layers":3},
        "vit_layers":[-1,2]
    }"#;
    assert!(
        MolmoVisionConfig::from_json_strs(duplicate, r#"{"max_crops":1}"#)
            .unwrap_err()
            .contains("unique")
    );
    let invalid = duplicate.replace("[-1,2]", "[-4]");
    assert!(
        MolmoVisionConfig::from_json_strs(&invalid, r#"{"max_crops":1}"#)
            .unwrap_err()
            .contains("out of range")
    );
}
