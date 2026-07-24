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
fn pinned_graph_contains_vit_pool_projector_and_static_output() {
    let mlir = emit_molmo_vision(&config());
    assert!(mlir.contains("stablehlo.dot_general"));
    assert!(mlir.contains("stablehlo.reduce"));
    assert!(mlir.contains("stablehlo.exponential"));
    assert!(mlir.contains("stablehlo.compare"));
    assert!(mlir.contains("tensor<13x144x3584xf32>"));
    assert!(mlir.contains("molmo.image_masks"));
}

#[cfg(feature = "iree")]
#[test]
fn pinned_graph_compiles_for_cpu() {
    let mlir = emit_molmo_vision(&config());
    let compiler = crate::iree::iree_compile_bin().unwrap();
    let cache = std::env::temp_dir().join("mlxcel-xla-molmo-vision-emitter-test");
    std::fs::create_dir_all(&cache).unwrap();
    let vmfb = crate::iree::compile_one(
        &compiler,
        &mlir,
        crate::iree::target_flags("local-task").unwrap(),
        &cache,
        "pinned-molmo-v1-vision",
        0,
    )
    .unwrap();
    assert!(vmfb.metadata().unwrap().len() > 0);
}
