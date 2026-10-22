use super::*;

#[test]
fn parses_legacy_segmentation_schema() {
    // This is the public mlx-community/sapiens2-seg-0.4b-bf16 config verbatim.
    // It has no `architectures` and uses the early nested schema.
    let raw: RawConfig = serde_json::from_str(include_str!(
        "../../../../../tests/fixtures/sapiens2/seg-0.4b-config.json"
    ))
    .unwrap();
    let config = Sapiens2Config::from_raw(raw).unwrap();
    assert_eq!(config.task, Sapiens2Task::Segmentation);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.rope_theta, 100.0);
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.kv_heads_per_layer[8], 8);
    assert_eq!(config.num_labels, 29);
    assert_eq!(
        config.head.unwrap().upsample_out_channels,
        [512, 256, 128, 64]
    );
}

#[test]
fn architecture_selects_pose_and_preserves_names() {
    let raw: RawConfig = serde_json::from_str(r#"{
        "architectures":["Sapiens2ForPoseEstimation"],"hidden_size":1024,"num_hidden_layers":2,"num_attention_heads":16,"num_key_value_heads_per_layer":[16,8],"intermediate_size":4096,"image_size":[1024,768],"patch_size":16,"num_register_tokens":8,
        "id2label":{"0":"nose","1":"left_eye"},"flip_pairs":[[0,1]],
        "head_config":{"upsample_out_channels":[1024,768],"upsample_kernel_sizes":[4,4]}
    }"#).unwrap();
    let config = Sapiens2Config::from_raw(raw).unwrap();
    assert_eq!(config.task, Sapiens2Task::Pose);
    assert_eq!(config.keypoint_names.unwrap(), ["nose", "left_eye"]);
    assert_eq!(config.flip_pairs, [(0, 1)]);
}

#[test]
fn legacy_upsample_channels_select_pixel_shuffle() {
    let raw: RawConfig = serde_json::from_str(
        r#"{
            "task":"pointmap",
            "backbone_config":{"embed_dims":1024,"num_layers":24,"num_heads":16,"feedforward_channels":4096,"image_size":[1024,768],"patch_size":16,"n_storage_tokens":8},
            "head_config":{"upsample_channels":[768,512,256,128],"conv_out_channels":[64,32,16],"conv_kernel_sizes":[3,3,3],"scale_conv_out_channels":[1536,512,128],"scale_conv_kernel_sizes":[1,1,1],"scale_final_layer":[6144,512,128,1]}
        }"#,
    )
    .unwrap();
    let config = Sapiens2Config::from_raw(raw).unwrap();
    let head = config.head.unwrap();
    assert!(head.use_pixel_shuffle);
    assert_eq!(head.upsample_out_channels, [768, 512, 256, 128]);
    assert_eq!(head.scale_final_input_size, Some(6144));
    assert_eq!(head.scale_final_hidden_sizes, [512, 128]);
}
