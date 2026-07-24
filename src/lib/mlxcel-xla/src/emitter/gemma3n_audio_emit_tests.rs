use super::*;

fn text() -> Gemma3nConfig {
    Gemma3nConfig::from_json_str(
        &serde_json::json!({
            "model_type": "gemma3n_text",
            "hidden_size": 8, "intermediate_size": [12, 12],
            "max_position_embeddings": 4096,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4, "rms_norm_eps": 1e-6,
            "vocab_size": 12, "vocab_size_per_layer_input": 10,
            "hidden_size_per_layer_input": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "activation_sparsity_pattern": [0.0, 0.0],
            "sliding_window": 2, "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0, "final_logit_softcapping": 30.0,
            "num_kv_shared_layers": 0, "altup_num_inputs": 2,
            "altup_active_idx": 0, "altup_coef_clip": 120.0,
            "altup_correct_scale": true, "laurel_rank": 2
        })
        .to_string(),
    )
    .unwrap()
    .with_context_capacity(8)
    .unwrap()
}

#[test]
fn split_audio_graphs_preserve_stage_and_weight_boundaries() {
    let text = text();
    let audio = Gemma3nXlaAudioConfig::default();
    let (encode, encode_layout) =
        emit_gemma3n_audio_encode(&text, &audio, 8, 1, Precision::F32).unwrap();
    let (merge, merge_layout) =
        emit_gemma3n_audio_merge_ple(&text, &audio, 1, Precision::F32).unwrap();

    assert!(encode.starts_with("module @audio_encode {"));
    assert!(encode.contains("stablehlo.convolution"));
    assert!(encode.contains("\"stablehlo.reduce_window\""));
    assert!(!encode.contains("loc(\"audio.hard_embeddings\")"));
    assert!(!encode.contains("model.language_model."));
    assert_eq!(encode_layout.stages.len(), 3 + 12 * 5 + 10);
    for name in [
        "sscp_conv_0",
        "sscp_conv_1",
        "input_projection",
        "conformer.0.feed_forward_start",
        "conformer.11.final_norm",
        "encoded_reduced",
        "soft_norm",
        "soft_linear",
        "soft_post_norm",
        "hard_embedding",
        "hard_norm",
        "hard_linear",
        "hard_post_norm",
        "soft_projection",
        "hard_projection",
    ] {
        assert!(encode_layout.stages.iter().any(|stage| stage.name == name));
    }

    assert!(merge.starts_with("module @audio_merge_ple {"));
    assert!(merge.contains("loc(\"audio.projected\")"));
    assert!(merge.contains("loc(\"audio.hard_embeddings\")"));
    assert!(merge.contains("model.language_model."));
    assert!(!merge.contains("audio_tower."));
    assert!(!merge.contains("embed_audio."));
    assert_eq!(
        merge_layout
            .stages
            .iter()
            .map(|stage| stage.name.as_str())
            .collect::<Vec<_>>(),
        ["merged_embeddings", "dense_ple"]
    );
}

fn compile(compiler: &std::ffi::OsStr, device: &str, stem: &str, artifact: &str, mlir: String) {
    let input = std::env::temp_dir().join(format!("{stem}-{artifact}.mlir"));
    let output = std::env::temp_dir().join(format!("{stem}-{artifact}.vmfb"));
    std::fs::write(&input, mlir).unwrap();
    let mut command = std::process::Command::new(compiler);
    command.arg("--iree-input-type=stablehlo");
    match device {
        "local" => {
            command
                .arg("--iree-hal-target-device=local")
                .arg("--iree-hal-local-target-device-backends=llvm-cpu");
        }
        "cuda" => {
            command
                .arg("--iree-hal-target-device=cuda")
                .arg("--iree-cuda-target=sm_80");
        }
        _ => panic!("unsupported IREE test device {device}"),
    }
    let result = command.arg(&input).arg("-o").arg(&output).output().unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    assert!(
        result.status.success(),
        "audio.{artifact} failed IREE {device} compile: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
#[ignore = "requires GEMMA3N_MODEL_DIR and the pinned IREE compiler"]
fn iree_compiles_split_real_audio_artifacts() {
    let model_dir =
        std::path::PathBuf::from(std::env::var_os("GEMMA3N_MODEL_DIR").expect("model directory"));
    let compiler = std::env::var_os("MLXCEL_XLA_IREE_COMPILE").expect("IREE compiler");
    let frame_bucket = std::env::var("GEMMA3N_AUDIO_TEST_BUCKET")
        .map(|value| value.parse::<usize>().expect("frame bucket"))
        .unwrap_or(8);
    let context_capacity = std::env::var("GEMMA3N_AUDIO_TEST_CONTEXT")
        .map(|value| value.parse::<usize>().expect("context capacity"))
        .unwrap_or(256);
    let device = std::env::var("GEMMA3N_AUDIO_TEST_DEVICE").unwrap_or_else(|_| "local".to_string());
    let text = Gemma3nConfig::from_json(&model_dir)
        .unwrap()
        .with_context_capacity(context_capacity)
        .unwrap();
    let audio = Gemma3nXlaAudioConfig::from_model_dir(&model_dir)
        .unwrap()
        .unwrap();
    let (encode, _) =
        emit_gemma3n_audio_encode(&text, &audio, frame_bucket, 1, Precision::Bf16).unwrap();
    let (merge, _) = emit_gemma3n_audio_merge_ple(&text, &audio, 1, Precision::Bf16).unwrap();
    let stem = format!(
        "mlxcel-gemma3n-audio-{frame_bucket}-{context_capacity}-{}",
        std::process::id()
    );
    compile(&compiler, &device, &stem, "encode", encode);
    compile(&compiler, &device, &stem, "merge-ple", merge);
}
