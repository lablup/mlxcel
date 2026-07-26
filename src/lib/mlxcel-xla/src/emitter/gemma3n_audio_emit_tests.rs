use super::*;
use crate::emitter::builder::Ty;

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
    assert!(!encode.contains("\"stablehlo.reduce_window\""));
    assert!(
        encode.matches("\"stablehlo.gather\"").count() >= 40,
        "both SSCP norms must pin four MLX CUDA XOR reduction trees"
    );
    assert!(!encode.contains("loc(\"audio.hard_embeddings\")"));
    assert!(!encode.contains("model.language_model."));
    assert_eq!(encode_layout.stages.len(), 3 + 4 + 12 * 5 + 11 + 8);
    for name in [
        "sscp_conv_0_convolution",
        "sscp_conv_0_norm_sum_at_time",
        "sscp_conv_0_norm_cumulative_sum",
        "sscp_conv_0_norm_mean",
        "sscp_conv_0_norm_squared_at_time",
        "sscp_conv_0_norm_cumulative_squared",
        "sscp_conv_0_norm_variance",
        "sscp_conv_0_norm_stabilized_variance",
        "sscp_conv_0_norm_inverse_stddev",
        "sscp_conv_0_norm_inverse_stddev_sqrt_reciprocal",
        "sscp_conv_0_norm",
        "sscp_conv_0",
        "sscp_conv_1_convolution",
        "sscp_conv_1_norm",
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

#[test]
fn first_sscp_convolution_rounds_mel_inputs_before_accumulation() {
    fn synthetic_convolution(input: &[f32], weights: &[f32], pre_cast: bool) -> f32 {
        let sum = input
            .iter()
            .zip(weights)
            .map(|(&value, &weight)| {
                let value = if pre_cast {
                    crate::weights::round_bf16_f32(value)
                } else {
                    value
                };
                value * crate::weights::round_bf16_f32(weight)
            })
            .sum::<f32>();
        crate::weights::round_bf16_f32(sum)
    }

    // Both values map to 1.0 in BF16, so the maintained MLX pre-cast cancels
    // them exactly. Rounding only after the convolution preserves the small
    // F32 difference and therefore produces a distinct non-zero output.
    let input = [1.003, 1.0];
    let weights = [1.0, -1.0];
    let pre_cast = synthetic_convolution(&input, &weights, true);
    let post_conv_only = synthetic_convolution(&input, &weights, false);
    assert_eq!(pre_cast, 0.0);
    assert_ne!(post_conv_only, 0.0);

    // Pin that same semantic boundary in the emitted graph even when global
    // contraction precision is F32: the reshaped mel is narrowed to BF16 and
    // widened back before padding feeds the first convolution.
    let (encode, _) = emit_gemma3n_audio_encode(
        &text(),
        &Gemma3nXlaAudioConfig::default(),
        8,
        1,
        Precision::F32,
    )
    .unwrap();
    let first_convolution = encode
        .find("\"stablehlo.convolution\"")
        .expect("Gemma3n audio graph must contain SSCP convolution");
    let prefix = &encode[..first_convolution];
    let narrow = prefix
        .find(
            "stablehlo.convert %0 : (tensor<1x8x128x1xf32>) -> \
             tensor<1x8x128x1xbf16>",
        )
        .expect("mel input must narrow to BF16 before the first SSCP convolution");
    let widen = prefix[narrow..]
        .find(
            "stablehlo.convert %1 : (tensor<1x8x128x1xbf16>) -> \
             tensor<1x8x128x1xf32>",
        )
        .expect("BF16 mel input must widen to the graph's F32 carrier");
    let pad = prefix[narrow + widen..]
        .find("\"stablehlo.pad\"(%2")
        .expect("the first SSCP padding must consume BF16-rounded mel");
    assert!(widen > 0);
    assert!(pad > 0);
}

#[test]
fn sscp_convolutions_accumulate_bf16_operands_into_f32_results() {
    let (encode, _) = emit_gemma3n_audio_encode(
        &text(),
        &Gemma3nXlaAudioConfig::default(),
        8,
        1,
        Precision::Bf16,
    )
    .unwrap();
    assert!(encode.contains(
        "(tensor<1x10x130x1xbf16>, tensor<3x3x1x128xbf16>) -> \
         tensor<1x4x64x128xf32>"
    ));
    assert!(encode.contains(
        "(tensor<1x6x66x128xbf16>, tensor<3x3x128x32xbf16>) -> \
         tensor<1x2x32x32xf32>"
    ));
    assert!(!encode.contains(
        "(tensor<1x10x130x1xbf16>, tensor<3x3x1x128xbf16>) -> \
         tensor<1x4x64x128xbf16>"
    ));
}

#[test]
#[ignore = "requires IREE_DIST"]
fn sscp_f32_accumulator_contract_compiles_for_cpu() {
    let mut builder = Builder::new().with_precision(Precision::Bf16);
    let input = Builder::arg(0, Ty::f32(vec![1, 3, 3, 1]));
    let kernel = Builder::arg(1, Ty::f32(vec![3, 3, 1, 4]));
    let output = builder.convolution_f32_accumulate(&input, &kernel, &[1, 1], &[(0, 0); 2], 1);
    let output_ty = output.ty.render();
    let mlir = format!(
        "module @sscp_f32_accumulator {{\n  func.func public @main(%arg0: \
         tensor<1x3x3x1xf32>, %arg1: tensor<3x3x1x4xf32>) -> {output_ty} {{\n{}    \
         return {} : {output_ty}\n  }}\n}}\n",
        builder.body(),
        output.name,
    );
    let compiler = std::path::PathBuf::from(std::env::var_os("IREE_DIST").expect("IREE_DIST"))
        .join("bin/iree-compile");
    let stem = format!("mlxcel-gemma3n-sscp-f32-accumulator-{}", std::process::id());
    compile(compiler.as_os_str(), "local", &stem, "contract", mlir);
}

#[test]
#[ignore = "requires IREE_DIST"]
fn sscp_cuda_norm_schedule_compiles_for_cpu() {
    use crate::emitter::gemma3n_audio_math::{mlx_cuda_cumsum_time_f32, mlx_cuda_row_sum_f32};

    let mut builder = Builder::new().with_precision(Precision::Bf16);
    let input = Builder::arg(0, Ty::f32(vec![1, 8, 64, 128]));
    let channels = mlx_cuda_row_sum_f32(&mut builder, &input, 3);
    let frequency = mlx_cuda_row_sum_f32(&mut builder, &channels, 2);
    let per_time = builder.reshape(&frequency, vec![1, 8, 1, 1]);
    let output = mlx_cuda_cumsum_time_f32(&mut builder, &per_time);
    let output_ty = output.ty.render();
    let mlir = format!(
        "module @sscp_cuda_norm_schedule {{\n  func.func public @main(%arg0: \
         tensor<1x8x64x128xf32>) -> {output_ty} {{\n{}    return {} : {output_ty}\n  \
         }}\n}}\n",
        builder.body(),
        output.name,
    );
    let compiler = std::path::PathBuf::from(std::env::var_os("IREE_DIST").expect("IREE_DIST"))
        .join("bin/iree-compile");
    let stem = format!("mlxcel-gemma3n-sscp-cuda-norm-{}", std::process::id());
    compile(compiler.as_os_str(), "local", &stem, "contract", mlir);
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
