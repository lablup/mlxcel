//! Decode compile flag selection (issues #3043 and #3363).

use super::*;

fn dense_runtime() -> RuntimeConfig {
    RuntimeConfig::Dense(Box::new(Config::llama_3_2_1b()))
}

fn tiny_gemma3n_runtime() -> RuntimeConfig {
    RuntimeConfig::Gemma3n(Box::new(
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
                "altup_correct_scale": true, "laurel_rank": 2,
                "tie_word_embeddings": true
            })
            .to_string(),
        )
        .unwrap()
        .with_context_capacity(4)
        .unwrap(),
    ))
}

fn has_aggressive_fusion(flags: &[String]) -> bool {
    flags.iter().any(|flag| flag == AGGRESSIVE_FUSION_FLAG)
}

#[test]
fn dense_decode_keeps_aggressive_fusion_on_every_device() {
    let cfg = dense_runtime();
    for device in ["cuda", "local-task"] {
        let flags = target_flags(device).unwrap();
        let decode = decode_compile_flags(&cfg, &flags);
        assert!(has_aggressive_fusion(&decode), "{device}: {decode:?}");
        // The decode flags extend the shared bundle flags, in order.
        assert_eq!(&decode[..flags.len()], flags.as_slice());
        assert!(!has_aggressive_fusion(&flags), "prefill keeps the defaults");
    }
}

#[test]
fn gemma3n_decode_drops_aggressive_fusion() {
    let cfg = tiny_gemma3n_runtime();
    for device in ["cuda", "local-task"] {
        let flags = target_flags(device).unwrap();
        let decode = decode_compile_flags(&cfg, &flags);
        assert!(!has_aggressive_fusion(&decode), "{device}: {decode:?}");
        assert_eq!(decode, flags);
    }
}

/// The checkpoint whose decode graph failed to compile, from
/// `GEMMA3N_MODEL_DIR` or the repository's `models` link farm.
fn gemma3n_model_dir() -> PathBuf {
    std::env::var_os("GEMMA3N_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../models/gemma-3n-e2b-it-4bit")
        })
}

/// A CUDA-capable `iree-compile` on a host with an NVIDIA driver, or `None`.
fn cuda_iree_compile() -> Option<PathBuf> {
    if !Path::new("/proc/driver/nvidia/version").exists() {
        return None;
    }
    let compiler = iree_compile_bin().ok().filter(|path| path.exists())?;
    let out = Command::new(&compiler)
        .arg("--iree-hal-list-target-backends")
        .output()
        .ok()?;
    let listed = String::from_utf8_lossy(&out.stdout);
    listed
        .split_whitespace()
        .any(|backend| backend == "cuda")
        .then_some(compiler)
}

/// issue #3363: the production Gemma3n decode graph, compiled for CUDA with the
/// production flag set, must compile. With aggressive fusion it failed on the
/// RMSNorm reduction dispatch (`'func.func' op failed to distribute`).
#[test]
fn gemma3n_cuda_decode_compiles() {
    let model_dir = gemma3n_model_dir();
    if !model_dir.join("config.json").is_file() {
        eprintln!("skipped: no Gemma3n checkpoint at {}", model_dir.display());
        return;
    }
    let Some(compiler) = cuda_iree_compile() else {
        eprintln!("skipped: no CUDA device or CUDA-capable iree-compile");
        return;
    };
    let context_capacity = 256;
    let cfg = RuntimeConfig::from_json(&model_dir, context_capacity).unwrap();
    let RuntimeConfig::Gemma3n(config) = &cfg else {
        panic!("{} is not a Gemma3n checkpoint", model_dir.display());
    };
    let precision = effective_precision("cuda", &cfg).unwrap();
    let native_qmv = gemma3n_native_qmv("cuda", config).unwrap();
    let mlir = emit_gemma3n_decode_with_qmv(config, true, precision, native_qmv);

    let dir = std::env::temp_dir().join(format!("mlxcel-gemma3n-decode-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let flags = bundle_compile_flags("cuda", &cfg, native_qmv, &dir).unwrap();
    let decode_flags = decode_compile_flags(&cfg, &flags);
    let result = compile_one_to(
        &compiler,
        &mlir,
        &decode_flags,
        &dir,
        "decode",
        context_capacity,
        &dir.join("decode.vmfb"),
    );
    std::fs::remove_dir_all(&dir).ok();
    if let Err(error) = result {
        // The compiler dumps the whole failing dispatch; its head names it.
        let head: String = error.lines().take(12).collect::<Vec<_>>().join("\n");
        panic!("Gemma3n CUDA decode failed to compile with {decode_flags:?}:\n{head}");
    }
}
