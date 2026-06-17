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

//! Quantization recommendation helpers for the CLI and server.
//!
//! This module bridges the low-level [`mlxcel_core::hardware`] recommendation
//! engine with the higher-level model loading path. It can estimate model
//! parameter counts from a `config.json` and emit human-readable advice.

use std::path::Path;

use mlxcel_core::hardware::{HardwareCapabilities, QuantRecommendation, recommend_quantization};

use crate::execution::kv_cache_advisor::{
    KvCacheModeAdvice, advise_kv_cache_modes, print_kv_cache_advice,
};

// ── Model-size estimation ─────────────────────────────────────────────────────

/// Try to estimate the approximate number of model parameters (in billions)
/// from the model's `config.json`.
///
/// This is a best-effort approximation that works for the most common dense
/// and MoE transformer architectures. Returns `None` when the config cannot be
/// read or when the required fields are absent.
///
/// The estimation formula mirrors what mlx-lm tooling uses for rough sizing:
/// ```text
/// params ≈ vocab_size * hidden_size          (embedding table)
///         + num_hidden_layers * (
///             4 * hidden_size²               (attention Q/K/V/O)
///             + 3 * hidden_size * ffn_size   (MLP gate/up/down, approx)
///           )
/// ```
///
/// For MoE models (`num_experts` > 1), the FFN term is scaled by the number
/// of experts, giving a rough total parameter count including inactive experts.
pub fn estimate_model_params_billions(model_path: &Path) -> Option<f64> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;

    estimate_params_from_config(&config)
}

fn estimate_params_from_config(config: &serde_json::Value) -> Option<f64> {
    // Support models that wrap text config under a "text_config" key (VLMs).
    let text_cfg = config.get("text_config").unwrap_or(config);

    let hidden_size = text_cfg
        .get("hidden_size")
        .or_else(|| text_cfg.get("d_model"))
        .or_else(|| text_cfg.get("dim"))
        .or_else(|| text_cfg.get("model_dim"))
        .and_then(|v| v.as_f64())?;

    let num_layers = text_cfg
        .get("num_hidden_layers")
        .or_else(|| text_cfg.get("n_layers"))
        .or_else(|| text_cfg.get("num_layers"))
        .and_then(|v| v.as_f64())?;

    let vocab_size = text_cfg
        .get("vocab_size")
        .and_then(|v| v.as_f64())
        .unwrap_or(32_000.0);

    // FFN intermediate size — try several field names used across architectures.
    let ffn_size = text_cfg
        .get("intermediate_size")
        .or_else(|| text_cfg.get("ffn_dim"))
        .or_else(|| text_cfg.get("ffn_hidden_size"))
        .and_then(|v| v.as_f64())
        .unwrap_or(hidden_size * 4.0); // fallback: 4× hidden as a rule of thumb

    // Number of experts (MoE). For dense models this stays at 1.
    let num_experts = text_cfg
        .get("num_experts")
        .or_else(|| text_cfg.get("num_local_experts"))
        .or_else(|| text_cfg.get("n_routed_experts"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0)
        .max(1.0);

    // Attention heads (used for GQA parameter correction, optional).
    let num_heads = text_cfg
        .get("num_attention_heads")
        .or_else(|| text_cfg.get("num_heads"))
        .or_else(|| text_cfg.get("n_heads"))
        .or_else(|| text_cfg.get("n_head"))
        .and_then(|v| v.as_f64())
        .unwrap_or(hidden_size / 64.0);

    let kv_heads = text_cfg
        .get("num_key_value_heads")
        .or_else(|| text_cfg.get("num_kv_heads"))
        .or_else(|| text_cfg.get("n_kv_heads"))
        .or_else(|| text_cfg.get("n_head_kv"))
        .or_else(|| text_cfg.get("multi_query_group_num"))
        .and_then(|v| v.as_f64())
        .unwrap_or(num_heads);

    let head_dim = text_cfg
        .get("head_dim")
        .or_else(|| text_cfg.get("head_size"))
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| {
            if num_heads > 0.0 && num_heads.is_finite() {
                (hidden_size / num_heads).round()
            } else {
                64.0
            }
        });

    // Parameter count estimate:
    //   embedding:  vocab_size × hidden_size (input + output embedding shared)
    //   per-layer attention:
    //     Q: hidden_size × (num_heads × head_dim)
    //     K: hidden_size × (kv_heads × head_dim)
    //     V: hidden_size × (kv_heads × head_dim)
    //     O: (num_heads × head_dim) × hidden_size
    //   per-layer MLP: 3 × hidden_size × ffn_size × num_experts
    //   norms: 2 × num_layers × hidden_size (small, included for accuracy)
    let embedding_params = vocab_size * hidden_size;

    let attn_params_per_layer = hidden_size * (num_heads * head_dim)          // Q proj
        + hidden_size * (kv_heads * head_dim)         // K proj
        + hidden_size * (kv_heads * head_dim)         // V proj
        + (num_heads * head_dim) * hidden_size; // O proj

    let ffn_params_per_layer = 3.0 * hidden_size * ffn_size * num_experts;
    let norm_params_per_layer = 2.0 * hidden_size;

    let total_params = embedding_params
        + num_layers * (attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer);

    Some(total_params / 1e9)
}

// ── BFloat16 detection ────────────────────────────────────────────────────────

/// Torch dtype strings that indicate BFloat16 weights.
const BFLOAT16_DTYPE_STRINGS: &[&str] = &["bfloat16", "bf16", "torch.bfloat16"];

/// Return `true` if the model's `config.json` declares BFloat16 as the
/// default weight dtype.
///
/// The M5 Neural Accelerator does **not** support BFloat16 computation, so
/// models with BF16 weights should be converted to FP16 or quantized before
/// running to avoid a fallback to the GPU shader pipeline.
pub fn model_uses_bfloat16(model_path: &Path) -> bool {
    let config_path = model_path.join("config.json");
    let Ok(config_str) = std::fs::read_to_string(&config_path) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) else {
        return false;
    };

    // Check both the top-level and text_config for the dtype field.
    let top_level_dtype = config
        .get("torch_dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let text_cfg_dtype = config
        .get("text_config")
        .and_then(|tc| tc.get("torch_dtype"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    BFLOAT16_DTYPE_STRINGS.contains(&top_level_dtype)
        || BFLOAT16_DTYPE_STRINGS.contains(&text_cfg_dtype)
}

// ── High-level advisor ────────────────────────────────────────────────────────

/// Advice produced by [`advise_quantization`].
#[derive(Debug)]
pub struct QuantAdvice {
    /// The recommended quantization mode.
    pub recommendation: QuantRecommendation,
    /// Estimated parameter count in billions (None if config parsing failed).
    pub estimated_params_billions: Option<f64>,
    /// Byte-accurate weight footprint read from the safetensors header before
    /// any tensors are loaded. `Some(bytes)` when available from
    /// `model.safetensors.index.json` (sharded) or a single `model.safetensors`
    /// binary header; `None` when neither is present and the analytical estimate
    /// from `estimated_params_billions` is the only available sizing signal.
    pub exact_weight_bytes: Option<u64>,
    /// True when the model config declares BFloat16 weights.
    pub model_uses_bfloat16: bool,
    /// KV cache memory estimate in bytes (None if architecture info is unavailable).
    pub kv_cache_bytes: Option<u64>,
    /// Advisory KV-cache-mode suggestions per context range (issue #327).
    ///
    /// Empty when `config.json` cannot be classified. These are opt-in
    /// suggestions printed by [`print_quant_advice`]; they never change the
    /// default inference path. See [`crate::execution::kv_cache_advisor`].
    pub kv_cache_advice: Vec<KvCacheModeAdvice>,
}

/// Produce a complete quantization recommendation for a model directory.
///
/// Uses the model's `config.json` to estimate the parameter count and KV cache
/// memory requirements, then calls [`recommend_quantization`] against the
/// provided hardware capabilities.
///
/// When `model_params_override` is `Some(n)`, that value is used instead of
/// both the safetensors-derived and config-derived estimates.
///
/// Resolution order for the size fed to [`recommend_quantization`]:
/// 1. `model_params_override` (caller-supplied explicit value)
/// 2. Exact bytes from the safetensors header, converted to billions via
///    `bytes / 2 / 1e9` (assumes FP16 as the reference dtype).
/// 3. Analytical estimate from `config.json`.
/// 4. Hard-coded 7 B fallback.
///
/// The KV cache headroom passed to [`recommend_quantization`] is derived from
/// the model architecture when it can be extracted from `config.json`. The
/// default context length is 8192 tokens; `int8_kv` defaults to `false`.
/// When architecture fields are unavailable the function falls back to the
/// built-in 2 GiB constant (`kv_cache_headroom_bytes = None`).
pub fn advise_quantization(
    model_path: &Path,
    hw: &HardwareCapabilities,
    model_params_override: Option<f64>,
) -> QuantAdvice {
    let estimated_params = estimate_model_params_billions(model_path);

    // Route weight + KV inputs through the unified estimator so the
    // advisor, the `mlxcel inspect` printer, and the
    // `--estimate-memory` preflight never disagree on the same model
    // (issue #56, epic #52 capstone). The estimator's resolution order
    // is the same one this function used historically:
    //   1. safetensors header (issue #53) — exact bytes.
    //   2. analytical estimate from config.json — fp16-equivalent.
    //   3. 7 B fallback.
    let estimate = crate::execution::memory_estimate::estimate_total_memory(
        model_path,
        crate::execution::memory_estimate::DEFAULT_CTX_LEN,
        1,
        crate::execution::memory_estimate::QuantHint::Default,
        false,
    );

    // The legacy `QuantAdvice` exposes `exact_weight_bytes: Option<u64>`,
    // which is `Some(_)` only when the safetensors header was read.
    let exact_weight_bytes = match estimate.weights_source {
        crate::execution::memory_estimate::WeightsSource::ExactSafetensors => {
            Some(estimate.weights_bytes)
        }
        _ => None,
    };

    // Convert weight-bytes to a billions-of-parameters estimate.
    // The safetensors total_size is raw parameter bytes (e.g. 2 bytes per BF16
    // parameter). Dividing by 2 yields an FP16-equivalent parameter count in
    // bytes; dividing by 1e9 converts to billions. This is conservative for
    // INT8/INT4 models (they will appear larger than they are), which is the
    // safe direction for memory-fit recommendations.
    let estimate_params_billions: Option<f64> = exact_weight_bytes.map(|b| b as f64 / 2.0 / 1e9);

    let params = model_params_override
        .or(estimate_params_billions)
        .or(estimated_params)
        .unwrap_or(7.0); // safe fallback: assume 7B when unknown

    // Carry KV bytes through directly when the estimator could read
    // the architecture; otherwise pass `None` so `recommend_quantization`
    // falls back to its built-in 2 GiB constant. Matches the legacy
    // contract on `kv_cache_headroom_bytes`.
    let kv_bytes = match estimate.kv_source {
        crate::execution::memory_estimate::KvSource::Config => Some(estimate.kv_cache_bytes),
        crate::execution::memory_estimate::KvSource::Unavailable => None,
    };

    let recommendation = recommend_quantization(params, hw.unified_memory_gb, hw, kv_bytes);

    let uses_bf16 = model_uses_bfloat16(model_path);

    // Advisory KV-cache-mode suggestions (issue #327). Reads only `config.json`
    // and returns data; it does not touch the weight-loading path, so it cannot
    // reintroduce the #289 bf16 to f16 quantized-weight promotion.
    let kv_cache_advice = advise_kv_cache_modes(model_path);

    QuantAdvice {
        recommendation,
        estimated_params_billions: estimated_params,
        exact_weight_bytes,
        model_uses_bfloat16: uses_bf16,
        kv_cache_bytes: kv_bytes,
        kv_cache_advice,
    }
}

/// Format a byte count as a human-readable string (GiB, MiB, or exact bytes).
fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB ({bytes} bytes)", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB ({bytes} bytes)", bytes as f64 / MIB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

/// Estimate KV cache memory in bytes from a model's `config.json`.
///
/// Returns `None` when the required architecture fields (`num_hidden_layers`,
/// `num_key_value_heads`, `hidden_size`, `num_attention_heads`) cannot be
/// extracted.
///
/// `ctx_len` is the requested context length (tokens); `int8_kv` controls
/// whether 1-byte (INT8) or 2-byte (FP16) KV storage is assumed.
pub fn estimate_kv_cache_bytes_from_path(
    model_path: &Path,
    ctx_len: u64,
    int8_kv: bool,
) -> Option<u64> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;
    estimate_kv_cache_bytes_from_config(&config, ctx_len, int8_kv)
}

fn estimate_kv_cache_bytes_from_config(
    config: &serde_json::Value,
    ctx_len: u64,
    int8_kv: bool,
) -> Option<u64> {
    // Delegate to the architecture-aware estimator so this advisor utility and
    // the unified `estimate_total_memory` path never diverge on KV sizing.
    crate::execution::kv_arch::estimate_kv_arch_from_config(config, ctx_len, int8_kv, 1)
        .map(|a| a.total_bytes)
}

/// Print a human-readable quantization recommendation to stdout.
pub fn print_quant_advice(advice: &QuantAdvice, hw: &HardwareCapabilities) {
    println!();
    println!("=== Quantization Recommendation ===");
    println!("  Hardware:   {} ({})", hw.silicon_gen, {
        if hw.has_neural_accelerator && hw.macos_supports_na {
            "Neural Accelerator active"
        } else if hw.has_neural_accelerator {
            "Neural Accelerator requires macOS 26.2+"
        } else {
            "no Neural Accelerator"
        }
    });
    println!("  Memory:     {} GB unified", hw.unified_memory_gb);

    if let Some(exact_bytes) = advice.exact_weight_bytes {
        println!(
            "  Model size: {} (exact, from safetensors header)",
            format_bytes(exact_bytes)
        );
        if let Some(params) = advice.estimated_params_billions {
            println!(
                "              ~{:.1}B parameters (analytical estimate for reference)",
                params
            );
        }
    } else if let Some(params) = advice.estimated_params_billions {
        println!("  Model size: ~{:.1}B parameters (estimated)", params);
    } else {
        println!("  Model size: unknown (could not parse config.json)");
    }

    println!();
    println!(
        "  Recommendation: {}",
        advice.recommendation.label().to_uppercase()
    );
    println!("  Reason:         {}", advice.recommendation.reason());

    if advice.model_uses_bfloat16 && hw.has_neural_accelerator {
        println!();
        println!("  WARNING: This model uses BFloat16 weights.");
        println!("  The M5 Neural Accelerator does not support BFloat16 computation.");
        println!("  For best performance, use an INT8 or FP16 quantized variant of this model.");
    }

    // Advisory KV-cache-mode suggestions (issue #327). Printed only; never
    // applied. No-op when the architecture could not be classified.
    print_kv_cache_advice(&advice.kv_cache_advice);

    println!();
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(hidden: u64, layers: u64, vocab: u64, ffn: u64) -> serde_json::Value {
        serde_json::json!({
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "intermediate_size": ffn,
            "num_attention_heads": 32,
            "num_key_value_heads": 8
        })
    }

    #[test]
    fn estimates_llama3_8b_in_correct_range() {
        // Llama 3 8B: 4096 hidden, 32 layers, 128256 vocab, 14336 ffn, GQA 32/8.
        let config = make_config(4096, 32, 128256, 14336);
        let params = estimate_params_from_config(&config).unwrap();
        // Should estimate between 7B and 10B.
        assert!(
            (7.0..=10.0).contains(&params),
            "Expected 7-10B for Llama3-8B config, got {:.2}B",
            params
        );
    }

    #[test]
    fn estimates_qwen2_0_5b_in_correct_range() {
        // Qwen2 0.5B: 896 hidden, 24 layers, 151936 vocab, 4864 ffn.
        let config = make_config(896, 24, 151936, 4864);
        let params = estimate_params_from_config(&config).unwrap();
        assert!(
            (0.3..=1.0).contains(&params),
            "Expected 0.3-1.0B for Qwen2-0.5B config, got {:.2}B",
            params
        );
    }

    #[test]
    fn text_config_nested_works() {
        // VLM style: text model params under "text_config".
        let config = serde_json::json!({
            "model_type": "llava",
            "text_config": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "vocab_size": 32000,
                "intermediate_size": 11008,
                "num_attention_heads": 32
            }
        });
        let params = estimate_params_from_config(&config).unwrap();
        assert!(
            params > 5.0,
            "Expected >5B for LLaVA-7B config, got {:.2}B",
            params
        );
    }

    #[test]
    fn kv_cache_estimate_prefers_explicit_head_dim() {
        let config = serde_json::json!({
            "text_config": {
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256
            }
        });

        let bytes = estimate_kv_cache_bytes_from_config(&config, 256, false).unwrap();
        assert_eq!(bytes, 35 * 2 * 256 * 2 * 256);
    }

    #[test]
    fn returns_none_for_empty_config() {
        let config = serde_json::json!({});
        let result = estimate_params_from_config(&config);
        assert!(result.is_none());
    }

    #[test]
    fn detects_bfloat16_from_torch_dtype() {
        // model_uses_bfloat16 reads from disk, so we test the underlying logic.
        let bfloat16_dtypes = ["bfloat16", "bf16", "torch.bfloat16"];
        for dtype in &bfloat16_dtypes {
            assert!(
                BFLOAT16_DTYPE_STRINGS.contains(dtype),
                "{dtype} should be recognized as bfloat16"
            );
        }
        assert!(!BFLOAT16_DTYPE_STRINGS.contains(&"float16"));
        assert!(!BFLOAT16_DTYPE_STRINGS.contains(&"fp16"));
    }

    #[test]
    fn advise_quantization_uses_override_params() {
        use mlxcel_core::hardware::{AppleSiliconGen, HardwareCapabilities};

        let hw = HardwareCapabilities {
            silicon_gen: AppleSiliconGen::M5,
            gpu_core_count: 10,
            has_neural_accelerator: true,
            metal_version: 4,
            macos_supports_na: true,
            memory_bandwidth_gbps: 150.0,
            unified_memory_gb: 32,
        };

        // Pass a temp dir — override_params bypasses config.json read.
        let tmp = std::env::temp_dir();
        let advice = advise_quantization(&tmp, &hw, Some(7.0));

        // With 7B model on 32 GB M5 with NA: expect INT8.
        assert_eq!(
            advice.recommendation,
            QuantRecommendation::Int8 {
                reason: "M5 NA delivers 2x throughput for INT8 vs FP16",
            }
        );
    }

    #[test]
    fn advise_quantization_exact_bytes_field_is_none_for_empty_dir() {
        use mlxcel_core::hardware::{AppleSiliconGen, HardwareCapabilities};

        let hw = HardwareCapabilities {
            silicon_gen: AppleSiliconGen::M5,
            gpu_core_count: 10,
            has_neural_accelerator: false,
            metal_version: 4,
            macos_supports_na: false,
            memory_bandwidth_gbps: 100.0,
            unified_memory_gb: 16,
        };

        let tmp = tempfile::tempdir().unwrap();
        let advice = advise_quantization(tmp.path(), &hw, None);
        assert_eq!(advice.exact_weight_bytes, None);
    }

    #[test]
    fn advise_quantization_uses_exact_bytes_from_index() {
        use mlxcel_core::hardware::{AppleSiliconGen, HardwareCapabilities};
        use std::io::Write;

        let hw = HardwareCapabilities {
            silicon_gen: AppleSiliconGen::M5,
            gpu_core_count: 10,
            has_neural_accelerator: false,
            metal_version: 4,
            macos_supports_na: false,
            memory_bandwidth_gbps: 100.0,
            unified_memory_gb: 16,
        };

        // Write an index.json with a known total_size (7B FP16 = ~14 GB = 14_000_000_000 bytes).
        let tmp = tempfile::tempdir().unwrap();
        let index_json =
            r#"{"metadata": {"total_size": 14000000000}, "weight_map": {"w": "x.safetensors"}}"#;
        let mut f = std::fs::File::create(tmp.path().join("model.safetensors.index.json")).unwrap();
        f.write_all(index_json.as_bytes()).unwrap();
        std::fs::File::create(tmp.path().join("x.safetensors")).unwrap();

        let advice = advise_quantization(tmp.path(), &hw, None);
        assert_eq!(advice.exact_weight_bytes, Some(14_000_000_000));
    }

    #[test]
    fn format_bytes_gib() {
        // 2 GiB exactly.
        let s = format_bytes(2 * 1024 * 1024 * 1024);
        assert!(s.contains("GiB"), "expected GiB in: {s}");
        assert!(s.contains("2147483648"), "expected raw bytes in: {s}");
    }

    #[test]
    fn format_bytes_mib() {
        let s = format_bytes(5 * 1024 * 1024);
        assert!(s.contains("MiB"), "expected MiB in: {s}");
    }

    #[test]
    fn format_bytes_small() {
        let s = format_bytes(42);
        assert_eq!(s, "42 bytes");
    }
}
