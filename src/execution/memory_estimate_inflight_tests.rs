// Copyright 2025-2026 Lablup Inc.
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

//! Tests for the backend in-flight reserve of the memory estimate (issue
//! #2155): the `MLX_ROCM_MAX_INFLIGHT_MB` parser, the compile-time gate that
//! keeps Metal and CUDA totals unchanged, and the measured gfx1151 peaks the
//! ROCm estimate must stay above.

use super::*;

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Sets or clears one env var for the guard's lifetime. Callers hold
/// `crate::test_support::env_lock::env_lock()` for longer than the guard.
struct EnvGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn new(key: &'static str, value: Option<&str>) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: the creating test holds the crate-wide env lock, which
        // serializes every env mutation in this test binary.
        unsafe {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: as for `new`; the env lock outlives this guard.
        unsafe {
            match &self.previous {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

/// Clears every env var the estimate reads besides the in-flight budget, so a
/// developer's calibration overrides cannot move the figures under test.
fn neutral_estimator_env() -> [EnvGuard; 3] {
    [
        EnvGuard::new(HEADROOM_FACTOR_ENV, None),
        EnvGuard::new(ACTIVATION_MULT_ENV, None),
        EnvGuard::new(MEMORY_LIMIT_ENV, None),
    ]
}

fn write_model(dir: &Path, config: &serde_json::Value, weights_total_size: u64) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string(config).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        format!(
            r#"{{"metadata": {{"total_size": {weights_total_size}}}, "weight_map": {{"w": "x.safetensors"}}}}"#
        ),
    )
    .unwrap();
    std::fs::File::create(dir.join("x.safetensors")).unwrap();
}

fn llama_8b_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    write_model(tmp.path(), &llama_31_8b_config(), LLAMA_31_8B_WEIGHTS);
    tmp
}

// ── Parser ───────────────────────────────────────────────────────────────────

#[test]
fn reserve_parser_matches_the_overlay_for_the_specified_inputs() {
    let default = ROCM_DEFAULT_MAX_INFLIGHT_MB * MIB;
    assert_eq!(default, GIB);
    // One past the largest MiB count that still fits once shifted by 20.
    let overflows_when_shifted = ((u64::MAX >> 20) + 1).to_string();
    let cases: [(Option<&str>, u64); 8] = [
        (None, default),
        (Some(""), default),
        (Some("0"), 0),
        (Some("256"), 256 * MIB),
        (Some("-1"), default),
        (Some("8abc"), default),
        (Some("1G"), default),
        (Some(overflows_when_shifted.as_str()), default),
    ];
    for (raw, expected) in cases {
        assert_eq!(
            rocm_inflight_reserve_bytes(raw),
            expected,
            "MLX_ROCM_MAX_INFLIGHT_MB={raw:?}"
        );
    }
}

#[test]
fn reserve_parser_follows_strtoull_where_the_overlay_does() {
    let default = ROCM_DEFAULT_MAX_INFLIGHT_MB * MIB;
    let largest = u64::MAX >> 20;
    let cases: [(&str, u64); 11] = [
        ("4096", 4 * GIB),
        // strtoull skips leading C whitespace and one `+`, and the overlay
        // only rejects a `-` in the first byte.
        ("+256", 256 * MIB),
        (" \t256", 256 * MIB),
        // A `-` after whitespace is negated in unsigned arithmetic: `-0` is
        // 0, anything else wraps far past the shift limit.
        (" -0", 0),
        (" -5", default),
        // Trailing bytes (`*end != '\0'`), no digits, or digits beyond 64
        // bits (ERANGE) keep the default.
        ("256 ", default),
        (" ", default),
        ("+", default),
        ("0x10", default),
        ("99999999999999999999999", default),
        // The largest accepted count shifts without overflowing.
        ("17592186044415", largest << 20),
    ];
    assert_eq!(largest, 17_592_186_044_415);
    for (raw, expected) in cases {
        assert_eq!(
            rocm_inflight_reserve_bytes(Some(raw)),
            expected,
            "MLX_ROCM_MAX_INFLIGHT_MB={raw:?}"
        );
    }
}

#[test]
fn default_budget_and_env_name_match_the_overlay_source() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/device.cpp");
    let source = std::fs::read_to_string(&path).unwrap();
    let declaration =
        format!("constexpr size_t default_max_inflight_mb = {ROCM_DEFAULT_MAX_INFLIGHT_MB};");
    assert!(
        source.contains(&declaration),
        "{} no longer declares `{declaration}`; update ROCM_DEFAULT_MAX_INFLIGHT_MB",
        path.display()
    );
    assert!(
        source.contains(&format!("std::getenv(\"{ROCM_MAX_INFLIGHT_ENV}\")")),
        "{} no longer reads {ROCM_MAX_INFLIGHT_ENV}",
        path.display()
    );
}

// ── Fit inversion ─────────────────────────────────────────────────────────────

#[test]
fn auto_kv_budget_leaves_room_for_the_inflight_reserve() {
    let est = MemoryEstimate {
        weights_bytes: 10_000_000_000,
        kv_cache_bytes: 0,
        runtime_headroom_bytes: 0,
        activation_bytes: 1_000_000_000,
        backend_inflight_bytes: 1_000_000_000,
        total_bytes: 0,
        available_bytes: 25_000_000_000,
        fits: true,
        weights_source: WeightsSource::AnalyticalConfig,
        kv_source: KvSource::Config,
        kv_detail: String::new(),
        headroom_factor: 1.20,
        ctx_len: DEFAULT_CTX_LEN,
        batch: 1,
        quant: QuantHint::Default,
        kv_dtype_int8: false,
    };
    // floor((25e9 - 1e9 - 1e9) / 1.2) - 10e9.
    let budget = auto_kv_budget_bytes(&est);
    assert_eq!(budget, 9_166_666_666);
    let total = (1.20_f64 * (est.weights_bytes + budget) as f64) as u64
        + est.activation_bytes
        + est.backend_inflight_bytes;
    assert!(total <= est.available_bytes);
}

// ── Compile-time gate ─────────────────────────────────────────────────────────

/// The total is exactly the sum of its five terms, with the allocator
/// overhead recomputed from the factor so a term dropped from `total_bytes`
/// cannot hide inside `runtime_headroom_bytes`.
fn assert_total_is_the_sum_of_its_terms(est: &MemoryEstimate) {
    let overhead =
        compute_runtime_headroom(est.weights_bytes + est.kv_cache_bytes, est.headroom_factor);
    assert_eq!(
        est.runtime_headroom_bytes,
        overhead + est.activation_bytes + est.backend_inflight_bytes
    );
    assert_eq!(
        est.total_bytes,
        est.weights_bytes
            + est.kv_cache_bytes
            + overhead
            + est.activation_bytes
            + est.backend_inflight_bytes
    );
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_estimate_reserves_the_inflight_budget() {
    let _lock = crate::test_support::env_lock::env_lock();
    let _neutral = neutral_estimator_env();
    let tmp = llama_8b_dir();

    let est = {
        let _budget = EnvGuard::new(ROCM_MAX_INFLIGHT_ENV, None);
        estimate_total_memory(tmp.path(), 640, 1, QuantHint::Default, false)
    };
    assert_eq!(est.backend_inflight_bytes, GIB);
    assert_total_is_the_sum_of_its_terms(&est);
    let text = format_estimate(tmp.path(), &est);
    assert!(
        text.contains(&format!(
            "Backend in-flight: {}  (MLX_ROCM_MAX_INFLIGHT_MB)",
            format_bytes(GIB)
        )),
        "{text}"
    );
    let report = InspectReport::from_estimate(tmp.path(), &est, "fp16".into(), None, None, None);
    assert_eq!(report.backend_inflight_bytes, GIB);
    assert_eq!(
        report.weights_bytes + report.kv_bytes_total + report.headroom_bytes,
        report.total_bytes
    );
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["backend_inflight_bytes"], GIB);

    let small = {
        let _budget = EnvGuard::new(ROCM_MAX_INFLIGHT_ENV, Some("256"));
        estimate_total_memory(tmp.path(), 640, 1, QuantHint::Default, false)
    };
    assert_eq!(small.backend_inflight_bytes, 256 * MIB);
    assert_eq!(small.total_bytes + (GIB - 256 * MIB), est.total_bytes);

    // `0` turns the backend's bound off; the estimate adds nothing and does
    // not print the line.
    let off = {
        let _budget = EnvGuard::new(ROCM_MAX_INFLIGHT_ENV, Some("0"));
        estimate_total_memory(tmp.path(), 640, 1, QuantHint::Default, false)
    };
    assert_eq!(off.backend_inflight_bytes, 0);
    assert_eq!(off.total_bytes + GIB, est.total_bytes);
    assert!(!format_estimate(tmp.path(), &off).contains("Backend in-flight"));
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_largest_accepted_budget_saturates_the_total() {
    let _lock = crate::test_support::env_lock::env_lock();
    let _neutral = neutral_estimator_env();
    let _budget = EnvGuard::new(ROCM_MAX_INFLIGHT_ENV, Some("17592186044415"));
    let tmp = llama_8b_dir();
    let est = estimate_total_memory(tmp.path(), 640, 1, QuantHint::Default, false);
    assert_eq!(est.backend_inflight_bytes, (u64::MAX >> 20) << 20);
    assert_eq!(est.total_bytes, u64::MAX);
    assert!(!est.fits);
}

#[cfg(not(feature = "rocm"))]
#[test]
fn non_rocm_estimate_reserves_nothing_for_inflight() {
    let _lock = crate::test_support::env_lock::env_lock();
    let _neutral = neutral_estimator_env();
    // Even with the ROCm variable set, Metal and CUDA builds add nothing, so
    // their totals are what they were before issue #2155.
    let _budget = EnvGuard::new(ROCM_MAX_INFLIGHT_ENV, Some("4096"));
    let tmp = llama_8b_dir();
    let est = estimate_total_memory(tmp.path(), 640, 1, QuantHint::Default, false);
    assert_eq!(est.backend_inflight_bytes, 0);
    assert_total_is_the_sum_of_its_terms(&est);
    assert!(!format_estimate(tmp.path(), &est).contains("Backend in-flight"));
    let report = InspectReport::from_estimate(tmp.path(), &est, "fp16".into(), None, None, None);
    assert_eq!(report.backend_inflight_bytes, 0);
}

// ── Measured gfx1151 peaks ────────────────────────────────────────────────────

/// `metadata.total_size` of the checkpoints the peaks were measured on.
const LLAMA_31_8B_WEIGHTS: u64 = 4_517_404_672;

fn llama_31_8b_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
        "max_position_embeddings": 131072, "tie_word_embeddings": false,
        "quantization": {"group_size": 64, "bits": 4}
    })
}

#[cfg(feature = "rocm")]
mod measured {
    use super::*;

    const QWEN25_7B_WEIGHTS: u64 = 4_284_263_424;
    const QWEN3_30B_A3B_WEIGHTS: u64 = 17_174_622_208;

    fn qwen25_7b_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
            "hidden_size": 3584, "intermediate_size": 18944, "num_hidden_layers": 28,
            "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064,
            "max_position_embeddings": 32768, "max_window_layers": 28,
            "sliding_window": 131072, "use_sliding_window": false,
            "tie_word_embeddings": false, "quantization": {"group_size": 64, "bits": 4}
        })
    }

    fn qwen3_30b_a3b_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
            "hidden_size": 2048, "intermediate_size": 6144, "moe_intermediate_size": 768,
            "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
            "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8,
            "decoder_sparse_step": 1, "mlp_only_layers": [], "vocab_size": 151936,
            "max_position_embeddings": 40960, "max_window_layers": 48,
            "sliding_window": null, "use_sliding_window": false,
            "tie_word_embeddings": false, "quantization": {"group_size": 64, "bits": 4}
        })
    }

    /// The checkpoints the peaks were measured on.
    type Model = (&'static str, fn() -> serde_json::Value, u64);
    const LLAMA: Model = ("Llama-3.1-8B", llama_31_8b_config, LLAMA_31_8B_WEIGHTS);
    const QWEN3_MOE: Model = ("Qwen3-30B-A3B", qwen3_30b_a3b_config, QWEN3_30B_A3B_WEIGHTS);
    const QWEN25: Model = ("Qwen2.5-7B", qwen25_7b_config, QWEN25_7B_WEIGHTS);

    /// One run per row: `(model, ctx_len, MLX_ROCM_MAX_INFLIGHT_MB, peak)`, from
    /// `mlxcel-bench-decode --prompt-tokens (ctx_len - 128) -n 128
    /// --warmup-tokens 20 --ignore-eos` on gfx1151 under
    /// `scripts/rocm_gpu_guard.sh`. `peak` is the printed `MLX peak memory`
    /// (bytes / 1e9, two decimals) in hundredths of a GB; [`peak_upper_bound`]
    /// adds the 0.005 GB its rounding can hide. The table is "Estimate against
    /// peak" in `docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`.
    #[rustfmt::skip]
    const MEASURED_PEAKS: &[(Model, u64, Option<&str>, u64)] = &[
        (LLAMA, 640, None, 614), (LLAMA, 640, Some("256"), 518), (LLAMA, 640, Some("4096"), 923),
        (LLAMA, 4096, None, 695), (LLAMA, 4096, Some("256"), 602), (LLAMA, 4096, Some("4096"), 1011),
        (QWEN3_MOE, 640, None, 1866), (QWEN3_MOE, 640, Some("256"), 1784),
        (QWEN3_MOE, 640, Some("4096"), 2181), (QWEN3_MOE, 4096, None, 1891),
        (QWEN3_MOE, 4096, Some("256"), 1878), (QWEN3_MOE, 4096, Some("4096"), 1926),
        (QWEN25, 640, None, 591), (QWEN25, 640, Some("256"), 485), (QWEN25, 640, Some("4096"), 912),
        (QWEN25, 4096, None, 617), (QWEN25, 4096, Some("256"), 541), (QWEN25, 4096, Some("4096"), 945),
    ];

    const fn peak_upper_bound(centi_gb: u64) -> u64 {
        centi_gb * 10_000_000 + 5_000_000
    }

    #[test]
    fn rocm_estimate_covers_every_measured_gfx1151_peak() {
        let _lock = crate::test_support::env_lock::env_lock();
        let _neutral = neutral_estimator_env();
        assert!(!MEASURED_PEAKS.is_empty());
        let mut uncovered_without_term = 0;
        for &((model, config, weights), ctx_len, budget_mb, centi_gb) in MEASURED_PEAKS {
            let peak = peak_upper_bound(centi_gb);
            let tmp = tempfile::tempdir().unwrap();
            write_model(tmp.path(), &config(), weights);
            let est = {
                let _budget = EnvGuard::new(ROCM_MAX_INFLIGHT_ENV, budget_mb);
                estimate_total_memory(tmp.path(), ctx_len, 1, QuantHint::Default, false)
            };
            assert!(
                est.total_bytes >= peak,
                "{model} at {ctx_len} tokens, MLX_ROCM_MAX_INFLIGHT_MB={budget_mb:?}: estimate \
                 {} B is below the measured peak {peak} B",
                est.total_bytes
            );
            if est.total_bytes - est.backend_inflight_bytes < peak {
                uncovered_without_term += 1;
            }
        }
        // The reserve is load-bearing: without it the estimate falls below the
        // measured peak on at least one row.
        assert!(uncovered_without_term > 0);
    }
}
