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

//! A two-layer transformer with random bf16 weights against upstream's
//! `Ideogram4Transformer` on the same weights and inputs
//! (`tests/fixtures/imagegen/ideogram4_tiny_refs.safetensors`, written by
//! `tests/fixtures/generate_ideogram4_fixtures.py --tiny` on mlx 0.32.4 CUDA,
//! NVIDIA GB10). It covers the interleaved M-RoPE at image positions past
//! 65536, the norms, the AdaLN gating, the LLM branch and the final layer.

use std::path::PathBuf;

use mlxcel_core::weights::{WeightMap, load_safetensors};

use super::*;
use crate::imagegen::testing::{rel_rms, values};

fn fixture() -> WeightMap {
    load_safetensors(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/imagegen/ideogram4_tiny_refs.safetensors"),
    )
    .expect("tiny reference loads")
}

fn tiny_config() -> Ideogram4TransformerConfig {
    Ideogram4TransformerConfig {
        emb_dim: 32,
        num_layers: 2,
        num_heads: 2,
        intermediate_size: 48,
        adaln_dim: 16,
        in_channels: 8,
        llm_features_dim: 24,
        rope_theta: 10000.0,
        mrope_section: [4, 2, 2],
        norm_eps: 1e-5,
    }
}

fn model(refs: &WeightMap) -> Ideogram4Transformer {
    let weights: WeightMap = refs
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("w.")
                .map(|k| (k.to_string(), mlxcel_core::copy(v)))
        })
        .collect();
    Ideogram4Transformer::from_weights(tiny_config(), &weights).unwrap()
}

fn slice_tokens(a: &MlxArray, start: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(a);
    let mut begin = vec![0; s.len()];
    begin[1] = start;
    mlxcel_core::slice(a, &begin, &s)
}

/// The tables take `cos`/`sin` of `position * inv_freq` with positions past
/// 65536, so the last bit of the device's `pow`, `cos` and `sin` shows up as
/// a few 1e-5 (ROCm: up to 3.2e-6 in cos and 3.0e-5 in sin here; the CPU stream is
/// within 2e-8 of the CUDA reference). The tables are rounded to bf16, whose
/// spacing is 4e-3, before use, so this is far below anything the model
/// sees. A wrong axis selector or frequency moves entries by 1e-1 or more.
const MAX_TABLE_ABS_ERR: f32 = 1e-4;

#[test]
fn the_rope_tables_match_upstream() {
    let refs = fixture();
    let rope = super::super::blocks::MRope::new(16, 10000.0, [4, 2, 2]);
    let (cos, sin) = rope.tables(&refs["position_ids"]);
    for (name, got, want) in [("cos", &cos, &refs["cos"]), ("sin", &sin, &refs["sin"])] {
        let (got, want) = (values(got), values(want));
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= MAX_TABLE_ABS_ERR,
                "{name}[{i}]: {g} vs {w}"
            );
        }
    }
}

/// Bound of the bf16 forwards against upstream's CUDA run.
///
/// CUDA reproduces upstream bit for bit at this size, so a last-bit
/// kernel change after an MLX bump is the only thing the 1e-3 allows for, and
/// a port with the pairwise (Z-Image) rotation, the fused RMSNorm kernel or a
/// missing mask on the biased projections moves the result by 1e-2 or more.
///
/// Any other backend rounds a few bf16 intermediates differently (a different
/// accumulation order in the matmuls, a different `exp`), and this random
/// model amplifies each flipped bit: upstream's own bf16 run is 0.37 from the
/// f32 forward of the same weights, so two bf16 runs that do not match bit for
/// bit land about 1e-2 apart. Measured against the CUDA reference: ROCm
/// (gfx1151) 1.41e-2 conditional and 2.37e-2 unconditional, the CPU stream
/// 1.61e-2 conditional, the M1 Ultra Metal GPU 1.53e-2 conditional and
/// 2.04e-2 unconditional. The same f32 forward on the GPU and on the CPU
/// agrees to 3e-5 on ROCm and passes the same 1e-4 on M1 Ultra, so the gap
/// is rounding and not a kernel defect
/// (`the_f32_forward_agrees_between_the_gpu_and_the_cpu`). The bound there is
/// 5e-2: the pairwise rotation measured 5.2e-1 and 3.2e-1 and the missing
/// mask 2.0e-1 on ROCm. The fused RMSNorm kernel (1.5e-2 on ROCm) is below
/// that noise and is only caught on CUDA.
fn max_rel_rms() -> f64 {
    use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
    if matches!(gpu_backend_kind(), GpuBackendKind::Cuda) {
        1e-3
    } else {
        5e-2
    }
}

#[test]
fn conditional_and_unconditional_forwards_match_upstream() {
    // The f32 test below moves the default device to the CPU while it holds this.
    let _lock = mlxcel_core::streams::lock_default_device();
    let max_rel_rms = max_rel_rms();
    let refs = fixture();
    let model = model(&refs);
    let llm = model
        .project_llm_features(&refs["llm"], &refs["indicator"])
        .unwrap();
    let cond = model
        .forward(
            Some(&llm),
            &refs["x"],
            &refs["t"],
            &TokenInputs {
                position_ids: &refs["position_ids"],
                indicator: &refs["indicator"],
                mask: None,
            },
        )
        .unwrap();
    let cond_err = rel_rms(&values(&cond), &values(&refs["cond"]));
    let uncond = model
        .forward(
            None,
            &slice_tokens(&refs["x"], 3),
            &refs["t"],
            &TokenInputs {
                position_ids: &slice_tokens(&refs["position_ids"], 3),
                indicator: &slice_tokens(&refs["indicator"], 3),
                mask: None,
            },
        )
        .unwrap();
    let uncond_err = rel_rms(&values(&uncond), &values(&refs["uncond"]));
    eprintln!(
        "tiny transformer: conditional rel-RMS {cond_err:.3e}, unconditional {uncond_err:.3e}"
    );
    assert_eq!(mlxcel_core::array_dtype(&cond), dtype::FLOAT32);
    assert!(cond_err < max_rel_rms, "conditional {cond_err}");
    assert!(uncond_err < max_rel_rms, "unconditional {uncond_err}");
}

/// Both forwards on the weights widened to f32, where the backends agree to
/// rounding error instead of flipping bf16 bits. This is the check that a
/// device kernel (a norm, the attention, a matmul) does what the CPU does;
/// 3e-5 was measured between the ROCm GPU and the CPU stream.
#[test]
fn the_f32_forward_agrees_between_the_gpu_and_the_cpu() {
    use mlxcel_core::streams::{DefaultDeviceGuard, lock_default_device};
    if !mlxcel_core::gpu_backend_available() {
        return;
    }
    let _lock = lock_default_device();
    let refs = fixture();
    let run = |cpu: bool| {
        let _device = cpu.then(DefaultDeviceGuard::cpu);
        let weights: WeightMap = refs
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("w.")
                    .map(|k| (k.to_string(), mlxcel_core::astype(v, dtype::FLOAT32)))
            })
            .collect();
        let model = Ideogram4Transformer::from_weights(tiny_config(), &weights).unwrap();
        let llm = model
            .project_llm_features(&refs["llm"], &refs["indicator"])
            .unwrap();
        let cond = model
            .forward(
                Some(&llm),
                &refs["x"],
                &refs["t"],
                &TokenInputs {
                    position_ids: &refs["position_ids"],
                    indicator: &refs["indicator"],
                    mask: None,
                },
            )
            .unwrap();
        let uncond = model
            .forward(
                None,
                &slice_tokens(&refs["x"], 3),
                &refs["t"],
                &TokenInputs {
                    position_ids: &slice_tokens(&refs["position_ids"], 3),
                    indicator: &slice_tokens(&refs["indicator"], 3),
                    mask: None,
                },
            )
            .unwrap();
        (values(&cond), values(&uncond))
    };
    let (gpu_cond, gpu_uncond) = run(false);
    let (cpu_cond, cpu_uncond) = run(true);
    let cond_err = rel_rms(&gpu_cond, &cpu_cond);
    let uncond_err = rel_rms(&gpu_uncond, &cpu_uncond);
    eprintln!("f32 gpu vs cpu: conditional {cond_err:.3e}, unconditional {uncond_err:.3e}");
    assert!(cond_err < 1e-4, "conditional {cond_err}");
    assert!(uncond_err < 1e-4, "unconditional {uncond_err}");
}

#[test]
fn a_single_segment_mask_is_all_zeros() {
    let ids = mlxcel_core::from_slice_i32(&[1, 1, 1], &[1, 3]);
    assert!(
        values(&segment_mask(&ids, dtype::FLOAT32))
            .iter()
            .all(|&v| v == 0.0)
    );
    let ids = mlxcel_core::from_slice_i32(&[1, 1, 2], &[1, 3]);
    let mask = values(&segment_mask(&ids, dtype::FLOAT32));
    assert_eq!(mask[..3], [0.0, 0.0, f32::NEG_INFINITY]);
    assert_eq!(mask[8], 0.0);
}

#[test]
fn llm_features_of_the_wrong_width_are_rejected() {
    let refs = fixture();
    let model = model(&refs);
    let bad = mlxcel_core::zeros(&[1, 9, 23], dtype::FLOAT32);
    assert!(
        model
            .project_llm_features(&bad, &refs["indicator"])
            .is_err()
    );
}

fn weights_with(
    refs: &WeightMap,
    key: &str,
    replacement: Option<UniquePtr<MlxArray>>,
) -> WeightMap {
    let mut weights: WeightMap = refs
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("w.")
                .map(|k| (k.to_string(), mlxcel_core::copy(v)))
        })
        .collect();
    if let Some(r) = replacement {
        weights.insert(key.to_string(), r);
    }
    weights
}

#[test]
fn a_weight_of_the_wrong_shape_is_rejected_at_load() {
    let refs = fixture();
    let cfg = tiny_config();
    let wrong = mlxcel_core::zeros(&[cfg.emb_dim, cfg.in_channels + 1], dtype::FLOAT32);
    let weights = weights_with(&refs, "input_proj.weight", Some(wrong));
    let err = Ideogram4Transformer::from_weights(cfg.clone(), &weights)
        .err()
        .expect("a mismatched input_proj must be rejected")
        .to_string();
    assert!(err.contains("input_proj.weight"), "{err}");
    assert!(err.contains("[32, 9]") && err.contains("[32, 8]"), "{err}");
    // A layer weight is checked too.
    let wrong = mlxcel_core::zeros(&[3 * cfg.emb_dim, cfg.emb_dim - 1], dtype::FLOAT32);
    let weights = weights_with(&refs, "layers.1.attention.qkv.weight", Some(wrong));
    let err = Ideogram4Transformer::from_weights(cfg, &weights)
        .err()
        .expect("a mismatched qkv must be rejected")
        .to_string();
    assert!(err.contains("layers.1.attention.qkv.weight"), "{err}");
}

#[test]
fn a_raw_fp8_weight_is_rejected_at_load() {
    let refs = fixture();
    let cfg = tiny_config();
    let raw = mlxcel_core::zeros(&[cfg.emb_dim, cfg.in_channels], dtype::UINT8);
    let weights = weights_with(&refs, "input_proj.weight", Some(raw));
    let err = Ideogram4Transformer::from_weights(cfg, &weights)
        .err()
        .expect("a uint8 weight must be rejected")
        .to_string();
    assert!(
        err.contains("input_proj.weight") && err.contains("floating"),
        "{err}"
    );
}

#[test]
fn the_llm_projection_is_optional_and_errors_when_absent() {
    let refs = fixture();
    let mut weights = weights_with(&refs, "", None);
    weights.remove("llm_cond_norm.weight");
    weights.remove("llm_cond_proj.weight");
    let model = Ideogram4Transformer::from_weights(tiny_config(), &weights).unwrap();
    assert!(!model.has_llm_conditioning());
    let features = mlxcel_core::zeros(&[1, 9, 24], dtype::FLOAT32);
    let err = model
        .project_llm_features(&features, &refs["indicator"])
        .err()
        .expect("no projection weights")
        .to_string();
    assert!(err.contains("llm_cond"), "{err}");
}
