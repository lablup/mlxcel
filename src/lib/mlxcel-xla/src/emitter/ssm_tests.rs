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

//! Pure-Rust emit checks for the SSM mixer primitives (issue #3337). The
//! numeric gate against HF is `spike/openxla/ssm_oracle.py` with
//! `examples/xla_ssm_probe.rs`.

use super::builder::{Builder, Precision, Ty};
use super::config::SsmConfig;
use super::model::{ProbeMode, emit_ssm_probe, emit_ssm_probe_with};
use super::ssm;
use super::ssm_scan::SsmPrefillForm;

const HIDDEN: usize = 16;

fn signature(graph: &str) -> &str {
    graph
        .lines()
        .find(|line| line.contains("func.func public @main"))
        .expect("probe has a main")
}

fn assert_lowerable(graph: &str) {
    for banned in ["scatter", "stablehlo.while", "stablehlo.custom_call"] {
        assert!(!graph.contains(banned), "probe must not emit {banned}");
    }
}

#[test]
fn ssm_config_validates_structure() {
    let c = SsmConfig::tiny_mamba2();
    c.validate().expect("tiny config is valid");
    assert_eq!(c.d_inner(), 32);
    assert_eq!(c.in_proj_dim(), 32 + c.conv_dim + 4);

    let mut bad = c.clone();
    bad.conv_dim += 1;
    assert!(bad.validate().unwrap_err().contains("conv_dim"));
    let mut bad = c.clone();
    bad.conv_kernel = 1;
    assert!(bad.validate().unwrap_err().contains("conv_kernel"));
    let mut bad = c.clone();
    bad.n_groups = 3;
    assert!(bad.validate().unwrap_err().contains("n_groups"));
    let mut bad = c.clone();
    bad.norm_group_size = 5;
    assert!(bad.validate().unwrap_err().contains("norm_group_size"));
    let mut bad = c.clone();
    bad.time_step_limit = (1.0, 0.5);
    assert!(bad.validate().unwrap_err().contains("time_step_limit"));
}

/// The dims will come from an untrusted config.json (#3338): products that wrap
/// must not satisfy the conv_dim equation, and clamps or eps that make every
/// state NaN are rejected up front.
#[test]
fn ssm_config_rejects_overflow_and_non_finite_parameters() {
    let c = SsmConfig::tiny_mamba2();
    // H * P wraps to 0 in release arithmetic, so conv_dim = 2*G*N would match.
    let mut bad = c.clone();
    bad.n_heads = 2;
    bad.n_groups = 2;
    bad.head_dim = usize::MAX / 2 + 1;
    bad.norm_group_size = 1;
    bad.conv_dim = 2 * bad.n_groups * bad.state_size;
    assert!(bad.validate().unwrap_err().contains("overflow"));
    let mut bad = c.clone();
    bad.conv_kernel = usize::MAX;
    assert!(bad.validate().unwrap_err().contains("overflow"));

    for limit in [
        (f32::INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::NEG_INFINITY),
    ] {
        let mut bad = c.clone();
        bad.time_step_limit = limit;
        assert!(
            bad.validate().unwrap_err().contains("time_step_limit"),
            "{limit:?}"
        );
    }
    let mut ok = c.clone();
    ok.time_step_limit = (f32::NEG_INFINITY, f32::INFINITY);
    ok.validate().expect("an open clamp is valid");
    for eps in [f32::NAN, f32::INFINITY, -1e-5] {
        let mut bad = c.clone();
        bad.norm_eps = eps;
        assert!(bad.validate().unwrap_err().contains("norm_eps"), "{eps}");
    }
}

#[test]
#[should_panic(expected = "rows and hidden must be positive")]
fn ssm_probe_refuses_zero_rows() {
    let _ = emit_ssm_probe(&SsmConfig::tiny_mamba2(), HIDDEN, 0, ProbeMode::Prefill);
}

#[test]
fn ssm_prefill_probe_signature_and_results() {
    let c = SsmConfig::tiny_mamba2();
    let graph = emit_ssm_probe(&c, HIDDEN, 16, ProbeMode::Prefill);
    let sig = signature(&graph);
    // weights (conv bias on, linear biases off), then x, real_len, h0, conv0.
    for arg in [
        "tensor<132x16xf32> loc(\"ssm.in_proj\")",
        "tensor<96x1x4xf32> loc(\"ssm.conv_w\")",
        "tensor<96xf32> loc(\"ssm.conv_b\")",
        "tensor<16x32xf32> loc(\"ssm.out_proj\")",
        "tensor<16x16xf32> loc(\"ssm.x\")",
        "tensor<i32> loc(\"ssm.real_len\")",
        "tensor<4x8x16xf32> loc(\"ssm.h0\")",
        "tensor<3x96xf32> loc(\"ssm.conv0\")",
    ] {
        assert!(sig.contains(arg), "missing {arg} in {sig}");
    }
    assert!(!sig.contains("in_proj_b"), "use_bias=false takes no bias");
    assert!(sig.contains("-> (tensor<16x16xf32>, tensor<4x8x16xf32>, tensor<3x96xf32>)"));
    // The tail is a dynamic_slice at real_len; padded rows are masked by select.
    assert!(graph.contains("stablehlo.dynamic_slice"));
    assert!(graph.contains("stablehlo.compare LT"));
    assert!(graph.contains("stablehlo.reduce_window"));
    assert_lowerable(&graph);
}

#[test]
fn ssm_decode_probe_carries_batched_state() {
    let mut c = SsmConfig::tiny_mamba2();
    c.use_bias = true;
    let graph = emit_ssm_probe(&c, HIDDEN, 3, ProbeMode::Decode);
    let sig = signature(&graph);
    assert!(sig.contains("loc(\"ssm.in_proj_b\")"));
    assert!(sig.contains("loc(\"ssm.out_proj_b\")"));
    assert!(sig.contains("tensor<3x4x8x16xf32> loc(\"ssm.h0\")"));
    assert!(sig.contains("tensor<3x3x96xf32> loc(\"ssm.conv0\")"));
    assert!(!sig.contains("real_len"), "decode takes no real_len");
    assert!(sig.contains("-> (tensor<3x16xf32>, tensor<3x4x8x16xf32>, tensor<3x3x96xf32>)"));
    assert!(!graph.contains("reduce_window"), "decode has no scan");
    assert_lowerable(&graph);
}

#[test]
fn ssm_every_prefill_form_emits_the_same_interface() {
    let c = SsmConfig::tiny_mamba2();
    let mut sigs = Vec::new();
    for form in ["chunked", "associative", "sequential"] {
        let form = SsmPrefillForm::parse(form).expect("known form");
        let graph = emit_ssm_probe_with(&c, HIDDEN, 16, ProbeMode::Prefill, form, Precision::F32);
        assert_lowerable(&graph);
        sigs.push(signature(&graph).to_string());
    }
    assert!(sigs.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(SsmPrefillForm::parse("unrolled").is_err());
}

#[test]
fn ssm_prefill_handles_ragged_and_short_capacities() {
    let c = SsmConfig::tiny_mamba2(); // chunk_size 8, K 4
    // T not a multiple of the chunk: padded inside the scan, sliced back out.
    let graph = emit_ssm_probe(&c, HIDDEN, 13, ProbeMode::Prefill);
    assert!(signature(&graph).contains("-> (tensor<13x16xf32>"));
    assert!(
        graph.contains("tensor<16x4x8xf32>"),
        "13 rows pad to two chunks"
    );
    // Capacities shorter than the conv tail (T < K-1) and a single row.
    for t in [1, 2] {
        let graph = emit_ssm_probe(&c, HIDDEN, t, ProbeMode::Prefill);
        assert!(signature(&graph).contains(&format!("-> (tensor<{t}x16xf32>")));
        assert!(graph.contains("tensor<3x96xf32>"), "tail stays K-1 rows");
    }
}

#[test]
fn ssm_scan_contractions_stay_f32_on_the_f16_path() {
    let c = SsmConfig::tiny_mamba2();
    let graph = emit_ssm_probe_with(
        &c,
        HIDDEN,
        16,
        ProbeMode::Prefill,
        SsmPrefillForm::ChunkedSsd,
        Precision::F16,
    );
    let dots: Vec<&str> = graph
        .lines()
        .filter(|line| line.contains("stablehlo.dot_general"))
        .collect();
    let demoted = dots.iter().filter(|line| line.contains("xf16>")).count();
    let full = dots.len() - demoted;
    // in_proj and out_proj demote; the five chunked-SSD contractions do not.
    assert_eq!(demoted, 2, "{dots:#?}");
    assert_eq!(full, 5, "{dots:#?}");
}

#[test]
fn ssm_conv_tail_and_step_shapes() {
    let mut b = Builder::new();
    let x = Builder::arg(0, Ty::f32(vec![5, 6]));
    let real_len = Builder::arg(1, Ty::scalar("i32"));
    let tail = ssm::conv1d_tail(&mut b, &x, &real_len, 4);
    assert_eq!(tail.ty.shape, vec![3, 6]);
    let w = Builder::arg(2, Ty::f32(vec![6, 1, 4]));
    let y = ssm::conv1d_causal_prefill(&mut b, &x, &w, None, 4);
    assert_eq!(y.ty.shape, vec![5, 6]);
    let x_t = Builder::arg(3, Ty::f32(vec![2, 6]));
    let carried = Builder::arg(4, Ty::f32(vec![2, 3, 6]));
    let (y_t, next) = ssm::conv1d_causal_step(&mut b, &x_t, &carried, &w, None, 4);
    assert_eq!(y_t.ty.shape, vec![2, 6]);
    assert_eq!(next.ty.shape, vec![2, 3, 6]);
}
