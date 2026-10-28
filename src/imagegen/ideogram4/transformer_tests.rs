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

#[test]
fn the_rope_tables_match_upstream() {
    let refs = fixture();
    let rope = super::super::blocks::MRope::new(16, 10000.0, [4, 2, 2]);
    let (cos, sin) = rope.tables(&refs["position_ids"]);
    assert_eq!(values(&cos), values(&refs["cos"]));
    assert_eq!(values(&sin), values(&refs["sin"]));
}

/// Both forwards must be bit-identical to upstream at this size; the bound
/// only allows for a last-bit kernel change after an MLX bump. A port with
/// the pairwise (Z-Image) rotation, the fused RMSNorm kernel or a missing
/// mask on the biased projections moves these by 1e-2 or more.
const MAX_REL_RMS: f64 = 1e-3;

#[test]
fn conditional_and_unconditional_forwards_match_upstream() {
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
    assert!(cond_err < MAX_REL_RMS, "conditional {cond_err}");
    assert!(uncond_err < MAX_REL_RMS, "unconditional {uncond_err}");
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
