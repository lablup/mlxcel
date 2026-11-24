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

//! Where a K-row verify block first parts from K single-token steps on a
//! real Inkling 2 checkpoint (issue #3280). Ignored; run under the GPU lock:
//!
//! ```text
//! cargo test --profile test-fast --features metal,accelerate --lib \
//!     models::inkling2::exactness_diag_tests -- --ignored --nocapture
//! ```
//!
//! `INKLING2_DIAG_MODEL` picks the checkpoint (default `models/inkling-2-mini`),
//! `INKLING2_DIAG_WIDTH` the block width (default 8), and
//! `INKLING2_DIAG_VERIFY=0` runs the block outside the speculative scope.

use mlxcel_core::MlxArray;

use super::Inkling2Model;
use mlxcel_core::sampling::TokenBiasMap;

fn row_f32(x: &MlxArray, row: i32) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(x);
    let rest: i32 = shape[2..].iter().product();
    let x = &*mlxcel_core::reshape(x, &[shape[0], shape[1], rest]);
    let shape = mlxcel_core::array_shape(x);
    let slab = mlxcel_core::slice(x, &[0, row, 0], &[shape[0], row + 1, shape[2]]);
    let slab = mlxcel_core::astype(&slab, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::utils::array_to_vec_f32(&slab)
}

fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    a.iter().zip(b).fold((0, 0.0f32), |(n, m), (x, y)| {
        let d = (x - y).abs();
        (n + usize::from(x.to_bits() != y.to_bits()), m.max(d))
    })
}

#[test]
#[ignore = "needs a real Inkling 2 checkpoint and the GPU lock"]
fn verify_block_vs_chain_per_visit() {
    let path =
        std::env::var("INKLING2_DIAG_MODEL").unwrap_or_else(|_| "models/inkling-2-mini".into());
    if !std::path::Path::new(&path).join("config.json").exists() {
        eprintln!("skipping: {path} is not present");
        return;
    }
    let width: usize = std::env::var("INKLING2_DIAG_WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let verify = std::env::var("INKLING2_DIAG_VERIFY").map_or(true, |v| v != "0");
    let (model, _) = Inkling2Model::load(&path).unwrap();
    let visits = model.layer_indices.len();
    let ids: Vec<usize> = (0..=visits).collect();
    let none = TokenBiasMap::new();
    let vocab = model.config.text_config.logits_width();
    let prompt: Vec<i32> = (0..32).map(|i| ((i * 7 + 2) % vocab) as i32).collect();
    let block: Vec<i32> = (0..width).map(|i| ((i * 13 + 4) % vocab) as i32).collect();
    let input = |t: &[i32]| mlxcel_core::from_slice_i32(t, &[1, t.len() as i32]);

    let mut chain = model.make_internal_caches();
    let _ = model.forward_speculative(&input(&prompt), &mut chain, &[], false, &none);
    let chain_out: Vec<_> = block
        .iter()
        .map(|&t| model.forward_speculative(&input(&[t]), &mut chain, &ids, false, &none))
        .collect();

    let mut caches = model.make_internal_caches();
    let _ = model.forward_speculative(&input(&prompt), &mut caches, &[], false, &none);
    let out = model.forward_speculative(&input(&block), &mut caches, &ids, verify, &none);

    println!("model {path} width {width} verify {verify} visits {visits}");
    for (v, &layer) in model
        .layer_indices
        .iter()
        .chain(std::iter::once(&usize::MAX))
        .enumerate()
    {
        let mut line = format!("capture {v:>2} (layer {layer:>2}):");
        for r in 0..width {
            let a = row_f32(out.hidden_states[v].as_ref().unwrap(), r as i32);
            let b = row_f32(chain_out[r].hidden_states[v].as_ref().unwrap(), 0);
            let (n, m) = compare(&a, &b);
            line.push_str(&format!(" r{r} {n}/{m:.2e}"));
        }
        println!("{line}");
    }
    let mut line = "logits:".to_string();
    for (r, chain_r) in chain_out.iter().enumerate() {
        let a = row_f32(&out.logits, r as i32);
        let b = row_f32(&chain_r.logits, 0);
        let (n, m) = compare(&a, &b);
        line.push_str(&format!(" r{r} {n}/{m:.2e}"));
    }
    println!("{line}");
}
