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

//! PLaMo 2 greedy-decode parity against the mlx-lm reference.
//!
//! PLaMo 2 ships a custom `PlamoTokenizer` (a numpy Unigram in `tokenizer.jsonl`)
//! that mlxcel's Rust tokenizer loader does not support, so end-to-end text
//! generation through the CLI is not yet available. The model architecture is
//! still validated here at the token-id level: identical input ids are fed to
//! mlxcel's `Plamo2Model` and the greedy decode is compared against the
//! reference ids captured from `mlx_lm` 0.31.3 (`mlx_lm.utils.load_model` runs
//! the upstream MLX PLaMo 2 with no tokenizer/torch dependency). Matching greedy
//! ids across the prefill -> first-token transition and 23 decode steps
//! exercises the interleaved Mamba/attention stack, the SSD scan (graph prefill
//! and fused decode kernel), the post-conv B/C/dt projection, and the offset /
//! double RMSNorms.
//!
//! The test skips when `models/plamo-2-1b` is absent (CI has no Metal and no
//! checkpoint), matching the other real-model parity tests.

use mlxcel::models::Plamo2Model;
use mlxcel_core::generate::LanguageModel;

const MODEL_DIR: &str = "models/plamo-2-1b";

// Fixed input ids (BOS = 1 plus arbitrary in-vocab ids). The point is parity for
// identical input, not semantic prompting; the greedy continuation is fully
// determined by the weights and architecture.
const INPUT_IDS: &[i32] = &[1, 2169, 290, 12978, 290, 8987, 466];

// Greedy (temp 0) continuation captured from mlx_lm 0.31.3 running the upstream
// MLX PLaMo 2 on INPUT_IDS.
const REF_GREEDY_OUT: &[i32] = &[
    515, 515, 10, 45311, 10, 45311, 10, 45311, 10, 45311, 10, 45311, 10, 45311, 10, 45311, 10,
    45311, 10, 45311, 10, 45311, 10, 45311,
];

fn argmax_last_token(logits: &mlxcel_core::MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let seq = shape[1];
    let vocab = shape[2];
    let last = mlxcel_core::slice(logits, &[0, seq - 1, 0], &[1, seq, vocab]);
    let argmax = mlxcel_core::argmax_last_axis(&last);
    mlxcel_core::eval(&argmax);
    mlxcel_core::item_i32(&argmax)
}

#[test]
fn plamo2_greedy_parity_matches_mlx_lm() {
    if !std::path::Path::new(MODEL_DIR).exists() {
        eprintln!("skipping plamo2_greedy_parity: {MODEL_DIR} not present");
        return;
    }

    let (model, _args) = Plamo2Model::load(MODEL_DIR).expect("load plamo-2-1b");

    // Reset the model-owned mixed cache for a fresh generation session.
    let mut caches = LanguageModel::make_caches(&model);

    let prompt = mlxcel_core::from_slice_i32(INPUT_IDS, &[1, INPUT_IDS.len() as i32]);
    let mut logits = LanguageModel::forward(&model, &prompt, &mut caches, None);

    let mut out = Vec::with_capacity(REF_GREEDY_OUT.len());
    for _ in 0..REF_GREEDY_OUT.len() {
        let tok = argmax_last_token(&logits);
        out.push(tok);
        let next = mlxcel_core::from_slice_i32(&[tok], &[1, 1]);
        logits = LanguageModel::forward(&model, &next, &mut caches, None);
    }

    assert_eq!(
        out, REF_GREEDY_OUT,
        "PLaMo 2 greedy decode diverged from the mlx-lm reference"
    );
}
