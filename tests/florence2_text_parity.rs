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

//! Florence-2 text-core parity against the mlx-vlm reference.
//!
//! Pins the BART seq2seq engine (post-norm blocks, position-table offset 2,
//! dual KV cache, tied/materialized LM head) against reference logits
//! captured from the mlx-vlm florence2 `LanguageModel`
//! (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/language.py)
//! running the same `models/Florence-2-base-ft-bf16` checkpoint with
//! weights cast bf16 -> f16, matching mlxcel's Apple Silicon precision
//! policy. Skips when the checkpoint is absent (CI has no Metal and no
//! weights).

use std::path::Path;

use mlxcel::models::Florence2TextModel;

const MODEL_DIR: &str = "models/Florence-2-base-ft-bf16";

// Arbitrary but fixed BART-vocab token sequence: <s> ... </s>.
const INPUT_IDS: &[i32] = &[0, 713, 16, 10, 1296, 4, 2];
// shift_tokens_right(INPUT_IDS) with decoder_start_token_id = 2.
const DEC_IDS: &[i32] = &[2, 0, 713, 16, 10, 1296, 4];

// Reference values from the mlx-vlm florence2 LanguageModel (f16 weights,
// f32 logits readout) on the same checkpoint and token ids.
const REF_TF_ARGMAX: &[i32] = &[0, 2362, 16, 10, 2170, 13, 2];
const REF_GREEDY: &[i32] = &[0, 2362]; // greedy stops at EOS (= 2) after these
// The full digit strings are the exact f16-representable reference values;
// truncating them would move the pins off the values the reference produced.
#[allow(clippy::excessive_precision)]
const REF_LOGITS_POS0_FIRST16: &[f32] = &[
    21.375,
    -7.11328125,
    5.90625,
    -7.12109375,
    -2.314453125,
    -3.068359375,
    0.37255859375,
    -0.1904296875,
    -0.478515625,
    -0.6201171875,
    -3.24609375,
    -0.97509765625,
    -0.95849609375,
    0.41015625,
    -2.615234375,
    -1.4384765625,
];
#[allow(clippy::excessive_precision)]
const REF_LOGITS_LAST_FIRST16: &[f32] = &[
    0.7998046875,
    -5.79296875,
    14.1875,
    -5.796875,
    5.0859375,
    5.7890625,
    1.525390625,
    3.5234375,
    4.36328125,
    1.8525390625,
    4.97265625,
    3.974609375,
    1.5615234375,
    2.51953125,
    4.37109375,
    3.396484375,
];

fn to_vec_f32(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, reference {w} (tol {tol})"
        );
    }
}

#[test]
fn florence2_text_core_matches_mlx_vlm_reference() {
    if !Path::new(MODEL_DIR).exists() {
        eprintln!("skipping florence2_text_parity: {MODEL_DIR} not present");
        return;
    }

    let model = Florence2TextModel::load(Path::new(MODEL_DIR)).expect("load florence2 text core");
    assert_eq!(model.config().decoder_attention_heads, 12);
    assert_eq!(model.config().decoder_start_token_id, 2);

    // shift_tokens_right must reproduce the pinned decoder input.
    let shifted = mlxcel::models::florence2::shift_tokens_right(
        INPUT_IDS,
        model.config().pad_token_id,
        model.config().decoder_start_token_id,
    );
    assert_eq!(shifted, DEC_IDS);

    // Teacher-forced full-sequence pass through encoder + decoder.
    let src = mlxcel_core::from_slice_i32(INPUT_IDS, &[1, INPUT_IDS.len() as i32]);
    let enc = model.encode_tokens(&src);
    let dec = mlxcel_core::from_slice_i32(DEC_IDS, &[1, DEC_IDS.len() as i32]);
    let mut cache = model.make_cache();
    let logits = model.decode(&dec, &enc, &mut cache);
    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(shape, vec![1, DEC_IDS.len() as i32, 51289]);

    // Per-position greedy argmax must match the reference exactly.
    let argmax = mlxcel_core::argmax(
        &mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32),
        -1,
        false,
    );
    mlxcel_core::eval(&argmax);
    let argmax_vals: Vec<i32> =
        mlxcel_core::array_to_raw_bytes(&mlxcel_core::astype(&argmax, mlxcel_core::dtype::INT32))
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
    assert_eq!(
        argmax_vals, REF_TF_ARGMAX,
        "teacher-forced argmax diverged from the mlx-vlm reference"
    );

    // Logit slices at the first and last positions, f16 execution on both
    // sides; tolerance covers op-ordering differences between the two
    // runtimes.
    let vocab = shape[2];
    let pos0 = mlxcel_core::slice(&logits, &[0, 0, 0], &[1, 1, 16]);
    assert_close(
        &to_vec_f32(&pos0),
        REF_LOGITS_POS0_FIRST16,
        0.1,
        "pos0 logits",
    );
    let last = mlxcel_core::slice(
        &logits,
        &[0, shape[1] - 1, 0],
        &[1, shape[1], vocab.min(16)],
    );
    assert_close(
        &to_vec_f32(&last),
        REF_LOGITS_LAST_FIRST16,
        0.1,
        "last-pos logits",
    );

    // Incremental greedy round trip through the dual KV cache must stop at
    // EOS after the same tokens the reference produces.
    let generated = model
        .generate_greedy(INPUT_IDS, 8)
        .expect("greedy generate");
    assert_eq!(
        generated, REF_GREEDY,
        "greedy decode diverged from the mlx-vlm reference"
    );
}
