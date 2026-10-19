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

//! Viterbi on synthetic emissions: illegal paths are repaired, each of the
//! six biases steers its own transition, ties resolve as upstream, and a
//! random lattice decodes to the path `mlx_vlm.privacy_filter.ViterbiDecoder`
//! produced (fixture `privacy_filter_mlx_vlm.json`, `viterbi_synthetic`).

use serde_json::{Value, json};

use super::*;
use crate::token_classification::argmax_path;

const O: usize = 0;
const B_X: usize = 1;
const I_X: usize = 2;
const E_X: usize = 3;
const S_X: usize = 4;
const B_Y: usize = 5;
const I_Y: usize = 6;
const E_Y: usize = 7;
const S_Y: usize = 8;

fn labels() -> Vec<String> {
    ["O", "B-x", "I-x", "E-x", "S-x", "B-y", "I-y", "E-y", "S-y"]
        .map(str::to_string)
        .to_vec()
}

fn decoder(biases: TransitionBiases) -> ViterbiDecoder {
    ViterbiDecoder::new(&labels(), biases).expect("valid BIOES set")
}

/// One row with `LOW` everywhere except the listed `(label, score)` pairs.
fn row(pairs: &[(usize, f32)]) -> Vec<f32> {
    let mut row = vec![-50.0; 9];
    for &(label, score) in pairs {
        row[label] = score;
    }
    row
}

fn is_legal(path: &[usize]) -> bool {
    let kind = |label: usize| -> (char, usize) {
        match label {
            O => ('O', 0),
            B_X | B_Y => ('B', if label == B_X { 1 } else { 2 }),
            I_X | I_Y => ('I', if label == I_X { 1 } else { 2 }),
            E_X | E_Y => ('E', if label == E_X { 1 } else { 2 }),
            _ => ('S', if label == S_X { 1 } else { 2 }),
        }
    };
    let mut open: Option<usize> = None;
    for &label in path {
        let (boundary, entity) = kind(label);
        match (boundary, open) {
            ('O' | 'B' | 'S', None) => {}
            ('I' | 'E', Some(current)) if current == entity => {}
            _ => return false,
        }
        open = match boundary {
            'B' | 'I' => Some(entity),
            _ => None,
        };
    }
    open.is_none()
}

#[test]
fn an_illegal_argmax_path_is_repaired() {
    // argmax reads O, I-x, I-x, O: an inside tag straight after background.
    let emissions = vec![
        row(&[(O, 0.0)]),
        row(&[(I_X, 0.0), (B_X, -1.0), (S_X, -3.0), (O, -4.0)]),
        row(&[(I_X, 0.0), (E_X, -1.0), (O, -4.0)]),
        row(&[(O, 0.0)]),
    ];
    let argmax = argmax_path(&emissions);
    assert_eq!(argmax, [O, I_X, I_X, O]);
    assert!(!is_legal(&argmax));
    let path = decoder(TransitionBiases::default()).decode(&emissions);
    assert_eq!(path, [O, B_X, E_X, O]);
    assert!(is_legal(&path));
}

#[test]
fn background_stay_keeps_background_over_a_slightly_better_single() {
    let emissions = vec![row(&[(O, 0.0)]), row(&[(S_X, 0.0), (O, -0.5)])];
    assert_eq!(
        decoder(TransitionBiases::default()).decode(&emissions),
        [O, S_X]
    );
    let biased = TransitionBiases {
        background_stay: 1.0,
        ..TransitionBiases::default()
    };
    assert_eq!(decoder(biased).decode(&emissions), [O, O]);
}

#[test]
fn background_to_start_opens_an_entity_over_a_slightly_better_background() {
    let emissions = vec![row(&[(O, 0.0)]), row(&[(O, 0.0), (S_X, -0.5)])];
    assert_eq!(
        decoder(TransitionBiases::default()).decode(&emissions),
        [O, O]
    );
    let biased = TransitionBiases {
        background_to_start: 1.0,
        ..TransitionBiases::default()
    };
    assert_eq!(decoder(biased).decode(&emissions), [O, S_X]);
}

#[test]
fn inside_to_continue_extends_the_entity() {
    // B-x, then a near tie between continuing (I-x) and closing (E-x) that a
    // following E-x / O pair leaves open either way.
    let emissions = vec![
        row(&[(B_X, 0.0)]),
        row(&[(E_X, 0.0), (I_X, -0.5)]),
        row(&[(O, 0.0), (E_X, 0.0)]),
    ];
    assert_eq!(
        decoder(TransitionBiases::default()).decode(&emissions),
        [B_X, E_X, O]
    );
    let biased = TransitionBiases {
        inside_to_continue: 1.0,
        ..TransitionBiases::default()
    };
    assert_eq!(decoder(biased).decode(&emissions), [B_X, I_X, E_X]);
}

#[test]
fn inside_to_end_closes_an_entity_over_staying_in_background() {
    // Every path with a multi-token entity pays `inside_to_end` exactly once,
    // so the bias is measured against the entity-free path.
    let emissions = vec![row(&[(O, 0.0), (B_X, -0.25)]), row(&[(O, 0.0), (E_X, 0.0)])];
    assert_eq!(
        decoder(TransitionBiases::default()).decode(&emissions),
        [O, O]
    );
    let biased = TransitionBiases {
        inside_to_end: 1.0,
        ..TransitionBiases::default()
    };
    assert_eq!(decoder(biased).decode(&emissions), [B_X, E_X]);
}

#[test]
fn end_to_background_returns_to_background_after_a_closed_entity() {
    let emissions = vec![row(&[(S_X, 0.0)]), row(&[(S_Y, 0.0), (O, -0.5)])];
    assert_eq!(
        decoder(TransitionBiases::default()).decode(&emissions),
        [S_X, S_Y]
    );
    let biased = TransitionBiases {
        end_to_background: 1.0,
        ..TransitionBiases::default()
    };
    assert_eq!(decoder(biased).decode(&emissions), [S_X, O]);
}

#[test]
fn end_to_start_chains_entities() {
    let emissions = vec![
        row(&[(S_X, 0.0)]),
        row(&[(O, 0.0), (B_Y, -0.5)]),
        row(&[(E_Y, 0.0), (O, 0.0)]),
    ];
    assert_eq!(
        decoder(TransitionBiases::default()).decode(&emissions),
        [S_X, O, O]
    );
    let biased = TransitionBiases {
        end_to_start: 1.0,
        ..TransitionBiases::default()
    };
    assert_eq!(decoder(biased).decode(&emissions), [S_X, B_Y, E_Y]);
}

#[test]
fn ties_resolve_as_upstream() {
    let plain = decoder(TransitionBiases::default());
    // Background: staying ties with closing; staying wins (`>=`).
    assert_eq!(
        plain.decode(&[row(&[(O, 0.0), (S_X, 0.0)]), row(&[(O, 0.0)])]),
        [O, O]
    );
    // Start: opening from background ties with reopening after a closed
    // entity; background wins.
    assert_eq!(
        plain.decode(&[row(&[(O, 0.0), (S_X, 0.0)]), row(&[(S_Y, 0.0)])]),
        [O, S_Y]
    );
    // Open: B ties with I; B wins (`I > B` only).
    assert_eq!(
        plain.decode(&[
            row(&[(O, 0.0), (B_X, 0.0)]),
            row(&[(I_X, 0.0), (B_X, 0.0)]),
            row(&[(E_X, 0.0)]),
        ]),
        [O, B_X, E_X]
    );
    // Closed states tie: the first in label order wins (S-x before S-y).
    assert_eq!(
        plain.decode(&[row(&[(S_X, 0.0), (S_Y, 0.0)]), row(&[(O, 0.0)])]),
        [S_X, O]
    );
    // End states tie: O comes first.
    assert_eq!(plain.decode(&[row(&[(O, 0.0), (S_X, 0.0)])]), [O]);
}

#[test]
fn a_single_token_is_background_or_single() {
    let plain = decoder(TransitionBiases::default());
    // B-x and I-x score higher, but neither can end a path.
    assert_eq!(
        plain.decode(&[row(&[(B_X, 0.0), (I_X, -1.0), (S_X, -2.0), (O, -3.0)])]),
        [S_X]
    );
    assert_eq!(plain.decode(&[]), Vec::<usize>::new());
}

#[test]
fn every_entity_needs_all_four_boundaries() {
    let labels: Vec<String> = ["O", "B-x", "I-x", "E-x", "S-x", "B-y", "I-y"]
        .map(str::to_string)
        .to_vec();
    let error = ViterbiDecoder::new(&labels, TransitionBiases::default())
        .err()
        .expect("missing E-y / S-y is refused")
        .to_string();
    assert_eq!(
        error,
        "Privacy Filter labels for 'y' are missing [\"E\", \"S\"]"
    );
    // A BIO set without the background label is refused by `build_label_info`.
    assert!(ViterbiDecoder::new(&["B-x".to_string()], TransitionBiases::default()).is_err());
}

#[test]
fn transition_biases_read_the_six_keys() {
    let full = json!({
        "transition_bias_background_stay": 0.5,
        "transition_bias_background_to_start": -0.25,
        "transition_bias_inside_to_continue": 1,
        "transition_bias_inside_to_end": 0,
        "transition_bias_end_to_background": 2.5,
        "transition_bias_end_to_start": -1.5,
        "something_else": 9,
    });
    assert_eq!(
        TransitionBiases::from_json(&full).unwrap(),
        TransitionBiases {
            background_stay: 0.5,
            background_to_start: -0.25,
            inside_to_continue: 1.0,
            inside_to_end: 0.0,
            end_to_background: 2.5,
            end_to_start: -1.5,
        }
    );
    let mut partial = full.clone();
    let object = partial.as_object_mut().unwrap();
    object.remove("transition_bias_inside_to_end");
    object.remove("transition_bias_background_stay");
    assert_eq!(
        TransitionBiases::from_json(&partial)
            .unwrap_err()
            .to_string(),
        "Viterbi calibration is missing biases: [\"transition_bias_background_stay\", \
         \"transition_bias_inside_to_end\"]"
    );
    let mut text = full.clone();
    text["transition_bias_end_to_start"] = json!("x");
    assert!(TransitionBiases::from_json(&text).is_err());
    // A value past the f32 range would turn into infinity and NaN the
    // lattice; it is refused instead.
    let mut huge = full.clone();
    huge["transition_bias_background_stay"] = json!(1e300);
    let error = TransitionBiases::from_json(&huge).unwrap_err().to_string();
    assert!(error.contains("must be finite"), "{error}");
    assert!(TransitionBiases::from_json(&json!([])).is_err());
}

/// The random lattice upstream decoded, under zero and under non-zero biases.
#[test]
fn a_random_lattice_decodes_like_mlx_vlm() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/privacy_filter_mlx_vlm.json"
    ))
    .expect("fixture parses");
    let synthetic = &fixture["viterbi_synthetic"];
    let labels: Vec<String> = serde_json::from_value(synthetic["labels"].clone()).unwrap();
    let emissions: Vec<Vec<f32>> = serde_json::from_value(synthetic["emissions"].clone()).unwrap();
    let cases = synthetic["cases"].as_array().expect("cases");
    assert!(cases.len() >= 2);
    for case in cases {
        let biases = TransitionBiases::from_json(&case["biases"]).unwrap();
        let want: Vec<usize> = serde_json::from_value(case["path"].clone()).unwrap();
        let got = ViterbiDecoder::new(&labels, biases)
            .unwrap()
            .decode(&emissions);
        assert_eq!(got, want, "biases {}", case["biases"]);
        assert!(is_legal(&got) || labels.len() != 9);
    }
}
