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

//! PLaMo 3 against mlx-lm with YaRN forced off (issue #2374).
//!
//! mlx-lm's `plamo3.py` ignores the checkpoint's `rope_scaling_factor`, so the
//! comparison installs the `--rope-scaling none` override first; it is a
//! process-wide setting, which is why this test has its own binary. The ids
//! are `mlx_lm.generate_step` greedy output (mlx-lm 0.32.2, mlx 0.32.5 CUDA,
//! `trust_remote_code`, eos ignored) on `pfnet/plamo-3-610m-fin-instruct` for
//! a Japanese and an English chat prompt built by the checkpoint's template.

use mlxcel::models::Plamo3Model;
use mlxcel::models::rope_overrides::{self, RopeRuntimeOverride};
use mlxcel_core::generate::LanguageModel;

const MODEL_DIR: &str = "models/plamo-3-610m-fin-instruct";

/// "<|plamo:begin_think:plamo|>まず、ユーザーの質問は「日本の首都はどこですか？」です。ユーザーは"...
const JA_PROMPT: &[i32] = &[
    1, 16, 1667, 19, 50623, 60134, 56881, 50698, 16, 39306, 1335, 17, 13018, 18, 21279, 19,
];

const JA_GREEDY: &[i32] = &[
    256, 21279, 258, 52519, 51174, 50100, 50786, 50215, 50623, 60134, 56881, 50698, 51203, 51174,
    50104, 89647, 50106, 50786, 57298, 89647, 50106, 51613, 50127, 52861, 519, 50623, 60134, 50104,
    50349, 50111, 50349, 50104, 50724, 50100, 51126, 50705, 52256, 50126, 50724, 55347, 50105,
    53976, 50157, 68792, 50182, 50724, 50100, 52891, 50112, 51742, 50100, 77759, 50150, 51224,
    52376, 519, 51613, 50104, 89647, 50192, 53104, 55627, 54045, 52861,
];

/// "<|plamo:begin_think:plamo|>The user asks: \"What is the capit"...
const EN_PROMPT: &[i32] = &[
    1, 16, 1667, 19, 20393, 4146, 1088, 5343, 63, 24607, 13045, 46, 16, 39306, 1335, 17, 13018, 18,
    21279, 19,
];

const EN_GREEDY: &[i32] = &[
    256, 21279, 258, 1100, 1867, 13983, 58, 290, 20393, 4146, 1088, 5343, 4215, 7452, 1629, 1087,
    4146, 1088, 5343, 46, 26580, 3453, 46, 26580, 2076, 3453, 46, 519, 1286, 1444, 3453, 58, 4944,
    46, 26580, 10994, 46, 519, 1286, 1792, 3285, 1116, 1091, 2970, 2661, 1646, 24559, 1771, 46,
    26580, 3453, 46, 519, 13811, 1516, 3453, 46, 257, 21279, 258, 27581, 50176, 4146, 1088,
];

fn greedy(model: &Plamo3Model, prompt: &[i32], steps: usize) -> Vec<i32> {
    let mut caches = LanguageModel::make_caches(model);
    let input = mlxcel_core::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let mut logits = LanguageModel::forward(model, &input, &mut caches, None);
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let shape = mlxcel_core::array_shape(&logits);
        let last = mlxcel_core::slice(&logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
        let next = mlxcel_core::argmax_last_axis(&last);
        mlxcel_core::eval(&next);
        let tok = mlxcel_core::item_i32(&next);
        out.push(tok);
        let step = mlxcel_core::from_slice_i32(&[tok], &[1, 1]);
        logits = LanguageModel::forward(model, &step, &mut caches, None);
    }
    out
}

#[test]
fn plamo3_yarn_off_greedy_matches_mlx_lm() {
    if !std::path::Path::new(MODEL_DIR).exists() {
        eprintln!("skipping plamo3 mlx-lm parity: {MODEL_DIR} not present");
        return;
    }
    let none = RopeRuntimeOverride::from_flags(Some("none"), None, None, None)
        .expect("--rope-scaling none parses");
    rope_overrides::install(none).expect("install the override once");
    let (model, _args) = Plamo3Model::load(MODEL_DIR).expect("load plamo-3-610m-fin-instruct");
    for (name, prompt, expected) in [("ja", JA_PROMPT, JA_GREEDY), ("en", EN_PROMPT, EN_GREEDY)] {
        let got = greedy(&model, prompt, expected.len());
        assert_eq!(got, expected, "{name}: greedy decode diverged from mlx-lm");
    }
}
