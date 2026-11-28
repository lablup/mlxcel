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

//! PLaMo 3 against the HF reference with YaRN on (issue #2374).
//!
//! `STREAM` is a 256-token teacher-forced stream on
//! `pfnet/plamo-3-610m-fin-instruct`: the chat prompt for
//! "日本の歴史について簡単に説明してください。" followed by a fixed answer
//! passage. `HF_TOP1[i]` is the argmax the checkpoint's own
//! `modeling_plamo.py` (transformers 5.24.1, `trust_remote_code`, f32 on CPU,
//! YaRN factor 256 on the full layers) predicts after `STREAM[..=i]`.
//! mlxcel runs the bf16 weights, so a few near-tie positions may differ; the
//! test pins the agreement floor and prints the exact count.

use mlxcel::models::Plamo3Model;
use mlxcel_core::generate::LanguageModel;

const MODEL_DIR: &str = "models/plamo-3-610m-fin-instruct";

const STREAM: &[i32] = &[
    1, 16, 1667, 19, 50623, 51409, 50178, 81757, 50628, 16, 39306, 1335, 17, 13018, 18, 21279, 19,
    50623, 51409, 50116, 51480, 50642, 50879, 50816, 61474, 50101, 84521, 50816, 50101, 78329,
    50816, 50101, 79534, 50816, 53386, 50101, 83158, 56518, 51370, 51345, 52622, 50532, 53715,
    50105, 55201, 50715, 57051, 50816, 50136, 50663, 51035, 52951, 50105, 51254, 51495, 65253,
    51112, 50105, 71608, 50170, 81553, 50136, 70026, 50100, 51112, 50105, 52547, 50131, 50152,
    88258, 51801, 50163, 50186, 57844, 50427, 57741, 50170, 61629, 50816, 50136, 76091, 50105,
    53714, 50100, 51307, 52622, 50103, 66327, 50101, 80606, 50816, 50101, 81746, 53386, 50101,
    58394, 50136, 75286, 75526, 73067, 50248, 50185, 51116, 51828, 50246, 50121, 64315, 55561,
    50105, 51514, 50170, 53655, 70897, 51677, 50101, 50206, 50104, 70906, 64015, 50310, 60645,
    53716, 50185, 72460, 53532, 79572, 53715, 50150, 77641, 62734, 50170, 1228, 2076, 44, 10029,
    2706, 5590, 1097, 13638, 23467, 115, 6902, 31310, 28945, 48092, 5434, 33825, 3915, 1089, 25851,
    24515, 38295, 7302, 20584, 4000, 40382, 1475, 23134, 1091, 7087, 19028, 1591, 1108, 1593, 1413,
    4858, 5679, 22106, 48092, 1143, 4217, 7291, 4426, 2803, 1304, 28672, 4327, 1722, 6026, 353,
    32501, 2861, 1090, 32289, 48094, 21568, 17304, 4399, 111, 2807, 50125, 2530, 4553, 7456, 2594,
    44, 7610, 2082, 32, 68494, 48092, 4399, 111, 16575, 50115, 42285, 44, 8707, 48092, 29195,
    31267, 1096, 3138, 305, 56, 54, 56, 1087, 1699, 19677, 2861, 4104, 32969, 115, 44, 32974, 353,
    30557, 50115, 50119, 32, 4238, 6454, 101, 11080, 5827, 4004, 36751, 8087, 33688, 5786, 44,
    2989, 1781, 8146, 7188, 44, 23988, 44, 17856, 44, 24521, 48092, 7259, 24206, 3267,
];

const HF_TOP1: &[i32] = &[
    256, 1667, 19, 35, 51742, 52106, 57372, 50628, 16, 39306, 1335, 19, 21279, 18, 21279, 19, 256,
    51409, 51147, 58230, 50642, 50879, 63831, 61474, 50101, 78329, 50816, 50101, 78329, 50816,
    50101, 79534, 50816, 50101, 50101, 83158, 50816, 57401, 51345, 52622, 53715, 53715, 61506,
    55201, 50537, 52155, 50816, 50101, 57637, 51035, 52951, 50109, 52865, 51495, 81553, 50105,
    50105, 52547, 50170, 81553, 50136, 70026, 50771, 51112, 50105, 52547, 50131, 61629, 88258,
    51801, 50163, 50112, 57844, 57741, 57741, 50170, 61629, 50816, 50136, 52359, 50100, 84837,
    50100, 51648, 52622, 50103, 66327, 50101, 80606, 50816, 50136, 81746, 53386, 50101, 58394,
    50136, 72794, 75526, 50216, 54939, 51, 51116, 50121, 50246, 50645, 64315, 55561, 50123, 51514,
    50170, 53655, 70897, 50396, 50101, 64015, 50104, 70906, 64015, 50310, 60645, 50101, 50185,
    72460, 53386, 58247, 53715, 50150, 52547, 62734, 50101, 61182, 5827, 44, 5827, 2706, 28789,
    1296, 41793, 1752, 32861, 44, 8454, 28945, 4000, 41119, 33825, 32861, 44, 2637, 44, 46, 7302,
    46, 2706, 1127, 4624, 23134, 1091, 1117, 2637, 48092, 46, 1519, 1413, 2807, 4000, 1087, 48092,
    8875, 4217, 1168, 4399, 2803, 1304, 2807, 4327, 1722, 6026, 1087, 41119, 1142, 50133, 32289,
    2014, 1160, 46, 4399, 111, 2807, 44, 1386, 4843, 1214, 3983, 48092, 25218, 1168, 32, 75890,
    48092, 4399, 111, 1202, 50115, 3983, 48092, 35042, 50115, 29195, 31267, 1096, 46, 1699, 56, 54,
    56, 50125, 1699, 19677, 11411, 2637, 25851, 1224, 44, 15396, 18571, 6473, 50115, 5827, 1699,
    4238, 6454, 101, 11080, 5827, 1305, 7648, 33688, 4144, 5786, 44, 1091, 25436, 8146, 5536, 44,
    23988, 44, 25981, 48092, 25981, 48092, 1602, 24206, 46, 1214,
];

#[test]
fn plamo3_teacher_forced_top1_agrees_with_the_hf_reference() {
    if !std::path::Path::new(MODEL_DIR).exists() {
        eprintln!("skipping plamo3 HF parity: {MODEL_DIR} not present");
        return;
    }
    let (model, _args) = Plamo3Model::load(MODEL_DIR).expect("load plamo-3-610m-fin-instruct");
    let mut caches = LanguageModel::make_caches(&model);
    let input = mlxcel_core::from_slice_i32(STREAM, &[1, STREAM.len() as i32]);
    let logits = LanguageModel::forward(&model, &input, &mut caches, None);
    let argmax = mlxcel_core::argmax_last_axis(&logits);
    let argmax = mlxcel_core::astype(
        &mlxcel_core::reshape(&argmax, &[-1]),
        mlxcel_core::dtype::INT32,
    );
    mlxcel_core::eval(&argmax);
    let n = STREAM.len() as i32;
    let ours: Vec<i32> = (0..n)
        .map(|i| mlxcel_core::item_i32(&mlxcel_core::slice(&argmax, &[i], &[i + 1])))
        .collect();
    let agree = ours.iter().zip(HF_TOP1).filter(|(a, b)| a == b).count();
    eprintln!("plamo3 HF teacher-forced top-1 agreement: {agree}/{n}");
    assert!(
        agree * 100 >= STREAM.len() * 95,
        "top-1 agreement {agree}/{n} is below 95%"
    );
}
