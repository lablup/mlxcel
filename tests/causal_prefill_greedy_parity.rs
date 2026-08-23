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

//! Causal-prefill regression gates for the families fixed by issue #999.
//!
//! `internlm3`, `hunyuan_v1_dense`, `gemma2` and `hunyuan_moe` all handed the
//! caller's `None` mask straight to an unmasked SDPA on a multi-token prefill,
//! so every prompt position attended to every later prompt position, every
//! layer wrote future-contaminated K/V into the cache, and the first sampled
//! token was already wrong. Nothing in the tree generated from a real
//! checkpoint of any of them, so the defect passed every unit test and every CI
//! job.
//!
//! Each family gets up to two independent gates:
//!
//! * A greedy-parity gate pins the generated ids against `mlx_lm` 0.31.3 greedy
//!   decode (`temp = 0`) on the same prompt ids. Reference and mlxcel are fed
//!   the same raw id sequence, so tokenizer differences cannot enter. The
//!   `internlm3` reference is the one exception to "stock mlx-lm": see
//!   [`INTERNLM3_REF_OUT`], whose values were re-captured for issue #1324 from
//!   an mlx-lm whose InternLM3 rotary schedule was corrected first, because the
//!   stock one shares the defect that issue fixed.
//! * A causality gate pins the property the bug violated without depending on
//!   any reference ids: the first prompt position's logits must not move when
//!   later prompt tokens are added. It is portable across families and
//!   survives a checkpoint swap.
//!
//! Prompt length matters. For the LAST query position a causal and a
//! bidirectional mask select the same keys, so a short prompt can look
//! perfectly correct while the prefill is broken. Every prompt here is 36
//! tokens or longer.
//!
//! All gates skip when the checkpoint is absent (CI has no Metal and no
//! weights). The `hunyuan_moe` gate additionally requires
//! `MLXCEL_TEST_HUNYUAN_MOE=1`, because the only local checkpoint of that
//! family is a 42 GB MoE that should not load on every `cargo test` run.

use mlxcel::models::{Gemma2Model, HunyuanMoeModel, HunyuanV1DenseModel, InternLM3Model};
use mlxcel_core::generate::LanguageModel;

// Model directories.
const INTERNLM3_DIR: &str = "models/internlm3-8b-4bit";
const HUNYUAN_DENSE_DIR: &str = "models/hunyuan-1.8b-4bit";
const GEMMA2_DIR: &str = "models/gemma2-2b-4bit";
const HUNYUAN_MOE_DIR: &str = "models/hunyuan-a13b-instruct-4bit";

// Per-family bounds on the position-0 deviation between a single-token prefill
// and the full prompt.
//
// A causal prefill does not drive this to zero. Position 0 attends only to
// itself in both runs, but the two runs reach that answer through different
// kernels: the single-query maskless SDPA fast path versus the masked
// multi-query one, and a 1-row versus an N-row quantized matmul, compounded
// over every layer. Each bound therefore sits between a family's measured
// residual and what the same measurement produced on the pre-fix code, and the
// `causality_ladder` diagnostic printed alongside is what tells the two apart:
// contaminated attention grows with the number of later tokens, kernel-path
// noise stays flat. Measured on the checkpoints named below (residual at the
// full prompt, then the same figure before the fix):
//
// | family             | ladder                          | residual | pre-fix |
// |--------------------|---------------------------------|----------|---------|
// | `internlm3`        | 6.7e-4, 6.4e-4, 1.2e-3 (14/28/56) | 1.2e-3  | 1.02e0  |
// | `gemma2`           | 0, 5.0e-4, 5.2e-4 (11/22/44)      | 5.2e-4  | 1.72e-1 |
// | `hunyuan_v1_dense` | 0, 1.43e-2, 1.49e-2 (9/18/36)     | 1.49e-2 | 1.50e-1 |
// | `hunyuan_moe`      | 0, 2.25e-2, 2.12e-2 (10/20/41)    | 2.12e-2 | 6.44e-1 |
//
// The last rung of every ladder is flat or falling against the one before it,
// on both the dense and the MoE families, which is the shape a causal prefill
// has to produce and the broken one cannot.

/// `internlm3` and `gemma2`: residual 1.2e-3 and 5.2e-4, pre-fix 1.02e0 and
/// 1.72e-1. At least a 15x margin below and an 80x margin above.
const CAUSALITY_TOLERANCE: f64 = 2e-2;

/// `hunyuan_v1_dense`: residual 1.49e-2, pre-fix 1.50e-1. Its 4-bit stack
/// carries more kernel-path noise than `internlm3` or `gemma2` do, so 2e-2
/// would leave only a 1.3x margin below.
const CAUSALITY_TOLERANCE_HUNYUAN_DENSE: f64 = 5e-2;

/// `hunyuan_moe`: residual 2.12e-2, pre-fix 6.44e-1. A 41-row `gather_qmm`
/// through 64 experts at 4 bits, 32 layers deep, is the noisiest path here.
const CAUSALITY_TOLERANCE_MOE: f64 = 1e-1;

/// 56 ids: the shared 44-word prompt under the InternLM3 tokenizer, with the
/// leading BOS that tokenizer adds.
const INTERNLM3_INPUT_IDS: &[i32] = &[
    1, 1019, 30517, 19969, 1752, 20657, 293, 19649, 25461, 899, 4542, 272, 5808, 16846, 18992, 371,
    17012, 353, 23134, 10389, 6420, 21011, 29530, 14212, 353, 15683, 17999, 27980, 581, 20051, 282,
    1689, 3598, 5629, 27980, 2040, 71491, 81698, 394, 27980, 10590, 365, 27980, 353, 272, 10664,
    14757, 331, 17415, 2878, 3401, 27974, 5563, 5568, 20748, 1235,
];

/// Greedy (temp 0) continuation from mlx-lm 0.31.3 **with its InternLM3 rotary
/// schedule corrected**, which is the oracle issue #1324 re-pinned this
/// constant from:
///
/// "the growth of a new class of industrial capitalists who owned the means of
///  production and employed workers to produce goods"
///
/// # Why the reference ids changed
///
/// The previous values were captured from stock mlx-lm 0.31.3, and stock
/// mlx-lm carries the defect #1324 fixed. Its
/// [`internlm3.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/internlm3.py)
/// computes `rope_scale = 1 / factor if rope_type == "linear" else 2.0`, so the
/// `{"factor": 6.0, "rope_type": "dynamic"}` block this checkpoint ships gets a
/// position scale of `2.0`: every token is rotated as if it sat at twice its
/// position. mlxcel had ported that expression verbatim, so the two agreed
/// exactly, and this gate went green on 24 of 24 ids while pinning the bug.
///
/// The oracle the values now come from is the same mlx-lm, same weights, same
/// prompt ids, with only the per-layer rope module replaced by the schedule the
/// checkpoint's own remote code implements (`modeling_internlm3.py` ->
/// transformers `_compute_dynamic_ntk_parameters`): positions unscaled,
/// `seq_len` clamped up to `max_position_embeddings`, and only the base moving
/// past that. The ids were **not** taken from mlxcel's own output; doing that
/// would re-set the same trap, since a self-captured pin agrees with whatever
/// the code does.
///
/// The two schedules diverge at step 2 of 24 here, where the corrected oracle's
/// top-2 logit margin is 0.31, so the step that separates them is decided by
/// the model rather than by rounding.
const INTERNLM3_REF_OUT: &[i32] = &[
    272, 14753, 331, 269, 510, 530, 331, 26264, 10011, 2969, 3491, 16354, 272, 1971, 331, 8161,
    353, 24324, 8032, 27964, 303, 6850, 2645, 27964,
];

/// 36 ids: the Hunyuan chat template applied to a one-turn user question. The
/// raw-text form of the same question degenerates under the reference itself on
/// this instruction-tuned 4-bit export, so the templated form is the one that
/// demonstrates a coherent continuation as well as parity.
const HUNYUAN_DENSE_INPUT_IDS: &[i32] = &[
    120000, 120006, 937, 926, 3686, 14412, 11, 5470, 3892, 252, 27189, 15924, 6641, 280, 9280,
    12681, 4587, 1134, 280, 9391, 11, 9544, 11, 445, 252, 25918, 11, 289, 1917, 252, 3548, 1502,
    2470, 3975, 13, 120007,
];

// mlx-lm 0.31.3 greedy (temp 0) continuation:
// "<think>\nGot it, let's tackle this question. The user wants to know why the
//  Industrial Revolution"
const HUNYUAN_DENSE_REF_OUT: &[i32] = &[
    120029, 185, 72520, 423, 11, 2264, 665, 34176, 549, 2708, 13, 424, 2388, 9821, 287, 1771, 3892,
    252, 27189, 15924, 6641, 280, 9280, 12681,
];

/// 44 ids: the shared prompt under the Gemma 2 tokenizer, with its BOS.
const GEMMA2_INPUT_IDS: &[i32] = &[
    2, 651, 17494, 21373, 6343, 575, 6553, 14401, 2290, 573, 5245, 61650, 7861, 578, 20914, 8151,
    4492, 41527, 4238, 578, 4612, 5783, 235269, 1004, 199524, 20630, 235269, 15394, 235269, 16216,
    235269, 578, 573, 7074, 6672, 576, 14814, 3434, 1461, 235265, 3428, 2845, 22212, 729,
];

// mlx-lm 0.31.3 greedy (temp 0) continuation:
// " the rise of factories, which transformed the nature of work and labor. \n\n
//  Here's a breakdown of the key"
const GEMMA2_REF_OUT: &[i32] = &[
    573, 9407, 576, 41047, 235269, 948, 30003, 573, 4460, 576, 1160, 578, 6379, 235265, 235248,
    109, 4858, 235303, 235256, 476, 25497, 576, 573, 2621,
];

/// 40 ids: a plain-text prompt for the Hunyuan MoE causality gate. No reference
/// ids are pinned for this family, so only the length matters.
const HUNYUAN_MOE_INPUT_IDS: &[i32] = &[
    628, 27189, 15924, 6956, 280, 9280, 12681, 2222, 252, 7153, 61622, 5477, 289, 17602, 6767,
    4584, 35720, 4349, 289, 6292, 6427, 11, 54636, 36507, 16993, 11, 15301, 11, 14936, 11, 289,
    252, 7394, 7610, 279, 17158, 3822, 1496, 13, 4710, 2470,
];

fn argmax_last_token(logits: &mlxcel_core::MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let last = mlxcel_core::slice(logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
    let argmax = mlxcel_core::argmax_last_axis(&last);
    mlxcel_core::eval(&argmax);
    mlxcel_core::item_i32(&argmax)
}

/// The `[1, vocab]` logit row at prompt position `pos`, as host floats.
fn logit_row(logits: &mlxcel_core::MlxArray, pos: i32) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    let row = mlxcel_core::slice(logits, &[0, pos, 0], &[1, pos + 1, shape[2]]);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    mlxcel_core::array_to_raw_bytes(&row)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// RMS of `a - b` divided by the RMS of `b`.
fn relative_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "logit row length mismatch");
    let mut diff_sq = 0f64;
    let mut ref_sq = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        diff_sq += (x - y) * (x - y);
        ref_sq += y * y;
    }
    if ref_sq == 0.0 {
        return diff_sq.sqrt();
    }
    (diff_sq / ref_sq).sqrt()
}

fn skip_if_absent(dir: &str) -> bool {
    if std::path::Path::new(dir).exists() {
        return false;
    }
    eprintln!("skipping causal-prefill gate: {dir} not present");
    true
}

/// Greedy-decode `ref_out.len()` tokens from `input_ids` and compare id for id.
fn assert_greedy_parity<M: LanguageModel>(
    model: &M,
    input_ids: &[i32],
    ref_out: &[i32],
    who: &str,
) {
    let mut caches = LanguageModel::make_caches(model);
    let prompt = mlxcel_core::from_slice_i32(input_ids, &[1, input_ids.len() as i32]);
    let mut logits = LanguageModel::forward(model, &prompt, &mut caches, None);

    let mut out = Vec::with_capacity(ref_out.len());
    for _ in 0..ref_out.len() {
        let tok = argmax_last_token(&logits);
        out.push(tok);
        let next = mlxcel_core::from_slice_i32(&[tok], &[1, 1]);
        logits = LanguageModel::forward(model, &next, &mut caches, None);
    }

    assert_eq!(
        out, ref_out,
        "{who} greedy decode diverged from the mlx-lm reference"
    );
}

/// Position-0 logit deviation from the single-token baseline, at a ladder of
/// prefill lengths.
///
/// Under a causal prefill position 0 sees only itself at every length, so the
/// whole ladder sits flat at whatever the two kernel paths (single-query
/// maskless SDPA versus masked multi-query SDPA, and a 1-row versus an N-row
/// quantized matmul) disagree by. Under a bidirectional prefill position 0
/// absorbs every later token, so the ladder climbs with length. The shape is
/// what separates a real causality violation from numerical noise, and it is
/// printed so the tolerance above stays auditable rather than magic.
fn causality_ladder<M: LanguageModel>(model: &M, input_ids: &[i32]) -> Vec<(usize, f64)> {
    let mut caches_single = LanguageModel::make_caches(model);
    let first = mlxcel_core::from_slice_i32(&input_ids[..1], &[1, 1]);
    let single = LanguageModel::forward(model, &first, &mut caches_single, None);
    let row_single = logit_row(&single, 0);

    let n = input_ids.len();
    let mut lengths = vec![2, n / 4, n / 2, n];
    lengths.retain(|&l| l >= 2 && l <= n);
    lengths.sort_unstable();
    lengths.dedup();

    lengths
        .into_iter()
        .map(|len| {
            let mut caches = LanguageModel::make_caches(model);
            let prompt = mlxcel_core::from_slice_i32(&input_ids[..len], &[1, len as i32]);
            let logits = LanguageModel::forward(model, &prompt, &mut caches, None);
            (len, relative_rms(&logit_row(&logits, 0), &row_single))
        })
        .collect()
}

/// Assert that the position-0 logits do not move when later prompt tokens are
/// appended. Under a causal prefill position 0 sees only itself in both runs, so
/// the two rows agree to within the kernel-path noise. Under the bidirectional
/// prefill that shipped before issue #999, position 0 also attended to every
/// later token and the rows diverged by tens of percent.
fn assert_prefill_is_causal<M: LanguageModel>(
    model: &M,
    input_ids: &[i32],
    tolerance: f64,
    who: &str,
) {
    let ladder = causality_ladder(model, input_ids);
    let rendered: Vec<String> = ladder
        .iter()
        .map(|(len, dev)| format!("{len}:{dev:.3e}"))
        .collect();
    eprintln!(
        "{who} position-0 deviation by prefill length: {}",
        rendered.join(" ")
    );

    let (len, deviation) = *ladder.last().expect("ladder is never empty");
    let extra = len - 1;
    assert!(
        deviation < tolerance,
        "{who} prefill is not causal: position-0 logits moved by {deviation:.4e} \
         relative RMS when {extra} later prompt tokens were added"
    );
}

#[test]
fn internlm3_greedy_parity_matches_mlx_lm() {
    if skip_if_absent(INTERNLM3_DIR) {
        return;
    }
    let (model, _args) = InternLM3Model::load(INTERNLM3_DIR).expect("load internlm3");
    assert_greedy_parity(&model, INTERNLM3_INPUT_IDS, INTERNLM3_REF_OUT, "InternLM3");
}

#[test]
fn internlm3_prefill_is_causal() {
    if skip_if_absent(INTERNLM3_DIR) {
        return;
    }
    let (model, _args) = InternLM3Model::load(INTERNLM3_DIR).expect("load internlm3");
    assert_prefill_is_causal(
        &model,
        INTERNLM3_INPUT_IDS,
        CAUSALITY_TOLERANCE,
        "InternLM3",
    );
}

#[test]
fn hunyuan_v1_dense_greedy_parity_matches_mlx_lm() {
    if skip_if_absent(HUNYUAN_DENSE_DIR) {
        return;
    }
    let (model, _args) =
        HunyuanV1DenseModel::load(HUNYUAN_DENSE_DIR).expect("load hunyuan_v1_dense");
    assert_greedy_parity(
        &model,
        HUNYUAN_DENSE_INPUT_IDS,
        HUNYUAN_DENSE_REF_OUT,
        "Hunyuan V1 dense",
    );
}

#[test]
fn hunyuan_v1_dense_prefill_is_causal() {
    if skip_if_absent(HUNYUAN_DENSE_DIR) {
        return;
    }
    let (model, _args) =
        HunyuanV1DenseModel::load(HUNYUAN_DENSE_DIR).expect("load hunyuan_v1_dense");
    assert_prefill_is_causal(
        &model,
        HUNYUAN_DENSE_INPUT_IDS,
        CAUSALITY_TOLERANCE_HUNYUAN_DENSE,
        "Hunyuan V1 dense",
    );
}

#[test]
fn gemma2_greedy_parity_matches_mlx_lm() {
    if skip_if_absent(GEMMA2_DIR) {
        return;
    }
    let (model, _args) = Gemma2Model::load(GEMMA2_DIR).expect("load gemma2");
    assert_greedy_parity(&model, GEMMA2_INPUT_IDS, GEMMA2_REF_OUT, "Gemma 2");
}

/// Gemma 2 is the one family here whose null mask reached `compiled_softcap_sdpa`
/// rather than the plain SDPA, because it sets `attn_logit_softcapping`. The
/// softcap composite applies no causality of its own, so this gate also covers
/// the softcap argument being threaded into the causal call rather than bypassed.
#[test]
fn gemma2_prefill_is_causal() {
    if skip_if_absent(GEMMA2_DIR) {
        return;
    }
    let (model, _args) = Gemma2Model::load(GEMMA2_DIR).expect("load gemma2");
    assert_prefill_is_causal(&model, GEMMA2_INPUT_IDS, CAUSALITY_TOLERANCE, "Gemma 2");
}

/// Opt-in: the only local `hunyuan_moe` checkpoint is a 42 GB MoE, so this does
/// not run on every `cargo test`. Enable with `MLXCEL_TEST_HUNYUAN_MOE=1`.
///
/// No reference ids are pinned: no `hunyuan_moe` checkpoint that both fits the
/// development machine and loads under the mlx-lm reference harness is
/// available, so this family is gated on the reference-free causality property
/// only.
#[test]
fn hunyuan_moe_prefill_is_causal() {
    if !matches!(
        std::env::var("MLXCEL_TEST_HUNYUAN_MOE").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    ) {
        eprintln!("skipping hunyuan_moe causality gate: set MLXCEL_TEST_HUNYUAN_MOE=1 to run");
        return;
    }
    if skip_if_absent(HUNYUAN_MOE_DIR) {
        return;
    }
    let (model, _args) = HunyuanMoeModel::load(HUNYUAN_MOE_DIR).expect("load hunyuan_moe");
    assert_prefill_is_causal(
        &model,
        HUNYUAN_MOE_INPUT_IDS,
        CAUSALITY_TOLERANCE_MOE,
        "Hunyuan MoE",
    );
}
