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

//! IQuest-Coder Loop greedy parity against an independent float32 oracle.
//!
//! The reference is a streaming NumPy implementation of the two-pass loop,
//! written from `modeling_iquestloopcoder.py` (the modeling file the checkpoint
//! itself ships) and cross-checked against that file directly: a tiny
//! random-weight fp32 torch model built from the vendor classes agrees with the
//! oracle on 16/16 shape cases spanning `L <= window`, `window < L <= 2*window`
//! and `L > 2*window`, to a max abs difference of 1.1e-06. The oracle
//! dequantizes the 4-bit affine weights itself, one layer at a time, so it
//! shares no code with mlxcel.
//!
//! # What each prompt is for
//!
//! `SHORT_CHAT_IDS` is 38 tokens, shorter than `loop_window_size` (64), so its
//! pass-2 local branch drops nothing. `LONG_CHAT_IDS` is 218 tokens, past
//! `2 * loop_window_size`, so the local branch genuinely windows: rewriting a
//! key more than 63 positions back cannot reach the final query row.
//!
//! # What this comparison can and cannot settle
//!
//! It settles the structure: the two passes, the per-head gate, pass 2 reading
//! pass 1's K/V, the shared RoPE positions, the 4-bit affine weight load, and
//! the prefill-then-decode cache handover. Measured against the oracle at the
//! last position, mlxcel agrees on the argmax for both prompts, reproduces the
//! oracle's greedy continuation for 12 tokens exactly, and lands at
//! `KL(oracle || mlxcel) = 0.0047` on the 218-token prompt (0.029 on the
//! 38-token one), with logit correlation 0.9980.
//!
//! It does **not** settle the window, and it cannot on this checkpoint. Every
//! layer's `gate_projections.{i}.bias` is +2.0 and the gate weights are small,
//! so the gate sits at ~0.88 in all 80 layers (per-layer min 0.8746, max
//! 0.8843, mean 0.8796) and the local branch carries only ~12% of the mixed
//! output. Re-running the oracle with the window removed moves the 218-token
//! logits by mean 5.0e-02 / max 3.4e-01 and changes neither the argmax nor the
//! top-10 order; at 521 tokens the gap shrinks further, to mean 8.9e-03. mlxcel
//! executes in f16 with a quantized `lm_head`, which puts its own deviation
//! from the f32 oracle at mean 2.7e-01 after removing the constant offset, 5.3x
//! the entire window effect. Measured directly: mlxcel sits 0.2674 from the
//! windowed oracle and 0.2678 from the unwindowed one, which is no separation
//! at all.
//!
//! So the window is pinned where it can be isolated instead of diluted:
//! `window_limits_local_branch` and `pass2_matches_only_the_correct_reference`
//! in `src/models/iquestloopcoder_tests.rs` compare the local branch's own
//! output rather than the final logits.
//!
//! Skips when the checkpoint is absent.

use mlxcel::models::IQuestLoopCoderModel;

const MODEL_DIR: &str = "models/mlx/iquest-coder-v1-40b-loop-instruct-4bit";

/// Chat-templated "Write a Python function to reverse a string.", 38 tokens.
const SHORT_CHAT_IDS: &[i32] = &[
    75863, 1505, 13, 6826, 490, 32049, 66605, 4476, 66560, 266, 8879, 20062, 6711, 528, 354, 46727,
    66576, 75864, 66542, 13, 75863, 1449, 13, 4719, 266, 6706, 773, 317, 13312, 266, 1078, 66576,
    75864, 66542, 13, 75863, 20062, 13,
];

/// Oracle last-position top-5 for `SHORT_CHAT_IDS`, `(id, logit)` in order.
const SHORT_CHAT_TOP5: &[(i32, f32)] = &[
    (66644, 37.591_75),
    (32349, 35.088_7),
    (66614, 34.305_7),
    (13162, 32.728_01),
    (1238, 30.638_454),
];

/// Chat-templated 218-token request about a sliding-window deduplicator. Past
/// `2 * loop_window_size`, so the pass-2 local branch is genuinely windowed.
const LONG_CHAT_IDS: &[i32] = &[
    75863, 1505, 13, 6826, 490, 32049, 66605, 4476, 66560, 266, 8879, 20062, 6711, 528, 354, 46727,
    66576, 75864, 66542, 13, 75863, 1449, 13, 66614, 1035, 4088, 266, 2226, 8328, 2109, 298, 6706,
    324, 354, 994, 1308, 409, 722, 3901, 66576, 477, 2109, 8507, 15401, 5859, 3980, 472, 266, 1993,
    9373, 66560, 60765, 989, 2666, 266, 9396, 66560, 7702, 56243, 989, 528, 1886, 960, 2547, 266,
    46252, 722, 7803, 4207, 66560, 324, 1133, 18516, 267, 43949, 317, 54521, 298, 52346, 314, 652,
    317, 4517, 10765, 9282, 66576, 12562, 1407, 267, 7702, 426, 2798, 6648, 345, 266, 12866, 14306,
    383, 30148, 2032, 5408, 66560, 712, 267, 1533, 7617, 662, 314, 5025, 1256, 814, 266, 1791, 314,
    467, 2149, 20761, 66576, 4864, 3156, 266, 549, 2591, 10588, 6771, 5368, 66621, 276, 8601, 862,
    383, 15714, 267, 5025, 53863, 33573, 66560, 441, 10117, 266, 4123, 66586, 2799, 66561, 305,
    66560, 13638, 66546, 1564, 15152, 4028, 837, 267, 960, 503, 2838, 10865, 4959, 267, 4207,
    66560, 6760, 56168, 36346, 11510, 1461, 29032, 66558, 4631, 1124, 409, 266, 2912, 5314, 66560,
    324, 345, 6453, 317, 1554, 472, 3237, 53333, 8880, 298, 267, 1562, 1886, 7525, 66576, 21982,
    997, 40805, 66560, 266, 2758, 4316, 1130, 369, 1492, 694, 1564, 66560, 324, 10821, 267, 811,
    17261, 314, 1406, 5911, 66576, 75864, 66542, 13, 75863, 20062, 13,
];

/// Oracle last-position top-5 for `LONG_CHAT_IDS`, `(id, logit)` in order.
///
/// Ranks 2 and 3 are separated by 0.0395 in the oracle's f32, which is below
/// the 0.125 spacing mlxcel's quantized `lm_head` resolves at this magnitude,
/// so their relative order is not a meaningful assertion here. See
/// [`TIE_GAP`].
const LONG_CHAT_TOP5: &[(i32, f32)] = &[
    (5123, 27.333_853),
    (32349, 27.166_67),
    (1545, 27.127_19),
    (1238, 26.227_745),
    (66628, 25.777_222),
];

/// Oracle greedy continuation from `SHORT_CHAT_IDS`, decoding to
/// `# Python Function to Reverse a String\n\nHere are several`.
const SHORT_CHAT_GREEDY: &[i32] = &[
    66644, 6706, 7348, 317, 47728, 266, 1139, 13, 13, 32349, 490, 3237,
];

/// How far a logit may sit from the oracle.
///
/// mlxcel runs 160 layer applications in f16 and projects through a quantized
/// `lm_head`; the oracle is f32 throughout. Across the two prompts the largest
/// observed top-5 deviation is 1.83, almost all of it a systematic shift shared
/// by the whole row (the fitted relation is `mlxcel = 1.095 * oracle - 0.155`
/// at 218 tokens), which softmax is insensitive to. 2.5 is that envelope with
/// room to spare.
///
/// This is the weakest of the assertions in this file and is not what carries
/// it: `iquestloopcoder_greedy_continuation_matches_the_oracle` is exact.
const LOGIT_TOLERANCE: f32 = 2.5;

/// Oracle logit gap below which two entries are treated as tied.
///
/// mlxcel's quantized `lm_head` resolves about 0.125 between adjacent logits at
/// magnitude 16 to 32, so any oracle pair closer than that has no defined order
/// here. 0.5 is four times that spacing.
const TIE_GAP: f32 = 0.5;

fn checkpoint_present() -> bool {
    std::path::Path::new(MODEL_DIR).join("config.json").exists()
}

fn top_k(logits: &mlxcel_core::MlxArray, k: usize) -> Vec<(i32, f32)> {
    let values = mlxcel_core::utils::array_to_vec_f32(logits);
    let mut indexed: Vec<(i32, f32)> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (i as i32, *v))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);
    indexed
}

/// Write the whole last-position logit row as little-endian f32, for the
/// off-line element-wise comparison described in the module docs.
fn dump_logits(label: &str, logits: &mlxcel_core::MlxArray) {
    let Ok(dir) = std::env::var("MLXCEL_IQUESTLOOP_LOGIT_DUMP") else {
        return;
    };
    let values = mlxcel_core::utils::array_to_vec_f32(logits);
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let path = std::path::Path::new(&dir).join(format!("mlxcel_{label}.f32"));
    std::fs::write(&path, bytes).expect("write logit dump");
    eprintln!("dumped {} logits to {}", values.len(), path.display());
}

fn last_logits(model: &IQuestLoopCoderModel, ids: &[i32]) -> Vec<(i32, f32)> {
    let mut caches = model.make_caches();
    let prompt = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
    let logits = model.forward_last_logits_with_caches(&prompt, &mut caches, ids.len() - 1);
    dump_logits(&format!("len{}", ids.len()), &logits);
    top_k(&logits, 5)
}

fn assert_matches_oracle(label: &str, got: &[(i32, f32)], expected: &[(i32, f32)]) {
    for ((id, value), (ref_id, reference)) in got.iter().zip(expected) {
        eprintln!(
            "{label}: got ({id}, {value}) oracle ({ref_id}, {reference}) delta {}",
            value - reference
        );
    }

    assert_eq!(
        got[0].0, expected[0].0,
        "{label}: argmax must match the oracle; got {got:?}"
    );

    let mut got_ids: Vec<i32> = got.iter().map(|(id, _)| *id).collect();
    let mut expected_ids: Vec<i32> = expected.iter().map(|(id, _)| *id).collect();
    got_ids.sort_unstable();
    expected_ids.sort_unstable();
    assert_eq!(
        got_ids, expected_ids,
        "{label}: the top-5 token set must match the oracle; got {got:?}"
    );

    // Order is asserted only between entries the oracle itself separates by
    // more than mlxcel can resolve. Demanding a total order would be asserting
    // noise.
    let rank_of = |id: i32| got.iter().position(|(g, _)| *g == id).expect("in top-5");
    for (i, (id_i, logit_i)) in expected.iter().enumerate() {
        for (id_j, logit_j) in &expected[i + 1..] {
            if logit_i - logit_j > TIE_GAP {
                assert!(
                    rank_of(*id_i) < rank_of(*id_j),
                    "{label}: oracle separates {id_i} from {id_j} by {} but mlxcel ranks them \
                     the other way; got {got:?}",
                    logit_i - logit_j
                );
            }
        }
    }

    for (id, value) in got {
        let reference = expected
            .iter()
            .find(|(e, _)| e == id)
            .map(|(_, v)| *v)
            .expect("id is in the oracle top-5");
        assert!(
            (value - reference).abs() <= LOGIT_TOLERANCE,
            "{label}: logit for token {id} is {value}, oracle says {reference} \
             (tolerance {LOGIT_TOLERANCE})"
        );
    }
}

/// The oracle's greedy continuation from `LONG_CHAT_IDS`, with the oracle's own
/// top-1 against top-2 margin at each step.
///
/// Produced by re-prefilling the whole growing sequence each step, so it carries
/// no cache state of its own and is a clean reference for a cached decoder.
/// Decoded, the seven tokens read `This implementation uses a `colle`.
const LONG_CHAT_GREEDY: &[(i32, f32)] = &[
    (5123, 0.167),
    (6275, 2.358),
    (4774, 3.096),
    (266, 0.706),
    (684, 0.719),
    (873, 0.271),
    (43409, 13.007),
];

/// Below this oracle margin, which token wins is not something mlxcel can be
/// held to: its quantized `lm_head` resolves about 0.125 at these magnitudes and
/// its deviation from the f32 oracle is mean 0.27. Steps closer than this are
/// still exercised, they just are not asserted against the oracle's choice.
const ORACLE_DECISIVE_MARGIN: f32 = 1.0;

/// Every check runs against one loaded model, inside one `#[test]`.
///
/// Not a style choice. The Rust harness runs `#[test]` functions in parallel by
/// default, and each of these checks needs the whole 40B checkpoint resident, so
/// five separate tests meant five simultaneous 21 GB models. On a 128 GB unified
/// -memory host that exhausts the GPU and wedges the driver rather than failing
/// cleanly. One load also cuts the wall clock by roughly the same factor.
///
/// Each check is a plain function so a failure still names the stage it came
/// from; the `eprintln!` banners separate them in `--nocapture` output.
#[test]
fn iquestloopcoder_matches_the_float32_oracle() {
    if !checkpoint_present() {
        eprintln!("skipping iquestloopcoder parity: {MODEL_DIR} not present");
        return;
    }

    let (model, args) = IQuestLoopCoderModel::load(MODEL_DIR).expect("load IQuest-Coder Loop");
    assert_eq!(args.loop_num, 2);
    assert_eq!(args.loop_window_size, 64);
    let window = args.loop_window_size as i32;

    eprintln!("== short prompt, 38 tokens, inside the window");
    check_prompt_matches_oracle(
        &model,
        "short chat (38 tokens, inside the window)",
        SHORT_CHAT_IDS,
        SHORT_CHAT_TOP5,
    );

    eprintln!("== long prompt, 218 tokens, past 2x the window");
    check_prompt_matches_oracle(
        &model,
        "long chat (218 tokens, past 2x the window)",
        LONG_CHAT_IDS,
        LONG_CHAT_TOP5,
    );

    eprintln!("== greedy continuation, inside the window");
    check_greedy_continuation(&model);

    eprintln!("== chunked prefill against single-pass");
    check_chunked_prefill(&model);

    eprintln!("== decode past the window against a full re-prefill");
    check_decode_past_the_window(&model, window);
}

fn check_prompt_matches_oracle(
    model: &IQuestLoopCoderModel,
    label: &str,
    ids: &[i32],
    expected: &[(i32, f32)],
) {
    assert_matches_oracle(label, &last_logits(model, ids), expected);
}

/// Incremental decode against the oracle's stateless re-prefill continuation,
/// from a prompt short enough that the pass-2 ring never fills its window.
fn check_greedy_continuation(model: &IQuestLoopCoderModel) {
    let mut caches = model.make_caches();
    let prompt = mlxcel_core::from_slice_i32(SHORT_CHAT_IDS, &[1, SHORT_CHAT_IDS.len() as i32]);
    let mut logits =
        model.forward_last_logits_with_caches(&prompt, &mut caches, SHORT_CHAT_IDS.len() - 1);

    let mut produced = Vec::with_capacity(SHORT_CHAT_GREEDY.len());
    for _ in 0..SHORT_CHAT_GREEDY.len() {
        let next = top_k(&logits, 1)[0].0;
        produced.push(next);
        let step = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        logits = model.forward_last_logits_with_caches(&step, &mut caches, 0);
    }

    assert_eq!(
        produced, SHORT_CHAT_GREEDY,
        "greedy continuation must match the oracle token for token"
    );
}

/// Chunked prefill must land where a single-pass prefill lands.
///
/// This is the regime the two caches disagree about most easily. A continuation
/// chunk finds the pass-1 cache holding every prior key and the pass-2 rotating
/// cache holding at most `loop_window_size - 1` of them, so
/// `RotatingKVCache::update_concat` returns `kept + L` keys while the pass-1
/// cache returns `S + L`, and the windowed mask
/// [`mlxcel_core::causal_attention`] builds has to match the first of those, not
/// the second. An off-by-one there windows the wrong keys, which stays fluent.
///
/// `prefill_matches_incremental_decode` in
/// `src/models/iquestloopcoder_tests.rs` covers the same seam on a synthetic
/// one-layer model; this pins it on the real 80-layer checkpoint, across chunk
/// sizes that land on, inside and across the 64-token window boundary.
fn check_chunked_prefill(model: &IQuestLoopCoderModel) {
    let chunked = |chunk: usize| -> Vec<(i32, f32)> {
        let mut caches = model.make_caches();
        let mut logits = None;
        let mut start = 0;
        while start < LONG_CHAT_IDS.len() {
            let end = (start + chunk).min(LONG_CHAT_IDS.len());
            let piece = &LONG_CHAT_IDS[start..end];
            let arr = mlxcel_core::from_slice_i32(piece, &[1, piece.len() as i32]);
            logits =
                Some(model.forward_last_logits_with_caches(&arr, &mut caches, piece.len() - 1));
            start = end;
        }
        top_k(logits.as_ref().expect("at least one chunk"), 5)
    };

    let single = chunked(LONG_CHAT_IDS.len());
    for chunk in [128, 97, 64, 32] {
        let split = chunked(chunk);
        assert_eq!(
            split.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            single.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            "chunk {chunk}: top-5 ids diverged from the single-pass prefill; got {split:?} \
             against {single:?}"
        );
        for ((id, value), (_, reference)) in split.iter().zip(&single) {
            assert!(
                (value - reference).abs() <= 0.25,
                "chunk {chunk}: logit for token {id} is {value}, single-pass says {reference}"
            );
        }
    }
}

/// Decode past the window: the rotating ring must agree with a full re-prefill,
/// and with the oracle wherever the oracle is decisive.
///
/// This is the one regime nothing else here covers. [`check_greedy_continuation`]
/// decodes from a 38-token prompt, where the pass-2 cache never fills its
/// 64-entry window, so the ring never wraps and never trims. Starting from 218
/// tokens every step is past the window: the ring holds exactly 64 keys and
/// drops one per token.
///
/// The load-bearing assertion is the self-consistency one. Incremental decode
/// reads a physically trimmed 64-key ring while a full re-prefill of the same
/// sequence keeps every key and windows them with a mask, so the two routes
/// reach the same answer through different code. It needs no reference and is
/// unaffected by mlxcel's f16 deviation from the f32 oracle, which is what makes
/// it able to catch a ring that trims the wrong end, wraps at the wrong point,
/// or loses the current token.
///
/// Tokens are teacher-forced onto the oracle's continuation so both routes and
/// the oracle see one sequence. Free-running, mlxcel follows the oracle for
/// three tokens and then flips step 4, where the oracle's own top three sit
/// within 0.744 of each other; forced back on, it reproduces the oracle's next
/// three exactly.
fn check_decode_past_the_window(model: &IQuestLoopCoderModel, window: i32) {
    for (n, (expected, margin)) in LONG_CHAT_GREEDY.iter().enumerate() {
        // Route A: prefill the prompt, then decode the forced tokens one by one.
        let mut caches = model.make_caches();
        let prompt = mlxcel_core::from_slice_i32(LONG_CHAT_IDS, &[1, LONG_CHAT_IDS.len() as i32]);
        let mut logits =
            model.forward_last_logits_with_caches(&prompt, &mut caches, LONG_CHAT_IDS.len() - 1);
        for (forced, _) in &LONG_CHAT_GREEDY[..n] {
            let step = mlxcel_core::from_slice_i32(&[*forced], &[1, 1]);
            logits = model.forward_last_logits_with_caches(&step, &mut caches, 0);
        }
        let incremental = top_k(&logits, 3);

        if n > 0 {
            assert_eq!(
                caches[0].pass2.visible_len(),
                window,
                "step {n}: the pass-2 ring must be holding exactly its window once decode has \
                 trimmed it, not the whole prompt"
            );
            assert_eq!(
                caches[0].pass1.offset,
                LONG_CHAT_IDS.len() as i32 + n as i32,
                "step {n}: the pass-1 cache must still be growing with the sequence"
            );
        }

        // Route B: one prefill over prompt plus the same forced tokens.
        let mut seq = LONG_CHAT_IDS.to_vec();
        seq.extend(LONG_CHAT_GREEDY[..n].iter().map(|(id, _)| *id));
        let mut reprefill_caches = model.make_caches();
        let whole = mlxcel_core::from_slice_i32(&seq, &[1, seq.len() as i32]);
        let reprefill_logits =
            model.forward_last_logits_with_caches(&whole, &mut reprefill_caches, seq.len() - 1);
        let reprefill = top_k(&reprefill_logits, 3);

        assert_eq!(
            incremental[0].0, reprefill[0].0,
            "step {n}: incremental decode over a trimmed ring and a full re-prefill over a \
             mask-windowed cache disagree; got {incremental:?} against {reprefill:?}"
        );

        if *margin > ORACLE_DECISIVE_MARGIN {
            assert_eq!(
                incremental[0].0, *expected,
                "step {n}: the oracle separates its choice by {margin}, so mlxcel must agree; \
                 got {incremental:?}"
            );
        } else {
            eprintln!(
                "step {n}: oracle margin {margin} is inside mlxcel's resolution, not asserted \
                 (oracle {expected}, mlxcel {})",
                incremental[0].0
            );
        }
    }
}
