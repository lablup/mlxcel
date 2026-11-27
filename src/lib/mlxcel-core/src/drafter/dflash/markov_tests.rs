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

//! Tests for the DSpark Markov token-transition head (issue #1339).

use super::*;
use crate::dtype;

const VOCAB: i32 = 16;

/// A head over `VOCAB` tokens at rank `VOCAB`: `w1` is the identity so
/// `w1[t] = e_t`, and `w2` is whatever `[vocab, rank]` matrix the test wants
/// the transition to be, so `bias(prev) = w2[:, prev]`.
fn head_with_transition(w2: Vec<f32>) -> VanillaMarkovHead {
    let mut identity = vec![0.0_f32; (VOCAB * VOCAB) as usize];
    for t in 0..VOCAB as usize {
        identity[t * VOCAB as usize + t] = 1.0;
    }
    let mut weights: WeightMap = std::collections::HashMap::new();
    weights.insert(
        "markov_head.markov_w1.weight".to_string(),
        ffi::from_slice_f32(&identity, &[VOCAB, VOCAB]),
    );
    weights.insert(
        "markov_head.markov_w2.weight".to_string(),
        ffi::from_slice_f32(&w2, &[VOCAB, VOCAB]),
    );
    VanillaMarkovHead::from_weights(
        &weights,
        "markov_head",
        VOCAB as usize,
        VOCAB as usize,
        64,
        4,
    )
    .expect("markov head must load from w1 / w2")
}

/// `[1, gamma, VOCAB]` logits that are flat (all zero) at every position.
fn flat_base_logits(gamma: i32) -> UniquePtr<MlxArray> {
    ffi::zeros(&[1, gamma, VOCAB], dtype::FLOAT32)
}

/// `[1, gamma, VOCAB]` logits where position `i` puts a `+5` on token
/// `first + i` and nothing else.
fn position_indexed_base_logits(gamma: i32, first: i32) -> UniquePtr<MlxArray> {
    let mut buf = vec![0.0_f32; (gamma * VOCAB) as usize];
    for i in 0..gamma {
        buf[(i * VOCAB + first + i) as usize] = 5.0;
    }
    ffi::from_slice_f32(&buf, &[1, gamma, VOCAB])
}

#[test]
fn markov_chain_feeds_previous_token_into_next_step() {
    // Transition: `w2[t + 1, t] = 100`, so `bias(prev)` is a large spike on
    // `prev + 1`. With flat base logits the chain must walk `anchor + 1`,
    // `anchor + 2`, ... which is only possible if each step's argmax is fed
    // back as the next step's `prev`.
    let mut w2 = vec![0.0_f32; (VOCAB * VOCAB) as usize];
    for t in 0..VOCAB - 1 {
        w2[((t + 1) * VOCAB + t) as usize] = 100.0;
    }
    let head = head_with_transition(w2);
    assert_eq!(head.rank(), VOCAB as usize);

    let gamma = 5;
    let anchor = 2;
    let proposals = head.sample_block(&flat_base_logits(gamma), anchor);
    assert_eq!(proposals, vec![3, 4, 5, 6, 7]);

    // The device-side variant carries the same ids as a `[1, gamma]` int32
    // row, which is what the round loop concatenates behind the anchor.
    let arr = head.sample_block_array(&flat_base_logits(gamma), anchor);
    assert_eq!(ffi::array_shape(&arr), vec![1, gamma]);
    assert_eq!(ffi::array_dtype(&arr), dtype::INT32);
    assert_eq!(
        super::super::materialize_argmax_i32_vec(&arr, gamma as usize),
        vec![3, 4, 5, 6, 7]
    );
}

#[test]
fn dspark_draft_block_uses_all_positions() {
    // No transition at all: the chain reduces to the per-position argmax of
    // the base logits, and it must consume position 0 as a proposal rather
    // than discard it the way a DFlash draft discards its bonus scaffold.
    let zero_transition = vec![0.0_f32; (VOCAB * VOCAB) as usize];
    let head = head_with_transition(zero_transition);
    let gamma = 5;
    let proposals = head.sample_block(&position_indexed_base_logits(gamma, 10), 3);
    assert_eq!(proposals, vec![10, 11, 12, 13, 14]);

    // Proposal 0 derives from the anchor row: a transition from the anchor
    // (token 3) that outweighs position 0's base spike wins that position,
    // and because the chosen token (1) has no outgoing transition, the
    // remaining positions fall back to their base spikes.
    let mut w2 = vec![0.0_f32; (VOCAB * VOCAB) as usize];
    w2[(VOCAB + 3) as usize] = 1000.0;
    let head = head_with_transition(w2);
    let proposals = head.sample_block(&position_indexed_base_logits(gamma, 10), 3);
    assert_eq!(proposals, vec![1, 11, 12, 13, 14]);
}

#[test]
fn markov_head_load_requires_both_factors() {
    let mut weights: WeightMap = std::collections::HashMap::new();
    weights.insert(
        "markov_head.markov_w1.weight".to_string(),
        ffi::zeros(&[VOCAB, 4], dtype::FLOAT32),
    );
    let Err(err) =
        VanillaMarkovHead::from_weights(&weights, "markov_head", VOCAB as usize, 4, 64, 4)
    else {
        panic!("a missing markov_w2 must fail the load");
    };
    assert!(
        err.contains("markov_head.markov_w2"),
        "error must name the missing tensor: {err}"
    );
}

/// Both factors are measured against the config that sizes them, before
/// either becomes a layer.
///
/// The row check is the security-critical half: `markov_w1` is gathered at a
/// token id once per chain step and MLX range-checks no positive gather
/// index, so a table with fewer rows than the vocabulary the chain draws from
/// reads past its own buffer into the logits. The rank check is the
/// availability half: a `markov_w2` of the wrong width throws inside
/// `quantized_matmul` or `matmul`, which crosses the cxx bridge as a process
/// abort rather than a load error.
#[test]
fn markov_factors_are_measured_against_the_config() {
    let full = |rows: i32, cols: i32| {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert(
            "markov_head.markov_w1.weight".to_string(),
            ffi::zeros(&[rows, cols], dtype::FLOAT32),
        );
        weights.insert(
            "markov_head.markov_w2.weight".to_string(),
            ffi::zeros(&[rows, cols], dtype::FLOAT32),
        );
        weights
    };

    let short = full(VOCAB - 1, 4);
    let Err(err) = VanillaMarkovHead::from_weights(&short, "markov_head", VOCAB as usize, 4, 64, 4)
    else {
        panic!("a table shorter than the vocabulary must fail the load");
    };
    assert!(err.contains("rows but the drafter config"), "{err}");

    let wide = full(VOCAB, 5);
    assert!(
        VanillaMarkovHead::from_weights(&wide, "markov_head", VOCAB as usize, 4, 64, 4).is_err(),
        "a rank-5 table under a rank-4 config must fail the load"
    );

    assert!(
        VanillaMarkovHead::from_weights(&full(VOCAB, 4), "markov_head", VOCAB as usize, 4, 64, 4)
            .is_ok(),
        "the declared shape must load"
    );
}

/// A quantized factor's stored width is `rank * bits / 32`, so the rank check
/// runs against the bit depths that could produce that width rather than
/// against the rank directly. Skipping it entirely is what is not acceptable:
/// every mlx-community conversion of these drafters is quantized, so the
/// packed case is the common one, not an exotic one.
///
/// `.scales` is what marks a factor quantized here, matching the loaders.
#[test]
fn a_quantized_markov_factor_has_its_packed_width_checked() {
    // rank 64 at 4 bits packs to 64 * 4 / 32 = 8 u32 columns.
    let packed = |cols: i32| {
        let mut weights: WeightMap = std::collections::HashMap::new();
        for factor in ["markov_w1", "markov_w2"] {
            weights.insert(
                format!("markov_head.{factor}.weight"),
                ffi::zeros(&[VOCAB, cols], dtype::UINT32),
            );
            weights.insert(
                format!("markov_head.{factor}.scales"),
                ffi::zeros(&[VOCAB, 1], dtype::FLOAT32),
            );
            weights.insert(
                format!("markov_head.{factor}.biases"),
                ffi::zeros(&[VOCAB, 1], dtype::FLOAT32),
            );
        }
        weights
    };

    // 8 packed columns is rank 64 at 4 bits, and also rank 128 at 2 bits and
    // rank 32 at 8: every one of those is a width the loader could really be
    // handed, so all three are admitted.
    for rank in [64, 128, 32] {
        assert!(
            VanillaMarkovHead::from_weights(&packed(8), "markov_head", VOCAB as usize, rank, 64, 4)
                .is_ok(),
            "rank {rank} is explained by a supported bit depth at 8 packed columns"
        );
    }

    // 8 packed columns cannot be rank 100 at any supported depth.
    let Err(err) =
        VanillaMarkovHead::from_weights(&packed(8), "markov_head", VOCAB as usize, 100, 64, 4)
    else {
        panic!("an unexplainable packed width must fail the load");
    };
    assert!(err.contains("no supported bit depth"), "{err}");

    // The row check still runs on a packed factor.
    let mut short: WeightMap = std::collections::HashMap::new();
    for factor in ["markov_w1", "markov_w2"] {
        short.insert(
            format!("markov_head.{factor}.weight"),
            ffi::zeros(&[VOCAB - 1, 8], dtype::UINT32),
        );
        short.insert(
            format!("markov_head.{factor}.scales"),
            ffi::zeros(&[VOCAB - 1, 1], dtype::FLOAT32),
        );
    }
    assert!(
        VanillaMarkovHead::from_weights(&short, "markov_head", VOCAB as usize, 64, 64, 4).is_err(),
        "a short packed table must still fail the row check"
    );
}

/// MLX's affine quantize kernel supports 5 bits, so a 5-bit factor must load.
/// The loader used to carry its own `[2, 3, 4, 6, 8]` list that omitted it
/// (issue #1790); the list now comes from `SUPPORTED_AFFINE_BITS`.
#[test]
fn a_five_bit_markov_factor_loads() {
    // rank 64 at 5 bits packs to 64 * 5 / 32 = 10 u32 columns.
    let mut weights: WeightMap = std::collections::HashMap::new();
    for factor in ["markov_w1", "markov_w2"] {
        weights.insert(
            format!("markov_head.{factor}.weight"),
            ffi::zeros(&[VOCAB, 10], dtype::UINT32),
        );
        weights.insert(
            format!("markov_head.{factor}.scales"),
            ffi::zeros(&[VOCAB, 1], dtype::FLOAT32),
        );
        weights.insert(
            format!("markov_head.{factor}.biases"),
            ffi::zeros(&[VOCAB, 1], dtype::FLOAT32),
        );
    }
    assert!(
        VanillaMarkovHead::from_weights(&weights, "markov_head", VOCAB as usize, 64, 64, 5).is_ok(),
        "10 packed columns is rank 64 at 5 bits"
    );
    // 7 bits is not a width MLX implements, so 7 * 64 / 32 = 14 stays refused.
    for factor in ["markov_w1", "markov_w2"] {
        weights.insert(
            format!("markov_head.{factor}.weight"),
            ffi::zeros(&[VOCAB, 14], dtype::UINT32),
        );
    }
    assert!(
        VanillaMarkovHead::from_weights(&weights, "markov_head", VOCAB as usize, 64, 64, 4)
            .is_err(),
        "an unexplainable width is still refused"
    );
}

// =========================================================================
// Reduced draft vocabulary and `d2t` (issue #3261)
// =========================================================================

/// Target vocabulary of the reduced-vocabulary fixtures.
const TARGET_VOCAB: i32 = 10;
/// Draft vocabulary of the reduced-vocabulary fixtures.
const DRAFT_VOCAB: i32 = 4;
/// Rank of the reduced-vocabulary fixtures (`w1` is then the identity).
const REDUCED_RANK: i32 = TARGET_VOCAB;

/// A head whose `markov_w1` is the `[TARGET_VOCAB, TARGET_VOCAB]` identity
/// (indexed by target ids) and whose `markov_w2` is `w2`,
/// `[DRAFT_VOCAB, TARGET_VOCAB]`: `bias(prev) = w2[:, prev]`, a score per
/// draft id.
fn reduced_head(w2: Vec<f32>, d2t: &[i32]) -> Result<VanillaMarkovHead, String> {
    let mut identity = vec![0.0_f32; (TARGET_VOCAB * TARGET_VOCAB) as usize];
    for t in 0..TARGET_VOCAB as usize {
        identity[t * TARGET_VOCAB as usize + t] = 1.0;
    }
    let mut weights: WeightMap = std::collections::HashMap::new();
    weights.insert(
        "markov_head.markov_w1.weight".to_string(),
        ffi::from_slice_f32(&identity, &[TARGET_VOCAB, REDUCED_RANK]),
    );
    weights.insert(
        "markov_head.markov_w2.weight".to_string(),
        ffi::from_slice_f32(&w2, &[DRAFT_VOCAB, REDUCED_RANK]),
    );
    let table = DraftToTarget::from_offsets(d2t, DRAFT_VOCAB as usize, TARGET_VOCAB as usize)?;
    VanillaMarkovHead::from_weights_with_draft_vocab(
        &weights,
        "markov_head",
        TARGET_VOCAB as usize,
        DRAFT_VOCAB as usize,
        REDUCED_RANK as usize,
        table,
        64,
        4,
    )
}

/// `[1, gamma, DRAFT_VOCAB]` base logits with a `+5` on `draft_ids[i]` at
/// position `i`.
fn draft_logits(draft_ids: &[i32]) -> UniquePtr<MlxArray> {
    let gamma = draft_ids.len() as i32;
    let mut buf = vec![0.0_f32; (gamma * DRAFT_VOCAB) as usize];
    for (i, &id) in draft_ids.iter().enumerate() {
        buf[i * DRAFT_VOCAB as usize + id as usize] = 5.0;
    }
    ffi::from_slice_f32(&buf, &[1, gamma, DRAFT_VOCAB])
}

#[test]
fn reduced_vocab_markov_and_d2t_map_to_target_ids() {
    // Draft vocabulary 4 inside target vocabulary 10, `d2t = [0, 2, 4, 6]`:
    // draft ids 0, 1, 2, 3 are target ids 0, 3, 6, 9. The head accepts a
    // `[4, rank]` `markov_w2` next to a `[10, rank]` `markov_w1`.
    let head = reduced_head(
        vec![0.0_f32; (DRAFT_VOCAB * REDUCED_RANK) as usize],
        &[0, 2, 4, 6],
    )
    .expect("a [draft_vocab, rank] markov_w2 must load");
    assert!(head.remaps_draft_ids());

    // Sampled draft id 3 leaves the drafter as target id 9.
    assert_eq!(head.sample_block(&draft_logits(&[3]), 0), vec![9]);
    // Every draft id, one per position.
    assert_eq!(
        head.sample_block(&draft_logits(&[0, 1, 2, 3]), 0),
        vec![0, 3, 6, 9]
    );
    // The device array carries the same target ids as int32.
    let arr = head.sample_block_array(&draft_logits(&[3, 1]), 0);
    assert_eq!(ffi::array_shape(&arr), vec![1, 2]);
    assert_eq!(ffi::array_dtype(&arr), dtype::INT32);
    assert_eq!(
        super::super::materialize_argmax_i32_vec(&arr, 2),
        vec![9, 3]
    );
}

/// The next step's `prev` is the TARGET id, not the draft id: `markov_w1`
/// is indexed by target ids, so feeding it the draft id would read another
/// token's transition row.
#[test]
fn reduced_vocab_chain_feeds_the_target_id_back() {
    // `w2[d, p]` is the score of draft id `d` after target id `p`. After
    // target 9 (draft 3) the chain must pick draft 1; after target 3 (draft
    // 1) it must pick draft 2. Column 3 also carries a decoy: if the chain
    // fed the DRAFT id 3 back instead of target 9, step 1 would read column
    // 3 and pick draft 0.
    let mut w2 = vec![0.0_f32; (DRAFT_VOCAB * REDUCED_RANK) as usize];
    let at = |d: i32, p: i32| (d * REDUCED_RANK + p) as usize;
    w2[at(1, 9)] = 100.0;
    w2[at(2, 3)] = 100.0;
    // Decoy for a chain that feeds a draft id back: after "3" pick draft 0.
    // Column 3 is also the legitimate transition out of target 3, so the
    // legitimate entry is made larger.
    w2[at(0, 3)] = 50.0;
    let head = reduced_head(w2, &[0, 2, 4, 6]).expect("head");

    // Step 0: flat transition out of anchor 0, base logits pick draft 3
    // (target 9). Steps 1 and 2: flat base logits, so the transition decides.
    let mut base = vec![0.0_f32; (3 * DRAFT_VOCAB) as usize];
    base[3] = 5.0;
    let logits = ffi::from_slice_f32(&base, &[1, 3, DRAFT_VOCAB]);
    assert_eq!(head.sample_block(&logits, 0), vec![9, 3, 6]);
}

/// A full-vocabulary head is unchanged by the new constructor: no table,
/// both factors at `vocab` rows, draft ids are target ids.
#[test]
fn full_vocab_head_has_no_remap() {
    let head = head_with_transition(vec![0.0_f32; (VOCAB * VOCAB) as usize]);
    assert!(!head.remaps_draft_ids());
    assert_eq!(
        head.sample_block(&position_indexed_base_logits(3, 10), 3),
        vec![10, 11, 12]
    );
    // An all-zero table is the identity and is dropped rather than gathered.
    assert!(
        DraftToTarget::from_offsets(&[0; 16], 16, 16)
            .expect("identity table")
            .is_none()
    );
    // A draft vocabulary equal to the target one may still carry a real
    // permutation; it is applied.
    let table = DraftToTarget::from_offsets(&[1, -1], 2, 2)
        .expect("a swap is in range")
        .expect("a swap is not the identity");
    let ids = ffi::from_slice_i32(&[0, 1, 1], &[1, 3]);
    assert_eq!(
        super::super::materialize_argmax_i32_vec(&table.to_target_ids(&ids), 3),
        vec![1, 0, 0]
    );
}

/// A `d2t` entry that maps outside the target vocabulary is refused at
/// load, naming the draft id and the target id; nothing is ever gathered
/// with it.
#[test]
fn d2t_out_of_range_is_refused_with_a_message() {
    // One past the last target id: draft 3 + 7 = 10 with vocab 10.
    let err = DraftToTarget::from_offsets(&[0, 2, 4, 7], 4, 10)
        .err()
        .expect("past the target vocabulary");
    assert!(err.contains("draft id 3"), "{err}");
    assert!(err.contains("target id 10"), "{err}");
    assert!(err.contains("[0, 10)"), "{err}");
    // The last valid id is accepted.
    assert!(DraftToTarget::from_offsets(&[0, 2, 4, 6], 4, 10).is_ok());
    // Negative target.
    let err = DraftToTarget::from_offsets(&[-1, 0, 0, 0], 4, 10)
        .err()
        .expect("negative target id");
    assert!(err.contains("target id -1"), "{err}");
    // Extreme offsets must not wrap into range.
    for extreme in [i32::MAX, i32::MIN] {
        let err = DraftToTarget::from_offsets(&[0, 0, 0, extreme], 4, 10)
            .err()
            .expect("extreme offset");
        assert!(err.contains("draft id 3"), "{err}");
    }
    // The wrong number of entries, on either side, zero included.
    for len in [0usize, 3, 5] {
        let err = DraftToTarget::from_offsets(&vec![0; len], 4, 10)
            .err()
            .expect("wrong table length");
        assert!(err.contains(&format!("{len} entries")), "{err}");
    }
    // The same refusal reaches the head constructor.
    assert!(
        reduced_head(
            vec![0.0_f32; (DRAFT_VOCAB * REDUCED_RANK) as usize],
            &[0, 2, 4, 7]
        )
        .is_err()
    );
}

/// The checkpoint tensor: any integer dtype is read, a float or a
/// mis-shaped tensor is refused.
#[test]
fn d2t_tensor_is_read_from_integer_dtypes_only() {
    let offsets = ffi::from_slice_i32(&[0, 2, 4, 6], &[4]);
    for dt in [dtype::INT32, dtype::INT64, dtype::UINT8, dtype::INT16] {
        let tensor = ffi::astype(&offsets, dt);
        let table = DraftToTarget::from_tensor(&tensor, 4, 10)
            .unwrap_or_else(|e| panic!("dtype {dt}: {e}"))
            .expect("not the identity");
        let ids = ffi::from_slice_i32(&[3, 0, 1, 2], &[1, 4]);
        assert_eq!(
            super::super::materialize_argmax_i32_vec(&table.to_target_ids(&ids), 4),
            vec![9, 0, 3, 6],
            "dtype {dt}"
        );
    }
    let float = ffi::astype(&offsets, dtype::FLOAT32);
    assert!(DraftToTarget::from_tensor(&float, 4, 10).is_err());
    let matrix = ffi::from_slice_i32(&[0, 2, 4, 6], &[2, 2]);
    assert!(DraftToTarget::from_tensor(&matrix, 4, 10).is_err());
}

/// Each factor is measured against its own vocabulary: `markov_w1` against
/// the target's, `markov_w2` against the draft one.
#[test]
fn reduced_vocab_factors_are_measured_against_their_own_vocabulary() {
    let build = |w1_rows: i32, w2_rows: i32| {
        let mut weights: WeightMap = std::collections::HashMap::new();
        weights.insert(
            "markov_head.markov_w1.weight".to_string(),
            ffi::zeros(&[w1_rows, 4], dtype::FLOAT32),
        );
        weights.insert(
            "markov_head.markov_w2.weight".to_string(),
            ffi::zeros(&[w2_rows, 4], dtype::FLOAT32),
        );
        VanillaMarkovHead::from_weights_with_draft_vocab(
            &weights,
            "markov_head",
            10,
            4,
            4,
            None,
            64,
            4,
        )
    };
    assert!(build(10, 4).is_ok());
    // `markov_w2` at the target vocabulary is the pre-#3261 shape and is the
    // wrong one for a reduced head.
    let Err(err) = build(10, 10) else {
        panic!("a [vocab, rank] markov_w2 must be refused for a reduced head");
    };
    assert!(err.contains("markov_w2"), "{err}");
    // `markov_w1` at the draft vocabulary would be gathered past its end.
    let Err(err) = build(4, 4) else {
        panic!("a [draft_vocab, rank] markov_w1 must be refused");
    };
    assert!(err.contains("markov_w1"), "{err}");

    // A table that covers a different number of draft ids than `markov_w2`
    // scores is refused by the head.
    let mut weights: WeightMap = std::collections::HashMap::new();
    weights.insert(
        "markov_head.markov_w1.weight".to_string(),
        ffi::zeros(&[10, 4], dtype::FLOAT32),
    );
    weights.insert(
        "markov_head.markov_w2.weight".to_string(),
        ffi::zeros(&[4, 4], dtype::FLOAT32),
    );
    let table = DraftToTarget::from_offsets(&[1, 1, 1], 3, 10).expect("in range");
    assert!(
        VanillaMarkovHead::from_weights_with_draft_vocab(
            &weights,
            "markov_head",
            10,
            4,
            4,
            table,
            64,
            4
        )
        .is_err()
    );
}

/// Zero proposals is an empty block, not a slice past the logits.
#[test]
fn an_empty_block_yields_no_proposals() {
    let head = head_with_transition(vec![0.0_f32; (VOCAB * VOCAB) as usize]);
    let empty = ffi::zeros(&[1, 0, VOCAB], dtype::FLOAT32);
    let arr = head.sample_block_array(&empty, 3);
    assert_eq!(ffi::array_shape(&arr), vec![1, 0]);
}

/// The remap is one gather per step, so a block over a table the size of a
/// real vocabulary costs what a small one costs; the table itself is
/// validated in one pass. 250k entries, the size of the largest target
/// vocabulary in the tree.
#[test]
fn a_vocabulary_sized_d2t_validates_in_one_pass() {
    let draft_vocab = 250_000usize;
    let vocab = 2 * draft_vocab;
    // Draft id `t` is target id `2 t`.
    let offsets: Vec<i32> = (0..draft_vocab as i32).collect();
    let start = std::time::Instant::now();
    let table = DraftToTarget::from_offsets(&offsets, draft_vocab, vocab)
        .expect("in range")
        .expect("not the identity");
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "validating {draft_vocab} offsets took {elapsed:?}"
    );
    let ids = ffi::from_slice_i32(&[0, 1, 249_999], &[1, 3]);
    assert_eq!(
        super::super::materialize_argmax_i32_vec(&table.to_target_ids(&ids), 3),
        vec![0, 2, 499_998]
    );
    // One bad entry at the very end is still found.
    let mut bad = offsets;
    *bad.last_mut().unwrap() = draft_vocab as i32 + 1;
    assert!(DraftToTarget::from_offsets(&bad, draft_vocab, vocab).is_err());
}
