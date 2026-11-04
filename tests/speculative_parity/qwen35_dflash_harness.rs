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

//! Shared harness for the Qwen 3.5 DFlash batched parity tests.
//!
//! One DFlash burst runner and one arbitrated parity rule serve both the
//! B = 4 versus B = 1 gate of issue #2353 and the identical-prompts control of
//! issue #2851. The rule is byte equality first and, failing that, a first
//! mismatch admitted only when a one-shot B = 1 arbiter ranks both tokens
//! within [`BATCHED_VERIFY_JITTER_MARGIN`] logits of its top token.

/// Tokens per row, counting the first token taken from the prefill.
pub(crate) const MAX_TOKENS: usize = 128;

/// Four prompts whose continuations the drafter predicts with very different
/// success, so rows accept ragged counts from the first rounds on. They are
/// truncated to the shortest tokenization because the batched burst prefills
/// a rectangular `[B, L]` block, as the server's window collector does.
pub(crate) const TEXTS: [&str; 4] = [
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox",
    "Write a haiku about a lighthouse keeper who finds a message in a bottle at dawn, then explain it.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# Now an iterative version",
    "Count from one to fifty in words, separated by commas: one, two, three, four, five, six, seven,",
];

/// Distinct `SequenceId` bases for the one-shot arbiters the two real-model
/// tests spawn, so an id released by one test is never reused by the other.
pub(crate) const ARBITER_SEQ_BASE_CONTROL: u64 = 9100;
pub(crate) const ARBITER_SEQ_BASE_GATE: u64 = 9200;

/// Evaluation-path jitter margin for the batched DFlash parity gate (issue
/// #2851).
///
/// The batched `[B, block]` target verify forward is not bitwise identical to
/// the `[1, block]` forward at B = 1: the leading dimension changes which
/// quantized-projection and SDPA kernels MLX dispatches and how their
/// reductions tile. That is the near-tie class this project already documents
/// for prefill partitioning (`docs/CONTINUOUS_BATCHING.md`, #2170) and gates
/// with `EVAL_PATH_JITTER_MARGIN` for the Gemma 4 MTP verifier
/// (`tests/speculative_parity.rs`). The accept policy is not involved: the
/// identical-prompts control runs four equal rows, so every round commits its
/// walk, no round clamps, and the batched stream still parts from the B = 1
/// stream.
///
/// Calibration on the M3 Ultra validation host (`qwen3.5-4b-4bit` target,
/// `qwen3.5-4b-dflash` drafter, block 16, 128 tokens, Metal), one-shot B = 1
/// arbiter at each row's first mismatch:
///
/// - Ragged gate (`greedy_parity_dflash_qwen35_b4_matches_b1`): row 0 diverges
///   at token 14 (batched 5388, B = 1 15217). The arbiter's argmax is the
///   batched token; top-gap(batched) = 0, top-gap(B = 1) = 0.25 logits.
/// - Identical-prompts control (`identical_prompts_dflash_qwen35_b4_matches_b1`):
///   all four rows diverge at the same token with the same gaps
///   (0.25 / 0.0), because the rows are equal.
/// - Every other row of both runs is byte-identical.
///
/// So the whole observed divergence is a quarter-logit evaluation-path
/// difference, and the margin is set to 1.0, four times the measured maximum.
/// That is not a free widening: the rule admits a first mismatch only when
/// both tokens are near-ties of a third evaluation path, not merely when the
/// batched one is, and the committed margin-removal control in the
/// identical-prompts test rejects every observed mismatch at margin 0.
///
/// Do not raise this without a fresh measurement and a rerun of that control.
///
/// This is outcome (b) of #2851: a measured jitter margin, not byte identity.
/// Outcome (a) was rejected on measurement, not convenience. The divergence is
/// produced by kernel dispatch on the leading dimension inside the batched
/// target verify forward; the drafter's proposals cannot change the emitted
/// chain, because under greedy decoding every emitted token is the target's
/// argmax over the accepted prefix whatever the draft quality. Forcing the
/// B = 1 shapes therefore means evaluating the target verify forward row by
/// row against per-row views of the rectangular KV/GDN caches, which is the
/// Gemma 4 per-row verify architecture (#2190). The clamp arm it would replace
/// measured only 1.07x over B = 1 on this pairing (#2442), so that rewrite
/// would spend the whole batching win to buy bitwise parity at B = 4.
pub(crate) const BATCHED_VERIFY_JITTER_MARGIN: f32 = 1.0;

/// What one batched burst produced.
pub(crate) struct BurstRun {
    /// Per-row tokens, the prefill's first token included.
    pub(crate) tokens: Vec<Vec<i32>>,
    /// Per-row, per-round accept counts as the speculative walk measured them.
    pub(crate) accept_lens: Vec<Vec<u32>>,
    /// Per-row, per-round accept counts the round committed to the cache.
    pub(crate) committed_accept_lens: Vec<Vec<u32>>,
    /// Rounds whose rows disagreed and were clamped to a uniform accept.
    pub(crate) clamped_rounds: usize,
}

/// One DFlash burst over `prompts` (all the same length), replicating the
/// server's `run_dflash_batched_on_target`: a `[B, L]` prefill, the first
/// token per row from the last prompt position, the last position's captured
/// hidden as the first draft input, then `DFlashBatchedGenerator::run_batched`.
pub(crate) fn run_burst(
    model: &mlxcel::models::Qwen35Model,
    drafter: Box<dyn mlxcel_core::drafter::Drafter>,
    prompts: &[Vec<i32>],
    block_size: u32,
) -> (BurstRun, Box<dyn mlxcel_core::drafter::Drafter>) {
    use mlxcel_core::drafter::dflash::{DFlashBatchedGenerator, SpeculativeTarget};
    use mlxcel_core::generate::{LanguageModel, SamplingConfig};

    let batch = prompts.len();
    let len = prompts[0].len();
    assert!(
        prompts.iter().all(|p| p.len() == len),
        "rectangular prompts"
    );
    let capture: Vec<usize> = drafter
        .dflash_target_layer_ids()
        .filter(|ids| !ids.is_empty())
        .map(<[usize]>::to_vec)
        .expect("the Qwen 3.5 DFlash drafter declares its target layer ids");

    let mut caches = model.make_speculative_caches_for_test();
    let flat: Vec<i32> = prompts.iter().flatten().copied().collect();
    let input = mlxcel_core::from_slice_i32(&flat, &[batch as i32, len as i32]);
    let out = model.prefill_forward_with_capture_layers(&input, &mut caches, &capture);

    let logits = out.logits.as_ref().expect("prefill logits");
    let shape = mlxcel_core::array_shape(logits);
    let last = len as i32 - 1;
    let last_logits = mlxcel_core::slice(logits, &[0, last, 0], &[shape[0], last + 1, shape[2]]);
    let ids = mlxcel_core::argmax_last_axis(&last_logits);
    let ids = mlxcel_core::reshape(&ids, &[batch as i32]);
    mlxcel_core::eval(&ids);
    let first: Vec<i32> = (0..batch as i32)
        .map(|r| {
            let cell = mlxcel_core::slice(&ids, &[r], &[r + 1]);
            mlxcel_core::item_i32(&mlxcel_core::reshape(&cell, &[]))
        })
        .collect();

    let mut hidden = mlxcel_core::copy(out.hidden_states[0].as_ref().expect("captured hidden"));
    for slab in out.hidden_states.iter().skip(1) {
        hidden = mlxcel_core::concatenate(&hidden, slab.as_ref().expect("captured hidden"), -1);
    }
    let hs = mlxcel_core::array_shape(&hidden);
    let first_hidden = mlxcel_core::slice(&hidden, &[0, last, 0], &[hs[0], last + 1, hs[2]]);
    drop(out);

    let eos = LanguageModel::eos_token_ids(model);
    let mut generator = DFlashBatchedGenerator::new(
        drafter,
        SamplingConfig::greedy(),
        block_size,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let output = generator
        .run_batched(
            model,
            model as &dyn LanguageModel,
            &mut caches,
            &first,
            first_hidden,
            &eos,
            MAX_TOKENS,
        )
        .expect("batched DFlash run must succeed");
    let tokens = output
        .tokens
        .into_iter()
        .zip(first)
        .map(|(row, first)| std::iter::once(first).chain(row).collect())
        .collect();
    let run = BurstRun {
        tokens,
        accept_lens: output.accept_lens,
        committed_accept_lens: output.committed_accept_lens,
        clamped_rounds: output.clamped_rounds,
    };
    (run, generator.into_drafter())
}

/// Mean accepted draft tokens per live row-round: the walk's count and the
/// count the round committed. A row is live in a round when it still had
/// tokens to emit when the round started; each round emits `committed + 1`
/// tokens per live row.
pub(crate) fn mean_accepts(run: &BurstRun) -> (f64, f64) {
    let (mut walk, mut committed, mut rounds) = (0u64, 0u64, 0u64);
    for ((accepts, kept), emitted) in run
        .accept_lens
        .iter()
        .zip(&run.committed_accept_lens)
        .zip(&run.tokens)
    {
        let mut produced = 1usize;
        for (&a, &c) in accepts.iter().zip(kept) {
            if produced >= emitted.len() {
                break;
            }
            walk += u64::from(a);
            committed += u64::from(c);
            rounds += 1;
            produced += c as usize + 1;
        }
    }
    let rounds = rounds.max(1) as f64;
    (walk as f64 / rounds, committed as f64 / rounds)
}

/// Whether some round had live rows whose walks accepted different counts.
/// A row is live in a round when it still had tokens to emit when the round
/// started; a finished row keeps a walk entry the round discards.
pub(crate) fn ragged_round(run: &BurstRun) -> bool {
    let rounds = run.accept_lens.first().map_or(0, Vec::len);
    let mut produced = vec![1usize; run.tokens.len()];
    for i in 0..rounds {
        let live: std::collections::BTreeSet<u32> = (0..run.tokens.len())
            .filter(|&r| produced[r] < run.tokens[r].len())
            .map(|r| run.accept_lens[r][i])
            .collect();
        if live.len() > 1 {
            return true;
        }
        for (r, p) in produced.iter_mut().enumerate() {
            *p += run.committed_accept_lens[r][i] as usize + 1;
        }
    }
    false
}

/// One row's first divergent token and the one-shot B = 1 arbiter's top-gaps
/// for the reference token and the batched token.
pub(crate) struct NearTie {
    pub(crate) row: usize,
    pub(crate) index: usize,
    pub(crate) want: i32,
    pub(crate) got: i32,
    /// `top - logit[want]` at the arbiter's last position.
    pub(crate) want_gap: f32,
    /// `top - logit[got]` at the arbiter's last position.
    pub(crate) got_gap: f32,
    /// The arbiter's own argmax at that position.
    pub(crate) argmax: i32,
}

/// One-shot B = 1 last-position logits for `prompt + emitted_prefix`, and the
/// top-gap of the reference token and the batched token against them.
///
/// This is the arbiter `check_row_parity_with_near_tie` uses in
/// `tests/speculative_parity.rs`, ported to the Qwen 3.5 text model: a full
/// prefill through `forward_with_sequence_id` on a fresh sequence, which is a
/// third evaluation path that neither the batched nor the B = 1 round loop
/// takes. A one-shot forward can itself differ from an incremental chain,
/// which is why both tokens are measured rather than only the batched one.
fn b1_arbiter_top_gaps(
    model: &mlxcel::models::Qwen35Model,
    seq_raw: u64,
    prompt: &[i32],
    emitted_prefix: &[i32],
    want: i32,
    got: i32,
) -> (f32, f32, i32) {
    use mlxcel_core::generate::LanguageModel;
    let mut tokens: Vec<i32> = prompt.to_vec();
    tokens.extend_from_slice(emitted_prefix);
    let input = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);
    let seq = mlxcel_core::cache::SequenceId::from_raw(seq_raw);
    let logits = model.forward_with_sequence_id(&input, Some(seq), &mut [], None);
    let row = mlxcel_core::slice_last_logits(&logits);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    let vals: Vec<f32> = mlxcel_core::array_to_raw_bytes(&row)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (argmax, top) = vals
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i as i32, v))
        .unwrap_or((-1, f32::NAN));
    LanguageModel::release_sequence_state_by_id(model, seq);
    (top - vals[want as usize], top - vals[got as usize], argmax)
}

/// Report every row whose batched stream parts from its B = 1 reference, with
/// the first divergent token and the one-shot arbiter's top-gaps for both
/// tokens.
///
/// Returns `(near_ties, failures)`. `near_ties` holds one entry per row that
/// diverged at a token position; the caller applies the margin to them.
/// `failures` holds rows that cannot be arbitrated at all (a length mismatch
/// with no token mismatch, i.e. EOS drift).
pub(crate) fn first_mismatch_arbiters(
    model: &mlxcel::models::Qwen35Model,
    label: &str,
    prompts: &[Vec<i32>],
    run: &BurstRun,
    reference: &[Vec<i32>],
    seq_base: u64,
) -> (Vec<NearTie>, Vec<String>) {
    assert_eq!(run.tokens.len(), reference.len(), "one reference per row");
    assert_eq!(run.tokens.len(), prompts.len(), "one prompt per row");
    let mut ties = Vec::new();
    let mut failures = Vec::new();
    for (row, (got_stream, want_stream)) in run.tokens.iter().zip(reference).enumerate() {
        let first = got_stream
            .iter()
            .zip(want_stream)
            .position(|(g, w)| g != w)
            .or_else(|| {
                (got_stream.len() != want_stream.len())
                    .then(|| got_stream.len().min(want_stream.len()))
            });
        eprintln!(
            "[{label}] row {row}: {} tokens (B=1: {}), accept_lens {:?}, first mismatch {first:?}",
            got_stream.len(),
            want_stream.len(),
            run.accept_lens[row],
        );
        let Some(index) = first else {
            continue;
        };
        let (Some(&got), Some(&want)) = (got_stream.get(index), want_stream.get(index)) else {
            failures.push(format!(
                "row {row}: length mismatch without a token mismatch (B=4 {} vs B=1 {} \
                 tokens): EOS handling drift",
                got_stream.len(),
                want_stream.len(),
            ));
            continue;
        };
        let (want_gap, got_gap, argmax) = b1_arbiter_top_gaps(
            model,
            seq_base + row as u64,
            &prompts[row],
            &want_stream[..index],
            want,
            got,
        );
        eprintln!(
            "[{label}] row {row}: first mismatch at token {index} (B=4 {got} vs B=1 {want}); \
             one-shot B=1 arbiter: argmax {argmax}, top-gap(ref)={want_gap:e}, \
             top-gap(batched)={got_gap:e}"
        );
        ties.push(NearTie {
            row,
            index,
            want,
            got,
            want_gap,
            got_gap,
            argmax,
        });
    }
    (ties, failures)
}

/// Apply the margin to every row's first mismatch. A row passes when both its
/// reference token and its batched token sit within `margin` logits of the
/// arbiter's top token; a structural break moves at least one of them far
/// outside it. Returns one message per rejected row, so `margin == 0.0` is the
/// margin-removal control.
pub(crate) fn rows_outside_margin(ties: &[NearTie], margin: f32) -> Vec<String> {
    ties.iter()
        .filter(|tie| {
            !(0.0..=margin).contains(&tie.want_gap) || !(0.0..=margin).contains(&tie.got_gap)
        })
        .map(|tie| {
            format!(
                "row {}: mismatch at token {} (B=4 {} vs B=1 {}, arbiter argmax {}) is not \
                 evaluation-path jitter (top-gaps ref {:e} / batched {:e} exceed {margin}): \
                 structural parity break",
                tie.row, tie.index, tie.got, tie.want, tie.argmax, tie.want_gap, tie.got_gap,
            )
        })
        .collect()
}
