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

//! Issue #2353: batched (B = 4) DFlash on a Qwen 3.5 target against four
//! B = 1 runs of the same round loop, byte for byte.
//!
//! Before the fix, a round whose rows accepted different draft counts left
//! the shorter rows' rejected K/V tail zeroed but attended, and the batched
//! stream drifted from the B = 1 stream a few rounds later. The fix clamps
//! every round to a uniform accept, so the rectangular cache never carries a
//! phantom key.
use super::{REACHABLE_PAIRINGS, pairing_present};

/// Tokens per row, counting the first token taken from the prefill.
const MAX_TOKENS: usize = 128;

/// Four prompts whose continuations the drafter predicts with very different
/// success, so rows accept ragged counts from the first rounds on. They are
/// truncated to the shortest tokenization because the batched burst prefills
/// a rectangular `[B, L]` block, as the server's window collector does.
const TEXTS: [&str; 4] = [
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox",
    "Write a haiku about a lighthouse keeper who finds a message in a bottle at dawn, then explain it.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n# Now an iterative version",
    "Count from one to fifty in words, separated by commas: one, two, three, four, five, six, seven,",
];

/// What one batched burst produced.
struct BurstRun {
    /// Per-row tokens, the prefill's first token included.
    tokens: Vec<Vec<i32>>,
    /// Per-row, per-round accept counts as the speculative walk measured them.
    accept_lens: Vec<Vec<u32>>,
    /// Per-row, per-round accept counts the round committed to the cache.
    committed_accept_lens: Vec<Vec<u32>>,
    /// Rounds whose rows disagreed and were clamped to a uniform accept.
    clamped_rounds: usize,
}

/// One DFlash burst over `prompts` (all the same length), replicating the
/// server's `run_dflash_batched_on_target`: a `[B, L]` prefill, the first
/// token per row from the last prompt position, the last position's captured
/// hidden as the first draft input, then `DFlashBatchedGenerator::run_batched`.
fn run_burst(
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
fn mean_accepts(run: &BurstRun) -> (f64, f64) {
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
fn ragged_round(run: &BurstRun) -> bool {
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

/// AC-3 of #2353. A B = 4 DFlash burst on the Qwen 3.5 4B target with its
/// DFlash drafter emits, for every row, exactly the greedy tokens a B = 1
/// burst of the same round loop emits for that row, and at least one round
/// had ragged accepts (otherwise the test proves nothing). Two B = 1 runs of
/// row 0 pin the baseline's own self-parity first.
///
/// The reference is the batched round loop at B = 1, not `DFlashGenerator`
/// (the server's B = 1 arm), so the only variable is the batch. The served
/// concurrent-versus-sequential comparison is the `mlxcel-server` smoke run
/// recorded on #2353's PR.
#[test]
#[ignore = "real-model heavy (Qwen3.5-4B target + DFlash drafter, B=4 vs B=1 batched runs); CI hardware lane only"]
fn greedy_parity_dflash_qwen35_batched_b4_matches_b1() {
    use mlxcel::{LoadedModel, initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};

    let pairing = &REACHABLE_PAIRINGS[0];
    let (target_path, draft_path, present) = pairing_present(pairing);
    if !present {
        eprintln!(
            "Skipping {} (batched B=4): target={:?} draft={:?}",
            pairing.name, target_path, draft_path,
        );
        return;
    }

    super::common::apply_server_mlx_cache_defaults();
    let _runtime = initialize_runtime();
    mlxcel_core::synchronize_default();
    mlxcel_core::clear_memory_cache();

    let (loaded, tokenizer) = load_model(&target_path).expect("target model must load");
    let model: &mlxcel::models::Qwen35Model = match &loaded {
        LoadedModel::Qwen35(m) | LoadedModel::Qwen35Moe(m) => m,
        LoadedModel::Qwen35VLM(vlm) | LoadedModel::Qwen35MoeVLM(vlm) => &vlm.text_model,
        _ => panic!("DFlash pairing requires a Qwen 3.5 family target"),
    };
    let encoded: Vec<Vec<i32>> = TEXTS
        .iter()
        .map(|t| {
            tokenizer
                .encode(t, true)
                .expect("prompt must tokenize")
                .into_iter()
                .map(|id| id as i32)
                .collect()
        })
        .collect();
    let len = encoded.iter().map(Vec::len).min().expect("four prompts");
    let prompts: Vec<Vec<i32>> = encoded.into_iter().map(|p| p[..len].to_vec()).collect();

    let (mut drafter, kind) =
        load_drafter(&draft_path, Some(DrafterKind::Dflash)).expect("DFlash drafter must load");
    assert_eq!(kind, DrafterKind::Dflash);
    let block_size = pairing.block_size;

    // Baseline self-parity: the B = 1 reference must reproduce itself.
    let (first, d) = run_burst(model, drafter, &prompts[..1], block_size);
    let (again, d) = run_burst(model, d, &prompts[..1], block_size);
    drafter = d;
    assert_eq!(
        first.tokens, again.tokens,
        "B = 1 DFlash is not self-consistent on this host; the B = 4 comparison would be noise"
    );

    let mut reference = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let (run, d) = run_burst(model, drafter, std::slice::from_ref(prompt), block_size);
        drafter = d;
        let (walk, committed) = mean_accepts(&run);
        eprintln!(
            "[dflash B=1] {} tokens, mean accept walk {walk:.3} committed {committed:.3}",
            run.tokens[0].len()
        );
        reference.push(run.tokens.into_iter().next().expect("one row"));
    }

    let (batched, _drafter) = run_burst(model, drafter, &prompts, block_size);
    let (walk, committed) = mean_accepts(&batched);
    eprintln!(
        "[dflash B=4] prompt length {len}, rounds {}, clamped rounds {}, mean accept per \
         row-round: walk {walk:.3}, committed {committed:.3}",
        batched.accept_lens[0].len(),
        batched.clamped_rounds,
    );
    let ragged = ragged_round(&batched);
    let mut failures = Vec::new();
    for (row, (got, want)) in batched.tokens.iter().zip(&reference).enumerate() {
        let first = got
            .iter()
            .zip(want)
            .position(|(g, w)| g != w)
            .or_else(|| (got.len() != want.len()).then(|| got.len().min(want.len())));
        eprintln!(
            "[dflash B=4] row {row}: {} tokens (B=1: {}), accept_lens {:?}, first mismatch {first:?}",
            got.len(),
            want.len(),
            batched.accept_lens[row],
        );
        if let Some(i) = first {
            failures.push(format!(
                "row {row}: first mismatch at token {i} (B=4 {:?} vs B=1 {:?})",
                got.get(i),
                want.get(i)
            ));
        }
    }
    assert!(
        ragged,
        "no round had ragged accepts; the prompts do not exercise the clamp"
    );
    assert!(
        failures.is_empty(),
        "batched DFlash is not byte-identical to B = 1:\n{}",
        failures.join("\n")
    );
}
