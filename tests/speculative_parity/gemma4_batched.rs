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

//! Issue #2190: batched Gemma 4 MTP against per-sequence classic greedy
//! decode, byte for byte, with no near-tie tolerance.
use super::{REACHABLE_PAIRINGS, UNIFIED_12B_MTP_PAIRING, gemma4_text_wrapper, pairing_present};

/// Tokens per row, counting the first token taken from the prefill.
const MAX_TOKENS: usize = 128;

/// Four prompts of different lengths; the equal-length set truncates them to
/// the shortest.
const EQUAL_TEXTS: [&str; 4] = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator keeps food cold, step by step, for a curious reader.",
    "List five interesting facts about octopuses and explain why each one matters.",
    "Describe the water cycle to a ten year old child using a simple everyday example.",
];

const RAGGED_TEXTS: [&str; 4] = [
    "Why is the sky blue?",
    "Translate to French: The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Hi! What is 17 times 23, and how did you get the answer?",
];

fn argmax(logits: &mlxcel_core::MlxArray) -> i32 {
    let ids = mlxcel_core::argmax_last_axis(logits);
    let shape = mlxcel_core::array_shape(&ids);
    let last = mlxcel_core::reshape(&ids, &[shape.iter().product::<i32>()]);
    let n = mlxcel_core::array_shape(&last)[0];
    let cell = mlxcel_core::slice(&last, &[n - 1], &[n]);
    mlxcel_core::item_i32(&mlxcel_core::reshape(&cell, &[]))
}

/// Per-sequence classic greedy decode: the scheduler's prefill (chunks of
/// `prefill_chunk_len`, last row through the LM head) and one-token decode
/// steps on the sequence's own cache.
fn classic_greedy(
    wrapper: &mlxcel::models::Gemma4Wrapper,
    seq_raw: u64,
    prompt: &[i32],
    eos: &[i32],
) -> Vec<i32> {
    use mlxcel_core::generate::LanguageModel;
    let seq = mlxcel_core::cache::SequenceId::from_raw(seq_raw);
    wrapper.prepare_sequence_state(seq);
    let chunk = mlxcel_core::generate::prefill_chunk_len().max(1);
    let mut logits = None;
    for part in prompt.chunks(chunk) {
        let input = mlxcel_core::from_slice_i32(part, &[1, part.len() as i32]);
        let out = wrapper.forward_last_logits_with_sequence_id(
            &input,
            Some(seq),
            &mut [],
            None,
            part.len() - 1,
        );
        mlxcel_core::eval(&out);
        logits = Some(out);
    }
    let mut next = argmax(&logits.expect("non-empty prompt"));
    let mut tokens = Vec::with_capacity(MAX_TOKENS);
    loop {
        tokens.push(next);
        if eos.contains(&next) || tokens.len() >= MAX_TOKENS {
            break;
        }
        let input = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        let out = wrapper.forward_with_sequence_id(&input, Some(seq), &mut [], None);
        next = argmax(&out);
    }
    wrapper.release_sequence_state_by_id(seq);
    tokens
}

/// Whether some round had rows accept different counts, i.e. the burst ran
/// at least one round whose rows' valid ends no longer line up.
fn has_divergent_round(accept_lens: &[Vec<u32>]) -> bool {
    let rounds = accept_lens.iter().map(Vec::len).min().unwrap_or(0);
    (0..rounds).any(|i| accept_lens.iter().any(|row| row[i] != accept_lens[0][i]))
}

/// AC-3 of #2190. For each row-wise pair (31B, and 12B on CUDA), batched MTP
/// at B = 2 and B = 4, on equal-length and on ragged prompts, must emit for
/// every row exactly the tokens per-sequence classic greedy decode emits.
/// Unlike the `greedy_parity_mtp_gemma4_batched_b4_*` gates this applies no
/// near-tie tolerance, and every case must contain a divergent accept round.
#[test]
#[ignore = "real-model heavy (Gemma-4-31B and 12B targets + drafters, B=2/B=4 batched runs); CI hardware lane only"]
fn greedy_parity_mtp_gemma4_batched_matches_classic() {
    use mlxcel::models::gemma4_mtp_target::Gemma4MtpBatchedTargetAdapter;
    use mlxcel::{initialize_runtime, load_model};
    use mlxcel_core::drafter::{DrafterKind, load_drafter};
    use mlxcel_core::generate::SamplingConfig;
    use mlxcel_core::speculative::mtp::MtpBatchedGenerator;
    use mlxcel_core::speculative::mtp::target::MtpTarget;

    let _runtime = initialize_runtime();
    let sampling = SamplingConfig::greedy();
    let mut failures = Vec::new();
    for pairing in [
        &REACHABLE_PAIRINGS[1],
        &REACHABLE_PAIRINGS[UNIFIED_12B_MTP_PAIRING],
    ] {
        let (target_path, draft_path, present) = pairing_present(pairing);
        if !present {
            eprintln!("Skipping {} (batched vs classic)", pairing.name);
            continue;
        }
        mlxcel_core::synchronize_default();
        mlxcel_core::clear_memory_cache();
        let (loaded, tokenizer) = load_model(&target_path).expect("target model must load");
        let wrapper = gemma4_text_wrapper(&loaded);
        assert!(
            wrapper.mtp_requires_linear_singleton(),
            "{} must be a row-wise geometry on this host",
            pairing.name
        );
        let encode = |text: &str| -> Vec<i32> {
            tokenizer
                .encode(text, true)
                .expect("prompt must tokenize")
                .into_iter()
                .map(|id| id as i32)
                .collect()
        };
        let equal: Vec<Vec<i32>> = {
            let encoded: Vec<Vec<i32>> = EQUAL_TEXTS.iter().map(|t| encode(t)).collect();
            let len = encoded.iter().map(Vec::len).min().expect("four prompts");
            encoded.into_iter().map(|p| p[..len].to_vec()).collect()
        };
        let ragged: Vec<Vec<i32>> = RAGGED_TEXTS.iter().map(|t| encode(t)).collect();
        assert!(
            ragged.iter().any(|p| p.len() != ragged[0].len()),
            "ragged prompts must differ in length"
        );
        let eos = Gemma4MtpBatchedTargetAdapter::new(wrapper, 1).eos_token_ids();

        let mut seq_raw = 7000_u64;
        let mut classic = |prompts: &[Vec<i32>]| -> Vec<Vec<i32>> {
            prompts
                .iter()
                .map(|prompt| {
                    seq_raw += 1;
                    classic_greedy(wrapper, seq_raw, prompt, &eos)
                })
                .collect()
        };
        let equal_ref = classic(&equal);
        let ragged_ref = classic(&ragged);

        for (set, prompts, reference) in [
            ("equal", &equal, &equal_ref),
            ("ragged", &ragged, &ragged_ref),
        ] {
            for batch in [2_usize, 4] {
                let label = format!("{} {set} B={batch}", pairing.name);
                let prompts = &prompts[..batch];
                let adapter = Gemma4MtpBatchedTargetAdapter::new_with_block_size(
                    wrapper,
                    batch,
                    pairing.block_size as usize,
                );
                let (mut drafter, _) = load_drafter(&draft_path, Some(DrafterKind::Mtp))
                    .expect("MTP drafter must load");
                drafter
                    .bind(wrapper as &dyn mlxcel_core::generate::LanguageModel)
                    .expect("drafter bind");
                let mut generator =
                    MtpBatchedGenerator::new(adapter, drafter, pairing.block_size as usize);
                let run = match generator.run_batched(prompts, &sampling, MAX_TOKENS) {
                    Ok(run) => run,
                    Err(e) => {
                        failures.push(format!("[{label}] batched run failed: {e:?}"));
                        continue;
                    }
                };
                let divergent = has_divergent_round(&run.accept_lens);
                eprintln!(
                    "[{label}] prompt lengths {:?}, divergent round: {divergent}",
                    prompts.iter().map(Vec::len).collect::<Vec<_>>()
                );
                if !divergent {
                    failures.push(format!("[{label}] no divergent accept round"));
                }
                for (row, (got, want)) in run.tokens.iter().zip(reference.iter()).enumerate() {
                    let first = got
                        .iter()
                        .zip(want.iter())
                        .position(|(g, w)| g != w)
                        .or_else(|| (got.len() != want.len()).then(|| got.len().min(want.len())));
                    eprintln!(
                        "[{label}] row {row}: {} tokens, accept_lens {:?}, first mismatch {first:?}",
                        got.len(),
                        run.accept_lens[row],
                    );
                    if let Some(i) = first {
                        failures.push(format!(
                            "[{label}] row {row}: first mismatch at token {i} (batched {:?} vs classic {:?})",
                            got.get(i),
                            want.get(i)
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "batched MTP is not byte-identical to classic decode:\n{}",
        failures.join("\n")
    );
}
