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

//! Issue #3410 on the tiny model: the server burst's prefill follows the
//! server's chunk size, and the acceptance-aware width (classic steps
//! included) keeps the drafter-on text equal to classic decode.

use mlxcel_core::prefill_plan::PrefillSplits;

use super::*;

/// The server chunk size the test runs at, below the 2048 default.
const CHUNK: usize = 16;

fn long_prompt() -> Vec<i32> {
    (0..40).map(|i| (i * 7 + 3) % 28).collect()
}

fn last_row_bytes(logits: &MlxArray) -> Vec<u8> {
    let shape = mlxcel_core::array_shape(logits);
    let last = shape[1] - 1;
    let row = mlxcel_core::slice(logits, &[0, last, 0], &[shape[0], last + 1, shape[2]]);
    mlxcel_core::array_to_raw_bytes(&row)
}

/// Classic decode as the scheduler runs it with `--prefill-chunk-size 16`:
/// the prompt in 16-token forwards, then one token per step. Returns the
/// last prefill row's logits bytes and `n` greedy tokens.
fn classic_chunked(model: &Inkling2Model, prompt: &[i32], n: usize) -> (Vec<u8>, Vec<i32>) {
    let mut caches = model.make_caches();
    let mut last = None;
    for piece in prompt.chunks(CHUNK) {
        last = Some(model.forward(&ids(piece), &mut caches, None));
    }
    let last = last.expect("a prompt");
    let bytes = last_row_bytes(&last);
    let mut next = *argmax_rows(&last).last().unwrap();
    let mut out = vec![next];
    while out.len() < n {
        let logits = model.forward(&ids(&[next]), &mut caches, None);
        next = argmax_rows(&logits)[0];
        out.push(next);
    }
    (bytes, out)
}

/// Run the round loop from a burst prefill at [`CHUNK`] and return the
/// emitted tokens (first bonus included) with a summary of the run.
fn burst(
    model: &Inkling2Model,
    prompt: &[i32],
    reference: &[i32],
    n: usize,
    adaptive: bool,
) -> (Vec<i32>, DFlashRunOutputSummary) {
    let mut caches = model.make_speculative_caches();
    let none = TokenBiasMap::new();
    let prefill = model.prefill_speculative_planned(
        &ids(prompt),
        &mut caches,
        &CAPTURE,
        &none,
        CHUNK,
        PrefillSplits::NONE,
    );
    let first_bonus = *argmax_rows(&prefill.logits).last().unwrap();
    let first_hidden = concat_captures(&prefill);
    drop(prefill);
    let seen_rows = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let drafter = ScheduledDrafter {
        reference: reference.to_vec(),
        // Mostly wrong proposals, so classic steps are worth taking.
        schedule: vec![0, 0, 1, 0, usize::MAX, 0],
        round: 0,
        next: 0,
        seen_rows: seen_rows.clone(),
    };
    let recorder = Recorder {
        model,
        verify_argmax: RefCell::new(Vec::new()),
        adaptive,
    };
    let sampler = SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    };
    let mut generator = DFlashGenerator::new(Box::new(drafter), sampler, BLOCK as u32, 0);
    let out = generator
        .run(
            &recorder,
            model as &dyn LanguageModel,
            &mut caches,
            first_bonus,
            first_hidden,
            &[],
            n,
            &AtomicBool::new(false),
            &LogprobsConfig::default(),
        )
        .expect("round loop runs");
    let mut tokens = vec![first_bonus];
    tokens.extend(&out.tokens);
    let widths = recorder
        .verify_argmax
        .into_inner()
        .iter()
        .map(Vec::len)
        .collect();
    let seen = seen_rows.lock().unwrap().clone();
    (
        tokens,
        DFlashRunOutputSummary {
            accept_lens: out.accept_lens,
            classic_steps: out.diagnostics.classic_steps,
            widths,
            seen_rows: seen,
        },
    )
}

struct DFlashRunOutputSummary {
    accept_lens: Vec<u32>,
    classic_steps: usize,
    widths: Vec<usize>,
    seen_rows: Vec<i32>,
}

/// A server started with `--prefill-chunk-size 16` prefills the burst in
/// 16-token forwards, as its classic path does. Before, the burst read
/// `MLXCEL_PREFILL_CHUNK` (2048 by default), ran this 40-token prompt as one
/// forward, and left caches classic decode never builds.
#[test]
fn the_burst_prefill_follows_the_server_chunk_size() {
    let model = model(3, false);
    let prompt = long_prompt();
    let n = 24;
    let (classic_bytes, reference) = classic_chunked(&model, &prompt, n + BLOCK);

    let none = TokenBiasMap::new();
    let prefill_at = |chunk: usize| {
        let mut caches = model.make_speculative_caches();
        let out = model.prefill_speculative_planned(
            &ids(&prompt),
            &mut caches,
            &CAPTURE,
            &none,
            chunk,
            PrefillSplits::NONE,
        );
        let rows = mlxcel_core::array_shape(&out.hidden_states[0])[1];
        (last_row_bytes(&out.logits), rows)
    };
    let (bytes, rows) = prefill_at(CHUNK);
    assert_eq!(rows, prompt.len() as i32, "captures cover every prompt row");
    assert_eq!(bytes, classic_bytes, "the burst prefill is the classic one");
    if mlxcel_core::row_invariant::available() {
        // What the server burst did before: one forward over the prompt. A
        // 40-row product takes MLX's gemm, a 16-row one the row-invariant
        // matvec, so the bits differ; this is what makes the test bite.
        let (single, _) = prefill_at(0);
        assert_ne!(single, classic_bytes, "a one-forward prefill differs");
    }

    for adaptive in [false, true] {
        let (tokens, _) = burst(&model, &prompt, &reference, n, adaptive);
        assert_eq!(
            tokens,
            reference[..tokens.len()],
            "adaptive {adaptive}: drafter-on text equals classic text"
        );
    }
}

/// With the target's opt-in the round loop narrows the block and runs
/// one-row classic steps (the third round always is one: it measures the
/// decode cost). The text stays classic decode's, and the drafter is handed
/// every committed row exactly once, the classic steps' rows included.
#[test]
fn classic_steps_keep_the_text_and_feed_the_drafter_every_row() {
    let model = model(3, false);
    let prompt = prompt();
    let n = 40;
    let reference = greedy_reference(&model, &prompt, n + BLOCK);
    let (tokens, run) = burst(&model, &prompt, &reference, n, true);
    assert_eq!(tokens, reference[..tokens.len()]);
    assert!(run.classic_steps >= 1, "widths {:?}", run.widths);
    assert_eq!(
        run.classic_steps,
        run.widths.iter().filter(|&&w| w == 1).count()
    );
    assert_eq!(run.widths.len(), run.accept_lens.len());
    // The drafter saw the prompt, then each committed row once: every row a
    // round committed before the last drafted round.
    let last_draft = run.widths.iter().rposition(|&w| w > 1).unwrap();
    let committed: u32 = run.accept_lens[..last_draft].iter().map(|a| a + 1).sum();
    let seen: i32 = run.seen_rows.iter().sum();
    assert_eq!(
        seen as usize,
        prompt.len() + committed as usize,
        "rows {:?}, widths {:?}",
        run.seen_rows,
        run.widths
    );
}
