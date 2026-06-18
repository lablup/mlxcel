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

//! Real-model regression test for #347 batched-prefill seed determinism.
//!
//! The first-token sampler draws from the process-global MLX RNG with no
//! per-call key (`fused_sample` takes only the scalar sampling params). A
//! batched cohort runs every row's `begin_prefill` (which seeds the global RNG)
//! up front, before any row reaches `finish_prefill` (which samples), so before
//! the fix the LAST cohort row's seed governed every row's first-token sample
//! ("last-seed-wins"): a seeded row's first token depended on its siblings'
//! seeds. The fix reseeds the global RNG to each row's own seed inside
//! `finish_prefill`, immediately before that row samples, so a row's first token
//! depends only on its own seed.
//!
//! Invariant choice (the careful part). This test does NOT assert byte-identity
//! of a batched (B > 1) row against a single-prefill (B = 1) row. Padded batched
//! prefill is not bitwise-identical to single-sequence prefill on Metal (the row
//! is padded up to the cohort's longest prompt and run through a wider matmul,
//! so f16 rounding differs), which is the documented #203 / #325 / #326 jitter
//! class, not a seed bug. To isolate the RNG variable, every assertion holds the
//! cohort COMPOSITION constant (same prompts in the same positions, so the
//! forward pass and therefore the measured row's logits are byte-identical) and
//! varies ONLY seed values. Seeds never touch the forward pass, so any change in
//! the measured row's first token under a constant composition is attributable
//! purely to RNG sequencing.
//!
//! The measured row is always enqueued at index 0, so it is begin- and
//! finish-prefilled first; the sibling is enqueued last, so under the old
//! "last-seed-wins" behavior the sibling's seed was the live global RNG state
//! when the measured row sampled. That is exactly the dependency the fix
//! removes.
//!
//! Checks (all under the fixed code):
//! - Sibling-seed invariance (the load-bearing, bug-discriminating check): a
//!   seeded stochastic row at index 0 produces the SAME first token across
//!   several distinct sibling seeds. This fails under "last-seed-wins" and holds
//!   exactly under the fix.
//! - Own-seed dependence (positive control, so the invariance check is not
//!   vacuous): the same row produces MORE THAN ONE distinct first token across
//!   several of its own seeds, i.e. the sampler genuinely consumes RNG at this
//!   distribution and the reseed reaches the row.
//! - Greedy invariance (AC #2): a `temperature == 0` row at index 0 produces the
//!   same first token regardless of its siblings' seeds (the argmax path
//!   consumes no RNG).
//! - Cohort-split coverage (AC #1 cohort-split clause / AC #3): a seeded row in a
//!   #332 cohort-split mixed window (`[cold A, cold B, adopted C]`, which the
//!   planner splits into a batched `{A, B}` cohort plus a sequential `{C}`
//!   cohort) produces the same first token as the same row in an all-cold
//!   batched window of the same cold composition (`[cold A, cold B]`). Both sides
//!   run the identical `{A, B}` padded forward and reseed A to its own seed, so
//!   the first token matches exactly; this is holistic coverage that the fix
//!   behaves identically across cohort structures, not a separate discriminator
//!   (the within-cohort sibling-seed-invariance check above is the discriminator).
//!
//! The test loads a real qwen3 checkpoint and runs GPU forwards, so it is
//! `#[ignore]` and soft-skips when the model directory is absent.
//!
//! Run with:
//! ```text
//! cargo test -p mlxcel --lib --release --features metal,accelerate \
//!     batched_prefill_seed_determinism -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Fetch the model with:
//! `./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit`.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;

use super::BatchScheduler;
use crate::server::batch::BatchObservability;
use crate::server::batch::sequence::{RequestPriority, SequenceInfo, SequenceState};
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy};
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

/// qwen3 checkpoint directory name (a dense family that opts into batched and
/// padded prefill, the scope of #347).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";

/// A high per-request budget so a request does not stop on the length limit at
/// the first token (the test only ever reads the first sampled token).
const MAX_TOKENS: usize = 64;

/// An open-ended prompt for the measured row: its next-token distribution has
/// real entropy, so stochastic sampling is genuinely RNG-sensitive and the
/// positive control (own-seed dependence) is meaningful. ("Hello, write me a
/// short poem about the moon. Please be concise.\n")
const MEASURED_PROMPT: &[i32] = &[
    9707, 11, 4332, 752, 264, 2805, 22692, 911, 279, 9396, 13, 5209, 387, 63594, 624,
];

/// A longer sibling prompt of a different length, so the batched cohort pads the
/// two rows to a common length (exercising the padded batched forward). The
/// sibling is always seeded so its seed is the "last seed" the old behavior
/// would leak into the measured row. ("...What is 2 + 2?")
const SIBLING_PROMPT: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358, 2776, 264,
    10950, 17847, 13, 6771, 594, 1438, 419, 1495, 3019, 553, 3019, 11, 323, 1473, 697, 975, 13,
    5209, 387, 2797, 624, 14374, 14582, 25, 3555, 374, 220, 17, 488, 220, 17, 30,
];

/// A third prompt routed to the sequential cohort via a non-zero adopted-prefix
/// offset, so a window containing it is a genuine cold + incompatible mix that
/// the planner splits. ("The capital of France is a city. Tell me about it.")
const ADOPTED_PROMPT: &[i32] = &[
    785, 6722, 315, 9625, 374, 264, 3283, 13, 22512, 752, 911, 432, 13,
];

/// `<CARGO_MANIFEST_DIR>/models/<name>` (the mlxcel crate root is the repo root).
fn repo_model_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

/// Build a scheduler that can batch a prefill window of up to four cold rows in
/// one pass (`max_batch_prefill = 4`) and hold the whole window in the active
/// batch (`max_batch_size = 4`).
fn build_scheduler(
    model: crate::LoadedModel,
    tokenizer: MlxcelTokenizer,
    config_eos: Vec<i32>,
) -> BatchScheduler {
    let (_req_tx, req_rx) = mpsc::channel();
    BatchScheduler::with_config(
        model,
        tokenizer,
        config_eos,
        req_rx,
        4,  // max_batch_size
        64, // max_queue_depth
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        512, // prefill_chunk_size > prompts so prefills run whole
        false,
        PreemptionPolicy::default(),
        4, // max_batch_prefill
        DecodeStorageBackend::Paged,
    )
}

/// A stochastic sampling config seeded with `seed`. `temperature > 0` with
/// `top_p < 1` forces the categorical (RNG-consuming) path of `fused_sample`,
/// not the argmax path, so the seed is load-bearing.
fn stochastic(seed: u64) -> SamplingConfig {
    SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.95,
        min_p: 0.0,
        seed: Some(seed),
        ..SamplingConfig::default()
    }
}

/// Build a request bound to `seq_id` with the given prompt, sampling config, and
/// optional adopted-prefix offset. The response receiver is returned and must be
/// kept alive so streamed tokens do not fail to send.
fn make_seq_with(
    seq_id: SequenceId,
    tokenizer: &MlxcelTokenizer,
    prompt_tokens: Vec<i32>,
    prefill_start_offset: usize,
    sampling: SamplingConfig,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let decode_state = StreamingDecodeState::new(tokenizer, &prompt_tokens);
    let seq = SequenceInfo {
        seq_id,
        state: SequenceState::Queued,
        prompt_tokens,
        sampling,
        max_tokens: MAX_TOKENS,
        eos_token_ids: Vec::new(),
        priority: RequestPriority::Normal,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset,
        already_cached_tokens: prefill_start_offset,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        sampler_state: None,
        merged_eos: Vec::new(),
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };
    (seq, rx)
}

/// Release a sequence's pool state if it is still active (a finished sequence has
/// already released its caches).
fn cleanup(sched: &mut BatchScheduler, seq_id: SequenceId) {
    if sched.active_batch.get_mut(seq_id).is_some() {
        sched.active_batch.remove(seq_id);
        sched.release_sequence_caches(seq_id);
    }
}

/// Enqueue a window of `(prompt, prefill_start_offset, sampling)` rows, run one
/// [`BatchScheduler::execute_batched_prefill`] (which classifies the rows, plans
/// cohorts, and dispatches them), then return the FIRST sampled token id of the
/// row at `target_idx`.
///
/// No decode steps are run: the first token is sampled in `finish_prefill` and
/// stored as the row's `generated_tokens[0]`, which is read straight from the
/// active batch. All rows are released afterward so the pool is clean for the
/// next window.
fn first_token_of_row(
    sched: &mut BatchScheduler,
    tokenizer: &MlxcelTokenizer,
    window: &[(&[i32], usize, SamplingConfig)],
    target_idx: usize,
) -> i32 {
    let mut ids = Vec::with_capacity(window.len());
    // Keep the receivers alive for the duration of the prefill so streamed
    // first-token sends do not fail (the token is read from the active batch,
    // not the channel).
    let mut rxs = Vec::with_capacity(window.len());
    for (prompt, offset, sampling) in window {
        let id = sched
            .allocate_sequence_state()
            .expect("allocate window sequence");
        let (seq, rx) = make_seq_with(id, tokenizer, prompt.to_vec(), *offset, sampling.clone());
        sched
            .prefill_queue
            .enqueue(seq)
            .expect("enqueue window sequence");
        ids.push(id);
        rxs.push(rx);
    }
    // One call drains the whole window, splitting it into cohorts as needed.
    sched.execute_batched_prefill();
    assert!(
        sched.prefill_queue.is_empty(),
        "the whole window must be drained by the cohort dispatch"
    );
    let target_id = ids[target_idx];
    let token = sched
        .active_batch
        .get(target_id)
        .and_then(|s| s.generated_tokens.first().copied())
        .unwrap_or_else(|| {
            panic!(
                "row {target_idx} produced no first token (it finished at prefill, e.g. an \
                 immediate EOS; pick a prompt that continues past the first token)"
            )
        });
    for &id in &ids {
        cleanup(sched, id);
    }
    drop(rxs);
    token
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn batched_prefill_seed_determinism_qwen3() {
    let _runtime = crate::initialize_runtime();
    let dir = repo_model_dir(QWEN3_DIR);
    if !dir.exists() {
        eprintln!(
            "Skipping {QWEN3_DIR}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit",
            dir.display()
        );
        return;
    }
    let (model, sched_tokenizer) =
        crate::load_model(&dir).unwrap_or_else(|e| panic!("load {QWEN3_DIR}: {e:?}"));
    let seq_tokenizer = crate::tokenizer::load_tokenizer(&dir).expect("load tokenizer");
    let config_eos = crate::read_eos_token_ids(&dir);
    let mut sched = build_scheduler(model, sched_tokenizer, config_eos);

    // Seed of the measured row (index 0). Held fixed for the sibling-invariance
    // check so only the sibling's seed varies.
    const MEASURED_SEED: u64 = 0xA5A5_A5A5;

    // ---- LOAD-BEARING: sibling-seed invariance (catches "last-seed-wins"). ----
    // The measured row (index 0, stochastic, seed = MEASURED_SEED) is batched
    // with a seeded sibling (index 1, the "last seed"). Across distinct sibling
    // seeds the composition is identical (same two prompts in the same order),
    // so the measured row's logits are byte-identical; only the sibling's seed
    // value changes. The measured row's first token must not move. Under the old
    // "last-seed-wins" behavior it tracked the sibling's seed and would differ.
    const SIBLING_SEEDS: [u64; 3] = [1, 2, 3];
    let mut sibling_varied_tokens = Vec::with_capacity(SIBLING_SEEDS.len());
    for sib in SIBLING_SEEDS {
        let tok = first_token_of_row(
            &mut sched,
            &seq_tokenizer,
            &[
                (MEASURED_PROMPT, 0, stochastic(MEASURED_SEED)),
                (SIBLING_PROMPT, 0, stochastic(sib)),
            ],
            0,
        );
        sibling_varied_tokens.push(tok);
    }
    eprintln!("--- #347 batched-prefill seed determinism (qwen3-0.6b-4bit) ---");
    eprintln!(
        "measured row first token across sibling seeds {SIBLING_SEEDS:?}: {sibling_varied_tokens:?}"
    );
    let baseline = sibling_varied_tokens[0];
    for (sib, &tok) in SIBLING_SEEDS.iter().zip(&sibling_varied_tokens) {
        assert_eq!(
            tok, baseline,
            "a seeded row's first token must not depend on a sibling's seed \
             (sibling seed {sib} changed it: {tok} vs baseline {baseline}); \
             this is the 'last-seed-wins' bug #347"
        );
    }

    // ---- POSITIVE CONTROL: the measured row's OWN seed is load-bearing. ----
    // With the sibling held constant, varying the measured row's own seed must
    // produce more than one distinct first token. This proves the sampler
    // genuinely consumes RNG at this distribution (so the invariance assertion
    // above is not vacuous) and that the reseed reaches the measured row.
    const OWN_SEEDS: [u64; 6] = [10, 20, 30, 40, 50, 60];
    const FIXED_SIBLING_SEED: u64 = 777;
    let mut own_varied_tokens = Vec::with_capacity(OWN_SEEDS.len());
    for own in OWN_SEEDS {
        let tok = first_token_of_row(
            &mut sched,
            &seq_tokenizer,
            &[
                (MEASURED_PROMPT, 0, stochastic(own)),
                (SIBLING_PROMPT, 0, stochastic(FIXED_SIBLING_SEED)),
            ],
            0,
        );
        own_varied_tokens.push(tok);
    }
    let distinct: HashSet<i32> = own_varied_tokens.iter().copied().collect();
    eprintln!(
        "measured row first token across own seeds {OWN_SEEDS:?}: {own_varied_tokens:?} \
         ({} distinct)",
        distinct.len()
    );
    assert!(
        distinct.len() >= 2,
        "positive control failed: the measured row's first token never changed across own \
         seeds {OWN_SEEDS:?}, so sampling is effectively deterministic here and the \
         sibling-invariance check would be vacuous; pick a higher-entropy prompt/temperature"
    );

    // ---- AC #2: a greedy (temperature == 0) row is unaffected by siblings. ----
    // The argmax path consumes no RNG, so a greedy row at index 0 must produce
    // the same first token regardless of any seeded sibling's seed.
    let mut greedy_tokens = Vec::with_capacity(SIBLING_SEEDS.len());
    for sib in SIBLING_SEEDS {
        let tok = first_token_of_row(
            &mut sched,
            &seq_tokenizer,
            &[
                (MEASURED_PROMPT, 0, SamplingConfig::greedy()),
                (SIBLING_PROMPT, 0, stochastic(sib)),
            ],
            0,
        );
        greedy_tokens.push(tok);
    }
    eprintln!("greedy row first token across sibling seeds {SIBLING_SEEDS:?}: {greedy_tokens:?}");
    let greedy_baseline = greedy_tokens[0];
    for (sib, &tok) in SIBLING_SEEDS.iter().zip(&greedy_tokens) {
        assert_eq!(
            tok, greedy_baseline,
            "a greedy row's first token must be independent of any sibling's seed \
             (sibling seed {sib} changed it: {tok} vs baseline {greedy_baseline})"
        );
    }

    // ---- AC #1 / AC #3: cohort-split coverage (#332 mixed window). ----
    // A seeded row A (index 0) in an all-cold batched window [cold A, cold B]
    // must produce the same first token as the same A in a cohort-split mixed
    // window [cold A, cold B, adopted C], where the planner forms BatchedCold
    // {A, B} + Sequential {C}. In both windows A is row 0 of an identical
    // BatchedCold {A, B} cohort (so its logits match) and A reseeds to its own
    // seed before sampling, so the first token matches exactly. C carries a
    // non-zero prefill_start_offset so the window is a genuine cold + adopted mix
    // that the planner splits; C is greedy and is not measured (it only forces
    // the split).
    const A_SEED: u64 = 0x00C0_FFEE;
    const B_SEED: u64 = 0x0000_BEEF;
    let allcold_a = first_token_of_row(
        &mut sched,
        &seq_tokenizer,
        &[
            (MEASURED_PROMPT, 0, stochastic(A_SEED)),
            (SIBLING_PROMPT, 0, stochastic(B_SEED)),
        ],
        0,
    );
    let mixed_a = first_token_of_row(
        &mut sched,
        &seq_tokenizer,
        &[
            (MEASURED_PROMPT, 0, stochastic(A_SEED)),
            (SIBLING_PROMPT, 0, stochastic(B_SEED)),
            (ADOPTED_PROMPT, 1, SamplingConfig::greedy()),
        ],
        0,
    );
    eprintln!(
        "cohort-split: all-cold A first token = {allcold_a}, mixed A first token = {mixed_a}"
    );
    assert_eq!(
        mixed_a, allcold_a,
        "a seeded row's first token must be identical in an all-cold batched window and in a \
         #332 cohort-split mixed window of the same cold composition"
    );
}
