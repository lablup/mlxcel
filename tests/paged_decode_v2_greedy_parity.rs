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

//! Greedy token parity for the production paged decode v2 path (issue #899).
//!
//! This is the regression guard the issue makes mandatory: on a pinned prompt
//! set, batch 4 with mixed lengths and long prompts, the greedy token stream
//! produced by the fused v2 decode must match the gather-then-SDPA baseline
//! exactly.
//!
//! # How both paths run in one process
//!
//! `MLXCEL_PAGED_ATTENTION_NATIVE` is read once per process behind a
//! `OnceLock`, so it cannot be toggled between two runs of the same test
//! binary. The test therefore selects the path structurally instead, using the
//! one property the fused kernels genuinely require: **every physical row of a
//! layer must live in one contiguous pool buffer**.
//!
//! - The **v2 arm** sets `CachePool::set_paged_slab_blocks` to a slab large
//!   enough for the whole batch, which is exactly what
//!   `resolve_paged_slab_blocks` does on the server. The fused path is
//!   eligible and, above the dispatch token floor, taken.
//! - The **gather arm** leaves the pool on its 32-row default, so every layer
//!   is multi-slab, the fused path declines, and
//!   `paged_batch_decode_attention` answers from `gather_visible` + SDPA. That
//!   is byte-for-byte the pre-#899 production path.
//!
//! Both arms run identical scheduler wiring, identical prompts, and identical
//! greedy sampling, so a token difference can only be the kernel. The test
//! asserts afterwards that the v2 arm really did build a plan, so a silent
//! fallback cannot pass as parity.
//!
//! # Choosing a checkpoint
//!
//! `MLXCEL_PARITY_MODEL` names a model directory (absolute, or relative to the
//! repository root). Without it, each family case looks for its default
//! directory under `models/` and soft-skips when absent, so the matrix can be
//! run family by family as checkpoints arrive.
//!
//! # Running
//!
//! ```text
//! # One explicit checkpoint (any pool-backed family):
//! MLXCEL_PARITY_MODEL=$HOME/.cache/mlxcel/models/Qwen2.5-0.5B-Instruct-4bit \
//!   cargo test --release --test paged_decode_v2_greedy_parity \
//!   --features metal,accelerate -- --ignored --nocapture
//!
//! # The default matrix (each case skips when its directory is absent):
//! cargo test --release --test paged_decode_v2_greedy_parity \
//!   --features metal,accelerate -- --ignored --nocapture
//! ```
//!
//! `MLXCEL_PARITY_PROMPT_LEN` (default 8192) sets the shortest prompt; the four
//! sequences use that length plus staggered offsets so the batch has mixed
//! lengths and the requests cross page boundaries on different steps.
//! `MLXCEL_PARITY_STEPS` (default 32) sets the number of compared decode steps.

mod common;
use common::repo_model_dir;

use std::path::PathBuf;

use mlxcel::{DecodeBatchContext, LanguageModel, initialize_runtime, load_model};
use mlxcel_core::cache::{CachePool, PagedKvLayout, SequenceStateLayout};

/// Paged block size, matching the scheduler's `DEFAULT_PAGED_BLOCK_SIZE`.
const BLOCK_SIZE: usize = 32;

/// Sequences in the batch. Four is the documented v0.4 `--parallel` default and
/// the batch size the issue's benchmark matrix uses.
const BATCH: usize = 4;

/// Token offsets added to the base prompt length, so the four sequences have
/// mixed lengths and open new pages on different decode steps.
const LENGTH_STAGGER: [usize; BATCH] = [0, 37, 96, 151];

/// Deterministic prompt vocabulary. Every id is < 128k, valid for the qwen and
/// llama vocabularies alike, so no tokenizer is needed and the two arms are fed
/// identical bytes.
const PROMPT_ALPHABET: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358, 1079,
    21815, 911, 1246, 498, 3705, 1293, 37597, 323, 1128, 8573, 979, 279, 6193, 27715, 13,
];

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn prompt_len() -> usize {
    env_usize("MLXCEL_PARITY_PROMPT_LEN", 8192)
}

fn decode_steps() -> usize {
    env_usize("MLXCEL_PARITY_STEPS", 32)
}

/// The prompt for sequence `slot`: a deterministic repeating pattern, offset by
/// the slot so no two sequences share a prompt.
fn prompt_for(slot: usize, len: usize) -> Vec<i32> {
    (0..len)
        .map(|i| PROMPT_ALPHABET[(i + slot * 7) % PROMPT_ALPHABET.len()])
        .collect()
}

/// Greedy argmax over the vocab at position `pos` of a `[batch, seq, vocab]`
/// logits tensor, for batch row `row`.
fn greedy_token(logits: &mlxcel_core::MlxArray, row: i32, pos: i32) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let vocab = shape[2];
    let at_pos = mlxcel_core::slice(logits, &[row, pos, 0], &[row + 1, pos + 1, vocab]);
    let flat = mlxcel_core::reshape(&at_pos, &[vocab]);
    mlxcel_core::eval(&flat);
    mlxcel_core::item_i32(&mlxcel_core::argmax_last_axis(&flat))
}

/// The paged sequence-state layout the scheduler builds for the default Fp16 KV
/// mode.
fn scheduler_paged_layout(num_layers: usize) -> SequenceStateLayout {
    SequenceStateLayout::paged_kv_cache(
        PagedKvLayout::uniform(num_layers, BLOCK_SIZE, BLOCK_SIZE).expect("valid paged layout"),
    )
}

/// Slab size that holds the whole batch's rows for one layer, which is what
/// makes the fused path eligible. Mirrors
/// `crate::memory_estimate::resolve_paged_slab_blocks` at this batch and
/// context.
fn slab_blocks_for(prompt: usize, steps: usize) -> usize {
    let longest = prompt + LENGTH_STAGGER[BATCH - 1] + steps + 1;
    longest.div_ceil(BLOCK_SIZE) * BATCH
}

/// Prefill every sequence, then run `steps` batched greedy decode steps through
/// the paged batched path, returning one token stream per sequence.
///
/// `slab_blocks` is the only difference between the two arms: `Some(n)` makes
/// each layer single-slab and the fused path eligible, `None` leaves the pool
/// default so it declines and the gather fallback answers.
fn run_batched(
    model: &mlxcel::LoadedModel,
    slab_blocks: Option<usize>,
    prompt: usize,
    steps: usize,
) -> (Vec<Vec<i32>>, bool) {
    let num_layers = model.num_layers();
    let mut pool = CachePool::new(BATCH + 2);
    pool.set_paged_slab_blocks(slab_blocks)
        .expect("a fresh pool has no storage yet");

    let ids: Vec<_> = (0..BATCH)
        .map(|slot| {
            pool.allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
                .unwrap_or_else(|e| panic!("paged allocate {slot}: {e}"))
        })
        .collect();

    // Prefill each sequence individually, as `execute_full_prefill` does.
    let mut next = vec![0i32; BATCH];
    for (slot, &id) in ids.iter().enumerate() {
        let len = prompt + LENGTH_STAGGER[slot];
        let tokens = prompt_for(slot, len);
        let input = mlxcel_core::from_slice_i32(&tokens, &[1, len as i32]);
        let mask = mlxcel_core::utils::create_causal_mask(len as i32, 0);
        let caches = pool.get_caches_mut(id).expect("caches");
        let logits = model.forward(&input, caches, Some(&mask));
        mlxcel_core::eval(&logits);
        next[slot] = greedy_token(&logits, 0, len as i32 - 1);
    }

    // Batched greedy decode, exactly the dispatch `execute_batched_decode`
    // performs for the paged storage backend.
    let context = DecodeBatchContext::paged_with_native(BLOCK_SIZE as i32, true);
    let mut streams: Vec<Vec<i32>> = vec![Vec::with_capacity(steps); BATCH];
    for _ in 0..steps {
        for (slot, stream) in streams.iter_mut().enumerate() {
            stream.push(next[slot]);
        }
        let input = mlxcel_core::from_slice_i32(&next, &[BATCH as i32, 1]);
        let logits = {
            let mut batch_caches = pool.get_batch_caches_mut(&ids).expect("batch caches");
            model.forward_batched_with_context_and_ids(
                &input,
                Some(&ids),
                &mut batch_caches,
                None,
                Some(&context),
            )
        };
        mlxcel_core::eval(&logits);
        for (slot, entry) in next.iter_mut().enumerate() {
            *entry = greedy_token(&logits, slot as i32, 0);
        }
    }

    // Did the fused path actually run? The plan cache is per pool, so this is
    // immune to any other test in the binary.
    let used_v2 = pool
        .paged_pool_ref()
        .is_some_and(|paged| paged.decode_plan_cache_stats().plan_rebuilds > 0);
    (streams, used_v2)
}

/// Resolve the checkpoint for a case: `MLXCEL_PARITY_MODEL` wins, then the
/// family's default directory under `models/`, then the downloader's cache at
/// `~/.cache/mlxcel/models/<repo>`.
fn resolve_model_dir(default_dir: &str, hf_repo: &str) -> Option<PathBuf> {
    if let Some(raw) = std::env::var_os("MLXCEL_PARITY_MODEL") {
        let path = PathBuf::from(raw);
        let path = if path.is_absolute() {
            path
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
        };
        if path.exists() {
            return Some(path);
        }
        eprintln!(
            "MLXCEL_PARITY_MODEL points at {}, which does not exist",
            path.display()
        );
        return None;
    }
    let path = repo_model_dir(default_dir);
    if path.exists() {
        return Some(path);
    }
    // Fall back to the downloader's cache, which is where `mlxcel download`
    // puts a checkpoint when no `--local-dir` was given.
    hf_cache_dir(hf_repo)
}

/// `~/.cache/mlxcel/models/<org>/<name>` for a `org/name` repository id, when
/// it exists.
fn hf_cache_dir(hf_repo: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home)
        .join(".cache")
        .join("mlxcel")
        .join("models")
        .join(hf_repo);
    path.exists().then_some(path)
}

/// Run the parity comparison for one checkpoint, soft-skipping when it is
/// absent.
fn assert_parity_for(default_dir: &str, fetch_repo: &str) {
    let Some(model_dir) = resolve_model_dir(default_dir, fetch_repo) else {
        eprintln!(
            "Skipping {default_dir}: checkpoint not found.\n\
             Fetch with: ./target/release/mlxcel download {fetch_repo}\n\
             Or point the test at one: MLXCEL_PARITY_MODEL=/path/to/model"
        );
        return;
    };
    let (model, _tokenizer) =
        load_model(&model_dir).unwrap_or_else(|e| panic!("load {}: {e:?}", model_dir.display()));
    if !model.supports_paged_decode_backend() {
        eprintln!(
            "Skipping {}: the family does not support the paged decode backend",
            model_dir.display()
        );
        return;
    }

    let prompt = prompt_len();
    let steps = decode_steps();
    let slab = slab_blocks_for(prompt, steps);
    eprintln!(
        "\n=== paged decode v2 greedy parity: {} ===\n\
         batch {BATCH}, prompts {prompt}{LENGTH_STAGGER:?}, {steps} decode steps, \
         slab {slab} blocks",
        model_dir.display()
    );

    let (gather_streams, gather_used_v2) = run_batched(&model, None, prompt, steps);
    assert!(
        !gather_used_v2,
        "the gather arm must not have reached the fused path; the slab default changed"
    );

    let (v2_streams, v2_used_v2) = run_batched(&model, Some(slab), prompt, steps);
    assert!(
        v2_used_v2,
        "the v2 arm never built a decode plan, so it silently fell back to gather; \
         parity would be vacuous. Check the slab size ({slab} blocks) and the \
         dispatch token floor."
    );

    for slot in 0..BATCH {
        assert_eq!(
            v2_streams[slot], gather_streams[slot],
            "sequence {slot} diverged.\ngather: {:?}\nv2:     {:?}",
            gather_streams[slot], v2_streams[slot]
        );
    }
    eprintln!(
        "OK: {BATCH} sequences x {steps} greedy decode steps identical between the fused v2 \
         path and the gather baseline."
    );
}

#[test]
#[ignore = "loads a real checkpoint and runs long real GPU forwards; run with --ignored"]
fn paged_decode_v2_greedy_parity_qwen3() {
    let _runtime = initialize_runtime();
    assert_parity_for("qwen3-0.6b-4bit", "mlx-community/Qwen3-0.6B-4bit");
}

#[test]
#[ignore = "loads a real checkpoint and runs long real GPU forwards; run with --ignored"]
fn paged_decode_v2_greedy_parity_llama3() {
    let _runtime = initialize_runtime();
    assert_parity_for("llama-3.2-1b-4bit", "mlx-community/Llama-3.2-1B-Instruct-4bit");
}

#[test]
#[ignore = "loads a real checkpoint and runs long real GPU forwards; run with --ignored"]
fn paged_decode_v2_greedy_parity_qwen25() {
    let _runtime = initialize_runtime();
    assert_parity_for(
        "qwen2.5-0.5b-instruct-4bit",
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    );
}
