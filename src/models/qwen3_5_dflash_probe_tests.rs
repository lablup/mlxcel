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

//! Real-checkpoint block-versus-chain diagnostics for the Qwen 3.5 DFlash
//! verify path (issue #1935).
//!
//! Served greedy output on `qwen3.5-4b-4bit` + `qwen3.5-4b-dflash` is not
//! byte-identical to classic decode at any DFlash verify width, and the
//! difference is the same for every width from 2 to 16. These tests locate
//! where in the forward the two arms part company, on the real checkpoint,
//! without a server.
//!
//! `#[ignore]`-gated: they need the checkpoint on disk and a GPU.
//!
//! ```text
//! cargo test --release --features cuda -p mlxcel --lib -- --ignored \
//!   --test-threads=1 --nocapture models::qwen3_5::qwen3_5_dflash_probe_tests
//! ```

use super::{Qwen3NextCache, Qwen35Model};
use crate::models::speculative_exactness::BlockChainExactness;
use mlxcel_core::{MlxArray, UniquePtr};

/// Default target checkpoint; override with `MLXCEL_Q35_PROBE_TARGET`.
const DEFAULT_TARGET: &str = "models/mlx/qwen3.5-4b-4bit";

/// Prompt length for the synthetic-id arms. Long enough that the full
/// attention layers have a real KV history behind the verify block.
const PROMPT_LEN: usize = 64;

fn target_dir() -> String {
    std::env::var("MLXCEL_Q35_PROBE_TARGET").unwrap_or_else(|_| DEFAULT_TARGET.to_string())
}

fn widths() -> Vec<usize> {
    match std::env::var("MLXCEL_Q35_PROBE_WIDTHS") {
        Ok(v) => v
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>().expect("width"))
            .collect(),
        Err(_) => vec![2, 4, 8, 16],
    }
}

/// Load the target and hand back its Qwen 3.5 text backbone.
///
/// Returns `None` (and says so) when the checkpoint is not on disk, so a
/// host without it reports a skip instead of a failure.
fn load_text_model() -> Option<(crate::LoadedModel, String)> {
    let dir = target_dir();
    if !std::path::Path::new(&dir).exists() {
        eprintln!("[1935] skipping: {dir} not on disk");
        return None;
    }
    let (model, _tokenizer) = crate::backend::select_backend()
        .load_model(std::path::Path::new(&dir))
        .expect("target model must load");
    Some((model, dir))
}

fn text_model(model: &crate::LoadedModel) -> &Qwen35Model {
    match model {
        crate::LoadedModel::Qwen35(m) => m,
        crate::LoadedModel::Qwen35VLM(vlm) => &vlm.text_model,
        _ => panic!("target is not a Qwen 3.5 text backbone"),
    }
}

fn ids_of(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

/// Deterministic synthetic ids, the same shape the in-tree probe draws.
fn synthetic(len: usize, vocab: usize, stride: usize, offset: usize) -> Vec<i32> {
    (0..len)
        .map(|i| ((i * stride + offset) % vocab) as i32)
        .collect()
}

/// Raw bytes of one `[1, T, W]` row.
fn row_bytes(t: &MlxArray, index: i32) -> Vec<u8> {
    let shape = mlxcel_core::array_shape(t);
    let row = mlxcel_core::slice(t, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
    mlxcel_core::array_to_raw_bytes(&row)
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Which arm each verdict came from, for the printed table.
fn verdict_line(label: &str, verdict: &BlockChainExactness) -> String {
    format!("[1935] {label}: {}", verdict.reason())
}

/// **Diagnostic 1.** Does `probe_block_chain_exactness` (the gate the issue
/// proposes to wire up for this family) actually fire on this checkpoint,
/// and at which widths?
///
/// A `Diverges` verdict means the proposed gate would decline the burst; an
/// `Equal` verdict at a width the served arms diverge at would mean the
/// probe measures the wrong pair and the gate would be a no-op.
#[test]
#[ignore = "needs the real Qwen 3.5 4B checkpoint and a GPU"]
fn block_chain_exactness_verdicts_on_the_real_checkpoint() {
    let Some((model, dir)) = load_text_model() else {
        return;
    };
    let text = text_model(&model);
    eprintln!("[1935] target {dir}");
    for width in widths() {
        let verdict = text.probe_block_chain_exactness(width);
        eprintln!("{}", verdict_line(&format!("width {width}"), &verdict));
    }
}

/// **Diagnostic 2.** Per-layer bisect of one verify block against the
/// single-token chain, on the real checkpoint.
///
/// Both arms run `forward_speculative` from the same prefill, so the only
/// difference is `T = width` against `T = 1`. Position 0 of the block sees
/// exactly the context the chain's first step sees, so a difference there is
/// a pure row-count effect with identical mathematics behind it, and the
/// first layer that shows one names the operator responsible.
///
/// Layers 0, 1, 2 are gated-delta (linear attention) and layer 3 is the
/// first full-attention layer on a `full_attention_interval = 4` checkpoint,
/// so the first differing layer index separates the two candidate families
/// the issue lists.
#[test]
#[ignore = "needs the real Qwen 3.5 4B checkpoint and a GPU"]
fn per_layer_block_versus_chain_bisect() {
    let Some((model, dir)) = load_text_model() else {
        return;
    };
    let text = text_model(&model);
    let vocab = text.config.vocab_size;
    let layers = text.num_layers();
    let capture: Vec<usize> = (0..layers).collect();
    let block_size: usize = std::env::var("MLXCEL_Q35_PROBE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    eprintln!("[1935] target {dir}, {layers} layers, block {block_size}");

    let prompt = synthetic(PROMPT_LEN, vocab, 7, 1);
    let block = synthetic(block_size, vocab, 13, 3);

    // Chain arm: prefill, then one token per forward.
    let mut chain_caches: Vec<Qwen3NextCache> = text.make_speculative_caches();
    let _ = text.forward_speculative(&ids_of(&prompt), &mut chain_caches, &capture);
    let mut chain_hidden: Vec<Vec<Vec<u8>>> = Vec::with_capacity(block_size);
    let mut chain_logits: Vec<Vec<u8>> = Vec::with_capacity(block_size);
    for token in &block {
        let out = text.forward_speculative(&ids_of(&[*token]), &mut chain_caches, &capture);
        chain_hidden.push(
            out.hidden_states
                .iter()
                .map(|h| row_bytes(h, 0))
                .collect::<Vec<_>>(),
        );
        chain_logits.push(row_bytes(&out.logits, 0));
    }

    // Block arm: fresh caches, same prefill, the whole block at once.
    let mut block_caches: Vec<Qwen3NextCache> = text.make_speculative_caches();
    let _ = text.forward_speculative(&ids_of(&prompt), &mut block_caches, &capture);
    let out = text.forward_speculative(&ids_of(&block), &mut block_caches, &capture);

    let mut first_layer: Option<usize> = None;
    for (layer, chain_row) in chain_hidden[0].iter().enumerate().take(layers) {
        let block_row = row_bytes(&out.hidden_states[layer], 0);
        let d = differing(&block_row, chain_row);
        if d > 0 {
            if first_layer.is_none() {
                first_layer = Some(layer);
            }
            if layer < 8 || first_layer == Some(layer) {
                eprintln!(
                    "[1935] position 0, layer {layer} ({}): {d} of {} hidden bytes differ",
                    if text.config.is_linear_layer(layer) {
                        "gated-delta"
                    } else {
                        "full attention"
                    },
                    block_row.len()
                );
            }
        }
    }
    match first_layer {
        None => eprintln!(
            "[1935] position 0: every layer's hidden state is byte-identical between the \
             {block_size}-row block and the single-token chain"
        ),
        Some(layer) => eprintln!(
            "[1935] position 0: FIRST divergence at layer {layer} ({})",
            if text.config.is_linear_layer(layer) {
                "gated-delta"
            } else {
                "full attention"
            }
        ),
    }
    let logit_diff = differing(&row_bytes(&out.logits, 0), &chain_logits[0]);
    eprintln!(
        "[1935] position 0: {logit_diff} of {} logit bytes differ",
        chain_logits[0].len()
    );
}

/// Bytes of a whole tensor, or an empty vector when absent.
fn all_bytes(t: Option<&UniquePtr<MlxArray>>) -> Vec<u8> {
    match t {
        Some(t) => mlxcel_core::array_to_raw_bytes(t.as_ref().expect("live tensor")),
        None => Vec::new(),
    }
}

/// One line describing how far two cache vectors agree.
fn compare_caches(a: &[Qwen3NextCache], b: &[Qwen3NextCache]) -> Option<(usize, String)> {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        match (x, y) {
            (Qwen3NextCache::Attention(kx), Qwen3NextCache::Attention(ky)) => {
                let dk = differing(&all_bytes(kx.keys.as_ref()), &all_bytes(ky.keys.as_ref()));
                let dv = differing(
                    &all_bytes(kx.values.as_ref()),
                    &all_bytes(ky.values.as_ref()),
                );
                if dk > 0 || dv > 0 || kx.offset != ky.offset {
                    return Some((
                        i,
                        format!(
                            "attention layer: {dk} key bytes, {dv} value bytes differ, \
                             offsets {} vs {}",
                            kx.offset, ky.offset
                        ),
                    ));
                }
            }
            (Qwen3NextCache::Linear(gx), Qwen3NextCache::Linear(gy)) => {
                let dc = differing(
                    &all_bytes(gx.conv_state.as_ref()),
                    &all_bytes(gy.conv_state.as_ref()),
                );
                let ds = differing(
                    &all_bytes(gx.state_cache.as_ref()),
                    &all_bytes(gy.state_cache.as_ref()),
                );
                if dc > 0 || ds > 0 || gx.offset != gy.offset {
                    return Some((
                        i,
                        format!(
                            "gated-delta layer: {dc} conv-state bytes, {ds} recurrent-state \
                             bytes differ, offsets {} vs {}",
                            gx.offset, gy.offset
                        ),
                    ));
                }
            }
            _ => return Some((i, "cache variants differ".to_string())),
        }
    }
    None
}

/// **Diagnostic 3.** The pair the block-versus-chain probe does not compare:
/// the classic decode forward (`forward_internal`, which is what
/// `LanguageModel::forward` reaches through the model-owned sequence state)
/// against the speculative verify forward (`forward_speculative`), both at
/// `T = 1` and both from their own prefill of the same prompt.
///
/// The DFlash burst runs every one of its forwards through
/// `forward_speculative` on caches of its own, so this is the pair a served
/// speculative completion is actually compared against when an operator
/// diffs it with classic decode. `probe_block_chain_exactness` holds both of
/// its arms inside `forward_speculative` and therefore cannot see a
/// difference here.
#[test]
#[ignore = "needs the real Qwen 3.5 4B checkpoint and a GPU"]
fn classic_forward_versus_speculative_forward_at_one_row() {
    let Some((model, dir)) = load_text_model() else {
        return;
    };
    let text = text_model(&model);
    let vocab = text.config.vocab_size;
    let steps: usize = std::env::var("MLXCEL_Q35_PROBE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    eprintln!("[1935] target {dir}, comparing classic and speculative forwards at T=1");

    let prompt = synthetic(PROMPT_LEN, vocab, 7, 1);
    let block = synthetic(steps, vocab, 13, 3);

    let mut classic_caches = text.make_internal_caches();
    let classic_prefill = text.forward_internal(&ids_of(&prompt), None, &mut classic_caches, None);
    let mut spec_caches = text.make_speculative_caches();
    let spec_prefill = text.forward_speculative(&ids_of(&prompt), &mut spec_caches, &[]);

    let last = PROMPT_LEN as i32 - 1;
    let prefill_diff = differing(
        &row_bytes(&classic_prefill, last),
        &row_bytes(&spec_prefill.logits, last),
    );
    eprintln!(
        "[1935] prefill last row: {prefill_diff} of {} logit bytes differ",
        row_bytes(&classic_prefill, last).len()
    );
    match compare_caches(&classic_caches, &spec_caches) {
        None => eprintln!("[1935] prefill: every layer's cache state is byte-identical"),
        Some((layer, why)) => {
            eprintln!("[1935] prefill: FIRST cache divergence at layer {layer}: {why}")
        }
    }

    for (i, token) in block.iter().enumerate() {
        let c = text.forward_internal(&ids_of(&[*token]), None, &mut classic_caches, None);
        let s = text.forward_speculative(&ids_of(&[*token]), &mut spec_caches, &[]);
        let d = differing(&row_bytes(&c, 0), &row_bytes(&s.logits, 0));
        eprintln!(
            "[1935] step {i}: {d} of {} logit bytes differ",
            row_bytes(&c, 0).len()
        );
        if d > 0 {
            if let Some((layer, why)) = compare_caches(&classic_caches, &spec_caches) {
                eprintln!("[1935] step {i}: FIRST cache divergence at layer {layer}: {why}");
            }
            break;
        }
    }
}

/// **Diagnostic 4.** Rollback exactness: after a partial accept, is the
/// rewound cache state byte-identical to the state a decode chain of the
/// accepted length would hold?
///
/// The round loop verifies a `block_size`-row block, accepts a prefix of it
/// and rolls the caches back to the accepted length. If that rewind is not
/// exact, every later token inherits the drift even when the verify block
/// itself is byte-identical to the chain, which on this checkpoint it is at
/// widths 2 and 4.
///
/// `MLXCEL_Q35_PROBE_BLOCK` sets the block width, `MLXCEL_Q35_PROBE_KEEP` the
/// number of rows accepted (the round loop's `accepted + 1`).
#[test]
#[ignore = "needs the real Qwen 3.5 4B checkpoint and a GPU"]
fn rollback_after_partial_accept_matches_the_decode_chain() {
    let Some((model, dir)) = load_text_model() else {
        return;
    };
    let text = text_model(&model);
    let vocab = text.config.vocab_size;
    let block_size: usize = std::env::var("MLXCEL_Q35_PROBE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let keep: usize = std::env::var("MLXCEL_Q35_PROBE_KEEP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    assert!(keep >= 1 && keep < block_size, "keep must be in 1..block");
    eprintln!("[1935] target {dir}, block {block_size}, keeping {keep} rows");

    let prompt = synthetic(PROMPT_LEN, vocab, 7, 1);
    let block = synthetic(block_size, vocab, 13, 3);

    // Chain arm: the state after `keep` single-token steps.
    let mut chain_caches = text.make_speculative_caches();
    let _ = text.forward_speculative(&ids_of(&prompt), &mut chain_caches, &[]);
    for token in block.iter().take(keep) {
        let _ = text.forward_speculative(&ids_of(&[*token]), &mut chain_caches, &[]);
    }

    // Block arm: verify the whole block, then roll back to `keep` rows.
    let mut block_caches = text.make_speculative_caches();
    let _ = text.forward_speculative(&ids_of(&prompt), &mut block_caches, &[]);
    let out = text.forward_speculative(&ids_of(&block), &mut block_caches, &[]);
    let kept = text.rollback_speculative_cache(
        &mut block_caches,
        &out.gdn_states,
        &[keep as i32 - 1],
        block_size as i32,
    );
    eprintln!("[1935] rollback returned max accepted index {kept}");

    match compare_caches(&chain_caches, &block_caches) {
        None => eprintln!("[1935] rollback: every layer's cache state is byte-identical"),
        Some((layer, why)) => {
            eprintln!("[1935] rollback: FIRST cache divergence at layer {layer}: {why}")
        }
    }

    // The token that follows is what the next round would verify from.
    let next = synthetic(1, vocab, 29, 11);
    let c = text.forward_speculative(&ids_of(&next), &mut chain_caches, &[]);
    let b = text.forward_speculative(&ids_of(&next), &mut block_caches, &[]);
    let d = differing(&row_bytes(&c.logits, 0), &row_bytes(&b.logits, 0));
    eprintln!(
        "[1935] next step after rollback: {d} of {} logit bytes differ",
        row_bytes(&c.logits, 0).len()
    );
}

fn env_ids(name: &str) -> Option<Vec<i32>> {
    let raw = std::env::var(name).ok()?;
    Some(
        raw.trim()
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<i32>().expect("integer token id"))
            .collect(),
    )
}

/// The capture list the served burst actually uses. Passing an empty slice
/// through `forward_speculative` captures nothing, and a capture inserts a
/// `copy` of the layer's output into the graph, which can change what MLX
/// fuses. `MLXCEL_Q35_PROBE_CAPTURE=0` turns it off for the A/B.
fn probe_capture_layer_ids() -> Vec<usize> {
    match std::env::var("MLXCEL_Q35_PROBE_CAPTURE").as_deref() {
        Ok("0") => Vec::new(),
        Ok(v) if !v.trim().is_empty() => v
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>().expect("layer id"))
            .collect(),
        _ => mlxcel_core::drafter::dflash::config::DEFAULT_TARGET_LAYER_IDS.to_vec(),
    }
}

/// `true` when the burst arm should take its per-position argmax the way the
/// round loop does, one `argmax_last_axis` over the whole `[1, T, vocab]`
/// block, rather than one call per sliced row. A multi-row reduction is not
/// obliged to break an exact tie the same way a single-row one does.
fn probe_block_argmax() -> bool {
    std::env::var("MLXCEL_Q35_PROBE_BLOCK_ARGMAX")
        .map(|v| v != "0")
        .unwrap_or(false)
}

/// Per-position argmax over a whole `[1, T, vocab]` block, which is what
/// `round_loop::argmax_logits_to_array` computes.
fn argmax_block(logits: &MlxArray) -> Vec<i32> {
    let arg = mlxcel_core::argmax_last_axis(logits);
    mlxcel_core::eval(&arg);
    let itemsize = mlxcel_core::array_itemsize(&arg);
    let bytes = mlxcel_core::array_to_raw_bytes(&arg);
    match itemsize {
        4 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        8 => bytes
            .chunks_exact(8)
            .map(|c| i64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as i32)
            .collect(),
        other => panic!("unexpected argmax itemsize {other}"),
    }
}

fn argmax_row(logits: &MlxArray, index: i32) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let row = mlxcel_core::slice(logits, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
    let arg = mlxcel_core::argmax_last_axis(&row);
    let arg = mlxcel_core::reshape(&arg, &[]);
    mlxcel_core::eval(&arg);
    mlxcel_core::item_i32(&arg)
}

/// **Diagnostic 5.** Replay a real classic-decode transcript through both
/// shapes and report where their greedy choices part.
///
/// The classic arm is `forward_internal`, prefill and one token per step,
/// which is what `mlxcel generate` and the server's classic decode both run
/// (verified byte-identical on GB10). The burst arm prefills through
/// `forward_prefill_with_capture_layers` and then feeds the same tokens in
/// `block_size`-row verify blocks through `forward_speculative`, which is
/// the shape a DFlash round takes when every proposal is correct. No
/// rollback is exercised, so a divergence here is in the prefill or the
/// verify block rather than in the rewind.
///
/// Supply the transcript from a real run:
///
/// ```text
/// MLXCEL_Q35_PROBE_PROMPT="[...]" MLXCEL_Q35_PROBE_REFERENCE="[...]"
/// ```
///
/// (`MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate -m <target> -p <prompt> --no-chat-template --temp 0`
/// prints both lines.)
#[test]
#[ignore = "needs the real Qwen 3.5 4B checkpoint, a GPU and a recorded transcript"]
fn reference_replay_classic_versus_burst_shape() {
    let Some((model, dir)) = load_text_model() else {
        return;
    };
    let text = text_model(&model);
    let Some(prompt) = env_ids("MLXCEL_Q35_PROBE_PROMPT") else {
        eprintln!("[1935] skipping: set MLXCEL_Q35_PROBE_PROMPT and MLXCEL_Q35_PROBE_REFERENCE");
        return;
    };
    let reference = env_ids("MLXCEL_Q35_PROBE_REFERENCE").expect("MLXCEL_Q35_PROBE_REFERENCE");
    let block_size: usize = std::env::var("MLXCEL_Q35_PROBE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    eprintln!(
        "[1935] target {dir}, prompt {} ids, reference {} ids, block {block_size}",
        prompt.len(),
        reference.len()
    );

    // Both prefills first, so the cache comparison below sees two caches that
    // have consumed the same tokens. Comparing after the classic chain has run
    // would only report the offset gap the chain itself opened.
    let mut classic_caches = text.make_internal_caches();
    let classic_prefill = text.forward_internal(&ids_of(&prompt), None, &mut classic_caches, None);
    let capture = probe_capture_layer_ids();
    eprintln!(
        "[1935] capture layers {capture:?}, block argmax {}",
        probe_block_argmax()
    );
    let mut burst_caches = text.make_speculative_caches();
    let burst_prefill =
        text.forward_prefill_with_capture_layers(&ids_of(&prompt), &mut burst_caches, &capture);

    let prefill_bytes = differing(
        &row_bytes(&classic_prefill, prompt.len() as i32 - 1),
        &row_bytes(&burst_prefill.logits, prompt.len() as i32 - 1),
    );
    eprintln!("[1935] prefill last row: {prefill_bytes} logit bytes differ");
    match compare_caches(&classic_caches, &burst_caches) {
        None => eprintln!("[1935] prefill: every layer's cache state is byte-identical"),
        Some((layer, why)) => {
            eprintln!("[1935] prefill: FIRST cache divergence at layer {layer}: {why}")
        }
    }

    let mut classic: Vec<i32> = vec![argmax_row(&classic_prefill, prompt.len() as i32 - 1)];
    for token in &reference {
        let out = text.forward_internal(&ids_of(&[*token]), None, &mut classic_caches, None);
        classic.push(argmax_row(&out, 0));
    }

    let mut burst: Vec<i32> = vec![argmax_row(&burst_prefill.logits, prompt.len() as i32 - 1)];
    for chunk in reference.chunks(block_size) {
        let out = text.forward_speculative(&ids_of(chunk), &mut burst_caches, &capture);
        if probe_block_argmax() {
            burst.extend(argmax_block(&out.logits));
        } else {
            for pos in 0..chunk.len() {
                burst.push(argmax_row(&out.logits, pos as i32));
            }
        }
    }

    let first = classic
        .iter()
        .zip(&burst)
        .position(|(a, b)| a != b)
        .map(|i| i as i64)
        .unwrap_or(-1);
    let disagreements = classic.iter().zip(&burst).filter(|(a, b)| a != b).count();
    eprintln!(
        "[1935] greedy argmax over {} positions: {disagreements} disagree, first at {first} \
         (-1 means none)",
        classic.len()
    );
    if first >= 0 {
        let i = first as usize;
        eprintln!(
            "[1935] at position {i}: classic {} against burst {}, reference has {:?}",
            classic[i],
            burst[i],
            reference.get(i)
        );
    }
}

/// **Diagnostic 6.** The round loop's cache dynamics, without a drafter.
///
/// Diagnostic 5 feeds the transcript in aligned blocks with every proposal
/// correct, so it never rolls back. A served burst rolls back on most rounds
/// (measured acceptance on this pairing is about 0.53), and a rollback trims
/// the attention caches and replays the gated-delta state. This arm walks the
/// same transcript but accepts a varying prefix of each block and rewinds the
/// rest, which is the cache history a served burst actually builds, and
/// compares its greedy argmax against the single-token chain position by
/// position.
///
/// `MLXCEL_Q35_PROBE_ACCEPTS` is a comma-separated cycle of accepted row
/// counts (default `1,2,3`, clamped to the block width).
#[test]
#[ignore = "needs the real Qwen 3.5 4B checkpoint, a GPU and a recorded transcript"]
fn round_loop_cache_dynamics_with_rollback_match_the_chain() {
    let Some((model, dir)) = load_text_model() else {
        return;
    };
    let text = text_model(&model);
    let Some(prompt) = env_ids("MLXCEL_Q35_PROBE_PROMPT") else {
        eprintln!("[1935] skipping: set MLXCEL_Q35_PROBE_PROMPT and MLXCEL_Q35_PROBE_REFERENCE");
        return;
    };
    let reference = env_ids("MLXCEL_Q35_PROBE_REFERENCE").expect("MLXCEL_Q35_PROBE_REFERENCE");
    let block_size: usize = std::env::var("MLXCEL_Q35_PROBE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let accepts: Vec<usize> = std::env::var("MLXCEL_Q35_PROBE_ACCEPTS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<usize>().expect("accept count"))
                .collect()
        })
        .unwrap_or_else(|| vec![1, 2, 3]);
    eprintln!(
        "[1935] target {dir}, block {block_size}, accept cycle {accepts:?}, \
         reference {} ids",
        reference.len()
    );

    let mut classic_caches = text.make_internal_caches();
    let classic_prefill = text.forward_internal(&ids_of(&prompt), None, &mut classic_caches, None);
    let mut classic: Vec<i32> = vec![argmax_row(&classic_prefill, prompt.len() as i32 - 1)];
    for token in &reference {
        let out = text.forward_internal(&ids_of(&[*token]), None, &mut classic_caches, None);
        classic.push(argmax_row(&out, 0));
    }

    let capture = probe_capture_layer_ids();
    eprintln!(
        "[1935] capture layers {capture:?}, block argmax {}",
        probe_block_argmax()
    );
    let mut burst_caches = text.make_speculative_caches();
    let burst_prefill =
        text.forward_prefill_with_capture_layers(&ids_of(&prompt), &mut burst_caches, &capture);
    let mut burst: Vec<i32> = vec![argmax_row(&burst_prefill.logits, prompt.len() as i32 - 1)];

    // A served round's rejected rows hold the drafter's WRONG proposals, not
    // the tokens that end up emitted. Their keys and values are written into
    // the attention caches and their projections into the gated-delta
    // snapshot before the rewind trims them, so a rewind that reads one row
    // too far shows up here and nowhere else. `MLXCEL_Q35_PROBE_WRONG` turns
    // that on; with it off the rejected rows carry the correct tokens, which
    // is the weaker test.
    let wrong: Option<i32> = std::env::var("MLXCEL_Q35_PROBE_WRONG")
        .ok()
        .filter(|v| v != "0")
        .map(|v| v.parse::<i32>().unwrap_or(9999));
    if let Some(w) = wrong {
        eprintln!("[1935] rejected rows carry the wrong token id {w}");
    }

    let mut i = 0usize;
    let mut round = 0usize;
    let mut rollbacks = 0usize;
    while i < reference.len() {
        let end = (i + block_size).min(reference.len());
        let rows = end - i;
        let keep_now = accepts[round % accepts.len()].clamp(1, rows);
        let mut fed: Vec<i32> = reference[i..end].to_vec();
        if let Some(w) = wrong {
            for slot in fed.iter_mut().skip(keep_now) {
                *slot = w;
            }
        }
        let out = text.forward_speculative(&ids_of(&fed), &mut burst_caches, &capture);
        let keep = keep_now;
        if probe_block_argmax() {
            let all = argmax_block(&out.logits);
            burst.extend(all.into_iter().take(keep));
        } else {
            for pos in 0..keep {
                burst.push(argmax_row(&out.logits, pos as i32));
            }
        }
        if keep < rows {
            // The round loop's own condition: rewind whenever the block was
            // not fully accepted. `accepted` is the index of the last kept row.
            text.rollback_speculative_cache(
                &mut burst_caches,
                &out.gdn_states,
                &[keep as i32 - 1],
                rows as i32,
            );
            rollbacks += 1;
        }
        i += keep;
        round += 1;
    }

    let first = classic
        .iter()
        .zip(&burst)
        .position(|(a, b)| a != b)
        .map(|i| i as i64)
        .unwrap_or(-1);
    let disagreements = classic.iter().zip(&burst).filter(|(a, b)| a != b).count();
    eprintln!(
        "[1935] {round} rounds, {rollbacks} rollbacks; greedy argmax over {} compared \
         positions: {disagreements} disagree, first at {first} (-1 means none)",
        classic.len().min(burst.len())
    );
    if first >= 0 {
        let i = first as usize;
        eprintln!(
            "[1935] at position {i}: classic {} against burst {}",
            classic[i], burst[i]
        );
    }
}
