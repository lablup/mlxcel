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

use super::{Qwen35Model, Qwen3NextCache};
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
    for layer in 0..layers {
        let block_row = row_bytes(&out.hidden_states[layer], 0);
        let chain_row = &chain_hidden[0][layer];
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
