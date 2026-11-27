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

//! Dense Qwen3 and Llama as DSpark targets (issue #3261), on tiny models.
//!
//! The drafter in the end-to-end cases is a real `DFlashDrafter` loaded from
//! a tiny checkpoint written to a temp directory in one of the two official
//! layouts, so the config parser, the loader, the pairing gate, the bind and
//! the round loop all run as they do for the published checkpoints.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use mlxcel_core::drafter::dflash::{DFlashDrafter, DFlashGenerator, SpeculativeTarget};
use mlxcel_core::drafter::{Drafter, DrafterError};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::layers::KVCache;
use mlxcel_core::prefill_plan::PrefillSplits;
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::*;

pub(crate) const VOCAB: i32 = 96;
pub(crate) const HIDDEN: i32 = 64;
const FF: i32 = 128;
const HEADS: i32 = 4;
const KV_HEADS: i32 = 2;
const HEAD_DIM: i32 = 64;
/// Depth of the tiny targets.
pub(crate) const LAYERS: usize = 4;
/// Layers the tiny drafters tap.
const TAPS: [usize; 2] = [1, 3];
const MASK_TOKEN: i32 = VOCAB - 1;

/// A varied, bounded fill, salted per tensor so no two weights repeat.
fn fill(shape: &[i32], salt: usize) -> Vec<f32> {
    let n: usize = shape.iter().product::<i32>() as usize;
    (0..n)
        .map(|i| ((((i + salt * 131) * 37 + 11) % 97) as f32 - 48.0) * 0.01)
        .collect()
}

/// Named `(shape, values)` tensors with norm weights near one.
fn tensors(mut shapes: Vec<(String, Vec<i32>)>) -> Vec<(String, Vec<i32>, Vec<f32>)> {
    shapes.sort();
    shapes
        .into_iter()
        .enumerate()
        .map(|(salt, (key, shape))| {
            let mut values = fill(&shape, salt);
            if key.contains("norm") {
                values.iter_mut().for_each(|v| *v += 1.0);
            }
            (key, shape, values)
        })
        .collect()
}

fn weight_map(tensors: &[(String, Vec<i32>, Vec<f32>)]) -> WeightMap {
    tensors
        .iter()
        .map(|(key, shape, values)| (key.clone(), mlxcel_core::from_slice_f32(values, shape)))
        .collect()
}

/// The attention planes, norms and MLP every tiny target shares; `qk_norm`
/// adds Qwen3's per-head Q/K norms.
fn target_shapes(qk_norm: bool) -> Vec<(String, Vec<i32>)> {
    let mut shapes = vec![
        ("model.embed_tokens.weight".to_owned(), vec![VOCAB, HIDDEN]),
        ("model.norm.weight".to_owned(), vec![HIDDEN]),
        ("lm_head.weight".to_owned(), vec![VOCAB, HIDDEN]),
    ];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let mut layer = vec![
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("self_attn.q_proj.weight", vec![HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.k_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.v_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, HEADS * HEAD_DIM]),
            ("mlp.gate_proj.weight", vec![FF, HIDDEN]),
            ("mlp.up_proj.weight", vec![FF, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, FF]),
        ];
        if qk_norm {
            layer.push(("self_attn.q_norm.weight", vec![HEAD_DIM]));
            layer.push(("self_attn.k_norm.weight", vec![HEAD_DIM]));
        }
        for (name, shape) in layer {
            shapes.push((format!("{p}.{name}"), shape));
        }
    }
    shapes
}

fn target_config(model_type: &str) -> String {
    format!(
        r#"{{"model_type": "{model_type}", "vocab_size": {VOCAB}, "hidden_size": {HIDDEN},
        "intermediate_size": {FF}, "num_hidden_layers": {LAYERS},
        "num_attention_heads": {HEADS}, "num_key_value_heads": {KV_HEADS},
        "head_dim": {HEAD_DIM}, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "tie_word_embeddings": false}}"#
    )
}

/// A four-layer untied Qwen3 with float weights.
pub(crate) fn tiny_qwen3() -> Qwen3Model {
    let args: crate::models::qwen3::ModelArgs =
        serde_json::from_str(&target_config("qwen3")).expect("qwen3 config");
    Qwen3Model::from_weights(&weight_map(&tensors(target_shapes(true))), &args).expect("tiny qwen3")
}

/// A four-layer untied Llama with float weights.
pub(crate) fn tiny_llama() -> Llama3Model {
    let args: crate::models::llama3::ModelArgs =
        serde_json::from_str(&target_config("llama")).expect("llama config");
    Llama3Model::from_weights(&weight_map(&tensors(target_shapes(false))), &args)
        .expect("tiny llama")
}

fn input(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn floats(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::eval(a);
    mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32))
}

fn argmax_rows(logits: &MlxArray) -> Vec<i32> {
    let shape = mlxcel_core::array_shape(logits);
    let values = floats(logits);
    let vocab = shape[2] as usize;
    values
        .chunks(vocab)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i as i32)
        })
        .collect()
}

/// Greedy decode through the plain `LanguageModel` forward, one token at a
/// time after a single-pass prefill: the classic reference.
fn classic_greedy(model: &dyn LanguageModel, prompt: &[i32], count: usize) -> Vec<i32> {
    let mut caches = model.make_caches();
    let logits = model.forward(&input(prompt), &mut caches, None);
    let mut next = *argmax_rows(&logits).last().expect("prompt logits");
    let mut out = vec![next];
    while out.len() < count {
        let logits = model.forward(&input(&[next]), &mut caches, None);
        next = argmax_rows(&logits)[0];
        out.push(next);
    }
    out
}

/// Write `tensors` (f32) and `ints` (i32) as one safetensors file.
fn write_safetensors(
    path: &Path,
    tensors: &[(String, Vec<i32>, Vec<f32>)],
    ints: &[(String, Vec<i32>)],
) {
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    let mut entry = |name: &str, dtype: &str, shape: &[i32], bytes: Vec<u8>, data: &mut Vec<u8>| {
        if header.len() > 1 {
            header.push(',');
        }
        let start = data.len();
        data.extend_from_slice(&bytes);
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    };
    for (name, shape, values) in tensors {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        entry(name, "F32", shape, bytes, &mut data);
    }
    for (name, values) in ints {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        entry(name, "I32", &[values.len() as i32], bytes, &mut data);
    }
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(path, file).expect("write safetensors");
}

/// Which official layout a tiny drafter checkpoint is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// Red Hat Speculators: nested backbone, sliding causal layers, own
    /// `embed_tokens` and `lm_head`, no `num_target_layers`.
    Speculators,
    /// OpenBMB: flat config, full attention, binds the target's table and
    /// head, declares `num_target_layers`.
    OpenBmb,
}

/// Knobs the cases below vary on a tiny drafter checkpoint.
#[derive(Clone)]
struct DrafterSpec {
    layout: Layout,
    taps: Vec<usize>,
    vocab: i32,
    /// `Some` writes a reduced-vocabulary head and this `d2t`.
    d2t: Option<Vec<i32>>,
    sliding_window: usize,
    block_size: usize,
}

impl DrafterSpec {
    fn new(layout: Layout) -> Self {
        Self {
            layout,
            taps: TAPS.to_vec(),
            vocab: VOCAB,
            d2t: None,
            sliding_window: 6,
            block_size: 3,
        }
    }
}

/// Write a two-layer DSpark drafter for the tiny targets into `dir`.
fn write_drafter(dir: &Path, spec: &DrafterSpec) {
    let draft_vocab = spec.d2t.as_ref().map_or(spec.vocab, |t| t.len() as i32);
    let rank = 8;
    let taps = spec.taps.len() as i32;
    let mut shapes = vec![
        ("fc.weight".to_owned(), vec![HIDDEN, taps * HIDDEN]),
        ("hidden_norm.weight".to_owned(), vec![HIDDEN]),
        ("norm.weight".to_owned(), vec![HIDDEN]),
        (
            "markov_head.markov_w1.weight".to_owned(),
            vec![spec.vocab, rank],
        ),
        (
            "markov_head.markov_w2.weight".to_owned(),
            vec![draft_vocab, rank],
        ),
        (
            "confidence_head.proj.weight".to_owned(),
            vec![1, HIDDEN + rank],
        ),
        ("confidence_head.proj.bias".to_owned(), vec![1]),
    ];
    if spec.layout == Layout::Speculators || spec.d2t.is_some() {
        shapes.push(("embed_tokens.weight".to_owned(), vec![spec.vocab, HIDDEN]));
        shapes.push(("lm_head.weight".to_owned(), vec![draft_vocab, HIDDEN]));
    }
    for l in 0..2 {
        for (name, shape) in [
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("self_attn.q_proj.weight", vec![HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.k_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.v_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, HEADS * HEAD_DIM]),
            ("self_attn.q_norm.weight", vec![HEAD_DIM]),
            ("self_attn.k_norm.weight", vec![HEAD_DIM]),
            ("mlp.gate_proj.weight", vec![FF, HIDDEN]),
            ("mlp.up_proj.weight", vec![FF, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, FF]),
        ] {
            shapes.push((format!("layers.{l}.{name}"), shape));
        }
    }
    let ints: Vec<(String, Vec<i32>)> = spec
        .d2t
        .iter()
        .map(|table| ("d2t".to_owned(), table.clone()))
        .collect();
    write_safetensors(&dir.join("model.safetensors"), &tensors(shapes), &ints);

    let backbone = serde_json::json!({
        "attention_bias": false, "head_dim": HEAD_DIM, "hidden_act": "silu",
        "hidden_size": HIDDEN, "intermediate_size": FF,
        "max_position_embeddings": 4096, "model_type": "qwen3",
        "num_attention_heads": HEADS, "num_hidden_layers": 2,
        "num_key_value_heads": KV_HEADS, "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "tie_word_embeddings": false, "vocab_size": spec.vocab,
    });
    let config = match spec.layout {
        Layout::Speculators => {
            let mut backbone = backbone;
            backbone["layer_types"] = serde_json::json!(["sliding_attention", "sliding_attention"]);
            backbone["sliding_window"] = serde_json::json!(spec.sliding_window);
            serde_json::json!({
                "architectures": ["DSparkDraftModel"],
                "speculators_model_type": "dspark",
                "aux_hidden_state_layer_ids": spec.taps,
                "block_size": spec.block_size,
                "draft_vocab_size": draft_vocab,
                "enable_confidence_head": true,
                "markov_head_type": "vanilla",
                "markov_rank": rank,
                "mask_token_id": MASK_TOKEN.min(spec.vocab - 1),
                "sample_from_anchor": true,
                "sliding_window_non_causal": false,
                "tie_word_embeddings": false,
                "transformer_layer_config": backbone,
            })
        }
        Layout::OpenBmb => {
            let mut config = backbone;
            config["architectures"] = serde_json::json!(["Qwen3DSparkModel"]);
            config["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
            config["sliding_window"] = serde_json::Value::Null;
            config["block_size"] = serde_json::json!(spec.block_size);
            config["draft_vocab_size"] = serde_json::json!(draft_vocab);
            config["num_target_layers"] = serde_json::json!(LAYERS);
            config["enable_confidence_head"] = serde_json::json!(true);
            config["markov_head_type"] = serde_json::json!("vanilla");
            config["markov_rank"] = serde_json::json!(rank);
            config["mask_token_id"] = serde_json::json!(MASK_TOKEN.min(spec.vocab - 1));
            config["projector_type"] = serde_json::json!("dspark");
            config["target_layer_ids"] = serde_json::json!(spec.taps);
            config
        }
    };
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .expect("write config.json");
}

fn load_drafter(spec: &DrafterSpec) -> (tempfile::TempDir, DFlashDrafter) {
    let dir = tempfile::tempdir().expect("temp dir");
    write_drafter(dir.path(), spec);
    let drafter = DFlashDrafter::load(dir.path()).expect("tiny DSpark drafter loads");
    (dir, drafter)
}

/// Drive the round loop the way the server burst and the offline driver do:
/// prefill with the drafter's taps captured, first token from the last
/// prompt row, every prompt row as the first draft context.
fn drafted<T: DenseSpeculativeTarget>(
    target: &T,
    spec: &DrafterSpec,
    prompt: &[i32],
    count: usize,
    width: usize,
    chunk: usize,
) -> (Vec<i32>, usize, usize) {
    let (_dir, drafter) = load_drafter(spec);
    drafter
        .validate_target_compat(target)
        .expect("the tiny pairing is compatible");
    assert!(target.dflash_drafter_pairing_error(&drafter).is_none());
    let taps = drafter.dflash_target_layer_ids().unwrap().to_vec();
    let mut caches = LanguageModel::make_caches(target);
    let out = target.dflash_prefill_planned(&input(prompt), &mut caches, &taps, chunk);
    let first = *argmax_rows(&out.logits).last().expect("prompt logits");
    let first_hidden = target.concat_hidden_for_drafter(&out);
    assert_eq!(
        mlxcel_core::array_shape(&first_hidden),
        vec![1, prompt.len() as i32, taps.len() as i32 * HIDDEN]
    );
    drop(out);
    let mut generator = DFlashGenerator::new(
        Box::new(drafter),
        SamplingConfig::greedy(),
        width as u32,
        mlxcel_core::drafter::dflash::round_loop::DEFAULT_MASK_TOKEN_ID,
    );
    let cancel = AtomicBool::new(false);
    let run = generator
        .run(
            target,
            target as &dyn LanguageModel,
            &mut caches,
            first,
            first_hidden,
            &[],
            count,
            &cancel,
            &LogprobsConfig::default(),
        )
        .expect("round loop");
    let mut tokens = vec![first];
    tokens.extend(run.tokens);
    (
        tokens,
        run.diagnostics.proposed_tokens,
        run.diagnostics.accepted_tokens,
    )
}

/// The capturing forward is the generation forward: same logits, and each
/// captured slab is the residual stream leaving that layer, before the
/// final norm.
#[test]
fn the_speculative_forward_returns_generation_logits_and_layer_outputs() {
    fn check<M: DenseSpeculativeBackbone>(model: &M) {
        let tokens = [3, 17, 42, 8, 60];
        let mut plain_caches = model.make_caches();
        let plain = model.forward(&input(&tokens), &mut plain_caches, None);

        let mut caches = model.make_caches();
        let out = forward_speculative(
            model,
            &input(&tokens),
            &mut caches,
            &[1, 3],
            Rows::VerifyBlock,
        );
        assert_eq!(mlxcel_core::array_shape(&out.logits), vec![1, 5, VOCAB]);
        // The verify scope may pick another causal dispatch than the plain
        // forward, so the comparison allows rounding.
        for (got, want) in floats(&out.logits).iter().zip(floats(&plain)) {
            assert!((got - want).abs() < 1e-3, "{}: {got} vs {want}", M::FAMILY);
        }
        assert_eq!(out.hidden_states.len(), 2);

        // `Rows::Last` is classic decode's prefill: one row of logits, the
        // same bytes `forward_last_logits` returns, and the same captures.
        let mut last_caches = model.make_caches();
        let last = forward_speculative(
            model,
            &input(&tokens),
            &mut last_caches,
            &[1, 3],
            Rows::Last,
        );
        assert_eq!(mlxcel_core::array_shape(&last.logits), vec![1, 1, VOCAB]);
        let mut classic_caches = model.make_caches();
        let classic = model.forward_last_logits(&input(&tokens), &mut classic_caches, None, 4);
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(&last.logits),
            mlxcel_core::array_to_raw_bytes(&classic),
            "{}: a prefill must sample its first token from classic decode's logits",
            M::FAMILY
        );
        assert_eq!(last.hidden_states.len(), 2);
        assert_eq!(
            mlxcel_core::array_shape(&last.hidden_states[0]),
            vec![1, 5, HIDDEN]
        );
        for slab in &out.hidden_states {
            assert_eq!(mlxcel_core::array_shape(slab), vec![1, 5, HIDDEN]);
        }
        // The last tap is the last layer: its slab, normed and projected,
        // is the logits.
        let mut tail_caches = model.make_caches();
        let mut captured = vec![None];
        let hidden = model.hidden_capturing(
            &input(&tokens),
            &mut tail_caches,
            &[LAYERS - 1],
            &mut captured,
        );
        let last = captured
            .pop()
            .flatten()
            .expect("the last layer is captured");
        assert_ne!(
            floats(&last),
            floats(&hidden),
            "captured before the final norm"
        );
        // The two slabs are different layers' outputs.
        assert_ne!(floats(&out.hidden_states[0]), floats(&out.hidden_states[1]));
        // No capture requested returns none.
        let mut bare_caches = model.make_caches();
        let bare = forward_speculative(model, &input(&tokens), &mut bare_caches, &[], Rows::Last);
        assert!(bare.hidden_states.is_empty());
    }
    check(&tiny_qwen3());
    check(&tiny_llama());
}

/// Tap boundaries: the last layer is captured, one past it is not (a zero
/// slab keeps the result aligned; the pairing gate refuses such a drafter
/// before any forward).
#[test]
fn a_tap_at_the_last_layer_is_captured_and_one_past_it_is_not() {
    let model = tiny_qwen3();
    let mut caches = model.make_caches();
    let out = forward_speculative(
        &model,
        &input(&[5, 9, 2]),
        &mut caches,
        &[LAYERS - 1, LAYERS],
        Rows::Last,
    );
    assert_eq!(out.hidden_states.len(), 2);
    assert!(floats(&out.hidden_states[0]).iter().any(|v| *v != 0.0));
    assert!(floats(&out.hidden_states[1]).iter().all(|v| *v == 0.0));
    assert_eq!(
        mlxcel_core::array_shape(&out.hidden_states[1]),
        vec![1, 3, HIDDEN]
    );
}

/// A partial accept trims exactly the rejected rows off every layer, and
/// the cache then continues as if only the kept rows had been forwarded.
#[test]
fn rollback_trims_the_rejected_rows() {
    fn check<M: DenseSpeculativeTarget + DenseSpeculativeBackbone>(model: &M) {
        let prompt = [3, 17, 42, 8];
        let block = [11, 23, 35, 47, 59];
        for accepted in 0..block.len() as i32 {
            let mut caches = LanguageModel::make_caches(model);
            let _ = forward_speculative(model, &input(&prompt), &mut caches, &[], Rows::Last);
            let out = model.verify_forward_with_capture_layers(&input(&block), &mut caches, &[1]);
            mlxcel_core::eval(&out.logits);
            assert!(caches.iter().all(|c| c.offset == 9));
            model.rollback_partial(&mut caches, &out, accepted, block.len() as i32);
            let kept = 4 + accepted + 1;
            assert!(
                caches.iter().all(|c| c.offset == kept),
                "{} accepted {accepted}",
                M::FAMILY
            );

            // The next token's logits equal those of a cache that only ever
            // saw the kept rows, up to the block-versus-chain rounding the
            // exactness probe measures.
            let next = forward_speculative(model, &input(&[70]), &mut caches, &[], Rows::Last);
            let mut clean = LanguageModel::make_caches(model);
            let _ = forward_speculative(model, &input(&prompt), &mut clean, &[], Rows::Last);
            let kept_block = &block[..=accepted as usize];
            let _ = forward_speculative(model, &input(kept_block), &mut clean, &[], Rows::Last);
            let want = forward_speculative(model, &input(&[70]), &mut clean, &[], Rows::Last);
            for (got, want) in floats(&next.logits).iter().zip(floats(&want.logits)) {
                assert!((got - want).abs() < 1e-3, "{}: {got} vs {want}", M::FAMILY);
            }
        }
        // A full accept trims nothing.
        let mut caches = LanguageModel::make_caches(model);
        let _ = forward_speculative(model, &input(&block), &mut caches, &[], Rows::VerifyBlock);
        rollback(&mut caches, 4, 5);
        assert!(caches.iter().all(|c| c.offset == 5));
        // Nor does an accept count past the block.
        rollback(&mut caches, 9, 5);
        assert!(caches.iter().all(|c| c.offset == 5));
    }
    check(&tiny_qwen3());
    check(&tiny_llama());
}

/// The planned prefill forwards classic decode's pieces: one forward for a
/// prompt within a chunk, several for a longer one, with the captures
/// concatenated back to one row per prompt token.
#[test]
fn the_planned_prefill_covers_every_prompt_row() {
    let model = tiny_llama();
    let prompt: Vec<i32> = (0..23).map(|i| (i * 7 + 3) % VOCAB).collect();
    let single = {
        let mut caches = model.make_caches();
        let out = prefill_planned(
            &model,
            &input(&prompt),
            &mut caches,
            &TAPS,
            0,
            PrefillSplits::default(),
        );
        assert!(caches.iter().all(|c| c.offset == 23));
        // One row of logits, however long the prompt: the head never
        // projects the whole prompt through the vocabulary.
        assert_eq!(mlxcel_core::array_shape(&out.logits), vec![1, 1, VOCAB]);
        out
    };
    for chunk in [1usize, 5, 22, 23, 24, 4096] {
        let mut caches = model.make_caches();
        let out = prefill_planned(
            &model,
            &input(&prompt),
            &mut caches,
            &TAPS,
            chunk,
            PrefillSplits::default(),
        );
        assert!(caches.iter().all(|c| c.offset == 23), "chunk {chunk}");
        assert_eq!(out.hidden_states.len(), 2);
        for (slab, want) in out.hidden_states.iter().zip(&single.hidden_states) {
            assert_eq!(mlxcel_core::array_shape(slab), vec![1, 23, HIDDEN]);
            for (got, want) in floats(slab).iter().zip(floats(want)) {
                assert!((got - want).abs() < 1e-3, "chunk {chunk}: {got} vs {want}");
            }
        }
        // The last row's logits pick the same first token.
        assert_eq!(mlxcel_core::array_shape(&out.logits), vec![1, 1, VOCAB]);
        assert_eq!(
            argmax_rows(&out.logits).last(),
            argmax_rows(&single.logits).last(),
            "chunk {chunk}"
        );
    }
    // A snapshot boundary splits the prompt too.
    let mut caches = model.make_caches();
    let out = prefill_planned(
        &model,
        &input(&prompt),
        &mut caches,
        &TAPS,
        4096,
        PrefillSplits::from_points(&[9]),
    );
    assert_eq!(
        mlxcel_core::array_shape(&out.hidden_states[0]),
        vec![1, 23, HIDDEN]
    );
    // A one-token prompt is one piece.
    let mut caches = model.make_caches();
    let out = prefill_planned(
        &model,
        &input(&[4]),
        &mut caches,
        &TAPS,
        2048,
        PrefillSplits::default(),
    );
    assert_eq!(
        mlxcel_core::array_shape(&out.hidden_states[0]),
        vec![1, 1, HIDDEN]
    );
}

/// The probe runs, on both targets, and refuses a width that drafts nothing.
/// Whether a block equals the chain is a property of the host's kernels, so
/// the verdict itself is not asserted here.
#[test]
fn the_exactness_probe_runs_on_both_targets() {
    let qwen3 = tiny_qwen3();
    let llama = tiny_llama();
    for width in [0, 1] {
        assert_eq!(
            probe_block_chain_exactness(&qwen3, width),
            BlockChainExactness::NotRun("block width below 2 drafts nothing")
        );
    }
    for width in [2, 4, 8] {
        for verdict in [
            probe_block_chain_exactness_at(&qwen3, width, &[8]),
            probe_block_chain_exactness_at(&llama, width, &[8]),
        ] {
            assert!(
                !matches!(verdict, BlockChainExactness::NotRun(_)),
                "width {width}: {verdict:?}"
            );
        }
    }
    assert_eq!(hidden_size(&qwen3), HIDDEN as usize);
    assert_eq!(logits_width(&llama), VOCAB as usize);
    assert_eq!(target_vocab_size(&qwen3), Some(VOCAB as usize));
    assert_eq!(target_vocab_size(&llama), Some(VOCAB as usize));
}

/// The target's half of the pairing gate, at its boundaries: a tap at the
/// last layer is admitted, a tap at the depth or past it is refused with the
/// index and the depth, and a vocabulary mismatch names both numbers.
#[test]
fn the_target_gate_refuses_a_tap_at_or_above_the_depth_by_index() {
    assert!(
        dense_target_pairing_error("Qwen3", 36, Some(151936), &[1, 9, 17, 25, 33], Some(151936))
            .is_none()
    );
    assert!(dense_target_pairing_error("Qwen3", 36, Some(151936), &[35], Some(151936)).is_none());
    let err = dense_target_pairing_error("Qwen3", 36, Some(151936), &[1, 36], Some(151936))
        .expect("a tap at the depth");
    assert!(err.contains("layer index 36"), "{err}");
    assert!(err.contains("36 layers"), "{err}");
    assert!(err.contains("0..=35"), "{err}");
    let err = dense_target_pairing_error("Llama", 42, None, &[1, 10, 20, 30, 39, 100], None)
        .expect("a tap past the depth");
    assert!(err.contains("layer index 100"), "{err}");
    assert!(err.contains("Llama"), "{err}");
    // Depth 0: every tap is past it, and the message does not underflow.
    let err = dense_target_pairing_error("Llama", 0, None, &[0], None).expect("empty target");
    assert!(err.contains("layer index 0"), "{err}");

    let err = dense_target_pairing_error("Qwen3", 36, Some(151936), &[1], Some(248320))
        .expect("vocabulary mismatch");
    assert!(err.contains("248320"), "{err}");
    assert!(err.contains("151936"), "{err}");
    // An unmeasured side is not guessed at.
    assert!(dense_target_pairing_error("Qwen3", 36, None, &[1], Some(248320)).is_none());
    assert!(dense_target_pairing_error("Qwen3", 36, Some(151936), &[1], None).is_none());
    // No taps at all is the drafter gate's error, not this one's.
    assert!(dense_target_pairing_error("Qwen3", 36, Some(151936), &[], Some(151936)).is_none());
}

/// Both targets admit a DSpark drafter in either official layout, bind it,
/// and refuse one whose taps or vocabulary do not fit them, through the
/// drafter's own gate and through the target's.
#[test]
fn both_targets_admit_a_dspark_drafter_and_refuse_a_mismatched_one() {
    let qwen3 = tiny_qwen3();
    let llama = tiny_llama();
    for layout in [Layout::Speculators, Layout::OpenBmb] {
        let (_dir, mut drafter) = load_drafter(&DrafterSpec::new(layout));
        assert!(drafter.is_dspark());
        assert_eq!(drafter.dflash_vocab_size(), Some(VOCAB as usize));
        for target in [&qwen3 as &dyn LanguageModel, &llama as &dyn LanguageModel] {
            drafter
                .validate_target_compat(target)
                .unwrap_or_else(|e| panic!("{layout:?}: {e}"));
            drafter
                .bind(target)
                .unwrap_or_else(|e| panic!("{layout:?}: {e}"));
        }
        assert!(qwen3.dflash_drafter_pairing_error(&drafter).is_none());
        assert!(llama.dflash_drafter_pairing_error(&drafter).is_none());
    }

    // A tap at the depth (index 4 of a 4-layer target), on the layout that
    // declares no depth of its own: nothing in the config can catch it, the
    // bound target does.
    let past = DrafterSpec {
        taps: vec![1, LAYERS],
        ..DrafterSpec::new(Layout::Speculators)
    };
    let (_dir, drafter) = load_drafter(&past);
    for target in [&qwen3 as &dyn LanguageModel, &llama as &dyn LanguageModel] {
        let Err(DrafterError::BindFailed { reason }) = drafter.validate_target_compat(target)
        else {
            panic!("a tap at the target depth must be refused");
        };
        assert!(reason.contains("layer index 4"), "{reason}");
        assert!(reason.contains("4 layers"), "{reason}");
    }
    let reason = qwen3
        .dflash_drafter_pairing_error(&drafter)
        .expect("target gate");
    assert!(reason.contains("layer index 4"), "{reason}");
    let reason = llama
        .dflash_drafter_pairing_error(&drafter)
        .expect("target gate");
    assert!(reason.contains("layer index 4"), "{reason}");

    // The last layer itself is a valid tap.
    let last = DrafterSpec {
        taps: vec![0, LAYERS - 1],
        ..DrafterSpec::new(Layout::Speculators)
    };
    let (_dir, drafter) = load_drafter(&last);
    drafter
        .validate_target_compat(&qwen3)
        .expect("a tap at the last layer");
    assert!(qwen3.dflash_drafter_pairing_error(&drafter).is_none());

    // A drafter for another vocabulary: both numbers are named.
    let other_vocab = DrafterSpec {
        vocab: VOCAB + 32,
        ..DrafterSpec::new(Layout::Speculators)
    };
    let (_dir, drafter) = load_drafter(&other_vocab);
    let Err(DrafterError::BindFailed { reason }) = drafter.validate_target_compat(&qwen3) else {
        panic!("a vocabulary mismatch must be refused");
    };
    assert!(reason.contains("128") && reason.contains("96"), "{reason}");
    let reason = llama
        .dflash_drafter_pairing_error(&drafter)
        .expect("target gate");
    assert!(reason.contains("128") && reason.contains("96"), "{reason}");

    // The OpenBMB layout declares its depth, so a drafter for a deeper
    // target is refused on the layer count.
    let dir = tempfile::tempdir().expect("temp dir");
    write_drafter(dir.path(), &DrafterSpec::new(Layout::OpenBmb));
    let path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["num_target_layers"] = serde_json::json!(42);
    std::fs::write(&path, config.to_string()).unwrap();
    let drafter = DFlashDrafter::load(dir.path()).expect("loads");
    let Err(DrafterError::BindFailed { reason }) = drafter.validate_target_compat(&llama) else {
        panic!("a layer-count mismatch must be refused");
    };
    assert!(reason.contains("num_target_layers = 42"), "{reason}");
    assert!(reason.contains("4 layers"), "{reason}");
}

/// Greedy text with the drafter equals greedy text without it, on both
/// targets and both layouts, at the narrowest and the default width, for a
/// prompt of one token, a prompt shorter than the drafter's sliding window
/// and one longer than it (and longer than a prefill chunk).
#[test]
fn drafted_greedy_tokens_equal_classic_greedy_tokens() {
    fn check<T: DenseSpeculativeTarget>(target: &T, family: &str) {
        let prompts: [Vec<i32>; 3] = [
            vec![7],
            vec![3, 17, 42],
            (0..40).map(|i| (i * 11 + 5) % VOCAB).collect(),
        ];
        for layout in [Layout::Speculators, Layout::OpenBmb] {
            let spec = DrafterSpec::new(layout);
            for prompt in &prompts {
                for width in [2usize, 4] {
                    let count = 24;
                    let classic = classic_greedy(target, prompt, count);
                    // Chunk 16 splits the 40-token prompt; the others are
                    // one piece.
                    let (tokens, proposed, _accepted) =
                        drafted(target, &spec, prompt, count, width, 16);
                    assert_eq!(
                        tokens,
                        classic,
                        "{family} {layout:?} prompt {} width {width}",
                        prompt.len()
                    );
                    assert!(proposed > 0, "the drafter must have proposed tokens");
                }
            }
        }
    }
    check(&tiny_qwen3(), "qwen3");
    check(&tiny_llama(), "llama");
}

/// A reduced draft vocabulary end to end: every proposal the target sees is
/// a target id from the `d2t` image, and the text is still the classic one.
#[test]
fn a_reduced_vocabulary_drafter_proposes_target_ids() {
    let target = tiny_qwen3();
    // 24 draft ids spread over the 96-id target vocabulary: draft id `t` is
    // target id `4 t + 1`.
    let d2t: Vec<i32> = (0..24).map(|t| 3 * t + 1).collect();
    let spec = DrafterSpec {
        d2t: Some(d2t.clone()),
        ..DrafterSpec::new(Layout::Speculators)
    };
    let (_dir, mut drafter) = load_drafter(&spec);
    drafter.validate_target_compat(&target).expect("compatible");
    drafter.bind(&target).expect("bind");
    let taps = drafter.dflash_target_layer_ids().unwrap().to_vec();
    let mut caches = LanguageModel::make_caches(&target);
    let out = target.dflash_prefill_planned(&input(&[3, 17, 42, 8]), &mut caches, &taps, 2048);
    let hidden = target.concat_hidden_for_drafter(&out);
    let proposals = drafter
        .draft_block(5, Some(&hidden), 4, &SamplingConfig::greedy())
        .expect("draft");
    assert_eq!(proposals.len(), 3);
    let image: Vec<i32> = d2t.iter().enumerate().map(|(t, o)| t as i32 + o).collect();
    for id in &proposals {
        assert!(image.contains(id), "{id} is not in the d2t image");
        assert!((0..VOCAB).contains(id));
    }

    let prompt = [3, 17, 42, 8];
    let classic = classic_greedy(&target, &prompt, 20);
    let (tokens, proposed, _) = drafted(&target, &spec, &prompt, 20, 4, 2048);
    assert_eq!(tokens, classic);
    assert!(proposed > 0);

    // A table that maps past the target vocabulary is refused when the
    // checkpoint loads, before any forward.
    let mut bad = d2t;
    bad[23] = VOCAB;
    let dir = tempfile::tempdir().expect("temp dir");
    write_drafter(
        dir.path(),
        &DrafterSpec {
            d2t: Some(bad),
            ..DrafterSpec::new(Layout::Speculators)
        },
    );
    let Err(err) = DFlashDrafter::load(dir.path()) else {
        panic!("an out-of-range d2t must be refused");
    };
    let err = err.to_string();
    assert!(err.contains("draft id 23"), "{err}");
}

/// A DSpark drafter is greedy-only on these targets as on LFM2: a sampling
/// request is an error from the drafter, not a silently greedy answer.
#[test]
fn a_sampling_request_is_refused_by_the_drafter() {
    let target = tiny_llama();
    let (_dir, mut drafter) = load_drafter(&DrafterSpec::new(Layout::OpenBmb));
    drafter.bind(&target).expect("bind");
    assert!(drafter.greedy_only());
    let taps = drafter.dflash_target_layer_ids().unwrap().to_vec();
    let mut caches: Vec<KVCache> = LanguageModel::make_caches(&target);
    let out = target.dflash_prefill_planned(&input(&[3, 17]), &mut caches, &taps, 2048);
    let hidden = target.concat_hidden_for_drafter(&out);
    // Greedy means temperature 0 or top_k 1, so a sampling request moves
    // both off it.
    let sampling = SamplingConfig {
        temperature: 0.8,
        top_k: 40,
        ..SamplingConfig::greedy()
    };
    assert!(!mlxcel_core::drafter::dflash::drafter::sampler_is_greedy(
        &sampling
    ));
    assert!(matches!(
        drafter.draft_block(5, Some(&hidden), 4, &sampling),
        Err(DrafterError::GreedyOnly { .. })
    ));
    // Block size 1 drafts nothing and is refused rather than sliced.
    assert!(
        drafter
            .draft_block(5, Some(&hidden), 1, &SamplingConfig::greedy())
            .is_err()
    );
}
