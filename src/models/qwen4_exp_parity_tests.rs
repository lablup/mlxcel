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

//! Parity against mlx-vlm's `qwen4_exp` on tiny float32 weights.
//!
//! `tests/fixtures/qwen4_exp/generate_fixtures.py` records upstream outputs.
//! The weights and inputs are not stored: [`hashed`] regenerates them from
//! their names with the generator's counter hash, bit for bit.

use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use super::Qwen4ExpModel;
use super::config::{Qwen4ExpConfig, ResolvedQwen4ExpConfig};
use super::gated_delta::{Qwen4ExpGatedDeltaNet, normalize_qk};
use super::hyper::GatedResidual;
use super::norm::ZeroCenteredRmsNorm;
use super::ple::{NgramHasher, PleCache, PleLayer};
use super::qsa::{QsaIndexCache, QsaIndexer, RopeContext};
use crate::models::gated_delta::GatedDeltaCache;
use crate::models::qwen3_vl::InterleavedMRoPE;
use mlxcel_core::generate::LanguageModel;

pub(super) fn fixture(name: &str) -> Value {
    let path = format!(
        "{}/tests/fixtures/qwen4_exp/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn fnv1a64(name: &str) -> u64 {
    name.bytes().fold(0xCBF2_9CE4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
    })
}

/// `generate_fixtures.hashed_tensor`, element for element.
pub(super) fn hashed(name: &str, shape: &[i32], scale: f64, offset: f64) -> UniquePtr<MlxArray> {
    let count: i64 = shape.iter().map(|&d| i64::from(d)).product();
    let seed = fnv1a64(name);
    let (scale, offset) = (scale as f32, offset as f32);
    let values: Vec<f32> = (1..=count as u64)
        .map(|index| {
            let mut z = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let unit = (z >> 40) as f32 * 2f32.powi(-23) - 1.0;
            unit * scale + offset
        })
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

pub(super) fn shape_of(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("shape")
        .iter()
        .map(|d| d.as_i64().expect("dim") as i32)
        .collect()
}

fn hashed_input(spec: &Value) -> UniquePtr<MlxArray> {
    hashed(
        spec["name"].as_str().expect("name"),
        &shape_of(&spec["shape"]),
        spec["scale"].as_f64().expect("scale"),
        0.0,
    )
}

pub(super) fn weights(fixture: &Value) -> WeightMap {
    let mut map = WeightMap::new();
    for spec in fixture["tensors"].as_array().expect("tensors") {
        let name = spec["name"].as_str().expect("name");
        let tensor = hashed(
            name,
            &shape_of(&spec["shape"]),
            spec["scale"].as_f64().expect("scale"),
            spec["offset"].as_f64().expect("offset"),
        );
        map.insert(name.to_string(), tensor);
    }
    map
}

pub(super) fn config(fixture: &Value, overrides: Option<&Value>) -> ResolvedQwen4ExpConfig {
    let mut raw = fixture["tiny_config"].clone();
    if let Some(overrides) = overrides.and_then(Value::as_object) {
        for (key, value) in overrides {
            raw[key] = value.clone();
        }
    }
    let config: Qwen4ExpConfig = serde_json::from_value(raw).expect("tiny config");
    config.resolve().expect("tiny config resolves")
}

fn values(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::array_to_raw_bytes(&array)
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
        .collect()
}

fn expected(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .expect("float list")
        .iter()
        .map(|v| v.as_f64().expect("float") as f32)
        .collect()
}

fn ints(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .expect("int list")
        .iter()
        .map(|v| v.as_i64().expect("int"))
        .collect()
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

/// Float32 on both sides: the tolerance is relative to the largest upstream
/// magnitude, so rows of tiny activations are held to the same standard.
const RTOL: f32 = 1e-5;

#[track_caller]
fn assert_close(label: &str, actual: &MlxArray, expected: &[f32]) {
    let actual = values(actual);
    let diff = max_abs_diff(&actual, expected);
    let scale = expected.iter().fold(0.0f32, |max, v| max.max(v.abs()));
    assert!(
        diff <= RTOL * scale,
        "{label}: max |actual - upstream| = {diff:e} exceeds {RTOL:e} x max |upstream| ({scale:e})"
    );
}

fn text_rope<'a>(mrope: &'a InterleavedMRoPE, batch: i32, start: i32, len: i32) -> RopeContext<'a> {
    let positions = mlxcel_core::arange_i32(start, start + len, 1);
    let positions = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&positions, &[1, 1, len]),
        &[3, batch, len],
    );
    RopeContext::new(mrope, positions)
}

#[test]
fn zero_centered_and_grouped_rms_norm_match_upstream() {
    let fixture = fixture("norms");
    let weights = weights(&fixture);
    let x = hashed_input(&fixture["input"]);
    let plain = ZeroCenteredRmsNorm::from_weights(&weights, "plain", 32, None, 1e-6).unwrap();
    let grouped =
        ZeroCenteredRmsNorm::from_weights(&weights, "grouped", 32, Some(8), 1e-6).unwrap();
    assert_close("plain", &plain.forward(&x), &expected(&fixture["plain"]));
    assert_close(
        "grouped",
        &grouped.forward(&x),
        &expected(&fixture["grouped"]),
    );
}

#[test]
fn gated_delta_net_matches_upstream_across_a_split_prefill() {
    let fixture = fixture("gated_delta");
    let config = config(&fixture, None);
    let gdn = Qwen4ExpGatedDeltaNet::from_weights(&weights(&fixture), &config, "gdn").unwrap();
    let x = hashed_input(&fixture["input"]);
    let split = fixture["split"].as_i64().unwrap() as i32;
    let total = shape_of(&fixture["input"]["shape"])[1];
    let width = config.raw.hidden_size as i32;
    let mut cache = GatedDeltaCache::new();
    let first = gdn.forward(
        &mlxcel_core::slice(&x, &[0, 0, 0], &[1, split, width]),
        Some(&mut cache),
    );
    let second = gdn.forward(
        &mlxcel_core::slice(&x, &[0, split, 0], &[1, total, width]),
        Some(&mut cache),
    );
    let output = mlxcel_core::concatenate(&first, &second, 1);
    assert_close("split prefill", &output, &expected(&fixture["output"]));

    let small = hashed_input(&fixture["small_input"]);
    let small_out = gdn.forward(&small, Some(&mut GatedDeltaCache::new()));
    assert_close(
        "near-epsilon rows",
        &small_out,
        &expected(&fixture["small_output"]),
    );
}

/// The float16 q/k normalization reproduces upstream bit for bit, and
/// Qwen3.5's fused normalization, mathematically the same function, does not.
#[test]
fn qk_normalization_matches_upstream_where_qwen35_normalization_does_not() {
    let fixture = fixture("gated_delta");
    let spec = &fixture["normalization"];
    let shape = shape_of(&spec["shape"]);
    let scale = spec["scale"].as_f64().unwrap();
    let q = mlxcel_core::astype(&hashed("input.norm_q", &shape, scale, 0.0), dtype::FLOAT16);
    let k = mlxcel_core::astype(&hashed("input.norm_k", &shape, scale, 0.0), dtype::FLOAT16);
    let (expected_q, expected_k) = (expected(&spec["q"]), expected(&spec["k"]));

    let (nq, nk) = normalize_qk(&q, &k);
    assert_eq!(values(&nq), expected_q, "q");
    assert_eq!(values(&nk), expected_k, "k");

    let head_dim = *shape.last().unwrap() as f32;
    let (q35, k35) = crate::models::gated_delta::normalize_qk(
        &q,
        &k,
        head_dim.powf(-0.5),
        crate::models::gated_delta::QK_L2NORM_EPS,
    );
    let q_gap = max_abs_diff(&values(&q35), &expected_q);
    let k_gap = max_abs_diff(&values(&k35), &expected_k);
    assert!(
        q_gap > 1e-3 && k_gap > 1e-3,
        "Qwen3.5 normalization is within {q_gap:e} / {k_gap:e} of upstream Qwen4"
    );
}

#[test]
fn gated_residual_matches_upstream_for_one_layer() {
    let fixture = fixture("hyper_connection");
    let config = config(&fixture, None);
    let weights = weights(&fixture);
    let residual = GatedResidual::from_weights(&weights, &config, "residual", true).unwrap();
    let mixer = GatedResidual::from_weights(&weights, &config, "mixer", false).unwrap();
    let h = hashed_input(&fixture["input"]);
    let branch = hashed_input(&fixture["branch"]);

    let (mixed, inject) = residual.split(&h);
    assert_close("mixed", &mixed, &expected(&fixture["mixed"]));
    assert_close("inject", &inject, &expected(&fixture["inject"]));
    let out = residual.inject(&branch, &h, &inject);
    assert_close("injected stream", &out, &expected(&fixture["output"]));
    assert_close("mixer", &mixer.collapse(&h), &expected(&fixture["mixer"]));
}

#[test]
fn ple_ngram_ids_match_upstream_across_eos_tokens() {
    let fixture = fixture("ple");
    let config = config(&fixture, None);
    let hasher = NgramHasher::new(&config, 0, u64::MAX).unwrap();
    assert_eq!(hasher.multipliers, ints(&fixture["layer_multipliers"]));
    assert_eq!(hasher.head_vocab_sizes, ints(&fixture["vocab_sizes"]));

    let tokens = ints(&fixture["tokens"]);
    let (ids, _) = hasher.ids(&hasher.initial_context(), &tokens);
    assert_eq!(ids, ints(&fixture["ngram_ids"]));

    let split = fixture["split"].as_u64().unwrap() as usize;
    let (first, context) = hasher.ids(&hasher.initial_context(), &tokens[..split]);
    let (second, _) = hasher.ids(&context, &tokens[split..]);
    assert_eq!(
        [first, second].concat(),
        ids,
        "chunked ids differ from one-shot ids"
    );
}

#[test]
fn released_checkpoint_ngram_hash_matches_upstream() {
    let fixture = fixture("ple");
    let released = &fixture["released"];
    let overrides = serde_json::json!({
        "vocab_size": released["config"]["vocab_size"],
        "ngram_size": released["config"]["ngram_size"],
        "heads_per_ngram": released["config"]["heads_per_ngram"],
        "ngram_vocab_size_base": released["config"]["ngram_vocab_size_base"],
        "make_ngram_vocab_size_divisible_by": released["config"]["make_ngram_vocab_size_divisible_by"],
        "seed": released["config"]["seed"],
        "eos_token_id": released["config"]["eos_token_id"],
        "ple_embed_dim": 128,
    });
    let config = config(&fixture, Some(&overrides));
    let hasher = NgramHasher::new(&config, 0, u64::MAX).unwrap();
    assert_eq!(hasher.multipliers, ints(&released["layer_multipliers"]));
    assert_eq!(hasher.head_vocab_sizes, ints(&released["vocab_sizes"]));
    assert_eq!(hasher.head_offsets, ints(&released["offsets"]));
    let (ids, _) = hasher.ids(&hasher.initial_context(), &ints(&released["tokens"]));
    assert_eq!(ids, ints(&released["ngram_ids"]));
}

#[test]
fn ple_layer_matches_upstream_and_split_prefill_equals_one_shot() {
    let fixture = fixture("ple");
    let config = config(&fixture, None);
    let layer = PleLayer::from_weights(&weights(&fixture), &config, "ple", 0).unwrap();
    let tokens = ints(&fixture["tokens"]);
    let h = hashed_input(&fixture["input"]);
    let shape = shape_of(&fixture["input"]["shape"]);
    let split = fixture["split"].as_i64().unwrap() as i32;

    let one_shot = layer.forward(&h, std::slice::from_ref(&tokens), &mut PleCache::default());
    assert_close("one-shot", &one_shot, &expected(&fixture["output"]));

    let mut cache = PleCache::default();
    let first = layer.forward(
        &mlxcel_core::slice(&h, &[0, 0, 0], &[1, split, shape[2]]),
        &[tokens[..split as usize].to_vec()],
        &mut cache,
    );
    let second = layer.forward(
        &mlxcel_core::slice(&h, &[0, split, 0], &[1, shape[1], shape[2]]),
        &[tokens[split as usize..].to_vec()],
        &mut cache,
    );
    let chunked = mlxcel_core::concatenate(&first, &second, 1);
    assert_close("two-chunk prefill", &chunked, &expected(&fixture["output"]));
    assert_close("two-chunk vs one-shot", &chunked, &values(&one_shot));
}

fn block_set(selection: &super::qsa::QsaSelection, row: i32) -> Vec<i64> {
    let shape = mlxcel_core::array_shape(&selection.selected);
    let row = mlxcel_core::slice(&selection.selected, &[0, row, 0], &[1, row + 1, shape[2]]);
    let mut blocks: Vec<i64> = values(&row)
        .into_iter()
        .map(|v| v as i64)
        .filter(|&b| b >= 0)
        .collect();
    blocks.sort_unstable();
    blocks
}

/// Runs the indexer over `prefill` tokens, then decodes one token at a time.
fn run_indexer(
    indexer: &QsaIndexer,
    mrope: &InterleavedMRoPE,
    hidden: &MlxArray,
    prefill: i32,
    decode_steps: i32,
) -> (
    Option<super::qsa::QsaSelection>,
    Vec<Option<super::qsa::QsaSelection>>,
) {
    let width = mlxcel_core::array_shape(hidden)[2];
    let mut cache = QsaIndexCache::default();
    let prompt = mlxcel_core::slice(hidden, &[0, 0, 0], &[1, prefill, width]);
    let prefill_selection = indexer.select(&prompt, &text_rope(mrope, 1, 0, prefill), &mut cache);
    let decode = (prefill..prefill + decode_steps)
        .map(|pos| {
            let token = mlxcel_core::slice(hidden, &[0, pos, 0], &[1, pos + 1, width]);
            indexer.select(&token, &text_rope(mrope, 1, pos, 1), &mut cache)
        })
        .collect();
    (prefill_selection, decode)
}

#[test]
fn qsa_selects_the_upstream_blocks_on_a_4096_token_sequence() {
    let fixture = fixture("qsa");
    let config = config(&fixture, Some(&fixture["config_overrides"]));
    let indexer = QsaIndexer::from_weights(&weights(&fixture), &config, "indexer").unwrap();
    let mrope = InterleavedMRoPE::new(
        config.rope_dims(),
        config.rope.rope_theta,
        config.rope.mrope_section.clone(),
    );
    let hidden = hashed_input(&fixture["input"]);
    let sparse = &fixture["sparse"];
    let prefill = sparse["prefill"].as_i64().unwrap() as i32;

    let (prefill_selection, decode) = run_indexer(&indexer, &mrope, &hidden, prefill, 1);
    let prefill_selection = prefill_selection.expect("4095 keys exceed the 512-block budget");
    for (row, blocks) in sparse["prefill_rows"].as_object().unwrap() {
        let row: i32 = row.parse().unwrap();
        assert_eq!(
            block_set(&prefill_selection, row),
            ints(blocks),
            "prefill row {row}"
        );
    }
    let decode = decode[0].as_ref().expect("decode at 4096 keys is sparse");
    assert_eq!(block_set(decode, 0), ints(&sparse["decode_blocks"]));
}

#[test]
fn qsa_attends_densely_until_the_budget_is_exceeded() {
    let fixture = fixture("qsa");
    let config = config(&fixture, Some(&fixture["config_overrides"]));
    let indexer = QsaIndexer::from_weights(&weights(&fixture), &config, "indexer").unwrap();
    let mrope = InterleavedMRoPE::new(
        config.rope_dims(),
        config.rope.rope_theta,
        config.rope.mrope_section.clone(),
    );
    let hidden = hashed_input(&fixture["input"]);
    let dense = &fixture["dense_fallback"];
    let prefill = dense["prefill"].as_i64().unwrap() as i32;
    let expected_selected: Vec<bool> = dense["selected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();

    let (prefill_selection, decode) = run_indexer(
        &indexer,
        &mrope,
        &hidden,
        prefill,
        expected_selected.len() as i32,
    );
    assert!(
        prefill_selection.is_none(),
        "2047 keys stay within the budget"
    );
    let selected: Vec<bool> = decode.iter().map(Option::is_some).collect();
    assert_eq!(selected, expected_selected);
    let last = decode.last().unwrap().as_ref().unwrap();
    assert_eq!(block_set(last, 0), ints(&dense["last_blocks"]));
}

#[test]
fn tiny_model_matches_upstream_logits_for_prefill_and_eight_decode_steps() {
    let fixture = fixture("model");
    let model = tiny_model(&fixture);

    let mut caches = model.make_internal_caches();
    let hidden = model.forward_hidden(&prompt_ids(&fixture), None, &mut caches, None);
    assert_close(
        "prefill logits",
        &model.logits(&hidden),
        &expected(&fixture["prefill_logits"]),
    );

    let steps = fixture["decode_tokens"].as_array().unwrap();
    for (step, (token, logits)) in steps
        .iter()
        .zip(fixture["decode_logits"].as_array().unwrap())
        .enumerate()
    {
        let token = mlxcel_core::from_slice_i32(&[token.as_i64().unwrap() as i32], &[1, 1]);
        let hidden = model.forward_hidden(&token, None, &mut caches, None);
        assert_close(
            &format!("decode step {step}"),
            &model.logits(&hidden),
            &expected(logits),
        );
    }
}

pub(super) fn tiny_model(fixture: &Value) -> Qwen4ExpModel {
    let config: Qwen4ExpConfig = serde_json::from_value(fixture["config"].clone()).unwrap();
    Qwen4ExpModel::from_weights(&weights(fixture), &config.resolve().unwrap()).unwrap()
}

pub(super) fn prompt_ids(fixture: &Value) -> UniquePtr<MlxArray> {
    let prompt: Vec<i32> = ints(&fixture["prompt"])
        .into_iter()
        .map(|t| t as i32)
        .collect();
    mlxcel_core::from_slice_i32(&prompt, &[1, prompt.len() as i32])
}

/// Chunk boundaries inside a QSA block, on a block edge, and splitting the
/// prompt before and after the index budget is first exceeded.
#[test]
fn tiny_model_chunked_prefill_matches_upstream_one_shot_logits() {
    let fixture = fixture("model");
    let model = tiny_model(&fixture);
    let ids = prompt_ids(&fixture);
    for chunks in [&[17][..], &[8, 13], &[1, 1, 30]] {
        let logits = model.forward_fresh(&ids, chunks);
        assert_close(
            &format!("chunks {chunks:?}"),
            &logits,
            &expected(&fixture["prefill_logits"]),
        );
    }
}

/// The scheduler-facing path: model-owned sequence state, last-position
/// logits, and text positions resolved per call.
#[test]
fn language_model_trait_path_matches_upstream() {
    let fixture = fixture("model");
    let model = tiny_model(&fixture);
    let ids = prompt_ids(&fixture);
    let prompt_len = mlxcel_core::array_shape(&ids)[1] as usize;
    let vocab = model.config().raw.vocab_size;
    let prefill = expected(&fixture["prefill_logits"]);
    let seq_id = mlxcel_core::cache::SequenceId::from_raw(7);

    model.prepare_sequence_state(seq_id);
    let last = model.forward_last_logits_with_sequence_id(
        &ids,
        Some(seq_id),
        &mut [],
        None,
        prompt_len - 1,
    );
    assert_close(
        "last prefill row",
        &last,
        &prefill[(prompt_len - 1) * vocab..],
    );
    for (step, (token, logits)) in fixture["decode_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .zip(fixture["decode_logits"].as_array().unwrap())
        .enumerate()
    {
        let token = mlxcel_core::from_slice_i32(&[token.as_i64().unwrap() as i32], &[1, 1]);
        let out = model.forward_with_sequence_id(&token, Some(seq_id), &mut [], None);
        assert_close(&format!("decode step {step}"), &out, &expected(logits));
    }
    model.release_sequence_state_by_id(seq_id);
}

/// A stored `[3, B, L]` position tensor (what a multimodal prefill leaves)
/// is sliced for the prompt and continued from `rope_deltas` while decoding.
/// Plain text positions stored this way must give the text logits.
#[test]
fn stored_mrope_positions_drive_prefill_and_decode() {
    let fixture = fixture("model");
    let model = tiny_model(&fixture);
    let ids = prompt_ids(&fixture);
    let len = mlxcel_core::array_shape(&ids)[1];
    let positions = mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&mlxcel_core::arange_i32(0, len, 1), &[1, 1, len]),
        &[3, 1, len],
    );
    model.reset_runtime_state();
    model.set_mrope_state(positions, 0);

    let logits = model.forward(&ids, &mut [], None);
    assert_close("prefill", &logits, &expected(&fixture["prefill_logits"]));
    let token = fixture["decode_tokens"][0].as_i64().unwrap() as i32;
    let out = model.forward(
        &mlxcel_core::from_slice_i32(&[token], &[1, 1]),
        &mut [],
        None,
    );
    assert_close(
        "decode step 0",
        &out,
        &expected(&fixture["decode_logits"][0]),
    );
}

/// Declared width of each quantized module in the mixed-width tiny model, in
/// the released quant's pattern: routed experts alternate 2- and 4-bit by
/// layer, hyper-connections are 8-bit, everything else takes the 4-bit
/// default. `None` leaves the tensor in float32: the router and shared-expert
/// gate (bf16 in the release), and inputs narrower than one 32-wide group.
fn mixed_width(name: &str, shape: &[i32]) -> Option<i32> {
    let module = name.strip_suffix(".weight")?;
    let float_only = module.ends_with(".mlp.gate")
        || module.ends_with(".shared_expert_gate")
        || module.contains("norm")
        || module.contains("conv1d");
    if float_only || shape.len() < 2 || shape.last()? % 32 != 0 {
        return None;
    }
    if module.contains(".switch_mlp.") {
        let layer: usize = module.split('.').nth(2)?.parse().ok()?;
        return Some(if layer.is_multiple_of(2) { 2 } else { 4 });
    }
    if module.contains("hyper_connection") {
        return Some(8);
    }
    Some(4)
}

#[test]
fn mixed_width_quantization_loads_each_module_at_its_declared_width() {
    let fixture = fixture("model");
    let mut quantized = WeightMap::new();
    let mut dequantized = WeightMap::new();
    let mut overrides = serde_json::Map::new();
    for (name, tensor) in weights(&fixture) {
        let shape = mlxcel_core::array_shape(&tensor);
        let Some(bits) = mixed_width(&name, &shape) else {
            dequantized.insert(name.clone(), mlxcel_core::copy(&tensor));
            quantized.insert(name, tensor);
            continue;
        };
        let module = name.strip_suffix(".weight").unwrap().to_string();
        let packed = mlxcel_core::quantize_weights(&tensor, 32, bits);
        let (w, scales, biases) = (
            mlxcel_core::quantized_weights_w(&packed),
            mlxcel_core::quantized_weights_scales(&packed),
            mlxcel_core::quantized_weights_biases(&packed),
        );
        let restored =
            unsafe { mlxcel_core::dequantize(&w, &scales, &*biases, 32, bits, "affine") };
        dequantized.insert(name, restored);
        quantized.insert(format!("{module}.weight"), w);
        quantized.insert(format!("{module}.scales"), scales);
        quantized.insert(format!("{module}.biases"), biases);
        overrides.insert(module, serde_json::json!({"group_size": 32, "bits": bits}));
    }
    // The stacked experts share one declared pair per layer even where a
    // projection is too narrow to pack.
    for layer in 0..4usize {
        let bits = if layer.is_multiple_of(2) { 2 } else { 4 };
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            overrides.insert(
                format!("model.layers.{layer}.mlp.switch_mlp.{projection}"),
                serde_json::json!({"group_size": 32, "bits": bits}),
            );
        }
    }
    let mut quantization = serde_json::json!({"group_size": 32, "bits": 4, "mode": "affine"});
    quantization.as_object_mut().unwrap().extend(overrides);

    let mut raw = fixture["config"].clone();
    let reference = Qwen4ExpModel::from_weights(
        &dequantized,
        &serde_json::from_value::<Qwen4ExpConfig>(raw.clone())
            .unwrap()
            .resolve()
            .unwrap(),
    )
    .unwrap();
    raw["quantization"] = quantization;
    let model = Qwen4ExpModel::from_weights(
        &quantized,
        &serde_json::from_value::<Qwen4ExpConfig>(raw)
            .unwrap()
            .resolve()
            .unwrap(),
    )
    .unwrap();

    let ids = prompt_ids(&fixture);
    let want = values(&reference.forward_fresh(&ids, &[]));
    assert_close(
        "mixed-width prefill",
        &model.forward_fresh(&ids, &[]),
        &want,
    );

    // Declaring the layer-0 experts at the 4-bit default fails the load
    // instead of aborting in `gather_qmm`.
    let mut misdeclared = fixture["config"].clone();
    misdeclared["quantization"] = serde_json::json!({"group_size": 32, "bits": 4});
    let config = serde_json::from_value::<Qwen4ExpConfig>(misdeclared)
        .unwrap()
        .resolve()
        .unwrap();
    let error = Qwen4ExpModel::from_weights(&quantized, &config)
        .err()
        .expect("misdeclared experts");
    assert!(
        error.contains("model.layers.0.mlp.switch_mlp.gate_proj"),
        "{error}"
    );
}

/// The image path end to end: sanitize the full checkpoint layout, encode the
/// image with the Qwen3-VL tower (no deepstack), scatter the features over
/// the image tokens, and place everything after the image at its MRoPE
/// positions, for prefill and while decoding.
#[test]
fn tiny_vlm_matches_upstream_on_an_image_prompt() {
    use crate::vision::encoders::qwen3_vl::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};
    use crate::vision::processors::qwen2_vl::Qwen2VLProcessor;

    let fixture = fixture("vlm");
    let config: Qwen4ExpConfig = serde_json::from_value(fixture["tiny_config"].clone()).unwrap();
    let config = config.resolve().unwrap();
    let sanitized = super::sanitize_weights(weights(&fixture), &config).unwrap();
    let vision_config: Qwen3VLVisionConfig =
        serde_json::from_value(fixture["vision_config"].clone()).unwrap();
    let grid = shape_of(&fixture["image_grid"]);
    let token = |name: &str| fixture["token_ids"][name].as_i64().unwrap() as i32;
    let vlm = crate::vision::Qwen4ExpVLModel {
        text_model: Qwen4ExpModel::from_weights(&sanitized.text, &config).unwrap(),
        vision_encoder: Qwen3VLVisionEncoder::from_weights(
            &sanitized.vision,
            &vision_config,
            "vision_tower",
        )
        .unwrap(),
        processor: Qwen2VLProcessor::new_with_norm(
            vision_config.patch_size,
            vision_config.temporal_patch_size,
            vision_config.spatial_merge_size,
            [0.5; 3],
            [0.5; 3],
        ),
        image_token_id: token("image"),
        video_token_id: token("video"),
        vision_start_token_id: token("vision_start"),
        spatial_merge_size: vision_config.spatial_merge_size,
    };

    let ids = prompt_ids(&fixture);
    // Upstream flattens each patch channel-first, `[N, (C, T, P, P)]`; the
    // mlxcel processor emits one row per temporal slice, `[N * T, (C, P, P)]`.
    let (channels, temporal, patch) = (
        vision_config.in_channels as i32,
        vision_config.temporal_patch_size as i32,
        vision_config.patch_size as i32,
    );
    let pixels = hashed_input(&fixture["pixels"]);
    let patches = mlxcel_core::array_shape(&pixels)[0];
    let pixels = mlxcel_core::reshape(&pixels, &[patches, channels, temporal, patch, patch]);
    let pixels = mlxcel_core::transpose_axes(&pixels, &[0, 2, 1, 3, 4]);
    let pixels = mlxcel_core::reshape(&pixels, &[patches * temporal, channels * patch * patch]);
    vlm.reset_runtime_state();
    let embeddings = vlm.get_input_embeddings(&ids, &pixels, &[(grid[0], grid[1], grid[2])]);
    let logits = vlm.forward_with_embeddings(&ids, Some(&embeddings.inputs_embeds), &mut [], None);
    assert_close(
        "image prefill",
        &logits,
        &expected(&fixture["prefill_logits"]),
    );
    for (step, (token, logits)) in fixture["decode_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .zip(fixture["decode_logits"].as_array().unwrap())
        .enumerate()
    {
        let token = mlxcel_core::from_slice_i32(&[token.as_i64().unwrap() as i32], &[1, 1]);
        let out = vlm.forward(&token, &mut [], None);
        assert_close(&format!("decode step {step}"), &out, &expected(logits));
    }
}

/// Anonymous batch rows must retain separate histories across prefill and decode.
#[test]
fn anonymous_batch_matches_independent_sequences() {
    let fixture = fixture("model");
    let batched = tiny_model(&fixture);
    let singles = [tiny_model(&fixture), tiny_model(&fixture)];
    let prompt: Vec<i32> = ints(&fixture["prompt"])
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let mut other = prompt.clone();
    other.reverse();
    for rows in [
        [prompt[..17].to_vec(), other[..17].to_vec()],
        [prompt[17..].to_vec(), other[17..].to_vec()],
        [vec![3], vec![7]],
        [vec![5], vec![11]],
    ] {
        let len = rows[0].len() as i32;
        let flat: Vec<i32> = rows.iter().flatten().copied().collect();
        let input = mlxcel_core::from_slice_i32(&flat, &[2, len]);
        let actual = batched.forward_batched(&input, &mut [], None);
        mlxcel_core::eval(&actual);
        let shape = mlxcel_core::array_shape(&actual);
        for (row, model) in singles.iter().enumerate() {
            let input = mlxcel_core::from_slice_i32(&rows[row], &[1, len]);
            let expected = model.forward(&input, &mut [], None);
            let actual_row = mlxcel_core::slice(
                &actual,
                &[row as i32, 0, 0],
                &[row as i32 + 1, len, shape[2]],
            );
            assert_close("anonymous batch row", &actual_row, &values(&expected));
        }
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn hyper_stream_mean_preserves_bfloat16_reference_rounding() {
    // mlx-vlm abd3b2d7: mx.mean(x, axis=-2), with bf16 x shaped [1, 1, 4, 1].
    // These stream values occur in the real OptiQ checkpoint. Promoting the
    // reduction to float32 changes 18.625 to 18.5 before the next projection.
    let streams = mlxcel_core::from_slice_f32(&[19.125, 17.75, 17.875, 19.5], &[1, 1, 4, 1]);
    let streams = mlxcel_core::astype(&streams, dtype::BFLOAT16);
    let mean = mlxcel_core::mean_axis_native_dtype(&streams, -2, false);
    assert_eq!(values(&mean), vec![18.625]);
    assert_eq!(mlxcel_core::array_dtype(&mean), dtype::BFLOAT16);
}

#[test]
fn quantized_gate_prefill_matches_independent_decode_rows() {
    use mlxcel_core::layers::UnifiedLinear;
    let dense = mlxcel_core::astype(&hashed("gate.weight", &[4, 256], 0.1, 0.0), dtype::BFLOAT16);
    let packed = mlxcel_core::quantize_weights(&dense, 32, 8);
    let mut weights = WeightMap::new();
    weights.insert(
        "gate.weight".into(),
        mlxcel_core::quantized_weights_w(&packed),
    );
    weights.insert(
        "gate.scales".into(),
        mlxcel_core::quantized_weights_scales(&packed),
    );
    weights.insert(
        "gate.biases".into(),
        mlxcel_core::quantized_weights_biases(&packed),
    );
    let linear = UnifiedLinear::from_weights(&weights, "gate", 32, 8).unwrap();
    let input = mlxcel_core::astype(
        &hashed("gate.input", &[2, 17, 256], 1.0, 0.0),
        dtype::BFLOAT16,
    );
    let batched = super::linear::row_invariant_projection(&linear, &input);
    for batch in 0..2 {
        for token in 0..17 {
            let row = mlxcel_core::slice(&input, &[batch, token, 0], &[batch + 1, token + 1, 256]);
            let expected = linear.forward(&mlxcel_core::contiguous(&row, false));
            let actual =
                mlxcel_core::slice(&batched, &[batch, token, 0], &[batch + 1, token + 1, 4]);
            assert_eq!(
                values(&actual),
                values(&expected),
                "batch {batch}, token {token}"
            );
        }
    }
}

#[test]
fn bf16_delta_preserves_float32_state_across_prefill_chunks() {
    use crate::models::gated_delta::gated_delta_update_precise;
    let input =
        |name| mlxcel_core::astype(&hashed(name, &[1, 5, 1, 128], 0.1, 0.0), dtype::BFLOAT16);
    let q = input("precision.q");
    let k = input("precision.k");
    let v = input("precision.v");
    let a = mlxcel_core::astype(
        &hashed("precision.a", &[1, 5, 1], 0.2, 0.0),
        dtype::BFLOAT16,
    );
    let b = mlxcel_core::astype(
        &hashed("precision.b", &[1, 5, 1], 0.2, 0.0),
        dtype::BFLOAT16,
    );
    let a_log = mlxcel_core::from_slice_f32(&[0.125], &[1]);
    let dt = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[0.125], &[1]),
        dtype::BFLOAT16,
    );
    let (whole, whole_state) =
        gated_delta_update_precise(&q, &k, &v, &a, &b, &a_log, &dt, None, None);
    assert_eq!(mlxcel_core::array_dtype(&whole_state), dtype::FLOAT32);
    assert_eq!(mlxcel_core::array_dtype(&whole), dtype::BFLOAT16);
    let mut state = None;
    for t in 0..5 {
        let row = |x: &MlxArray| mlxcel_core::slice(x, &[0, t, 0, 0], &[1, t + 1, 1, 128]);
        let gate = |x: &MlxArray| mlxcel_core::slice(x, &[0, t, 0], &[1, t + 1, 1]);
        let (out, next) = gated_delta_update_precise(
            &row(&q),
            &row(&k),
            &row(&v),
            &gate(&a),
            &gate(&b),
            &a_log,
            &dt,
            state.as_deref(),
            None,
        );
        assert_eq!(mlxcel_core::array_dtype(&next), dtype::FLOAT32);
        assert_close("chunked delta output", &out, &values(&row(&whole)));
        state = Some(next);
    }
    assert_close(
        "chunked delta state",
        state.as_ref().unwrap(),
        &values(&whole_state),
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn sparse_qsa_keeps_only_selected_blocks_and_causal_tail() {
    let q = mlxcel_core::zeros(&[1, 2, 4, 32], dtype::BFLOAT16);
    let k = mlxcel_core::zeros(&[1, 1, 10, 32], dtype::BFLOAT16);
    let v: Vec<f32> = (0..10)
        .flat_map(|t| std::iter::repeat_n(t as f32, 32))
        .collect();
    let v = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&v, &[1, 1, 10, 32]),
        dtype::BFLOAT16,
    );
    let blocks = mlxcel_core::from_slice_i32(&[-1, -1, -1, 0, -1, 1, -1, 0], &[1, 4, 2]);
    let ends = mlxcel_core::from_slice_i32(&[3, 5, 8, 10], &[1, 4]);
    let out = mlxcel_core::qwen4_qsa_attention(&q, &k, &v, &blocks, &ends, 0.125, 4);
    // Uniform attention: [0,1,2], [0..4], [4..7], and [0..3,8,9].
    let expected: Vec<f32> = (0..2)
        .flat_map(|_| {
            [1.0, 2.0, 5.5, 23.0 / 6.0]
                .into_iter()
                .flat_map(|x| std::iter::repeat_n(x, 32))
        })
        .collect();
    let expected = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&expected, &[1, 2, 4, 32]),
        dtype::BFLOAT16,
    );
    assert_eq!(values(&out), values(&expected));
    // Sparse decode uses this same reduction after the index budget is exceeded.
    let decode_q = mlxcel_core::slice(&q, &[0, 0, 3, 0], &[1, 2, 4, 32]);
    let decode_blocks = mlxcel_core::slice(&blocks, &[0, 3, 0], &[1, 4, 2]);
    let decode_ends = mlxcel_core::from_slice_i32(&[10], &[1, 1]);
    let decoded =
        mlxcel_core::qwen4_qsa_attention(&decode_q, &k, &v, &decode_blocks, &decode_ends, 0.125, 4);
    let last = mlxcel_core::slice(&out, &[0, 0, 3, 0], &[1, 2, 4, 32]);
    assert_eq!(values(&decoded), values(&last));
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bf16_multimodal_rope_matches_upstream_at_long_positions() {
    let fixture = fixture("rotary_bf16");
    let x = mlxcel_core::astype(&hashed_input(&fixture["input"]), dtype::BFLOAT16);
    let positions: Vec<i32> = ints(&fixture["positions"])
        .into_iter()
        .map(|p| p as i32)
        .collect();
    let positions = mlxcel_core::from_slice_i32(&positions, &[3, 1, 3]);
    let rope = InterleavedMRoPE::new_with_backend_frequencies(64, 10_000_000.0, vec![11, 11, 10]);
    let output = rope
        .apply_metal(&x, &positions)
        .expect("Metal fixture requires Metal");
    assert_eq!(values(&output), expected(&fixture["output"]));
    let tail = |a: &MlxArray| mlxcel_core::slice(a, &[0, 0, 0, 64], &[1, 1, 3, 128]);
    assert_eq!(values(&tail(&output)), values(&tail(&x)));
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bf16_moe_decode_matches_reference_expert_rounding() {
    let fixture = fixture("moe_bf16");
    let config = config(&fixture, None);
    let mut quantized = WeightMap::new();
    for (name, weight) in weights(&fixture) {
        let weight = mlxcel_core::astype(&weight, dtype::BFLOAT16);
        if name.ends_with("_proj.weight") {
            let prefix = name.strip_suffix(".weight").unwrap();
            quantized.insert(
                format!("{prefix}.scales"),
                mlxcel_core::quantize_weights_scales(&weight, 32, 4),
            );
            quantized.insert(
                format!("{prefix}.biases"),
                mlxcel_core::quantize_weights_biases(&weight, 32, 4),
            );
            quantized.insert(name, mlxcel_core::quantize_weights_w(&weight, 32, 4));
        } else {
            quantized.insert(name, weight);
        }
    }
    let moe = super::moe::Qwen4ExpMoeBlock::from_weights(&quantized, &config, "moe").unwrap();
    let input = mlxcel_core::astype(&hashed_input(&fixture["input"]), dtype::BFLOAT16);
    assert_eq!(values(&moe.forward(&input)), expected(&fixture["output"]));
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn fused_delta_projection_matches_singleton_quantized_outputs() {
    use mlxcel_core::layers::{QuantizedWeight, UnifiedLinear};
    let make = |name: &str, rows: i32, bits: i32| {
        let w = mlxcel_core::astype(&hashed(name, &[rows, 64], 0.125, 0.0), dtype::BFLOAT16);
        UnifiedLinear::new(
            QuantizedWeight::new(
                mlxcel_core::quantize_weights_w(&w, 32, bits),
                mlxcel_core::quantize_weights_scales(&w, 32, bits),
                mlxcel_core::quantize_weights_biases(&w, 32, bits),
                32,
                bits,
            ),
            None,
        )
    };
    let layers = [
        make("qkv", 128, 4),
        make("z", 64, 4),
        make("b", 16, 4),
        make("a", 16, 4),
    ];
    let refs: Vec<_> = layers.iter().collect();
    let fused = super::linear::fused_decode_projection(&refs).unwrap();
    let input = mlxcel_core::astype(
        &hashed("delta_input", &[1, 1, 64], 2.0, 0.0),
        dtype::BFLOAT16,
    );
    let actual = values(&fused.forward(&input));
    let expected: Vec<_> = layers
        .iter()
        .flat_map(|linear| values(&linear.forward(&input)))
        .collect();
    assert_eq!(actual, expected);
    let different = make("different", 16, 8);
    assert!(super::linear::fused_decode_projection(&[&layers[0], &different]).is_none());
}
