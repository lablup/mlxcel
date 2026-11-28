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

//! Token-only verify blocks of the dense and MoE families against the
//! single-token chain (issue #3421): Llama 3 (K2 Horizon), Arcee and OLMoE,
//! on float and 4-bit weights. With `MLXCEL_MLA_ROW_INVARIANT=1` the block
//! equals the chain byte for byte; without it the block parts from it.
//!
//! Head dim 64 so MLX's fused SDPA takes the attention (it does not take
//! the tiny head dims, and its unfused fallback is a path no checkpoint
//! runs); every product width a multiple of 64 so the 4-bit planes quantize.

use mlxcel_core::cache::DecodeLookaheadAppendScope;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::mla_batched::RowInvariantTestMode;

const VOCAB: i32 = 96;
const HIDDEN: i32 = 64;
const FF: i32 = 128;
const HEADS: i32 = 4;
const KV_HEADS: i32 = 2;
const HEAD_DIM: i32 = 64;
const EXPERTS: i32 = 4;
const LAYERS: usize = 2;

/// A varied, bounded fill, salted per tensor so no two weights repeat.
fn fill(shape: &[i32], salt: usize) -> UniquePtr<MlxArray> {
    let n: usize = shape.iter().product::<i32>() as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| ((((i + salt * 131) * 37 + 11) % 97) as f32 - 48.0) * 0.01)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// Norm weights near one, every other weight from [`fill`]; with `quantized`
/// every projection plane becomes an affine 4-bit group-64 triple.
fn weights(shapes: Vec<(String, Vec<i32>)>, quantized: bool) -> WeightMap {
    let mut shapes = shapes;
    shapes.sort();
    let mut out = WeightMap::new();
    for (salt, (key, shape)) in shapes.into_iter().enumerate() {
        let w = fill(&shape, salt);
        if key.contains("norm") {
            let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
            out.insert(key, mlxcel_core::add(&w, &one));
            continue;
        }
        let projection = key.ends_with(".weight") && !key.contains("embed_tokens");
        if quantized && projection {
            let base = key.trim_end_matches(".weight");
            out.insert(
                format!("{base}.scales"),
                mlxcel_core::quantize_weights_scales(&w, 64, 4),
            );
            out.insert(
                format!("{base}.biases"),
                mlxcel_core::quantize_weights_biases(&w, 64, 4),
            );
            out.insert(key, mlxcel_core::quantize_weights_w(&w, 64, 4));
        } else {
            out.insert(key, w);
        }
    }
    out
}

fn quantization(quantized: bool) -> &'static str {
    if quantized {
        r#", "quantization": {"group_size": 64, "bits": 4}"#
    } else {
        ""
    }
}

/// The attention planes and norms every family here shares.
fn decoder_shapes(mlp: &[(&str, Vec<i32>)]) -> Vec<(String, Vec<i32>)> {
    let mut shapes = vec![
        ("model.embed_tokens.weight".to_owned(), vec![VOCAB, HIDDEN]),
        ("model.norm.weight".to_owned(), vec![HIDDEN]),
        ("lm_head.weight".to_owned(), vec![VOCAB, HIDDEN]),
    ];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        for (name, shape) in [
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("self_attn.q_proj.weight", vec![HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.k_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.v_proj.weight", vec![KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, HEADS * HEAD_DIM]),
        ] {
            shapes.push((format!("{p}.{name}"), shape));
        }
        for (name, shape) in mlp {
            shapes.push((format!("{p}.mlp.{name}"), shape.clone()));
        }
    }
    shapes
}

fn common_config() -> String {
    format!(
        r#""vocab_size": {VOCAB}, "hidden_size": {HIDDEN}, "intermediate_size": {FF},
        "num_hidden_layers": {LAYERS}, "num_attention_heads": {HEADS},
        "num_key_value_heads": {KV_HEADS}, "head_dim": {HEAD_DIM}, "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0, "tie_word_embeddings": false"#
    )
}

fn llama3(quantized: bool) -> super::Llama3Model {
    let json = format!(
        r#"{{"model_type": "llama", {}{}}}"#,
        common_config(),
        quantization(quantized)
    );
    let args: super::llama3::ModelArgs = serde_json::from_str(&json).expect("llama config");
    let shapes = decoder_shapes(&[
        ("gate_proj.weight", vec![FF, HIDDEN]),
        ("up_proj.weight", vec![FF, HIDDEN]),
        ("down_proj.weight", vec![HIDDEN, FF]),
    ]);
    super::Llama3Model::from_weights(&weights(shapes, quantized), &args).expect("tiny llama")
}

fn arcee(quantized: bool) -> super::ArceeModel {
    let json = format!(
        r#"{{"model_type": "arcee", "hidden_act": "relu2", "max_position_embeddings": 4096,
        {}{}}}"#,
        common_config(),
        quantization(quantized)
    );
    let config = super::arcee::parse_config(&json).expect("arcee config");
    let shapes = decoder_shapes(&[
        ("up_proj.weight", vec![FF, HIDDEN]),
        ("down_proj.weight", vec![HIDDEN, FF]),
    ]);
    super::ArceeModel::from_weights(&weights(shapes, quantized), &config).expect("tiny arcee")
}

fn olmoe(quantized: bool) -> super::OlmoeModel {
    let json = format!(
        r#"{{"model_type": "olmoe", "num_experts": {EXPERTS}, "num_experts_per_tok": 2,
        {}{}}}"#,
        common_config(),
        quantization(quantized)
    );
    let args: super::olmoe::ModelArgs = serde_json::from_str(&json).expect("olmoe config");
    let mut shapes = decoder_shapes(&[
        ("gate.weight", vec![EXPERTS, HIDDEN]),
        ("switch_mlp.gate_proj.weight", vec![EXPERTS, FF, HIDDEN]),
        ("switch_mlp.up_proj.weight", vec![EXPERTS, FF, HIDDEN]),
        ("switch_mlp.down_proj.weight", vec![EXPERTS, HIDDEN, FF]),
    ]);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.self_attn");
        shapes.push((format!("{p}.q_norm.weight"), vec![HEADS * HEAD_DIM]));
        shapes.push((format!("{p}.k_norm.weight"), vec![KV_HEADS * HEAD_DIM]));
    }
    super::OlmoeModel::from_weights(&weights(shapes, quantized), &args).expect("tiny olmoe")
}

fn input(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn row_bytes(logits: &MlxArray, row: i32) -> Vec<u8> {
    mlxcel_core::array_to_raw_bytes(&mlxcel_core::utils::slice_axis(logits, 1, row, row + 1))
}

/// Rows of a `width`-token block in the speculative append scope that differ
/// from the single-token chain over the same tokens, through the plain
/// `LanguageModel::forward` the token-only verify round runs.
fn differing_rows(model: &dyn LanguageModel, width: usize) -> Vec<usize> {
    let prompt: Vec<i32> = (0..9).map(|i| (i * 7 + 3) % VOCAB).collect();
    let block: Vec<i32> = (0..width as i32).map(|i| (i * 13 + 5) % VOCAB).collect();
    let prefilled = || {
        let mut caches = model.make_caches();
        mlxcel_core::eval(&model.forward(&input(&prompt), &mut caches, None));
        caches
    };

    let mut chain_caches = prefilled();
    let chain: Vec<Vec<u8>> = block
        .iter()
        .map(|&token| row_bytes(&model.forward(&input(&[token]), &mut chain_caches, None), 0))
        .collect();

    let mut block_caches = prefilled();
    let logits = {
        let _speculative = DecodeLookaheadAppendScope::enter();
        model.forward(&input(&block), &mut block_caches, None)
    };
    (0..width)
        .filter(|&row| row_bytes(&logits, row as i32) != chain[row])
        .collect()
}

fn models(quantized: bool) -> Vec<(&'static str, Box<dyn LanguageModel>)> {
    vec![
        ("llama3", Box::new(llama3(quantized))),
        ("arcee", Box::new(arcee(quantized))),
        ("olmoe", Box::new(olmoe(quantized))),
    ]
}

/// MLX sends a quantized product of two or more rows down `qmv_wide`, which
/// sums in another order than the one-row `qmv`, so a quantized block equals
/// the chain only with that switch off (`MLXCEL_QMV_WIDE=0`), as on the
/// DeepSeek V3 family. The switch is process-global; it is restored on drop,
/// and only this test's quantized half runs under it.
struct QmvWideOff {
    previous: bool,
}

impl QmvWideOff {
    fn enter() -> Self {
        let previous = mlxcel_core::qmv_wide_enabled();
        mlxcel_core::set_qmv_wide(false);
        Self { previous }
    }
}

impl Drop for QmvWideOff {
    fn drop(&mut self) {
        mlxcel_core::set_qmv_wide(self.previous);
    }
}

fn assert_block_matches_chain(quantized: bool) {
    for (family, model) in models(quantized) {
        for kernel in [true, false] {
            let _mode = RowInvariantTestMode::set(Some(kernel));
            for width in [2, 4, 8, 12] {
                assert_eq!(
                    differing_rows(model.as_ref(), width),
                    Vec::<usize>::new(),
                    "{family} (quantized {quantized}), width {width}, matvec kernel {kernel}: \
                     block rows differ from the chain"
                );
            }
        }
    }
}

#[test]
fn a_verify_block_matches_the_single_token_chain_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    assert_block_matches_chain(false);
    let _narrow = QmvWideOff::enter();
    assert_block_matches_chain(true);
}

/// The default, and the code before #3421: without the scope the same float
/// block parts from the chain (gemv against gemm, the block SDPA, the
/// compiled MLP and the expert kernels).
#[test]
fn without_the_scope_a_block_differs_from_the_chain_on_metal() {
    if !mlxcel_core::metal_is_available() {
        return;
    }
    let _mode = RowInvariantTestMode::set(None);
    for (family, model) in models(false) {
        let differing: usize = [2, 4, 8]
            .into_iter()
            .map(|width| differing_rows(model.as_ref(), width).len())
            .sum();
        assert!(
            differing > 0,
            "{family}: the block already equals the chain without the scope"
        );
    }
}
