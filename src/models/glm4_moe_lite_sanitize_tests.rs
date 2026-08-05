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

//! Tests for the GLM4 MoE Lite MLA `kv_b_proj` decomposition (issue #1029).
//!
//! Every assertion is on a `Result` from `sanitize_weights` or from the real
//! `Glm4MoeLiteModel::from_weights`, and no test runs a forward pass. That is
//! deliberate: a load that produced a malformed `embed_q` would reach MLX
//! through the cxx bridge as `UniquePtr<MlxArray>` rather than `Result`, so a
//! C++ throw there is an uncatchable abort that takes the whole test binary
//! down instead of failing one test.

use super::sanitize_weights;
use crate::models::glm4_moe_lite::{Glm4MoeLiteModel, ModelArgs};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const VOCAB: i32 = 16;
const HIDDEN: i32 = 8;
const INTERMEDIATE: i32 = 8;
const HEADS: i32 = 2;
const KV_LORA_RANK: i32 = 16;
const QK_NOPE: i32 = 4;
const QK_ROPE: i32 = 2;
/// Deliberately different from [`QK_NOPE`]. While the two half-widths were
/// equal, `[HEADS, KV_LORA_RANK, QK_NOPE]` and `[HEADS, V_HEAD, KV_LORA_RANK]`
/// stayed valid with the nope and v halves swapped, so every shape assertion in
/// this file passed whichever half each output was built from.
const V_HEAD: i32 = 6;

/// `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]`, the layout
/// every public `glm4_moe_lite` checkpoint stores `kv_b_proj` in.
const KV_B_ROWS: i32 = HEADS * (QK_NOPE + V_HEAD);

/// The per-head row block of `kv_b_proj`: `qk_nope_head_dim` rows of the nope
/// half first, then `v_head_dim` rows of the v half.
const HEAD_DIM: i32 = QK_NOPE + V_HEAD;

const ATTN: &str = "model.layers.0.self_attn";

/// One dense layer, so `is_moe_layer` stays false (`n_routed_experts` is left
/// unset) and the fixture does not have to build a stacked expert plane, and
/// `q_lora_rank` is null so the Q side is a single `q_proj`. Neither choice
/// touches the MLA decomposition under test.
fn tiny_config() -> ModelArgs {
    let json = format!(
        r#"{{
        "model_type": "glm4_moe_lite",
        "vocab_size": {VOCAB},
        "hidden_size": {HIDDEN},
        "intermediate_size": {INTERMEDIATE},
        "moe_intermediate_size": {INTERMEDIATE},
        "num_hidden_layers": 1,
        "num_attention_heads": {HEADS},
        "num_key_value_heads": {HEADS},
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "kv_lora_rank": {KV_LORA_RANK},
        "q_lora_rank": null,
        "qk_rope_head_dim": {QK_ROPE},
        "qk_nope_head_dim": {QK_NOPE},
        "v_head_dim": {V_HEAD}
    }}"#
    );
    serde_json::from_str(&json).expect("parse tiny glm4_moe_lite config")
}

fn dense(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![0.0f32; n as usize], shape)
}

/// Everything `Glm4MoeLiteModel::from_weights` needs except the MLA pair, so
/// the only reason a load can fail below is the `kv_b_proj` handling.
fn checkpoint_without_the_mla_pair() -> WeightMap {
    let q_head_dim = QK_NOPE + QK_ROPE;
    let mut weights = WeightMap::new();
    for (key, shape) in [
        ("model.embed_tokens.weight", vec![VOCAB, HIDDEN]),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![HEADS * q_head_dim, HIDDEN],
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![KV_LORA_RANK + QK_ROPE, HIDDEN],
        ),
        (
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            vec![KV_LORA_RANK],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![HIDDEN, HEADS * V_HEAD],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![INTERMEDIATE, HIDDEN],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![INTERMEDIATE, HIDDEN],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![HIDDEN, INTERMEDIATE],
        ),
        ("model.layers.0.input_layernorm.weight", vec![HIDDEN]),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![HIDDEN],
        ),
        ("model.norm.weight", vec![HIDDEN]),
        ("lm_head.weight", vec![VOCAB, HIDDEN]),
    ] {
        weights.insert(key.to_string(), dense(&shape));
    }
    weights
}

/// The layout the three public `glm4_moe_lite` repos actually ship: a float
/// `kv_b_proj`, no `.scales`, no `.biases`, and no `embed_q` at all.
fn float_kv_b_proj_checkpoint() -> WeightMap {
    let mut weights = checkpoint_without_the_mla_pair();
    weights.insert(
        format!("{ATTN}.kv_b_proj.weight"),
        dense(&[KV_B_ROWS, KV_LORA_RANK]),
    );
    weights
}

/// The same float layout, but with `kv_b_proj` filled with a `0, 1, 2, ...`
/// ramp in row-major order over `[KV_B_ROWS, KV_LORA_RANK]`, so every element
/// carries its own flat source index and the decomposition's element mapping
/// becomes observable. The zero-filled fixtures above cannot show it.
fn ramp_kv_b_proj_checkpoint() -> (WeightMap, Vec<f32>) {
    let kv_b: Vec<f32> = (0..KV_B_ROWS * KV_LORA_RANK).map(|i| i as f32).collect();
    let mut weights = checkpoint_without_the_mla_pair();
    weights.insert(
        format!("{ATTN}.kv_b_proj.weight"),
        mlxcel_core::from_slice_f32(&kv_b, &[KV_B_ROWS, KV_LORA_RANK]),
    );
    (weights, kv_b)
}

/// An affine 4-bit `kv_b_proj`. The geometry is honest: `packed_in * 32 == bits
/// * num_groups * group_size` (2 * 32 == 4 * 1 * 16) against `kv_lora_rank` 16,
/// which is what `infer_mla_quantization_params` solves the pair from.
///
/// The packed plane is UINT32, not a float dtype: `dequantize` rejects any
/// other packed dtype by throwing, and that throw is the uncatchable abort this
/// file exists to keep out of the test binary. A float fixture here would take
/// the process down on the positive control.
fn quantized_kv_b_proj_checkpoint() -> WeightMap {
    let mut weights = checkpoint_without_the_mla_pair();
    weights.insert(
        format!("{ATTN}.kv_b_proj.weight"),
        mlxcel_core::zeros(&[KV_B_ROWS, 2], mlxcel_core::dtype::UINT32),
    );
    weights.insert(format!("{ATTN}.kv_b_proj.scales"), dense(&[KV_B_ROWS, 1]));
    weights.insert(format!("{ATTN}.kv_b_proj.biases"), dense(&[KV_B_ROWS, 1]));
    weights
}

/// The same plane an mxfp4 / nvfp4 / mxfp8 export ships: `.scales` present,
/// `.biases` absent, because the block-float modes carry no zero points.
fn block_float_kv_b_proj_checkpoint() -> WeightMap {
    let mut weights = quantized_kv_b_proj_checkpoint();
    weights.remove(&format!("{ATTN}.kv_b_proj.biases"));
    weights
}

/// A checkpoint that already carries the decomposed pair, and a stale
/// `kv_b_proj` alongside it.
fn predecomposed_checkpoint() -> WeightMap {
    let mut weights = float_kv_b_proj_checkpoint();
    weights.insert(
        format!("{ATTN}.embed_q.weight"),
        dense(&[HEADS, KV_LORA_RANK, QK_NOPE]),
    );
    weights.insert(
        format!("{ATTN}.unembed_out.weight"),
        dense(&[HEADS, V_HEAD, KV_LORA_RANK]),
    );
    weights
}

fn shape_of(weights: &WeightMap, key: &str) -> Vec<i32> {
    mlxcel_core::array_shape(weights.get(key).unwrap_or_else(|| panic!("missing {key}")))
}

/// Read an array back as flat row-major `f32`. Reading values out over the cxx
/// bridge is safe in a way a forward pass is not: it neither reshapes nor
/// matmuls, so there is no C++ throw to abort the test binary.
fn read_f32(arr: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Issue #1029: `glm4_moe_lite` had no `sanitize_weights` and no `kv_b_proj`
/// decomposition anywhere, while `MlaAttention::from_weights` loads `embed_q`
/// and `unembed_out` unconditionally. Since no published checkpoint ships
/// either tensor, the family could not load the canonical layout of its own
/// architecture.
///
/// The first assertion is the failure the issue reports, taken from the real
/// loader rather than restated, so this test would still fail if the
/// decomposition were removed and `MlaAttention` were relaxed to tolerate a
/// missing `embed_q` instead.
#[test]
fn sanitize_synthesizes_the_mla_pair_no_checkpoint_ships() {
    let args = tiny_config();

    let err = Glm4MoeLiteModel::from_weights(&float_kv_b_proj_checkpoint(), &args)
        .err()
        .expect("an unsanitized kv_b_proj checkpoint has no embed_q to load");
    assert_eq!(
        err, "Weight not found: model.layers.0.self_attn.embed_q.weight",
        "the unsanitized failure this test pins must stay the one issue #1029 reports"
    );

    let sanitized = sanitize_weights(float_kv_b_proj_checkpoint(), &args)
        .expect("a float kv_b_proj must decompose");

    assert!(
        !sanitized.contains_key(&format!("{ATTN}.kv_b_proj.weight")),
        "the decomposed kv_b_proj must be consumed, not left beside the pair"
    );
    assert_eq!(
        shape_of(&sanitized, &format!("{ATTN}.embed_q.weight")),
        vec![HEADS, KV_LORA_RANK, QK_NOPE],
        "embed_q holds the nope half with its last two axes swapped"
    );
    assert_eq!(
        shape_of(&sanitized, &format!("{ATTN}.unembed_out.weight")),
        vec![HEADS, V_HEAD, KV_LORA_RANK],
        "unembed_out holds the v half unswapped"
    );

    Glm4MoeLiteModel::from_weights(&sanitized, &args)
        .expect("the sanitized checkpoint must load through the real loader");
}

/// The split runs per head along the row axis, so it has to happen in dense
/// space: slicing the packed plane would leave each half carrying group scales
/// and biases that describe the whole row.
#[test]
fn sanitize_dequantizes_kv_b_proj_before_the_per_head_split() {
    let args = tiny_config();

    let sanitized = sanitize_weights(quantized_kv_b_proj_checkpoint(), &args)
        .expect("an honest affine kv_b_proj must decompose");

    for suffix in ["weight", "scales", "biases"] {
        assert!(
            !sanitized.contains_key(&format!("{ATTN}.kv_b_proj.{suffix}")),
            "all three kv_b_proj planes are consumed by the dequantize, `{suffix}` survived"
        );
    }
    // No `.scales` on either half is what makes `MultiLinear::from_weights`
    // take its dense branch; a re-quantized pair would need planes this
    // decomposition never rebuilds.
    for name in ["embed_q", "unembed_out"] {
        assert!(
            !sanitized.contains_key(&format!("{ATTN}.{name}.scales")),
            "{name} must be stored dense after the dequantize"
        );
    }
    assert_eq!(
        shape_of(&sanitized, &format!("{ATTN}.embed_q.weight")),
        vec![HEADS, KV_LORA_RANK, QK_NOPE],
        "the dequantized plane must split to the same shape the float one does"
    );

    Glm4MoeLiteModel::from_weights(&sanitized, &args)
        .expect("the dequantized pair must load through the real loader");
}

/// Issue #1026 / PR #1027: the quantized branch decides `kv_b_proj` is
/// quantized on `.scales` alone. Affine stores zero points; the block-float
/// modes (mxfp4 / nvfp4 / mxfp8) do not. A block-float `kv_b_proj` therefore
/// satisfies the `.scales` gate and reaches the `.biases` removal with nothing
/// to take, and an `.unwrap()` there is a panic during weight sanitization,
/// which in the server takes the process down rather than rejecting one load.
///
/// The decomposition still dequantizes as `"affine"`, so a genuine block-float
/// `kv_b_proj` is not supported either way. What this pins is that it is
/// refused by name, in the wording the other five sanitizers share.
#[test]
fn sanitize_rejects_scales_with_no_biases() {
    let args = tiny_config();

    // Positive control first, so a check that rejected every quantized
    // kv_b_proj could not pass this test.
    sanitize_weights(quantized_kv_b_proj_checkpoint(), &args)
        .expect("a kv_b_proj carrying both planes must still sanitize");

    let err = sanitize_weights(block_float_kv_b_proj_checkpoint(), &args)
        .err()
        .expect("scales with no biases must be refused at load, not unwrapped");
    assert!(
        err.contains("model.layers.0.self_attn.kv_b_proj.biases"),
        "the error must name the key that is missing, got: {err}"
    );
    assert!(
        err.contains("scales but no biases"),
        "the wording must match the other five MLA sanitizers, got: {err}"
    );
}

/// A checkpoint that already ships the pair is skipped, matching the `continue`
/// in every sibling sanitizer. This is not hypothetical: once a converter emits
/// a pre-decomposed quantized `embed_q`, rebuilding it here from a stale
/// `kv_b_proj` would silently replace the quantized plane with a dense one.
#[test]
fn sanitize_leaves_an_already_decomposed_checkpoint_alone() {
    let args = tiny_config();

    let sanitized = sanitize_weights(predecomposed_checkpoint(), &args)
        .expect("a pre-decomposed checkpoint must sanitize");

    // The decomposition removes `kv_b_proj.weight`, so its survival is the
    // observable proof that the skip fired rather than the split running.
    assert!(
        sanitized.contains_key(&format!("{ATTN}.kv_b_proj.weight")),
        "an existing embed_q must stop the decomposition before it consumes kv_b_proj"
    );
    assert_eq!(
        shape_of(&sanitized, &format!("{ATTN}.embed_q.weight")),
        vec![HEADS, KV_LORA_RANK, QK_NOPE],
        "the shipped embed_q must survive untouched"
    );
}

/// The reshape below the split is where a `config.json` that disagrees with the
/// stored tensor stops being recoverable: MLX reports a bad reshape by
/// throwing, and that throw crosses the cxx bridge as `UniquePtr<MlxArray>`
/// rather than `Result`, so it aborts during weight sanitization instead of
/// failing the load. The cross-check turns that into an error naming both
/// shapes.
#[test]
fn sanitize_rejects_a_kv_b_proj_that_disagrees_with_the_config() {
    let args = tiny_config();

    let mut weights = checkpoint_without_the_mla_pair();
    weights.insert(
        format!("{ATTN}.kv_b_proj.weight"),
        dense(&[KV_B_ROWS + QK_NOPE + V_HEAD, KV_LORA_RANK]),
    );

    let err = sanitize_weights(weights, &args)
        .err()
        .expect("a kv_b_proj carrying an extra head must be refused before the reshape");
    assert!(
        err.contains("shape mismatch"),
        "the error must say what is wrong, got: {err}"
    );
    assert!(
        err.contains(&format!("[{KV_B_ROWS}, {KV_LORA_RANK}]")),
        "the error must name the shape the config describes, got: {err}"
    );
}

/// Pins the element mapping, not just the shapes: which half of `kv_b_proj`
/// each output is built from, and which axis order it lands in.
///
/// Every other assertion in this file is blind to both. Shapes cannot catch a
/// swapped nope / v half once the two half-widths are told apart only by
/// [`QK_NOPE`] and [`V_HEAD`], and cannot catch a wrong reshape or transpose
/// axis order at all, so the ramp fixture is what makes either failure
/// observable.
///
/// This matters more than a load error would. A decomposition that picks the
/// wrong half or the wrong axis order still produces tensors of the right
/// shape, so the model loads, runs, and emits plausible text built on weights
/// that mean something else. A missing `embed_q` at least stops at the loader.
///
/// The two expectations come from the HF definition: `kv_b_proj =
/// nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))`, so
/// the stored `weight` is `[num_heads * head_dim, kv_lora_rank]` row-major and
/// head-major, and within each head the `qk_nope` rows precede the `v_head`
/// ones (`expand_kv` views `(..., -1, qk_nope + v_head)` then splits on the
/// last axis). `embed_q` is the nope half transposed; `unembed_out` is the v
/// half as stored.
#[test]
fn sanitize_splits_the_halves_in_the_order_the_reference_layout_stores_them() {
    let args = tiny_config();
    let (weights, kv_b) = ramp_kv_b_proj_checkpoint();

    let sanitized = sanitize_weights(weights, &args).expect("a float kv_b_proj must decompose");

    let embed_q = read_f32(
        sanitized
            .get(&format!("{ATTN}.embed_q.weight"))
            .expect("embed_q"),
    );
    let unembed_out = read_f32(
        sanitized
            .get(&format!("{ATTN}.unembed_out.weight"))
            .expect("unembed_out"),
    );
    assert_eq!(embed_q.len(), (HEADS * KV_LORA_RANK * QK_NOPE) as usize);
    assert_eq!(unembed_out.len(), (HEADS * V_HEAD * KV_LORA_RANK) as usize);

    for h in 0..HEADS {
        // embed_q[h][r][c] is the nope row `c` of head `h`, column `r`: the
        // last two axes of the nope half swapped.
        for r in 0..KV_LORA_RANK {
            for c in 0..QK_NOPE {
                let got = embed_q[((h * KV_LORA_RANK + r) * QK_NOPE + c) as usize];
                let want = kv_b[((h * HEAD_DIM + c) * KV_LORA_RANK + r) as usize];
                assert_eq!(
                    got, want,
                    "embed_q[{h}][{r}][{c}] must be the transposed nope half, not the v half \
                     or an untransposed view"
                );
            }
        }
        // unembed_out[h][d][r] is v row `d` of head `h`, stored as it is: the v
        // rows start `QK_NOPE` into the head's row block.
        for d in 0..V_HEAD {
            for r in 0..KV_LORA_RANK {
                let got = unembed_out[((h * V_HEAD + d) * KV_LORA_RANK + r) as usize];
                let want = kv_b[((h * HEAD_DIM + QK_NOPE + d) * KV_LORA_RANK + r) as usize];
                assert_eq!(
                    got, want,
                    "unembed_out[{h}][{d}][{r}] must be the untransposed v half, not the nope \
                     half"
                );
            }
        }
    }
}
