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

//! Operator declarations of every production auxiliary family (issue #3211).
//!
//! Each list names the operator classes the family's emitter writes, one entry
//! per distinct operand origin and input policy. Pooling, RoPE, masking, and
//! gathers are layout or elementwise steps with a single rounding and are left
//! out; the MLIR digest in the generation identity still binds them.

use crate::numeric_dtype_contract::NumericDType;

use super::production::{DeclaredOperator, InputPolicy, OperandOrigin};
use super::{AssociationPolicy, OperatorClass};

use AssociationPolicy::{BackendAlgorithm, CudaOneBlockScan, Sequential};
use InputPolicy::{F32Carrier, FollowsPrecision, PinnedF32};
use OperandOrigin::{Activation, FloatParameter, MatrixWeight};

const fn weighted(id: &'static str, class: OperatorClass) -> DeclaredOperator {
    DeclaredOperator::new(id, class, MatrixWeight, FollowsPrecision, BackendAlgorithm)
}

const fn norm(id: &'static str, class: OperatorClass) -> DeclaredOperator {
    DeclaredOperator::new(id, class, FloatParameter, F32Carrier, BackendAlgorithm)
}

/// Patch embedding through `Builder::convolution` or `linear_seq` on a float
/// kernel that quantization never packs.
const fn float_kernel(id: &'static str, class: OperatorClass) -> DeclaredOperator {
    DeclaredOperator::new(
        id,
        class,
        FloatParameter,
        FollowsPrecision,
        BackendAlgorithm,
    )
}

/// Query/key scores, softmax, and value context of one attention block.
const fn attention(
    scores: &'static str,
    softmax: &'static str,
    context: &'static str,
) -> [DeclaredOperator; 3] {
    [
        DeclaredOperator::new(
            scores,
            OperatorClass::DenseMatmul,
            Activation,
            FollowsPrecision,
            BackendAlgorithm,
        ),
        DeclaredOperator::new(
            softmax,
            OperatorClass::AttentionSoftmax,
            Activation,
            F32Carrier,
            BackendAlgorithm,
        ),
        DeclaredOperator::new(
            context,
            OperatorClass::DenseMatmul,
            Activation,
            FollowsPrecision,
            BackendAlgorithm,
        ),
    ]
}

const fn residual(id: &'static str) -> DeclaredOperator {
    DeclaredOperator::new(
        id,
        OperatorClass::ResidualAdd,
        Activation,
        F32Carrier,
        Sequential,
    )
}

/// The pre-norm transformer encoder layer shared by the vision towers: norm,
/// q/k/v/o projections, attention, residual, and an activated MLP.
fn encoder_layer(
    norm_class: OperatorClass,
    activation: OperatorClass,
    gated_up: bool,
) -> Vec<DeclaredOperator> {
    let mut operators = vec![
        norm("encoder.norm", norm_class),
        weighted("encoder.attention-projection", OperatorClass::DenseMatmul),
    ];
    operators.extend(attention(
        "encoder.attention-scores",
        "encoder.attention-softmax",
        "encoder.attention-context",
    ));
    operators.push(residual("encoder.residual-add"));
    operators.push(weighted("encoder.mlp-activation-projection", activation));
    if gated_up {
        operators.push(weighted(
            "encoder.mlp-up-projection",
            OperatorClass::DenseMatmul,
        ));
    }
    operators.push(weighted(
        "encoder.mlp-down-projection",
        OperatorClass::DenseMatmul,
    ));
    operators
}

/// `vision_runtime.rs`: LLaVA, SigLIP, and Gemma3 towers (`emitter/vision.rs`).
/// The NCHW patch convolution never demotes its inputs.
pub(crate) fn llava_vision_operators(gemma3_projector: bool) -> Vec<DeclaredOperator> {
    let mut operators = vec![DeclaredOperator::new(
        "patch-embedding",
        OperatorClass::Convolution,
        FloatParameter,
        PinnedF32,
        BackendAlgorithm,
    )];
    operators.extend(encoder_layer(
        OperatorClass::LayerNorm,
        OperatorClass::GeluProjection,
        false,
    ));
    if gemma3_projector {
        operators.extend([
            norm("projector.post-layer-norm", OperatorClass::LayerNorm),
            norm("projector.rms-norm", OperatorClass::RmsNorm),
            weighted("projector.output-projection", OperatorClass::DenseMatmul),
        ]);
    } else {
        operators.extend([
            weighted("projector.gelu-projection", OperatorClass::GeluProjection),
            weighted("projector.output-projection", OperatorClass::DenseMatmul),
        ]);
    }
    operators
}

fn merger(norm_class: OperatorClass, prefix: &'static [&'static str; 3]) -> [DeclaredOperator; 3] {
    [
        norm(prefix[0], norm_class),
        weighted(prefix[1], OperatorClass::GeluProjection),
        weighted(prefix[2], OperatorClass::DenseMatmul),
    ]
}

const MERGER: [&str; 3] = [
    "merger.norm",
    "merger.gelu-projection",
    "merger.output-projection",
];

/// `qwen2_vl_runtime.rs`: Qwen2-VL (`emitter/qwen2_vl.rs`) and Qwen2.5-VL
/// (`emitter/qwen2_5_vl.rs`, including its diagnostics graph).
pub(crate) fn qwen2_vl_operators(qwen2_5: bool) -> Vec<DeclaredOperator> {
    let mut operators = vec![float_kernel("patch-embedding", OperatorClass::DenseMatmul)];
    let norm_class = if qwen2_5 {
        OperatorClass::RmsNorm
    } else {
        OperatorClass::LayerNorm
    };
    let activation = if qwen2_5 {
        OperatorClass::SiluProjection
    } else {
        OperatorClass::GeluProjection
    };
    operators.extend(encoder_layer(norm_class, activation, qwen2_5));
    operators.extend(merger(norm_class, &MERGER));
    operators
}

/// `qwen3_vl_runtime.rs`: Qwen3-VL main merger plus the DeepStack mergers
/// (`emitter/qwen3_vl.rs`). The graph stores every operator result on eager
/// MLX's BF16 grid (#3291, `emitter/qwen3_vl_ops.rs`). Only the attention
/// scores stay f32: the fused SDPA keeps them in its f32 softmax statistics.
pub(crate) fn qwen3_vl_operators() -> Vec<DeclaredOperator> {
    let bf16 = NumericDType::Bf16;
    let mut operators = vec![float_kernel("patch-embedding", OperatorClass::DenseMatmul)];
    operators.extend(encoder_layer(
        OperatorClass::LayerNorm,
        OperatorClass::GeluProjection,
        false,
    ));
    operators.extend(merger(OperatorClass::LayerNorm, &MERGER));
    operators.extend(merger(
        OperatorClass::LayerNorm,
        &[
            "deepstack-merger.norm",
            "deepstack-merger.gelu-projection",
            "deepstack-merger.output-projection",
        ],
    ));
    operators
        .into_iter()
        .map(|operator| {
            if operator.id == "encoder.attention-scores" {
                operator
            } else {
                operator.rounded_to(bf16)
            }
        })
        .collect()
}

/// Attention pooling and the SiLU-gated projector shared by Molmo and Molmo2.
fn molmo_pool_and_projector() -> Vec<DeclaredOperator> {
    let mut operators = vec![weighted(
        "pool.attention-projection",
        OperatorClass::DenseMatmul,
    )];
    operators.extend(attention(
        "pool.attention-scores",
        "pool.attention-softmax",
        "pool.attention-context",
    ));
    operators.extend([
        weighted("projector.silu-projection", OperatorClass::SiluProjection),
        weighted("projector.up-projection", OperatorClass::DenseMatmul),
        weighted("projector.down-projection", OperatorClass::DenseMatmul),
    ]);
    operators
}

/// `molmo_vision_runtime.rs` (`emitter/molmo_vision.rs`). Every projection
/// weight stays packed and `Builder::dequant_affine` unpacks it in the graph.
///
/// The graph rounds to F16 where eager MLX stores an F16 tensor for the F16
/// checkpoint (`emitter/molmo_vision_f16.rs`, #3390): the dequantized weight,
/// every linear product and bias add, the patch embedding, LayerNorm,
/// quick_gelu, the encoder attention context and the residual adds. The
/// attention scores, softmax and the pooling attention stay f32 as eager
/// computes them (eager casts the pool attention context back to F16 before
/// `wo`, but the graph keeps the pool path in f32; every observable stage
/// passes with headroom).
pub(crate) fn molmo_vision_operators(bits: usize) -> Vec<DeclaredOperator> {
    let dequantize = if bits == 4 {
        OperatorClass::AffineQ4Dequantize
    } else {
        OperatorClass::AffineQ8Dequantize
    };
    let mut operators = vec![
        DeclaredOperator::new(
            "weight-dequantize",
            dequantize,
            MatrixWeight,
            F32Carrier,
            Sequential,
        ),
        float_kernel("patch-embedding", OperatorClass::DenseMatmul),
    ];
    operators.extend(encoder_layer(
        OperatorClass::LayerNorm,
        OperatorClass::GeluProjection,
        false,
    ));
    operators.extend(molmo_pool_and_projector());
    let f16 = NumericDType::F16;
    operators
        .into_iter()
        .map(|operator| match operator.id {
            "encoder.attention-scores"
            | "encoder.attention-softmax"
            | "pool.attention-scores"
            | "pool.attention-softmax"
            | "pool.attention-context" => operator,
            _ => operator.rounded_to(f16),
        })
        .collect()
}

/// `molmo2_vision_runtime.rs` (`emitter/molmo2_vision.rs`).
pub(crate) fn molmo2_vision_operators() -> Vec<DeclaredOperator> {
    let mut operators = vec![weighted("patch-embedding", OperatorClass::DenseMatmul)];
    operators.extend(encoder_layer(
        OperatorClass::LayerNorm,
        OperatorClass::GeluProjection,
        false,
    ));
    operators.extend(molmo_pool_and_projector());
    operators
}

/// `youtu_vl_runtime.rs`, main and diagnostics graphs (`emitter/youtu_vl.rs`).
pub(crate) fn youtu_vl_operators() -> Vec<DeclaredOperator> {
    let mut operators = vec![weighted("patch-embedding", OperatorClass::DenseMatmul)];
    operators.extend(encoder_layer(
        OperatorClass::LayerNorm,
        OperatorClass::GeluProjection,
        false,
    ));
    operators.extend([
        norm("merger.post-layer-norm", OperatorClass::LayerNorm),
        norm("merger.rms-norm", OperatorClass::RmsNorm),
        weighted("merger.gelu-projection", OperatorClass::GeluProjection),
        weighted("merger.output-projection", OperatorClass::DenseMatmul),
    ]);
    operators
}

/// `gemma3n_audio_runtime.rs` encode graph (`emitter/gemma3n_audio_*.rs`). The
/// subsampler accumulates its BF16 convolution in F32 and rounds the result to
/// BF16, and its cumulative group norm runs the shared MLX CUDA prefix scan.
pub(crate) fn gemma3n_audio_encode_operators() -> Vec<DeclaredOperator> {
    let bf16 = NumericDType::Bf16;
    let mut operators = vec![
        float_kernel("subsample.convolution", OperatorClass::Convolution).rounded_to(bf16),
        DeclaredOperator::new(
            "subsample.cumulative-group-norm",
            OperatorClass::PrefixReduction,
            Activation,
            PinnedF32,
            CudaOneBlockScan,
        ),
        weighted("subsample.input-projection", OperatorClass::DenseMatmul).rounded_to(bf16),
        norm("conformer.rms-norm", OperatorClass::RmsNorm).rounded_to(bf16),
        weighted(
            "conformer.ffw-silu-projection",
            OperatorClass::SiluProjection,
        ),
        weighted("conformer.ffw-down-projection", OperatorClass::DenseMatmul),
        weighted("conformer.attention-projection", OperatorClass::DenseMatmul),
    ];
    operators.extend(attention(
        "conformer.attention-scores",
        "conformer.attention-softmax",
        "conformer.attention-context",
    ));
    operators.extend([
        float_kernel("conformer.light-conv-depthwise", OperatorClass::Convolution),
        residual("conformer.residual-add"),
        norm("embedder.rms-norm", OperatorClass::RmsNorm),
        weighted("embedder.projection", OperatorClass::DenseMatmul),
    ]);
    operators
}

/// `gemma3n_audio_runtime.rs` merge graph: scaled token embeddings rounded to
/// BF16, then the per-layer-embedding projection and its RMS norm.
pub(crate) fn gemma3n_audio_merge_operators() -> Vec<DeclaredOperator> {
    let bf16 = NumericDType::Bf16;
    vec![
        weighted("per-layer-projection", OperatorClass::DenseMatmul).rounded_to(bf16),
        norm("per-layer-projection-norm", OperatorClass::RmsNorm).rounded_to(bf16),
        residual("per-layer-combine"),
    ]
}

/// `phi4_audio.rs`, encoder and diagnostics graphs (`emitter/phi4_audio.rs`).
pub(crate) fn phi4_audio_operators() -> Vec<DeclaredOperator> {
    let mut operators = vec![
        float_kernel("subsample.convolution", OperatorClass::Convolution),
        norm("conformer.layer-norm", OperatorClass::LayerNorm),
        weighted(
            "conformer.ffn-silu-projection",
            OperatorClass::SiluProjection,
        ),
        weighted("conformer.ffn-down-projection", OperatorClass::DenseMatmul),
        weighted("conformer.attention-projection", OperatorClass::DenseMatmul),
    ];
    operators.extend(attention(
        "conformer.attention-scores",
        "conformer.attention-softmax",
        "conformer.attention-context",
    ));
    operators.extend([
        float_kernel(
            "conformer.pointwise-convolution",
            OperatorClass::Convolution,
        ),
        float_kernel(
            "conformer.depthwise-convolution",
            OperatorClass::Convolution,
        ),
        residual("conformer.residual-add"),
        weighted("projector.gelu-projection", OperatorClass::GeluProjection),
        weighted("projector.output-projection", OperatorClass::DenseMatmul),
    ]);
    operators
}
