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

//! Pinned operator sets of every production family (issue #3211).
//!
//! Each family is built at its default CUDA and CPU precision. The pins fail
//! when an operator, an input cast, an output rounding, or an association
//! policy is added, removed, or changed.

use crate::emitter::{Precision, default_precision};
use crate::numeric_dtype_contract::NumericDType;

use super::CheckpointDType;
use super::families::{
    gemma3n_audio_encode_operators, gemma3n_audio_merge_operators, llava_vision_operators,
    molmo_vision_operators, molmo2_vision_operators, phi4_audio_operators, qwen2_vl_operators,
    qwen3_vl_operators, youtu_vl_operators,
};
use super::production::{
    AffineExecution, CheckpointSourceObserver, CheckpointWeights, DeclaredOperator,
    ProductionBackend, production_operator_contract,
};

const CUDA: &str = "cuda";
const CPU: &str = "local-task";

fn float(dtype: CheckpointDType) -> CheckpointWeights {
    let mut observer = CheckpointSourceObserver::default();
    observer.observe_float(dtype).unwrap();
    observer.resolve(AffineExecution::Host).unwrap()
}

fn affine(
    float: CheckpointDType,
    bits: usize,
    scale: NumericDType,
    execution: AffineExecution,
) -> CheckpointWeights {
    let mut observer = CheckpointSourceObserver::default();
    observer.observe_float(float).unwrap();
    observer.observe_affine(bits, 64, scale).unwrap();
    observer.resolve(execution).unwrap()
}

/// One line per entry without the backend identity, which is pinned by the
/// cache-rejection tests instead.
fn summary(
    family: &str,
    operators: &[DeclaredOperator],
    precision: Precision,
    weights: CheckpointWeights,
    device: &str,
) -> Vec<String> {
    let backend = ProductionBackend::new("iree-test", "runtime-test", device);
    let set =
        production_operator_contract(family, operators, precision, weights, &backend).unwrap();
    let identity = set.canonical_identity();
    let target = format!("target={device};");
    identity
        .split('\0')
        .skip(1)
        .map(|entry| {
            assert!(entry.contains(&target), "{entry} must name {device}");
            let (operator, _) = entry.split_once(";framework=").unwrap();
            let field = |name: &str| {
                operator
                    .split(';')
                    .find_map(|part| part.strip_prefix(&format!("{name}=")))
                    .unwrap()
                    .to_string()
            };
            let id = field("id");
            format!(
                "{} {} {} {}>{} {} {}",
                id.strip_prefix(&format!("{family}.")).unwrap(),
                field("class"),
                field("checkpoint"),
                field("input"),
                field("result"),
                field("association"),
                field("weights"),
            )
        })
        .collect()
}

/// Families with one fixed precision must declare the same operators on both
/// devices; returns that shared summary.
fn fixed(
    family: &str,
    operators: &[DeclaredOperator],
    precision: Precision,
    weights: CheckpointWeights,
) -> Vec<String> {
    let cuda = summary(family, operators, precision, weights, CUDA);
    assert_eq!(cuda, summary(family, operators, precision, weights, CPU));
    cuda
}

const LLAVA: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul f16 f32>f32 backend-algorithm dense",
    "encoder.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection gelu-projection f16 f32>f32 backend-algorithm dense",
    "encoder.mlp-down-projection dense-matmul f16 f32>f32 backend-algorithm dense",
    "encoder.norm layer-norm f16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "patch-embedding convolution f16 f32>f32 backend-algorithm dense",
    "projector.gelu-projection gelu-projection f16 f32>f32 backend-algorithm dense",
    "projector.output-projection dense-matmul f16 f32>f32 backend-algorithm dense",
];

const GEMMA3: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "encoder.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection gelu-projection bf16 f32>f32 backend-algorithm dense",
    "encoder.mlp-down-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "encoder.norm layer-norm bf16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "patch-embedding convolution bf16 f32>f32 backend-algorithm dense",
    "projector.output-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "projector.post-layer-norm layer-norm bf16 f32>f32 backend-algorithm dense",
    "projector.rms-norm rms-norm bf16 f32>f32 backend-algorithm dense",
];

const QWEN2_CUDA: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f16>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul affine-u4 f16>f32 backend-algorithm host-affine-dequantized",
    "encoder.attention-scores dense-matmul f32 f16>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection gelu-projection affine-u4 f16>f32 backend-algorithm host-affine-dequantized",
    "encoder.mlp-down-projection dense-matmul affine-u4 f16>f32 backend-algorithm host-affine-dequantized",
    "encoder.norm layer-norm f16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "merger.gelu-projection gelu-projection affine-u4 f16>f32 backend-algorithm host-affine-dequantized",
    "merger.norm layer-norm f16 f32>f32 backend-algorithm dense",
    "merger.output-projection dense-matmul affine-u4 f16>f32 backend-algorithm host-affine-dequantized",
    "patch-embedding dense-matmul f16 f16>f32 backend-algorithm dense",
];

const QWEN2_5_CUDA: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f16>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul f16 f16>f32 backend-algorithm dense",
    "encoder.attention-scores dense-matmul f32 f16>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection silu-projection f16 f16>f32 backend-algorithm dense",
    "encoder.mlp-down-projection dense-matmul f16 f16>f32 backend-algorithm dense",
    "encoder.mlp-up-projection dense-matmul f16 f16>f32 backend-algorithm dense",
    "encoder.norm rms-norm f16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "merger.gelu-projection gelu-projection f16 f16>f32 backend-algorithm dense",
    "merger.norm rms-norm f16 f32>f32 backend-algorithm dense",
    "merger.output-projection dense-matmul f16 f16>f32 backend-algorithm dense",
    "patch-embedding dense-matmul f16 f16>f32 backend-algorithm dense",
];

const QWEN2_CPU: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul affine-u4 f32>f32 backend-algorithm host-affine-dequantized",
    "encoder.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection gelu-projection affine-u4 f32>f32 backend-algorithm host-affine-dequantized",
    "encoder.mlp-down-projection dense-matmul affine-u4 f32>f32 backend-algorithm host-affine-dequantized",
    "encoder.norm layer-norm f16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "merger.gelu-projection gelu-projection affine-u4 f32>f32 backend-algorithm host-affine-dequantized",
    "merger.norm layer-norm f16 f32>f32 backend-algorithm dense",
    "merger.output-projection dense-matmul affine-u4 f32>f32 backend-algorithm host-affine-dequantized",
    "patch-embedding dense-matmul f16 f32>f32 backend-algorithm dense",
];

const QWEN2_5_CPU: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul f16 f32>f32 backend-algorithm dense",
    "encoder.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection silu-projection f16 f32>f32 backend-algorithm dense",
    "encoder.mlp-down-projection dense-matmul f16 f32>f32 backend-algorithm dense",
    "encoder.mlp-up-projection dense-matmul f16 f32>f32 backend-algorithm dense",
    "encoder.norm rms-norm f16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "merger.gelu-projection gelu-projection f16 f32>f32 backend-algorithm dense",
    "merger.norm rms-norm f16 f32>f32 backend-algorithm dense",
    "merger.output-projection dense-matmul f16 f32>f32 backend-algorithm dense",
    "patch-embedding dense-matmul f16 f32>f32 backend-algorithm dense",
];

const QWEN3: &[&str] = &[
    "deepstack-merger.gelu-projection gelu-projection bf16 f32>f32 backend-algorithm dense",
    "deepstack-merger.norm layer-norm bf16 f32>f32 backend-algorithm dense",
    "deepstack-merger.output-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "encoder.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "encoder.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection gelu-projection bf16 f32>f32 backend-algorithm dense",
    "encoder.mlp-down-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "encoder.norm layer-norm bf16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "merger.gelu-projection gelu-projection bf16 f32>f32 backend-algorithm dense",
    "merger.norm layer-norm bf16 f32>f32 backend-algorithm dense",
    "merger.output-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "patch-embedding dense-matmul bf16 f32>f32 backend-algorithm dense",
];

const MOLMO: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul affine-u4 f32>f32 backend-algorithm packed-affine-in-graph",
    "encoder.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection gelu-projection affine-u4 f32>f32 backend-algorithm packed-affine-in-graph",
    "encoder.mlp-down-projection dense-matmul affine-u4 f32>f32 backend-algorithm packed-affine-in-graph",
    "encoder.norm layer-norm f16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "patch-embedding dense-matmul f16 f32>f32 backend-algorithm dense",
    "pool.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "pool.attention-projection dense-matmul affine-u4 f32>f32 backend-algorithm packed-affine-in-graph",
    "pool.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "pool.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "projector.down-projection dense-matmul affine-u4 f32>f32 backend-algorithm packed-affine-in-graph",
    "projector.silu-projection silu-projection affine-u4 f32>f32 backend-algorithm packed-affine-in-graph",
    "projector.up-projection dense-matmul affine-u4 f32>f32 backend-algorithm packed-affine-in-graph",
    "weight-dequantize affine-q4-dequantize affine-u4 f32>f32 sequential packed-affine-in-graph",
];

const MOLMO2: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection gelu-projection f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-down-projection dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.norm layer-norm f32 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "patch-embedding dense-matmul f32 f32>f32 backend-algorithm dense",
    "pool.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "pool.attention-projection dense-matmul f32 f32>f32 backend-algorithm dense",
    "pool.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "pool.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "projector.down-projection dense-matmul f32 f32>f32 backend-algorithm dense",
    "projector.silu-projection silu-projection f32 f32>f32 backend-algorithm dense",
    "projector.up-projection dense-matmul f32 f32>f32 backend-algorithm dense",
];

const YOUTU: &[&str] = &[
    "encoder.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "encoder.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "encoder.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "encoder.mlp-activation-projection gelu-projection bf16 f32>f32 backend-algorithm dense",
    "encoder.mlp-down-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "encoder.norm layer-norm bf16 f32>f32 backend-algorithm dense",
    "encoder.residual-add residual-add f32 f32>f32 sequential dense",
    "merger.gelu-projection gelu-projection bf16 f32>f32 backend-algorithm dense",
    "merger.output-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "merger.post-layer-norm layer-norm bf16 f32>f32 backend-algorithm dense",
    "merger.rms-norm rms-norm bf16 f32>f32 backend-algorithm dense",
    "patch-embedding dense-matmul bf16 f32>f32 backend-algorithm dense",
];

const GEMMA3N_ENCODE: &[&str] = &[
    "conformer.attention-context dense-matmul f32 bf16>f32 backend-algorithm dense",
    "conformer.attention-projection dense-matmul affine-u4 bf16>f32 backend-algorithm host-affine-dequantized",
    "conformer.attention-scores dense-matmul f32 bf16>f32 backend-algorithm dense",
    "conformer.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "conformer.ffw-down-projection dense-matmul affine-u4 bf16>f32 backend-algorithm host-affine-dequantized",
    "conformer.ffw-silu-projection silu-projection affine-u4 bf16>f32 backend-algorithm host-affine-dequantized",
    "conformer.light-conv-depthwise convolution bf16 bf16>f32 backend-algorithm dense",
    "conformer.residual-add residual-add f32 f32>f32 sequential dense",
    "conformer.rms-norm rms-norm bf16 f32>bf16 backend-algorithm dense",
    "embedder.projection dense-matmul affine-u4 bf16>f32 backend-algorithm host-affine-dequantized",
    "embedder.rms-norm rms-norm bf16 f32>f32 backend-algorithm dense",
    "subsample.convolution convolution bf16 bf16>bf16 backend-algorithm dense",
    "subsample.cumulative-group-norm prefix-reduction f32 f32>f32 cuda-one-block-scan dense",
    "subsample.input-projection dense-matmul affine-u4 bf16>bf16 backend-algorithm host-affine-dequantized",
];

const GEMMA3N_MERGE: &[&str] = &[
    "per-layer-combine residual-add f32 f32>f32 sequential dense",
    "per-layer-projection dense-matmul affine-u4 bf16>bf16 backend-algorithm host-affine-dequantized",
    "per-layer-projection-norm rms-norm bf16 f32>bf16 backend-algorithm dense",
];

const PHI4: &[&str] = &[
    "conformer.attention-context dense-matmul f32 f32>f32 backend-algorithm dense",
    "conformer.attention-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "conformer.attention-scores dense-matmul f32 f32>f32 backend-algorithm dense",
    "conformer.attention-softmax attention-softmax f32 f32>f32 backend-algorithm dense",
    "conformer.depthwise-convolution convolution bf16 f32>f32 backend-algorithm dense",
    "conformer.ffn-down-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "conformer.ffn-silu-projection silu-projection bf16 f32>f32 backend-algorithm dense",
    "conformer.layer-norm layer-norm bf16 f32>f32 backend-algorithm dense",
    "conformer.pointwise-convolution convolution bf16 f32>f32 backend-algorithm dense",
    "conformer.residual-add residual-add f32 f32>f32 sequential dense",
    "projector.gelu-projection gelu-projection bf16 f32>f32 backend-algorithm dense",
    "projector.output-projection dense-matmul bf16 f32>f32 backend-algorithm dense",
    "subsample.convolution convolution bf16 f32>f32 backend-algorithm dense",
];

fn assert_pinned(name: &str, actual: Vec<String>, expected: &[&str]) {
    let actual = actual.iter().map(String::as_str).collect::<Vec<_>>();
    assert_eq!(actual, expected, "{name} operator set drifted");
}

#[test]
fn vision_runtime_families_pin_llava_and_gemma3_operator_sets() {
    let llava = llava_vision_operators(false);
    let f16 = float(CheckpointDType::F16);
    assert_pinned(
        "llava",
        fixed("llava-vision", &llava, Precision::F32, f16),
        LLAVA,
    );
    let gemma3 = llava_vision_operators(true);
    let bf16 = float(CheckpointDType::Bf16);
    assert_pinned(
        "gemma3",
        fixed("llava-vision", &gemma3, Precision::F32, bf16),
        GEMMA3,
    );
}

#[test]
fn qwen2_vl_families_follow_the_device_default_precision() {
    let q4 = affine(
        CheckpointDType::F16,
        4,
        NumericDType::F16,
        AffineExecution::Host,
    );
    let f16 = float(CheckpointDType::F16);
    assert_eq!(default_precision(CUDA), Precision::F16);
    assert_eq!(default_precision(CPU), Precision::F32);
    for (device, qwen2, qwen2_5) in [
        (CUDA, QWEN2_CUDA, QWEN2_5_CUDA),
        (CPU, QWEN2_CPU, QWEN2_5_CPU),
    ] {
        let precision = default_precision(device);
        let ops = qwen2_vl_operators(false);
        assert_pinned("qwen2-vl", summary("q", &ops, precision, q4, device), qwen2);
        let ops = qwen2_vl_operators(true);
        assert_pinned(
            "qwen2.5-vl",
            summary("q", &ops, precision, f16, device),
            qwen2_5,
        );
    }
}

#[test]
fn fixed_precision_vision_families_pin_their_operator_sets() {
    let bf16 = float(CheckpointDType::Bf16);
    let ops = qwen3_vl_operators();
    assert_pinned("qwen3-vl", fixed("q", &ops, Precision::F32, bf16), QWEN3);
    let packed = affine(
        CheckpointDType::F16,
        4,
        NumericDType::F16,
        AffineExecution::InGraph,
    );
    let ops = molmo_vision_operators(4);
    assert_pinned("molmo", fixed("m", &ops, Precision::F32, packed), MOLMO);
    let ops = molmo2_vision_operators();
    let f32 = float(CheckpointDType::F32);
    assert_pinned("molmo2", fixed("m", &ops, Precision::F32, f32), MOLMO2);
    let ops = youtu_vl_operators();
    assert_pinned("youtu-vl", fixed("y", &ops, Precision::F32, bf16), YOUTU);
}

#[test]
fn audio_families_pin_their_operator_sets() {
    let q4 = affine(
        CheckpointDType::Bf16,
        4,
        NumericDType::Bf16,
        AffineExecution::HostBf16Sequential,
    );
    let ops = gemma3n_audio_encode_operators();
    assert_pinned(
        "gemma3n-encode",
        fixed("g", &ops, Precision::Bf16, q4),
        GEMMA3N_ENCODE,
    );
    let ops = gemma3n_audio_merge_operators();
    assert_pinned(
        "gemma3n-merge",
        fixed("g", &ops, Precision::Bf16, q4),
        GEMMA3N_MERGE,
    );
    let ops = phi4_audio_operators();
    let bf16 = float(CheckpointDType::Bf16);
    assert_pinned("phi4", fixed("p", &ops, Precision::F32, bf16), PHI4);
}

#[test]
fn molmo_eight_bit_checkpoints_declare_q8_dequantization() {
    let packed = affine(
        CheckpointDType::F16,
        8,
        NumericDType::F16,
        AffineExecution::InGraph,
    );
    let lines = fixed("m", &molmo_vision_operators(8), Precision::F32, packed);
    assert!(lines.contains(
        &"weight-dequantize affine-q8-dequantize affine-u8 f32>f32 sequential packed-affine-in-graph"
            .to_string()
    ));
}

#[test]
fn precision_mismatch_between_emitter_and_contract_changes_identity() {
    let backend = ProductionBackend::new("iree-test", "runtime-test", CUDA);
    let ops = qwen2_vl_operators(true);
    let f16 = float(CheckpointDType::F16);
    let build = |precision| {
        production_operator_contract("q", &ops, precision, f16, &backend)
            .unwrap()
            .canonical_identity()
    };
    assert_ne!(build(Precision::F16), build(Precision::F32));
    assert_ne!(build(Precision::F16), build(Precision::Bf16));
}

#[test]
fn removing_an_operator_or_policy_breaks_the_pin() {
    let mut ops = qwen3_vl_operators();
    let bf16 = float(CheckpointDType::Bf16);
    let removed = ops.pop().unwrap();
    assert_ne!(fixed("q", &ops, Precision::F32, bf16), QWEN3);
    ops.push(DeclaredOperator {
        association: super::AssociationPolicy::Sequential,
        ..removed
    });
    assert_ne!(fixed("q", &ops, Precision::F32, bf16), QWEN3);
}

#[test]
fn host_dequantization_rounding_is_part_of_the_identity() {
    // Gemma3n rounds each host dequantization step to BF16; switching it to
    // the F32 multiply-then-add must change the digest even though the operator
    // dtypes and weight execution stay the same.
    let backend = ProductionBackend::new("iree-test", "runtime-test", CUDA);
    let ops = gemma3n_audio_encode_operators();
    let build = |execution| {
        let weights = affine(CheckpointDType::Bf16, 4, NumericDType::Bf16, execution);
        production_operator_contract("g", &ops, Precision::Bf16, weights, &backend)
            .unwrap()
            .canonical_identity()
    };
    let bf16_rounded = build(AffineExecution::HostBf16Sequential);
    assert!(bf16_rounded.contains("evaluation=separate-multiply-then-add-bf16-rounded"));
    assert_ne!(bf16_rounded, build(AffineExecution::Host));
}
