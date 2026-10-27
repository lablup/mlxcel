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

//! A synthetic two-module `transformer-packed-mflux/`: a listed ternary
//! linear stored as upstream ships it (`uint32` packed weight, bf16 scales and
//! biases) and an unlisted dense one.

use std::collections::HashMap;
use std::path::Path;

use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde_json::json;

use super::*;
use crate::imagegen::bonsai::config::{PACKED_FORMAT, PackedQuantConfig};

const IN: usize = 256;
const OUT: usize = 8;
/// Listed in `quantized_modules` under its file name; the transformer asks
/// for it as `...to_out` (`.to_out.0.` is renamed).
const PACKED: &str = "transformer_blocks.0.attn.to_out.0";
const PACKED_RENAMED: &str = "transformer_blocks.0.attn.to_out";
const DENSE: &str = "proj_out";

struct Tensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn bf16_ones(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor {
        dtype: Dtype::BF16,
        shape: shape.to_vec(),
        data: std::iter::repeat_n(0x3f80_u16.to_le_bytes(), n)
            .flatten()
            .collect(),
    }
}

/// `{name: tensor}` for the packed module (2-bit, group 128) and the dense one.
fn tensors(packed_scales: bool, dense_scales: bool) -> HashMap<String, Tensor> {
    let mut t = HashMap::new();
    let packed_in = IN * 2 / 32;
    t.insert(
        format!("{PACKED}.weight"),
        Tensor {
            dtype: Dtype::U32,
            shape: vec![OUT, packed_in],
            data: vec![0x55; OUT * packed_in * 4],
        },
    );
    if packed_scales {
        t.insert(format!("{PACKED}.scales"), bf16_ones(&[OUT, IN / 128]));
    }
    t.insert(format!("{PACKED}.biases"), bf16_ones(&[OUT, IN / 128]));
    t.insert(format!("{DENSE}.weight"), bf16_ones(&[OUT, IN]));
    if dense_scales {
        t.insert(format!("{DENSE}.scales"), bf16_ones(&[OUT, IN / 128]));
    }
    t
}

/// Write the packed directory and load it back the way the transformer does.
fn packed_dir(dir: &Path, tensors: &HashMap<String, Tensor>, quant: serde_json::Value) {
    std::fs::write(
        dir.join(super::QUANT_CONFIG_FILE),
        serde_json::to_vec(&quant).unwrap(),
    )
    .unwrap();
    let views: HashMap<String, TensorView<'_>> = tensors
        .iter()
        .map(|(k, t)| {
            (
                k.clone(),
                TensorView::new(t.dtype, t.shape.clone(), &t.data).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(
        &views,
        None,
        &dir.join("diffusion_pytorch_model.safetensors"),
    )
    .unwrap();
}

fn quant(format: &str, bits: i64) -> serde_json::Value {
    json!({
        "format": format,
        "bits": bits,
        "group_size": 128,
        "quantized_modules": [PACKED],
    })
}

fn load(dir: &Path) -> (PackedLinears, WeightMap) {
    let source = PackedLinears::new(PackedQuantConfig::from_dir(dir).unwrap());
    let raw = mlxcel_core::weights::load_weights_from_dir(dir).unwrap();
    (source, raw)
}

#[test]
fn listed_modules_load_as_2_bit_group_128_and_the_rest_dense() {
    let tmp = tempfile::tempdir().unwrap();
    packed_dir(tmp.path(), &tensors(true, false), quant(PACKED_FORMAT, 2));
    let (source, raw) = load(tmp.path());
    source.check_weights(&raw).unwrap();
    let weights = crate::imagegen::flux2::transformer::prepare_transformer_weights(
        raw,
        mlxcel_core::dtype::BFLOAT16,
    );

    let packed = source.load_linear(&weights, PACKED_RENAMED).unwrap();
    let qw = packed
        .quantized_weight()
        .expect("listed module is quantized");
    assert_eq!((qw.bits, qw.group_size), (2, 128));
    assert_eq!(qw.mode, "affine");
    // Scales and biases stay as stored.
    assert_eq!(
        mlxcel_core::array_dtype(&qw.scales),
        mlxcel_core::dtype::BFLOAT16
    );

    let dense = source.load_linear(&weights, DENSE).unwrap();
    assert!(!dense.is_quantized());

    // The packed layer runs: every 2-bit code 0b01 is 1 * scale + bias = 2.
    let x = mlxcel_core::full_f32(&[1, IN as i32], 1.0, mlxcel_core::dtype::BFLOAT16);
    let y = mlxcel_core::utils::array_to_vec_f32(&packed.forward(&x));
    assert_eq!(y.len(), OUT);
    assert!(
        y.iter().all(|v| (*v - 2.0 * IN as f32).abs() < 1.0),
        "{y:?}"
    );
}

#[test]
fn an_unsupported_format_and_one_bit_are_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    packed_dir(tmp.path(), &tensors(true, false), quant("gguf", 2));
    let err = PackedQuantConfig::from_dir(tmp.path()).unwrap_err();
    assert!(format!("{err:#}").contains("\"gguf\""), "{err:#}");

    packed_dir(tmp.path(), &tensors(true, false), quant(PACKED_FORMAT, 1));
    let err = PackedQuantConfig::from_dir(tmp.path()).unwrap_err();
    assert!(format!("{err:#}").contains("bits (1)"), "{err:#}");
}

#[test]
fn a_listed_module_without_scales_errors_naming_it() {
    let tmp = tempfile::tempdir().unwrap();
    packed_dir(tmp.path(), &tensors(false, false), quant(PACKED_FORMAT, 2));
    let (source, raw) = load(tmp.path());
    let err = source.check_weights(&raw).unwrap_err().to_string();
    assert!(err.contains(PACKED) && err.contains(".scales"), "{err}");
    let weights = crate::imagegen::flux2::transformer::prepare_transformer_weights(
        raw,
        mlxcel_core::dtype::BFLOAT16,
    );
    let err = source.load_linear(&weights, PACKED_RENAMED).err().unwrap();
    assert!(err.to_string().contains(PACKED_RENAMED), "{err}");
}

#[test]
fn an_unlisted_module_with_scales_errors() {
    let tmp = tempfile::tempdir().unwrap();
    packed_dir(tmp.path(), &tensors(true, true), quant(PACKED_FORMAT, 2));
    let (source, raw) = load(tmp.path());
    let err = source.check_weights(&raw).unwrap_err().to_string();
    assert!(err.contains(DENSE) && err.contains("not in"), "{err}");
    let err = source.load_linear(&raw, DENSE).err().unwrap().to_string();
    assert!(err.contains(DENSE) && err.contains("not in"), "{err}");
}
