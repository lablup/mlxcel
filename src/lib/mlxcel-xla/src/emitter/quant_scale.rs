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

//! Element type of the MLX affine `scales` / `biases` on the packed in-graph
//! dequant path (issue #603).
//!
//! `config.json` does not record whether a checkpoint stores its affine
//! metadata as f16 or bf16 (mlx-lm writes bf16 for Qwen3, Gemma3-27B and
//! Qwen3-MoE), so the loader reads it from the safetensors header and stores
//! the result in [`Config::quant_scale_dtype`](super::Config). That one field
//! decides both the emitted argument type (`take_weight`, `take_lm_head`) and
//! the uploaded buffer's dtype code, so the graph and its buffers cannot drift.

/// Element type of the packed path's `scales` / `biases` arguments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum QuantScaleDtype {
    /// IEEE half, the mlx-lm default and the pre-#603 packed path.
    #[default]
    F16,
    /// bfloat16, which mlx-lm stores for some families.
    Bf16,
}

impl QuantScaleDtype {
    /// The StableHLO element type the emitter declares for the argument.
    #[must_use]
    pub fn elt(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// The per-weight dtype code the loader hands to `xla_upload_one_weight_impl`
    /// in `csrc/xla_iree.c` (0 f32, 1 f16, 2 packed u32, 3 bf16).
    #[must_use]
    pub fn weight_dtype_code(self) -> i32 {
        match self {
            Self::F16 => 1,
            Self::Bf16 => 3,
        }
    }
}

/// Resolve the single scale dtype of the packed path from the dtype of every
/// `scales` / `biases` tensor it uploads. Each entry is the tensor name and its
/// dtype, `None` when the dtype is neither f16 nor bf16 (named in the error by
/// `raw`). An empty list keeps the f16 default, so an unquantized checkpoint is
/// unaffected.
///
/// A checkpoint mixing f16 and bf16 metadata is rejected: the emitter declares
/// one element type for every packed projection, and a per-tensor type would
/// fork the compiled graph for a layout no known converter writes. The caller
/// falls back to the dequant-at-load path with a warning.
pub(crate) fn resolve_quant_scale_dtype<'a, I>(parts: I) -> Result<QuantScaleDtype, String>
where
    I: IntoIterator<Item = (&'a str, Option<QuantScaleDtype>, &'a str)>,
{
    let mut first: Option<(&str, QuantScaleDtype)> = None;
    for (name, dtype, raw) in parts {
        let Some(dtype) = dtype else {
            return Err(format!(
                "{name} is {raw}; the packed path accepts F16 or BF16 scales/biases"
            ));
        };
        match first {
            None => first = Some((name, dtype)),
            Some((first_name, first_dtype)) if first_dtype != dtype => {
                return Err(format!(
                    "mixed affine metadata dtypes: {first_name} is {first_dtype:?} but {name} \
                     is {dtype:?}; the packed path needs one scale dtype per checkpoint"
                ));
            }
            Some(_) => {}
        }
    }
    Ok(first.map_or(QuantScaleDtype::default(), |(_, dtype)| dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_bf16_metadata_resolves_bf16() {
        let parts = [
            ("a.scales", Some(QuantScaleDtype::Bf16), "BF16"),
            ("a.biases", Some(QuantScaleDtype::Bf16), "BF16"),
        ];
        assert_eq!(resolve_quant_scale_dtype(parts), Ok(QuantScaleDtype::Bf16));
    }

    #[test]
    fn empty_and_f16_metadata_keep_the_f16_default() {
        assert_eq!(resolve_quant_scale_dtype([]), Ok(QuantScaleDtype::F16));
        let parts = [("a.scales", Some(QuantScaleDtype::F16), "F16")];
        assert_eq!(resolve_quant_scale_dtype(parts), Ok(QuantScaleDtype::F16));
    }

    #[test]
    fn mixed_metadata_is_rejected_naming_both_tensors() {
        let parts = [
            ("a.scales", Some(QuantScaleDtype::F16), "F16"),
            ("b.scales", Some(QuantScaleDtype::Bf16), "BF16"),
        ];
        let error = resolve_quant_scale_dtype(parts).unwrap_err();
        assert!(
            error.contains("a.scales") && error.contains("b.scales"),
            "{error}"
        );
    }

    #[test]
    fn unsupported_metadata_dtype_is_rejected() {
        let parts = [("a.scales", None, "F32")];
        let error = resolve_quant_scale_dtype(parts).unwrap_err();
        assert!(error.contains("a.scales is F32"), "{error}");
    }

    /// The packed graph declares the scales/biases with the config's dtype and
    /// nothing else moves: with f32 contractions the only f16 types in the f16
    /// form are that metadata, so swapping them to bf16 yields the bf16 graph.
    #[test]
    fn packed_graphs_declare_the_config_scale_dtype() {
        use super::super::builder::{Precision, ProjectionLayout};
        use super::super::{Config, QuantConfig, emit_decode_with, emit_prefill_with};
        for layout in [ProjectionLayout::OutIn, ProjectionLayout::InOut] {
            let mut c = Config::llama_3_2_1b();
            c.projection_layout = layout;
            c.packed_args = true;
            c.quantization = Some(QuantConfig {
                bits: 4,
                group_size: 64,
            });
            let f16_decode = emit_decode_with(&c, true, Precision::F32);
            let f16_prefill = emit_prefill_with(&c, true, Precision::F32);
            assert!(
                f16_decode
                    .contains("tensor<8192x32xf16> loc(\"params['layers'][0]['gate'].scales\")"),
                "the default packed graph keeps f16 metadata"
            );
            c.quant_scale_dtype = QuantScaleDtype::Bf16;
            let bf16_decode = emit_decode_with(&c, true, Precision::F32);
            let bf16_prefill = emit_prefill_with(&c, true, Precision::F32);
            assert!(
                bf16_decode
                    .contains("tensor<8192x32xbf16> loc(\"params['layers'][0]['gate'].biases\")"),
                "bf16 metadata is declared bf16"
            );
            assert!(!bf16_decode.contains("xf16>"), "no f16 type remains");
            assert_eq!(f16_decode.replace("xf16>", "xbf16>"), bf16_decode);
            assert_eq!(f16_prefill.replace("xf16>", "xbf16>"), bf16_prefill);
        }
    }

    /// The Rust dtype codes must match the `switch` in the C shim, which is the
    /// only consumer; a drift would upload bf16 bytes as another element type.
    #[test]
    fn weight_dtype_codes_match_the_c_shim() {
        let shim = include_str!("../../csrc/xla_iree.c");
        let case = |code: i32, elt: &str| {
            let needle = format!("case {code}:\n      elt = {elt};\n      esize = 2;");
            assert!(shim.contains(&needle), "xla_iree.c lacks `{needle}`");
        };
        case(
            QuantScaleDtype::F16.weight_dtype_code(),
            "IREE_HAL_ELEMENT_TYPE_FLOAT_16",
        );
        case(
            QuantScaleDtype::Bf16.weight_dtype_code(),
            "IREE_HAL_ELEMENT_TYPE_BFLOAT_16",
        );
    }
}
