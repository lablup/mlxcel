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

//! IREE execution path (issue #449 Phase 3 M2), compiled only under the `iree`
//! feature.
//!
//! [`IreeLlama`] is the safe Rust owner of the C shim (`csrc/xla_iree.c`). On
//! [`IreeLlama::load`] it (1) reads the model's `config.json` into an emitter
//! [`Config`], (2) emits the `prefill` / `decode_step` StableHLO graphs from that
//! config (the #451 emitter, ending in an on-device argmax) and compiles them to
//! vmfbs with the dist's `iree-compile`, cached by content hash, (3) loads the
//! weights as f32 (widening bf16 / f16, copying f32, or dequantizing MLX 4 / 8-bit
//! affine weights) in the emitter's arg order, from either a single-file
//! `model.safetensors` or a sharded checkpoint (via its
//! `model.safetensors.index.json`, which is how the big untied models ship), and
//! (4) hands all of it to the
//! shim, which keeps the weights resident on the device and threads the KV cache
//! across steps. Then [`IreeLlama::prefill`] / [`IreeLlama::decode`] are token-in
//! / token-out. Emitting from config (issue #449 M3 Stage 2d) replaced the bundled
//! Llama-3.2-1B `.mlir` assets, so any checkpoint of a supported dense family loads:
//! Llama, Qwen2, Qwen3, Gemma1/2/3, SmolLM3, OLMo2/3, Seed-OSS, MiMo, InternLM3,
//! ExaOne (issues #497 / #499), and the parallel-block / norm-variant pack Cohere,
//! Cohere2, Phi3, StableLM, StarCoder2, Granite, MiniCPM (issue #498). Each family's
//! per-layer weight order in `weight_specs` (`weights.rs`) mirrors the emitter's arg
//! schedule: the q/k/v biases, the Qwen3 / Gemma3 / OLMo2/3 q/k norms, the Gemma2/3
//! feed-forward norms, the OLMo2/3 absence of an `input_layernorm`, the #498
//! LayerNorm / o_proj / MLP biases and dense StarCoder2 MLP, and the fused Phi3
//! `qkv_proj` / `gate_up_proj` (read once and row-sliced into the separate args). An
//! untied checkpoint (`tie_word_embeddings = false`, e.g. Llama-3.1-8B, larger
//! Qwen2.5, OLMo2/3, Phi3, StableLM, MiniCPM) adds its `lm_head.weight`, matching the
//! separate `params['lm_head']` arg the emitter takes for the final projection.
//!
//! Proven token-exact against the HF temp-0 reference in
//! `spike/openxla/artifacts/results.json` before being vendored from the
//! standalone gate (`spike/iree-ffi`).

use std::ffi::CString;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::os::raw::{c_char, c_int, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use memmap2::Mmap;
use mlxcel_core::session::PreparedPrefill;
use safetensors::{Dtype, SafeTensors};

use crate::emitter::{
    Config, DeepStackConfig, Gemma3nConfig, Gemma3nWeightSpec, Precision, QuantConfig,
    check_packed_supported, emit_decode_ragged_with, emit_decode_with,
    emit_gemma3n_decode_ragged_with_qmv, emit_gemma3n_decode_with_qmv,
    emit_gemma3n_prefill_embeddings_ple_with_qmv, emit_gemma3n_prefill_with_qmv,
    emit_prefill_embeddings_deepstack_with, emit_prefill_embeddings_with, emit_prefill_with,
    gemma3n_qmv_artifact_identity, gemma3n_qmv_is_available, gemma3n_weight_specs, quant_in_graph,
    resolve_precision, resolve_precision_checked,
};
#[cfg(feature = "diagnostics")]
use crate::emitter::{
    Gemma3nDiagnosticLayout, emit_gemma3n_all_layer_diagnostics_with_qmv,
    emit_gemma3n_prefill_diagnostics_with_qmv,
};
use crate::prepared::{
    DecodePositionState, PreparedIreePrefill, PreparedPositionMode, canonical_text_positions,
    validate_slot,
};
use crate::prepared_deepstack::PreparedDeepStack;
use crate::{DeepStackFeatures, DeepStackPreparedPrefill, Gemma3nDensePle, Gemma3nPreparedPrefill};
// The loader reads the per-architecture checkpoint-weight order from
// `weights::weight_specs`, which sources its names from `weight_names::scheme_names`
// (issue #499 naming schemes) and covers the #498 dense pack (LayerNorm / o_proj /
// MLP biases, the dense StarCoder2 MLP, and the row-sliced fused Phi3 projections)
// and the #500 MoE expert bank (the stacked `switch_mlp` weights, dequantized with
// `dequantize_affine_stacked`). Both are pure-Rust and unit-tested without `iree`.
use crate::weights::{
    QuantPart, WeightSpec, bf16_to_f32, dequantize_affine, dequantize_affine_bf16_fused,
    dequantize_affine_stacked, f16_to_f32, f32_le_to_f32, pack_f16, slice_rows, weight_specs,
};

/// Weight-buffer element dtype passed to the C shim (issue #516 per-weight ABI):
/// f32 (a widened / dequantized weight), or f16 / packed-U32 for the raw parts of an
/// MLX affine-quantized projection on the packed path. Kept in sync with the
/// `switch` in `csrc/xla_iree.c`.
const WDT_F32: c_int = 0;
const WDT_F16: c_int = 1;
const WDT_U32: c_int = 2;

/// C-side tensor element types. Keep in sync with `xla_tensor_desc` validation
/// in `csrc/xla_iree.c`.
const TENSOR_F32: c_int = 0;
const TENSOR_I32: c_int = 1;

/// Explicit tensor descriptor crossing the C ABI. Every input includes its byte
/// length, dtype, rank, and complete static shape; the shim revalidates these
/// fields before constructing an IREE buffer view.
#[repr(C)]
struct XlaTensorDesc {
    data: *const c_void,
    byte_length: usize,
    dtype: c_int,
    rank: c_int,
    dims: [i64; 4],
}

impl XlaTensorDesc {
    fn f32(data: &[f32], shape: &[usize]) -> Result<Self, String> {
        Self::new(
            data.as_ptr().cast(),
            data.len(),
            std::mem::size_of::<f32>(),
            TENSOR_F32,
            shape,
        )
    }

    fn i32(data: &[i32], shape: &[usize]) -> Result<Self, String> {
        Self::new(
            data.as_ptr().cast(),
            data.len(),
            std::mem::size_of::<i32>(),
            TENSOR_I32,
            shape,
        )
    }

    fn new(
        data: *const c_void,
        elements: usize,
        element_size: usize,
        dtype: c_int,
        shape: &[usize],
    ) -> Result<Self, String> {
        if shape.len() > 4 {
            return Err(format!(
                "IREE tensor descriptor rank {} exceeds 4",
                shape.len()
            ));
        }
        let expected = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| "IREE tensor descriptor element count overflowed".to_string())?;
        if elements != expected {
            return Err(format!(
                "IREE tensor descriptor element count {elements} disagrees with shape {shape:?} ({expected})"
            ));
        }
        let byte_length = elements
            .checked_mul(element_size)
            .ok_or_else(|| "IREE tensor descriptor byte count overflowed".to_string())?;
        let mut dims = [0i64; 4];
        for (index, &dim) in shape.iter().enumerate() {
            dims[index] = i64::try_from(dim)
                .map_err(|_| format!("IREE tensor dimension {dim} does not fit i64"))?;
        }
        Ok(Self {
            data,
            byte_length,
            dtype,
            rank: c_int::try_from(shape.len())
                .map_err(|_| format!("IREE tensor rank {} does not fit c_int", shape.len()))?,
            dims,
        })
    }
}

/// A resident-weight host buffer, kept alive until the shim copies it to the device.
/// f32 values (a widened / dequantized weight, the proven path), f16 values (an
/// f16-resident projection, issue #572), or raw bytes (an MLX packed-U32 weight or
/// its f16 scales / biases, issue #516 packed path).
enum WeightBuf {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Raw(Vec<u8>),
}

impl WeightBuf {
    /// Host pointer to the buffer bytes (the shim copies `nelem * dtype_size` of them).
    fn as_u8_ptr(&self) -> *const u8 {
        match self {
            WeightBuf::F32(v) => v.as_ptr() as *const u8,
            WeightBuf::F16(v) => v.as_ptr() as *const u8,
            WeightBuf::Raw(v) => v.as_ptr(),
        }
    }
}

/// Opaque handle to the C-side execution context.
#[repr(C)]
struct XlaCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn xla_llama_create(
        device: *const c_char,
        prefill: *const c_char,
        prefill_embeddings: *const c_char,
        prefill_embeddings_deepstack: *const c_char,
        decode: *const c_char,
        prefill_diagnostics: *const c_char,
        compatibility_fingerprint: u64,
        n_weights: c_int,
        weight_data: *const *const c_void,
        weight_dtypes: *const c_int,
        weight_ranks: *const c_int,
        weight_dims: *const i64,
        context_capacity: c_int,
        hidden_size: c_int,
        position_mode: c_int,
        prefill_embeddings_kind: c_int,
        dense_ple_layers: c_int,
        dense_ple_hidden: c_int,
        model_layers: c_int,
        deepstack_layers: c_int,
        deepstack_visual_positions: c_int,
        deepstack_target_layers: *const c_int,
    ) -> *mut XlaCtx;
    fn xla_llama_prefill(
        c: *mut XlaCtx,
        tokens: *const c_int,
        lp: c_int,
        positions: *const c_int,
        real_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    fn xla_llama_decode(
        c: *mut XlaCtx,
        token: c_int,
        pos: c_int,
        cache_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    fn xla_llama_decode_mrope(
        c: *mut XlaCtx,
        token: c_int,
        positions: *const c_int,
        cache_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    // Ragged continuous-batching (#449 M3 Stage 2b/2d). `ragged_reset` sizes the
    // batch; the `_logits` calls run one slot's prefill / all B slots' decode and
    // copy the per-row `[vocab]` logits to host so the engine samples there (Stage
    // 2d). `prefill_slot_logits` also does the device-side KV slot write.
    fn xla_llama_ragged_reset(c: *mut XlaCtx, bsz: c_int) -> c_int;
    fn xla_llama_prefill_slot_logits(
        c: *mut XlaCtx,
        slot: c_int,
        tokens: *const c_int,
        lp: c_int,
        positions: *const c_int,
        real_len: c_int,
        vocab: c_int,
        out_logits: *mut f32,
    ) -> c_int;
    #[cfg(feature = "diagnostics")]
    fn xla_llama_prefill_diagnostics_slot(
        c: *mut XlaCtx,
        slot: c_int,
        tokens: *const c_int,
        lp: c_int,
        positions: *const c_int,
        real_len: c_int,
        diagnostic_len: c_int,
        out_diagnostics: *mut f32,
    ) -> c_int;
    fn xla_llama_prefill_embeddings(
        c: *mut XlaCtx,
        position_mode: c_int,
        embeddings: *const XlaTensorDesc,
        positions: *const XlaTensorDesc,
        attention_bias: *const XlaTensorDesc,
        real_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    fn xla_llama_prefill_embeddings_slot_logits(
        c: *mut XlaCtx,
        slot: c_int,
        position_mode: c_int,
        embeddings: *const XlaTensorDesc,
        positions: *const XlaTensorDesc,
        attention_bias: *const XlaTensorDesc,
        real_len: c_int,
        vocab: c_int,
        out_logits: *mut f32,
    ) -> c_int;
    fn xla_llama_prefill_embeddings_ple(
        c: *mut XlaCtx,
        position_mode: c_int,
        embeddings: *const XlaTensorDesc,
        dense_ple: *const XlaTensorDesc,
        positions: *const XlaTensorDesc,
        attention_bias: *const XlaTensorDesc,
        real_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    fn xla_llama_prefill_embeddings_ple_slot_logits(
        c: *mut XlaCtx,
        slot: c_int,
        position_mode: c_int,
        embeddings: *const XlaTensorDesc,
        dense_ple: *const XlaTensorDesc,
        positions: *const XlaTensorDesc,
        attention_bias: *const XlaTensorDesc,
        real_len: c_int,
        vocab: c_int,
        out_logits: *mut f32,
    ) -> c_int;
    fn xla_llama_prefill_embeddings_deepstack(
        c: *mut XlaCtx,
        position_mode: c_int,
        embeddings: *const XlaTensorDesc,
        positions: *const XlaTensorDesc,
        attention_bias: *const XlaTensorDesc,
        visual_positions: *const XlaTensorDesc,
        layer_features: *const XlaTensorDesc,
        layer_indices: *const XlaTensorDesc,
        actual_layer_count: c_int,
        actual_visual_count: c_int,
        real_len: c_int,
        out_token: *mut c_int,
    ) -> c_int;
    fn xla_llama_prefill_embeddings_deepstack_slot_logits(
        c: *mut XlaCtx,
        slot: c_int,
        position_mode: c_int,
        embeddings: *const XlaTensorDesc,
        positions: *const XlaTensorDesc,
        attention_bias: *const XlaTensorDesc,
        visual_positions: *const XlaTensorDesc,
        layer_features: *const XlaTensorDesc,
        layer_indices: *const XlaTensorDesc,
        actual_layer_count: c_int,
        actual_visual_count: c_int,
        real_len: c_int,
        vocab: c_int,
        out_logits: *mut f32,
    ) -> c_int;
    fn xla_llama_decode_ragged_logits(
        c: *mut XlaCtx,
        bsz: c_int,
        tokens: *const c_int,
        pos: *const c_int,
        cache_len: *const c_int,
        vocab: c_int,
        out_logits: *mut f32,
    ) -> c_int;
    fn xla_llama_decode_ragged_mrope_logits(
        c: *mut XlaCtx,
        bsz: c_int,
        tokens: *const c_int,
        positions: *const c_int,
        cache_len: *const c_int,
        vocab: c_int,
        out_logits: *mut f32,
    ) -> c_int;
    fn xla_llama_free(c: *mut XlaCtx);
}

/// The slot counts (`B_max`) the serve worker maps a request's batch size to. The
/// ragged decode graph for the chosen `b_max` is emitted from the model config at
/// load (any `b_max` is emittable; the worker selects from this set).
pub(crate) const RAGGED_B_VALUES: &[usize] = &[4, 8];

#[derive(Clone, Debug)]
enum RuntimeConfig {
    Dense(Box<Config>),
    Gemma3n(Box<Gemma3nConfig>),
}

fn checked_ffi_int(value: usize, name: &str) -> Result<c_int, String> {
    c_int::try_from(value).map_err(|_| format!("{name}={value} does not fit the IREE C ABI c_int"))
}

fn checked_ffi_i64(value: usize, name: &str) -> Result<i64, String> {
    i64::try_from(value).map_err(|_| format!("{name}={value} does not fit the IREE C ABI i64"))
}

#[derive(Debug)]
struct RuntimeFfiDimensions {
    n_weights: c_int,
    context_capacity: c_int,
    hidden: c_int,
    dense_ple_layers: c_int,
    dense_ple_hidden: c_int,
    model_layers: c_int,
    deepstack_layers: c_int,
    deepstack_visual: c_int,
}

fn runtime_ffi_dimensions(
    cfg: &RuntimeConfig,
    n_weights: usize,
) -> Result<RuntimeFfiDimensions, String> {
    let dense_ple = cfg.dense_ple_shape();
    checked_ffi_int(cfg.vocab(), "vocab_size")?;
    Ok(RuntimeFfiDimensions {
        n_weights: checked_ffi_int(n_weights, "n_weights")?,
        context_capacity: checked_ffi_int(cfg.context_capacity(), "context_capacity")?,
        hidden: checked_ffi_int(cfg.hidden(), "hidden_size")?,
        dense_ple_layers: checked_ffi_int(
            dense_ple.map_or(0, |shape| shape[1]),
            "dense_ple_layers",
        )?,
        dense_ple_hidden: checked_ffi_int(
            dense_ple.map_or(0, |shape| shape[2]),
            "dense_ple_hidden",
        )?,
        model_layers: checked_ffi_int(cfg.n_layers(), "model_layers")?,
        deepstack_layers: checked_ffi_int(
            cfg.deepstack()
                .map_or(0, |schema| schema.target_layer_indices.len()),
            "deepstack_layers",
        )?,
        deepstack_visual: checked_ffi_int(
            cfg.deepstack()
                .map_or(0, |schema| schema.max_visual_positions),
            "deepstack_visual_positions",
        )?,
    })
}

impl RuntimeConfig {
    fn from_json(model_dir: &Path, context_capacity: usize) -> Result<Self, String> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        let root: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| format!("parse {}: {error}", path.display()))?;
        let model_type = root.get("model_type").and_then(serde_json::Value::as_str);
        if matches!(model_type, Some("gemma3n" | "gemma3n_text")) {
            return Gemma3nConfig::from_json_str(&text)
                .and_then(|config| config.with_context_capacity(context_capacity))
                .map(Box::new)
                .map(Self::Gemma3n)
                .map_err(|error| format!("{}: {error}", path.display()));
        }
        Config::from_json(model_dir)?
            .with_context_capacity(context_capacity)
            .map(Box::new)
            .map(Self::Dense)
    }

    fn context_capacity(&self) -> usize {
        match self {
            Self::Dense(config) => config.context_capacity,
            Self::Gemma3n(config) => config.context_capacity,
        }
    }

    fn hidden(&self) -> usize {
        match self {
            Self::Dense(config) => config.hidden,
            Self::Gemma3n(config) => config.hidden,
        }
    }

    fn vocab(&self) -> usize {
        match self {
            Self::Dense(config) => config.vocab,
            Self::Gemma3n(config) => config.vocab,
        }
    }

    fn n_layers(&self) -> usize {
        match self {
            Self::Dense(config) => config.n_layers,
            Self::Gemma3n(config) => config.n_layers,
        }
    }

    fn quantization(&self) -> Option<QuantConfig> {
        match self {
            Self::Dense(config) => config.quantization,
            Self::Gemma3n(config) => config.quantization,
        }
    }

    fn weight_specs(&self) -> Vec<WeightSpec> {
        match self {
            Self::Dense(config) => weight_specs(config),
            Self::Gemma3n(config) => gemma3n_weight_specs(config)
                .into_iter()
                .map(|spec| match spec {
                    Gemma3nWeightSpec::Tensor(name) => WeightSpec::Whole(name),
                    Gemma3nWeightSpec::Projection(name) => WeightSpec::Proj(name),
                })
                .collect(),
        }
    }

    fn artifact_identity(&self) -> String {
        match self {
            Self::Dense(config) => format!("dense:{config:?}"),
            Self::Gemma3n(config) => config.compatibility_fingerprint(),
        }
    }

    fn supports_f16_resident(&self) -> bool {
        matches!(self, Self::Dense(config) if config.supports_f16_resident())
    }

    fn dense_ple_shape(&self) -> Option<[usize; 3]> {
        match self {
            Self::Dense(_) => None,
            Self::Gemma3n(config) => Some(config.dense_ple_shape()),
        }
    }

    fn position_mode(&self) -> PreparedPositionMode {
        match self {
            Self::Dense(config) if config.uses_mrope() => PreparedPositionMode::Mrope3D,
            Self::Dense(_) | Self::Gemma3n(_) => PreparedPositionMode::OneD,
        }
    }

    fn deepstack(&self) -> Option<&DeepStackConfig> {
        match self {
            Self::Dense(config) => config.deepstack.as_ref(),
            Self::Gemma3n(_) => None,
        }
    }
}

fn effective_precision(device: &str, cfg: &RuntimeConfig) -> Result<Precision, String> {
    match cfg {
        RuntimeConfig::Dense(_) => resolve_precision_checked(device),
        RuntimeConfig::Gemma3n(_) if device == "metal" => Err(
            "Gemma3n uses an exact BF16 activation stream with F32 AltUp coefficient islands, \
             but the Metal target cannot lower BF16. Use CUDA or a CPU local-task/local-sync \
             target."
                .to_string(),
        ),
        RuntimeConfig::Gemma3n(_) => Ok(Precision::Bf16),
    }
}

fn gemma3n_native_qmv_eligibility(
    device: &str,
    quantization: Option<QuantConfig>,
    qmv_available: bool,
) -> Result<bool, String> {
    if device != "cuda" {
        return Ok(false);
    }
    let Some(quantization) = quantization else {
        return Ok(false);
    };
    if quantization.bits == 8 {
        return Ok(false);
    }
    if quantization.bits != 4 {
        return Err(format!(
            "Gemma3n CUDA native QMV supports 4-bit checkpoints; got {} bits",
            quantization.bits
        ));
    }
    if quantization.group_size != 64 {
        return Err(format!(
            "Gemma3n CUDA native QMV currently pins the Q4 group_size=64 contract; got {}",
            quantization.group_size
        ));
    }
    if !qmv_available {
        return Err(
            "Gemma3n CUDA Q4 requires the token-exact native QMV, but this binary was \
             built without the CUDA-IREE PTX (`cfg(xla_iree_cuda)`). Rebuild with \
             IREE_CUDA_HOME and nvcc; refusing the numerically different dot fallback."
                .to_string(),
        );
    }
    Ok(true)
}

fn gemma3n_native_qmv(device: &str, config: &Gemma3nConfig) -> Result<bool, String> {
    let enabled =
        gemma3n_native_qmv_eligibility(device, config.quantization, gemma3n_qmv_is_available())?;
    if enabled {
        config.validate_native_cuda_dispatches()?;
    }
    Ok(enabled)
}

#[cfg(test)]
mod gemma3n_qmv_eligibility_tests {
    use super::*;

    const Q4_GROUP64: Option<QuantConfig> = Some(QuantConfig {
        bits: 4,
        group_size: 64,
    });

    #[test]
    fn cuda_q4_requires_native_qmv_and_exact_group_contract() {
        assert!(gemma3n_native_qmv_eligibility("cuda", Q4_GROUP64, true).unwrap());
        assert!(
            gemma3n_native_qmv_eligibility("cuda", Q4_GROUP64, false)
                .unwrap_err()
                .contains("refusing the numerically different dot fallback")
        );
        assert!(
            gemma3n_native_qmv_eligibility(
                "cuda",
                Some(QuantConfig {
                    bits: 4,
                    group_size: 32,
                }),
                true,
            )
            .unwrap_err()
            .contains("group_size=64")
        );
    }

    #[test]
    fn only_cuda_q4_selects_native_qmv() {
        assert!(!gemma3n_native_qmv_eligibility("local-task", Q4_GROUP64, true).unwrap());
        assert!(
            !gemma3n_native_qmv_eligibility(
                "cuda",
                Some(QuantConfig {
                    bits: 8,
                    group_size: 64,
                }),
                true,
            )
            .unwrap()
        );
        assert!(!gemma3n_native_qmv_eligibility("cuda", None, true).unwrap());
    }
}

// The per-architecture checkpoint-weight order the loader reads lives in
// `weights::weight_specs` (pure logic, unit-tested without the IREE runtime, and
// kept in lock-step with the emitter's arg order). It covers the dense arch pack
// (issue #498): LayerNorm biases, the o_proj / MLP biases, the dense StarCoder2
// MLP, and the Phi3 fused `qkv_proj` / `gate_up_proj` (loaded once and row-sliced
// into the emitter's separate args); and the MoE expert bank (issue #500): the
// router and the stacked `switch_mlp` gate/up/down (plus the optional shared
// expert), appended after each MoE layer's attention weights / norms.

/// Locate the IREE distribution: a runtime `IREE_DIST` override first, else the
/// path baked at build time (the dist whose runtime is linked into this binary).
/// Only the dist (CPU/vulkan) build uses this; the cuda build links a source-built
/// runtime and takes its iree-compile from `MLXCEL_XLA_IREE_COMPILE` instead.
#[cfg(not(xla_iree_cuda))]
fn iree_dist() -> Result<PathBuf, String> {
    if let Ok(d) = std::env::var("IREE_DIST") {
        return Ok(PathBuf::from(d));
    }
    match option_env!("MLXCEL_XLA_IREE_DIST") {
        Some(d) => Ok(PathBuf::from(d)),
        None => Err("IREE_DIST is not set and no dist path was baked at build time".to_string()),
    }
}

/// The `iree-compile` binary used to lower the bundled graphs to vmfbs. In the
/// cuda build the source-built runtime ships no compiler, so a cuda-capable
/// iree-compile (version-matched to that runtime) is required via
/// `MLXCEL_XLA_IREE_COMPILE` (runtime env, else baked at build); in the dist
/// build it is the dist's own `bin/iree-compile`.
fn iree_compile_bin() -> Result<PathBuf, String> {
    if let Ok(ic) = std::env::var("MLXCEL_XLA_IREE_COMPILE") {
        return Ok(PathBuf::from(ic));
    }
    if let Some(ic) = option_env!("MLXCEL_XLA_IREE_COMPILE") {
        return Ok(PathBuf::from(ic));
    }
    #[cfg(xla_iree_cuda)]
    {
        Err("the cuda build needs a cuda-capable iree-compile: set \
             MLXCEL_XLA_IREE_COMPILE to one matching the source runtime version \
             (e.g. the pip iree-compile)"
            .to_string())
    }
    #[cfg(not(xla_iree_cuda))]
    {
        let ic = iree_dist()?.join("bin/iree-compile");
        if !ic.exists() {
            return Err(format!(
                "iree-compile not found at {} (set IREE_DIST to a valid dist)",
                ic.display()
            ));
        }
        Ok(ic)
    }
}

/// iree-compile target flags for a HAL device. `local-task`/`local-sync` -> the
/// CPU (llvm-cpu) target; `cuda` -> the CUDA target (only usable in a cuda
/// build, whose runtime registers the cuda driver and whose iree-compile has
/// cuda codegen); `metal` -> the Metal target (metal-spirv codegen; the Apple
/// Silicon dev path, where the macOS runtime registers the metal driver and the
/// pinned macOS universal2 iree-compile has metal-spirv codegen).
fn target_flags(device: &str) -> Result<&'static [&'static str], String> {
    if device == "cuda" {
        Ok(&["--iree-hal-target-device=cuda"])
    } else if device == "metal" {
        Ok(&["--iree-hal-target-device=metal"])
    } else if device.starts_with("local") {
        Ok(&[
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ])
    } else {
        Err(format!(
            "unsupported OpenXLA device {device:?}; use \"local-task\" (CPU), \
             \"metal\" (Apple Silicon GPU, macOS build), or \"cuda\" (cuda build)"
        ))
    }
}

/// Compile one bundled graph to a vmfb, cached by a hash of its text + flags so
/// repeated loads skip the ~3 s compile.
fn compile_one(
    iree_compile: &Path,
    mlir: &str,
    flags: &[&str],
    cache: &Path,
    tag: &str,
    context_capacity: usize,
) -> Result<PathBuf, String> {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    mlir.hash(&mut h);
    for f in flags {
        f.hash(&mut h);
    }
    // Include the compiler path so a cuda vmfb is never reused for a cpu build
    // (or across iree-compile versions) just because the graph text matches.
    iree_compile.to_string_lossy().hash(&mut h);
    context_capacity.hash(&mut h);
    let key = h.finish();
    let mlir_path = cache.join(format!("{tag}-c{context_capacity}-{key:016x}.mlir"));
    let vmfb_path = cache.join(format!("{tag}-c{context_capacity}-{key:016x}.vmfb"));
    if vmfb_path.exists() {
        return Ok(vmfb_path);
    }
    std::fs::write(&mlir_path, mlir).map_err(|e| format!("write {}: {e}", mlir_path.display()))?;
    let out = Command::new(iree_compile)
        .arg("--iree-input-type=stablehlo")
        .args(flags)
        .arg(&mlir_path)
        .arg("-o")
        .arg(&vmfb_path)
        .output()
        .map_err(|e| format!("run {}: {e}", iree_compile.display()))?;
    if !out.status.success() {
        return Err(format!(
            "iree-compile failed for {tag}: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(vmfb_path)
}

/// Compatible entry modules loaded atomically into one runtime session.
struct CompiledBundle {
    prefill_vmfb: PathBuf,
    prefill_embeddings_vmfb: PathBuf,
    prefill_embeddings_deepstack_vmfb: Option<PathBuf>,
    decode_vmfb: PathBuf,
    compatibility_fingerprint: u64,
}

/// Compile the token prefill, embeddings prefill, and decode entries as one
/// compatibility bundle. The fingerprint binds their architecture/config text,
/// weight argument schema, precision, context/hidden/KV dimensions, and ABI
/// schema version; a member compiled from a different contract therefore cannot
/// be substituted without changing the fingerprint.
#[allow(clippy::too_many_arguments)]
fn compile_bundle(
    device: &str,
    cfg: &RuntimeConfig,
    gemma3n_qmv: bool,
    prefill_mlir: &str,
    prefill_tag: &str,
    prefill_embeddings_mlir: &str,
    prefill_embeddings_tag: &str,
    prefill_embeddings_deepstack: Option<(&str, &str)>,
    decode_mlir: &str,
    decode_tag: &str,
) -> Result<CompiledBundle, String> {
    let iree_compile = iree_compile_bin()?;
    if !iree_compile.exists() {
        return Err(format!(
            "iree-compile not found at {}",
            iree_compile.display()
        ));
    }
    let mut flags = target_flags(device)?.to_vec();
    if gemma3n_qmv {
        flags.push("--iree-cuda-target=sm_80");
    }
    let cache = std::env::temp_dir().join("mlxcel-xla-vmfb");
    std::fs::create_dir_all(&cache).map_err(|e| format!("mkdir {}: {e}", cache.display()))?;
    let pre = compile_one(
        &iree_compile,
        prefill_mlir,
        &flags,
        &cache,
        prefill_tag,
        cfg.context_capacity(),
    )?;
    let pre_embeddings = compile_one(
        &iree_compile,
        prefill_embeddings_mlir,
        &flags,
        &cache,
        prefill_embeddings_tag,
        cfg.context_capacity(),
    )?;
    let pre_embeddings_deepstack = prefill_embeddings_deepstack
        .map(|(mlir, tag)| {
            compile_one(
                &iree_compile,
                mlir,
                &flags,
                &cache,
                tag,
                cfg.context_capacity(),
            )
        })
        .transpose()?;
    let dec = compile_one(
        &iree_compile,
        decode_mlir,
        &flags,
        &cache,
        decode_tag,
        cfg.context_capacity(),
    )?;
    let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
    "mlxcel-xla-runtime-bundle-v2-explicit-position-mode".hash(&mut fingerprint);
    cfg.artifact_identity().hash(&mut fingerprint);
    format!("{:?}", cfg.weight_specs()).hash(&mut fingerprint);
    format!("{:?}", effective_precision(device, cfg)?).hash(&mut fingerprint);
    if gemma3n_qmv {
        gemma3n_qmv_artifact_identity().hash(&mut fingerprint);
    }
    prefill_mlir.hash(&mut fingerprint);
    prefill_embeddings_mlir.hash(&mut fingerprint);
    prefill_embeddings_deepstack
        .map(|(mlir, _)| mlir)
        .hash(&mut fingerprint);
    decode_mlir.hash(&mut fingerprint);
    let compatibility_fingerprint = fingerprint.finish();
    Ok(CompiledBundle {
        prefill_vmfb: pre,
        prefill_embeddings_vmfb: pre_embeddings,
        prefill_embeddings_deepstack_vmfb: pre_embeddings_deepstack,
        decode_vmfb: dec,
        compatibility_fingerprint: compatibility_fingerprint.max(1),
    })
}

/// Emit and compile the argmax prefill bundle for a single-sequence engine.
fn compile_vmfbs(device: &str, cfg: &RuntimeConfig) -> Result<CompiledBundle, String> {
    let precision = effective_precision(device, cfg)?;
    let native_qmv = match cfg {
        RuntimeConfig::Dense(_) => false,
        RuntimeConfig::Gemma3n(config) => gemma3n_native_qmv(device, config)?,
    };
    let (prefill, prefill_embeddings, prefill_embeddings_deepstack, decode) = match cfg {
        RuntimeConfig::Dense(config) => {
            check_packed_supported(device, quant_in_graph() && config.supports_packed_quant())?;
            (
                emit_prefill_with(config, true, precision),
                emit_prefill_embeddings_with(config, true, precision),
                config
                    .deepstack
                    .as_ref()
                    .map(|_| emit_prefill_embeddings_deepstack_with(config, true, precision)),
                emit_decode_with(config, true, precision),
            )
        }
        RuntimeConfig::Gemma3n(config) => (
            emit_gemma3n_prefill_with_qmv(config, true, precision, native_qmv),
            emit_gemma3n_prefill_embeddings_ple_with_qmv(config, true, precision, native_qmv),
            None,
            emit_gemma3n_decode_with_qmv(config, true, precision, native_qmv),
        ),
    };
    compile_bundle(
        device,
        cfg,
        native_qmv,
        &prefill,
        "prefill",
        &prefill_embeddings,
        "prefill_embeddings",
        prefill_embeddings_deepstack
            .as_deref()
            .map(|mlir| (mlir, "prefill_embeddings_deepstack")),
        &decode,
        "decode",
    )
}

#[cfg(feature = "diagnostics")]
fn compile_gemma3n_diagnostics(
    device: &str,
    cfg: &RuntimeConfig,
    all_layers: bool,
) -> Result<(PathBuf, Gemma3nDiagnosticLayout), String> {
    let RuntimeConfig::Gemma3n(config) = cfg else {
        return Err("Gemma3n diagnostics require a Gemma3n runtime config".to_string());
    };
    let precision = effective_precision(device, cfg)?;
    let native_qmv = gemma3n_native_qmv(device, config)?;
    let (mlir, layout) = if all_layers {
        emit_gemma3n_all_layer_diagnostics_with_qmv(config, precision, native_qmv)
    } else {
        emit_gemma3n_prefill_diagnostics_with_qmv(config, precision, native_qmv)
    };
    layout.validate()?;
    let compiler = iree_compile_bin()?;
    if !compiler.exists() {
        return Err(format!("iree-compile not found at {}", compiler.display()));
    }
    let cache = std::env::temp_dir().join("mlxcel-xla-vmfb");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let mut flags = target_flags(device)?.to_vec();
    if native_qmv {
        flags.push("--iree-cuda-target=sm_80");
    }
    let vmfb = compile_one(
        &compiler,
        &mlir,
        &flags,
        &cache,
        if all_layers {
            "prefill_all_layer_diagnostics"
        } else {
            "prefill_diagnostics"
        },
        cfg.context_capacity(),
    )?;
    Ok((vmfb, layout))
}

/// Map each needed weight name to the safetensors file that holds it, in the same
/// order as `names`. A single-file checkpoint (`model.safetensors` present) maps
/// every name to that one file; a sharded checkpoint (no `model.safetensors`, only
/// `model-0000k-of-...safetensors` shards) reads `model.safetensors.index.json`'s
/// `weight_map`. The big untied models (e.g. Llama-3.1-8B) ship sharded, so this
/// is what lets them load. A model dir that has BOTH a `model.safetensors` and an
/// index uses the single file (the index then just points back at it).
fn resolve_weight_shards(model_dir: &Path, names: &[String]) -> Result<Vec<PathBuf>, String> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single; names.len()]);
    }
    let index = model_dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index).map_err(|e| {
        format!(
            "no model.safetensors and no readable model.safetensors.index.json in {}: {e}",
            model_dir.display()
        )
    })?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", index.display()))?;
    let map = v
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{}: missing object `weight_map`", index.display()))?;
    names
        .iter()
        .map(|name| {
            map.get(name)
                .and_then(serde_json::Value::as_str)
                .map(|f| model_dir.join(f))
                .ok_or_else(|| format!("{}: weight_map has no entry for `{name}`", index.display()))
        })
        .collect()
}

fn language_model_tensor_name(name: &str) -> String {
    format!("language_model.{name}")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseCheckpointNames {
    Canonical,
    LanguageModelPrefixed,
}

/// Resolve dense-language weights from either a standalone checkpoint or the
/// `language_model.*` namespace used by LLaVA wrappers.
///
/// A single-file checkpoint defers namespace detection until its safetensors
/// header is open. A sharded checkpoint must choose one complete namespace
/// from the index so a partially mixed wrapper is rejected.
fn resolve_dense_weight_sources(
    model_dir: &Path,
    names: &[String],
) -> Result<(Vec<PathBuf>, Option<DenseCheckpointNames>), String> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok((vec![single; names.len()], None));
    }
    if let Ok(paths) = resolve_weight_shards(model_dir, names) {
        return Ok((paths, Some(DenseCheckpointNames::Canonical)));
    }
    let wrapped: Vec<String> = names
        .iter()
        .map(|name| language_model_tensor_name(name))
        .collect();
    resolve_weight_shards(model_dir, &wrapped)
        .map(|paths| (paths, Some(DenseCheckpointNames::LanguageModelPrefixed)))
        .map_err(|error| {
            format!(
                "dense checkpoint has neither a complete canonical nor `language_model.*` \
                 weight namespace: {error}"
            )
        })
}

/// mlx-community Gemma3n quantized repacks retain the upstream index but rewrite
/// the language backbone's safetensors prefix. Keep the graph's canonical HF
/// argument order and only try this one architecture-specific alternate after
/// the canonical name is absent.
fn gemma3n_repackaged_tensor_name(name: &str) -> Option<String> {
    name.strip_prefix("model.language_model.")
        .map(|suffix| format!("language_model.model.{suffix}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gemma3nCheckpointNames {
    Canonical,
    Repackaged,
}

fn merge_gemma3n_name_scheme(
    current: Option<Gemma3nCheckpointNames>,
    observed: Gemma3nCheckpointNames,
) -> Result<Option<Gemma3nCheckpointNames>, String> {
    match current {
        Some(current) if current != observed => Err(
            "Gemma3n checkpoint mixes canonical `model.language_model.*` and repackaged \
             `language_model.model.*` tensor names"
                .to_string(),
        ),
        Some(current) => Ok(Some(current)),
        None => Ok(Some(observed)),
    }
}

fn gemma3n_candidate_shards(model_dir: &Path, names: &[String]) -> Result<Vec<PathBuf>, String> {
    let indexed = resolve_weight_shards(model_dir, names);
    if let Ok(paths) = &indexed
        && paths.iter().all(|path| path.is_file())
    {
        let mut unique = paths.clone();
        unique.sort();
        unique.dedup();
        return Ok(unique);
    }
    let mut actual: Vec<PathBuf> = std::fs::read_dir(model_dir)
        .map_err(|error| format!("read {}: {error}", model_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".safetensors"))
        })
        .collect();
    actual.sort();
    if actual.is_empty() {
        return Err(match indexed {
            Err(error) => error,
            Ok(_) => format!("no safetensors files found in {}", model_dir.display()),
        });
    }
    Ok(actual)
}

fn resolve_gemma3n_weight_sources(
    model_dir: &Path,
    names: &[String],
) -> Result<(Vec<PathBuf>, Gemma3nCheckpointNames), String> {
    let candidate_shards = gemma3n_candidate_shards(model_dir, names)?;
    let mut resolved: Vec<Option<PathBuf>> = vec![None; names.len()];
    let mut scheme = None;
    for shard in candidate_shards {
        let file = File::open(&shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        // Safety: the file is read-only for the lifetime of this header scan.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", shard.display()))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("parse {}: {e}", shard.display()))?;
        for (index, canonical) in names.iter().enumerate() {
            let alias = gemma3n_repackaged_tensor_name(canonical).ok_or_else(|| {
                format!("Gemma3n graph weight has an unsupported canonical name: {canonical}")
            })?;
            let has_canonical = st.tensor(canonical).is_ok();
            let has_alias = st.tensor(&alias).is_ok();
            let observed = match (has_canonical, has_alias) {
                (false, false) => continue,
                (true, false) => Gemma3nCheckpointNames::Canonical,
                (false, true) => Gemma3nCheckpointNames::Repackaged,
                (true, true) => {
                    return Err(format!(
                        "Gemma3n checkpoint {} contains both `{canonical}` and `{alias}`",
                        shard.display()
                    ));
                }
            };
            if let Some(previous) = &resolved[index] {
                return Err(format!(
                    "Gemma3n tensor `{canonical}` is duplicated in {} and {}",
                    previous.display(),
                    shard.display()
                ));
            }
            resolved[index] = Some(shard.clone());
            scheme = merge_gemma3n_name_scheme(scheme, observed)?;
        }
    }
    let missing: Vec<&str> = resolved
        .iter()
        .zip(names)
        .filter_map(|(path, name)| path.is_none().then_some(name.as_str()))
        .collect();
    if !missing.is_empty() {
        let preview = missing
            .iter()
            .take(8)
            .copied()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Gemma3n checkpoint is missing {} graph weight(s): {preview}",
            missing.len()
        ));
    }
    let scheme = scheme.ok_or_else(|| "Gemma3n checkpoint has no graph weights".to_string())?;
    Ok((
        resolved
            .into_iter()
            .map(|path| path.expect("missing paths rejected above"))
            .collect(),
        scheme,
    ))
}

fn checkpoint_tensor_name(
    scheme: Option<Gemma3nCheckpointNames>,
    canonical: &str,
) -> Result<String, String> {
    match scheme {
        Some(Gemma3nCheckpointNames::Repackaged) => gemma3n_repackaged_tensor_name(canonical)
            .ok_or_else(|| format!("unsupported Gemma3n tensor name: {canonical}")),
        Some(Gemma3nCheckpointNames::Canonical) | None => Ok(canonical.to_string()),
    }
}

#[cfg(test)]
mod checkpoint_name_tests {
    use super::*;

    fn tiny_gemma3n_runtime() -> RuntimeConfig {
        RuntimeConfig::Gemma3n(Box::new(
            Gemma3nConfig::from_json_str(
                &serde_json::json!({
                    "model_type": "gemma3n_text",
                    "hidden_size": 8, "intermediate_size": [12, 12],
                    "max_position_embeddings": 4096,
                    "num_hidden_layers": 2, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 4, "rms_norm_eps": 1e-6,
                    "vocab_size": 12, "vocab_size_per_layer_input": 10,
                    "hidden_size_per_layer_input": 2,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "activation_sparsity_pattern": [0.0, 0.0],
                    "sliding_window": 2, "rope_theta": 1000000.0,
                    "rope_local_base_freq": 10000.0, "final_logit_softcapping": 30.0,
                    "num_kv_shared_layers": 0, "altup_num_inputs": 2,
                    "altup_active_idx": 0, "altup_coef_clip": 120.0,
                    "altup_correct_scale": true, "laurel_rank": 2,
                    "tie_word_embeddings": true
                })
                .to_string(),
            )
            .unwrap()
            .with_context_capacity(4)
            .unwrap(),
        ))
    }

    #[test]
    fn gemma3n_runtime_precision_is_fixed_bf16_and_rejects_metal() {
        let cfg = tiny_gemma3n_runtime();
        assert_eq!(effective_precision("cuda", &cfg).unwrap(), Precision::Bf16);
        assert_eq!(
            effective_precision("local-task", &cfg).unwrap(),
            Precision::Bf16
        );
        assert!(
            effective_precision("metal", &cfg)
                .unwrap_err()
                .contains("BF16")
        );
    }

    #[test]
    fn runtime_ffi_preflight_rejects_oversized_config_and_weight_count() {
        let mut cfg = tiny_gemma3n_runtime();
        let RuntimeConfig::Gemma3n(config) = &mut cfg else {
            unreachable!()
        };
        config.hidden = i32::MAX as usize + 1;
        let error = runtime_ffi_dimensions(&cfg, 1).unwrap_err();
        assert!(error.contains("hidden_size"));
        assert!(error.contains("does not fit"));

        let cfg = tiny_gemma3n_runtime();
        let error = runtime_ffi_dimensions(&cfg, i32::MAX as usize + 1).unwrap_err();
        assert!(error.contains("n_weights"));
        assert!(error.contains("does not fit"));
    }

    #[test]
    fn gemma3n_repack_alias_covers_weight_and_quant_companions() {
        assert_eq!(
            gemma3n_repackaged_tensor_name("model.language_model.layers.0.self_attn.q_proj.weight")
                .as_deref(),
            Some("language_model.model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            gemma3n_repackaged_tensor_name("model.language_model.embed_tokens.scales").as_deref(),
            Some("language_model.model.embed_tokens.scales")
        );
        assert_eq!(
            gemma3n_repackaged_tensor_name("model.language_model.embed_tokens.biases").as_deref(),
            Some("language_model.model.embed_tokens.biases")
        );
    }

    #[test]
    fn gemma3n_name_scheme_rejects_mixed_checkpoints() {
        let scheme = merge_gemma3n_name_scheme(None, Gemma3nCheckpointNames::Canonical).unwrap();
        let error =
            merge_gemma3n_name_scheme(scheme, Gemma3nCheckpointNames::Repackaged).unwrap_err();
        assert!(error.contains("mixes canonical"));
    }

    #[test]
    #[ignore = "requires GEMMA3N_MODEL_DIR pointing at a real local checkpoint"]
    fn real_gemma3n_headers_resolve_every_graph_weight_once() {
        let model_dir =
            PathBuf::from(std::env::var_os("GEMMA3N_MODEL_DIR").expect("GEMMA3N_MODEL_DIR"));
        let cfg = RuntimeConfig::from_json(&model_dir, 8).unwrap();
        let names: Vec<String> = cfg
            .weight_specs()
            .iter()
            .map(|spec| spec.tensor_name().to_string())
            .collect();
        let (paths, _) = resolve_gemma3n_weight_sources(&model_dir, &names).unwrap();
        assert_eq!(paths.len(), names.len());
        assert!(paths.iter().all(|path| path.is_file()));
    }
}

/// Load the weights bf16 -> f32 in the emitter's arg order, returning the owned
/// f32 buffers (kept alive until the shim copies them) plus the flat (ptr, rank,
/// dims) arrays the C ABI takes. `cfg` fixes the layer count and weight order.
///
/// Single-file and sharded checkpoints both load: [`resolve_weight_shards`] maps
/// each weight to its file, and the weights are read shard by shard (each shard
/// mmap'd exactly once, its tensors copied out as owned f32) and placed back into
/// the emitter's arg order by index, so the buffers line up with the graph args
/// regardless of how the checkpoint was split.
#[allow(clippy::type_complexity)]
fn load_weights(
    model_dir: &Path,
    cfg: &RuntimeConfig,
    resident_f16: bool,
) -> Result<(Vec<WeightBuf>, Vec<c_int>, Vec<c_int>, Vec<i64>), String> {
    let specs = cfg.weight_specs();
    let names: Vec<String> = specs.iter().map(|s| s.tensor_name().to_string()).collect();
    let (shard_paths, mut dense_name_scheme, gemma3n_name_scheme) = match cfg {
        RuntimeConfig::Gemma3n(_) => {
            let (paths, scheme) = resolve_gemma3n_weight_sources(model_dir, &names)?;
            (paths, None, Some(scheme))
        }
        RuntimeConfig::Dense(_) => {
            let (paths, scheme) = resolve_dense_weight_sources(model_dir, &names)?;
            (paths, scheme, None)
        }
    };

    // Group weight indices by shard so each shard file is opened/mmap'd once; the
    // results are placed by index, preserving the emitter's arg order. A Phi3
    // fused checkpoint references one tensor (`qkv_proj` / `gate_up_proj`) from
    // several specs; each reads and row-slices the (memmapped) shard.
    let mut by_shard: std::collections::BTreeMap<&Path, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, p) in shard_paths.iter().enumerate() {
        by_shard.entry(p.as_path()).or_default().push(i);
    }

    let n = specs.len();
    let mut bufs: Vec<WeightBuf> = (0..n).map(|_| WeightBuf::F32(Vec::new())).collect();
    let mut dtypes: Vec<c_int> = vec![WDT_F32; n];
    let mut ranks: Vec<c_int> = vec![0; n];
    let mut dims: Vec<i64> = vec![0; n * 4];
    for (shard, idxs) in by_shard {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        // Safety: the file is read-only for the lifetime of the mmap below.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", shard.display()))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("parse {}: {e}", shard.display()))?;
        for &i in &idxs {
            let name = &names[i];
            let (checkpoint_name, t) = match cfg {
                RuntimeConfig::Gemma3n(_) => {
                    let checkpoint_name = checkpoint_tensor_name(gemma3n_name_scheme, name)?;
                    let tensor = st.tensor(&checkpoint_name).map_err(|error| {
                        format!(
                            "resolved weight {checkpoint_name} (canonical {name}) in {}: {error}",
                            shard.display()
                        )
                    })?;
                    (checkpoint_name, tensor)
                }
                RuntimeConfig::Dense(_) => {
                    let checkpoint_name = match dense_name_scheme {
                        Some(DenseCheckpointNames::Canonical) => name.clone(),
                        Some(DenseCheckpointNames::LanguageModelPrefixed) => {
                            language_model_tensor_name(name)
                        }
                        None => {
                            if st.tensor(name).is_ok() {
                                dense_name_scheme = Some(DenseCheckpointNames::Canonical);
                                name.clone()
                            } else {
                                let wrapped = language_model_tensor_name(name);
                                st.tensor(&wrapped).map_err(|error| {
                                    format!(
                                        "resolved neither canonical weight {name} nor wrapped \
                                         weight {wrapped} in {}: {error}",
                                        shard.display()
                                    )
                                })?;
                                dense_name_scheme =
                                    Some(DenseCheckpointNames::LanguageModelPrefixed);
                                wrapped
                            }
                        }
                    };
                    let tensor = st.tensor(&checkpoint_name).map_err(|error| {
                        format!(
                            "resolved weight {checkpoint_name} (canonical {name}) in {}: {error}",
                            shard.display()
                        )
                    })?;
                    (checkpoint_name, tensor)
                }
            };

            // issue #516 packed path: upload the raw quantized part (the packed U32
            // weight, or its f16 scales / biases) as-is, no dequant. Its native dtype
            // and 2-D shape go straight to the device; the graph dequants in-place.
            if let WeightSpec::QuantRaw { part, .. } = &specs[i] {
                let (code, expect) = match part {
                    QuantPart::Packed => (WDT_U32, Dtype::U32),
                    QuantPart::Scales | QuantPart::Biases => (WDT_F16, Dtype::F16),
                };
                if t.dtype() != expect {
                    return Err(format!(
                        "quant part {name} dtype {:?}, expected {expect:?}",
                        t.dtype()
                    ));
                }
                let shape = t.shape();
                if shape.len() != 2 {
                    return Err(format!(
                        "quant part {name} is rank {} (expected 2)",
                        shape.len()
                    ));
                }
                dtypes[i] = code;
                ranks[i] = 2;
                dims[i * 4] = checked_ffi_i64(shape[0], &format!("weight {name} dim 0"))?;
                dims[i * 4 + 1] = checked_ffi_i64(shape[1], &format!("weight {name} dim 1"))?;
                bufs[i] = WeightBuf::Raw(t.data().to_vec());
                continue;
            }

            // Widen the whole tensor to row-major `[out, in]` (or `[out]`) f32 (the
            // graph's dtype): an MLX affine `U32`-packed weight is dequantized (the
            // layernorms / biases are not quantized and take the widen path).
            let (data, shape): (Vec<f32>, Vec<usize>) = if t.dtype() == Dtype::U32 {
                let qc = cfg.quantization().ok_or_else(|| {
                    format!(
                        "weight {name} is U32 (quantized) but config.json has no `quantization`"
                    )
                })?;
                let prefix = checkpoint_name.strip_suffix(".weight").ok_or_else(|| {
                    format!("quantized weight {checkpoint_name} does not end in `.weight`")
                })?;
                let scales_name = format!("{prefix}.scales");
                let biases_name = format!("{prefix}.biases");
                let scales = st.tensor(&scales_name).map_err(|e| {
                    format!("{scales_name} (for {name}) in {}: {e}", shard.display())
                })?;
                let biases = st.tensor(&biases_name).map_err(|e| {
                    format!("{biases_name} (for {name}) in {}: {e}", shard.display())
                })?;
                // mlx-lm emits the affine scales/biases as either f16 or bf16
                // (Qwen3 / Gemma3-27B / Qwen3-MoE checkpoints use bf16); accept a
                // matching 16-bit pair and widen it to f32 in the dequantizer.
                let sb_bf16 = match (scales.dtype(), biases.dtype()) {
                    (Dtype::F16, Dtype::F16) => false,
                    (Dtype::BF16, Dtype::BF16) => true,
                    (s, b) => {
                        return Err(format!(
                            "{prefix} scales/biases dtype {s:?}/{b:?}, expected a matching F16 or BF16 pair"
                        ));
                    }
                };
                let shape = t.shape();
                match shape.len() {
                    // Ordinary `[out, in_packed]` weight (attention, dense MLP,
                    // shared expert, router when quantized).
                    2 => {
                        let (out, in_packed) = (shape[0], shape[1]);
                        let in_ = in_packed * (32 / qc.bits);
                        let d = if matches!(cfg, RuntimeConfig::Gemma3n(_)) {
                            if !sb_bf16 {
                                return Err(format!(
                                    "{prefix} uses F16 affine scales/biases, but Gemma3n requires BF16 affine metadata"
                                ));
                            }
                            dequantize_affine_bf16_fused(
                                t.data(),
                                scales.data(),
                                biases.data(),
                                out,
                                in_packed,
                                qc.bits,
                                qc.group_size,
                            )
                        } else {
                            dequantize_affine(
                                t.data(),
                                scales.data(),
                                biases.data(),
                                out,
                                in_packed,
                                qc.bits,
                                qc.group_size,
                                sb_bf16,
                            )
                        }
                        .map_err(|e| format!("dequantize {name}: {e}"))?;
                        (d, vec![out, in_])
                    }
                    // Stacked `[experts, out, in_packed]` MoE `switch_mlp` weight
                    // (issue #500); dequantize each expert slab and keep the leading
                    // expert axis so it feeds the emitter's `[E, out, in]` arg.
                    3 => {
                        let (experts, out, in_packed) = (shape[0], shape[1], shape[2]);
                        let in_ = in_packed * (32 / qc.bits);
                        let d = dequantize_affine_stacked(
                            t.data(),
                            scales.data(),
                            biases.data(),
                            experts,
                            out,
                            in_packed,
                            qc.bits,
                            qc.group_size,
                            sb_bf16,
                        )
                        .map_err(|e| format!("dequantize stacked {name}: {e}"))?;
                        (d, vec![experts, out, in_])
                    }
                    r => {
                        return Err(format!("quantized weight {name} rank {r} not in {{2, 3}}"));
                    }
                }
            } else {
                // bf16 and f16 are the common checkpoint dtypes; f32 is a
                // passthrough. The widening is exact for all three, matching HF's
                // f32 reference.
                let d = match t.dtype() {
                    Dtype::BF16 => bf16_to_f32(t.data()),
                    Dtype::F16 => f16_to_f32(t.data()),
                    Dtype::F32 => f32_le_to_f32(t.data()),
                    other => {
                        return Err(format!(
                            "weight {name} dtype {other:?}, expected BF16/F16/F32 or \
                             MLX-quantized U32"
                        ));
                    }
                };
                let shape = t.shape();
                if shape.len() > 4 {
                    return Err(format!("weight {name} rank {} > 4", shape.len()));
                }
                (d, shape.to_vec())
            };

            // Place the whole tensor, or (Phi3 fused) a row-slice, into arg slot i.
            match &specs[i] {
                WeightSpec::Whole(_) => {
                    ranks[i] = checked_ffi_int(shape.len(), &format!("weight {name} rank"))?;
                    for (k, &s) in shape.iter().enumerate() {
                        dims[i * 4 + k] = checked_ffi_i64(s, &format!("weight {name} dim {k}"))?;
                    }
                    bufs[i] = WeightBuf::F32(data);
                }
                // issue #572: a linear projection. On the f16 GPU path pack it to an
                // f16 resident buffer (WDT_F16), matching the emitter's f16 weight arg
                // and halving its per-step DRAM read; otherwise upload f32, identical
                // to the old `Whole`. `data` / `shape` are the widened row-major weight.
                WeightSpec::Proj(_) => {
                    ranks[i] = checked_ffi_int(shape.len(), &format!("weight {name} rank"))?;
                    for (k, &s) in shape.iter().enumerate() {
                        dims[i * 4 + k] = checked_ffi_i64(s, &format!("weight {name} dim {k}"))?;
                    }
                    if resident_f16 {
                        dtypes[i] = WDT_F16;
                        bufs[i] = WeightBuf::F16(pack_f16(&data));
                    } else {
                        bufs[i] = WeightBuf::F32(data);
                    }
                }
                WeightSpec::Rows { start, end, .. } => {
                    if shape.len() != 2 {
                        return Err(format!(
                            "row-slice weight {name} is rank {} (expected 2)",
                            shape.len()
                        ));
                    }
                    bufs[i] = WeightBuf::F32(
                        slice_rows(&data, shape[0], *start, *end)
                            .map_err(|e| format!("row-slice {name}: {e}"))?,
                    );
                    ranks[i] = 2;
                    dims[i * 4] =
                        checked_ffi_i64(*end - *start, &format!("weight {name} sliced rows"))?;
                    dims[i * 4 + 1] = checked_ffi_i64(shape[1], &format!("weight {name} dim 1"))?;
                }
                WeightSpec::QuantRaw { .. } => {
                    unreachable!("QuantRaw is handled by the early-continue above")
                }
            }
        }
    }
    Ok((bufs, dtypes, ranks, dims))
}

fn path_cstring(p: &Path) -> Result<CString, String> {
    CString::new(p.as_os_str().as_bytes())
        .map_err(|_| format!("path has an interior nul byte: {}", p.display()))
}

fn prepared_descriptors(
    prepared: &PreparedIreePrefill,
) -> Result<(XlaTensorDesc, XlaTensorDesc, XlaTensorDesc), String> {
    let embeddings = XlaTensorDesc::f32(
        &prepared.embeddings,
        &[prepared.context_capacity, prepared.hidden_size],
    )?;
    let positions = match prepared.positions.mode() {
        PreparedPositionMode::OneD => {
            XlaTensorDesc::i32(prepared.positions.values(), &[prepared.context_capacity])?
        }
        PreparedPositionMode::Mrope3D => {
            XlaTensorDesc::i32(prepared.positions.values(), &[3, prepared.context_capacity])?
        }
    };
    let attention_bias = XlaTensorDesc::f32(
        &prepared.attention_bias,
        &[prepared.context_capacity, prepared.context_capacity],
    )?;
    Ok((embeddings, positions, attention_bias))
}

fn dense_ple_descriptor(dense_ple: &Gemma3nDensePle) -> Result<XlaTensorDesc, String> {
    XlaTensorDesc::f32(dense_ple.as_slice(), &dense_ple.shape())
}

fn deepstack_descriptors(
    deepstack: &PreparedDeepStack,
) -> Result<(XlaTensorDesc, XlaTensorDesc, XlaTensorDesc), String> {
    Ok((
        XlaTensorDesc::i32(&deepstack.visual_positions, &[deepstack.max_visual_count])?,
        XlaTensorDesc::f32(
            &deepstack.layer_features,
            &[
                deepstack.max_layer_count,
                deepstack.max_visual_count,
                deepstack.hidden_size,
            ],
        )?,
        XlaTensorDesc::i32(&deepstack.layer_indices, &[deepstack.max_layer_count])?,
    ))
}

/// Load the weights and create the C execution context for a (prefill, decode)
/// vmfb pair on `device`. Shared by the single-sequence ([`IreeLlama`]) and ragged
/// ([`IreeRaggedLlama`]) engines, which differ only in which decode vmfb they pass.
fn create_ctx(
    model_dir: &Path,
    cfg: &RuntimeConfig,
    device: &str,
    bundle: &CompiledBundle,
) -> Result<*mut XlaCtx, String> {
    create_ctx_with_diagnostics(model_dir, cfg, device, bundle, None)
}

fn create_ctx_with_diagnostics(
    model_dir: &Path,
    cfg: &RuntimeConfig,
    device: &str,
    bundle: &CompiledBundle,
    prefill_diagnostics_vmfb: Option<&Path>,
) -> Result<*mut XlaCtx, String> {
    // issue #572: the f16 GPU path uploads the projection weights f16-resident to
    // match the emitter's f16 args. Uses the same resolve_precision(device) the graph
    // emit uses, so the uploaded buffer dtype always lines up with the emitted arg.
    let resident_f16 = resolve_precision(device) == Precision::F16 && cfg.supports_f16_resident();
    let (bufs, dtypes, ranks, dims) = load_weights(model_dir, cfg, resident_f16)?;
    let ffi = runtime_ffi_dimensions(cfg, bufs.len())?;
    let ptrs: Vec<*const c_void> = bufs
        .iter()
        .map(|b| b.as_u8_ptr() as *const c_void)
        .collect();
    let c_dev = CString::new(device).map_err(|_| "device has interior nul".to_string())?;
    let c_pre = path_cstring(&bundle.prefill_vmfb)?;
    let c_pre_embeddings = path_cstring(&bundle.prefill_embeddings_vmfb)?;
    let c_pre_embeddings_deepstack = bundle
        .prefill_embeddings_deepstack_vmfb
        .as_deref()
        .map(path_cstring)
        .transpose()?;
    let c_dec = path_cstring(&bundle.decode_vmfb)?;
    let c_diagnostics = prefill_diagnostics_vmfb.map(path_cstring).transpose()?;
    let deepstack_target_layers = cfg
        .deepstack()
        .map(|schema| {
            schema
                .target_layer_indices
                .iter()
                .map(|&layer| checked_ffi_int(layer, "deepstack target layer"))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    // Safety: pointers are valid for the duration of the call; the shim copies
    // the weight data into device buffers before returning.
    let ctx = unsafe {
        xla_llama_create(
            c_dev.as_ptr(),
            c_pre.as_ptr(),
            c_pre_embeddings.as_ptr(),
            c_pre_embeddings_deepstack
                .as_ref()
                .map_or(std::ptr::null(), |path| path.as_ptr()),
            c_dec.as_ptr(),
            c_diagnostics
                .as_ref()
                .map_or(std::ptr::null(), |path| path.as_ptr()),
            bundle.compatibility_fingerprint,
            ffi.n_weights,
            ptrs.as_ptr(),
            dtypes.as_ptr(),
            ranks.as_ptr(),
            dims.as_ptr(),
            ffi.context_capacity,
            ffi.hidden,
            cfg.position_mode().ffi_code(),
            i32::from(cfg.dense_ple_shape().is_some()),
            ffi.dense_ple_layers,
            ffi.dense_ple_hidden,
            ffi.model_layers,
            ffi.deepstack_layers,
            ffi.deepstack_visual,
            if deepstack_target_layers.is_empty() {
                std::ptr::null()
            } else {
                deepstack_target_layers.as_ptr()
            },
        )
    };
    // Weights are resident on the device now; free the host copy.
    drop(ptrs);
    drop(bufs);
    if ctx.is_null() {
        return Err(
            "xla_llama_create failed (IREE runtime; see stderr for the status)".to_string(),
        );
    }
    Ok(ctx)
}

/// Owns the IREE execution context: one session with the prefill + decode
/// modules, the resident weights, and the threaded KV cache. Not `Send`/`Sync`
/// (the raw context is single-threaded), matching the single-sequence session.
pub struct IreeLlama {
    ctx: *mut XlaCtx,
    context_capacity: usize,
    hidden_size: usize,
    dense_ple_shape: Option<[usize; 3]>,
    position_mode: PreparedPositionMode,
    decode_position: DecodePositionState,
    deepstack_schema: Option<DeepStackConfig>,
}

impl IreeLlama {
    /// Prepare execution for a model directory on a HAL `device`
    /// (`"local-task"` for CPU). Compiles the bundled graphs, uploads the
    /// weights resident, and readies the prefill / decode calls.
    pub fn load(model_dir: &Path, device: &str, context_capacity: usize) -> Result<Self, String> {
        let cfg = RuntimeConfig::from_json(model_dir, context_capacity)?;
        runtime_ffi_dimensions(&cfg, cfg.weight_specs().len())?;
        let bundle = compile_vmfbs(device, &cfg)?;
        let ctx = create_ctx(model_dir, &cfg, device, &bundle)?;
        Ok(Self {
            ctx,
            context_capacity: cfg.context_capacity(),
            hidden_size: cfg.hidden(),
            dense_ple_shape: cfg.dense_ple_shape(),
            position_mode: cfg.position_mode(),
            decode_position: DecodePositionState::default(),
            deepstack_schema: cfg.deepstack().cloned(),
        })
    }

    /// Static sequence capacity compiled into the paired graphs and KV buffers.
    #[must_use]
    pub fn context_capacity(&self) -> usize {
        self.context_capacity
    }

    /// Seed the KV cache with `token_ids` (length <= the configured capacity) via the
    /// bucketed prefill graph. The returned token (the graph's argmax at
    /// `real_len-1`) is unused by the seed-then-decode drive loop.
    pub fn prefill_seed(&mut self, token_ids: &[i32]) -> Result<(), String> {
        self.prefill_padded(token_ids).map(|_| ())
    }

    /// Run the bucketed prefill over the FULL prompt and return its first token
    /// (the argmax at `prompt.len() - 1`). Unlike [`prefill_seed`], which seeds
    /// the KV and discards the token for the seed-then-decode loop, this returns
    /// the token, matching the batched engine's slot-seed convention
    /// ([`IreeRaggedLlama::prefill_slot`]); so a single-seq stream captured with
    /// `prefill_first` + [`decode`](Self::decode) is the right reference for it.
    ///
    /// [`prefill_seed`]: Self::prefill_seed
    pub fn prefill_first(&mut self, prompt: &[i32]) -> Result<i32, String> {
        if prompt.is_empty() {
            return Err("prefill_first requires a non-empty prompt".to_string());
        }
        self.prefill_padded(prompt)
    }

    /// Validate an owned embeddings payload, seed the resident KV through the
    /// dedicated embeddings module, and return the first sampled token.
    pub fn prefill_prepared_first(&mut self, prepared: &PreparedPrefill) -> Result<i32, String> {
        if self.dense_ple_shape.is_some() {
            return Err(
                "Gemma3n requires the distinct embeddings-plus-dense-PLE prefill entry".to_string(),
            );
        }
        let prepared = PreparedIreePrefill::prepare_for_mode(
            prepared,
            self.hidden_size,
            self.context_capacity,
            self.position_mode,
        )
        .map_err(|error| error.to_string())?;
        let rope_delta = prepared.positions.rope_delta();
        let (embeddings, positions, attention_bias) = prepared_descriptors(&prepared)?;
        let real_len = i32::try_from(prepared.effective_len)
            .map_err(|_| "prepared effective length does not fit i32".to_string())?;
        let mut out = 0i32;
        // Safety: every descriptor references a validated owned vector that
        // outlives the call; the C shim repeats dtype/rank/shape/byte checks.
        let rc = unsafe {
            xla_llama_prefill_embeddings(
                self.ctx,
                prepared.positions.mode().ffi_code(),
                &embeddings,
                &positions,
                &attention_bias,
                real_len,
                &mut out,
            )
        };
        let result = if rc == 0 {
            Ok(out)
        } else {
            Err(format!("xla_llama_prefill_embeddings failed (status {rc})"))
        };
        self.decode_position.complete_prefill(rope_delta, result)
    }

    /// Seed KV through the distinct sparse DeepStack embeddings entry.
    pub fn prefill_deepstack_prepared_first(
        &mut self,
        request: &DeepStackPreparedPrefill,
    ) -> Result<i32, String> {
        let schema = self.deepstack_schema.as_ref().ok_or_else(|| {
            "DeepStack prefill was requested from a runtime bundle without that capability"
                .to_string()
        })?;
        let prepared = PreparedIreePrefill::prepare_for_mode(
            request.prepared(),
            self.hidden_size,
            self.context_capacity,
            self.position_mode,
        )
        .map_err(|error| error.to_string())?;
        let rope_delta = prepared.positions.rope_delta();
        let deepstack = PreparedDeepStack::prepare(request.deepstack(), schema, self.hidden_size)
            .map_err(|error| error.to_string())?;
        let (embeddings, positions, attention_bias) = prepared_descriptors(&prepared)?;
        let (visual_positions, layer_features, layer_indices) = deepstack_descriptors(&deepstack)?;
        let actual_layer_count =
            checked_ffi_int(deepstack.actual_layer_count, "DeepStack actual layer count")?;
        let actual_visual_count = checked_ffi_int(
            deepstack.actual_visual_count,
            "DeepStack actual visual count",
        )?;
        let real_len = checked_ffi_int(prepared.effective_len, "prepared effective length")?;
        let mut out = 0i32;
        // Safety: all descriptors reference validated owned buffers that outlive
        // the call. The C shim repeats descriptor, value, and bound checks.
        let rc = unsafe {
            xla_llama_prefill_embeddings_deepstack(
                self.ctx,
                prepared.positions.mode().ffi_code(),
                &embeddings,
                &positions,
                &attention_bias,
                &visual_positions,
                &layer_features,
                &layer_indices,
                actual_layer_count,
                actual_visual_count,
                real_len,
                &mut out,
            )
        };
        let result = if rc == 0 {
            Ok(out)
        } else {
            Err(format!(
                "xla_llama_prefill_embeddings_deepstack failed (status {rc})"
            ))
        };
        self.decode_position.complete_prefill(rope_delta, result)
    }

    /// Seed Gemma3n KV from post-scale merged embeddings plus a canonical dense
    /// projected PLE tensor.
    pub fn prefill_gemma3n_prepared_first(
        &mut self,
        request: &Gemma3nPreparedPrefill,
    ) -> Result<i32, String> {
        let expected = self.dense_ple_shape.ok_or_else(|| {
            "dense Gemma3n PLE prefill was requested from a non-Gemma3n bundle".to_string()
        })?;
        let prepared = PreparedIreePrefill::prepare(
            request.prepared(),
            self.hidden_size,
            self.context_capacity,
        )
        .map_err(|error| error.to_string())?;
        if request.dense_ple().shape() != expected {
            return Err(format!(
                "Gemma3n dense PLE shape {:?} does not match runtime bundle {expected:?}",
                request.dense_ple().shape()
            ));
        }
        let (embeddings, positions, attention_bias) = prepared_descriptors(&prepared)?;
        let dense_ple = dense_ple_descriptor(request.dense_ple())?;
        let real_len = i32::try_from(prepared.effective_len)
            .map_err(|_| "prepared effective length does not fit i32".to_string())?;
        let mut out = 0i32;
        // Safety: all typed descriptors reference validated owned vectors that
        // outlive the call; the C shim repeats exact dtype/rank/shape/byte checks.
        let rc = unsafe {
            xla_llama_prefill_embeddings_ple(
                self.ctx,
                prepared.positions.mode().ffi_code(),
                &embeddings,
                &dense_ple,
                &positions,
                &attention_bias,
                real_len,
                &mut out,
            )
        };
        let result = if rc == 0 {
            Ok(out)
        } else {
            Err(format!(
                "xla_llama_prefill_embeddings_ple failed (status {rc})"
            ))
        };
        self.decode_position.complete_prefill(0, result)
    }

    /// Pad `prompt` into the configured static bucket, run the prefill, and return its
    /// first token. Accepts an empty prompt (the seed-then-decode loop prefills a
    /// zero-length prefix when the prompt is a single token).
    fn prefill_padded(&mut self, prompt: &[i32]) -> Result<i32, String> {
        if prompt.len() > self.context_capacity {
            return Err(format!(
                "prompt prefix of {} exceeds context_capacity={}",
                prompt.len(),
                self.context_capacity
            ));
        }
        let mut tokens = vec![0i32; self.context_capacity];
        tokens[..prompt.len()].copy_from_slice(prompt);
        let capacity = checked_ffi_int(self.context_capacity, "context_capacity")?;
        let real_len = checked_ffi_int(prompt.len(), "prompt length")?;
        let positions = canonical_text_positions(self.position_mode, self.context_capacity)
            .map_err(|error| format!("cannot build text-only prefill positions: {error}"))?;
        let mut out = 0i32;
        // Safety: buffers outlive the call; the shim stores the returned KV.
        let rc = unsafe {
            xla_llama_prefill(
                self.ctx,
                tokens.as_ptr(),
                capacity,
                positions.as_ptr(),
                real_len,
                &mut out,
            )
        };
        let result = if rc == 0 {
            Ok(out)
        } else {
            Err(format!("xla_llama_prefill failed (status {rc})"))
        };
        self.decode_position.complete_prefill(0, result)
    }

    /// Advance one token at `cache_len` (== position), returning the next token
    /// id (on-device argmax) and writing the new K/V into the resident cache.
    pub fn decode(&mut self, token: i32, cache_len: i32) -> Result<i32, String> {
        if cache_len < 0 || cache_len as usize >= self.context_capacity {
            return Err(format!(
                "decode position {cache_len} is outside context_capacity={}",
                self.context_capacity
            ));
        }
        let mut out = 0i32;
        let rc = match self.position_mode {
            PreparedPositionMode::OneD => {
                // Safety: the shim threads its own resident KV; only scalars cross here.
                unsafe { xla_llama_decode(self.ctx, token, cache_len, cache_len, &mut out) }
            }
            PreparedPositionMode::Mrope3D => {
                let coordinate = self
                    .decode_position
                    .mrope_coordinate(cache_len)
                    .map_err(|error| error.to_string())?;
                let positions = [coordinate; 3];
                // Safety: positions is exactly the explicit rank-1 `[3]` ABI payload.
                unsafe {
                    xla_llama_decode_mrope(self.ctx, token, positions.as_ptr(), cache_len, &mut out)
                }
            }
        };
        if rc != 0 {
            return Err(format!("xla_llama_decode failed (status {rc})"));
        }
        Ok(out)
    }
}

impl Drop for IreeLlama {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // Safety: ctx was produced by xla_llama_create and not freed yet.
            unsafe { xla_llama_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

/// Owns a ragged (continuous-batching) IREE context (#449 M3 Stage 2b/2d): the
/// logits prefill module plus a fixed-`B_max` ragged decode module, the resident
/// weights, and the rank-5 per-slot KV. Slots are seeded with [`prefill_slot_logits`]
/// (a device-side KV write) and advanced together by [`decode_ragged_logits`], each
/// row at its own position; both return logits so the wrapping engine samples on the
/// host. Not `Send`/`Sync` (the raw context is single-threaded); the engine that
/// wraps it owns it on one thread.
///
/// [`prefill_slot_logits`]: Self::prefill_slot_logits
/// [`decode_ragged_logits`]: Self::decode_ragged_logits
pub struct IreeRaggedLlama {
    ctx: *mut XlaCtx,
    b_max: usize,
    /// Vocabulary size (logits per row), from the model config; the readback
    /// buffers and the per-row logits slice are sized by it.
    vocab: usize,
    context_capacity: usize,
    hidden_size: usize,
    dense_ple_shape: Option<[usize; 3]>,
    position_mode: PreparedPositionMode,
    deepstack_schema: Option<DeepStackConfig>,
    #[cfg(feature = "diagnostics")]
    diagnostic_layout: Option<Gemma3nDiagnosticLayout>,
}

impl IreeRaggedLlama {
    /// Prepare a ragged engine for `model_dir` on `device` with `b_max` slots.
    ///
    /// Verifies the architecture, compiles the bundled prefill + the ragged decode
    /// graph for `b_max`, uploads the weights resident, and sizes the batch.
    /// `b_max` must be one of the bundled graphs ([`RAGGED_B_VALUES`]).
    pub fn load(
        model_dir: &Path,
        device: &str,
        b_max: usize,
        context_capacity: usize,
    ) -> Result<Self, String> {
        Self::load_inner(model_dir, device, b_max, context_capacity, false, false)
    }

    #[cfg(feature = "diagnostics")]
    pub fn load_with_diagnostics(
        model_dir: &Path,
        device: &str,
        b_max: usize,
        context_capacity: usize,
    ) -> Result<Self, String> {
        Self::load_inner(model_dir, device, b_max, context_capacity, true, false)
    }

    #[cfg(feature = "diagnostics")]
    pub fn load_with_all_layer_diagnostics(
        model_dir: &Path,
        device: &str,
        b_max: usize,
        context_capacity: usize,
    ) -> Result<Self, String> {
        Self::load_inner(model_dir, device, b_max, context_capacity, false, true)
    }

    fn load_inner(
        model_dir: &Path,
        device: &str,
        b_max: usize,
        context_capacity: usize,
        diagnostics: bool,
        all_layer_diagnostics: bool,
    ) -> Result<Self, String> {
        debug_assert!(!(diagnostics && all_layer_diagnostics));
        let cfg = RuntimeConfig::from_json(model_dir, context_capacity)?;
        runtime_ffi_dimensions(&cfg, cfg.weight_specs().len())?;
        checked_ffi_int(b_max, "b_max")?;
        if !RAGGED_B_VALUES.contains(&b_max) {
            return Err(format!(
                "the OpenXLA serve worker selects B_max from {RAGGED_B_VALUES:?}; \
                 {b_max} is not one of them"
            ));
        }
        // Emit the logits prefill + the ragged decode graph for this model + b_max
        // at the device-resolved precision (f16 on GPU by default, f32 on CPU).
        let precision = effective_precision(device, &cfg)?;
        let native_qmv = match &cfg {
            RuntimeConfig::Dense(_) => false,
            RuntimeConfig::Gemma3n(config) => gemma3n_native_qmv(device, config)?,
        };
        let (prefill_mlir, prefill_embeddings_mlir, prefill_embeddings_deepstack_mlir, decode_mlir) =
            match &cfg {
                RuntimeConfig::Dense(config) => {
                    check_packed_supported(
                        device,
                        quant_in_graph() && config.supports_packed_quant(),
                    )?;
                    (
                        emit_prefill_with(config, false, precision),
                        emit_prefill_embeddings_with(config, false, precision),
                        config.deepstack.as_ref().map(|_| {
                            emit_prefill_embeddings_deepstack_with(config, false, precision)
                        }),
                        emit_decode_ragged_with(config, b_max, false, precision),
                    )
                }
                RuntimeConfig::Gemma3n(config) => (
                    emit_gemma3n_prefill_with_qmv(config, false, precision, native_qmv),
                    emit_gemma3n_prefill_embeddings_ple_with_qmv(
                        config, false, precision, native_qmv,
                    ),
                    None,
                    emit_gemma3n_decode_ragged_with_qmv(
                        config, b_max, false, precision, native_qmv,
                    ),
                ),
            };
        let bundle = compile_bundle(
            device,
            &cfg,
            native_qmv,
            &prefill_mlir,
            "prefill_logits",
            &prefill_embeddings_mlir,
            "prefill_embeddings_logits",
            prefill_embeddings_deepstack_mlir
                .as_deref()
                .map(|mlir| (mlir, "prefill_embeddings_deepstack_logits")),
            &decode_mlir,
            &format!("decode_ragged_logits_b{b_max}"),
        )?;
        #[cfg(feature = "diagnostics")]
        let (diagnostic_vmfb, diagnostic_layout) = if diagnostics || all_layer_diagnostics {
            let (vmfb, layout) = compile_gemma3n_diagnostics(device, &cfg, all_layer_diagnostics)?;
            (Some(vmfb), Some(layout))
        } else {
            (None, None)
        };
        #[cfg(not(feature = "diagnostics"))]
        debug_assert!(!diagnostics && !all_layer_diagnostics);
        #[cfg(feature = "diagnostics")]
        let ctx = create_ctx_with_diagnostics(
            model_dir,
            &cfg,
            device,
            &bundle,
            diagnostic_vmfb.as_deref(),
        )?;
        #[cfg(not(feature = "diagnostics"))]
        let ctx = create_ctx(model_dir, &cfg, device, &bundle)?;
        // Safety: ctx is a fresh valid context from create_ctx; free it on error.
        let b_max_ffi = checked_ffi_int(b_max, "b_max")?;
        let rc = unsafe { xla_llama_ragged_reset(ctx, b_max_ffi) };
        if rc != 0 {
            unsafe { xla_llama_free(ctx) };
            return Err(format!("xla_llama_ragged_reset failed (status {rc})"));
        }
        Ok(Self {
            ctx,
            b_max,
            vocab: cfg.vocab(),
            context_capacity: cfg.context_capacity(),
            hidden_size: cfg.hidden(),
            dense_ple_shape: cfg.dense_ple_shape(),
            position_mode: cfg.position_mode(),
            deepstack_schema: cfg.deepstack().cloned(),
            #[cfg(feature = "diagnostics")]
            diagnostic_layout,
        })
    }

    /// The fixed slot count this engine was compiled for.
    #[must_use]
    pub fn b_max(&self) -> usize {
        self.b_max
    }

    /// The vocabulary size (logits per row). The engine slices the ragged decode's
    /// flat `[b_max * vocab]` logits by this.
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Static sequence capacity compiled into the paired graphs and KV buffers.
    #[must_use]
    pub fn context_capacity(&self) -> usize {
        self.context_capacity
    }

    /// Model hidden width compiled into the embeddings entry.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub(crate) const fn position_mode(&self) -> PreparedPositionMode {
        self.position_mode
    }

    pub(crate) fn prepare_deepstack(
        &self,
        features: &DeepStackFeatures,
    ) -> Result<PreparedDeepStack, String> {
        let schema = self.deepstack_schema.as_ref().ok_or_else(|| {
            "DeepStack prefill was requested from a runtime bundle without that capability"
                .to_string()
        })?;
        PreparedDeepStack::prepare(features, schema, self.hidden_size)
            .map_err(|error| error.to_string())
    }

    pub fn validate_gemma3n_dense_ple(&self, dense_ple: &Gemma3nDensePle) -> Result<(), String> {
        let expected = self.dense_ple_shape.ok_or_else(|| {
            "dense Gemma3n PLE was submitted to a non-Gemma3n runtime bundle".to_string()
        })?;
        if dense_ple.shape() != expected {
            return Err(format!(
                "Gemma3n dense PLE shape {:?} does not match runtime bundle {expected:?}",
                dense_ple.shape()
            ));
        }
        Ok(())
    }

    #[cfg(feature = "diagnostics")]
    pub fn diagnostic_layout(&self) -> Result<&Gemma3nDiagnosticLayout, String> {
        self.diagnostic_layout
            .as_ref()
            .ok_or_else(|| "this runtime bundle has no diagnostic module".to_string())
    }

    #[cfg(feature = "diagnostics")]
    pub fn prefill_diagnostics_slot(
        &mut self,
        slot: usize,
        prompt: &[i32],
    ) -> Result<Vec<f32>, String> {
        validate_slot(slot, self.b_max).map_err(|error| error.to_string())?;
        if prompt.is_empty() || prompt.len() > self.context_capacity {
            return Err(format!(
                "diagnostic prompt length {} is outside 1..={}",
                prompt.len(),
                self.context_capacity
            ));
        }
        let layout = self.diagnostic_layout()?.clone();
        layout.validate()?;
        let mut tokens = vec![0i32; self.context_capacity];
        tokens[..prompt.len()].copy_from_slice(prompt);
        let slot = checked_ffi_int(slot, "slot")?;
        let capacity = checked_ffi_int(self.context_capacity, "context_capacity")?;
        let real_len = checked_ffi_int(prompt.len(), "prompt length")?;
        let positions = canonical_text_positions(self.position_mode, self.context_capacity)
            .map_err(|error| format!("cannot build text-only diagnostic positions: {error}"))?;
        let diagnostic_len = c_int::try_from(layout.total_len)
            .map_err(|_| "diagnostic output length does not fit c_int".to_string())?;
        let mut output = vec![0.0f32; layout.total_len];
        // Safety: every input/output slice has the statically validated length;
        // the C shim validates the optional module and exact output element count.
        let rc = unsafe {
            xla_llama_prefill_diagnostics_slot(
                self.ctx,
                slot,
                tokens.as_ptr(),
                capacity,
                positions.as_ptr(),
                real_len,
                diagnostic_len,
                output.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_prefill_diagnostics_slot failed (status {rc})"
            ));
        }
        Ok(output)
    }

    /// Seed slot `slot` with `prompt` (up to the configured capacity) and return its
    /// first-token `[vocab]` LOGITS (#449 M3 Stage 2d). The prompt's KV is written
    /// device-side into the slot's region of the rank-5 cache; the other slots are
    /// untouched, so a mid-stream admit does not disturb live sequences. The caller
    /// samples the first token from the returned logits.
    pub fn prefill_slot_logits(&mut self, slot: usize, prompt: &[i32]) -> Result<Vec<f32>, String> {
        validate_slot(slot, self.b_max).map_err(|error| error.to_string())?;
        if prompt.is_empty() {
            return Err("prefill_slot_logits requires a non-empty prompt".to_string());
        }
        if prompt.len() > self.context_capacity {
            return Err(format!(
                "prompt of {} exceeds context_capacity={}",
                prompt.len(),
                self.context_capacity
            ));
        }
        let mut tokens = vec![0i32; self.context_capacity];
        tokens[..prompt.len()].copy_from_slice(prompt);
        let slot = checked_ffi_int(slot, "slot")?;
        let capacity = checked_ffi_int(self.context_capacity, "context_capacity")?;
        let real_len = checked_ffi_int(prompt.len(), "prompt length")?;
        let vocab = checked_ffi_int(self.vocab, "vocab_size")?;
        let positions = canonical_text_positions(self.position_mode, self.context_capacity)
            .map_err(|error| format!("cannot build text-only slot positions: {error}"))?;
        let mut logits = vec![0f32; self.vocab];
        // Safety: input buffers outlive the call; `logits` has self.vocab elements,
        // which the shim fills; the shim also writes the slot's KV device-side.
        let rc = unsafe {
            xla_llama_prefill_slot_logits(
                self.ctx,
                slot,
                tokens.as_ptr(),
                capacity,
                positions.as_ptr(),
                real_len,
                vocab,
                logits.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_prefill_slot_logits failed (status {rc})"
            ));
        }
        Ok(logits)
    }

    /// Seed one batch slot from a validated owned embeddings payload. The C
    /// shim copies the returned rank-4 KV directly into this slot's rank-5
    /// device cache, sharing the token-prefill slot population path.
    pub fn prefill_prepared_slot_logits(
        &mut self,
        slot: usize,
        prepared: &PreparedIreePrefill,
    ) -> Result<Vec<f32>, String> {
        if self.dense_ple_shape.is_some() {
            return Err(
                "Gemma3n requires the distinct embeddings-plus-dense-PLE prefill entry".to_string(),
            );
        }
        validate_slot(slot, self.b_max).map_err(|error| error.to_string())?;
        if prepared.hidden_size != self.hidden_size
            || prepared.context_capacity != self.context_capacity
            || prepared.effective_len == 0
            || prepared.effective_len > self.context_capacity
            || prepared.positions.mode() != self.position_mode
        {
            return Err(
                "prepared IREE payload is incompatible with this runtime bundle".to_string(),
            );
        }
        let (embeddings, positions, attention_bias) = prepared_descriptors(prepared)?;
        let real_len = i32::try_from(prepared.effective_len)
            .map_err(|_| "prepared effective length does not fit i32".to_string())?;
        let slot = checked_ffi_int(slot, "slot")?;
        let vocab = checked_ffi_int(self.vocab, "vocab_size")?;
        let mut logits = vec![0.0; self.vocab];
        // Safety: descriptors and output buffers outlive the call and carry
        // exact lengths; the shim validates slot/vocab and every descriptor.
        let rc = unsafe {
            xla_llama_prefill_embeddings_slot_logits(
                self.ctx,
                slot,
                prepared.positions.mode().ffi_code(),
                &embeddings,
                &positions,
                &attention_bias,
                real_len,
                vocab,
                logits.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_prefill_embeddings_slot_logits failed (status {rc})"
            ));
        }
        Ok(logits)
    }

    /// Seed one batch slot through the sparse DeepStack prefill entry.
    pub fn prefill_deepstack_prepared_slot_logits(
        &mut self,
        slot: usize,
        prepared: &PreparedIreePrefill,
        deepstack: &PreparedDeepStack,
    ) -> Result<Vec<f32>, String> {
        let schema = self.deepstack_schema.as_ref().ok_or_else(|| {
            "DeepStack prefill was requested from a runtime bundle without that capability"
                .to_string()
        })?;
        validate_slot(slot, self.b_max).map_err(|error| error.to_string())?;
        if prepared.hidden_size != self.hidden_size
            || prepared.context_capacity != self.context_capacity
            || prepared.effective_len == 0
            || prepared.effective_len > self.context_capacity
            || prepared.positions.mode() != self.position_mode
            || deepstack.hidden_size != self.hidden_size
            || deepstack.max_layer_count != schema.target_layer_indices.len()
            || deepstack.max_visual_count != schema.max_visual_positions
        {
            return Err(
                "DeepStack prepared payload is incompatible with this runtime bundle".to_string(),
            );
        }
        let (embeddings, positions, attention_bias) = prepared_descriptors(prepared)?;
        let (visual_positions, layer_features, layer_indices) = deepstack_descriptors(deepstack)?;
        let slot = checked_ffi_int(slot, "slot")?;
        let actual_layer_count =
            checked_ffi_int(deepstack.actual_layer_count, "DeepStack actual layer count")?;
        let actual_visual_count = checked_ffi_int(
            deepstack.actual_visual_count,
            "DeepStack actual visual count",
        )?;
        let real_len = checked_ffi_int(prepared.effective_len, "prepared effective length")?;
        let vocab = checked_ffi_int(self.vocab, "vocab_size")?;
        let mut logits = vec![0.0; self.vocab];
        // Safety: descriptors and output remain live for the call; the C shim
        // repeats all descriptor and compact-index checks before device upload.
        let rc = unsafe {
            xla_llama_prefill_embeddings_deepstack_slot_logits(
                self.ctx,
                slot,
                prepared.positions.mode().ffi_code(),
                &embeddings,
                &positions,
                &attention_bias,
                &visual_positions,
                &layer_features,
                &layer_indices,
                actual_layer_count,
                actual_visual_count,
                real_len,
                vocab,
                logits.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_prefill_embeddings_deepstack_slot_logits failed (status {rc})"
            ));
        }
        Ok(logits)
    }

    /// Seed one Gemma3n batch slot from owned prepared embeddings and dense PLE.
    pub fn prefill_gemma3n_prepared_slot_logits(
        &mut self,
        slot: usize,
        prepared: &PreparedIreePrefill,
        dense_ple: &Gemma3nDensePle,
    ) -> Result<Vec<f32>, String> {
        validate_slot(slot, self.b_max).map_err(|error| error.to_string())?;
        let expected = self.dense_ple_shape.ok_or_else(|| {
            "dense Gemma3n PLE prefill was requested from a non-Gemma3n bundle".to_string()
        })?;
        if dense_ple.shape() != expected
            || prepared.hidden_size != self.hidden_size
            || prepared.context_capacity != self.context_capacity
            || prepared.effective_len == 0
            || prepared.effective_len > self.context_capacity
        {
            return Err(
                "Gemma3n prepared payload is incompatible with this runtime bundle".to_string(),
            );
        }
        let (embeddings, positions, attention_bias) = prepared_descriptors(prepared)?;
        let dense_ple = dense_ple_descriptor(dense_ple)?;
        let real_len = i32::try_from(prepared.effective_len)
            .map_err(|_| "prepared effective length does not fit i32".to_string())?;
        let slot = checked_ffi_int(slot, "slot")?;
        let vocab = checked_ffi_int(self.vocab, "vocab_size")?;
        let mut logits = vec![0.0; self.vocab];
        // Safety: the typed descriptors and output remain live for the call; the
        // shim validates the Gemma3n-specific descriptor before device upload.
        let rc = unsafe {
            xla_llama_prefill_embeddings_ple_slot_logits(
                self.ctx,
                slot,
                prepared.positions.mode().ffi_code(),
                &embeddings,
                &dense_ple,
                &positions,
                &attention_bias,
                real_len,
                vocab,
                logits.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_prefill_embeddings_ple_slot_logits failed (status {rc})"
            ));
        }
        Ok(logits)
    }

    /// Advance all `b_max` slots one token, returning the flat `[b_max * vocab]`
    /// LOGITS (#449 M3 Stage 2d). `tokens` / `pos` / `cache_len` are per-row (length
    /// `b_max`); an inactive slot carries zeros (a masked no-op whose logits the
    /// caller discards). The caller samples a token per row from `logits[s*vocab..]`.
    pub fn decode_ragged_logits(
        &mut self,
        tokens: &[i32],
        pos: &[i32],
        cache_len: &[i32],
    ) -> Result<Vec<f32>, String> {
        if self.position_mode != PreparedPositionMode::OneD {
            return Err("1D decode was requested from an M-RoPE runtime bundle".to_string());
        }
        if tokens.len() != self.b_max || pos.len() != self.b_max || cache_len.len() != self.b_max {
            return Err(format!(
                "decode_ragged_logits expects per-row arrays of length b_max = {}",
                self.b_max
            ));
        }
        for row in 0..self.b_max {
            if pos[row] != cache_len[row]
                || pos[row] < 0
                || pos[row] as usize >= self.context_capacity
            {
                return Err(format!(
                    "decode row {row} has pos={} and cache_len={} outside context_capacity={} or not equal",
                    pos[row], cache_len[row], self.context_capacity
                ));
            }
        }
        let mut logits = vec![0f32; self.b_max * self.vocab];
        let b_max = checked_ffi_int(self.b_max, "b_max")?;
        let vocab = checked_ffi_int(self.vocab, "vocab_size")?;
        // Safety: the three input slices are length b_max == bsz; `logits` has
        // b_max*self.vocab elements, which the shim fills; it threads its rank-5 KV.
        let rc = unsafe {
            xla_llama_decode_ragged_logits(
                self.ctx,
                b_max,
                tokens.as_ptr(),
                pos.as_ptr(),
                cache_len.as_ptr(),
                vocab,
                logits.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_decode_ragged_logits failed (status {rc})"
            ));
        }
        Ok(logits)
    }

    /// Advance an M-RoPE batch with explicit temporal/height/width coordinates
    /// per row while retaining physical KV writes at `cache_len`.
    pub fn decode_ragged_mrope_logits(
        &mut self,
        tokens: &[i32],
        positions: &[[i32; 3]],
        cache_len: &[i32],
    ) -> Result<Vec<f32>, String> {
        if self.position_mode != PreparedPositionMode::Mrope3D {
            return Err("M-RoPE decode was requested from a 1D runtime bundle".to_string());
        }
        if tokens.len() != self.b_max
            || positions.len() != self.b_max
            || cache_len.len() != self.b_max
        {
            return Err(format!(
                "decode_ragged_mrope_logits expects per-row arrays of length b_max = {}",
                self.b_max
            ));
        }
        for (row, (&physical, coordinates)) in cache_len.iter().zip(positions.iter()).enumerate() {
            if physical < 0 || physical as usize >= self.context_capacity {
                return Err(format!(
                    "decode row {row} has cache_len={physical} outside context_capacity={}",
                    self.context_capacity
                ));
            }
            if coordinates.iter().any(|&coordinate| coordinate < 0) {
                return Err(format!(
                    "decode row {row} has negative M-RoPE coordinates {coordinates:?}"
                ));
            }
        }
        let flat_positions: Vec<i32> = positions
            .iter()
            .flat_map(|coordinates| coordinates.iter().copied())
            .collect();
        let mut logits = vec![0f32; self.b_max * self.vocab];
        let b_max = checked_ffi_int(self.b_max, "b_max")?;
        let vocab = checked_ffi_int(self.vocab, "vocab_size")?;
        // Safety: all per-row inputs have b_max rows, each position row has
        // exactly three coordinates, and logits has b_max*vocab elements.
        let rc = unsafe {
            xla_llama_decode_ragged_mrope_logits(
                self.ctx,
                b_max,
                tokens.as_ptr(),
                flat_positions.as_ptr(),
                cache_len.as_ptr(),
                vocab,
                logits.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "xla_llama_decode_ragged_mrope_logits failed (status {rc})"
            ));
        }
        Ok(logits)
    }
}

impl Drop for IreeRaggedLlama {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // Safety: ctx was produced by create_ctx and not freed yet.
            unsafe { xla_llama_free(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}
