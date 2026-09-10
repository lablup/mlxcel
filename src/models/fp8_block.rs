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

//! Vendor fine-grained FP8 checkpoints: block-scale detection and load-time
//! requantization to MLX's native `mxfp8`.
//!
//! The vLLM-style FP8 releases of the Qwen3.5 family (`Qwen/Qwen3.8-27B-FP8`
//! and siblings) store every converted projection as two tensors:
//!
//! * `<name>.weight` — raw `E4M3` bytes, one per weight element,
//! * `<name>.weight_scale_inv` — a `bfloat16` multiplier per 128x128 block.
//!
//! The true weight is `e4m3_decode(byte) * scale_inv[block]`. MLX has no
//! float8 dtype and its safetensors reader maps `F8_E4M3` to `uint8`, so the
//! bytes arrive intact but the scale is never applied: without this pass the
//! checkpoint loads and generates, and every projection is wrong by its own
//! per-block factor.
//!
//! [`requantize_block_fp8_weights`] reconstructs each pair on device and
//! requantizes it to `mxfp8` (`group_size` 32, `bits` 8), which is the closest
//! native layout MLX executes: 8 bits per weight plus one shared E8M0 exponent
//! per 32 values, with no per-token dequantization in the forward pass. The
//! reconstruction is transient and per tensor, so peak memory stays close to
//! the checkpoint's own footprint rather than its dequantized size.
//!
//! This module deliberately lives outside `sanitize.rs`: that file is already
//! 3k lines, and the FP8 path is a self-contained pre-pass with its own tests.
//!
//! Used by: the Qwen3.5 text loader (`Qwen35Model::load`), the Qwen3.5 VLM
//! loader (`load_qwen3_5_vlm_with_variant`), and the adapter-side special
//! loader (`try_load_special_model_from_weights`).

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::{Value, json};

use super::sanitize::f8_e4m3_to_f32;

/// Suffix identifying a vendor FP8 block-scale sidecar.
pub const FP8_BLOCK_SCALE_SUFFIX: &str = ".weight_scale_inv";

/// MLX fixes `mxfp8` at group size 32 and 8 bits (`fp_quantize` in
/// `mlx/ops.cpp` throws for anything else), so these are constants rather than
/// tunables.
pub const MXFP8_GROUP_SIZE: i32 = 32;
/// See [`MXFP8_GROUP_SIZE`].
pub const MXFP8_BITS: i32 = 8;
/// See [`MXFP8_GROUP_SIZE`].
pub const MXFP8_MODE: &str = "mxfp8";

/// The only block geometry this path accepts.
///
/// Every public Qwen3.5-family FP8 release declares `[128, 128]`. A different
/// geometry is refused rather than assumed, because the block extent is what
/// pairs a scale with its weight elements and a wrong pairing silently
/// mis-scales the whole tensor.
pub const SUPPORTED_FP8_BLOCK: usize = 128;

/// A detected fine-grained FP8 layout and the MLX quantization it maps onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8BlockQuantization {
    /// Block extent along the output (row) axis.
    pub block_rows: usize,
    /// Block extent along the input (column) axis.
    pub block_cols: usize,
    /// Group size of the MLX quantization the weights are converted to.
    pub group_size: i32,
    /// Bit width of the MLX quantization the weights are converted to.
    pub bits: i32,
    /// MLX quantization mode name.
    pub mode: &'static str,
}

impl Fp8BlockQuantization {
    /// The `quantization` block to merge into a model config so the model
    /// constructor builds quantized layers for the converted tensors.
    ///
    /// `mode` is included for readers that thread an explicit mode
    /// (`QuantizationArgs`); the Qwen3.5 config carries only `group_size` and
    /// `bits` and lets `infer_quantization_mode` derive `mxfp8` from
    /// `bits == 8` plus the absent `.biases` plane.
    #[must_use]
    pub fn effective_config(&self) -> Value {
        json!({
            "group_size": self.group_size,
            "bits": self.bits,
            "mode": self.mode,
        })
    }
}

/// Read the `quantization_config` block and report whether this checkpoint is
/// a fine-grained (block-scaled) FP8 release this loader can reconstruct.
///
/// Accepts the block at the top level or inside `text_config`, since the VLM
/// wrappers place it in either. Returns:
///
/// * `Ok(None)` when the checkpoint declares no FP8 quantization at all, which
///   covers every dense and affine-quantized checkpoint,
/// * `Ok(Some(..))` for `quant_method: "fp8"` with `fmt` absent or `"e4m3"`
///   and `weight_block_size: [128, 128]`,
/// * `Err` for an FP8 declaration whose geometry or float format is something
///   other than that. Failing the load is the point: the alternative is
///   applying a 128x128 pairing to scales that do not have it, which produces
///   fluent but wrong output rather than an error.
pub fn qwen_fp8_block_quantization(
    full_config: &Value,
) -> Result<Option<Fp8BlockQuantization>, String> {
    let Some(quant_config) = full_config.get("quantization_config").or_else(|| {
        full_config
            .get("text_config")
            .and_then(|text| text.get("quantization_config"))
    }) else {
        return Ok(None);
    };

    let Some(quant_config) = quant_config.as_object() else {
        return Ok(None);
    };

    let method = quant_config
        .get("quant_method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !method.eq_ignore_ascii_case("fp8") {
        return Ok(None);
    }

    match quant_config.get("fmt").and_then(Value::as_str) {
        None => {}
        Some(fmt) if fmt.eq_ignore_ascii_case("e4m3") => {}
        Some(fmt) => {
            return Err(format!(
                "quantization_config declares quant_method \"fp8\" with fmt \"{fmt}\": mlxcel \
                 decodes only the E4M3 float8 format, and decoding E4M3 bytes as another format \
                 (or the reverse) produces a wrong exponent for every weight without failing"
            ));
        }
    }

    let Some(block) = quant_config.get("weight_block_size") else {
        return Err(
            "quantization_config declares quant_method \"fp8\" but carries no weight_block_size: \
             mlxcel reconstructs only the fine-grained block-scaled layout (a bf16 scale per \
             128x128 block in a *.weight_scale_inv sidecar). Per-tensor and per-channel FP8 \
             scales use a different sidecar layout and are not supported"
                .to_string(),
        );
    };

    let dims = block
        .as_array()
        .ok_or_else(|| {
            format!("quantization_config.weight_block_size must be an array, got {block}")
        })?
        .iter()
        .map(|dim| {
            dim.as_u64().ok_or_else(|| {
                format!(
                    "quantization_config.weight_block_size entry {dim} is not a positive integer"
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let block_extent = u64::try_from(SUPPORTED_FP8_BLOCK).unwrap_or(u64::MAX);
    if dims.as_slice() != [block_extent, block_extent] {
        return Err(format!(
            "quantization_config.weight_block_size is {block}, but mlxcel reconstructs only \
             [{SUPPORTED_FP8_BLOCK}, {SUPPORTED_FP8_BLOCK}] fine-grained FP8 blocks. The block \
             extent is what pairs each scale with its weight elements, so applying the wrong one \
             mis-scales every tensor without failing"
        ));
    }

    Ok(Some(Fp8BlockQuantization {
        block_rows: SUPPORTED_FP8_BLOCK,
        block_cols: SUPPORTED_FP8_BLOCK,
        group_size: MXFP8_GROUP_SIZE,
        bits: MXFP8_BITS,
        mode: MXFP8_MODE,
    }))
}

/// Merge the effective `quantization` block into a model config when the
/// checkpoint is a fine-grained FP8 release.
///
/// Mirrors how `qwen35_text_config` merges a top-level `quantization` into
/// `text_config`: an explicit `quantization` already present always wins, so a
/// pre-converted checkpoint is untouched. Returns whether a block was
/// inserted.
pub fn merge_fp8_block_quantization(
    config: &mut Value,
    detected: Fp8BlockQuantization,
) -> Result<bool, String> {
    if config.get("quantization").is_some() {
        return Ok(false);
    }
    let object = config.as_object_mut().ok_or_else(|| {
        "Cannot merge the FP8 quantization block into a non-object config".to_string()
    })?;
    object.insert("quantization".to_string(), detected.effective_config());
    Ok(true)
}

/// Whether `weights` carries at least one FP8 block-scale sidecar.
///
/// Cheap enough to gate the pre-pass on: a detected `quantization_config` says
/// what the publisher intended, this says what the shards actually hold.
#[must_use]
pub fn has_block_fp8_weights(weights: &WeightMap) -> bool {
    weights
        .keys()
        .any(|key| key.ends_with(FP8_BLOCK_SCALE_SUFFIX))
}

/// A 256-entry f32 lookup table mapping an E4M3 byte to its value.
///
/// Built once per call and consumed by `take` so the decode runs on device:
/// the alternative, a host loop over every byte, would walk 28 GB through the
/// CPU and materialize an f32 copy four times that size.
fn e4m3_lookup_table() -> UniquePtr<MlxArray> {
    let table: Vec<f32> = (0..=u8::MAX).map(f8_e4m3_to_f32).collect();
    mlxcel_core::from_slice_f32(&table, &[256])
}

/// Reconstruct one block-scaled FP8 tensor into a plain f32 array.
///
/// `raw` holds the E4M3 bytes as `uint8`, `scale_inv` the per-block bf16
/// multipliers. The result is `[rows, cols]` f32, matching the original weight
/// shape. Exposed so tests can compare this route against an independently
/// expressed reference (see `fp8_block_tests.rs`).
pub fn reconstruct_block_fp8_tensor(
    name: &str,
    raw: &MlxArray,
    scale_inv: &MlxArray,
    block: usize,
    lut: &MlxArray,
) -> Result<UniquePtr<MlxArray>, String> {
    let weight_shape = mlxcel_core::array_shape(raw);
    if weight_shape.len() != 2 {
        return Err(format!(
            "FP8 block-scaled tensor {name} has rank {} ({weight_shape:?}); only rank-2 \
             projections carry a weight_scale_inv sidecar",
            weight_shape.len()
        ));
    }
    let weight_dtype = mlxcel_core::array_dtype(raw);
    if weight_dtype != dtype::UINT8 {
        return Err(format!(
            "FP8 block-scaled tensor {name} loaded as MLX dtype {weight_dtype} rather than uint8 \
             ({}): the raw E4M3 bytes are needed to pair each element with its block scale, and a \
             pre-decoded float tensor has already lost that pairing",
            dtype::UINT8
        ));
    }

    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    if rows <= 0 || cols <= 0 {
        return Err(format!(
            "FP8 block-scaled tensor {name} has a degenerate shape {weight_shape:?}"
        ));
    }
    if cols % MXFP8_GROUP_SIZE != 0 {
        return Err(format!(
            "FP8 block-scaled tensor {name} has {cols} input features, which is not a multiple of \
             the mxfp8 group size {MXFP8_GROUP_SIZE}. MLX packs one E8M0 scale per {MXFP8_GROUP_SIZE} \
             values along the last axis and cannot represent a partial group"
        ));
    }

    let block_i32 = i32::try_from(block)
        .map_err(|_| format!("FP8 block extent {block} exceeds i32 for tensor {name}"))?;
    // `i32::div_ceil` is still unstable on the pinned toolchain; both operands
    // are positive here, so the classic form is exact.
    let row_blocks = (rows + block_i32 - 1) / block_i32;
    let col_blocks = (cols + block_i32 - 1) / block_i32;

    let scale_shape = mlxcel_core::array_shape(scale_inv);
    if scale_shape != vec![row_blocks, col_blocks] {
        return Err(format!(
            "FP8 block-scale sidecar for {name} has shape {scale_shape:?}, expected \
             [{row_blocks}, {col_blocks}] for a {rows}x{cols} weight at block extent {block}"
        ));
    }

    // Decode the E4M3 bytes through the LUT on device.
    let indices = mlxcel_core::astype(raw, dtype::UINT32);
    let decoded = mlxcel_core::take(lut, &indices, 0);

    // Pad to whole blocks so the reshape is exact. The padding is zero and is
    // sliced back off after scaling, so it can only affect the trailing block
    // of a tensor whose extent is not a multiple of `block`.
    let padded_rows = row_blocks * block_i32;
    let padded_cols = col_blocks * block_i32;
    let padded = if padded_rows == rows && padded_cols == cols {
        decoded
    } else {
        mlxcel_core::pad(
            &decoded,
            &[0, padded_rows - rows, 0, padded_cols - cols],
            0.0,
        )
    };

    let blocked = mlxcel_core::reshape(&padded, &[row_blocks, block_i32, col_blocks, block_i32]);
    let scales = mlxcel_core::astype(scale_inv, dtype::FLOAT32);
    let scales = mlxcel_core::reshape(&scales, &[row_blocks, 1, col_blocks, 1]);
    let scaled = mlxcel_core::multiply(&blocked, &scales);
    let flat = mlxcel_core::reshape(&scaled, &[padded_rows, padded_cols]);

    Ok(if padded_rows == rows && padded_cols == cols {
        flat
    } else {
        mlxcel_core::slice(&flat, &[0, 0], &[rows, cols])
    })
}

/// Reconstruct every block-scaled FP8 tensor in `weights` and replace it with
/// an MLX-native `mxfp8` pair.
///
/// For each `<name>.weight` / `<name>.weight_scale_inv` pair the sidecar is
/// removed, `<name>.weight` is replaced by the packed `uint32` plane, and
/// `<name>.scales` is added. Tensors without a sidecar (norms, embeddings,
/// `conv1d`, `A_log`, `dt_bias`, and the whole vision tower on the released
/// checkpoints) are left untouched, and `UnifiedLinear` picks the quantized or
/// the dense path per tensor from the presence of `.scales`.
///
/// Conversion is one tensor at a time and each quantized pair is evaluated
/// before the next tensor starts, so the f32 reconstruction of the largest
/// projection is the only transient and is released before the next one is
/// built. Doing the whole map lazily would instead hold every reconstruction
/// alive until the first `eval`, which for a 27B checkpoint is four times the
/// checkpoint's own size.
pub fn requantize_block_fp8_weights(
    mut weights: WeightMap,
    block: usize,
) -> Result<WeightMap, String> {
    if block == 0 {
        return Err("FP8 block extent must be positive".to_string());
    }

    let mut scale_keys: Vec<String> = weights
        .keys()
        .filter(|key| key.ends_with(FP8_BLOCK_SCALE_SUFFIX))
        .cloned()
        .collect();
    if scale_keys.is_empty() {
        return Ok(weights);
    }
    // Deterministic order so a failure reports the same tensor on every run.
    scale_keys.sort();

    let started = std::time::Instant::now();
    let total = scale_keys.len();
    tracing::info!(
        tensors = total,
        block,
        "reconstructing fine-grained FP8 weights and requantizing to mxfp8"
    );

    let lut = e4m3_lookup_table();

    for scale_key in scale_keys {
        let prefix = scale_key
            .strip_suffix(FP8_BLOCK_SCALE_SUFFIX)
            .ok_or_else(|| format!("Internal error: {scale_key} lost its FP8 sidecar suffix"))?
            .to_string();
        let weight_key = format!("{prefix}.weight");

        let scale_inv = weights
            .remove(&scale_key)
            .ok_or_else(|| format!("Internal error: FP8 sidecar {scale_key} disappeared"))?;
        let raw = weights.remove(&weight_key).ok_or_else(|| {
            format!(
                "FP8 block-scale sidecar {scale_key} has no matching {weight_key} in the \
                 checkpoint; the scale cannot be applied to anything"
            )
        })?;

        let restored = reconstruct_block_fp8_tensor(&weight_key, &raw, &scale_inv, block, &lut)?;
        // Drop the E4M3 bytes and the sidecar before quantizing so the peak
        // holds the reconstruction and the packed result, not all four.
        drop(raw);
        drop(scale_inv);

        let quantized = mlxcel_core::quantize_weights_with_mode(
            &restored,
            MXFP8_GROUP_SIZE,
            MXFP8_BITS,
            MXFP8_MODE,
        );
        if mlxcel_core::quantized_weights_has_biases(&quantized) {
            return Err(format!(
                "MLX returned affine biases for a {MXFP8_MODE} quantization of {weight_key}; \
                 the block-float path must not carry a zero-point plane"
            ));
        }
        let packed = mlxcel_core::quantized_weights_w(&quantized);
        let scales = mlxcel_core::quantized_weights_scales(&quantized);

        // Force realization here rather than at first forward: this is what
        // keeps the reconstruction transient and the peak bounded.
        mlxcel_core::eval(&packed);
        mlxcel_core::eval(&scales);

        weights.insert(weight_key, packed);
        weights.insert(format!("{prefix}.scales"), scales);
    }

    tracing::info!(
        tensors = total,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "fine-grained FP8 requantization complete"
    );

    Ok(weights)
}

#[cfg(test)]
#[path = "fp8_block_tests.rs"]
mod fp8_block_tests;
