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

//! Laguna weight sanitization.
//!
//! Brings every published layout onto the one the constructor reads:
//!
//! 1. `language_model.` prefixes are dropped, and `rotary_emb.inv_freq`,
//!    `self_attn.k_scale` / `v_scale` and `input_global_scale` tensors are
//!    removed.
//! 2. The router moves to `mlp.gate.proj.weight`, its correction bias to
//!    `mlp.gate.e_score_correction_bias`.
//! 3. compressed-tensors `nvfp4-pack-quantized` experts (per-expert
//!    `weight_packed` / `weight_scale` / `weight_global_scale`) are stacked
//!    into MLX native NVFP4 planes: the packed E2M1 bytes are reinterpreted
//!    as little-endian `uint32` words (MLX's own nibble order), the E4M3
//!    block scales are kept byte for byte, and `1 / weight_global_scale`
//!    becomes a per-expert `.global_scale` sidecar applied after the routed
//!    matmul. Dense triplets (the shared expert) become a `UnifiedLinear`
//!    NVFP4 triple with the same sidecar.
//! 4. Pre-stacked MLX checkpoints that fuse `gate_up_proj` are split back
//!    into `gate_proj` / `up_proj`; per-expert bf16 experts are stacked.
//! 5. A tied `lm_head.weight` is dropped.
//!
//! Running the sanitizer twice is a no-op.

use crate::models::laguna::ModelArgs;
use crate::models::sanitize::f32_to_f8_e4m3;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const PROJECTIONS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Sanitize `weights` in place. Returns the number of compressed-tensors
/// NVFP4 planes that were transcoded (experts and dense linears).
pub fn sanitize_weights(weights: &mut WeightMap, args: &ModelArgs) -> Result<usize, String> {
    strip_language_model_prefix(weights);
    drop_auxiliary_keys(weights);
    rename_router_keys(weights);

    let mut transcoded = 0;
    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        transcoded += transcode_expert_planes(weights, &prefix, args.num_experts)?;
    }
    transcoded += transcode_dense_planes(weights)?;
    if transcoded > 0 {
        eprintln!("Transcoded {transcoded} compressed-tensors NVFP4 planes to MLX native NVFP4");
    }

    for layer in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        split_prestacked_gate_up(weights, &prefix);
        for proj in PROJECTIONS {
            stack_bf16_experts(weights, &prefix, proj, args.num_experts)?;
        }
    }

    if args.tie_word_embeddings {
        weights.remove("lm_head.weight");
        weights.remove("lm_head.scales");
        weights.remove("lm_head.biases");
    }
    Ok(transcoded)
}

fn strip_language_model_prefix(weights: &mut WeightMap) {
    let renames: Vec<(String, String)> = weights
        .keys()
        .filter_map(|k| {
            if let Some(rest) = k.strip_prefix("model.language_model.") {
                Some((k.clone(), format!("model.{rest}")))
            } else {
                k.strip_prefix("language_model.")
                    .map(|rest| (k.clone(), rest.to_string()))
            }
        })
        .collect();
    for (old, new) in renames {
        if let Some(v) = weights.remove(&old) {
            weights.insert(new, v);
        }
    }
}

fn drop_auxiliary_keys(weights: &mut WeightMap) {
    let doomed: Vec<String> = weights
        .keys()
        .filter(|k| {
            k.contains("rotary_emb.inv_freq")
                || k.ends_with(".self_attn.k_scale")
                || k.ends_with(".self_attn.v_scale")
                || k.ends_with(".input_global_scale")
        })
        .cloned()
        .collect();
    for key in doomed {
        weights.remove(&key);
    }
}

fn rename_router_keys(weights: &mut WeightMap) {
    let renames: Vec<(String, String)> = weights
        .keys()
        .filter_map(|k| {
            if let Some(stem) = k.strip_suffix(".mlp.gate.weight") {
                Some((k.clone(), format!("{stem}.mlp.gate.proj.weight")))
            } else if let Some(stem) = k.strip_suffix(".mlp.gate.scales") {
                Some((k.clone(), format!("{stem}.mlp.gate.proj.scales")))
            } else if let Some(stem) = k.strip_suffix(".mlp.gate.biases") {
                Some((k.clone(), format!("{stem}.mlp.gate.proj.biases")))
            } else {
                k.strip_suffix(".mlp.experts.e_score_correction_bias")
                    .map(|stem| {
                        (
                            k.clone(),
                            format!("{stem}.mlp.gate.e_score_correction_bias"),
                        )
                    })
            }
        })
        .collect();
    for (old, new) in renames {
        if let Some(v) = weights.remove(&old) {
            weights.insert(new, v);
        }
    }
}

/// Stack `{prefix}.mlp.experts.{e}.{proj}.{weight_packed,weight_scale,
/// weight_global_scale}` into `{prefix}.mlp.switch_mlp.{proj}.{weight,scales,
/// global_scale}`. Returns the number of planes transcoded (0 or 3).
fn transcode_expert_planes(
    weights: &mut WeightMap,
    prefix: &str,
    num_experts: usize,
) -> Result<usize, String> {
    let mut count = 0;
    for proj in PROJECTIONS {
        let probe = format!("{prefix}.mlp.experts.0.{proj}.weight_packed");
        if !weights.contains_key(&probe) {
            continue;
        }
        if num_experts == 0 {
            return Err(format!(
                "{probe} is present but config.json declares num_experts = 0"
            ));
        }
        check_no_expert_beyond(weights, prefix, proj, num_experts, "weight_packed")?;
        let mut packed = Vec::with_capacity(num_experts);
        let mut scales = Vec::with_capacity(num_experts);
        let mut globals = Vec::with_capacity(num_experts);
        let mut layout: Option<PlaneLayout> = None;
        for e in 0..num_experts {
            let base = format!("{prefix}.mlp.experts.{e}.{proj}");
            let triplet = take_triplet(weights, &base)?;
            let found = PlaneLayout::of(&triplet);
            match &layout {
                None => layout = Some(found),
                Some(first) if *first != found => {
                    return Err(format!(
                        "{base}: every expert plane must share one shape and dtype; expert 0 is \
                         {first:?} and expert {e} is {found:?}. mlx::core::stack throws on a \
                         mismatch, and that throw crosses the cxx bridge as an uncatchable abort \
                         mid-load. A promoted mixed dtype would be worse: the packed view would \
                         reinterpret it at the wrong width and load silently."
                    ));
                }
                _ => {}
            }
            packed.push(triplet.0);
            scales.push(triplet.1);
            globals.push(triplet.2);
        }
        let weight = view_packed_as_u32(&mlxcel_core::stack_owned(&packed, 0));
        let scales = encode_block_scales(&mlxcel_core::stack_owned(&scales, 0))?;
        let global = reciprocal_global_scale(&mlxcel_core::stack_owned(&globals, 0), num_experts)?;
        let ptrs: Vec<*const MlxArray> = [&weight, &scales, &global]
            .into_iter()
            .map(|a| a.as_ref().unwrap() as *const MlxArray)
            .collect();
        unsafe { mlxcel_core::eval_all(&ptrs) };

        let out = format!("{prefix}.mlp.switch_mlp.{proj}");
        weights.insert(format!("{out}.weight"), weight);
        weights.insert(format!("{out}.scales"), scales);
        weights.insert(format!("{out}.global_scale"), global);
        weights.remove(&format!("{out}.biases"));
        count += 1;
    }
    Ok(count)
}

/// Transcode every remaining dense `{p}.weight_packed` triplet (the shared
/// expert, and anything else the export quantized) into a `UnifiedLinear`
/// NVFP4 triple plus a `[1]` `global_scale` sidecar.
fn transcode_dense_planes(weights: &mut WeightMap) -> Result<usize, String> {
    let prefixes: Vec<String> = weights
        .keys()
        .filter_map(|k| k.strip_suffix(".weight_packed").map(str::to_string))
        .collect();
    for base in &prefixes {
        let (packed, scale, global) = take_triplet(weights, base)?;
        let weight = view_packed_as_u32(&packed);
        let scales = encode_block_scales(&scale)?;
        let global = reciprocal_global_scale(&global, 1)?;
        let ptrs: Vec<*const MlxArray> = [&weight, &scales, &global]
            .into_iter()
            .map(|a| a.as_ref().unwrap() as *const MlxArray)
            .collect();
        unsafe { mlxcel_core::eval_all(&ptrs) };
        weights.insert(format!("{base}.weight"), weight);
        weights.insert(format!("{base}.scales"), scales);
        weights.insert(format!("{base}.global_scale"), global);
        weights.remove(&format!("{base}.biases"));
    }
    Ok(prefixes.len())
}

type Triplet = (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
);

/// Shape and dtype of one expert's packed weight and block scales.
#[derive(Debug, PartialEq, Eq)]
struct PlaneLayout {
    packed_shape: Vec<i32>,
    packed_dtype: i32,
    scale_shape: Vec<i32>,
    scale_dtype: i32,
}

impl PlaneLayout {
    fn of(triplet: &Triplet) -> Self {
        Self {
            packed_shape: mlxcel_core::array_shape(&triplet.0),
            packed_dtype: mlxcel_core::array_dtype(&triplet.0),
            scale_shape: mlxcel_core::array_shape(&triplet.1),
            scale_dtype: mlxcel_core::array_dtype(&triplet.1),
        }
    }
}

/// Refuse a checkpoint that carries an expert past the declared count.
///
/// Stacking only the first `num_experts` planes leaves the router able to emit
/// an index past the end of the stack, because the router width comes from
/// `mlp.gate.proj` and not from this key, and `gather_qmm` does not range-check
/// a positive index. On the packed path it would additionally send every
/// leftover plane through [`transcode_dense_planes`] as a dense linear nothing
/// ever reads, which for a 256-expert 40-layer export declaring 8 experts is
/// tens of thousands of pointless f32 materializations and E4M3 re-encodes.
fn check_no_expert_beyond(
    weights: &WeightMap,
    prefix: &str,
    proj: &str,
    num_experts: usize,
    suffix: &str,
) -> Result<(), String> {
    let overflow = format!("{prefix}.mlp.experts.{num_experts}.{proj}.{suffix}");
    if weights.contains_key(&overflow) {
        return Err(format!(
            "{overflow} is present but config.json declares num_experts = {num_experts}; the \
             router is sized by mlp.gate.proj, so stacking only the declared experts would let it \
             index past the stacked plane and gather_qmm reads a positive index without a range \
             check"
        ));
    }
    Ok(())
}

fn take_triplet(weights: &mut WeightMap, base: &str) -> Result<Triplet, String> {
    let packed_key = format!("{base}.weight_packed");
    let scale_key = format!("{base}.weight_scale");
    let global_key = format!("{base}.weight_global_scale");
    let packed = weights
        .remove(&packed_key)
        .ok_or_else(|| format!("Weight not found: {packed_key}"))?;
    let scale = weights
        .remove(&scale_key)
        .ok_or_else(|| format!("Weight not found: {scale_key}"))?;
    let global = weights
        .remove(&global_key)
        .ok_or_else(|| format!("Weight not found: {global_key}"))?;
    let packed_dtype = mlxcel_core::array_dtype(&packed);
    if packed_dtype != mlxcel_core::dtype::UINT8 && packed_dtype != mlxcel_core::dtype::INT8 {
        return Err(format!(
            "{packed_key}: expected packed E2M1 bytes (U8), got dtype {packed_dtype}"
        ));
    }
    let packed_shape = mlxcel_core::array_shape(&packed);
    let scale_shape = mlxcel_core::array_shape(&scale);
    if packed_shape.len() != 2 || scale_shape.len() != 2 {
        return Err(format!(
            "{packed_key}: expected 2-D packed weight and block scales, got {packed_shape:?} and \
             {scale_shape:?}"
        ));
    }
    let in_dim = packed_shape[1] * 2;
    if in_dim % 16 != 0 || scale_shape[0] != packed_shape[0] || scale_shape[1] != in_dim / 16 {
        return Err(format!(
            "{packed_key}: packed weight {packed_shape:?} does not pair with block scales \
             {scale_shape:?} at group size 16"
        ));
    }
    Ok((packed, scale, global))
}

/// Reinterpret `[.., out, in/2]` packed E2M1 bytes as `[.., out, in/8]`
/// little-endian `uint32` words: two codes per byte, low nibble first, which
/// is exactly MLX native NVFP4's eight-codes-per-word order.
fn view_packed_as_u32(packed: &MlxArray) -> UniquePtr<MlxArray> {
    let packed = if mlxcel_core::array_dtype(packed) == mlxcel_core::dtype::UINT8 {
        mlxcel_core::copy(packed)
    } else {
        mlxcel_core::view(packed, mlxcel_core::dtype::UINT8)
    };
    let contiguous = mlxcel_core::contiguous(&packed, false);
    mlxcel_core::view(&contiguous, mlxcel_core::dtype::UINT32)
}

/// Re-encode block scales as raw E4M3 bytes. `U8` input is passed through
/// verbatim; a float input (the loader promotes `F8_E4M3` to f16) is decoded
/// to f32 and encoded back, which is lossless for values that came from E4M3.
fn encode_block_scales(scales: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(scales);
    if mlxcel_core::array_dtype(scales) == mlxcel_core::dtype::UINT8 {
        return Ok(mlxcel_core::contiguous(scales, false));
    }
    let as_f32 = mlxcel_core::astype(scales, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&as_f32);
    let bytes = mlxcel_core::array_to_raw_bytes(&as_f32);
    let count: usize = shape.iter().map(|&d| d as usize).product();
    if bytes.len() < count * 4 {
        return Err(format!(
            "block scales {shape:?} yielded {} bytes, expected {}",
            bytes.len(),
            count * 4
        ));
    }
    let encoded = encode_f32_bytes_to_e4m3(&bytes[..count * 4]);
    Ok(mlxcel_core::from_bytes(
        &encoded,
        &shape,
        mlxcel_core::dtype::UINT8,
    ))
}

/// Encode little-endian f32 bytes to E4M3 bytes, in parallel over chunks.
fn encode_f32_bytes_to_e4m3(bytes: &[u8]) -> Vec<u8> {
    let count = bytes.len() / 4;
    let mut out = vec![0u8; count];
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16);
    let chunk = count.div_ceil(workers).max(1 << 16);
    std::thread::scope(|scope| {
        for (i, dst) in out.chunks_mut(chunk).enumerate() {
            let src = &bytes[i * chunk * 4..(i * chunk + dst.len()) * 4];
            scope.spawn(move || {
                for (d, w) in dst.iter_mut().zip(src.chunks_exact(4)) {
                    *d = f32_to_f8_e4m3(f32::from_le_bytes([w[0], w[1], w[2], w[3]]));
                }
            });
        }
    });
    out
}

/// `1 / weight_global_scale` as an f32 `[num_entries]` vector.
fn reciprocal_global_scale(
    global: &MlxArray,
    num_entries: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    let size: usize = mlxcel_core::array_shape(global)
        .iter()
        .map(|&d| d as usize)
        .product();
    if size != num_entries {
        return Err(format!(
            "weight_global_scale has {size} elements, expected {num_entries}"
        ));
    }
    let flat = mlxcel_core::reshape(global, &[num_entries as i32]);
    let flat = mlxcel_core::astype(&flat, mlxcel_core::dtype::FLOAT32);
    let ones = mlxcel_core::ones(&[num_entries as i32], mlxcel_core::dtype::FLOAT32);
    Ok(mlxcel_core::divide(&ones, &flat))
}

/// Split a pre-stacked `switch_mlp.gate_up_proj.{weight,scales,biases}`
/// (`[E, 2I, ..]`) into `gate_proj` (rows `0..I`) and `up_proj` (rows
/// `I..2I`).
fn split_prestacked_gate_up(weights: &mut WeightMap, prefix: &str) {
    for suffix in ["weight", "scales", "biases"] {
        let key = format!("{prefix}.mlp.switch_mlp.gate_up_proj.{suffix}");
        let Some(fused) = weights.remove(&key) else {
            continue;
        };
        let shape = mlxcel_core::array_shape(&fused);
        // An odd fused axis cannot be a gate/up pair; putting the plane back
        // untouched surfaces as a missing `gate_proj` at load rather than as
        // two silently misaligned halves.
        if shape.len() < 2 || shape[1] % 2 != 0 {
            weights.insert(key, fused);
            continue;
        }
        let half = shape[1] / 2;
        let mut gate_stop = shape.clone();
        gate_stop[1] = half;
        let mut up_start = vec![0; shape.len()];
        up_start[1] = half;
        let zeros = vec![0; shape.len()];
        let gate = mlxcel_core::slice(&fused, &zeros, &gate_stop);
        let up = mlxcel_core::slice(&fused, &up_start, &shape);
        weights.insert(
            format!("{prefix}.mlp.switch_mlp.gate_proj.{suffix}"),
            mlxcel_core::contiguous(&gate, false),
        );
        weights.insert(
            format!("{prefix}.mlp.switch_mlp.up_proj.{suffix}"),
            mlxcel_core::contiguous(&up, false),
        );
    }
}

/// Stack per-expert `{prefix}.mlp.experts.{e}.{proj}.{suffix}` tensors into
/// `{prefix}.mlp.switch_mlp.{proj}.{suffix}`.
fn stack_bf16_experts(
    weights: &mut WeightMap,
    prefix: &str,
    proj: &str,
    num_experts: usize,
) -> Result<(), String> {
    for suffix in ["weight", "scales", "biases"] {
        let first = format!("{prefix}.mlp.experts.0.{proj}.{suffix}");
        if !weights.contains_key(&first) {
            continue;
        }
        check_no_expert_beyond(weights, prefix, proj, num_experts, suffix)?;
        // Check the whole run before removing any of it: a half-stacked export
        // would otherwise leave the map short of the experts already taken,
        // and the failure would surface as a missing `switch_mlp` plane with
        // no trace of what was dropped.
        let keys: Vec<String> = (0..num_experts)
            .map(|e| format!("{prefix}.mlp.experts.{e}.{proj}.{suffix}"))
            .collect();
        if !keys.iter().all(|k| weights.contains_key(k)) {
            continue;
        }
        let mut parts = Vec::with_capacity(num_experts);
        let mut layout: Option<(Vec<i32>, i32)> = None;
        for key in &keys {
            let Some(w) = weights.remove(key) else {
                return Err(format!("Weight not found: {key}"));
            };
            let found = (mlxcel_core::array_shape(&w), mlxcel_core::array_dtype(&w));
            match &layout {
                None => layout = Some(found),
                Some(expected) if *expected != found => {
                    return Err(format!(
                        "{key}: every expert plane must share one shape and dtype; expert 0 is \
                         {expected:?} and this one is {found:?}. mlx::core::stack throws on a \
                         mismatch, and that throw crosses the cxx bridge as an uncatchable abort \
                         mid-load."
                    ));
                }
                _ => {}
            }
            parts.push(w);
        }
        let stacked = mlxcel_core::stack_owned(&parts, 0);
        weights.insert(format!("{prefix}.mlp.switch_mlp.{proj}.{suffix}"), stacked);
    }
    Ok(())
}
