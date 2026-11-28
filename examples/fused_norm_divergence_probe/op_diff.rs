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

//! Same-input diff of the second residual join (issue #3501): one arm's
//! dumped layer inputs (`x`, `attn_out`) fed to both `graph_add_rms_norm` and
//! the fused launcher in the same process, so the only variable is the op.
//!
//! Each output is also scored against a float64 reference of the graph's
//! formula (`w * bf16(x * rsqrt(mean(x^2) + eps))`, rounded once per step),
//! which says whether either side is wrong or both sit within rounding of it.

use std::path::Path;

use anyhow::{Result, bail};
use mlxcel::models::Llama3Model;
use mlxcel_core::layers::graph_add_rms_norm;
use mlxcel_core::{UniquePtr, dtype};

use crate::tensors::{Deviation, DumpWriter, decode, find, read_dump};

/// Round an f32 to bf16 (nearest, ties to even) and back.
fn round_bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Elements of `normed` (bf16 values) that differ from the float64 reference
/// computed row by row from `residual` (the identical bf16 new_residual) and
/// `weight`, whose length is the row width.
fn reference_misses(residual: &[f32], weight: &[f32], eps: f32, normed: &[f32]) -> usize {
    let dim = weight.len();
    residual
        .chunks_exact(dim)
        .zip(normed.chunks_exact(dim))
        .map(|(row, got)| {
            let mean_sq = row
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                / dim as f64;
            let inv = 1.0 / (mean_sq + f64::from(eps)).sqrt();
            row.iter()
                .zip(weight)
                .zip(got)
                .filter(|((x, w), got)| {
                    let scaled = round_bf16((f64::from(**x) * inv) as f32);
                    round_bf16(**w * scaled).to_bits() != got.to_bits()
                })
                .count()
        })
        .sum()
}

/// With `out`, both sides' outputs are written to `out/graph` and `out/fused`
/// as dumps (`Lnn_h`, `Lnn_normed`), for offline analysis of the same input.
pub fn run(model: &Llama3Model, dump: &Path, out: Option<&Path>) -> Result<()> {
    if !mlxcel_core::fused_add_rms_norm_available() {
        bail!("the fused_add_rms_norm port is not available on this device");
    }
    let mut writers = match out {
        Some(dir) => Some((
            DumpWriter::create(&dir.join("graph"))?,
            DumpWriter::create(&dir.join("fused"))?,
        )),
        None => None,
    };
    // One read of the whole dump; each layer looks its two tensors up by name.
    let tensors = read_dump(dump)?;
    let mut first = None;
    for (i, layer) in model.layers.iter().enumerate() {
        let x = find(&tensors, &format!("L{i:02}_x"))?;
        let attn = find(&tensors, &format!("L{i:02}_attn_out"))?;
        if x.dtype != dtype::BFLOAT16 {
            bail!("this diff decodes bf16 activations, got dtype {}", x.dtype);
        }
        let (x_arr, attn_arr) = (x.to_array(), attn.to_array());
        let norm = &layer.post_attention_layernorm;
        let (g_normed, g_res) = graph_add_rms_norm(norm, &attn_arr, &x_arr);
        let mut f_normed = UniquePtr::null();
        let mut f_res = UniquePtr::null();
        mlxcel_core::fused_add_rms_norm(
            &attn_arr,
            &x_arr,
            &norm.weight,
            norm.eps,
            0.0,
            &mut f_normed,
            &mut f_res,
        )?;
        for a in [&g_normed, &g_res, &f_normed, &f_res] {
            mlxcel_core::eval(a);
        }
        if let Some((graph, fused)) = writers.as_mut() {
            graph.write(&format!("L{i:02}_h"), &g_res)?;
            graph.write(&format!("L{i:02}_normed"), &g_normed)?;
            fused.write(&format!("L{i:02}_h"), &f_res)?;
            fused.write(&format!("L{i:02}_normed"), &f_normed)?;
        }
        let bytes = |a: &UniquePtr<mlxcel_core::MlxArray>| mlxcel_core::array_to_raw_bytes(a);
        let res_dev = Deviation::between(dtype::BFLOAT16, &bytes(&f_res), &bytes(&g_res))?;
        let normed_dev = Deviation::between(dtype::BFLOAT16, &bytes(&f_normed), &bytes(&g_normed))?;
        let (residual, _) = decode(dtype::BFLOAT16, &bytes(&g_res))?;
        let weight_bf16 = mlxcel_core::astype(&norm.weight, dtype::BFLOAT16);
        let (weight, _) = decode(dtype::BFLOAT16, &bytes(&weight_bf16))?;
        let (graph_vals, _) = decode(dtype::BFLOAT16, &bytes(&g_normed))?;
        let (fused_vals, _) = decode(dtype::BFLOAT16, &bytes(&f_normed))?;
        let graph_misses = reference_misses(&residual, &weight, norm.eps, &graph_vals);
        let fused_misses = reference_misses(&residual, &weight, norm.eps, &fused_vals);
        if first.is_none() && (res_dev.differing > 0 || normed_dev.differing > 0) {
            first = Some(i);
        }
        println!(
            "[op-diff] layer {i:02} new_residual {} | normed {} | f64-reference misses graph={graph_misses} fused={fused_misses}",
            res_dev.summary(),
            normed_dev.summary()
        );
    }
    match first {
        Some(i) => println!("[op-diff] first layer whose join differs on the same input: {i}"),
        None => println!("[op-diff] the two ops are byte-identical on every layer's input"),
    }
    Ok(())
}
