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

//! Load-time shape contract of a Limite checkpoint (issue #2360). The MLX
//! ops behind `mlxcel_core` abort the whole process on a shape error and
//! GPU gathers do not bounds-check, so every tensor the forward reads is
//! checked against the config here, before any op touches it.

use super::{ModelArgs, weight};
use mlxcel_core::weights::WeightMap;

/// One expected extent: exactly `n`, or at least `n`.
#[derive(Clone, Copy)]
enum Dim {
    Eq(usize),
    Min(usize),
}
use Dim::{Eq, Min};

fn expect_shape(weights: &WeightMap, name: &str, want: &[Dim]) -> Result<(), String> {
    let got: Vec<usize> = mlxcel_core::array_shape(weight(weights, name)?)
        .iter()
        .map(|&d| d.max(0) as usize)
        .collect();
    let fits = got.len() == want.len()
        && got.iter().zip(want).all(|(&g, w)| match *w {
            Eq(n) => g == n,
            Min(n) => g >= n,
        });
    if fits {
        return Ok(());
    }
    let want: Vec<String> = want
        .iter()
        .map(|w| match *w {
            Eq(n) => n.to_string(),
            Min(n) => format!(">={n}"),
        })
        .collect();
    Err(format!(
        "Limite: {name} has shape {got:?}, expected [{}]",
        want.join(", ")
    ))
}

/// `n` elements in a tensor of rank 0 or 1 (a higher rank would broadcast
/// the activations it scales into a different shape).
fn expect_len(weights: &WeightMap, name: &str, n: usize) -> Result<(), String> {
    let w = weight(weights, name)?;
    let (size, rank) = (mlxcel_core::array_size(w), mlxcel_core::array_ndim(w));
    if size == n && rank <= 1 {
        return Ok(());
    }
    Err(format!(
        "Limite: {name} has {size} elements in rank {rank}, expected {n} in rank 0 or 1"
    ))
}

pub(super) fn check_shapes(weights: &WeightMap, args: &ModelArgs) -> Result<(), String> {
    let hidden = args.hidden_size;
    let (h, kv, d) = (
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
    );
    let embed = [Eq(args.vocab_size), Eq(hidden)];
    expect_shape(weights, "model.embed_tokens.weight", &embed)?;
    if !args.ve_layers.is_empty() {
        // Every valid token id gathers a row of this table, unchecked.
        let ve = [Eq(args.vocab_size), Eq(args.ve_stored_heads * args.ve_dim)];
        expect_shape(weights, "model.value_embeds.weight", &ve)?;
    }
    let inter = args.intermediate_size;
    for i in 0..args.num_hidden_layers {
        let p = format!("model.layers.{i}");
        for (name, rows, cols) in [
            ("self_attn.q_proj.weight", h * d, hidden),
            ("self_attn.k_proj.weight", kv * d, hidden),
            ("self_attn.v_proj.weight", kv * d, hidden),
            ("self_attn.o_proj.weight", hidden, h * d),
            ("mlp.gate_proj.weight", inter, hidden),
            ("mlp.up_proj.weight", inter, hidden),
            ("mlp.down_proj.weight", hidden, inter),
        ] {
            expect_shape(weights, &format!("{p}.{name}"), &[Eq(rows), Eq(cols)])?;
        }
        for name in [
            "self_attn.qkv_scale",
            "self_attn.o_scale",
            "resid_lambda_attn",
            "post_lambda_attn",
            "resid_lambda_mlp",
            "post_lambda_mlp",
        ] {
            expect_len(weights, &format!("{p}.{name}"), 1)?;
        }
        if args.attn_gate_channels > 0 {
            let gate = [Eq(h), Eq(args.attn_gate_channels)];
            expect_shape(weights, &format!("{p}.self_attn.attn_gate"), &gate)?;
        }
        if args.ve_layers.contains(&i) {
            let gate = [Min(args.ve_stored_heads.min(kv)), Eq(args.ve_gate_channels)];
            expect_shape(weights, &format!("{p}.self_attn.ve_gate"), &gate)?;
        }
        if args.xsa && args.xsa_layers.contains(&i) {
            expect_len(weights, &format!("{p}.self_attn.xsa_alpha"), h)?;
        }
    }
    if args.mudd && !args.mudd_layers.is_empty() {
        let taps = args.tap_indices()?;
        // `rows_as` slices `[layer, :taps]` out of each table.
        let layers = taps.keys().max().map_or(0, |&l| l + 1);
        let n = taps.values().map(Vec::len).max().unwrap_or(0);
        let mi = args.mudd_inter;
        expect_shape(weights, "model.mudd.dense1", &[Eq(mi), Eq(hidden)])?;
        let mut ways = vec![("model.mudd.dense2", "model.mudd.bias")];
        if args.mudd_mlp {
            ways.push(("model.mudd.dense2_mlp", "model.mudd.bias_mlp"));
        }
        for (dense2, bias) in ways {
            expect_shape(weights, dense2, &[Min(layers), Min(n), Eq(mi)])?;
            expect_shape(weights, bias, &[Min(layers), Min(n)])?;
        }
    }
    Ok(())
}
