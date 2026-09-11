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

//! Tests for the GOT-OCR 2.0 runtime's vision path.
//!
//! The geometry claim these pin is the one the whole port rests on: a page
//! becomes exactly 256 feature rows of width 1024, so the prompt's `<imgpad>`
//! run can be a constant. Losing a stride or a conv would change the count and
//! the scatter would then fail loudly rather than silently, but only at load
//! time on a real checkpoint; here it fails in a second.

use super::*;
use crate::vision::encoders::deepseekocr_sam::SamConfig;
use mlxcel_core::weights::WeightMap;

/// Deterministic pseudo-random fill. A real normal distribution is not needed:
/// these tests assert shapes and finiteness, and a fixed sequence keeps a
/// failure reproducible.
fn filled(shape: &[i32], seed: u64) -> UniquePtr<MlxArray> {
    let count: usize = shape.iter().map(|&d| d as usize).product();
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map to roughly [-0.05, 0.05]; small enough that a 12-layer stack of
        // random weights stays finite.
        let unit = ((state >> 40) as f32 / 16_777_216.0) - 0.5;
        data.push(unit * 0.1);
    }
    mlxcel_core::from_slice_f32(&data, shape)
}

/// A weight map for [`SamEncoder::from_weights`] under `config`, using the
/// canonical key names [`crate::loading`]'s canonicalizer produces.
fn sam_weights(config: &SamConfig, prefix: &str) -> WeightMap {
    let mut w = WeightMap::new();
    let ed = config.embed_dim;
    let oc = config.out_chans;
    let mut seed = 1u64;
    let put = |w: &mut WeightMap, name: String, shape: Vec<i32>, seed: &mut u64| {
        *seed += 1;
        w.insert(name, filled(&shape, *seed));
    };

    // Channels-last conv weights, the layout the MLX conversion ships; the
    // encoder's shape gate leaves them alone.
    put(
        &mut w,
        format!("{prefix}.patch_embed.proj.weight"),
        vec![ed, 16, 16, 3],
        &mut seed,
    );
    put(
        &mut w,
        format!("{prefix}.patch_embed.proj.bias"),
        vec![ed],
        &mut seed,
    );
    put(
        &mut w,
        format!("{prefix}.pos_embed"),
        vec![1, config.grid, config.grid, ed],
        &mut seed,
    );

    let head_dim = ed / config.num_heads;
    for i in 0..config.depth {
        let bp = format!("{prefix}.blocks.{i}");
        let is_global = config.global_attn_indexes.contains(&i);
        let span = if is_global {
            2 * config.grid - 1
        } else {
            2 * config.window_size - 1
        };
        for norm in ["norm1", "norm2"] {
            put(&mut w, format!("{bp}.{norm}.weight"), vec![ed], &mut seed);
            put(&mut w, format!("{bp}.{norm}.bias"), vec![ed], &mut seed);
        }
        put(
            &mut w,
            format!("{bp}.attn.qkv.weight"),
            vec![3 * ed, ed],
            &mut seed,
        );
        put(
            &mut w,
            format!("{bp}.attn.qkv.bias"),
            vec![3 * ed],
            &mut seed,
        );
        put(
            &mut w,
            format!("{bp}.attn.proj.weight"),
            vec![ed, ed],
            &mut seed,
        );
        put(&mut w, format!("{bp}.attn.proj.bias"), vec![ed], &mut seed);
        for axis in ["rel_pos_h", "rel_pos_w"] {
            put(
                &mut w,
                format!("{bp}.attn.{axis}"),
                vec![span, head_dim],
                &mut seed,
            );
        }
        put(
            &mut w,
            format!("{bp}.mlp.lin1.weight"),
            vec![4 * ed, ed],
            &mut seed,
        );
        put(
            &mut w,
            format!("{bp}.mlp.lin1.bias"),
            vec![4 * ed],
            &mut seed,
        );
        put(
            &mut w,
            format!("{bp}.mlp.lin2.weight"),
            vec![ed, 4 * ed],
            &mut seed,
        );
        put(&mut w, format!("{bp}.mlp.lin2.bias"), vec![ed], &mut seed);
    }

    put(
        &mut w,
        format!("{prefix}.neck.0.weight"),
        vec![oc, 1, 1, ed],
        &mut seed,
    );
    put(
        &mut w,
        format!("{prefix}.neck.1.weight"),
        vec![oc],
        &mut seed,
    );
    put(&mut w, format!("{prefix}.neck.1.bias"), vec![oc], &mut seed);
    put(
        &mut w,
        format!("{prefix}.neck.2.weight"),
        vec![oc, 3, 3, oc],
        &mut seed,
    );
    put(
        &mut w,
        format!("{prefix}.neck.3.weight"),
        vec![oc],
        &mut seed,
    );
    put(&mut w, format!("{prefix}.neck.3.bias"), vec![oc], &mut seed);
    put(
        &mut w,
        format!("{prefix}.net_2.weight"),
        vec![2 * oc, 3, 3, oc],
        &mut seed,
    );
    put(
        &mut w,
        format!("{prefix}.net_3.weight"),
        vec![config.final_out_chans, 3, 3, 2 * oc],
        &mut seed,
    );
    w
}

/// A depth-2 tower (one windowed block, one global) at the checkpoint's real
/// 1024x1024 geometry.
///
/// The 256-token claim comes from the geometry, not the depth: patch 16 gives a
/// 64x64 grid, the two stride-2 compressor convs halve it twice to 16x16, and
/// `net_3` widens to 1024. Every block preserves `(H, W, C)`, so trimming the
/// stack from 12 keeps the assertion exact while keeping the test's global
/// attention to a single 4096-token pass.
fn thin_config() -> SamConfig {
    SamConfig {
        depth: 2,
        global_attn_indexes: vec![1],
        ..SamConfig::default()
    }
}

/// The tower turns one 1024x1024 page into 256 rows of width 1024, and the
/// projector keeps that shape.
#[test]
fn tower_output_is_256_tokens_of_1024() {
    let config = thin_config();
    let weights = sam_weights(&config, "vision_tower");
    let tower = SamEncoder::from_weights(&weights, "vision_tower", config.clone())
        .expect("build tower from canonical keys");

    let pixels = filled(&[1, 1024, 1024, 3], 99);
    let grid = tower.forward(&pixels);
    assert_eq!(
        mlxcel_core::array_shape(&grid),
        vec![1, 16, 16, config.final_out_chans]
    );

    // The projector is `Linear(1024, 1024)`; the flatten in between is what
    // produces the 256 rows the prompt's `<imgpad>` run has to match.
    let projector = UnifiedLinear::from_weights(
        &{
            let mut w = WeightMap::new();
            w.insert(
                "multi_modal_projector.weight".to_string(),
                filled(&[1024, 1024], 7),
            );
            w.insert("multi_modal_projector.bias".to_string(), filled(&[1024], 8));
            w
        },
        "multi_modal_projector",
        64,
        4,
    )
    .expect("build projector");

    let flat = mlxcel_core::reshape(&grid, &[1, 256, config.final_out_chans]);
    let projected = projector.forward(&flat);
    assert_eq!(mlxcel_core::array_shape(&projected), vec![1, 256, 1024]);

    // Finite logits are the real acceptance bar: a wrong conv layout or a
    // dropped `1e-6` LayerNorm epsilon shows up here as NaN, not as a shape.
    let total = mlxcel_core::sum_all(&mlxcel_core::abs(&projected));
    mlxcel_core::eval(&total);
    let value = mlxcel_core::item_f32(&total);
    assert!(value.is_finite(), "tower produced non-finite features");
}

/// The flatten that turns the tower's `(1, 16, 16, C)` grid into `(1, 256, C)`
/// orders rows by `h * 16 + w`.
///
/// Upstream reaches the same order from NCHW with `flatten(2).permute(0, 2, 1)`.
/// Getting this wrong transposes the page under the decoder, which reads as
/// plausible-but-wrong text rather than as an error.
#[test]
fn grid_flatten_is_row_major_over_the_16x16_grid() {
    let c = 2i32;
    let mut data = Vec::new();
    for h in 0..16 {
        for w in 0..16 {
            for ch in 0..c {
                data.push((h * 16 + w) as f32 + ch as f32 * 1000.0);
            }
        }
    }
    let grid = mlxcel_core::from_slice_f32(&data, &[1, 16, 16, c]);
    let flat = mlxcel_core::reshape(&grid, &[1, 256, c]);
    for (index, expected) in [(0i32, 0.0f32), (1, 1.0), (16, 16.0), (255, 255.0)] {
        let cell = mlxcel_core::slice(&flat, &[0, index, 0], &[1, index + 1, 1]);
        mlxcel_core::eval(&cell);
        assert_eq!(mlxcel_core::item_f32(&cell), expected, "row {index}");
    }
}
