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

//! Per-expert NVFP4 global scales on `SwitchLinear` (issue #2351).
//!
//! Two scales share the `global_scale` name upstream and mean opposite things:
//! mlx-lm's `quantize_model(..., mode="nvfp4", global_scale=True)` writes
//! `max|w[e]|`, which `gather_qmm` folds into the dequantization as
//! `global_scale / (448 * 6)`; the compressed-tensors and ModelOpt transcodes
//! produce a multiplier for the matmul output. These tests pin the native
//! path against a dense reference, the config flag that selects it, and the
//! stacking of the ModelOpt per-expert sidecar into the output multiplier.

use super::*;
use mlxcel_core::dtype;
use serde_json::json;

const EXPERTS: i32 = 4;
const OUT: i32 = 32;
const IN: i32 = 64;
const PREFIX: &str = "model.layers.0.mlp.switch_mlp.gate_proj";

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32);
        (unit * 2.0 - 1.0) * scale
    }
}

fn to_vec(arr: &MlxArray) -> Vec<f32> {
    let f32_arr = mlxcel_core::astype(arr, dtype::FLOAT32);
    mlxcel_core::eval(&f32_arr);
    mlxcel_core::utils::array_to_vec_f32(&f32_arr)
}

/// A random `[EXPERTS, OUT, IN]` expert stack quantized the way mlx-lm's
/// `_quantize_experts` does: one `mx::quantize(w[e], 16, 4, "nvfp4",
/// global_scale = max|w[e]|)` per expert. Experts get different magnitudes so
/// a scale applied to the wrong expert is visible.
struct NativePlane {
    weights: WeightMap,
    dense: Vec<Vec<f32>>,
    reference: Vec<Vec<f32>>,
    global: Vec<f32>,
}

fn native_plane() -> NativePlane {
    let mut rng = Lcg(0x2351);
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut dense = Vec::new();
    let mut reference = Vec::new();
    let mut global = Vec::new();
    for e in 0..EXPERTS {
        let magnitude = 0.05 * (1 + e) as f32;
        let w: Vec<f32> = (0..OUT * IN).map(|_| rng.next_f32(magnitude)).collect();
        let gs = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        let gs_arr = mlxcel_core::from_slice_f32(&[gs], &[]);
        let q = mlxcel_core::quantize_weights_with_global_scale(
            &mlxcel_core::from_slice_f32(&w, &[OUT, IN]),
            16,
            4,
            "nvfp4",
            &gs_arr,
        );
        let wq = mlxcel_core::quantized_weights_w(&q);
        let sq = mlxcel_core::quantized_weights_scales(&q);
        // `dequantize` without the tensor scale decodes `code * E4M3(scale)`;
        // MLX's kernels multiply that by `global_scale / (448 * 6)`.
        let deq = unsafe { mlxcel_core::dequantize(&wq, &sq, std::ptr::null(), 16, 4, "nvfp4") };
        let factor = gs / (448.0 * 6.0);
        reference.push(to_vec(&deq).iter().map(|v| v * factor).collect());
        packed.push(wq);
        scales.push(sq);
        dense.push(w);
        global.push(gs);
    }
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{PREFIX}.weight"),
        mlxcel_core::stack_owned(&packed, 0),
    );
    weights.insert(
        format!("{PREFIX}.scales"),
        mlxcel_core::stack_owned(&scales, 0),
    );
    weights.insert(
        format!("{PREFIX}.global_scale"),
        mlxcel_core::from_slice_f32(&global, &[EXPERTS]),
    );
    NativePlane {
        weights,
        dense,
        reference,
        global,
    }
}

/// The `config.json` `quantization` block mlx-lm writes for the plane.
fn flagged_config() -> serde_json::Value {
    json!({
        "model_type": "olmoe",
        "quantization": {
            "group_size": 16, "bits": 4, "mode": "nvfp4",
            PREFIX: {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": true}
        }
    })
}

fn unflagged_config() -> serde_json::Value {
    json!({
        "model_type": "olmoe",
        "quantization": {
            "group_size": 16, "bits": 4, "mode": "nvfp4",
            PREFIX: {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        }
    })
}

/// `x @ w.T` for one row.
fn matvec(row: &[f32], w: &[f32]) -> Vec<f32> {
    (0..OUT as usize)
        .map(|o| {
            let w_row = &w[o * IN as usize..(o + 1) * IN as usize];
            row.iter().zip(w_row).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

#[test]
fn native_global_scale_matches_dense_dequantized_reference() {
    let plane = native_plane();
    let mut weights = plane.weights;
    apply_native_global_scale_flags(&mut weights, Some(&flagged_config()))
        .expect("flagged plane resolves");
    let layer = SwitchLinear::from_weights(&weights, PREFIX, 16, 4).expect("native plane loads");
    match &layer {
        SwitchLinear::Quantized {
            mode,
            native_global_scale,
            inv_global_scale,
            ..
        } => {
            assert_eq!(mode, "nvfp4");
            assert!(native_global_scale.is_some());
            assert!(inv_global_scale.is_none());
        }
        SwitchLinear::Regular { .. } => panic!("expected a quantized plane"),
    }
    assert!(
        layer.quantized_parts().is_none(),
        "fused paths have no slot for the native scale"
    );

    let tokens = 5i32;
    let mut rng = Lcg(7);
    let x_rows: Vec<Vec<f32>> = (0..tokens)
        .map(|_| (0..IN).map(|_| rng.next_f32(1.0)).collect())
        .collect();
    let flat: Vec<f32> = x_rows.iter().flatten().copied().collect();
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&flat, &[tokens, 1, 1, IN]),
        dtype::BFLOAT16,
    );
    // Every expert is selected at least once, in both orders.
    let ids = [0, 3, 1, 2, 2, 0, 3, 1, 1, 3];
    let indices = mlxcel_core::from_slice_i32(&ids, &[tokens, 2]);

    let check = |got: &[f32],
                 slot_expert: &dyn Fn(usize) -> usize,
                 slot_token: &dyn Fn(usize) -> usize,
                 label: &str| {
        let slots = got.len() / OUT as usize;
        let mut err = 0f32;
        let mut scale = 0f32;
        let mut dense_err2 = 0f64;
        let mut dense_norm2 = 0f64;
        for s in 0..slots {
            let e = slot_expert(s);
            let row = &x_rows[slot_token(s)];
            let want = matvec(row, &plane.reference[e]);
            let truth = matvec(row, &plane.dense[e]);
            let have = &got[s * OUT as usize..(s + 1) * OUT as usize];
            for ((h, w), t) in have.iter().zip(&want).zip(&truth) {
                err = err.max((h - w).abs());
                dense_err2 += f64::from(h - t).powi(2);
                dense_norm2 += f64::from(*t).powi(2);
            }
            scale = scale.max(max_abs(&want));
        }
        // bf16 activations and output: a few ULP of the largest output.
        assert!(
            err <= 2e-2 * scale,
            "{label}: max |gather_qmm - x @ dequantize(w).T| = {err} against output scale {scale}"
        );
        // And the dequantized reference itself sits within NVFP4's error of
        // the float weights, so the tensor scale was applied once, not lost.
        let rel = (dense_err2 / dense_norm2).sqrt();
        assert!(
            rel < 0.2,
            "{label}: relative L2 error against the float weights {rel}"
        );
    };

    let unsorted = to_vec(&layer.forward(&x, &indices, false));
    assert_eq!(unsorted.len(), (tokens * 2 * OUT) as usize);
    check(&unsorted, &|s| ids[s] as usize, &|s| s / 2, "unsorted");

    // The sorted prefill layout, unsorted back into `[tokens, k, OUT]` with
    // the inverse permutation `SwitchGLU::forward` uses.
    let (sorted_x, sorted_idx, inv) = gather_sort(&x, &indices);
    let sorted = layer.forward(&sorted_x, &sorted_idx, true);
    let restored = to_vec(&scatter_unsort(&sorted, &inv, &[tokens, 2]));
    check(&restored, &|s| ids[s] as usize, &|s| s / 2, "sorted");
    assert!(plane.global.iter().all(|g| *g > 0.0));
}

#[test]
fn global_scale_without_the_config_flag_fails_to_load() {
    let plane = native_plane();
    let mut weights = plane.weights;
    apply_native_global_scale_flags(&mut weights, Some(&unflagged_config()))
        .expect("nothing is flagged, so nothing is resolved");
    let err = SwitchLinear::from_weights(&weights, PREFIX, 16, 4)
        .err()
        .expect("a .global_scale tensor without the config flag must not load");
    assert!(err.contains(PREFIX), "{err}");
    assert!(err.contains("\"global_scale\": true"), "{err}");

    // No config at all behaves the same.
    let plane = native_plane();
    let mut weights = plane.weights;
    apply_native_global_scale_flags(&mut weights, None).unwrap();
    assert!(SwitchLinear::from_weights(&weights, PREFIX, 16, 4).is_err());
}

#[test]
fn config_flag_without_global_scale_tensor_is_a_load_error() {
    let plane = native_plane();
    let mut weights = plane.weights;
    weights.remove(&format!("{PREFIX}.global_scale"));
    let err = apply_native_global_scale_flags(&mut weights, Some(&flagged_config()))
        .expect_err("a flag naming no tensor must fail the load");
    assert!(err.contains(PREFIX), "{err}");
    assert!(err.contains(".global_scale"), "{err}");

    // `global_scale: false` is not a flag.
    let mut config = flagged_config();
    config["quantization"][PREFIX]["global_scale"] = json!(false);
    let plane = native_plane();
    let mut weights = plane.weights;
    weights.remove(&format!("{PREFIX}.global_scale"));
    apply_native_global_scale_flags(&mut weights, Some(&config)).unwrap();
}

#[test]
fn checkpoint_cannot_supply_the_resolved_native_key_itself() {
    let plane = native_plane();
    let mut weights = plane.weights;
    let gs = weights.remove(&format!("{PREFIX}.global_scale")).unwrap();
    weights.insert(format!("{PREFIX}.{NATIVE_GLOBAL_SCALE_LEAF}"), gs);
    let err = apply_native_global_scale_flags(&mut weights, Some(&unflagged_config()))
        .expect_err("the resolved key is reserved for the loader");
    assert!(err.contains(NATIVE_GLOBAL_SCALE_LEAF), "{err}");
}

#[test]
fn native_global_scale_needs_nvfp4_and_one_entry_per_expert() {
    let plane = native_plane();
    let mut weights = plane.weights;
    apply_native_global_scale_flags(&mut weights, Some(&flagged_config())).unwrap();
    let err = SwitchLinear::from_weights_with_mode(&weights, PREFIX, 32, 4, "mxfp4")
        .err()
        .expect("MLX rejects a tensor scale outside nvfp4");
    assert!(err.contains(PREFIX) && err.contains("nvfp4"), "{err}");

    weights.insert(
        format!("{PREFIX}.{NATIVE_GLOBAL_SCALE_LEAF}"),
        mlxcel_core::from_slice_f32(&[1.0, 2.0], &[2]),
    );
    let err = SwitchLinear::from_weights(&weights, PREFIX, 16, 4)
        .err()
        .expect("one scale per expert");
    assert!(err.contains(PREFIX), "{err}");

    weights.insert(
        format!("{PREFIX}.{NATIVE_GLOBAL_SCALE_LEAF}"),
        mlxcel_core::from_slice_f32(&[1.0; EXPERTS as usize], &[EXPERTS]),
    );
    weights.insert(
        format!("{PREFIX}.{INV_GLOBAL_SCALE_LEAF}"),
        mlxcel_core::from_slice_f32(&[1.0; EXPERTS as usize], &[EXPERTS]),
    );
    let err = SwitchLinear::from_weights(&weights, PREFIX, 16, 4)
        .err()
        .expect("the two scales describe different layouts");
    assert!(err.contains(PREFIX), "{err}");
}

/// The ModelOpt repack leaves one `[1]` `weight_scale_2` sidecar per expert
/// plane; the shared stacker turns them into the `[num_experts]` output
/// multiplier, so the routed output of expert `e` is scaled by its own entry.
#[test]
fn per_expert_sidecars_stack_into_the_output_multiplier() {
    let root = "model.layers.0.mlp";
    let mut rng = Lcg(11);
    let mut weights = WeightMap::new();
    let factors = [0.5f32, 2.0, 1.5, 0.25];
    for e in 0..EXPERTS {
        let w: Vec<f32> = (0..OUT * IN).map(|_| rng.next_f32(0.1)).collect();
        let q = mlxcel_core::quantize_weights_with_mode(
            &mlxcel_core::from_slice_f32(&w, &[OUT, IN]),
            16,
            4,
            "nvfp4",
        );
        let base = format!("{root}.experts.{e}.gate_proj");
        weights.insert(
            format!("{base}.weight"),
            mlxcel_core::quantized_weights_w(&q),
        );
        weights.insert(
            format!("{base}.scales"),
            mlxcel_core::quantized_weights_scales(&q),
        );
        weights.insert(
            format!("{base}.global_scale"),
            mlxcel_core::from_slice_f32(&[factors[e as usize]], &[1]),
        );
    }
    let prefix = format!("{root}.switch_mlp.gate_proj");
    let scaled = SwitchLinear::from_weights(&weights, &prefix, 16, 4).expect("stacked plane");
    match &scaled {
        SwitchLinear::Quantized {
            inv_global_scale: Some(inv),
            native_global_scale: None,
            ..
        } => {
            assert_eq!(mlxcel_core::array_shape(inv), vec![EXPERTS]);
            assert_eq!(to_vec(inv), factors.to_vec());
        }
        _ => panic!("expected the stacked multiplier"),
    }
    for e in 0..EXPERTS {
        weights.remove(&format!("{root}.experts.{e}.gate_proj.global_scale"));
    }
    let plain = SwitchLinear::from_weights(&weights, &prefix, 16, 4).unwrap();

    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(
            &(0..2 * IN).map(|_| rng.next_f32(1.0)).collect::<Vec<_>>(),
            &[2, 1, 1, IN],
        ),
        dtype::BFLOAT16,
    );
    let ids = [3, 0, 1, 2];
    let indices = mlxcel_core::from_slice_i32(&ids, &[2, 2]);
    let base = to_vec(&plain.forward(&x, &indices, false));
    let got = to_vec(&scaled.forward(&x, &indices, false));
    for (i, (b, g)) in base.iter().zip(&got).enumerate() {
        let f = factors[ids[i / OUT as usize] as usize];
        assert!(
            (g - b * f).abs() <= 1e-2 * (1.0 + b.abs()),
            "slot {i}: {g} vs {b} * {f}"
        );
    }
}

/// A sidecar on some experts but not others is a corrupted transcode; the
/// short vector fails the per-expert shape check instead of being applied to
/// the wrong experts.
#[test]
fn a_partial_per_expert_sidecar_is_refused() {
    let root = "model.layers.0.mlp";
    let mut weights = WeightMap::new();
    for e in 0..EXPERTS {
        let q = mlxcel_core::quantize_weights_with_mode(
            &mlxcel_core::from_slice_f32(&vec![0.01; (OUT * IN) as usize], &[OUT, IN]),
            16,
            4,
            "nvfp4",
        );
        let base = format!("{root}.experts.{e}.gate_proj");
        weights.insert(
            format!("{base}.weight"),
            mlxcel_core::quantized_weights_w(&q),
        );
        weights.insert(
            format!("{base}.scales"),
            mlxcel_core::quantized_weights_scales(&q),
        );
        if e != 2 {
            weights.insert(
                format!("{base}.global_scale"),
                mlxcel_core::from_slice_f32(&[1.0], &[1]),
            );
        }
    }
    let prefix = format!("{root}.switch_mlp.gate_proj");
    let err = SwitchLinear::from_weights(&weights, &prefix, 16, 4)
        .err()
        .expect("partial sidecar");
    assert!(err.contains(&prefix), "{err}");
}

#[test]
fn stack_expert_leaf_maps_the_sidecar_to_the_output_multiplier() {
    let parts: Vec<UniquePtr<MlxArray>> = [0.5f32, 2.0, 3.0]
        .iter()
        .map(|v| mlxcel_core::from_slice_f32(&[*v], &[1]))
        .collect();
    let ptrs: Vec<*const MlxArray> = parts
        .iter()
        .map(|p| p.as_ref().unwrap() as *const MlxArray)
        .collect();
    let (leaf, stacked) = stack_expert_leaf("global_scale", &ptrs);
    assert_eq!(leaf, INV_GLOBAL_SCALE_LEAF);
    assert_eq!(mlxcel_core::array_shape(&stacked), vec![3]);
    assert_eq!(to_vec(&stacked), vec![0.5, 2.0, 3.0]);

    let (leaf, stacked) = stack_expert_leaf("weight", &ptrs);
    assert_eq!(leaf, "weight");
    assert_eq!(mlxcel_core::array_shape(&stacked), vec![3, 1]);
}

/// A family stacking loop that knows only `weight` / `scales` / `biases`
/// (OLMoE, dots1, ...) stacks the repacked ModelOpt codes and block scales
/// and leaves each expert's `weight_scale_2` sidecar behind. The plane must
/// not then load as plain nvfp4 with every routed expert off by that factor.
#[test]
fn a_plane_whose_family_left_the_modelopt_sidecars_unstacked_is_refused() {
    let root = "model.layers.0.mlp";
    let mut weights = WeightMap::new();
    for e in 0..EXPERTS {
        insert_modelopt_expert_plane(
            &mut weights,
            &format!("{root}.experts.{e}.gate_proj"),
            OUT,
            IN,
            0.5 + e as f32,
            e as u32,
        );
    }
    let prefix = format!("{root}.switch_mlp.gate_proj");
    for leaf in ["weight", "scales"] {
        let parts: Vec<UniquePtr<MlxArray>> = (0..EXPERTS)
            .map(|e| {
                weights
                    .remove(&format!("{root}.experts.{e}.gate_proj.{leaf}"))
                    .expect("per-expert leaf")
            })
            .collect();
        let ptrs: Vec<*const MlxArray> = parts
            .iter()
            .map(|p| p.as_ref().unwrap() as *const MlxArray)
            .collect();
        weights.insert(format!("{prefix}.{leaf}"), mlxcel_core::stack(&ptrs, 0));
    }
    let err = SwitchLinear::from_weights(&weights, &prefix, 16, 4)
        .err()
        .expect("unstacked sidecars must fail the load");
    assert!(err.contains("left unstacked"), "{err}");
    assert!(err.contains(&format!("{root}.experts.")), "{err}");

    for e in 0..EXPERTS {
        weights.remove(&format!("{root}.experts.{e}.gate_proj.global_scale"));
    }
    SwitchLinear::from_weights(&weights, &prefix, 16, 4)
        .expect("the same plane without leftover sidecars still loads");
}
