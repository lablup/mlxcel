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

//! Unit tests for the embedding-driven Gemma 3 backbone (random weights).

use super::*;
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;

fn rand(key: &mut u64, shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
    *key += 1;
    let k = mlxcel_core::random_key(*key);
    let x = unsafe { mlxcel_core::random_normal(shape, dtype::FLOAT32, &*k) };
    mlxcel_core::multiply_scalar(&x, scale)
}

fn tiny_args() -> ModelArgs {
    ModelArgs {
        hidden_size: 8,
        num_hidden_layers: 4,
        intermediate_size: 16,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        sliding_window: 4,
        sliding_window_pattern: 2,
        vocab_size: 1,
        ..ModelArgs::default()
    }
}

fn tiny_weights(prefix: &str, args: &ModelArgs) -> WeightMap {
    let mut key = 7u64;
    let mut w = WeightMap::new();
    let (h, i, hd) = (
        args.hidden_size as i32,
        args.intermediate_size as i32,
        args.head_dim as i32,
    );
    let qd = (args.num_attention_heads * args.head_dim) as i32;
    let kd = (args.num_key_value_heads * args.head_dim) as i32;
    for l in 0..args.num_hidden_layers {
        let p = format!("{prefix}.layers.{l}");
        let mut put = |name: &str, shape: &[i32], scale: f32| {
            w.insert(format!("{p}.{name}"), rand(&mut key, shape, scale));
        };
        put("self_attn.q_proj.weight", &[qd, h], 0.3);
        put("self_attn.k_proj.weight", &[kd, h], 0.3);
        put("self_attn.v_proj.weight", &[kd, h], 0.3);
        put("self_attn.o_proj.weight", &[h, qd], 0.3);
        put("self_attn.q_norm.weight", &[hd], 0.1);
        put("self_attn.k_norm.weight", &[hd], 0.1);
        put("mlp.gate_proj.weight", &[i, h], 0.3);
        put("mlp.up_proj.weight", &[i, h], 0.3);
        put("mlp.down_proj.weight", &[h, i], 0.3);
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            put(&format!("{norm}.weight"), &[h], 0.1);
        }
    }
    w.insert(format!("{prefix}.norm.weight"), rand(&mut key, &[h], 0.1));
    w
}

fn to_vec(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

#[test]
fn caches_follow_the_sliding_global_pattern() {
    let args = tiny_args();
    let backbone = Gemma3Backbone::from_weights(&tiny_weights("bb", &args), "bb", &args).unwrap();
    let caches = backbone.make_caches();
    assert_eq!(caches.len(), 4);
    let globals: Vec<bool> = (0..4).map(|i| caches.is_global(i)).collect();
    assert_eq!(globals, vec![false, true, false, true]);
    assert_eq!(caches.offset(), 0);
}

#[test]
fn missing_layer_weight_is_an_error() {
    let args = tiny_args();
    let mut weights = tiny_weights("bb", &args);
    weights.remove("bb.layers.3.mlp.up_proj.weight");
    let err = Gemma3Backbone::from_weights(&weights, "bb", &args)
        .err()
        .unwrap();
    assert!(err.contains("bb.layers.3.mlp.up_proj"), "{err}");
}

#[test]
fn incremental_steps_match_one_shot_prefill() {
    let args = tiny_args();
    let backbone = Gemma3Backbone::from_weights(&tiny_weights("bb", &args), "bb", &args).unwrap();
    let mut key = 99u64;
    // Five tokens overflow the four-token sliding window in the one-shot
    // call, exercising the explicit sliding mask.
    let x = rand(&mut key, &[2, 5, 8], 1.0);

    let mut full_caches = backbone.make_caches();
    let full = backbone.forward_embeds(&x, &mut full_caches).unwrap();
    assert_eq!(mlxcel_core::array_shape(&full), vec![2, 5, 8]);
    assert_eq!(full_caches.offset(), 5);

    let mut caches = backbone.make_caches();
    let head = mlxcel_core::slice(&x, &[0, 0, 0], &[2, 2, 8]);
    let _ = backbone.forward_embeds(&head, &mut caches).unwrap();
    let mut last = None;
    for t in 2..5 {
        let xt = mlxcel_core::slice(&x, &[0, t, 0], &[2, t + 1, 8]);
        last = Some(backbone.forward_embeds(&xt, &mut caches).unwrap());
    }
    let last = to_vec(&last.unwrap());
    let want = to_vec(&mlxcel_core::slice(&full, &[0, 4, 0], &[2, 5, 8]));
    for (a, b) in last.iter().zip(&want) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn wrong_cache_count_is_rejected() {
    let args = tiny_args();
    let backbone = Gemma3Backbone::from_weights(&tiny_weights("bb", &args), "bb", &args).unwrap();
    let other = Gemma3Backbone::from_weights(
        &tiny_weights(
            "x",
            &ModelArgs {
                num_hidden_layers: 2,
                ..tiny_args()
            },
        ),
        "x",
        &ModelArgs {
            num_hidden_layers: 2,
            ..tiny_args()
        },
    )
    .unwrap();
    let mut key = 1u64;
    let x = rand(&mut key, &[1, 1, 8], 1.0);
    assert!(
        backbone
            .forward_embeds(&x, &mut other.make_caches())
            .is_err()
    );
}

/// `mlx.nn.gelu_approx` as the pinned MLX source defines it
/// (`python/mlx/nn/layers/activations.py`):
///
/// ```python
/// return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
/// ```
///
/// transcribed with Python's evaluation order and scalar rules: `*` and `+`
/// are left-associative, so the leading factor is `(0.5 * x)`; `x**3` is
/// `mx.power`; and every Python float becomes a weak scalar, which the MLX
/// bindings build as `array(float(v), x.dtype)` (an `f32` rounding first, then
/// a cast to `x`'s dtype). This is written from that line, not from
/// [`gelu_approx`], so the comparison below checks the port against the
/// upstream definition on whatever backend runs the test.
///
/// Upstream wraps the function in `@partial(mx.compile, shapeless=True)`.
/// Both the port and this reference are the uncompiled expression: the
/// compiled kernel evaluates the fused graph and can round the last `f32` bit
/// differently (one ulp at x = 4.1 on Metal with mlx 0.32, as #2037 measured),
/// while its bf16 results matched the uncompiled expression exactly there.
fn mlx_nn_gelu_approx_reference(x: &MlxArray) -> UniquePtr<MlxArray> {
    use mlxcel_core::{add, multiply, power, tanh};
    let weak = |v: f64| mlxcel_core::full_f32(&[], v as f32, mlxcel_core::array_dtype(x));
    let x_cubed = power(x, &weak(3.0)); // x**3
    let poly = add(x, &multiply(&weak(0.044715), &x_cubed)); // x + 0.044715 * x**3
    let arg = multiply(&weak((2.0 / std::f64::consts::PI).sqrt()), &poly); // math.sqrt(2 / math.pi) * (...)
    let one_plus = add(&weak(1.0), &tanh(&arg)); // 1 + mx.tanh(...)
    let half_x = multiply(&weak(0.5), x); // 0.5 * x
    multiply(&half_x, &one_plus) // (0.5 * x) * (...)
}

/// The same formula in `f64` on the host: backend-independent, and only
/// accurate to the tolerance its callers allow.
fn gelu_approx_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn bits(a: &MlxArray) -> Vec<u32> {
    to_vec(a).iter().map(|v| v.to_bits()).collect()
}

#[test]
fn gelu_approx_matches_mlx_nn_bit_for_bit() {
    // Bit-for-bit against the upstream expression, evaluated at run time on
    // the same device with the same ops. Hardcoding values instead ties the
    // test to one backend's `tanh`/`power` kernels: Metal and ROCm disagree by
    // one f32 ulp at x = 4.1 (4.099957 vs 4.0999565), and both are correct
    // for their backend. A dense grid over [-8, 8] makes a changed op
    // (`x * x * x` for `power`, a literal rounded in another dtype, a single
    // rounding at the end, the fused GeGLU kernel) show up as a differing ulp
    // somewhere even when a handful of points happen to round the same way:
    // on ROCm, `x * x * x` differs from `power` at only 2 of these 4097 f32
    // inputs.
    let grid: Vec<f32> = (0..=4096)
        .map(|i| -8.0 + 16.0 * i as f32 / 4096.0)
        .collect();
    let xs = mlxcel_core::from_slice_f32(&grid, &[grid.len() as i32]);
    for (name, dt) in [
        ("f32", dtype::FLOAT32),
        ("bf16", dtype::BFLOAT16),
        ("f16", dtype::FLOAT16),
    ] {
        let x = mlxcel_core::astype(&xs, dt);
        let got = gelu_approx(&x);
        assert_eq!(mlxcel_core::array_dtype(&got), dt, "{name}: dtype");
        let want = mlx_nn_gelu_approx_reference(&x);
        let (got, want) = (bits(&got), bits(&want));
        let mismatches: Vec<_> = grid
            .iter()
            .zip(got.iter().zip(&want))
            .filter(|(_, (g, w))| g != w)
            .map(|(x, (g, w))| (*x, f32::from_bits(*g), f32::from_bits(*w)))
            .collect();
        assert!(
            mismatches.is_empty(),
            "{name}: {} of {} elements differ from mlx.nn.gelu_approx (x, got, want): {:?}",
            mismatches.len(),
            grid.len(),
            &mismatches[..mismatches.len().min(8)]
        );
    }

    // Backend-independent sanity values, so the check above cannot pass by
    // comparing a wrong formula with an equally wrong reference. Tolerances:
    // f32 per-op rounding stays within 1e-6 here; bf16 rounds after every op,
    // which moves the value at -3.0 from -0.00364 to -0.00586 on every
    // backend, so its bound is a few bf16 ulps of the largest intermediate.
    let x = [-3.0f32, -1.3, -0.2, 0.0, 0.7, 1.9, 4.1];
    let xs = mlxcel_core::from_slice_f32(&x, &[x.len() as i32]);
    for (name, dt, abs, rel) in [
        ("f32", dtype::FLOAT32, 1e-6, 1e-6),
        ("bf16", dtype::BFLOAT16, 4e-3, 1e-2),
    ] {
        let got = to_vec(&gelu_approx(&mlxcel_core::astype(&xs, dt)));
        for (&xi, &g) in x.iter().zip(&got) {
            let want = gelu_approx_f64(f64::from(xi));
            let err = (f64::from(g) - want).abs();
            assert!(
                err <= abs + rel * want.abs(),
                "{name}: gelu_approx({xi}) = {g}, expected {want} (error {err})"
            );
        }
    }
}
