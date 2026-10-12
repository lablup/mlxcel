// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

/// A zero carried state and x = B = 1 make every next-state element equal
/// to the processed timestep. This isolates compute_dt from A_log rounding
/// and the output reduction. CUDA's bf16-preserving mixed-dtype promotion
/// must not narrow that timestep when the stored bias is bf16 (#2142).
#[test]
fn ssm_update_kernel_keeps_timestep_f32_with_bf16_bias() {
    let _guard = crate::test_support::env_lock::env_lock();
    if skip() {
        return;
    }
    // A small, distinct geometry isolates the timestep from the model-sized
    // random recurrence cases above.
    let shape = Shape {
        name: "f32 timestep with mixed bias dtype",
        batch: 1,
        heads: 2,
        head_dim: 8,
        groups: 1,
        state_dim: 32,
        dt_limits: (0.0, f32::INFINITY),
    };
    // Both -1.5 and -1.0 are exactly representable in f32 and bf16, so the
    // expected value depends on no pre-quantization random inputs or GPU op.
    let expected_dt = (-2.5_f32).exp().ln_1p();
    let state_shape = [1, 2, 8, 32];
    let expected_state = full_f32(&state_shape, expected_dt, dtype::FLOAT32);
    eval(&expected_state);

    for act_dtype in [dtype::FLOAT32, dtype::BFLOAT16] {
        // Exercise both bias-dtype transitions at the same geometry. The
        // processed timestep must stay f32 regardless of the stored bias.
        for bias_dtype in [dtype::BFLOAT16, dtype::FLOAT32, dtype::BFLOAT16] {
            let case = Case {
                x: full_f32(&[1, 1, 2, 8], 1.0, act_dtype),
                a_log: full_f32(&[2], 0.0, dtype::FLOAT32),
                b: full_f32(&[1, 1, 1, 32], 1.0, act_dtype),
                c: full_f32(&[1, 1, 1, 32], 0.0, act_dtype),
                d: full_f32(&[2], 0.0, act_dtype),
                dt: full_f32(&[1, 1, 2], -1.5, act_dtype),
                dt_bias: full_f32(&[2], -1.0, bias_dtype),
                state: full_f32(&state_shape, 0.0, dtype::FLOAT32),
            };
            let (out, state) = run_kernel(&shape, &case);
            let label = format!("act dtype {act_dtype}, dt_bias dtype {bias_dtype}");
            assert_eq!(array_dtype(&out), act_dtype, "{label}: output dtype");
            assert_eq!(array_dtype(&state), dtype::FLOAT32, "{label}: state dtype");
            assert_within(&label, &state, &expected_state, dtype::FLOAT32);
        }
    }
}
