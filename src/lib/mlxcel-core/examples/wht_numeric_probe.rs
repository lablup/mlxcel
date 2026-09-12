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
//! Numeric reference for the Walsh-Hadamard transform used by the TurboQuant KV
//! cache, measured so another backend's port has a line to pass.
//!
//! Reports, per head_dim and dtype: the forward norm ratio ||wht(x)|| / ||x||,
//! which is 1 for the orthonormal 1/sqrt(N) scaling MLX applies by default and
//! sqrt(N) for an unscaled Hadamard matrix, and the round-trip error of
//! wht(wht(x)) against x, which the involution makes zero in exact arithmetic.
//!
//! The norm ratio is the column that catches a scale mistake, and the round-trip
//! alone cannot: an unscaled Hadamard is not an involution either, but it shows
//! up at once as sqrt(N), which is 8, 11.31 and 16 for the three head dims here.
//!
//!     cargo run --release -p mlxcel-core --example wht_numeric_probe
//!
//! Metal reference, M1 Ultra (Apple GPU generation 13), mlxcel `1f107710`, MLX
//! pin `81ba1c6a`. The norm ratio is 1.000000 on every row to six decimals, in
//! both dtypes. Round-trip error: fp16 relative RMS 3.8e-4 to 5.0e-4 with max
//! absolute 1.953e-3, one row at 3.906e-3; f32 relative RMS about 1e-7 with max
//! absolute 3.0e-7 to 7.2e-7.
//!
//! Another backend's port passes when its norm ratio is 1 within 1e-3 and its
//! round-trip stays inside those bands: fp16 relative RMS at or under 1e-3 with
//! max at or under 4e-3, f32 relative RMS at or under 1e-6 with max at or under
//! 5e-6. Reading f32 against the fp16 band hides a real defect.
//!
//! `wht()` does not pass a scale to MLX, so this measures the default 1/sqrt(N).
//! MLX's `hadamard_transform` takes an optional scale that the bridge does not
//! expose today; a port should keep the same default.

use mlxcel_core::{self, MlxArray, UniquePtr, dtype};

const HEAD_DIMS: &[i32] = &[64, 128, 256];
const SHAPES: &[(&str, [i32; 3])] = &[("decode [1,32,1,d]", [1, 32, 1]), ("prefill [1,32,512,d]", [1, 32, 512])];

fn random_input(shape: &[i32], seed: u64, dt: i32) -> UniquePtr<MlxArray> {
    let key = mlxcel_core::random_key(seed);
    // SAFETY: `key` is owned by this scope, lives across the call.
    let f32_arr = unsafe {
        mlxcel_core::random_normal(
            shape,
            dtype::FLOAT32,
            key.as_ref().unwrap() as *const MlxArray,
        )
    };
    mlxcel_core::astype(&f32_arr, dt)
}

/// L2 norm of an array, computed in f32 so the reduction itself is not the
/// thing being measured.
fn l2_norm(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, dtype::FLOAT32);
    let sq = mlxcel_core::square(&f);
    let total = mlxcel_core::sum_all(&sq);
    mlxcel_core::eval(&total);
    mlxcel_core::item_f32(&total).sqrt()
}

fn max_abs(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, dtype::FLOAT32);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&f));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn main() {
    println!("host: M1 Ultra, mlxcel wht() = mlx::core::hadamard_transform with no explicit scale\n");
    println!(
        "{:<22}{:>6}{:>8}{:>14}{:>16}{:>16}{:>14}",
        "shape", "d", "dtype", "norm ratio", "roundtrip max", "roundtrip rms", "rel rms"
    );
    for (label, base) in SHAPES {
        for &d in HEAD_DIMS {
            for (dt, dt_name) in [(dtype::FLOAT16, "fp16"), (dtype::FLOAT32, "f32")] {
                let shape = [base[0], base[1], base[2], d];
                let x = random_input(&shape, 0xB0_C0_DE_42 ^ (d as u64), dt);
                mlxcel_core::eval(&x);
                let y = mlxcel_core::wht(&x);
                mlxcel_core::eval(&y);
                let z = mlxcel_core::wht(&y);
                mlxcel_core::eval(&z);

                let nx = l2_norm(&x);
                let ny = l2_norm(&y);
                let diff = mlxcel_core::subtract(&mlxcel_core::astype(&z, dtype::FLOAT32), &mlxcel_core::astype(&x, dtype::FLOAT32));
                let n_diff = l2_norm(&diff);
                let elems = (shape.iter().product::<i32>()) as f32;
                println!(
                    "{:<22}{:>6}{:>8}{:>14.6}{:>16.3e}{:>16.3e}{:>14.3e}",
                    label,
                    d,
                    dt_name,
                    ny / nx,
                    max_abs(&diff),
                    n_diff / elems.sqrt(),
                    n_diff / nx
                );
            }
        }
    }
}
