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
//! GPU-against-CPU check for the device FFT, plus the plan-cache stress that
//! found the hipFFT hang in lablup/mlxcel#1876.
//!
//! Two modes, because the two questions are different and one of them can hang.
//!
//!     cargo run --profile test-fast -p mlxcel-core --features rocm --example fft_numeric_probe
//!     cargo run --profile test-fast -p mlxcel-core --features rocm --example fft_numeric_probe -- --plans 40
//!
//! Select `cuda` or `metal` instead for those backends. Without a backend feature,
//! both arms use the CPU; that is a smoke test, not GPU validation.
//!
//! `correctness` (the default) runs `rfft`, `irfft`, a round trip and a complex
//! `fft` on the GPU and on the CPU stream and reports the relative difference.
//! It covers the lengths mlxcel actually asks for (Kokoro's STFT at 1200,
//! Phi-4-multimodal at 512) plus a non-power-of-two, and batches from 3 to 512.
//!
//! `--plans N` is the stress: N distinct transform shapes in one process,
//! exercising plan creation and eviction at the configured cache limit.
//! Each shape runs both a forward and an inverse transform. On ROCm, past roughly a
//! dozen live hipFFT plans the next plan creation never returns, with the GPU
//! idle and the calling thread asleep (lablup/mlxcel#1876). The cap that bounds
//! it is `MLX_ROCM_FFT_CACHE_SIZE`, default 8, so
//!
//!     cargo build --profile test-fast -p mlxcel-core --features rocm --example fft_numeric_probe
//!     MLX_ROCM_FFT_CACHE_SIZE=32 timeout 120s target/test-fast/examples/fft_numeric_probe --plans 40
//!
//! is the reproduction, and the same command without the variable is the
//! control. Run it under `timeout` if you are scripting it: the failure is a
//! hang, not an error, so nothing comes back on its own.
//! Build before applying the timeout so compilation is not mistaken for a hang.
//! Numerical failures exit with status 1; invalid arguments exit with status 2.
//!
//! The reference is the CPU stream rather than another host. Two accelerators
//! are not expected to agree bit for bit, and the CPU arm is available wherever
//! this runs, so a difference here is this backend's and not a portability
//! question. See `docs/benchmark_results/versions/v0.8.0.md#rocm-correctness-gfx1151-2026-09-12`
//! for why byte-identity is not the cross-backend criterion.

use std::process::ExitCode;

use mlxcel_core::streams::DefaultDeviceGuard;
use mlxcel_core::{self, MlxArray, UniquePtr, dtype};

/// Transform lengths worth covering: 2 and 8 are degenerate, 400 is Whisper's
/// frame and not a power of two, 512 is the Phi-4-multimodal front end, 1200 is
/// Kokoro's STFT, and 1024/2048 bracket them. Odd lengths exercise the inverse
/// real-transform output size rather than only its even-length default.
const LENGTHS: &[i32] = &[2, 3, 7, 8, 400, 401, 512, 1024, 2048];
const KOKORO_N_FFT: i32 = 1200;
const PLAN_BATCH: i32 = 4;
const FIRST_PLAN_LENGTH: i32 = 64;
const PLAN_LENGTH_STEP: i32 = 8;
// Bound both the transform dimension and input's i32 batch * length product.
const MAX_PLAN_COUNT: usize =
    ((i32::MAX / PLAN_BATCH - FIRST_PLAN_LENGTH) / PLAN_LENGTH_STEP + 1) as usize;
const USAGE: &str = "Usage: fft_numeric_probe [--plans N | --help]";

#[derive(Debug, PartialEq, Eq)]
enum Mode {
    Correctness,
    Plans(usize),
    Help,
}

fn parse_mode(args: &[String]) -> Result<Mode, String> {
    match args {
        [] => Ok(Mode::Correctness),
        [flag] if flag == "--help" || flag == "-h" => Ok(Mode::Help),
        [flag, value] if flag == "--plans" => {
            let count = value
                .parse::<usize>()
                .map_err(|_| format!("invalid plan count: {value}"))?;
            if !(1..=MAX_PLAN_COUNT).contains(&count) {
                return Err(format!("plan count must be in 1..={MAX_PLAN_COUNT}"));
            }
            Ok(Mode::Plans(count))
        }
        _ => Err("expected no arguments, --plans N, or --help".to_owned()),
    }
}

#[derive(Debug, Default)]
struct Comparisons {
    failures: usize,
}

impl Comparisons {
    fn record(&mut self, relative_error: f32) -> bool {
        let passed = relative_error.is_finite() && (0.0..TOLERANCE).contains(&relative_error);
        self.failures += usize::from(!passed);
        passed
    }

    fn exit_code(&self) -> ExitCode {
        if self.failures == 0 {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    }
}

fn pseudo_random(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs(a: &MlxArray) -> f32 {
    let f = mlxcel_core::astype(a, dtype::FLOAT32);
    let m = mlxcel_core::max_all(&mlxcel_core::abs(&f));
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m)
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    max_abs(&mlxcel_core::subtract(
        &mlxcel_core::astype(a, dtype::FLOAT32),
        &mlxcel_core::astype(b, dtype::FLOAT32),
    ))
}

/// Relative difference of two complex arrays, real and imaginary parts summed
/// and scaled by the reference's magnitude.
fn complex_rel(gpu: &MlxArray, cpu: &MlxArray) -> f32 {
    let scale = max_abs(&mlxcel_core::real_part(cpu)) + max_abs(&mlxcel_core::imag_part(cpu));
    let d = max_abs_diff(&mlxcel_core::real_part(gpu), &mlxcel_core::real_part(cpu))
        + max_abs_diff(&mlxcel_core::imag_part(gpu), &mlxcel_core::imag_part(cpu));
    d / if scale > 0.0 { scale } else { 1.0 }
}

fn real_rel(gpu: &MlxArray, cpu: &MlxArray) -> f32 {
    let scale = max_abs(cpu);
    max_abs_diff(gpu, cpu) / if scale > 0.0 { scale } else { 1.0 }
}

fn input(batch: i32, n: i32, seed: u32) -> UniquePtr<MlxArray> {
    let v = pseudo_random((batch * n) as usize, seed);
    mlxcel_core::from_slice_f32(&v, &[batch, n])
}

/// Run `f` with the default device pinned, evaluating before the guard drops so
/// the work actually lands on that device rather than wherever it resolves
/// later.
fn on_gpu<F: FnOnce() -> UniquePtr<MlxArray>>(f: F) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::gpu();
    let out = f();
    mlxcel_core::eval(&out);
    out
}

fn on_cpu<F: FnOnce() -> UniquePtr<MlxArray>>(f: F) -> UniquePtr<MlxArray> {
    let _guard = DefaultDeviceGuard::cpu();
    let out = f();
    mlxcel_core::eval(&out);
    out
}

const TOLERANCE: f32 = 1e-5;

fn correctness() -> Comparisons {
    println!(
        "FFT, GPU against the CPU stream. Pass is a relative difference under {TOLERANCE:.0e}.\n"
    );
    println!(
        "{:<10}{:>7}{:>8}{:>14}{:>8}",
        "case", "batch", "n", "rel", "verdict"
    );

    let mut comparisons = Comparisons::default();
    let mut report = |case: &str, batch: i32, n: i32, rel: f32| {
        let ok = comparisons.record(rel);
        println!(
            "{:<10}{:>7}{:>8}{:>14.3e}{:>8}",
            case,
            batch,
            n,
            rel,
            if ok { "ok" } else { "FAIL" }
        );
    };

    let mut lengths = LENGTHS.to_vec();
    lengths.push(KOKORO_N_FFT);

    for &n in &lengths {
        let x = input(4, n, 0xC0FFEE ^ n as u32);
        let g = on_gpu(|| mlxcel_core::rfft(&x, n, 1));
        let c = on_cpu(|| mlxcel_core::rfft(&x, n, 1));
        report("rfft", 4, n, complex_rel(&g, &c));
    }

    for &n in &lengths {
        let x = input(4, n, 0xBEEF00 ^ n as u32);
        // The spectrum comes from the CPU so the inverse is compared on the same
        // input rather than on each backend's own forward result.
        let spec = on_cpu(|| mlxcel_core::rfft(&x, n, 1));
        let g = on_gpu(|| mlxcel_core::irfft(&spec, n, 1));
        let c = on_cpu(|| mlxcel_core::irfft(&spec, n, 1));
        report("irfft", 4, n, real_rel(&g, &c));
    }

    for &n in &lengths {
        let x = input(3, n, 0x1234 ^ n as u32);
        let back = on_gpu(|| {
            let spec = mlxcel_core::rfft(&x, n, 1);
            mlxcel_core::irfft(&spec, n, 1)
        });
        report("roundtrip", 3, n, real_rel(&back, &x));
    }

    for &n in &[7, 8, 512, KOKORO_N_FFT] {
        let x = mlxcel_core::astype(&input(4, n, 0x77 ^ n as u32), dtype::COMPLEX64);
        let g = on_gpu(|| mlxcel_core::fft(&x, n, 1));
        let c = on_cpu(|| mlxcel_core::fft(&x, n, 1));
        report("fft", 4, n, complex_rel(&g, &c));
    }

    // A batch large enough that the launch geometry is not the trivial one.
    {
        let (batch, n) = (512, 512);
        let x = input(batch, n, 0xABCD);
        let g = on_gpu(|| mlxcel_core::rfft(&x, n, 1));
        let c = on_cpu(|| mlxcel_core::rfft(&x, n, 1));
        report("rfft", batch, n, complex_rel(&g, &c));
    }

    println!();
    let failures = comparisons.failures;
    if failures == 0 {
        println!("all FFT results match the CPU stream");
    } else {
        println!("{failures} comparison(s) disagreed with the CPU stream");
    }
    comparisons
}

/// Exercise `count` distinct transform shapes in one process. The backend's
/// cache capacity determines how many forward/inverse plans remain live.
fn plans(count: usize) -> Comparisons {
    println!(
        "Plan-cache stress: {count} distinct shapes in one process.\n\
         On ROCm this hangs rather than failing once too many plans are live \
         (lablup/mlxcel#1876);\n\
         run it under `timeout` and treat no output as the failure.\n"
    );
    let mut comparisons = Comparisons::default();
    for i in 0..count {
        // The parser bounds count so the dimension and batch product fit i32.
        let n = FIRST_PLAN_LENGTH + (i as i32) * PLAN_LENGTH_STEP;
        let x = input(PLAN_BATCH, n, 0x5EED ^ n as u32);
        let spec = on_gpu(|| mlxcel_core::rfft(&x, n, 1));
        let back = on_gpu(|| mlxcel_core::irfft(&spec, n, 1));
        let rel = real_rel(&back, &x);
        let ok = comparisons.record(rel);
        println!(
            "plan {:>3}  n={:<6} roundtrip rel={:.3e} {}",
            i,
            n,
            rel,
            if ok { "ok" } else { "FAIL" }
        );
    }
    println!(
        "\ncompleted {count} transform shapes without blocking; {} numerical failure(s)",
        comparisons.failures
    );
    comparisons
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_mode(&args) {
        Ok(Mode::Correctness) => correctness().exit_code(),
        Ok(Mode::Plans(count)) => plans(count).exit_code(),
        Ok(Mode::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Mode, String> {
        parse_mode(&args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn accepts_default_help_and_explicit_plan_counts() {
        assert_eq!(parse(&[]), Ok(Mode::Correctness));
        assert_eq!(parse(&["--help"]), Ok(Mode::Help));
        assert_eq!(parse(&["-h"]), Ok(Mode::Help));
        assert_eq!(parse(&["--plans", "40"]), Ok(Mode::Plans(40)));
        assert_eq!(
            parse(&["--plans", &MAX_PLAN_COUNT.to_string()]),
            Ok(Mode::Plans(MAX_PLAN_COUNT))
        );
    }

    #[test]
    fn rejects_missing_malformed_empty_and_overflowing_plan_requests() {
        for args in [
            vec!["--plans"],
            vec!["--plans", "no"],
            vec!["--plans", "-1"],
            vec!["--plans", "0"],
            vec!["--plans", "1", "extra"],
            vec!["--unknown"],
        ] {
            assert!(parse(&args).is_err(), "accepted {args:?}");
        }
        assert!(parse(&["--plans", &(MAX_PLAN_COUNT + 1).to_string()]).is_err());
        assert!(parse(&["--plans", &usize::MAX.to_string()]).is_err());
    }

    #[test]
    fn bounded_plan_geometry_does_not_overflow_the_input_product() {
        let last = FIRST_PLAN_LENGTH + (MAX_PLAN_COUNT as i32 - 1) * PLAN_LENGTH_STEP;
        assert!(last.checked_mul(PLAN_BATCH).is_some());
        assert!((last + PLAN_LENGTH_STEP).checked_mul(PLAN_BATCH).is_none());
    }

    #[test]
    fn numerical_failures_survive_later_passing_comparisons() {
        let mut comparisons = Comparisons::default();
        assert!(comparisons.record(0.0));
        assert!(comparisons.record(TOLERANCE / 2.0));
        assert_eq!(comparisons.exit_code(), ExitCode::SUCCESS);
        for error in [TOLERANCE, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            assert!(!comparisons.record(error));
        }
        assert!(comparisons.record(0.0));
        assert_eq!(comparisons.failures, 5);
        assert_eq!(comparisons.exit_code(), ExitCode::FAILURE);
    }
}
