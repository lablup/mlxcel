//! Op-level measurement of the fused tanh-GELU f32 Metal kernel against the two
//! chains it replaces.
//!
//! `gelu_tanh_f32` applies one eager elementwise formula and passes about a
//! dozen times over a full-width intermediate. `compiled_gelu_tanh_f32` folds
//! that formula into one `mlx::core::compile` region. The fused kernel is a
//! single dispatch, and this harness prints all three for the shapes the
//! tanh-switched vision encoders actually feed, so the wiring decision has
//! evidence rather than a guess behind it.
//!
//! # Method
//!
//! Every arm allocates its input once and reuses it, so this measures a warm
//! last-level cache; see "Warm vs cold last-level cache" in
//! `docs/benchmarks.md` for what that does and does not say about production.
//! Each `body` is evaluated per iteration so the lazy graph actually runs, with
//! `synchronize_default()` bracketing the timed region.
//!
//! Trials are interleaved and the per-arm minimum is reported. Background load
//! and host throttling only ever add time, so the minimum is the estimator least
//! biased by them, and interleaving puts all three arms under the same
//! conditions. A single non-interleaved pass proved noisy enough to invert the
//! ordering on a loaded host, so do not read one trial.
//!
//! The absolute microseconds are not the claim. The op-level ratio is also a
//! lower bound on the end-to-end effect: the fused kernel removes the whole
//! intermediate from the graph, which shows up as allocator and
//! dependency-tracking pressure rather than as kernel time.
//!
//! Reproduce (use `caffeinate -i` so the host does not idle-throttle the GPU
//! mid-run, and let the machine run cool between sweeps):
//!
//! ```text
//! caffeinate -i cargo run --release --features metal,accelerate \
//!     --example fused_gelu_tanh_microbench
//! ```
//!
//! `fused_gelu_tanh_f32_available()` is false off the GPU stream, so with
//! `MLXCEL_DEVICE=cpu` the harness prints why and exits instead of timing the
//! eager path three times.

use std::time::{Duration, Instant};

use mlxcel_core::utils::{compiled_gelu_tanh_f32, fused_gelu_tanh_f32, gelu_tanh_f32};
use mlxcel_core::{
    MlxArray, UniquePtr, astype, dtype, eval, from_slice_f32, fused_gelu_tanh_f32_available,
    synchronize_default,
};

const WARMUP: usize = 64;
const ITERS: usize = 512;
const TRIALS: usize = 7;

/// Time `iters` calls of `body`, evaluating each result so the lazy graph is
/// actually executed, with `synchronize_default()` bracketing the timed region
/// so queued GPU work is not attributed to the next arm.
fn time_body<F>(warmup: usize, iters: usize, mut body: F) -> Duration
where
    F: FnMut() -> UniquePtr<MlxArray>,
{
    for _ in 0..warmup {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    let start = Instant::now();
    for _ in 0..iters {
        let out = body();
        eval(&out);
    }
    synchronize_default();
    start.elapsed()
}

fn per_call_us(total: Duration, iters: usize) -> f64 {
    total.as_secs_f64() * 1e6 / iters as f64
}

/// Interleaved trials, minimum per arm. Returns `(eager, compiled, fused)`.
fn sweep(x: &MlxArray) -> (f64, f64, f64) {
    let (mut eager, mut compiled, mut fused) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for _ in 0..TRIALS {
        eager = eager.min(per_call_us(
            time_body(WARMUP, ITERS, || gelu_tanh_f32(x)),
            ITERS,
        ));
        compiled = compiled.min(per_call_us(
            time_body(WARMUP, ITERS, || compiled_gelu_tanh_f32(x)),
            ITERS,
        ));
        fused = fused.min(per_call_us(
            time_body(WARMUP, ITERS, || fused_gelu_tanh_f32(x)),
            ITERS,
        ));
    }
    (eager, compiled, fused)
}

fn main() {
    if !fused_gelu_tanh_f32_available() {
        println!(
            "fused_gelu_tanh_f32 has no port for this backend; run with MLXCEL_DEVICE=gpu \
             on a Metal build"
        );
        return;
    }

    // The tanh-switched encoders feed [batch, hidden] f32 GEMM output in bf16,
    // so that is the dtype under test. The tower row is the shape the GELU
    // tests use; the decode row is one token's worth; the large prefill row is
    // an image chunk the tower can actually hand over.
    let shapes: [(&str, [i32; 2]); 3] = [
        ("tower prefill", [64, 4304]),
        ("decode row", [1, 4304]),
        ("large prefill", [1024, 4096]),
    ];

    println!("dtype bf16, warmup {WARMUP}, iters {ITERS}, trials {TRIALS}, min per arm");
    println!();
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "shape", "eager us", "compiled", "fused", "f/compiled", "f/eager"
    );

    let mut csv =
        String::from("shape,eager_us,compiled_us,fused_us,fused_vs_compiled,fused_vs_eager\n");
    for (label, shape) in shapes {
        let n: i32 = shape.iter().product();
        // A deterministic ramp rather than random: the tanh CDF is monotone, so
        // this spans the saturated negative tail, the linear region and the
        // saturated positive tail without depending on an RNG.
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() * 4.0).collect();
        let x = from_slice_f32(&data, &shape);
        let x = astype(&x, dtype::BFLOAT16);
        eval(&x);

        let (eager, compiled, fused) = sweep(&x);
        let vs_compiled = fused / compiled;
        let vs_eager = fused / eager;

        let shape_name = format!("{label} {:?}", shape);
        println!(
            "{:<16} {eager:>10.2} {compiled:>10.2} {fused:>10.2} {vs_compiled:>9.2}x {vs_eager:>9.2}x",
            shape_name
        );
        csv.push_str(&format!(
            "\"{shape_name}\",{eager:.2},{compiled:.2},{fused:.2},{vs_compiled:.4},{vs_eager:.4}\n"
        ));
    }

    println!();
    println!("A ratio below 1.00 is the fused kernel winning.");
    println!();
    println!("```csv");
    print!("{csv}");
    println!("```");
}
