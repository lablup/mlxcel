/*
 * Standalone cuda-oxide smoke harness (issue #2831).
 *
 * This crate has no MLX dependency. It compiles one bounded elementwise vector
 * add kernel to PTX/cubin with the pinned cuda-oxide toolchain and executes it
 * directly through the CUDA runtime, verifying host-computed reference outputs
 * for small, odd, block-aligned, and tail sizes.
 *
 * Everything here is deliberately outside the mlxcel workspace
 * (spike/cuda-oxide-smoke/Cargo.toml has an empty [workspace] table), so an
 * ordinary mlxcel build never resolves cuda-oxide and never needs LLVM 21.
 */

use std::process::ExitCode;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Threads per block. The manifest records this as a launch constraint, and
/// `scripts/validate_manifest.py` cross-checks the recorded value against the
/// harness output for every case.
const BLOCK_SIZE: u32 = 256;

/// The kernel under test. The host-side launch ABI is generated from this
/// signature: each slice lowers to a pointer plus a length, so the PTX entry
/// takes six `.u64` parameters in the order a_ptr, a_len, b_ptr, b_len,
/// c_ptr, c_len.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        // Threads beyond the output length return here. For a non-multiple of
        // BLOCK_SIZE the grid is rounded up, so this check is the tail guard.
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}

/// One bounded test case. `label` names the size class used by the evidence
/// report; `n` is both the logical length and the exact allocation length of
/// all three buffers.
struct Case {
    label: &'static str,
    n: usize,
}

/// Small (including empty-adjacent single-element), odd, block-aligned, and
/// tail sizes. Every case is a real launch; nothing here is compile-only.
const CASES: &[Case] = &[
    Case {
        label: "small",
        n: 1,
    },
    Case {
        label: "small",
        n: 2,
    },
    Case {
        label: "small",
        n: 3,
    },
    Case {
        label: "small",
        n: 16,
    },
    Case {
        label: "small",
        n: 17,
    },
    Case {
        label: "odd",
        n: 33,
    },
    Case {
        label: "odd",
        n: 999,
    },
    Case {
        label: "tail",
        n: 255,
    },
    Case {
        label: "block",
        n: 256,
    },
    Case {
        label: "tail",
        n: 257,
    },
    Case {
        label: "block",
        n: 512,
    },
    Case {
        label: "tail",
        n: 1000,
    },
    Case {
        label: "tail",
        n: 1025,
    },
    Case {
        label: "tail",
        n: 65537,
    },
];

/// Deterministic 64-bit LCG. Values are exact binary fractions in [-32, 32),
/// generated independently per case so a case cannot pass by sharing data
/// with its neighbour.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in [-32, 32) with a 2^-10 step. Both the value and the exact sum
    /// of two such values fit comfortably in an f32 mantissa for this range,
    /// so a reference comparison can be bitwise rather than tolerance-based.
    fn next_value(&mut self) -> f32 {
        let raw = (self.next_u32() % 65_536) as i32 - 32_768;
        raw as f32 / 1024.0
    }
}

fn inputs(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Lcg::new(seed);
    let a: Vec<f32> = (0..n).map(|_| rng.next_value()).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.next_value()).collect();
    (a, b)
}

fn launch_config(n: usize) -> Result<LaunchConfig, String> {
    let n = u32::try_from(n).map_err(|_| format!("case length {n} exceeds the u32 grid limit"))?;
    if n == 0 {
        return Err("a zero-length launch is not a valid bounded case".to_string());
    }
    Ok(LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn run() -> Result<(), String> {
    if let Some(arg) = std::env::args().nth(1) {
        return match arg.as_str() {
            "-h" | "--help" => {
                println!(
                    "usage: cuda-oxide-smoke\n\nRuns {n} bounded vector-add cases on CUDA device 0 \
                     and compares every element with a host reference.",
                    n = CASES.len()
                );
                Ok(())
            }
            other => Err(format!("unexpected argument: {other}")),
        };
    }

    let ctx = CudaContext::new(0).map_err(|e| format!("failed to create CUDA context: {e}"))?;
    let stream = ctx.default_stream();
    let module =
        kernels::load(&ctx).map_err(|e| format!("failed to load the cuda-oxide module: {e}"))?;

    let mut elements_checked = 0usize;
    for (case_index, case) in CASES.iter().enumerate() {
        let config = launch_config(case.n)?;
        let (a_host, b_host) = inputs(case.n as u64 + case_index as u64, case.n);
        let expected: Vec<f32> = a_host
            .iter()
            .zip(b_host.iter())
            .map(|(a, b)| a + b)
            .collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)
            .map_err(|e| format!("case {}: upload a failed: {e}", case.label))?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)
            .map_err(|e| format!("case {}: upload b failed: {e}", case.label))?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, case.n)
            .map_err(|e| format!("case {}: output allocation failed: {e}", case.label))?;

        // SAFETY: one thread per element, grid and block are derived from the
        // same `n` that sizes all three buffers, and the kernel bounds-checks
        // the output write, so out-of-range threads return before reading a.
        unsafe { module.vecadd(&stream, config, &a_dev, &b_dev, &mut c_dev) }
            .map_err(|e| format!("case {}: launch failed: {e}", case.label))?;

        let got = c_dev
            .to_host_vec(&stream)
            .map_err(|e| format!("case {}: download failed: {e}", case.label))?;
        if got.len() != case.n {
            return Err(format!(
                "case {}: downloaded {} elements, expected {}",
                case.label,
                got.len(),
                case.n
            ));
        }

        let mut mismatches = 0usize;
        let mut first: Option<(usize, f32, f32)> = None;
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            // Bitwise equality: both sides perform exactly one IEEE-754 f32
            // add on exactly representable inputs, so any difference is a real
            // defect rather than a tolerance question.
            if g.to_bits() != e.to_bits() {
                mismatches += 1;
                if first.is_none() {
                    first = Some((i, *g, *e));
                }
            }
        }
        if mismatches != 0 {
            let (i, g, e) = first.expect("mismatches implies a first mismatch");
            return Err(format!(
                "case {} (n={}): {mismatches} of {} elements differ; first at [{i}] got {g:?} expected {e:?}",
                case.label, case.n, case.n
            ));
        }

        elements_checked += case.n;
        println!(
            "PASS case={} n={} grid={} block={} elements={}",
            case.label, case.n, config.grid_dim.0, config.block_dim.0, case.n
        );
    }

    println!(
        "RESULT standalone=1 mlx=0 cases_passed={} cases_total={} elements_checked={} comparison=exact-bitwise",
        CASES.len(),
        CASES.len(),
        elements_checked
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FAIL error={error}");
            ExitCode::FAILURE
        }
    }
}
