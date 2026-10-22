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

//! Custom CUDA kernels are compiled once per machine, not once per process
//! (issue #2492).
//!
//! Upstream MLX hands `use_disk_cache = false` to the JIT module of every
//! `mx.fast.cuda_kernel`, so the NVRTC compile of mlxcel's fused decode
//! kernels was paid again by every process, inside its first decode step. On
//! GB10 the fused MoE pair alone cost about 1.2 s per `mlxcel generate`, which
//! read as a 20% granite-4.0-h decode regression at `-n 256` although the
//! kernel is faster per token
//! (`docs/benchmark_results/fused-moe-jit-cache-gb10-2026-10-22.md`). The
//! `jit_module.cpp` overlay now writes custom kernels to the disk cache too,
//! publishing every file by rename.
//!
//! The test launches the fused MoE kernel pair against a fresh
//! `MLX_PTX_CACHE_DIR`, checks that both custom kernels left a complete cache
//! entry and no temporary file behind, then re-runs itself as a child process
//! on the same directory and requires that the child compiles nothing: the JIT
//! notice that every NVRTC compile prints must be absent from its stderr.
//!
//! The same child also runs the compiled f32 tanh GELU on bf16 input (#2423).
//! Its fused module name is 234 characters, short enough to stay one path
//! component but long enough that the overlay's old `<name>.txt.tmp<pid>_<tick>`
//! temporary passed NAME_MAX, so the entry was never written and every process
//! compiled it again.
//!
//! `MLX_PTX_CACHE_DIR` is read once per process at the first JIT compile, so
//! this file holds a single test and sets the variable before any kernel runs.
//!
//! ```sh
//! cargo test --profile test-fast --features cuda --test cuda_custom_kernel_disk_cache --no-run
//! gpu-lock run --tag disk-cache -- target/test-fast/deps/cuda_custom_kernel_disk_cache-<hash>
//! ```

#![cfg(feature = "cuda")]

use std::path::Path;
use std::process::Command;

use mlxcel_core::{MlxArray, UniquePtr, dtype};

const TEST_NAME: &str = "custom_kernels_are_compiled_once_per_cache_dir";
const CHILD_ENV: &str = "MLXCEL_TEST_DISK_CACHE_CHILD";
const OUT_TAG: &str = "MLXCEL_DISK_CACHE_OUT=";
const JIT_NOTICE: &str = "Compiling CUDA kernels";

const DIN: i32 = 64;
const DFF: i32 = 64;
const EXPERTS: i32 = 2;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

fn values(len: i32, start: f32, step: f32) -> Vec<f32> {
    (0..len).map(|idx| start + idx as f32 * step).collect()
}

/// `[EXPERTS, rows, cols]` affine 4-bit weight, scales and biases, quantized
/// from a deterministic ramp.
fn expert_stack(
    rows: i32,
    cols: i32,
    start: f32,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    let dense = mlxcel_core::from_slice_f32(
        &values(EXPERTS * rows * cols, start, 1.0 / 4096.0),
        &[EXPERTS * rows, cols],
    );
    let dense = mlxcel_core::astype(&dense, dtype::BFLOAT16);
    let w = mlxcel_core::quantize_weights_w(&dense, GROUP_SIZE, BITS);
    let s = mlxcel_core::quantize_weights_scales(&dense, GROUP_SIZE, BITS);
    let b = mlxcel_core::quantize_weights_biases(&dense, GROUP_SIZE, BITS);
    let packed = *mlxcel_core::array_shape(&w)
        .last()
        .expect("quantized weight shape");
    let groups = *mlxcel_core::array_shape(&s).last().expect("scales shape");
    (
        mlxcel_core::reshape(&w, &[EXPERTS, rows, packed]),
        mlxcel_core::reshape(&s, &[EXPERTS, rows, groups]),
        mlxcel_core::reshape(&b, &[EXPERTS, rows, groups]),
    )
}

/// One single-token fused MoE step: launches the gate-up and the down custom
/// kernels (`moe_gateup_kernel_cu`, `moe_down_kernel_cu`).
fn launch_fused_moe_pair() -> Vec<f32> {
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&values(DIN, -0.4, 0.0125), &[DIN]),
        dtype::BFLOAT16,
    );
    let indices = mlxcel_core::from_slice_u32(&[0, 1], &[EXPERTS]);
    let scores = mlxcel_core::from_slice_f32(&[0.25, 0.75], &[EXPERTS]);
    let (gate_w, gate_s, gate_b) = expert_stack(DFF, DIN, -0.5);
    let (up_w, up_s, up_b) = expert_stack(DFF, DIN, -0.3);
    let (down_w, down_s, down_b) = expert_stack(DIN, DFF, -0.4);
    let out = mlxcel_core::fused_moe_expert_kernel(
        &x, &indices, &gate_w, &gate_s, &gate_b, &up_w, &up_s, &up_b, &down_w, &down_s, &down_b,
        &scores, DIN, DFF, EXPERTS, BITS, BITS, GROUP_SIZE,
    )
    .expect("the CUDA build has both fused MoE kernel ports");
    mlxcel_core::eval(&out);
    let out = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(&out, dtype::FLOAT32));
    assert!(
        out.iter().all(|v| v.is_finite()),
        "the fused MoE kernel pair must produce finite output"
    );
    out
}

/// The compiled f32 tanh GELU on a bf16 input: its fused kernel has a module
/// name in the length window the cache once failed to write (#2423).
fn launch_compiled_tanh_gelu() -> Vec<f32> {
    let x = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&values(DIN, -6.0, 0.1875), &[DIN]),
        dtype::BFLOAT16,
    );
    let out = mlxcel_core::compiled_gelu_tanh_f32(&x);
    mlxcel_core::eval(&out);
    mlxcel_core::utils::array_to_vec_f32(&out)
}

/// Every kernel this test checks, in a fixed order.
fn launch_all() -> Vec<f32> {
    let mut out = launch_fused_moe_pair();
    out.extend(launch_compiled_tanh_gelu());
    out
}

/// Print the output as hex bit patterns so a parent can compare it exactly.
fn emit(out: &[f32]) {
    let hex: Vec<String> = out.iter().map(|v| format!("{:08x}", v.to_bits())).collect();
    eprintln!("{OUT_TAG}{}", hex.join(","));
}

fn parse_out(stderr: &str) -> Vec<u32> {
    stderr
        .lines()
        .find_map(|l| l.strip_prefix(OUT_TAG))
        .expect("child printed its output")
        .split(',')
        .map(|h| u32::from_str_radix(h, 16).expect("hex word"))
        .collect()
}

/// The `.ptx` cache entry of the custom kernel named `kernel`, if any. The
/// file name is `custom_kernel_<kernel><template-args hash>_<source hash>.ptx`
/// (`mlx/backend/cuda/custom_kernel.cpp`), well under the length at which
/// `get_ptx_path` nests directories.
fn cache_entry(dir: &Path, kernel: &str) -> Option<std::path::PathBuf> {
    let prefix = format!("custom_kernel_{kernel}");
    std::fs::read_dir(dir)
        .expect("cache dir is readable")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .find(|path| {
            path.extension().is_some_and(|ext| ext == "ptx")
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&prefix))
        })
}

#[test]
fn custom_kernels_are_compiled_once_per_cache_dir() {
    if std::env::var_os(CHILD_ENV).is_some() {
        // Child: the parent already populated MLX_PTX_CACHE_DIR.
        emit(&launch_all());
        return;
    }
    if mlxcel_core::hardware::gpu_backend_kind() != mlxcel_core::hardware::GpuBackendKind::Cuda {
        eprintln!("skipping: no CUDA device");
        return;
    }

    let cache = tempfile::tempdir().expect("temp cache dir");
    // SAFETY: this binary holds this one test, and nothing has touched the
    // CUDA JIT yet, so no other thread reads the environment here; MLX reads
    // the variable once, at the first JIT compile below.
    unsafe { std::env::set_var("MLX_PTX_CACHE_DIR", cache.path()) };

    let parent_out = launch_all();

    for kernel in ["moe_gateup_kernel_cu", "moe_down_kernel_cu"] {
        let ptx = cache_entry(cache.path(), kernel).unwrap_or_else(|| {
            panic!("custom kernel {kernel} left no disk cache entry (issue #2492)")
        });
        let ptx_len = std::fs::metadata(&ptx).map(|m| m.len()).unwrap_or(0);
        let map = std::fs::read_to_string(ptx.with_extension("txt")).unwrap_or_default();
        assert!(ptx_len > 0, "{kernel}: empty compiled-code cache file");
        assert!(
            map.lines().any(|line| line.contains('\t')),
            "{kernel}: the kernel-name map must be complete once the .ptx exists"
        );
    }
    let leftovers: Vec<String> = std::fs::read_dir(cache.path())
        .expect("cache dir is readable")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "every cache file must be published by rename: {leftovers:?}"
    );

    let child = Command::new(std::env::current_exe().expect("test binary path"))
        .args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env("MLX_PTX_CACHE_DIR", cache.path())
        .env_remove("MLXCEL_QUIET_JIT")
        .output()
        .expect("re-run the test binary as a child");
    let stderr = String::from_utf8_lossy(&child.stderr);
    assert!(
        child.status.success(),
        "child run failed: {}\n{stderr}",
        child.status
    );
    assert!(
        !stderr.contains(JIT_NOTICE),
        "a second process on the same cache dir must compile nothing, but it \
         printed the JIT notice:\n{stderr}"
    );
    // The kernels are deterministic (fixed reduction order, no atomics on the
    // output), so code loaded from the cache must reproduce the freshly
    // compiled output bit for bit.
    let parent_bits: Vec<u32> = parent_out.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        parse_out(&stderr),
        parent_bits,
        "cached kernels must match the freshly compiled output bit for bit"
    );

    // Self-healing: truncate one entry's .ptx and break the other's kernel map;
    // a third process must treat both as misses, recompile (JIT notice
    // printed), republish complete entries, and still produce the same output.
    let gateup = cache_entry(cache.path(), "moe_gateup_kernel_cu").expect("gateup entry");
    let down = cache_entry(cache.path(), "moe_down_kernel_cu").expect("down entry");
    let full_len = std::fs::metadata(&gateup).expect("gateup meta").len();
    std::fs::write(&gateup, b"").expect("truncate gateup .ptx");
    std::fs::write(down.with_extension("txt"), b"no-tab-here\n").expect("corrupt down map");
    let healed = Command::new(std::env::current_exe().expect("test binary path"))
        .args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env("MLX_PTX_CACHE_DIR", cache.path())
        .env_remove("MLXCEL_QUIET_JIT")
        .output()
        .expect("re-run the test binary on the damaged cache");
    let healed_err = String::from_utf8_lossy(&healed.stderr);
    assert!(
        healed.status.success(),
        "a damaged cache entry must be recompiled, not fatal: {}\n{healed_err}",
        healed.status
    );
    assert!(
        healed_err.contains(JIT_NOTICE),
        "damaged entries must be recompiled:\n{healed_err}"
    );
    assert_eq!(parse_out(&healed_err), parent_bits);
    assert_eq!(
        std::fs::metadata(&gateup).map(|m| m.len()).unwrap_or(0),
        full_len,
        "the truncated .ptx must be republished complete"
    );
    let repaired_map = std::fs::read_to_string(down.with_extension("txt")).unwrap_or_default();
    assert!(
        repaired_map.lines().any(|l| l.contains('\t')),
        "the corrupt kernel map must be republished"
    );
}
