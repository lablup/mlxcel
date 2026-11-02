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

//! Custom CUDA kernel launches shared by the JIT disk-cache tests
//! (`tests/cuda_custom_kernel_disk_cache.rs`, `tests/cuda_ptx_cache_trust.rs`).
//!
//! Both files re-run their own test binary as a child process and compare the
//! child's kernel output with a reference bit for bit, so the launch, the
//! output encoding on stderr and its parser live here.

use std::path::{Path, PathBuf};

use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// The stderr prefix of the line [`emit`] prints and [`parse_out`] reads.
pub const OUT_TAG: &str = "MLXCEL_DISK_CACHE_OUT=";
/// The first words of the notice the `jit_module.cpp` overlay prints once per
/// process before its first NVRTC compile.
pub const JIT_NOTICE: &str = "Compiling CUDA kernels";

const DIN: i32 = 64;
const DFF: i32 = 64;
const EXPERTS: i32 = 2;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;

/// A deterministic ramp of `len` values.
pub fn values(len: i32, start: f32, step: f32) -> Vec<f32> {
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
pub fn launch_fused_moe_pair() -> Vec<f32> {
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

/// Print the output as hex bit patterns so a parent can compare it exactly.
pub fn emit(out: &[f32]) {
    let hex: Vec<String> = out.iter().map(|v| format!("{:08x}", v.to_bits())).collect();
    eprintln!("{OUT_TAG}{}", hex.join(","));
}

/// The bit patterns a child printed with [`emit`].
pub fn parse_out(stderr: &str) -> Vec<u32> {
    stderr
        .lines()
        .find_map(|l| l.strip_prefix(OUT_TAG))
        .expect("child printed its output")
        .split(',')
        .map(|h| u32::from_str_radix(h, 16).expect("hex word"))
        .collect()
}

/// A fresh temporary directory with mode 0700. The JIT overlay refuses a
/// group- or other-writable cache directory (#2539), and `tempfile` leaves the
/// mode to the umask, which is 002 on hosts with user private groups.
pub fn private_tempdir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700))
            .expect("make the temp dir private");
    }
    dir
}

/// Whether `name` is a device subdirectory of the PTX cache root,
/// `sm<major><minor>-nvrtc<major>.<minor>` (issue #2539).
pub fn is_device_dir_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("sm") else {
        return false;
    };
    let Some((cc, nvrtc)) = rest.split_once("-nvrtc") else {
        return false;
    };
    let Some((major, minor)) = nvrtc.split_once('.') else {
        return false;
    };
    [cc, major, minor]
        .iter()
        .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

/// The single device subdirectory of the PTX cache root `root`, if exactly one
/// exists.
pub fn device_dir(root: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|entry| entry.file_name().to_str().is_some_and(is_device_dir_name))
        .map(|entry| entry.path())
        .collect();
    if dirs.len() == 1 { dirs.pop() } else { None }
}

/// Every file under `dir` (recursively, without following symlinks) whose
/// extension is `ext`.
pub fn files_with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.filter_map(|entry| entry.ok()) {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == ext) {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}
