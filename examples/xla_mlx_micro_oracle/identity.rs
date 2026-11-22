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

//! Build and device identity of the MLX CUDA reference.

use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

/// Overlay trees a CUDA build applies over the pinned MLX checkout
/// (`src/lib/mlxcelverse/CMakeLists.txt`).
const CUDA_OVERLAY_TREES: [&str; 3] = ["core", "cpu", "cuda"];

/// The affine quantized-matmul sources compiled into this binary's MLX. Read
/// at compile time, so this example rebuilds whenever they change, as
/// `mlxcel-core` does through its `rerun-if-changed` on the overlay tree.
const Q4_KERNEL_SOURCES: [(&str, &[u8]); 3] = [
    (
        "mlx/backend/cuda/quantized/quantized.cpp",
        include_bytes!(
            "../../src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/quantized/quantized.cpp"
        ),
    ),
    (
        "mlx/backend/cuda/quantized/qmm/qmm_sm80.cu",
        include_bytes!(
            "../../src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/quantized/qmm/qmm_sm80.cu"
        ),
    ),
    (
        "mlx/backend/cuda/quantized/qmm/qmv.cu",
        include_bytes!(
            "../../src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/quantized/qmm/qmv.cu"
        ),
    ),
];

/// JIT module revision #923 introduced with the `cp.async` drain before the
/// `qmm_sm80` epilogue; every pre-#923 kernel lacks it.
const POST_923_QMM_MARKER: &str = "qmm_sm80_r2";

/// Everything the reports must name about the MLX side.
#[derive(Clone, Debug, Serialize)]
pub struct MlxIdentity {
    pub mlx_commit: &'static str,
    pub overlay_sha256: String,
    pub q4_kernel_sha256: String,
    pub q4_kernel_baseline: &'static str,
    pub mlx_enable_tf32: &'static str,
    pub cuda_architectures: &'static str,
    pub device: String,
    pub compute_capability: String,
    pub mlxcel_version: &'static str,
}

impl MlxIdentity {
    /// The `patch` component of the reference backend identity.
    pub fn patch(&self) -> String {
        format!(
            "overlays-sha256={};q4-kernel-sha256={};q4-kernel={}",
            self.overlay_sha256, self.q4_kernel_sha256, self.q4_kernel_baseline
        )
    }
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn collect_files(directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("read {}: {error}", directory.display()))?
            .path();
        if path.is_dir() {
            collect_files(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

/// SHA-256 over every overlay file's relative path and contents, in path order.
fn overlay_digest(root: &Path) -> Result<String, String> {
    let overlays = root.join("src/lib/mlxcelverse/overlays");
    let mut hasher = Sha256::new();
    for tree in CUDA_OVERLAY_TREES {
        let mut files = Vec::new();
        collect_files(&overlays.join(tree), &mut files)?;
        files.sort();
        for file in files {
            let relative = file
                .strip_prefix(&overlays)
                .map_err(|error| format!("{}: {error}", file.display()))?;
            let bytes = std::fs::read(&file)
                .map_err(|error| format!("read {}: {error}", file.display()))?;
            hasher.update(relative.to_string_lossy().as_bytes());
            hasher.update([0]);
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
    }
    Ok(hex(hasher.finalize()))
}

/// Digest of the compiled-in Q4 kernel sources, refusing a binary whose
/// sources no longer match the checkout or predate #923.
fn q4_kernel_digest(root: &Path) -> Result<String, String> {
    let overlay = root.join("src/lib/mlxcelverse/overlays/cuda");
    let mut hasher = Sha256::new();
    for (relative, compiled) in Q4_KERNEL_SOURCES {
        let current = std::fs::read(overlay.join(relative))
            .map_err(|error| format!("read {relative}: {error}"))?;
        if current != compiled {
            return Err(format!(
                "{relative} changed after this example was built; rebuild it so the Q4 report \
                 names the kernel that actually runs"
            ));
        }
        hasher.update(relative.as_bytes());
        hasher.update([0]);
        hasher.update(compiled);
    }
    let sm80 = String::from_utf8_lossy(Q4_KERNEL_SOURCES[1].1);
    if !sm80.contains(POST_923_QMM_MARKER) {
        return Err(format!(
            "qmm_sm80.cu lacks the {POST_923_QMM_MARKER} revision; this is pre-#923 Q4 evidence"
        ));
    }
    Ok(hex(hasher.finalize()))
}

/// Resolve the identity of the MLX CUDA reference on GPU 0.
pub fn mlx_identity(mlx_enable_tf32: &'static str) -> Result<MlxIdentity, String> {
    if mlxcel_core::gpu_device_count() < 1 {
        return Err("no MLX GPU device is visible".to_string());
    }
    let (major, minor) = mlxcel_core::cuda_arch::cuda_compute_capability()
        .ok_or_else(|| "MLX reports no CUDA compute capability for GPU 0".to_string())?;
    let cuda_architectures = mlxcel_core::cuda_arch::compiled_cuda_architectures();
    if cuda_architectures.is_empty() {
        return Err("this build records no compiled CUDA architecture list".to_string());
    }
    let root = repository_root();
    Ok(MlxIdentity {
        mlx_commit: mlxcel_core::autotune::store::mlx_commit(),
        overlay_sha256: overlay_digest(&root)?,
        q4_kernel_sha256: q4_kernel_digest(&root)?,
        q4_kernel_baseline: "post-#923 (qmm_sm80_r2 cp.async drain)",
        mlx_enable_tf32,
        cuda_architectures,
        device: mlxcel_core::gpu_device_name(0),
        compute_capability: format!("sm_{major}{minor}"),
        mlxcel_version: env!("CARGO_PKG_VERSION"),
    })
}
