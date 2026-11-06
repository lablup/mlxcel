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

//! Lowering tuning specs passed to `iree-compile`.
//!
//! The compiler ships a default tuning module for the target, and a user spec
//! is applied first (issue #2873). The specs are plain MLIR assets so they can
//! be inspected and reviewed like any other compiler input.

use std::path::Path;

/// ROCm lowering-tuning spec for single-row contractions with a long K
/// (issue #2873).
///
/// The compiler's default tile for a `vecmat` is `reduction = [0, 4]`. For the
/// down projection the contraction runs over the intermediate size (8192 on
/// llama-3.2-1b), and that tile leaves the kernel at about 50 GB/s, roughly
/// three times slower than the same shape reaches at `reduction = [0, 32]`.
/// The wider tile is slower on short-K vecmats, so the spec matches only
/// contractions whose reduction axis is at least 4096 elements; every other
/// shape keeps the compiler's default.
pub(crate) const ROCM_BIGK_VECMAT_TUNING: &str =
    include_str!("../assets/rocm/bigk_vecmat_tuning.mlir");

/// Write [`ROCM_BIGK_VECMAT_TUNING`] under a content-addressed name in `dir` and
/// return the `--iree-codegen-tuning-spec-path` flag that selects it.
///
/// The content hash goes into both the file name and the flag, so changing the
/// spec changes the vmfb cache key instead of silently reusing a stale build.
pub(crate) fn rocm_tuning_spec_flag_in(dir: &Path) -> Result<String, String> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ROCM_BIGK_VECMAT_TUNING.hash(&mut h);
    let path = dir.join(format!("rocm-bigk-vecmat-{:016x}.mlir", h.finish()));
    if !path.exists() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        std::fs::write(&path, ROCM_BIGK_VECMAT_TUNING)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(format!(
        "--iree-codegen-tuning-spec-path={}",
        path.display()
    ))
}

/// The tuning-spec flag for the model compile cache
/// (`<temp>/mlxcel-xla-vmfb`), which every model graph shares.
pub(crate) fn rocm_tuning_spec_flag() -> Result<String, String> {
    let dir = std::env::temp_dir().join("mlxcel-xla-vmfb");
    rocm_tuning_spec_flag_in(&dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning_spec_flag_is_content_addressed() {
        let dir =
            std::env::temp_dir().join(format!("mlxcel-rocm-tuning-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let flag = rocm_tuning_spec_flag_in(&dir).expect("write the spec");
        let path = std::path::PathBuf::from(
            flag.strip_prefix("--iree-codegen-tuning-spec-path=")
                .expect("the flag names the spec path"),
        );
        assert!(path.is_file(), "{}", path.display());
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            name.starts_with("rocm-bigk-vecmat-") && name.ends_with(".mlir"),
            "{name}"
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            ROCM_BIGK_VECMAT_TUNING
        );
        assert_eq!(
            flag,
            rocm_tuning_spec_flag_in(&dir).expect("reuse the spec"),
            "the same contents keep the same cache name"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
