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

/// ROCm lowering-tuning spec for the contraction shapes the compiler defaults
/// lower poorly on gfx1151 (issues #2873, #2921).
///
/// The compiler's default tile for a `vecmat` is `reduction = [0, 4]`. For the
/// down projection the contraction runs over the intermediate size (8192 on
/// llama-3.2-1b), and that tile leaves the kernel at about 50 GB/s, roughly
/// three times slower than the same shape reaches at `reduction = [0, 32]`.
/// The wider tile is slower on short-K vecmats, so the spec matches only
/// contractions whose reduction axis is at least 4096 elements.
///
/// The second match is the grouped-query decode score contraction, a batched
/// `[n_kv, group, head_dim]` by `[n_kv, head_dim, keys]` f16 matmul with an f32
/// result. The default config for that shape runs it at about 74 us per
/// dispatch on llama-3.2-1b, where the same MAC count in the context matmul
/// took about 6.5 us under its own default config; the spec's tile drops the
/// score dispatch to about 1.5 us (issue #2921). It matches only
/// `batch = n_kv = 8`, `group = 4` and `head_dim = 64` with at least 128 keys,
/// the measured shape of the single-sequence decode graph. The batch predicate
/// is load-bearing: the ragged and uniform-B `dot_general`s collapse to the
/// same rank-3 contraction with the same maps, but with `bsz * n_kv` as the
/// batch, so without it a multi-row ragged graph would silently pick up a
/// config that was measured only for one row.
///
/// The third match is the companion context contraction, the same batched
/// layout with the operands' roles swapped: `[n_kv, group, keys]` by
/// `[n_kv, keys, head_dim]`. It reduces over the key axis and writes 64
/// outputs, and the compiler default leaves it at about 6.4 us per dispatch at
/// capacity 256, where the score dispatch already runs at about 1.5 us. The
/// spec's tile brings it to about 1.8 us at capacity 256, 2.8 us at 512 and
/// 5.9 us at 1024 (issue #2956). It is pinned the same way: batch 8, group 4
/// and head_dim 64 exactly, at least 128 keys. The M predicate also keeps
/// prefill out, since prefill's M axis carries the prompt length. Every other
/// shape keeps the compiler default.
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

    /// issues #2921, #2956: the spec must keep every contraction match wired
    /// into the entrypoint. The grouped-query score match is intentionally
    /// narrow (batch 8, group 4, head_dim 64, at least 128 keys) because only
    /// that shape was measured; widening it would apply an unmeasured config to
    /// other models, and dropping the batch predicate would apply it to the
    /// ragged and uniform-B score matmuls, whose collapsed batch is
    /// `bsz * n_kv`.
    #[test]
    fn tuning_spec_matches_the_gqa_decode_score_shape() {
        assert!(ROCM_BIGK_VECMAT_TUNING.contains("@match_bigk_vecmat"));
        assert!(ROCM_BIGK_VECMAT_TUNING.contains("@match_gqa_score"));
        let entrypoint = ROCM_BIGK_VECMAT_TUNING
            .split("@__kernel_config")
            .nth(1)
            .expect("the tuning-spec entrypoint");
        assert!(
            entrypoint.contains("@match_bigk_vecmat -> @apply_op_config")
                && entrypoint.contains("@match_gqa_score -> @apply_op_config")
                && entrypoint.contains("@match_gqa_context -> @apply_op_config"),
            "every matcher must run from the tuning-spec entrypoint"
        );
        let score = ROCM_BIGK_VECMAT_TUNING
            .split("@match_gqa_score")
            .nth(1)
            .expect("the score matcher");
        assert!(
            score.contains("transform.iree.match.dims_equal %batch_dims, [8]"),
            "the score matcher targets the single-sequence batch of 8 KV heads"
        );
        assert!(
            score.contains("transform.iree.match.dims_equal %m_dims, [4]"),
            "the score matcher targets group = 4"
        );
        assert!(
            score.contains("transform.iree.match.dims_equal %k_dims, [64]"),
            "the score matcher targets head_dim = 64"
        );
        assert!(
            score.contains("transform.iree.match.dim_bounds %rhs[2], umin = 128, none"),
            "the score matcher targets at least 128 keys"
        );
        assert!(
            score.contains("workgroup_size = [32, 4, 1]"),
            "the score matcher sets the 128-thread workgroup measured in #2921"
        );
    }

    /// issue #2956: the context contraction gets its own narrow matcher. The
    /// reduction is the key axis and the output is 64 wide, the mirror image of
    /// the score matcher, so the predicates are batch 8, group 4, head_dim 64
    /// and at least 128 keys. Prefill is kept out by the M predicate (its M
    /// axis carries the prompt length) and the ragged and uniform-B graphs by
    /// the batch predicate.
    #[test]
    fn tuning_spec_matches_the_gqa_decode_context_shape() {
        let context = ROCM_BIGK_VECMAT_TUNING
            .split("@match_gqa_context")
            .nth(1)
            .expect("the context matcher")
            .split("@__kernel_config")
            .next()
            .expect("the context matcher body");
        assert!(
            context.contains("transform.iree.match.dims_equal %batch_dims, [8]"),
            "the context matcher targets the single-sequence batch of 8 KV heads"
        );
        assert!(
            context.contains("transform.iree.match.dims_equal %m_dims, [4]"),
            "the context matcher targets group = 4"
        );
        assert!(
            context.contains("transform.iree.match.dims_equal %n_dims, [64]"),
            "the context matcher targets head_dim = 64 as the output width"
        );
        assert!(
            context.contains("transform.iree.match.dim_bounds %rhs[1], umin = 128, none"),
            "the context matcher targets at least 128 keys on the reduction axis"
        );
        assert!(
            context.contains(
                "reduction = [0, 0, 0, 64], thread = [1, 1, 2, 0], workgroup = [1, 4, 64, 1]"
            ),
            "the context matcher sets the tile measured in #2956"
        );
        assert!(
            context.contains("workgroup_size = [32, 4, 1]"),
            "the context matcher sets the 128-thread workgroup measured in #2956"
        );
    }
}
