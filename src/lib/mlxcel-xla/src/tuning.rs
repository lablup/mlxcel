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
/// lower poorly on gfx1151 (issues #2873, #2921, #2956).
///
/// The compiler's default tile for a rank-1 vecmat is `thread = [8, 0]`,
/// `workgroup = [256, 0]` for every width at or above 2048, so the launch
/// carries N/256 workgroups of one warp each: 8 workgroups for the 2048-wide
/// q and o projections and 32 for gate/up on llama-3.2-1b, 14 for the
/// 3584-wide projections of a model with hidden 3584. That is less than one
/// wave on a 20-WGP device, and the projections run at about 42 GB/s where the
/// 128256-wide LM head reaches 228 GB/s (issue #2873). The spec gives every
/// single-row vecmat with `128 <= N <= 8192` one output per thread and
/// `reduction = [0, 32]`, which launches N/32 warps and measures llama's q/o at
/// about 219 GB/s and its K = 8192 down projection at about 230. The tile does
/// not depend on K (every output still walks the reduction serially; only the
/// output count sets the occupancy), so the same match covers the down
/// projections of larger models. The LM head and other wide shapes keep the
/// config they had before: the compiler default unless N >= 8193 and
/// K >= 4096, where the previous `thread = [8, 0]` config still applies. Their
/// activation re-read grows with N, and they already launch enough warps (see
/// the asset for the exact predicates).
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

    /// issue #2873: the assertions above are textual, and the failure mode this
    /// asset can produce silently is a matcher that still parses but no longer
    /// matches: the graph compiles on the compiler default and only the
    /// throughput changes. This compiles the single-sequence decode graph the
    /// `hip` backend emits and reads the selected lowering configs back out of
    /// the executable dump, so the narrow tile is observed on the projections it
    /// was measured for and a wide output is observed keeping its own config.
    ///
    /// Ignored by default because it needs the version-matched ROCm-capable
    /// `iree-compile`. Run it after an edit to the asset or the emitter:
    /// `MLXCEL_XLA_IREE_COMPILE=<iree-compile> cargo test -p mlxcel-xla --lib
    /// tuning::tests::the_spec_reaches -- --ignored`.
    #[test]
    #[ignore = "requires MLXCEL_XLA_IREE_COMPILE pointing at an iree-compile with the ROCm target"]
    fn the_spec_reaches_the_shapes_the_decode_graph_emits() {
        use crate::emitter::{
            Config, Precision, ProjectionLayout, emit_decode_with_native,
            target_flags_with_rocm_target,
        };
        use std::process::Command;

        let compiler =
            std::env::var("MLXCEL_XLA_IREE_COMPILE").expect("set MLXCEL_XLA_IREE_COMPILE");
        let dir =
            std::env::temp_dir().join(format!("mlxcel-rocm-tuning-apply-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create the scratch directory");
        let spec = rocm_tuning_spec_flag_in(&dir).expect("write the spec");

        // The graph the `hip` engine compiles for a single-sequence decode: the
        // `InOut` layout with f16-resident projections. The other layouts and
        // precisions emit rank-2 contractions that this spec deliberately leaves
        // alone, so they cannot observe the matcher.
        let mut config = Config::llama_3_2_1b();
        config.projection_layout = ProjectionLayout::InOut;
        let graph = emit_decode_with_native(&config, false, Precision::F16, None);
        let input = dir.join("decode.mlir");
        std::fs::write(&input, &graph).expect("write the decode graph");
        let dumped = dir.join("configurations");
        std::fs::create_dir_all(&dumped).expect("create the dump directory");

        // Cross-compiling is enough to observe the selected configs, so the
        // matcher test pins the measured architecture unless the caller names
        // another one.
        let rocm_target =
            std::env::var("MLXCEL_XLA_ROCM_TARGET").unwrap_or_else(|_| "gfx1151".to_string());
        let mut cmd = Command::new(&compiler);
        cmd.arg("--iree-input-type=stablehlo");
        for flag in
            target_flags_with_rocm_target("hip", Some(&rocm_target)).expect("hip target flags")
        {
            cmd.arg(flag);
        }
        let out = cmd
            .arg(spec)
            .arg("--iree-dispatch-creation-enable-aggressive-fusion")
            .arg(format!(
                "--iree-hal-dump-executable-configurations-to={}",
                dumped.display()
            ))
            .arg(&input)
            .arg("-o")
            .arg(dir.join("decode.vmfb"))
            .output()
            .expect("run iree-compile");
        assert!(
            out.status.success(),
            "iree-compile failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );

        let mut seen: Vec<(String, String)> = Vec::new();
        for entry in std::fs::read_dir(&dumped).expect("read the dump directory") {
            let text = std::fs::read_to_string(entry.expect("dump entry").path())
                .expect("read a dumped configuration");
            let export = text
                .split("hal.executable.export public @")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .expect("an export name");
            if !export.contains("_vecmat_") {
                continue;
            }
            let start = text
                .find("lowering_config = #iree_gpu.lowering_config")
                .expect("a lowering config on the vecmat dispatch");
            let rest = &text[start..];
            let config = &rest[..rest.find("}>").expect("a closed lowering config") + 2];
            seen.push((export.to_string(), config.to_string()));
        }
        assert!(
            !seen.is_empty(),
            "the decode graph carried no vecmat dispatch to observe"
        );

        let narrow = "lowering_config = #iree_gpu.lowering_config<{reduction = [0, 32], thread = [1, 0], workgroup = [32, 0]}";
        let (mut narrow_hits, mut others) = (0usize, 0usize);
        for (export, config) in &seen {
            // The dispatch name is `vecmat_<N>x<K>...`, N first (the weight's
            // output axis), which is the axis the narrow matcher bounds.
            let n: usize = export
                .split("_vecmat_")
                .nth(1)
                .and_then(|rest| rest.split('x').next())
                .expect("an output width in the dispatch name")
                .parse()
                .expect("a numeric output width");
            if (128..=8192).contains(&n) {
                assert!(
                    config.starts_with(narrow),
                    "{export} (N = {n}) lost the narrow tile: {config}"
                );
                narrow_hits += 1;
            } else {
                assert!(
                    !config.starts_with(narrow),
                    "{export} (N = {n}) picked up the narrow tile the N bound excludes: {config}"
                );
                others += 1;
            }
        }
        assert!(
            narrow_hits > 0,
            "no narrow vecmat was observed, so the matcher did not apply at all"
        );
        assert!(
            others > 0,
            "the wide LM head was not in the dump, so the N bound is unobserved"
        );
        // Reached only when every observation held; a failed run keeps the dump
        // and the emitted graph in the scratch directory for inspection.
        let _ = std::fs::remove_dir_all(&dir);
    }

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

    /// issue #2873: the narrow single-row vecmats (q, k, v, o and the down
    /// projection, at any hidden size) get one output per thread, so the launch
    /// carries N/32 warps instead of N/256 and fills the device. The matcher
    /// must stay wired from the entrypoint, must keep the batch predicate (the
    /// ragged graph's projections are multi-row), and must stay bounded at
    /// N <= 8192 so the LM head keeps its wider thread tile. The K axis is not
    /// bounded: the tile is the same for every reduction length, and bounding
    /// it at the 1b model's hidden size would leave q/o on a 7b checkpoint
    /// (`K = 3584`) on the compiler default.
    ///
    /// The wide fallback's two predicates are pinned as well: it must keep its
    /// K >= 4096 gate (without it the wide tile would reach every LM head,
    /// including llama's K = 2048 one that is deliberately left on the compiler
    /// default) and start exactly where the narrow matcher stops, N >= 8193, so
    /// the two predicates are disjoint. Because they are disjoint their order
    /// in `transform.foreach_match` cannot change the result, so it is not
    /// asserted.
    #[test]
    fn tuning_spec_matches_the_narrow_single_row_vecmats() {
        let entrypoint = ROCM_BIGK_VECMAT_TUNING
            .split("@__kernel_config")
            .nth(1)
            .expect("the tuning-spec entrypoint");
        assert!(
            entrypoint.contains("@match_single_row_vecmat -> @apply_op_config")
                && entrypoint.contains("@match_bigk_vecmat_wide -> @apply_op_config"),
            "every vecmat matcher must run from the tuning-spec entrypoint"
        );
        let body = ROCM_BIGK_VECMAT_TUNING
            .split("@match_single_row_vecmat")
            .nth(1)
            .expect("the single-row matcher")
            .split("transform.named_sequence")
            .next()
            .expect("the matcher body");
        assert!(
            body.contains("thread = [1, 0], workgroup = [32, 0]"),
            "the narrow matcher sets one output per thread"
        );
        assert!(
            body.contains("transform.iree.match.dims_equal %batch_dims, []"),
            "the narrow matcher keeps the ragged multi-row projections out"
        );
        assert!(
            body.contains("transform.iree.match.dim_bounds %rhs[1], umin = 128, umax = 8192"),
            "the narrow matcher bounds only the output width, so the LM head keeps its config"
        );
        assert!(
            !body.contains("match.dim_bounds %rhs[0]"),
            "the narrow matcher must not bound the reduction axis"
        );
        let wide = ROCM_BIGK_VECMAT_TUNING
            .split("@match_bigk_vecmat_wide")
            .nth(1)
            .expect("the wide matcher")
            .split("transform.named_sequence")
            .next()
            .expect("the wide matcher body");
        assert!(
            wide.contains("thread = [8, 0], workgroup = [256, 0]"),
            "the wide long-K fallback keeps the previous thread tile"
        );
        assert!(
            wide.contains("transform.iree.match.dims_equal %batch_dims, []"),
            "the wide matcher keeps the ragged multi-row projections out"
        );
        assert!(
            wide.contains("transform.iree.match.dim_bounds %rhs[0], umin = 4096, none"),
            "the wide fallback stays gated on the reduction axis, so the K = 2048 LM head keeps the compiler default"
        );
        assert!(
            wide.contains("transform.iree.match.dim_bounds %rhs[1], umin = 8193, none"),
            "the wide fallback starts where the narrow matcher stops"
        );
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
        assert!(ROCM_BIGK_VECMAT_TUNING.contains("@match_single_row_vecmat"));
        assert!(ROCM_BIGK_VECMAT_TUNING.contains("@match_gqa_score"));
        let entrypoint = ROCM_BIGK_VECMAT_TUNING
            .split("@__kernel_config")
            .nth(1)
            .expect("the tuning-spec entrypoint");
        assert!(
            entrypoint.contains("@match_single_row_vecmat -> @apply_op_config")
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
