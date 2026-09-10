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

//! `mlxcel split-mtp`: extract the GLM-4.7-Flash next-token-prediction block
//! out of a raw `zai-org/GLM-4.7-Flash` checkpoint into a standalone
//! `glm4_moe_lite_mtp` drafter directory (issue #1326).
//!
//! The transform itself is [`mlxcel_surgery::split_mtp_dir`]; this module is
//! the argument surface and the report printer. The community 4-bit
//! conversions drop `model.layers.47.*`, so the source must be the raw
//! checkpoint; the tool reads only the shards the safetensors index names
//! for that layer.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::Args;
use mlxcel::initialize_runtime_checked;
use mlxcel_surgery::{SplitMtpOptions, refuse_output_is_source, split_mtp_dir};

/// Arguments for `mlxcel split-mtp`.
#[derive(Args, Debug)]
pub(crate) struct SplitMtpArgs {
    /// Raw GLM-4.7-Flash checkpoint directory (`config.json` plus the
    /// safetensors shards and index). Must still hold the
    /// `model.layers.{num_hidden_layers}.*` tensors; community 4-bit
    /// conversions do not.
    #[arg(short, long, value_name = "PATH")]
    pub(crate) model: PathBuf,

    /// Output drafter directory. Created if missing; pass `--draft-model`
    /// this path to `mlxcel generate` or `--model-draft` to `mlxcel-server`.
    #[arg(short, long, value_name = "PATH")]
    pub(crate) output: PathBuf,

    /// Verify block size recorded in the drafter config, bonus token
    /// included. Defaults to `num_nextn_predict_layers + 1` (2 on
    /// GLM-4.7-Flash). Larger values draft past the trained depth.
    #[arg(long, value_name = "N")]
    pub(crate) block_size: Option<usize>,

    /// Affine-quantize every quantizable projection to this bit width
    /// (2, 3, 4, 5, 6 or 8). Omit to keep the drafter bf16. The router
    /// (`mlp.gate.weight`), the norms and the selection bias are never
    /// quantized.
    #[arg(long, value_name = "BITS")]
    pub(crate) q_bits: Option<i32>,

    /// Quantization group size. Only read with `--q-bits`.
    #[arg(long, default_value_t = 64, value_name = "N")]
    pub(crate) q_group_size: i32,

    /// Overwrite an output directory that already holds a checkpoint.
    #[arg(long)]
    pub(crate) force: bool,
}

pub(crate) fn run_split_mtp(args: SplitMtpArgs) -> Result<()> {
    if !args.model.join("config.json").is_file() {
        return Err(anyhow!(
            "split-mtp: {} has no config.json; pass the raw checkpoint directory",
            args.model.display()
        ));
    }
    preflight_checks(&args.model, &args.output, args.q_bits, args.q_group_size)?;
    prepare_output_dir(&args.output, args.force)?;
    if args.q_bits.is_none() && args.q_group_size != 64 {
        eprintln!("split-mtp: --q-group-size has no effect without --q-bits");
    }

    let _runtime = initialize_runtime_checked()?;

    let opts = SplitMtpOptions {
        block_size: args.block_size,
        q_bits: args.q_bits,
        q_group_size: args.q_group_size,
    };
    println!(
        "Splitting the MTP block out of {} into {}{}",
        args.model.display(),
        args.output.display(),
        match args.q_bits {
            Some(bits) => format!(" ({bits}-bit affine, group size {})", args.q_group_size),
            None => " (bf16)".to_string(),
        }
    );
    let report = split_mtp_dir(&args.model, &args.output, &opts)
        .with_context(|| format!("split-mtp: {}", args.model.display()))?;

    println!(
        "Wrote {} tensor(s), {:.2} GB, from model.layers.{} ({} quantized)",
        report.tensors,
        report.bytes_written as f64 / 1e9,
        report.source_layer,
        report.quantized_tensors,
    );
    if !report.copied_files.is_empty() {
        println!("Copied: {}", report.copied_files.join(", "));
    }
    for name in &report.missing_files {
        eprintln!(
            "warning: {name} not found in {}; copy the tokenizer beside the drafter by hand",
            args.model.display()
        );
    }
    println!(
        "Drafter written to {}. Use it with:\n  mlxcel generate -m <glm4_moe_lite target> --draft-model {} -p ...",
        report.output_dir.display(),
        report.output_dir.display()
    );
    Ok(())
}

/// Whether `name` is a safetensors weight file this tool must never mix into
/// the drafter: a numbered shard (`model-00001-of-00003.safetensors`) or any
/// other checkpoint's file (`consolidated.safetensors`). `model.safetensors`,
/// the drafter's own single-file output, is excluded on purpose: the
/// loader's `glob_safetensors` (`weights.rs`) reads every `*.safetensors` in
/// a directory once no index is present, so any other safetensors file left
/// beside it would silently join the drafter as one checkpoint (issue #1778
/// review).
///
/// Used by: [`existing_checkpoint_marker`], [`find_foreign_safetensors_file`].
fn is_foreign_safetensors_file(name: &str) -> bool {
    name != "model.safetensors" && name.ends_with(".safetensors")
}

/// The first foreign safetensors file in `dir` (see
/// [`is_foreign_safetensors_file`]), by filename order, `Ok(None)` when the
/// directory holds none, or `Err` when the directory could not be listed.
///
/// Fails closed on a listing error rather than reporting "no weight files
/// found": [`prepare_output_dir`] would otherwise proceed to delete a stale
/// index (or, previously, nothing at all) believing the directory was clear
/// when it was merely unreadable (issue #1778 review).
///
/// Used by: [`prepare_output_dir`].
fn find_foreign_safetensors_file(dir: &std::path::Path) -> std::io::Result<Option<String>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if is_foreign_safetensors_file(&name) {
            found.push(name);
        }
    }
    found.sort();
    Ok(found.into_iter().next())
}

/// Non-mutating checks that must all pass before [`prepare_output_dir`] (or
/// anything else) touches the filesystem: an unsupported `--q-bits` /
/// `--q-group-size`, or an `--output` that resolves to `--model`, must be
/// refused before any mutation rather than after one.
///
/// With the old ordering, `prepare_output_dir` ran first: `-o` equal to a
/// sharded `-m` under `--force` told the user to remove their own source
/// directory by hand (the "shard" it named was the source's), and
/// `-m X -o X --force` on a single-file source deleted the source's own
/// stale index before the same-path refusal in `split_mtp_dir` ever ran
/// (issue #1778 review).
///
/// Used by: [`run_split_mtp`].
fn preflight_checks(
    model: &std::path::Path,
    output: &std::path::Path,
    q_bits: Option<i32>,
    q_group_size: i32,
) -> Result<()> {
    if let Some(bits) = q_bits {
        mlxcel_core::layers::validate_affine_quantization_bits(bits)
            .map_err(|e| anyhow!("split-mtp: --q-bits: {e}"))?;
        mlxcel_core::layers::validate_affine_quantization_group_size(q_group_size)
            .map_err(|e| anyhow!("split-mtp: --q-group-size: {e}"))?;
    }
    refuse_output_is_source(model, output)?;
    Ok(())
}

/// Guard `args.output` before [`split_mtp_dir`] writes into it.
///
/// No existing marker: `Ok(())`, nothing to guard. A marker with `force`
/// unset: today's `already holds a checkpoint (...); pass --force to
/// overwrite` refusal, unchanged. A marker with `force` set and at least one
/// foreign safetensors file present ([`is_foreign_safetensors_file`]): a new
/// refusal naming one such file, because `--force` overwrites the drafter's
/// own `model.safetensors` and `config.json` but does not promise to delete
/// weight files, and the file it would otherwise leave orphaned may be the
/// only surviving copy (issue #1763). A marker with `force` set and no
/// foreign files: remove a stale `model.safetensors.index.json` as before
/// and return `Ok(())`, since `collect_shard_paths` prefers an index over a
/// bare `model.safetensors` and a stale one left beside the drafter's
/// single-file output would send the loader to the victim's shards.
///
/// Used by: [`run_split_mtp`] (after [`preflight_checks`] has already
/// refused anything that does not require touching the filesystem).
fn prepare_output_dir(dir: &std::path::Path, force: bool) -> Result<()> {
    let Some(existing) = existing_checkpoint_marker(dir) else {
        return Ok(());
    };
    if !force {
        return Err(anyhow!(
            "split-mtp: {} already holds a checkpoint ({existing}); pass --force to overwrite",
            dir.display()
        ));
    }
    let foreign = find_foreign_safetensors_file(dir).with_context(|| {
        format!(
            "split-mtp: failed to check {} for weight files to protect under --force",
            dir.display()
        )
    })?;
    if let Some(file) = foreign {
        return Err(anyhow!(
            "split-mtp: {} holds weight file {file}; --force overwrites the drafter's own \
             model.safetensors and config.json but does not delete weight files, remove the \
             directory by hand and retry",
            dir.display()
        ));
    }
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        std::fs::remove_file(&index)
            .with_context(|| format!("split-mtp: failed to remove stale {}", index.display()))?;
    }
    Ok(())
}

/// Name of the first artifact that makes `dir` look like an existing
/// checkpoint, or `None` when writing there would clobber nothing.
///
/// `split_mtp_dir` writes `model.safetensors` and `config.json` and copies
/// five tokenizer files over whatever is present, so the guard has to screen
/// on more than the single-file weight name: a sharded checkpoint has an
/// index and shards instead, and would lose its config and tokenizer while
/// keeping shards nothing can load.
///
/// Used by: [`run_split_mtp`] (through [`prepare_output_dir`]).
fn existing_checkpoint_marker(dir: &std::path::Path) -> Option<String> {
    for name in [
        "model.safetensors",
        "model.safetensors.index.json",
        "config.json",
    ] {
        if dir.join(name).exists() {
            return Some(name.to_string());
        }
    }
    let shard = std::fs::read_dir(dir).ok()?.flatten().find(|entry| {
        entry
            .file_name()
            .to_str()
            .is_some_and(is_foreign_safetensors_file)
    })?;
    Some(shard.file_name().to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::{
        existing_checkpoint_marker, find_foreign_safetensors_file, preflight_checks,
        prepare_output_dir,
    };

    /// The guard this replaces probed only `model.safetensors`, so a sharded
    /// checkpoint passed it and lost its `config.json` and tokenizer files to
    /// the drafter's (issue #1326). `consolidated.safetensors` covers the
    /// broader case (issue #1778 review): the loader's `glob_safetensors`
    /// reads every `*.safetensors` file, not only ones named `model-*`.
    #[test]
    fn marker_names_every_shape_of_existing_checkpoint() {
        for name in [
            "model.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "model-00001-of-00003.safetensors",
            "consolidated.safetensors",
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(dir.path().join(name), b"x").expect("write");
            assert_eq!(
                existing_checkpoint_marker(dir.path()).as_deref(),
                Some(name),
                "{name} must be recognised as an existing checkpoint"
            );
        }
    }

    #[test]
    fn marker_is_none_for_an_empty_or_unrelated_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(existing_checkpoint_marker(dir.path()), None);
        std::fs::write(dir.path().join("README.md"), b"x").expect("write");
        assert_eq!(existing_checkpoint_marker(dir.path()), None);
        assert_eq!(
            existing_checkpoint_marker(&dir.path().join("missing")),
            None,
            "a directory that does not exist clobbers nothing"
        );
    }

    /// `--force` must refuse rather than delete when the output directory
    /// still holds a previously sharded checkpoint's weight shards: nothing
    /// promised by a flag named `--force` covers deleting weight files, and
    /// those shards may be the only surviving copy (issue #1763).
    #[test]
    fn force_refuses_a_previously_sharded_output_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shard = dir.path().join("model-00001-of-00003.safetensors");
        let index = dir.path().join("model.safetensors.index.json");
        let config = dir.path().join("config.json");
        std::fs::write(&shard, b"shard").expect("write shard");
        std::fs::write(&index, b"index").expect("write index");
        std::fs::write(&config, b"config").expect("write config");

        let err = prepare_output_dir(dir.path(), true).expect_err("must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("model-00001-of-00003.safetensors"),
            "must name the shard: {msg}"
        );

        assert!(shard.exists(), "the shard must not be deleted");
        assert!(index.exists(), "the index must not be deleted");
    }

    /// A stale index with no surviving shards is still cleared under
    /// `--force`, as before: there is nothing left for it to misdirect the
    /// loader toward.
    #[test]
    fn force_clears_a_stale_index_when_no_shards_remain() {
        let dir = tempfile::tempdir().expect("tempdir");
        let weights = dir.path().join("model.safetensors");
        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(&weights, b"weights").expect("write weights");
        std::fs::write(&index, b"index").expect("write index");

        prepare_output_dir(dir.path(), true).expect("must be accepted");

        assert!(!index.exists(), "the stale index must be removed");
        assert!(weights.exists(), "model.safetensors must be left alone");
    }

    /// `--force` must also refuse a directory holding a foreign safetensors
    /// file that is not a `model-*`-named shard: `consolidated.safetensors`
    /// passes the loader's `glob_safetensors` glob just as readily as a
    /// numbered shard would, and would be silently read alongside the
    /// drafter's own `model.safetensors` (issue #1778 review).
    #[test]
    fn force_refuses_a_directory_with_a_foreign_safetensors_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let foreign = dir.path().join("consolidated.safetensors");
        let config = dir.path().join("config.json");
        std::fs::write(&foreign, b"weights").expect("write foreign file");
        std::fs::write(&config, b"config").expect("write config");

        let err = prepare_output_dir(dir.path(), true).expect_err("must refuse");
        assert!(
            err.to_string().contains("consolidated.safetensors"),
            "must name the foreign file: {err}"
        );
        assert!(foreign.exists(), "the foreign file must not be deleted");
    }

    /// A directory that cannot be listed must refuse rather than report "no
    /// weight files found": the old `.ok()?` / `.flatten()` shape silently
    /// treated an unreadable directory as shard-free, after which
    /// `prepare_output_dir` would go on to delete a stale index believing
    /// nothing else was there (issue #1778 review).
    #[cfg(unix)]
    #[test]
    fn find_foreign_safetensors_file_fails_closed_on_an_unlistable_directory() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o000))
            .expect("chmod 000");

        let result = find_foreign_safetensors_file(dir.path());

        // Restore permissions before the tempdir's Drop tries to remove it.
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700))
            .expect("restore permissions");

        assert!(
            result.is_err(),
            "an unlistable directory must fail closed, not report no weight files"
        );
    }

    /// `preflight_checks` must refuse an `--output` that resolves to the
    /// same directory as `--model` before anything mutates: with the old
    /// ordering, `prepare_output_dir` ran first and either misdirected the
    /// user (telling them to remove their own source directory by hand) or,
    /// for a single-file source, deleted the source's own stale index before
    /// this refusal ever ran (issue #1778 review).
    #[test]
    fn preflight_refuses_an_output_that_equals_the_model_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), b"{}").expect("write");

        let err = preflight_checks(dir.path(), dir.path(), None, 64).expect_err("must refuse");
        assert!(
            err.to_string().contains("is the source checkpoint"),
            "{err}"
        );
    }
}
