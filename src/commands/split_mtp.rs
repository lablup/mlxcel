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
use mlxcel_surgery::{SplitMtpOptions, split_mtp_dir};

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
    // A sharded checkpoint carries no `model.safetensors`, so probing for
    // that one filename let `--output` point at one and silently replace its
    // `config.json` and tokenizer files while orphaning the shards. Screen on
    // whatever makes the directory a checkpoint instead (issue #1326).
    if let Some(existing) = existing_checkpoint_marker(&args.output) {
        if !args.force {
            return Err(anyhow!(
                "split-mtp: {} already holds a checkpoint ({existing}); pass --force to overwrite",
                args.output.display()
            ));
        }
        // `collect_shard_paths` prefers an index over a bare
        // `model.safetensors`, so a stale index left beside the drafter's
        // single-file output would send the loader to the victim's shards.
        let index = args.output.join("model.safetensors.index.json");
        if index.exists() {
            std::fs::remove_file(&index).with_context(|| {
                format!("split-mtp: failed to remove stale {}", index.display())
            })?;
        }
    }
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

/// Name of the first artifact that makes `dir` look like an existing
/// checkpoint, or `None` when writing there would clobber nothing.
///
/// `split_mtp_dir` writes `model.safetensors` and `config.json` and copies
/// five tokenizer files over whatever is present, so the guard has to screen
/// on more than the single-file weight name: a sharded checkpoint has an
/// index and shards instead, and would lose its config and tokenizer while
/// keeping shards nothing can load.
///
/// Used by: [`run_split_mtp`].
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
            .is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
    })?;
    Some(shard.file_name().to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::existing_checkpoint_marker;

    /// The guard this replaces probed only `model.safetensors`, so a sharded
    /// checkpoint passed it and lost its `config.json` and tokenizer files to
    /// the drafter's (issue #1326).
    #[test]
    fn marker_names_every_shape_of_existing_checkpoint() {
        for name in [
            "model.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "model-00001-of-00003.safetensors",
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
}
