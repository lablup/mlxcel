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

//! Entry-point plumbing for `--mla-latent-quant` (issue #3179).
//!
//! The option itself, its resolution rule and its warning texts live in
//! `mlxcel_core::mla::latent_option`. This module is the model-directory side:
//! it fixes the option for the process at startup, and turns a model path into
//! the lines an explicit value prints and the latent width the model stores.
//!
//! Used by: every binary's `main` ([`init_mla_latent_quant`]),
//! `cli::turbo_args` (the announce path), `server::cli_input` (startup
//! notices, the `--kv-bits` table) and the `generate` banner.

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::mla::{MlaLatentQuant, mla_latent_quant};

/// clap value parser for `--mla-latent-quant`: one of `auto`, `fp16`, `int8`,
/// `int4`; anything else is a parse error that lists them.
pub fn parse_mla_latent_quant_arg(value: &str) -> Result<MlaLatentQuant, String> {
    MlaLatentQuant::parse(value)
}

/// Fix the MLA latent option for this process: the CLI value when one was
/// given, else `MLXCEL_MLA_LATENT_QUANT`, else `auto`.
///
/// Called once by each binary right after argument parsing, before anything
/// resolves a KV cache mode, so the banner, the estimators and the caches all
/// read the same value. An unknown value in the environment variable is an
/// error here, as it is on the command line.
pub fn init_mla_latent_quant(cli: Option<MlaLatentQuant>) -> anyhow::Result<MlaLatentQuant> {
    mlxcel_core::mla::init_mla_latent_quant(cli.map(MlaLatentQuant::as_str))
        .map_err(|err| anyhow::anyhow!("{err}"))
}

/// The `model_type` the KV rules key on and whether absorbed MLA decode is on,
/// or `None` when the model directory has no readable `config.json`.
fn rules_model_type(model_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&content).ok()?;
    super::turbo_args::kv_rules_model_type(&config)
}

/// The lines an explicit `--mla-latent-quant` prints for the model at
/// `model_path` (`mlxcel_core::mla::latent_option_notices`): the option being
/// ignored on a model with no MLA latent cache, a disagreement with the
/// latent precision `requested` implies, and the experimental notice with its
/// measurements. Empty under `auto` and for a model whose `config.json`
/// cannot be read.
///
/// `requested` is the KV cache mode the operator's flags select before family
/// resolution.
#[must_use]
pub fn mla_latent_option_notices(requested: KVCacheMode, model_path: &Path) -> Vec<String> {
    let option = mla_latent_quant();
    if !option.is_explicit() {
        return Vec::new();
    }
    let Some(model_type) = rules_model_type(model_path) else {
        return Vec::new();
    };
    mlxcel_core::mla::latent_option_notices(
        &model_type,
        requested,
        mlxcel_core::mla::absorbed_enabled(),
        option,
    )
}

/// Whether `line` has not been announced by this process yet (and record it).
fn first_announcement(line: &str) -> bool {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .map(|mut seen| seen.insert(line.to_string()))
        .unwrap_or(false)
}

/// Print [`mla_latent_option_notices`] on stderr, each line at most once per
/// process however many entry points resolve the mode. stderr only, as the
/// substitution warning beside it is: the CLI's `tracing` subscriber writes to
/// stderr too, so logging the text there as well would print it twice.
///
/// Used by: `cli::turbo_args::resolve_and_announce_kv_cache_mode` (the CLI
/// generation entry points) and `mlxcel inspect`. The server carries the same
/// lines as startup notices instead.
pub fn announce_mla_latent_option(requested: KVCacheMode, model_path: &Path) {
    for line in mla_latent_option_notices(requested, model_path) {
        if first_announcement(&line) {
            eprintln!("{line}");
        }
    }
}

/// Whether an explicit `--mla-latent-quant` decides the KV caches of the
/// model at `model_path`: the option is not `auto` and the model holds an MLA
/// latent (a latent-pair family, or DeepSeek V2 while it absorbs), which is
/// `mlxcel_core::mla::option_decides_latent` on the directory's `model_type`.
///
/// Every KV cache of such a model is a latent cache, so the `--kv-bits` table
/// has nothing left to decide there.
///
/// Used by: `server::cli_input::disable_batch_kv_quant_for_family`.
#[must_use]
pub fn option_decides_model(model_path: &Path) -> bool {
    let option = mla_latent_quant();
    option.is_explicit()
        && rules_model_type(model_path).is_some_and(|model_type| {
            mlxcel_core::mla::option_decides_latent(
                &model_type,
                mlxcel_core::mla::absorbed_enabled(),
                option,
            )
        })
}

/// The width a cache built in `mode` for the model at `model_path` packs its
/// MLA latent at (8, or 4 under `int4`), or `None` when it stores no
/// quantized latent.
///
/// Used by: the `generate` KV cache banner.
#[must_use]
pub fn model_latent_bits(mode: KVCacheMode, model_path: &Path) -> Option<i32> {
    super::turbo_args::model_stores_quantized_latent(mode, model_path)
        .then(|| mlxcel_core::mla::latent_cache_bits(mode))
        .flatten()
}

/// The `KV dtype` a memory estimate reports for a cache built in `mode` for
/// the model at `model_path`: `fp16`, `int8`, or, when the option packs the
/// model's MLA latent at another width inside the `int8` storage, that width
/// and the flag that chose it (`int4 (MLA latent, --mla-latent-quant int4)`).
///
/// Used by: `execution::memory_estimate::format_estimate` (`mlxcel inspect`
/// and the `--estimate-memory` preflights).
#[must_use]
pub fn kv_dtype_label(mode: KVCacheMode, model_path: &Path) -> String {
    if mode != KVCacheMode::Int8 {
        return "fp16".to_string();
    }
    match model_latent_bits(mode, model_path) {
        Some(bits) if bits != 8 => format!(
            "int{bits} (MLA latent, {} {})",
            mlxcel_core::mla::MLA_LATENT_QUANT_FLAG,
            mla_latent_quant()
        ),
        _ => "int8".to_string(),
    }
}

/// The `generate` banner's description of an `int8` KV cache that stores an
/// MLA latent at `bits`. The default 8-bit wording is unchanged; a width the
/// option forced names the option.
#[must_use]
pub fn latent_codec_label(bits: i32) -> String {
    let option = mla_latent_quant();
    if option.is_explicit() {
        format!(
            "affine group-64 {bits}-bit MLA latent, fp16 rope keys; {} {option}",
            mlxcel_core::mla::MLA_LATENT_QUANT_FLAG
        )
    } else {
        "affine group-64 MLA latent, fp16 rope keys".to_string()
    }
}

#[cfg(test)]
#[path = "mla_latent_option_tests.rs"]
mod tests;
