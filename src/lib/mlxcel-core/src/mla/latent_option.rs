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

//! The experimental MLA latent precision option (issue #3179).
//!
//! `--mla-latent-quant <auto|fp16|int8|int4>` (env
//! [`MLA_LATENT_QUANT_ENV`]; the CLI wins) lets an operator pick the
//! precision every MLA latent cache of the loaded model is stored at,
//! including on the families the default quality gate keeps at fp16. `auto`
//! is the default and reproduces the gate exactly.
//!
//! [`effective_latent_bits`] is the one place a family, a requested
//! [`KVCacheMode`] and the option become a latent bit width. Everything else
//! derives from it:
//!
//! * the mode resolver (`cache::turbo::allowlist`) builds a model's latent
//!   caches in [`KVCacheMode::Int8`] storage when it returns a width and in
//!   fp16 otherwise, so no new `KVCacheMode` variant exists;
//! * [`latent_cache_bits`] is the width such a cache packs its latent at
//!   (8, or 4 under `int4`), read by the latent view and by sizing;
//! * [`crate::mla::stores_quantized_latent_pair`] and
//!   [`crate::mla::latent_pair_supports_mode`] answer through it.
//!
//! The option is process state: [`init_mla_latent_quant`] sets it once at
//! startup and [`mla_latent_quant`] reads it. A process that never sets it
//! (a library user, an example binary) reads the environment variable on the
//! first query, in the same function.

use std::cell::Cell;
use std::sync::OnceLock;

use crate::cache::KVCacheMode;
use crate::mla::cache::{caches_mla_latent_pair, stores_latent_through_view};
use crate::mla::latent_keys::caches_dsa_indexer_key;
use crate::mla::latent_pair::QUANTIZED_LATENT_FAMILIES;

/// Environment variable carrying the option when the CLI flag is absent.
pub const MLA_LATENT_QUANT_ENV: &str = "MLXCEL_MLA_LATENT_QUANT";

/// The CLI flag, for messages.
pub const MLA_LATENT_QUANT_FLAG: &str = "--mla-latent-quant";

/// Precision an operator asked the MLA latent caches to be stored at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MlaLatentQuant {
    /// Follow the default quality gate: the listed families take the affine
    /// 8-bit latent under `int8`, the rest keep fp16.
    #[default]
    Auto,
    /// Keep the latent fp16 whatever the KV cache mode asks for.
    Fp16,
    /// Store the latent as affine group-64 8-bit on every MLA family.
    Int8,
    /// Store the latent as affine group-64 4-bit on every MLA family.
    Int4,
}

impl MlaLatentQuant {
    /// The accepted spellings, in help order.
    pub const VALUES: [&'static str; 4] = ["auto", "fp16", "int8", "int4"];

    /// Parse one of [`Self::VALUES`] (trimmed, case-insensitive). The error
    /// lists them.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "fp16" => Ok(Self::Fp16),
            "int8" => Ok(Self::Int8),
            "int4" => Ok(Self::Int4),
            other => Err(format!(
                "invalid MLA latent quantization '{other}': expected one of {}",
                Self::VALUES.join(", ")
            )),
        }
    }

    /// The spelling [`Self::parse`] accepts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }

    /// Whether the operator named a precision (anything but `auto`).
    #[must_use]
    pub const fn is_explicit(self) -> bool {
        !matches!(self, Self::Auto)
    }
}

impl std::fmt::Display for MlaLatentQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for MlaLatentQuant {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

static OPTION: OnceLock<MlaLatentQuant> = OnceLock::new();

thread_local! {
    /// Per-thread override installed by [`with_mla_latent_quant`].
    static OVERRIDE: Cell<Option<MlaLatentQuant>> = const { Cell::new(None) };
}

/// The option from the CLI value when one was given, else from
/// [`MLA_LATENT_QUANT_ENV`], else `auto`. An unknown value in either place is
/// an error naming where it came from.
///
/// The only read of the environment variable.
pub fn resolve_mla_latent_quant(cli: Option<&str>) -> Result<MlaLatentQuant, String> {
    if let Some(value) = cli {
        return MlaLatentQuant::parse(value)
            .map_err(|err| format!("{MLA_LATENT_QUANT_FLAG}: {err}"));
    }
    match std::env::var(MLA_LATENT_QUANT_ENV) {
        Ok(value) if !value.trim().is_empty() => {
            MlaLatentQuant::parse(&value).map_err(|err| format!("{MLA_LATENT_QUANT_ENV}: {err}"))
        }
        _ => Ok(MlaLatentQuant::Auto),
    }
}

/// Fix the process-wide option. The first call wins; a later call with the
/// same value is a no-op and one with a different value returns the value
/// already in force.
pub fn set_mla_latent_quant(option: MlaLatentQuant) -> Result<(), MlaLatentQuant> {
    let current = *OPTION.get_or_init(|| option);
    if current == option {
        Ok(())
    } else {
        Err(current)
    }
}

/// Resolve the option ([`resolve_mla_latent_quant`]) and fix it for the
/// process ([`set_mla_latent_quant`]). Every binary calls this once, before
/// anything resolves a KV cache mode.
pub fn init_mla_latent_quant(cli: Option<&str>) -> Result<MlaLatentQuant, String> {
    let option = resolve_mla_latent_quant(cli)?;
    set_mla_latent_quant(option).map_err(|current| {
        format!(
            "{MLA_LATENT_QUANT_FLAG} {option} cannot replace the value already in force for this \
             process ({current})"
        )
    })?;
    Ok(option)
}

/// The option in force: the calling thread's [`with_mla_latent_quant`]
/// override, else the process value, which a process that never called
/// [`init_mla_latent_quant`] takes from the environment on first use (an
/// unknown value there is reported once and read as `auto`).
#[must_use]
pub fn mla_latent_quant() -> MlaLatentQuant {
    if let Some(option) = OVERRIDE.with(Cell::get) {
        return option;
    }
    *OPTION.get_or_init(|| {
        resolve_mla_latent_quant(None).unwrap_or_else(|err| {
            tracing::warn!("{err}; using auto");
            MlaLatentQuant::Auto
        })
    })
}

/// Run `f` with the option overridden on this thread, restoring the previous
/// state afterwards (also on unwind).
///
/// A seam for tests and measurement harnesses, which share one process and
/// cannot each fix the process-wide value. The MLX graph for a latent write
/// or read is built on the calling thread, so the override covers a model
/// forward run inside `f`.
pub fn with_mla_latent_quant<R>(option: MlaLatentQuant, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<MlaLatentQuant>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(OVERRIDE.with(|cell| cell.replace(Some(option))));
    f()
}

/// Whether `model_type` holds an MLA latent cache the option can act on: a
/// latent-pair family, or `deepseek_v2` (whose latent exists while absorbed
/// decode is on, which the caller knows and this does not).
#[must_use]
pub fn has_mla_latent_cache(model_type: &str) -> bool {
    caches_mla_latent_pair(model_type) || stores_latent_through_view(model_type)
}

/// The bit width `model_type` stores its MLA latent at when the operator
/// requested KV cache mode `requested` under `option`; `None` is the fp16
/// latent, and the answer for every type with no MLA latent cache.
///
/// `auto` is the default gate: a family in [`QUANTIZED_LATENT_FAMILIES`], or
/// `deepseek_v2`, stores the width `requested` selects
/// ([`KVCacheMode::latent_affine_bits`], 8 under `int8`), and every other
/// family keeps fp16. An explicit value ignores both the list and
/// `requested`.
///
/// This is the single resolution point of issue #3179; see the module docs
/// for what reads it.
#[must_use]
pub fn effective_latent_bits(
    model_type: &str,
    requested: KVCacheMode,
    option: MlaLatentQuant,
) -> Option<i32> {
    if !has_mla_latent_cache(model_type) {
        return None;
    }
    match option {
        MlaLatentQuant::Fp16 => None,
        MlaLatentQuant::Int8 => Some(8),
        MlaLatentQuant::Int4 => Some(4),
        MlaLatentQuant::Auto => {
            let needle = model_type.trim().to_ascii_lowercase();
            let listed = QUANTIZED_LATENT_FAMILIES.contains(&needle.as_str());
            if listed || stores_latent_through_view(model_type) {
                requested.latent_affine_bits()
            } else {
                None
            }
        }
    }
}

/// [`latent_cache_bits`] for an explicit option value.
#[must_use]
pub const fn latent_cache_bits_with(mode: KVCacheMode, option: MlaLatentQuant) -> Option<i32> {
    match (mode.latent_affine_bits(), option) {
        (Some(_), MlaLatentQuant::Int4) => Some(4),
        (bits, _) => bits,
    }
}

/// The width a latent cache built in `mode` packs its latent at under the
/// option in force: [`KVCacheMode::latent_affine_bits`], narrowed to 4 under
/// `int4`. `None` for a mode that stores no affine latent.
///
/// The resolver has already decided that the cache is built in `int8`
/// storage, so this only picks the width inside it.
///
/// Used by: `mla::MlaLatentCache` (a fresh cache; one that already holds rows
/// keeps their width), `mlxcel::execution::kv_arch` (sizing) and the
/// prompt-cache adoption check.
#[must_use]
pub fn latent_cache_bits(mode: KVCacheMode) -> Option<i32> {
    latent_cache_bits_with(mode, mla_latent_quant())
}

/// One measured result an experimental latent precision is known by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExperimentalLatentNote {
    /// The `model_type` values the note is about; empty means every MLA
    /// family.
    pub families: &'static [&'static str],
    /// The latent widths the note applies to.
    pub bits: &'static [i32],
    /// The measurement, one sentence.
    pub text: &'static str,
}

/// What is known about each precision the default gate does not ship, keyed
/// by family and width. The startup warning quotes these, so a new
/// measurement is recorded here and nowhere else.
pub static EXPERIMENTAL_LATENT_NOTES: &[ExperimentalLatentNote] = &[
    ExperimentalLatentNote {
        families: &["glm4_moe_lite"],
        bits: &[8, 4],
        text: "GLM-4.7-Flash with an int8 latent showed 1 decided-position disagreement against \
               the fp16 latent behind 1024 tokens (docs/benchmark_results/\
               mla-quantized-latent-glm-longcat-m3ultra-2026-11-19.md)",
    },
    ExperimentalLatentNote {
        families: &["glm4_moe_lite"],
        bits: &[4],
        text: "GLM-4.7-Flash with an int4 latent showed 3 decided-position disagreements behind \
               1024 tokens and 3 behind 4096, and moved perplexity by -19.5% and -16.1% \
               (docs/benchmark_results/mla-latent-quant-option-m3ultra-2026-11-20.md)",
    },
    ExperimentalLatentNote {
        families: &["deepseek_v32", "deepseek_v3.2", "glm_moe_dsa"],
        bits: &[8, 4],
        text: "DeepSeek-V3.2 with an int8 key slot passed the logit trace, but the DSA indexer's \
               top-k overlap with fp16 fell below 0.99 on 49 of 61 layers behind 4096 tokens \
               (docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md)",
    },
    ExperimentalLatentNote {
        families: &[],
        bits: &[4],
        text: "a 4-bit latent on DeepSeek-V2-Lite had 0 decided disagreements but flipped the \
               top-1 token at 8.8% and 5.5% of positions, with near-tie gaps up to 0.86 logits \
               (docs/benchmark_results/mla-quantized-latent-m3ultra-2026-11-19.md); it always \
               decodes on the composed path, never the fused Metal kernel",
    },
    ExperimentalLatentNote {
        families: &[],
        bits: &[4],
        text: "on Moonlight-16B-A3B and LongCat-Flash-Lite a 4-bit latent also had 0 decided \
               disagreements, with near-tie flips up to 0.84 and 1.13 logits \
               (docs/benchmark_results/mla-latent-quant-option-m3ultra-2026-11-20.md)",
    },
];

/// The [`EXPERIMENTAL_LATENT_NOTES`] texts for `model_type` at `bits`.
#[must_use]
pub fn experimental_latent_notes(model_type: &str, bits: i32) -> Vec<&'static str> {
    let needle = model_type.trim().to_ascii_lowercase();
    EXPERIMENTAL_LATENT_NOTES
        .iter()
        .filter(|note| note.bits.contains(&bits))
        .filter(|note| note.families.is_empty() || note.families.contains(&needle.as_str()))
        .map(|note| note.text)
        .collect()
}

/// The operator-facing lines an explicit option produces for one model, to be
/// printed once at startup. Empty under `auto`.
///
/// * On a model with no MLA latent cache (or `deepseek_v2` with absorbed
///   decode off, `absorbed == false`) the option is ignored, and one line
///   says so.
/// * When `requested` is a quantized KV cache mode whose own latent precision
///   differs from the explicit value, one line names both; the explicit value
///   wins.
/// * An explicit `int8` on a family outside the default list, and any `int4`,
///   gets one line stating that the precision is experimental, with the
///   measured results from [`EXPERIMENTAL_LATENT_NOTES`].
///
/// `requested` is the mode the operator's flags select before family
/// resolution (`--kv-cache-mode`, or the `--kv-bits` table's base mode).
#[must_use]
pub fn latent_option_notices(
    model_type: &str,
    requested: KVCacheMode,
    absorbed: bool,
    option: MlaLatentQuant,
) -> Vec<String> {
    if !option.is_explicit() {
        return Vec::new();
    }
    let v2 = stores_latent_through_view(model_type);
    if !has_mla_latent_cache(model_type) || (v2 && !absorbed) {
        let why = if v2 {
            "absorbed MLA decode is off, so it caches decompressed per-head K/V"
        } else {
            "it has no MLA latent cache"
        };
        return vec![format!(
            "warning: {MLA_LATENT_QUANT_FLAG} {option} is ignored for this model \
             (model_type=\"{model_type}\"): {why}."
        )];
    }
    let mut notices = Vec::new();
    let explicit = effective_latent_bits(model_type, requested, option);
    let precision = |bits: Option<i32>| bits.map_or("fp16".to_string(), |b| format!("int{b}"));
    if requested != KVCacheMode::Fp16 && requested.latent_affine_bits() != explicit {
        notices.push(format!(
            "warning: --kv-cache-mode {requested} and {MLA_LATENT_QUANT_FLAG} {option} disagree \
             on the MLA latent precision (model_type=\"{model_type}\"); the explicit \
             {MLA_LATENT_QUANT_FLAG} wins and the latent is stored as {}.",
            precision(explicit)
        ));
    }
    if let Some(bits) = explicit {
        let default_gate =
            effective_latent_bits(model_type, KVCacheMode::Int8, MlaLatentQuant::Auto);
        if default_gate != Some(bits) {
            let mut line = format!(
                "warning: {MLA_LATENT_QUANT_FLAG} {option} is experimental \
                 (model_type=\"{model_type}\"): this latent precision is outside the default \
                 quality gate and is not validated for this family."
            );
            for note in experimental_latent_notes(model_type, bits) {
                line.push_str("\n\tMeasured: ");
                line.push_str(note);
                line.push('.');
            }
            if bits != crate::mla::latent_keys::SIDECAR_KEY_BITS
                && caches_dsa_indexer_key(model_type)
            {
                line.push_str("\n\tThe DSA indexer key stored beside the latent stays int8.");
            }
            notices.push(line);
        }
    }
    notices
}

#[cfg(test)]
#[path = "latent_option_tests.rs"]
mod tests;
