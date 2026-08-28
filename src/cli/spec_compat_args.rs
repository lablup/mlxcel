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

//! llama-server b10621 speculative-decoding compatibility options (#1433).
//!
//! b10621 exposes forty-plus speculative flags. mlxcel's speculative decoding
//! is MTP / DFlash through `--model-draft` (alias `--spec-draft-model`),
//! `--draft-kind`, `--draft-block-size`, and the draft-token cap `--draft` /
//! `--draft-max` / `--spec-draft-n-max`, all handled by
//! [`crate::cli::speculative_args`] and the server flags proper. Everything
//! here covers the REMAINDER: options that configure GGML draft-side process
//! placement, the n-gram speculation subsystem, or draft-sampler thresholds
//! that MTP / DFlash verification does not use.
//!
//! Same rule as [`crate::cli::ggml_compat_args`]: each option is accepted
//! (hidden) and its VALUE is classified. A value b10621's semantics make
//! inert on mlxcel is accepted silently; a value that would change behavior
//! mlxcel cannot reproduce fails startup with a diagnostic naming the option,
//! the platform limitation, and the mlxcel alternative where one exists.
//!
//! The n-gram tuning knobs (`--spec-ngram-*` sizes and hit counts) are a
//! deliberate exception: they only take effect when `--spec-type` selects an
//! n-gram mode, and mlxcel rejects every non-`none` `--spec-type`. With the
//! selector pinned to `none`, b10621 itself ignores the tuning values, so
//! accepting any value here reproduces upstream behavior exactly and the
//! whole family stays inert by construction.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference: <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>

use clap::Args;

use super::ggml_compat_args::{
    GgmlCompatRejection, env_flag, numeric_at_most, numeric_equals, present, reject_owned,
};

/// llama-server b10621 speculative compatibility options (#1433).
///
/// Flattened into both server binaries so the two surfaces cannot drift.
#[derive(Args, Debug, Clone, Default)]
pub struct SpecCompatArgs {
    /// b10621 `--spec-draft-n-min`: minimum draft tokens per step. Inert at
    /// its 0 default (no minimum); MTP / DFlash verify whole blocks and have
    /// no per-step minimum to enforce.
    #[arg(
        long = "spec-draft-n-min",
        env = "LLAMA_ARG_SPEC_DRAFT_N_MIN",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_n_min: Option<String>,

    /// b10621 `--spec-draft-p-min`: minimum draft probability. Inert at its
    /// 0.00 default (greedy, no threshold).
    #[arg(
        long = "spec-draft-p-min",
        env = "LLAMA_ARG_SPEC_DRAFT_P_MIN",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_p_min: Option<String>,

    /// b10621 `--spec-draft-p-split`: tree-split probability. Inert at its
    /// 0.10 default; mlxcel drafts a single sequence and never splits.
    #[arg(
        long = "spec-draft-p-split",
        env = "LLAMA_ARG_SPEC_DRAFT_P_SPLIT",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_p_split: Option<String>,

    /// b10621 `--spec-type`: which speculation subsystems to run. Only the
    /// `none` default is accepted (drafting is configured through
    /// `--model-draft` instead); every n-gram mode is rejected.
    #[arg(
        long = "spec-type",
        env = "LLAMA_ARG_SPEC_TYPE",
        value_name = "TYPES",
        hide = true
    )]
    pub spec_type: Option<String>,

    /// b10621 `--spec-draft-backend-sampling` (default: enabled). Inert:
    /// mlxcel's draft sampling always runs as MLX ops on the accelerator.
    #[arg(long = "spec-draft-backend-sampling", hide = true)]
    pub spec_draft_backend_sampling: bool,

    /// b10621 `--spec-draft-hf`: pull the draft model from a HF repo through
    /// GGML loading. Rejected; `--model-draft` accepts a repo-id directly.
    #[arg(
        long = "spec-draft-hf",
        env = "LLAMA_ARG_SPEC_DRAFT_HF_REPO",
        value_name = "REPO",
        hide = true
    )]
    pub spec_draft_hf: Option<String>,

    // ── GGML draft-side process placement (no GGML backend to place) ──
    /// b10621 `--spec-draft-cpu-mask` (draft CPU affinity). Rejected.
    #[arg(long = "spec-draft-cpu-mask", value_name = "M", hide = true)]
    pub spec_draft_cpu_mask: Option<String>,
    /// b10621 `--spec-draft-cpu-mask-batch`. Rejected.
    #[arg(long = "spec-draft-cpu-mask-batch", value_name = "M", hide = true)]
    pub spec_draft_cpu_mask_batch: Option<String>,
    /// b10621 `--spec-draft-cpu-range`. Rejected.
    #[arg(long = "spec-draft-cpu-range", value_name = "lo-hi", hide = true)]
    pub spec_draft_cpu_range: Option<String>,
    /// b10621 `--spec-draft-cpu-strict`. Inert at 0.
    #[arg(long = "spec-draft-cpu-strict", value_name = "0|1", hide = true)]
    pub spec_draft_cpu_strict: Option<String>,
    /// b10621 `--spec-draft-cpu-strict-batch`. Inert at 0.
    #[arg(long = "spec-draft-cpu-strict-batch", value_name = "0|1", hide = true)]
    pub spec_draft_cpu_strict_batch: Option<String>,
    /// b10621 `--spec-draft-poll`. Inert at 50 (the upstream default).
    #[arg(long = "spec-draft-poll", value_name = "0-100", hide = true)]
    pub spec_draft_poll: Option<String>,
    /// b10621 `--spec-draft-poll-batch`. Inert at 50.
    #[arg(long = "spec-draft-poll-batch", value_name = "0-100", hide = true)]
    pub spec_draft_poll_batch: Option<String>,
    /// b10621 `--spec-draft-prio`. Inert at 0 (normal priority).
    #[arg(long = "spec-draft-prio", value_name = "N", hide = true)]
    pub spec_draft_prio: Option<String>,
    /// b10621 `--spec-draft-prio-batch`. Inert at 0.
    #[arg(long = "spec-draft-prio-batch", value_name = "N", hide = true)]
    pub spec_draft_prio_batch: Option<String>,
    /// b10621 `--spec-draft-threads`. Inert for any non-positive count
    /// ("use hardware concurrency", the upstream default request).
    #[arg(long = "spec-draft-threads", value_name = "N", hide = true)]
    pub spec_draft_threads: Option<String>,
    /// b10621 `--spec-draft-threads-batch`. Same rule.
    #[arg(long = "spec-draft-threads-batch", value_name = "N", hide = true)]
    pub spec_draft_threads_batch: Option<String>,
    /// b10621 `--spec-draft-device`: GGML offload device list. Rejected.
    #[arg(long = "spec-draft-device", value_name = "dev1,dev2", hide = true)]
    pub spec_draft_device: Option<String>,
    /// b10621 `--spec-draft-ngl`: draft VRAM layer count. `auto` / `all` are
    /// inert (mlxcel runs every draft layer on the accelerator); a numeric
    /// partial offload is rejected.
    #[arg(
        long = "spec-draft-ngl",
        env = "LLAMA_ARG_N_GPU_LAYERS_DRAFT",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_ngl: Option<String>,
    /// b10621 `--spec-draft-override-tensor`. Rejected.
    #[arg(long = "spec-draft-override-tensor", value_name = "SPEC", hide = true)]
    pub spec_draft_override_tensor: Option<String>,
    /// b10621 `--spec-draft-type-k`: draft KV cache dtype for K. Inert at
    /// the `f16` default, which is also mlxcel's draft KV dtype.
    #[arg(
        long = "spec-draft-type-k",
        env = "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K",
        value_name = "TYPE",
        hide = true
    )]
    pub spec_draft_type_k: Option<String>,
    /// b10621 `--spec-draft-type-v`: draft KV cache dtype for V. Same rule.
    #[arg(
        long = "spec-draft-type-v",
        env = "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V",
        value_name = "TYPE",
        hide = true
    )]
    pub spec_draft_type_v: Option<String>,
    /// b10621 `--spec-draft-cpu-moe`: keep draft MoE weights on the CPU.
    /// Rejected when set.
    #[arg(long = "spec-draft-cpu-moe", hide = true)]
    pub spec_draft_cpu_moe: bool,
    /// b10621 `--spec-draft-n-cpu-moe`: first-N-layers CPU MoE. Rejected.
    #[arg(
        long = "spec-draft-n-cpu-moe",
        env = "LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_n_cpu_moe: Option<String>,

    // ── lookup decoding ──
    /// b10621 `--lookup-cache-static`: lookup-decoding cache path. Rejected.
    #[arg(long = "lookup-cache-static", value_name = "PATH", hide = true)]
    pub lookup_cache_static: Option<String>,
    /// b10621 `--lookup-cache-dynamic`. Rejected.
    #[arg(long = "lookup-cache-dynamic", value_name = "PATH", hide = true)]
    pub lookup_cache_dynamic: Option<String>,

    // ── n-gram tuning knobs: inert by construction (see module docs) ──
    /// b10621 `--spec-ngram-simple-min-hits`. Accepted; inert while
    /// `--spec-type` stays `none`, which mlxcel enforces.
    #[arg(long = "spec-ngram-simple-min-hits", value_name = "N", hide = true)]
    pub spec_ngram_simple_min_hits: Option<String>,
    /// b10621 `--spec-ngram-simple-size-m`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-simple-size-m", value_name = "N", hide = true)]
    pub spec_ngram_simple_size_m: Option<String>,
    /// b10621 `--spec-ngram-simple-size-n`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-simple-size-n", value_name = "N", hide = true)]
    pub spec_ngram_simple_size_n: Option<String>,
    /// b10621 `--spec-ngram-map-k-min-hits`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-map-k-min-hits", value_name = "N", hide = true)]
    pub spec_ngram_map_k_min_hits: Option<String>,
    /// b10621 `--spec-ngram-map-k-size-m`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-map-k-size-m", value_name = "N", hide = true)]
    pub spec_ngram_map_k_size_m: Option<String>,
    /// b10621 `--spec-ngram-map-k-size-n`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-map-k-size-n", value_name = "N", hide = true)]
    pub spec_ngram_map_k_size_n: Option<String>,
    /// b10621 `--spec-ngram-map-k4v-min-hits`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-map-k4v-min-hits", value_name = "N", hide = true)]
    pub spec_ngram_map_k4v_min_hits: Option<String>,
    /// b10621 `--spec-ngram-map-k4v-size-m`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-map-k4v-size-m", value_name = "N", hide = true)]
    pub spec_ngram_map_k4v_size_m: Option<String>,
    /// b10621 `--spec-ngram-map-k4v-size-n`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-map-k4v-size-n", value_name = "N", hide = true)]
    pub spec_ngram_map_k4v_size_n: Option<String>,
    /// b10621 `--spec-ngram-mod-n-match`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-mod-n-match", value_name = "N", hide = true)]
    pub spec_ngram_mod_n_match: Option<String>,
    /// b10621 `--spec-ngram-mod-n-max`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-mod-n-max", value_name = "N", hide = true)]
    pub spec_ngram_mod_n_max: Option<String>,
    /// b10621 `--spec-ngram-mod-n-min`. Accepted; inert (same rule).
    #[arg(long = "spec-ngram-mod-n-min", value_name = "N", hide = true)]
    pub spec_ngram_mod_n_min: Option<String>,
}

const NO_GGML_DRAFT: &str = "mlxcel loads the draft model through MLX on the same accelerator as \
     the target; there is no GGML draft process whose CPU placement, threads, or buffers could \
     be configured";
const NO_NGRAM: &str = "mlxcel's speculative decoding is MTP / DFlash draft-model verification; \
     it has no n-gram or lookup speculation subsystem";

impl SpecCompatArgs {
    /// Reject every value whose b10621 meaning mlxcel cannot reproduce.
    ///
    /// Runs before the model load; every check depends only on flag strings.
    ///
    /// # Errors
    ///
    /// Returns the first rejection in a fixed order, so a command line
    /// carrying several unsupported values always reports the same one.
    pub fn ensure_inert(&self) -> Result<(), GgmlCompatRejection> {
        self.rejection().map_or(Ok(()), Err)
    }

    fn rejection(&self) -> Option<GgmlCompatRejection> {
        // ── speculation subsystem selector ──
        // b10621 reads a comma-separated list; `none` (and an absent or
        // empty value) is the default and the only selection mlxcel honors.
        if let Some(spec_type) = present(self.spec_type.as_deref())
            && spec_type != "none"
        {
            return Some(reject_owned(
                "--spec-type",
                spec_type,
                NO_NGRAM,
                Some(
                    "--model-draft <path-or-repo-id> (with --draft-kind mtp|dflash) for draft-model speculation",
                ),
            ));
        }
        for (option, value) in [
            ("--lookup-cache-static", self.lookup_cache_static.as_deref()),
            (
                "--lookup-cache-dynamic",
                self.lookup_cache_dynamic.as_deref(),
            ),
        ] {
            if let Some(value) = present(value) {
                return Some(reject_owned(option, value, NO_NGRAM, None));
            }
        }

        // ── draft-sampler thresholds ──
        // MTP / DFlash verification accepts or rejects whole draft blocks
        // against the target distribution; the per-token draft thresholds
        // below have no place to act, so only their inert defaults pass.
        if let Some(value) = present(self.spec_draft_n_min.as_deref())
            && !numeric_equals(value, 0)
        {
            return Some(reject_owned(
                "--spec-draft-n-min",
                value,
                "mlxcel's MTP / DFlash drafters emit fixed verification blocks and have no \
                 per-step minimum draft count",
                Some("--draft-block-size to size the verification block"),
            ));
        }
        if let Some(value) = present(self.spec_draft_p_min.as_deref())
            && value.trim().parse::<f32>() != Ok(0.0)
        {
            return Some(reject_owned(
                "--spec-draft-p-min",
                value,
                "mlxcel verifies draft tokens against the target distribution (modified \
                 rejection sampling) instead of thresholding draft probabilities",
                None,
            ));
        }
        if let Some(value) = present(self.spec_draft_p_split.as_deref())
            && value.trim().parse::<f32>() != Ok(0.1)
        {
            return Some(reject_owned(
                "--spec-draft-p-split",
                value,
                "mlxcel drafts a single sequence per step and never tree-splits",
                None,
            ));
        }
        if let Some(value) = present(self.spec_draft_hf.as_deref()) {
            return Some(reject_owned(
                "--spec-draft-hf",
                value,
                "mlxcel does not load GGUF draft checkpoints from HuggingFace",
                Some("--model-draft <repo-id>, which auto-downloads the MLX checkpoint"),
            ));
        }

        // ── GGML draft-side process placement ──
        for (option, value) in [
            ("--spec-draft-cpu-mask", self.spec_draft_cpu_mask.as_deref()),
            (
                "--spec-draft-cpu-mask-batch",
                self.spec_draft_cpu_mask_batch.as_deref(),
            ),
            (
                "--spec-draft-cpu-range",
                self.spec_draft_cpu_range.as_deref(),
            ),
            ("--spec-draft-device", self.spec_draft_device.as_deref()),
            (
                "--spec-draft-override-tensor",
                self.spec_draft_override_tensor.as_deref(),
            ),
            (
                "--spec-draft-n-cpu-moe",
                self.spec_draft_n_cpu_moe.as_deref(),
            ),
        ] {
            if let Some(value) = present(value) {
                return Some(reject_owned(option, value, NO_GGML_DRAFT, None));
            }
        }
        if self.spec_draft_cpu_moe {
            return Some(reject_owned(
                "--spec-draft-cpu-moe",
                "--spec-draft-cpu-moe",
                NO_GGML_DRAFT,
                None,
            ));
        }
        for (option, value, inert) in [
            (
                "--spec-draft-cpu-strict",
                self.spec_draft_cpu_strict.as_deref(),
                0,
            ),
            (
                "--spec-draft-cpu-strict-batch",
                self.spec_draft_cpu_strict_batch.as_deref(),
                0,
            ),
            ("--spec-draft-poll", self.spec_draft_poll.as_deref(), 50),
            (
                "--spec-draft-poll-batch",
                self.spec_draft_poll_batch.as_deref(),
                50,
            ),
            ("--spec-draft-prio", self.spec_draft_prio.as_deref(), 0),
            (
                "--spec-draft-prio-batch",
                self.spec_draft_prio_batch.as_deref(),
                0,
            ),
        ] {
            if let Some(value) = present(value)
                && !numeric_equals(value, inert)
            {
                return Some(reject_owned(option, value, NO_GGML_DRAFT, None));
            }
        }
        for (option, value) in [
            ("--spec-draft-threads", self.spec_draft_threads.as_deref()),
            (
                "--spec-draft-threads-batch",
                self.spec_draft_threads_batch.as_deref(),
            ),
        ] {
            if let Some(value) = present(value)
                && !numeric_at_most(value, 0)
            {
                return Some(reject_owned(option, value, NO_GGML_DRAFT, None));
            }
        }
        if let Some(value) = present(self.spec_draft_ngl.as_deref())
            && value != "auto"
            && value != "all"
        {
            return Some(reject_owned(
                "--spec-draft-ngl",
                value,
                "mlxcel keeps every draft layer on the accelerator; a partial VRAM offload \
                 cannot be reproduced",
                None,
            ));
        }
        for (option, value) in [
            ("--spec-draft-type-k", self.spec_draft_type_k.as_deref()),
            ("--spec-draft-type-v", self.spec_draft_type_v.as_deref()),
        ] {
            if let Some(value) = present(value)
                && value != "f16"
            {
                return Some(reject_owned(
                    option,
                    value,
                    "mlxcel's draft KV cache runs at f16; the GGML quantized draft cache \
                     types have no counterpart",
                    None,
                ));
            }
        }

        // `--spec-draft-backend-sampling` and every `--spec-ngram-*` tuning
        // knob are accepted as inert (see the struct field docs).
        None
    }

    /// Fold the b10621 value-less environment bindings in, using upstream's
    /// own truthy rule (see [`crate::cli::ggml_compat_args::env_flag`]).
    pub fn apply_env_bindings(&mut self) {
        self.spec_draft_backend_sampling |= env_flag("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING");
        self.spec_draft_cpu_moe |= env_flag("LLAMA_ARG_SPEC_DRAFT_CPU_MOE");
    }
}

#[cfg(test)]
#[path = "spec_compat_args_tests.rs"]
mod tests;
