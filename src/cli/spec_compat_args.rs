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
//! `--spec-type` is b10621's speculation-subsystem selector (draft-simple,
//! draft-eagle3, draft-mtp, draft-dflash, draft-dspark, and five n-gram
//! modes; `none` disables speculation outright, draft model or not). mlxcel
//! translates the values it can honor exactly: `none` disables the drafter
//! (matching upstream, which stops inferring a speculation type once the
//! selector is explicit), `draft-mtp` maps onto `--draft-kind mtp`, and
//! `draft-dflash` onto `--draft-kind dflash`. Every other value, and any
//! comma-list mlxcel cannot run as a single subsystem, fails startup with a
//! per-value diagnostic.
//!
//! The n-gram tuning knobs (`--spec-ngram-*` sizes and hit counts) only take
//! effect when the selector picks an n-gram mode, which mlxcel rejects, so
//! any tuning value here is inert exactly as it is upstream with a
//! non-n-gram selector and the whole family is accepted.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference: <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>

use clap::Args;

use super::ggml_compat_args::{
    GgmlCompatRejection, env_bool_pair, env_flag, numeric_at_most, numeric_equals, present,
    reject_owned,
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
        alias = "draft-p-min",
        env = "LLAMA_ARG_SPEC_DRAFT_P_MIN",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_p_min: Option<String>,

    /// b10621 `--spec-draft-p-split`: tree-split probability. Inert at its
    /// 0.10 default; mlxcel drafts a single sequence and never splits.
    #[arg(
        long = "spec-draft-p-split",
        alias = "draft-p-split",
        env = "LLAMA_ARG_SPEC_DRAFT_P_SPLIT",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_p_split: Option<String>,

    /// b10621 `--spec-type`: which speculation subsystems to run. `none`
    /// disables speculation (dropping any configured draft model, matching
    /// upstream), `draft-mtp` / `draft-dflash` translate onto
    /// `--draft-kind`; everything else is rejected. See
    /// [`SpecCompatArgs::resolved_spec_type`].
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

    /// b10621 `--no-spec-draft-backend-sampling`: move draft sampling to a
    /// host-side CPU sampler. The operator-active half of the pair; mlxcel
    /// cannot reproduce it, so it is rejected.
    #[arg(long = "no-spec-draft-backend-sampling", hide = true)]
    pub no_spec_draft_backend_sampling: bool,

    /// b10621 `--spec-draft-hf`: pull the draft model from a HF repo through
    /// GGML loading. Rejected; `--model-draft` accepts a repo-id directly.
    #[arg(
        long = "spec-draft-hf",
        alias = "hf-repo-draft",
        env = "LLAMA_ARG_SPEC_DRAFT_HF_REPO",
        value_name = "REPO",
        hide = true
    )]
    pub spec_draft_hf: Option<String>,

    // ── GGML draft-side process placement (no GGML backend to place) ──
    /// b10621 `--spec-draft-cpu-mask` (draft CPU affinity). Rejected.
    #[arg(
        long = "spec-draft-cpu-mask",
        alias = "cpu-mask-draft",
        value_name = "M",
        hide = true
    )]
    pub spec_draft_cpu_mask: Option<String>,
    /// b10621 `--spec-draft-cpu-mask-batch`. Rejected.
    #[arg(
        long = "spec-draft-cpu-mask-batch",
        alias = "cpu-mask-batch-draft",
        value_name = "M",
        hide = true
    )]
    pub spec_draft_cpu_mask_batch: Option<String>,
    /// b10621 `--spec-draft-cpu-range`. Rejected.
    #[arg(
        long = "spec-draft-cpu-range",
        alias = "cpu-range-draft",
        value_name = "lo-hi",
        hide = true
    )]
    pub spec_draft_cpu_range: Option<String>,
    /// b10621 `--spec-draft-cpu-strict`. Inert at 0.
    #[arg(
        long = "spec-draft-cpu-strict",
        alias = "cpu-strict-draft",
        value_name = "0|1",
        hide = true
    )]
    pub spec_draft_cpu_strict: Option<String>,
    /// b10621 `--spec-draft-cpu-strict-batch`. Inert at 0.
    #[arg(
        long = "spec-draft-cpu-strict-batch",
        alias = "cpu-strict-batch-draft",
        value_name = "0|1",
        hide = true
    )]
    pub spec_draft_cpu_strict_batch: Option<String>,
    /// b10621 `--spec-draft-poll`. Inert at 50 (the upstream default).
    #[arg(
        long = "spec-draft-poll",
        alias = "poll-draft",
        value_name = "0-100",
        hide = true
    )]
    pub spec_draft_poll: Option<String>,
    /// b10621 `--spec-draft-poll-batch`. Inert at 50.
    #[arg(
        long = "spec-draft-poll-batch",
        alias = "poll-batch-draft",
        value_name = "0-100",
        hide = true
    )]
    pub spec_draft_poll_batch: Option<String>,
    /// b10621 `--spec-draft-prio`. Inert at 0 (normal priority).
    #[arg(
        long = "spec-draft-prio",
        alias = "prio-draft",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_prio: Option<String>,
    /// b10621 `--spec-draft-prio-batch`. Inert at 0.
    #[arg(
        long = "spec-draft-prio-batch",
        alias = "prio-batch-draft",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_prio_batch: Option<String>,
    /// b10621 `--spec-draft-threads`. Inert for any non-positive count
    /// ("use hardware concurrency", the upstream default request).
    #[arg(
        long = "spec-draft-threads",
        alias = "threads-draft",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_threads: Option<String>,
    /// b10621 `--spec-draft-threads-batch`. Same rule.
    #[arg(
        long = "spec-draft-threads-batch",
        alias = "threads-batch-draft",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_threads_batch: Option<String>,
    /// b10621 `--spec-draft-device`: GGML offload device list. Rejected.
    #[arg(
        long = "spec-draft-device",
        alias = "device-draft",
        value_name = "dev1,dev2",
        hide = true
    )]
    pub spec_draft_device: Option<String>,
    /// b10621 `--spec-draft-ngl`: draft VRAM layer count. `auto` / `all` are
    /// inert (mlxcel runs every draft layer on the accelerator); a numeric
    /// partial offload is rejected.
    #[arg(
        long = "spec-draft-ngl",
        alias = "gpu-layers-draft",
        alias = "n-gpu-layers-draft",
        env = "LLAMA_ARG_N_GPU_LAYERS_DRAFT",
        value_name = "N",
        hide = true
    )]
    pub spec_draft_ngl: Option<String>,
    /// b10621 `--spec-draft-override-tensor`. Rejected.
    #[arg(
        long = "spec-draft-override-tensor",
        alias = "override-tensor-draft",
        value_name = "SPEC",
        hide = true
    )]
    pub spec_draft_override_tensor: Option<String>,
    /// b10621 `--spec-draft-type-k`: draft KV cache dtype for K. Inert at
    /// the `f16` default, which is also mlxcel's draft KV dtype.
    #[arg(
        long = "spec-draft-type-k",
        alias = "cache-type-k-draft",
        env = "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K",
        value_name = "TYPE",
        hide = true
    )]
    pub spec_draft_type_k: Option<String>,
    /// b10621 `--spec-draft-type-v`: draft KV cache dtype for V. Same rule.
    #[arg(
        long = "spec-draft-type-v",
        alias = "cache-type-v-draft",
        env = "LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V",
        value_name = "TYPE",
        hide = true
    )]
    pub spec_draft_type_v: Option<String>,
    /// b10621 `--spec-draft-cpu-moe`: keep draft MoE weights on the CPU.
    /// Rejected when set.
    #[arg(long = "spec-draft-cpu-moe", alias = "cpu-moe-draft", hide = true)]
    pub spec_draft_cpu_moe: bool,
    /// b10621 `--spec-draft-n-cpu-moe`: first-N-layers CPU MoE. Rejected.
    #[arg(
        long = "spec-draft-n-cpu-moe",
        alias = "spec-draft-ncmoe",
        alias = "n-cpu-moe-draft",
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

/// How a `--spec-type` value resolves onto mlxcel's speculative controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpecTypeResolution {
    /// `--spec-type none`: b10621 disables speculation outright (the
    /// explicit selector stops the draft-sidecar type inference), so the
    /// caller must drop its draft model.
    pub disable_speculation: bool,
    /// `--spec-type draft-mtp` / `draft-dflash`: the exact `--draft-kind`
    /// translation the caller must apply (and error on if it conflicts with
    /// an explicit `--draft-kind`).
    pub draft_kind: Option<&'static str>,
}

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

    /// Resolve `--spec-type` onto mlxcel's speculative controls (#1433).
    ///
    /// b10621's selector takes a comma-separated list of speculation
    /// subsystems. mlxcel honors exactly the values it can translate:
    /// `none` disables speculation (upstream semantics: an explicit selector
    /// stops the draft-sidecar type inference, so a draft model with
    /// `--spec-type none` runs no speculation), `draft-mtp` and
    /// `draft-dflash` translate onto `--draft-kind`. Any other subsystem
    /// (the n-gram modes, `draft-simple`, `draft-eagle3`, `draft-dspark`),
    /// and any list asking for more than one subsystem at once, is rejected
    /// with a per-value diagnostic.
    ///
    /// # Errors
    ///
    /// Returns the rejection for the first unsupported value.
    pub fn resolved_spec_type(&self) -> Result<SpecTypeResolution, GgmlCompatRejection> {
        let Some(raw) = present(self.spec_type.as_deref()) else {
            return Ok(SpecTypeResolution::default());
        };
        let values: Vec<&str> = raw
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .collect();
        let mut resolution = SpecTypeResolution::default();
        let mut selected: Option<&str> = None;
        for value in values {
            match value {
                "none" => resolution.disable_speculation = true,
                "draft-mtp" | "draft-dflash" => {
                    if let Some(previous) = selected
                        && previous != value
                    {
                        return Err(reject_owned(
                            "--spec-type",
                            raw,
                            "mlxcel runs one speculative subsystem per server; a list selecting \
                             several draft types cannot be honored",
                            Some("pick one of draft-mtp or draft-dflash"),
                        ));
                    }
                    selected = Some(value);
                    resolution.draft_kind = Some(if value == "draft-mtp" {
                        "mtp"
                    } else {
                        "dflash"
                    });
                }
                "draft-simple" | "draft-eagle3" | "draft-dspark" => {
                    return Err(reject_owned(
                        "--spec-type",
                        value,
                        "mlxcel's draft verification implements the MTP and DFlash subsystems \
                         only; this draft type has no mlxcel implementation",
                        Some(
                            "--spec-type draft-mtp or draft-dflash with --model-draft <path-or-repo-id>",
                        ),
                    ));
                }
                other if other.starts_with("ngram") => {
                    return Err(reject_owned(
                        "--spec-type",
                        other,
                        NO_NGRAM,
                        Some(
                            "--spec-type draft-mtp or draft-dflash with --model-draft <path-or-repo-id> for draft-model speculation",
                        ),
                    ));
                }
                other => {
                    return Err(reject_owned(
                        "--spec-type",
                        other,
                        "not a b10621 speculation type mlxcel recognizes",
                        Some("none, draft-mtp, or draft-dflash"),
                    ));
                }
            }
        }
        if resolution.disable_speculation && resolution.draft_kind.is_some() {
            return Err(reject_owned(
                "--spec-type",
                raw,
                "none disables speculation and cannot be combined with a draft type",
                Some("pass either none or a single draft type"),
            ));
        }
        Ok(resolution)
    }

    fn rejection(&self) -> Option<GgmlCompatRejection> {
        // ── speculation subsystem selector ──
        // Validated through `resolved_spec_type` so `ensure_inert` rejects
        // unsupported subsystems even when a caller skips the translation
        // step; the translation itself is applied by the binaries.
        if let Err(rejection) = self.resolved_spec_type() {
            return Some(rejection);
        }
        if self.no_spec_draft_backend_sampling {
            return Some(reject_owned(
                "--no-spec-draft-backend-sampling",
                "--no-spec-draft-backend-sampling",
                "mlxcel cannot move draft sampling to a host-side CPU sampler; its sampling \
                 always runs as MLX ops on the accelerator",
                None,
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
            // A negative count is llama.cpp's historical spelling of "all",
            // which is what mlxcel already does (mirrors `--gpu-layers`).
            && !numeric_at_most(value, -1)
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

    /// Fold the b10621 value-less environment bindings in.
    ///
    /// `--spec-draft-backend-sampling` is a `--x` / `--no-x` pair upstream,
    /// so its variable goes through [`env_bool_pair`]: a truthy value
    /// re-affirms the (inert) default, a falsey value or the
    /// `LLAMA_ARG_NO_*` alias selects the rejected CPU-sampler half, and an
    /// unrecognized value is an error exactly as b10621's `parse_bool_value`
    /// throws. `--spec-draft-cpu-moe` is a plain flag and uses [`env_flag`].
    ///
    /// # Errors
    ///
    /// Returns the variable name and raw value when a boolean variable holds
    /// a value b10621 would reject.
    pub fn apply_env_bindings(&mut self) -> Result<(), (&'static str, String)> {
        match env_bool_pair("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING") {
            Some(Ok(true)) => self.spec_draft_backend_sampling = true,
            Some(Ok(false)) => self.no_spec_draft_backend_sampling = true,
            Some(Err(raw)) => return Err(("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING", raw)),
            None => {}
        }
        self.spec_draft_cpu_moe |= env_flag("LLAMA_ARG_SPEC_DRAFT_CPU_MOE");
        Ok(())
    }
}

#[cfg(test)]
#[path = "spec_compat_args_tests.rs"]
mod tests;
