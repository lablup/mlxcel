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

//! RoPE and YaRN runtime-override flag group (llama-server b10621 parity).
//!
//! One canonical clap definition of `--rope-scaling`, `--rope-scale`,
//! `--rope-freq-base`, `--rope-freq-scale` and the five `--yarn-*` knobs,
//! flattened into both server binaries so `mlxcel serve` and `mlxcel-server`
//! accept the same command line. The upstream definitions are in
//! [`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp).
//!
//! # The YaRN knobs (#1472)
//!
//! All nine flags are implemented. The four RoPE flags rewrite the
//! checkpoint's `rope_scaling` block and `rope_theta` before the model is
//! built, through [`crate::models::rope_overrides`]. The five YaRN flags
//! resolve to [`crate::models::rope_overrides::YarnKnobs`] and tune the YaRN
//! frequency table wherever the shared RoPE path builds one (see the `Yarn`
//! arm of [`crate::models::rope_utils::RopeScalingKind`]), whether
//! `--rope-scaling yarn` forced it or the checkpoint's own block declared it.
//!
//! b10621 gives all five a sentinel default that means "leave the model's own
//! values alone" (`-1.00`, and `0` for `--yarn-orig-ctx`); the sentinel maps
//! to an absent knob here, so a deployment script that spells out the
//! upstream defaults asks for the checkpoint's own behavior and gets it. With
//! a non-YaRN rotation in force the knobs are inert, exactly as they are in
//! b10621, whose `llama_context` reads them only under
//! `LLAMA_ROPE_SCALING_TYPE_YARN`.
//!
//! Used by: mlxcel serve, mlxcel-server.

use clap::Args;

use crate::models::rope_overrides::{RopeRuntimeOverride, YarnKnobs};

/// Shared RoPE / YaRN runtime-override flag group.
///
/// Every flag is `Option<_>` with no clap default, so "not given" stays
/// distinguishable from "given at b10621's default". That distinction is what
/// lets a sentinel `--yarn-ext-factor -1` resolve to "the model's own value"
/// while an ordinary value becomes a knob, and it keeps the resolved override
/// out of the way entirely for the overwhelmingly common case where no
/// rotation flag was passed.
#[derive(Args, Debug, Default, Clone)]
#[command(next_help_heading = "RoPE / YaRN Options")]
pub struct RopeOverrideArgs {
    /// RoPE frequency scaling method: `none`, `linear`, or `yarn`.
    ///
    /// Defaults to the scheme the checkpoint's own `rope_scaling` block names.
    /// `none` drops that block and rotates with the plain `base^(2i/d)` table;
    /// `linear` divides positions by `--rope-scale` (or by the checkpoint's own
    /// factor when that flag is absent); `yarn` builds the YaRN table with the
    /// factor from `--rope-scale` / `--rope-freq-scale` and the `--yarn-*`
    /// knobs.
    ///
    /// Applied before the model is constructed, so before any KV cache exists.
    /// Refused at startup when the loaded architecture computes its RoPE
    /// frequencies outside the shared path and would ignore the request.
    #[arg(
        long = "rope-scaling",
        env = "LLAMA_ARG_ROPE_SCALING_TYPE",
        value_name = "{none,linear,yarn}"
    )]
    pub rope_scaling: Option<String>,

    /// RoPE context scaling factor: expands context by a factor of N.
    ///
    /// The reciprocal spelling of `--rope-freq-scale`; `--rope-scale 8` and
    /// `--rope-freq-scale 0.125` are the same request. Passing both with values
    /// that are not reciprocals is refused rather than resolved by precedence.
    #[arg(long = "rope-scale", env = "LLAMA_ARG_ROPE_SCALE", value_name = "N")]
    pub rope_scale: Option<f32>,

    /// RoPE base frequency, used by NTK-aware scaling.
    ///
    /// Replaces the checkpoint's `rope_theta`. On Gemma 3 this reaches the
    /// global-attention layers only, matching llama.cpp's separate SWA rope
    /// parameters.
    #[arg(
        long = "rope-freq-base",
        env = "LLAMA_ARG_ROPE_FREQ_BASE",
        value_name = "N"
    )]
    pub rope_freq_base: Option<f32>,

    /// RoPE frequency scaling factor: expands context by a factor of 1/N.
    #[arg(
        long = "rope-freq-scale",
        env = "LLAMA_ARG_ROPE_FREQ_SCALE",
        value_name = "N"
    )]
    pub rope_freq_scale: Option<f32>,

    /// YaRN: original context size of the model (0 = model training context size).
    ///
    /// Tunes the YaRN correction band when a YaRN rotation is in force; see
    /// the module docs.
    #[arg(
        long = "yarn-orig-ctx",
        env = "LLAMA_ARG_YARN_ORIG_CTX",
        value_name = "N"
    )]
    pub yarn_orig_ctx: Option<i64>,

    /// YaRN: extrapolation mix factor (0.0 = full interpolation).
    ///
    /// Tunes the interpolation/extrapolation mix when a YaRN rotation is in
    /// force; the sentinel (-1.0) resolves to 1.0 there, as it does upstream.
    #[arg(
        long = "yarn-ext-factor",
        env = "LLAMA_ARG_YARN_EXT_FACTOR",
        value_name = "N"
    )]
    pub yarn_ext_factor: Option<f32>,

    /// YaRN: scale sqrt(t) or attention magnitude.
    ///
    /// Participates only when the resolved extrapolation mix is 0, which is
    /// b10621's own resolution order; see the module docs.
    #[arg(
        long = "yarn-attn-factor",
        env = "LLAMA_ARG_YARN_ATTN_FACTOR",
        value_name = "N"
    )]
    pub yarn_attn_factor: Option<f32>,

    /// YaRN: high correction dim or alpha (default 1).
    ///
    /// Tunes the YaRN correction band when a YaRN rotation is in force.
    #[arg(
        long = "yarn-beta-slow",
        env = "LLAMA_ARG_YARN_BETA_SLOW",
        value_name = "N"
    )]
    pub yarn_beta_slow: Option<f32>,

    /// YaRN: low correction dim or beta (default 32).
    ///
    /// Tunes the YaRN correction band when a YaRN rotation is in force.
    #[arg(
        long = "yarn-beta-fast",
        env = "LLAMA_ARG_YARN_BETA_FAST",
        value_name = "N"
    )]
    pub yarn_beta_fast: Option<f32>,
}

/// b10621's `-1.0` sentinel for the four floating-point YaRN knobs, meaning
/// "keep whatever the model was trained with".
///
/// `common_params` initializes `yarn_ext_factor`, `yarn_attn_factor`,
/// `yarn_beta_fast` and `yarn_beta_slow` to `-1.0f`, and `llama_context`
/// substitutes the model's own value for any that is still negative.
const YARN_SENTINEL: f32 = -1.0;

/// b10621's `0` sentinel for `--yarn-orig-ctx`, meaning "the model's training
/// context size".
const YARN_ORIG_CTX_SENTINEL: i64 = 0;

impl RopeOverrideArgs {
    /// Resolve this group into a RoPE override, refusing what cannot be served.
    ///
    /// A self-contradictory RoPE request and a YaRN knob outside its value
    /// domain are both startup errors, never warnings
    /// ([`RopeRuntimeOverride::from_flags_with_yarn`]).
    pub fn resolve(&self) -> Result<Option<RopeRuntimeOverride>, String> {
        RopeRuntimeOverride::from_flags_with_yarn(
            self.rope_scaling.as_deref(),
            self.rope_scale,
            self.rope_freq_scale,
            self.rope_freq_base,
            self.yarn_knobs(),
        )
    }

    /// The five YaRN flags as knobs, with b10621's sentinels mapped to
    /// "use the model's own value" (#1472).
    ///
    /// `common_params` initializes the four float knobs to `-1.0f` and the
    /// original context to `0`, and `llama_context` substitutes the model's
    /// own value for any that is still at the sentinel; an absent knob here is
    /// that substitution.
    pub fn yarn_knobs(&self) -> YarnKnobs {
        YarnKnobs {
            orig_ctx: self
                .yarn_orig_ctx
                .filter(|&value| value != YARN_ORIG_CTX_SENTINEL),
            ext_factor: self.yarn_ext_factor.filter(|&value| value != YARN_SENTINEL),
            attn_factor: self
                .yarn_attn_factor
                .filter(|&value| value != YARN_SENTINEL),
            beta_fast: self.yarn_beta_fast.filter(|&value| value != YARN_SENTINEL),
            beta_slow: self.yarn_beta_slow.filter(|&value| value != YARN_SENTINEL),
        }
    }
}

/// Resolve the group from the environment alone.
///
/// The server binaries resolve their own [`RopeOverrideArgs`] through clap,
/// where a flag beats the environment. Out-of-band harnesses have no command
/// line to parse but still need the override installed in their own process,
/// because it is process-wide by construction (see
/// [`crate::models::rope_overrides`]): `examples/logit_trace` compares two
/// rotations by running two processes, exactly as `docs/benchmarks.md`
/// prescribes for `MLXCEL_QMV_WIDE`.
///
/// Parsing an argv of just the program name makes clap fill every field from
/// the `LLAMA_ARG_*` variable it is already bound to, so the harness and the
/// server read the same environment through the same definitions rather than
/// through a second hand-written reader that can drift.
pub fn from_env() -> Result<Option<RopeRuntimeOverride>, String> {
    use clap::Parser;

    #[derive(Parser)]
    #[command(allow_negative_numbers = true)]
    struct EnvOnly {
        #[command(flatten)]
        rope: RopeOverrideArgs,
    }

    EnvOnly::try_parse_from(["mlxcel"])
        .map_err(|e| e.to_string())?
        .rope
        .resolve()
}

/// Resolve the group from the environment and install it process-wide.
///
/// Convenience for harnesses; the server installs its own resolved value.
pub fn install_from_env() -> Result<Option<RopeRuntimeOverride>, String> {
    let resolved = from_env()?;
    crate::models::rope_overrides::install(resolved)?;
    Ok(resolved)
}

#[cfg(test)]
#[path = "rope_args_tests.rs"]
mod tests;
