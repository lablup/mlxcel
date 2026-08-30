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

//! Operator-supplied RoPE overrides, applied to a checkpoint's own
//! `rope_scaling` block and `rope_theta` before the model is constructed.
//!
//! This is the mlxcel side of llama-server b10621's `--rope-scaling`,
//! `--rope-scale`, `--rope-freq-scale` and `--rope-freq-base`
//! ([`common/arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp)).
//! Upstream keeps them on `common_params` and hands them to `llama_model_load`,
//! which rebuilds the rotation before any KV cache exists. mlxcel has no
//! equivalent single parameter object reaching every family's loader: each
//! family reads `config.json` inside its own `load()`, so there is no argument
//! to thread. The override is therefore a process-wide value installed once,
//! before the first load, and read at the two seams every family on the shared
//! RoPE path already goes through.
//!
//! # Why the applications counter exists
//!
//! An override that reaches no seam is the exact failure epic #1431 is about:
//! the flag parses, the server starts, and the model rotates with the
//! checkpoint's own frequencies while the operator believes otherwise. Only six
//! families route through [`crate::models::rope_utils`]; the rest
//! (DeepSeek V2/V3/V4, Gemma 4, gpt-oss, Phi-3, Mellum, TeleChat3, Exaone 4,
//! Hunyuan MoE, and every MRoPE VLM) compute their frequencies inline and would
//! ignore it silently.
//!
//! Rather than maintain a hand-written table of which `ModelType` honors the
//! override (which goes stale the moment a family is ported), every seam that
//! consumes the override increments [`applications`]. Server startup compares
//! that count against zero after the model is loaded and refuses to serve when
//! an override was requested and never applied. A new family that wires itself
//! into `rope_utils` starts being accepted without anyone editing a list, and a
//! family that does not is named in the error.
//!
//! # YaRN (#1472)
//!
//! [`RopeScalingKind`](crate::models::rope_utils::RopeScalingKind) implements
//! `default`, `linear`, `llama3` and, since #1472, `yarn`. `--rope-scaling
//! yarn` forces the YaRN table on the shared path, and the five `--yarn-*`
//! knobs ([`YarnKnobs`]) tune whatever YaRN rotation ends up in force, whether
//! forced by the flag or declared by the checkpoint's own block; with a
//! non-YaRN rotation in force they are inert, exactly as they are in b10621
//! (`llama_context` reads them only under `LLAMA_ROPE_SCALING_TYPE_YARN`).
//! The families that implement YaRN outside this seam (DeepSeek V2/V3.2/V4,
//! gpt-oss, Mellum, TeleChat3) build it from their checkpoint's own block; an
//! override installed against one of them still refuses to serve through
//! [`verify_applied`] rather than being ignored.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::rope_utils::RopeScalingSpec;

/// The scaling scheme `--rope-scaling` can force.
///
/// Mirrors b10621's `{none,linear,yarn}` value domain exactly, including
/// `yarn`: the value parses so the diagnostic can name what is missing, rather
/// than clap reporting "invalid value" for a scheme llama.cpp accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingTypeOverride {
    /// `LLAMA_ROPE_SCALING_TYPE_NONE`: rotate with the plain `base^(2i/d)`
    /// table whatever the checkpoint declares.
    None,
    /// `LLAMA_ROPE_SCALING_TYPE_LINEAR`: divide positions by a factor.
    Linear,
    /// `LLAMA_ROPE_SCALING_TYPE_YARN`: force the YaRN table on the shared
    /// RoPE path (#1472). See the module documentation.
    Yarn,
}

impl RopeScalingTypeOverride {
    /// Parse the b10621 spelling. `Err` carries the accepted domain.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "none" => Ok(Self::None),
            "linear" => Ok(Self::Linear),
            "yarn" => Ok(Self::Yarn),
            other => Err(format!(
                "--rope-scaling {other:?} is not one of {{none,linear,yarn}}"
            )),
        }
    }

    /// The spelling this variant was parsed from.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Yarn => "yarn",
        }
    }
}

/// A resolved RoPE override, ready to apply to a checkpoint's config.
///
/// `freq_scale` is stored in llama.cpp's orientation (`rope_freq_scale`, the
/// multiplier applied to a position), not in the `rope_scaling.factor`
/// orientation of a HuggingFace config. b10621 derives it two ways and they are
/// reciprocals of each other: `--rope-scale N` sets `rope_freq_scale = 1/N`
/// while `--rope-freq-scale N` sets it to `N`. Normalizing at the edge keeps
/// the inversion in one place instead of at every consumer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeRuntimeOverride {
    scaling_type: Option<RopeScalingTypeOverride>,
    freq_base: Option<f32>,
    freq_scale: Option<f32>,
    yarn: YarnKnobs,
}

/// The five llama-server b10621 `--yarn-*` runtime knobs (#1472).
///
/// `None` is the b10621 sentinel (`-1.0`, and `0` for the original context),
/// meaning "use the values the model was trained with". The knobs tune a YaRN
/// rotation wherever one is in force, whether `--rope-scaling yarn` forced it
/// or the checkpoint's own `rope_scaling` block declared it, and are inert
/// against any other rotation, exactly as they are upstream.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct YarnKnobs {
    /// `--yarn-orig-ctx`: the original (pre-extension) context length.
    pub orig_ctx: Option<i64>,
    /// `--yarn-ext-factor`: extrapolation mix factor (`0.0` = full
    /// interpolation).
    pub ext_factor: Option<f32>,
    /// `--yarn-attn-factor`: attention magnitude. Participates only when the
    /// resolved extrapolation mix is `0`, which is b10621's own resolution
    /// order (`llama-context.cpp` recomputes the factor whenever the mix is
    /// non-zero).
    pub attn_factor: Option<f32>,
    /// `--yarn-beta-fast`: low correction rotation count (default 32).
    pub beta_fast: Option<f32>,
    /// `--yarn-beta-slow`: high correction rotation count (default 1).
    pub beta_slow: Option<f32>,
}

impl YarnKnobs {
    /// Whether any knob was given at a non-sentinel value.
    pub fn any_set(&self) -> bool {
        self.orig_ctx.is_some()
            || self.ext_factor.is_some()
            || self.attn_factor.is_some()
            || self.beta_fast.is_some()
            || self.beta_slow.is_some()
    }

    /// Screen the knob values a table could not be built from.
    ///
    /// The startup error names the flag and the value, in the same shape the
    /// RoPE scalar screening uses.
    fn validate(&self) -> Result<(), String> {
        if let Some(value) = self.orig_ctx
            && value < 0
        {
            return Err(format!(
                "--yarn-orig-ctx {value} is negative; pass the model's original context length, \
                 or 0 to use the value the model was trained with"
            ));
        }
        for (flag, value, min_exclusive) in [
            ("--yarn-ext-factor", self.ext_factor, false),
            ("--yarn-attn-factor", self.attn_factor, true),
            ("--yarn-beta-fast", self.beta_fast, true),
            ("--yarn-beta-slow", self.beta_slow, true),
        ] {
            if let Some(value) = value {
                let ok = value.is_finite()
                    && if min_exclusive {
                        value > 0.0
                    } else {
                        value >= 0.0
                    };
                if !ok {
                    return Err(format!(
                        "{flag} {value} is not a {} finite number; a YaRN parameter outside its \
                         domain produces a frequency table that decodes wrongly without an error",
                        if min_exclusive {
                            "positive"
                        } else {
                            "non-negative"
                        }
                    ));
                }
            }
        }
        Ok(())
    }
}

impl RopeRuntimeOverride {
    /// Build an override from the four b10621 flags, or `Ok(None)` when none of
    /// them was given.
    ///
    /// `rope_scale` and `freq_scale` are the raw `--rope-scale` and
    /// `--rope-freq-scale` values. b10621 writes both into the same
    /// `rope_freq_scale` field, so the later flag on the command line wins
    /// there; clap gives mlxcel two independent options, so a request that sets
    /// both to values that are not reciprocals is ambiguous and is refused
    /// rather than resolved by an arbitrary precedence.
    pub fn from_flags(
        scaling: Option<&str>,
        rope_scale: Option<f32>,
        freq_scale: Option<f32>,
        freq_base: Option<f32>,
    ) -> Result<Option<Self>, String> {
        Self::from_flags_with_yarn(
            scaling,
            rope_scale,
            freq_scale,
            freq_base,
            YarnKnobs::default(),
        )
    }

    /// [`Self::from_flags`] with the five b10621 `--yarn-*` knobs (#1472).
    pub fn from_flags_with_yarn(
        scaling: Option<&str>,
        rope_scale: Option<f32>,
        freq_scale: Option<f32>,
        freq_base: Option<f32>,
        yarn: YarnKnobs,
    ) -> Result<Option<Self>, String> {
        let scaling_type = scaling.map(RopeScalingTypeOverride::parse).transpose()?;

        yarn.validate()?;

        for (flag, value) in [
            ("--rope-scale", rope_scale),
            ("--rope-freq-scale", freq_scale),
            ("--rope-freq-base", freq_base),
        ] {
            if let Some(value) = value
                && !(value.is_finite() && value > 0.0)
            {
                return Err(format!(
                    "{flag} {value} is not a positive finite number; a non-positive or \
                     non-finite RoPE factor produces infinite or NaN frequencies and every \
                     logit becomes NaN without an error"
                ));
            }
        }

        // `--rope-scale N` is `rope_freq_scale = 1/N`; `--rope-freq-scale N` is
        // `rope_freq_scale = N`. Both are accepted, and both spellings of the
        // same request agree.
        let from_rope_scale = rope_scale.map(|n| 1.0 / n);
        let resolved_freq_scale = match (from_rope_scale, freq_scale) {
            (Some(a), Some(b)) if (a - b).abs() > f32::EPSILON * a.abs().max(b.abs()).max(1.0) => {
                return Err(format!(
                    "--rope-scale {} and --rope-freq-scale {} disagree: they are two spellings \
                     of the same setting and must be reciprocals (--rope-scale N means \
                     --rope-freq-scale 1/N). Pass one of them.",
                    rope_scale.unwrap_or_default(),
                    b
                ));
            }
            (Some(a), _) => Some(a),
            (None, b) => b,
        };

        if scaling_type.is_none()
            && resolved_freq_scale.is_none()
            && freq_base.is_none()
            && !yarn.any_set()
        {
            return Ok(None);
        }

        Ok(Some(Self {
            scaling_type,
            freq_base,
            freq_scale: resolved_freq_scale,
            yarn,
        }))
    }

    /// The scheme this override forces, if it forces one.
    pub fn scaling_type(&self) -> Option<RopeScalingTypeOverride> {
        self.scaling_type
    }

    /// The `rope_theta` replacement, if any.
    pub fn freq_base(&self) -> Option<f32> {
        self.freq_base
    }

    /// The `rope_freq_scale` replacement, if any.
    pub fn freq_scale(&self) -> Option<f32> {
        self.freq_scale
    }

    /// The five `--yarn-*` knobs carried by this override (#1472).
    pub fn yarn_knobs(&self) -> &YarnKnobs {
        &self.yarn
    }

    /// A one-line description for the startup banner and error messages.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(kind) = self.scaling_type {
            parts.push(format!("--rope-scaling {}", kind.as_str()));
        }
        if let Some(scale) = self.freq_scale {
            parts.push(format!("--rope-freq-scale {scale}"));
        }
        if let Some(base) = self.freq_base {
            parts.push(format!("--rope-freq-base {base}"));
        }
        if let Some(n) = self.yarn.orig_ctx {
            parts.push(format!("--yarn-orig-ctx {n}"));
        }
        if let Some(v) = self.yarn.ext_factor {
            parts.push(format!("--yarn-ext-factor {v}"));
        }
        if let Some(v) = self.yarn.attn_factor {
            parts.push(format!("--yarn-attn-factor {v}"));
        }
        if let Some(v) = self.yarn.beta_slow {
            parts.push(format!("--yarn-beta-slow {v}"));
        }
        if let Some(v) = self.yarn.beta_fast {
            parts.push(format!("--yarn-beta-fast {v}"));
        }
        parts.join(" ")
    }

    /// Rewrite a checkpoint's `rope_scaling` block into the one this override
    /// asks for.
    ///
    /// Pure, so the whole decision table is testable without touching the
    /// process-wide slot: the caller supplies the block the checkpoint declared
    /// and gets back the block the model should be built from.
    ///
    /// | requested scheme | `--rope-freq-scale` | result |
    /// |---|---|---|
    /// | `none` | ignored | no block: the plain table |
    /// | `linear` | given | `{"rope_type": "linear", "factor": 1/scale}` |
    /// | `linear` | absent | the checkpoint's own factor, or `1.0` |
    /// | not given | given | as `linear`, matching b10621's "defaults to linear unless specified by the model" |
    /// | not given | absent | the checkpoint's block, untouched |
    ///
    /// The one case with no defensible answer is a scale applied on top of a
    /// scheme that is neither linear nor YaRN, `llama3` above all: b10621
    /// would multiply its own `rope_freq_scale` into an NTK or banded
    /// rotation, mlxcel's `llama3` table has no such multiplier, and composing
    /// them by dropping one half silently changes the rotation. (A declared
    /// YaRN block composes: the scale replaces its factor, which is where
    /// b10621 writes `rope_freq_scale` for a YaRN model.) That returns `Err`,
    /// which
    /// [`crate::models::rope_utils::RopeScalingKind::resolve`] turns into a
    /// recorded rejection and startup turns into a refusal to serve.
    pub fn apply_to_spec(
        &self,
        declared: Option<&RopeScalingSpec>,
    ) -> Result<Option<RopeScalingSpec>, String> {
        let declared_type = declared.map(|s| s.rope_type().to_string());

        match self.scaling_type {
            Some(RopeScalingTypeOverride::None) => Ok(None),
            Some(RopeScalingTypeOverride::Yarn) => {
                // Force the YaRN table (#1472). The factor is
                // `1 / --rope-freq-scale` when a scale was requested,
                // otherwise the checkpoint's own factor when its block already
                // declares YaRN (whose band and mscale keys are carried
                // through), otherwise `1.0`, which is b10621's resolution: an
                // unspecified `rope_freq_scale` keeps the model's own.
                let mut spec = if declared_type.as_deref() == Some("yarn") {
                    declared.cloned().unwrap_or_default()
                } else {
                    RopeScalingSpec::default()
                };
                spec.rope_type = Some("yarn".to_string());
                if let Some(freq_scale) = self.freq_scale {
                    spec.factor = Some(1.0 / freq_scale);
                } else if spec.factor.is_none() {
                    spec.factor = Some(1.0);
                }
                Ok(Some(spec))
            }
            Some(RopeScalingTypeOverride::Linear) => Ok(Some(RopeScalingSpec {
                rope_type: Some("linear".to_string()),
                factor: Some(self.linear_factor(declared)),
                ..RopeScalingSpec::default()
            })),
            None => {
                let Some(freq_scale) = self.freq_scale else {
                    // Base-only override: the block is the checkpoint's.
                    return Ok(declared.cloned());
                };
                match declared_type.as_deref() {
                    None | Some("default") | Some("linear") => Ok(Some(RopeScalingSpec {
                        rope_type: Some("linear".to_string()),
                        factor: Some(1.0 / freq_scale),
                        ..RopeScalingSpec::default()
                    })),
                    // A scale over a checkpoint-declared YaRN rotation feeds
                    // the rotation's own factor, which is where b10621 writes
                    // `rope_freq_scale` for a YaRN model.
                    Some("yarn") => {
                        let mut spec = declared.cloned().unwrap_or_default();
                        spec.factor = Some(1.0 / freq_scale);
                        Ok(Some(spec))
                    }
                    Some(other) => Err(format!(
                        "the checkpoint declares rope_scaling type \"{other}\" and the request \
                         asks for a frequency scale of {freq_scale} without naming a scheme. \
                         llama-server would multiply the scale into that scheme's rotation; \
                         mlxcel's \"{other}\" table has no such multiplier, so composing them \
                         would change the rotation in a way neither side describes. Pass \
                         --rope-scaling linear to replace the scheme, or --rope-scaling none \
                         to drop it."
                    )),
                }
            }
        }
    }

    /// The `factor` a forced `linear` scheme should use.
    ///
    /// `--rope-freq-scale` when given (as its reciprocal), then the
    /// checkpoint's own factor when the block already named one, then `1.0`.
    /// The last case is a deliberate no-op rather than an error: `--rope-scaling
    /// linear` alone is how an operator asks for "linear, whatever the model
    /// says", and b10621 resolves it the same way.
    fn linear_factor(&self, declared: Option<&RopeScalingSpec>) -> f32 {
        if let Some(freq_scale) = self.freq_scale {
            return 1.0 / freq_scale;
        }
        declared
            .and_then(|s| s.factor)
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or(1.0)
    }

    /// The RoPE base after this override, given the checkpoint's own.
    pub fn apply_to_base(&self, declared: f32) -> f32 {
        self.freq_base.unwrap_or(declared)
    }
}

/// The process-wide override, installed once before the first model load.
static INSTALLED: OnceLock<Option<RopeRuntimeOverride>> = OnceLock::new();

/// How many times a seam has consumed the installed override.
static APPLICATIONS: AtomicUsize = AtomicUsize::new(0);

/// The first rejection a seam recorded, if any.
static REJECTION: OnceLock<String> = OnceLock::new();

/// Install the process-wide override before any model is loaded.
///
/// Returns `Err` when an override was already installed with a different value,
/// which can only mean two startup paths raced or a caller installed after the
/// first load. Installing the same value twice is accepted so a re-entrant
/// startup (the server's model-switch path reloads through the same code) is
/// not an error.
pub fn install(override_value: Option<RopeRuntimeOverride>) -> Result<(), String> {
    match INSTALLED.set(override_value) {
        Ok(()) => Ok(()),
        Err(_) if INSTALLED.get() == Some(&override_value) => Ok(()),
        Err(_) => Err(format!(
            "a RoPE runtime override is already installed ({:?}); it must be set once, before \
             the first model load",
            INSTALLED.get()
        )),
    }
}

/// The installed override, if the process has one.
pub fn installed() -> Option<&'static RopeRuntimeOverride> {
    INSTALLED.get().and_then(|slot| slot.as_ref())
}

/// The installed `--yarn-*` knobs, when an override carrying any is installed.
///
/// Consulted by the YaRN table builder in [`crate::models::rope_utils`], so
/// the knobs reach a YaRN rotation whether the operator forced it
/// (`--rope-scaling yarn`) or the checkpoint's own block declared it, which is
/// b10621's behavior. With no YaRN rotation in force nothing ever reads them,
/// which is also b10621's behavior.
pub(crate) fn installed_yarn_knobs() -> Option<&'static YarnKnobs> {
    installed().map(|over| &over.yarn).filter(|k| k.any_set())
}

/// How many RoPE seams have consumed the installed override.
///
/// Zero after a model load, with an override installed, means the model's
/// family does not route through [`crate::models::rope_utils`] and the override
/// had no effect. See the module documentation.
pub fn applications() -> usize {
    APPLICATIONS.load(Ordering::Relaxed)
}

/// Record that a seam consumed the override.
pub(crate) fn note_application() {
    APPLICATIONS.fetch_add(1, Ordering::Relaxed);
}

/// Record why a seam could not apply the override.
///
/// Kept rather than returned because the seams sit inside infallible config
/// readers ([`crate::models::rope_utils::RopeScalingKind::resolve`] cannot
/// return a `Result` without a load error for every checkpoint that declares an
/// unimplemented scheme; see its own documentation). Startup reads this after
/// the load and refuses to serve.
pub(crate) fn note_rejection(reason: String) {
    let _ = REJECTION.set(reason);
}

/// The first rejection a seam recorded since process start.
pub fn rejection() -> Option<&'static str> {
    REJECTION.get().map(String::as_str)
}

/// Confirm that the override an operator asked for actually reached the model.
///
/// Called once, on the worker thread, immediately after the checkpoint loads.
/// Three outcomes:
///
/// - No override installed: `Ok(())`, and nothing was ever consulted.
/// - Override installed and a seam applied it: `Ok(())`.
/// - Override installed and either no seam saw it, or a seam saw it and could
///   not serve it: `Err`, naming the checkpoint and what was asked for.
///
/// The error is fatal to serving on purpose. A server that starts, answers
/// requests, and rotates with the checkpoint's own frequencies while the
/// operator passed `--rope-freq-base` is the exact silent-acceptance failure
/// epic #1431 exists to remove, and it is invisible from the outside: the
/// output is fluent either way.
pub fn verify_applied(model_label: &str) -> Result<(), String> {
    let Some(over) = installed() else {
        return Ok(());
    };
    if let Some(reason) = rejection() {
        return Err(format!(
            "{model_label}: {} could not be applied: {reason}",
            over.describe()
        ));
    }
    if applications() == 0 {
        return Err(format!(
            "{model_label}: {} was accepted on the command line but reached no RoPE code path,              so the model would rotate with the frequencies its own config.json declares. This              architecture computes its rotary frequencies outside the shared              `models::rope_utils` path, where the override is applied; the families that do              route through it are Llama 3.x (and through it Qwen2 / Qwen2.5, Helium, the mllama              text decoder, and every VLM whose text backbone is one of those), Qwen3,              Qwen3-MoE, Apertus, Gemma 3, and InternLM2 / InternLM3. Drop the flag to serve this              checkpoint with its own rotation.",
            over.describe()
        ));
    }
    Ok(())
}

/// Apply the installed override to a checkpoint's block, counting the seam.
///
/// Returns the block the model should be built from. With no override
/// installed this is the checkpoint's own block and nothing is counted, so the
/// hot path for every ordinary load is one relaxed atomic load.
pub(crate) fn resolve_spec(declared: Option<&RopeScalingSpec>) -> Option<RopeScalingSpec> {
    let Some(over) = installed() else {
        return declared.cloned();
    };
    note_application();
    match over.apply_to_spec(declared) {
        Ok(spec) => spec,
        Err(reason) => {
            note_rejection(reason);
            declared.cloned()
        }
    }
}

/// Apply the installed override to a checkpoint's `rope_theta`, counting the
/// seam.
///
/// Split from [`resolve_spec`] because a model stores its base and its
/// frequency table in different places: MLX's `fast_rope` takes a base and
/// `fast_rope_with_freqs` takes a table, never both, so a family keeps the base
/// on the attention block and the table on the side. Both have to move together
/// or a `--rope-freq-base` override would reach the table and not the plain
/// rotation.
// Used by: Llama 3.x text (and through it Qwen2 / Qwen2.5, Helium, the mllama
// text decoder, every VLM whose text backbone is Llama3Model or Qwen2Model),
// Qwen3, Qwen3-MoE, Apertus, Gemma 3, InternLM3 / InternLM2 via
// dynamic_ntk_rope, and rope_utils' own table builder.
pub(crate) fn resolve_base(declared: f32) -> f32 {
    let Some(over) = installed() else {
        return declared;
    };
    note_application();
    over.apply_to_base(declared)
}

#[cfg(test)]
#[path = "rope_overrides_tests.rs"]
mod tests;
