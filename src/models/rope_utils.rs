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

//! Shared reader for the `rope_scaling` block, and the frequency tables it
//! selects.
//!
//! Port of
//! [`mlx_lm/models/rope_utils.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py)
//! (`initialize_rope` and `Llama3RoPE`), which is the single place upstream
//! turns a `config.json` `rope_scaling` block into a RoPE module. Two families
//! in this tree need the same decision, and until #1355 only one of them made
//! it: Apertus computed the `llama3` table inline while the shared Llama
//! attention parsed the block and dropped it.
//!
//! Used by: Llama 3.x text (`llama3`, and through it Qwen2 / Qwen2.5, Helium,
//! the `mllama` text decoder, every VLM whose text backbone is `Llama3Model` or
//! `Qwen2Model`, the `llama` / `mistral` pipeline stage executors and the
//! tensor-parallel Llama runtime), Apertus.

use mlxcel_core::{MlxArray, UniquePtr};
use serde::{Deserialize, Deserializer};

/// The `rope_scaling` fields this module reads.
///
/// # Why this is not a plain `#[derive(Deserialize)]` struct
///
/// The key holding the scheme name is spelled two ways in the wild. Llama 3.x,
/// Apertus and every recent HuggingFace conversion write `rope_type`; older
/// configs (and some conversions that carry both) write `type`. A derived
/// struct can name only one of them, and naming one with
/// `#[serde(rename = "type", alias = "rope_type")]` turns a config that carries
/// *both* into a hard `duplicate field` parse error, which would stop
/// checkpoints that load today (`models/internvl3-1b` spells both). Reading the
/// block as a JSON map first sidesteps that: a map keeps the last value for a
/// repeated key instead of failing, and unknown keys are ignored the way serde's
/// default already ignores them.
///
/// The lookup order is `type` then `rope_type`, matching upstream's
/// `scaling_config.get("type") or scaling_config.get("rope_type", "default")`.
/// A JSON `null` under either key reads as absent, so `{"type": null,
/// "rope_type": "llama3"}` resolves to `llama3` exactly as it does upstream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RopeScalingSpec {
    /// The scheme name, from `type` or `rope_type`. `None` means the block
    /// named none, which upstream treats as `"default"`.
    pub rope_type: Option<String>,
    pub factor: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<f32>,
}

impl RopeScalingSpec {
    /// Read the block through a key lookup.
    ///
    /// Generic over the lookup rather than over the container so both shapes in
    /// this tree work unchanged: `serde_json::Value::get` for a parsed block and
    /// `HashMap<String, Value>::get` for a config that deserializes the block
    /// into a map (Apertus).
    pub fn from_lookup<'a, F>(lookup: F) -> Self
    where
        F: Fn(&str) -> Option<&'a serde_json::Value>,
    {
        let string_at = |key: &str| lookup(key).and_then(|v| v.as_str()).map(|s| s.to_string());
        let f32_at = |key: &str| lookup(key).and_then(|v| v.as_f64()).map(|v| v as f32);

        Self {
            rope_type: string_at("type").or_else(|| string_at("rope_type")),
            factor: f32_at("factor"),
            low_freq_factor: f32_at("low_freq_factor"),
            high_freq_factor: f32_at("high_freq_factor"),
            original_max_position_embeddings: f32_at("original_max_position_embeddings"),
        }
    }

    /// The scheme name, defaulting to `"default"` the way upstream does.
    pub fn rope_type(&self) -> &str {
        self.rope_type.as_deref().unwrap_or("default")
    }
}

impl<'de> Deserialize<'de> for RopeScalingSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let block = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
        Ok(Self::from_lookup(|key| block.get(key)))
    }
}

/// The frequency table a `rope_scaling` block selects.
///
/// Mirrors the three branches of upstream's `initialize_rope`, minus the ones
/// this path does not implement (see [`RopeScalingKind::resolve`]).
pub enum RopeScalingKind {
    /// Plain `base^(2i/d)` frequencies at position scale `1.0`. Selected by an
    /// absent block, `"default"`, and any scheme this path cannot serve.
    Default,
    /// Plain frequencies with positions divided by `factor`. MLX's `fast::rope`
    /// multiplies the position by `scale`, so `scale = 1 / factor`.
    Linear { scale: f32 },
    /// The banded `llama3` table, precomputed as a float32 `[dims / 2]` array
    /// and handed to `fast_rope_with_freqs`. `base` is not used once a table is
    /// supplied.
    Llama3 { freqs: UniquePtr<MlxArray> },
}

impl RopeScalingKind {
    /// Resolve a parsed block into a frequency table.
    ///
    /// # Why an unimplemented scheme is a warning, not an error
    ///
    /// #1355 asked for a named load error on any scheme this path does not
    /// implement. That cannot ship as written: the shared Llama args are also
    /// what several VLM loaders parse a `text_config` into, and
    /// `models/internvl3-1b` declares `"rope_type": "dynamic"` in exactly that
    /// position. It loads today (the block is parsed and ignored), so a load
    /// error would take a working checkpoint offline to fix a scheme that is not
    /// implemented on either side of the change.
    ///
    /// The failure a load error was meant to prevent is silence, and a warning
    /// removes the silence without removing the model. The scheme stays on the
    /// unscaled table, which is what it decoded with before this change, and the
    /// operator is told which model and which scheme. Wire a scheme in properly
    /// and the warning stops on its own.
    pub fn resolve(
        spec: Option<&RopeScalingSpec>,
        dims: usize,
        base: f32,
        model_label: &str,
    ) -> Self {
        let Some(spec) = spec else {
            return Self::Default;
        };

        match spec.rope_type() {
            "default" => Self::Default,
            "linear" => {
                // Upstream indexes `scaling_config["factor"]` here, so a
                // `linear` block without one is malformed. Falling back to 1.0
                // keeps that config loading with the graph it already had.
                let factor = spec.factor.unwrap_or(1.0);
                if is_usable_scalar(factor) {
                    Self::Linear {
                        scale: 1.0 / factor,
                    }
                } else {
                    report_unusable_rope_scaling_once(
                        model_label,
                        "linear",
                        &format!("factor {factor} is not a positive finite number"),
                    );
                    Self::Default
                }
            }
            "llama3" => match llama3_rope_freqs(spec, dims, base) {
                Ok(freqs) => Self::Llama3 { freqs },
                Err(reason) => {
                    report_unusable_rope_scaling_once(model_label, "llama3", &reason);
                    Self::Default
                }
            },
            other => {
                report_unusable_rope_scaling_once(
                    model_label,
                    other,
                    "this scheme is not implemented on the shared Llama RoPE path",
                );
                Self::Default
            }
        }
    }

    /// The position scale to hand `fast_rope`. `1.0` unless `linear`.
    pub fn scale(&self) -> f32 {
        match self {
            Self::Default | Self::Llama3 { .. } => 1.0,
            Self::Linear { scale } => *scale,
        }
    }

    /// The precomputed frequency table, when the scheme has one.
    pub fn freqs(&self) -> Option<&MlxArray> {
        match self {
            Self::Llama3 { freqs } => Some(freqs),
            _ => None,
        }
    }

    /// A per-layer handle on the same table.
    ///
    /// The table is computed once per model and duplicated into every attention
    /// block, so a 32-layer model runs one `powf` loop rather than 32. The
    /// duplicate is an MLX `copy` of an already-evaluated `[dims / 2]` float32
    /// array, which is a few hundred bytes; what it buys is that each block owns
    /// its handle and the blocks stay independently constructible.
    ///
    /// `mlxcel_core::copy` is `mlx::core::copy`, a lazy `Copy` graph node over
    /// an evaluated source rather than a refcounted share of one buffer, so each
    /// block's handle materialises on that block's first forward. That is a
    /// one-off `[dims / 2]` element copy per layer, not a re-run of the band
    /// arithmetic, and the source stays evaluated for all of them.
    pub fn duplicate(&self) -> Self {
        match self {
            Self::Default => Self::Default,
            Self::Linear { scale } => Self::Linear { scale: *scale },
            Self::Llama3 { freqs } => Self::Llama3 {
                freqs: mlxcel_core::copy(freqs),
            },
        }
    }
}

/// The `llama3` frequency table, as a float32 `[dims / 2]` array.
///
/// Port of `Llama3RoPE.__init__`
/// ([`mlx_lm/models/rope_utils.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py)).
/// With `d = dims`, `F = factor`, `lf = low_freq_factor`, `hf =
/// high_freq_factor` and `L = original_max_position_embeddings`, for each pair
/// `i < d / 2`:
///
/// ```text
/// freq    = base ^ (2i / d)          // the value fast::rope divides the position by
/// wavelen = 2 * pi * freq
/// if wavelen > L / lf:                          freq * F        // low band
/// else if L / hf < wavelen < L / lf:            freq / ((1 - s) / F + s)
/// else:                                         freq            // high band
/// where s = (L / wavelen - lf) / (hf - lf)
/// ```
///
/// The middle branch reproduces upstream's `is_medium_freq` mask exactly,
/// including its strict `wavelen < low_freq_wavelen`: a wavelength that lands
/// exactly on `L / lf` is in neither mask and keeps its unscaled frequency.
///
/// The arithmetic stays in f32 because upstream's does (`mx.arange` is float32
/// and the whole expression evaluates in float32), so the table matches the
/// reference to the same rounding rather than to a more accurate one. It is
/// scalar rather than vectorized, so it can differ from a table built by
/// mlx-lm's `mx.power` by a couple of ULP; assert closeness against a
/// Python-generated table, never bit-equality.
///
/// `Err` carries the reason the block cannot build a table, for the caller to
/// report. See [`Llama3Params::from_spec`] for what is screened and why.
pub fn llama3_rope_freqs(
    spec: &RopeScalingSpec,
    dims: usize,
    base: f32,
) -> Result<UniquePtr<MlxArray>, String> {
    let Llama3Params {
        factor,
        low_freq_factor,
        high_freq_factor,
        old_context_len,
    } = Llama3Params::from_spec(spec)?;

    let low_freq_wavelen = old_context_len / low_freq_factor;
    let high_freq_wavelen = old_context_len / high_freq_factor;

    let half_dims = dims / 2;
    let mut freq_vals = Vec::with_capacity(half_dims);
    for i in 0..half_dims {
        let exp = (2 * i) as f32 / dims as f32;
        let freq = base.powf(exp);
        let wavelen = 2.0 * std::f32::consts::PI * freq;

        let adjusted = if wavelen > low_freq_wavelen {
            // Low frequency (long wavelength): scale by factor.
            freq * factor
        } else if wavelen > high_freq_wavelen && wavelen < low_freq_wavelen {
            // Medium frequency: smooth interpolation between the two bands.
            let smooth = (old_context_len / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            freq / ((1.0 - smooth) / factor + smooth)
        } else {
            // High frequency (short wavelength): unchanged.
            freq
        };
        freq_vals.push(adjusted);
    }

    let freqs = mlxcel_core::from_slice_f32(&freq_vals, &[half_dims as i32]);
    // Evaluated at load so the band arithmetic is a materialised buffer rather
    // than a graph the first forward has to run. Each layer's
    // [`RopeScalingKind::duplicate`] is a `copy` node over this evaluated
    // source, so what a block defers is one element copy, not this loop.
    mlxcel_core::eval(&freqs);
    Ok(freqs)
}

/// Whether a `rope_scaling` scalar can be used as a multiplier or a divisor.
///
/// One predicate for every scalar the two implemented schemes read, so the
/// `linear` and `llama3` arms cannot screen to different standards.
fn is_usable_scalar(value: f32) -> bool {
    value > 0.0 && value.is_finite()
}

/// The four `llama3` scalars, screened for values the band arithmetic can use.
struct Llama3Params {
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    old_context_len: f32,
}

impl Llama3Params {
    /// Read the four scalars, or name the one that is unusable.
    ///
    /// # Why this screens at all
    ///
    /// Upstream reads these straight out of the block and hands the result to
    /// `mx.where`, so nothing there rejects a value. Feeding an unusable one
    /// through the same arithmetic here produces a table that loads, allocates
    /// and evaluates without a single error, and then decodes wrongly:
    ///
    /// - `factor` zero sends `freq * factor` to `0.0` for every pair in the low
    ///   band (35 of 64 entries at Llama 3.1 geometry). `fast::rope` divides the
    ///   position by the table entry, `reciprocal(0.0)` is `inf`, and every
    ///   logit is `NaN` for every token. Nothing panics on the way: the sampler
    ///   compares with `partial_cmp(..).unwrap_or(Equal)`.
    /// - `factor` absent, or a JSON string rather than a number, would default
    ///   to `1.0` and build a table bit-identical to the plain one. That is the
    ///   silent no-op #1355 exists to remove, so it is reported instead.
    ///   Upstream indexes `scaling_config["factor"]` with no default, which is
    ///   the same judgement expressed as a `KeyError`.
    /// - `factor` beyond f32 range (a JSON `1e39`) saturates to `inf` through
    ///   the `as` cast and gives 29 `inf` entries, which is an identity rotation
    ///   on that band.
    /// - `factor` negative reverses the rotation direction on the low band.
    ///
    /// A screened-out block falls back to the plain table, which is the graph
    /// the checkpoint had before the block was read at all, and says so once on
    /// stderr.
    ///
    /// `low_freq_factor == high_freq_factor` is *not* screened out:
    /// `Llama-4-Scout-17B-16E` ships exactly that (both `1.0`), and it is
    /// well defined, because the medium band `wavelen > L / hf && wavelen < L /
    /// lf` is then empty and the interpolation whose denominator would be zero
    /// is never evaluated. Upstream reaches the same table through `mx.where`
    /// discarding the unselected branch.
    fn from_spec(spec: &RopeScalingSpec) -> Result<Self, String> {
        let Some(factor) = spec.factor else {
            return Err("it names no numeric factor".to_string());
        };
        let params = Self {
            factor,
            low_freq_factor: spec.low_freq_factor.unwrap_or(1.0),
            high_freq_factor: spec.high_freq_factor.unwrap_or(4.0),
            old_context_len: spec.original_max_position_embeddings.unwrap_or(8192.0),
        };

        for (name, value) in [
            ("factor", params.factor),
            ("low_freq_factor", params.low_freq_factor),
            ("high_freq_factor", params.high_freq_factor),
            ("original_max_position_embeddings", params.old_context_len),
        ] {
            if !is_usable_scalar(value) {
                return Err(format!("{name} {value} is not a positive finite number"));
            }
        }

        Ok(params)
    }
}

/// Report, at most once per `(model, scheme)` pair, that a `rope_scaling` block
/// was read and could not be honored.
///
/// `eprintln!` rather than `tracing::warn!` for the same reason
/// `report_fused_rope_bypass_once` in `llama3.rs` uses it: only the server
/// installs a `tracing` subscriber, so a `warn!` is a no-op in the `mlxcel` CLI,
/// which is where a model is most often loaded by hand.
///
/// Deduplicated on the pair rather than through a `Once` because a process can
/// load more than one model (the server's model switching, the pipeline stage
/// executors, the tensor-parallel ranks), and because a per-layer loader would
/// otherwise print the same line once per decoder layer.
///
/// `model_label` is the checkpoint the loader read when it knows the directory
/// (see `llama3::ModelArgs::set_checkpoint_label`) and the architecture from
/// `model_type` otherwise. Keying on the checkpoint is what makes the dedup
/// argument above hold: two different `qwen2` checkpoints that both declare an
/// unimplemented scheme each get a warning, where keying on the architecture
/// would report the first and let the second degrade in silence.
fn report_unusable_rope_scaling_once(model_label: &str, rope_type: &str, reason: &str) {
    use std::collections::BTreeSet;
    use std::sync::{Mutex, OnceLock};

    static REPORTED: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();

    let model_label = printable_label(model_label);
    let rope_type = printable_label(rope_type);

    let key = format!("{model_label}\u{0}{rope_type}");
    let reported = REPORTED.get_or_init(|| Mutex::new(BTreeSet::new()));
    // A poisoned lock here means another thread panicked while printing; the set
    // is still a valid set, and losing the warning is worse than reusing it.
    let mut reported = reported.lock().unwrap_or_else(|err| err.into_inner());
    if !reported.insert(key) {
        return;
    }

    eprintln!(
        "warning: {model_label} declares rope_scaling type \"{rope_type}\", but {reason}. \
         The model is loaded with unscaled RoPE frequencies, which is the behavior it had \
         before this block was read at all. Short prompts are unaffected; long-context \
         quality past the checkpoint's original_max_position_embeddings may be."
    );
}

/// Make checkpoint-controlled text safe to put on a line of stderr, or into an
/// error message.
///
/// Both halves of the warning come out of the checkpoint: `model_type` (or a
/// directory name) and the `rope_type` string. Neither is bounded or validated
/// anywhere upstream of here, so a config can embed a newline and forge a whole
/// log line after it, or run to megabytes and bury the message. Control
/// characters become `U+FFFD` and the label is truncated, which also bounds the
/// dedup set's keys.
///
/// Used by: this module's warning, and `gemma3::ModelArgs::global_rope_scale`,
/// whose load error names the same checkpoint-controlled `rope_type` string.
pub(crate) fn printable_label(label: &str) -> String {
    const MAX_CHARS: usize = 64;

    let cleaned: String = label
        .chars()
        .take(MAX_CHARS)
        .map(|c| if c.is_control() { '\u{fffd}' } else { c })
        .collect();

    if label.chars().nth(MAX_CHARS).is_some() {
        format!("{cleaned}...")
    } else {
        cleaned
    }
}

#[cfg(test)]
#[path = "rope_utils_tests.rs"]
mod tests;
