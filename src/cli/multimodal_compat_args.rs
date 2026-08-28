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

//! llama-server b10621 multimodal projector and media options (issue #1451).
//!
//! b10621 is a GGUF runtime whose multimodal support is a *separate artifact*:
//! `libmtmd` loads a projector file (`--mmproj`, `--mmproj-url`) alongside the
//! language model, places it on a device (`--mmproj-device`, `--mmproj-offload`)
//! and can be told not to load one at all (`--no-mmproj`). mlxcel loads an
//! integrated MLX VLM checkpoint: the vision tower, the audio tower and the
//! multimodal projector are tensors inside the same SafeTensors snapshot as the
//! language model, resolved by the same `-m` reference.
//!
//! That difference decides every classification in this module.
//!
//! # The rule
//!
//! - A flag that can only name a **separate GGUF projector artifact**
//!   (`--mmproj`, `--mmproj-url`) is refused at startup. Before #1451 `--mmproj`
//!   parsed into an ignored field, so a command line that asked mlxcel to attach
//!   a GGUF projector started a server that silently was not using it.
//! - A flag that only describes **where the projector runs** (`--mmproj-offload`,
//!   `--mmproj-device`) is inert at the value that asks for nothing and refused
//!   otherwise: mlxcel runs the whole checkpoint, vision tower included, on the
//!   one MLX device selected by `MLXCEL_DEVICE`, and has no host projector path
//!   to move it to.
//! - `--mmproj-auto` / `--no-mmproj` is **honored**, because its observable
//!   effect has a translation: upstream without a projector answers a media
//!   request with `image input is not supported` and mlxcel can decline the same
//!   requests through the same wording. See [`MediaAdmission`].
//! - `--image-min-tokens` / `--image-max-tokens` are **honored** through
//!   [`crate::vision::image_token_overrides`], which reproduces upstream's own
//!   `clip_hparams::set_limit_image_tokens` arithmetic
//!   (`pixels = tokens * patch_size^2 * merge_size^2`) against the dynamic
//!   resolution processors that express the same bounds as `min_pixels` /
//!   `max_pixels`. An architecture that has no such bound refuses to serve
//!   rather than ignoring the flag.
//! - `--mtmd-batch-max-tokens` is inert at b10621's own default and refused
//!   otherwise: mlxcel encodes each image in one vision-tower forward and has no
//!   image-token batch to bound.
//! - `--media-path` is implemented, with a confined root; see
//!   [`crate::server::media_root`].
//!
//! Everything except `--media-path` is `hide = true`. Rendering the projector
//! family in `--help` would imply that a GGUF `mmproj` file can be attached to
//! an MLX checkpoint, which is the one thing this module exists to deny.
//!
//! Used by: mlxcel serve, mlxcel-server.
//!
//! Upstream reference:
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>,
//! <https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/mtmd/clip-model.h>

use std::path::{Path, PathBuf};

use clap::Args;

use super::ggml_compat_args::{env_bool_pair, env_flag};

/// b10621's own `--mtmd-batch-max-tokens` default.
///
/// The only value mlxcel accepts: it names an image-token batch width, and
/// mlxcel encodes every image in a single vision-tower forward.
pub const B10621_MTMD_BATCH_MAX_TOKENS: i64 = 1024;

/// Whether the server admits multimodal content parts at the request boundary.
///
/// This is the mlxcel translation of b10621's `--mmproj-auto` /
/// `--no-mmproj`. Upstream decides it by whether a projector was loaded:
/// `--no-mmproj` leaves `mctx` null and every image, audio or video part in a
/// request is refused with `... input is not supported - hint: if this is
/// unexpected, you may need to provide the mmproj`. mlxcel cannot leave the
/// projector unloaded (it is inside the checkpoint), so it declines the same
/// requests with the same wording instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MediaAdmission {
    /// b10621's default: media parts are admitted when the checkpoint can
    /// consume them.
    #[default]
    Auto,
    /// `--no-mmproj` / `--no-mmproj-auto`: every media part is refused.
    Disabled,
}

impl MediaAdmission {
    /// True when the operator asked for media to be refused outright.
    #[must_use]
    pub const fn is_disabled(self) -> bool {
        matches!(self, Self::Disabled)
    }
}

/// llama-server b10621 multimodal projector / media options.
///
/// Flattened into both server binaries so the two surfaces cannot drift; see
/// `tests/cli_help_consistency.rs`.
#[derive(Args, Debug, Clone, Default)]
pub struct MultimodalCompatArgs {
    // ── projector artifact ──────────────────────────────────────────────
    /// b10621 `-mm` / `--mmproj`: path to a multimodal projector file.
    #[arg(
        long = "mmproj",
        env = "LLAMA_ARG_MMPROJ",
        value_name = "FILE",
        hide = true
    )]
    pub mmproj: Option<String>,

    /// b10621 `-mmu` / `--mmproj-url`: URL to a multimodal projector file.
    #[arg(
        long = "mmproj-url",
        env = "LLAMA_ARG_MMPROJ_URL",
        value_name = "URL",
        hide = true
    )]
    pub mmproj_url: Option<String>,

    /// b10621 `--mmproj-auto` (positive half of the pair, upstream's default).
    #[arg(long = "mmproj-auto", overrides_with_all = ["no_mmproj", "no_mmproj_auto"], hide = true)]
    pub mmproj_auto: bool,

    /// b10621 `--no-mmproj` (negative half of the pair).
    #[arg(long = "no-mmproj", overrides_with = "mmproj_auto", hide = true)]
    pub no_mmproj: bool,

    /// b10621 `--no-mmproj-auto` (second spelling of the negative half).
    #[arg(long = "no-mmproj-auto", overrides_with = "mmproj_auto", hide = true)]
    pub no_mmproj_auto: bool,

    // ── projector placement ─────────────────────────────────────────────
    /// b10621 `--mmproj-offload` (positive half, upstream's default).
    #[arg(
        long = "mmproj-offload",
        overrides_with = "no_mmproj_offload",
        hide = true
    )]
    pub mmproj_offload: bool,

    /// b10621 `--no-mmproj-offload` (negative half).
    #[arg(
        long = "no-mmproj-offload",
        overrides_with = "mmproj_offload",
        hide = true
    )]
    pub no_mmproj_offload: bool,

    /// b10621 `-mmdev` / `--mmproj-device`: device for the projector.
    ///
    /// Bound to `MTMD_BACKEND_DEVICE`, not a `LLAMA_ARG_*` variable, exactly as
    /// upstream binds it.
    #[arg(
        long = "mmproj-device",
        env = "MTMD_BACKEND_DEVICE",
        value_name = "DEVICE",
        hide = true
    )]
    pub mmproj_device: Option<String>,

    // ── image token budget ──────────────────────────────────────────────
    /// b10621 `--image-min-tokens`: minimum tokens one image may take.
    #[arg(
        long = "image-min-tokens",
        env = "LLAMA_ARG_IMAGE_MIN_TOKENS",
        value_name = "N",
        hide = true
    )]
    pub image_min_tokens: Option<String>,

    /// b10621 `--image-max-tokens`: maximum tokens one image may take.
    #[arg(
        long = "image-max-tokens",
        env = "LLAMA_ARG_IMAGE_MAX_TOKENS",
        value_name = "N",
        hide = true
    )]
    pub image_max_tokens: Option<String>,

    /// b10621 `--mtmd-batch-max-tokens`: image tokens per encode batch.
    #[arg(
        long = "mtmd-batch-max-tokens",
        env = "LLAMA_ARG_MTMD_BATCH_MAX_TOKENS",
        value_name = "N",
        hide = true
    )]
    pub mtmd_batch_max_tokens: Option<String>,

    // ── local media root ────────────────────────────────────────────────
    /// Directory that local `file://` media URLs are resolved against.
    ///
    /// Disabled by default: without it, a `file://` media URL in a request is
    /// refused. The value is canonicalized once at startup, and every resolved
    /// file must stay inside it, symlink targets included. llama-server spells
    /// this `--media-path` too.
    #[arg(
        long = "media-path",
        value_name = "PATH",
        help_heading = "Multimodal Options"
    )]
    pub media_path: Option<PathBuf>,
}

/// One rejected multimodal option, ready to render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultimodalCompatRejection {
    /// The option as the operator wrote it, for example `--mmproj`.
    pub option: &'static str,
    /// The requested value, or the flag name again for a value-less flag.
    pub value: String,
    /// Why mlxcel cannot reproduce b10621's behavior for that value.
    pub limitation: &'static str,
    /// What to use instead, or `None` when nothing corresponds.
    pub alternative: Option<&'static str>,
}

impl std::fmt::Display for MultimodalCompatRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} is not supported: {}",
            self.option, self.value, self.limitation
        )?;
        match self.alternative {
            Some(alternative) => write!(f, "\nUse instead: {alternative}"),
            None => write!(
                f,
                "\nThere is no mlxcel equivalent; drop the flag to start the server."
            ),
        }
    }
}

/// The operator-requested image token budget, in b10621's units.
///
/// Both halves are token counts, converted to the pixel bounds a dynamic
/// resolution processor expresses by
/// [`crate::vision::image_token_overrides`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageTokenBounds {
    /// `--image-min-tokens`, when set to a positive value.
    pub min_tokens: Option<u32>,
    /// `--image-max-tokens`, when set to a positive value.
    pub max_tokens: Option<u32>,
}

impl ImageTokenBounds {
    /// True when neither half was requested.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.min_tokens.is_none() && self.max_tokens.is_none()
    }
}

const PROJECTOR_LIMITATION: &str = "mlxcel loads an integrated MLX VLM checkpoint whose vision tower, audio tower and \
     multimodal projector are tensors inside the same SafeTensors snapshot as the language \
     model, so there is no separate GGUF projector artifact to attach";

const PROJECTOR_ALTERNATIVE: &str = "point `-m` / `--model` at an MLX VLM checkpoint (for example \
     mlx-community/Qwen2.5-VL-7B-Instruct-4bit); its projector loads with it and needs no flag";

const PLACEMENT_LIMITATION: &str = "mlxcel evaluates the whole checkpoint, vision and audio towers included, on the one MLX \
     device, and has no host projector path to move it to";

impl MultimodalCompatArgs {
    /// Reject every value whose b10621 meaning mlxcel cannot reproduce.
    ///
    /// Runs before the model reference resolves, so `--mmproj proj.gguf` is
    /// reported immediately rather than after a multi-gigabyte download. Every
    /// option in this group is model-independent: what a projector artifact is,
    /// where it runs, and how wide an encode batch is are all decided without
    /// reading `config.json`. Whether the *loaded* checkpoint honors
    /// `--image-min-tokens` is a separate, post-load check owned by
    /// [`crate::vision::image_token_overrides::verify_applied`].
    ///
    /// # Errors
    ///
    /// Returns the first rejection in a fixed order, so a command line carrying
    /// several unsupported values always reports the same one.
    pub fn ensure_inert(&self) -> Result<(), MultimodalCompatRejection> {
        self.rejection().map_or(Ok(()), Err)
    }

    /// The first unsupported request in this argument set, if any.
    #[must_use]
    pub fn rejection(&self) -> Option<MultimodalCompatRejection> {
        if let Some(value) = present(self.mmproj.as_deref()) {
            return Some(reject(
                "--mmproj",
                value,
                PROJECTOR_LIMITATION,
                Some(PROJECTOR_ALTERNATIVE),
            ));
        }
        if let Some(value) = present(self.mmproj_url.as_deref()) {
            return Some(reject(
                "--mmproj-url",
                value,
                PROJECTOR_LIMITATION,
                Some(PROJECTOR_ALTERNATIVE),
            ));
        }
        if self.no_mmproj_offload {
            return Some(reject(
                "--no-mmproj-offload",
                "--no-mmproj-offload",
                PLACEMENT_LIMITATION,
                None,
            ));
        }
        if let Some(value) = present(self.mmproj_device.as_deref()) {
            // Upstream's `none` means "do not offload the projector", the same
            // request as `--no-mmproj-offload`; anything else names a GGML
            // device, which mlxcel selects with its own environment variable.
            let alternative = if value.eq_ignore_ascii_case("none") {
                None
            } else {
                Some("`MLXCEL_DEVICE` selects the MLX device for the whole checkpoint")
            };
            return Some(reject(
                "--mmproj-device",
                value,
                PLACEMENT_LIMITATION,
                alternative,
            ));
        }
        if let Some(value) = present(self.mtmd_batch_max_tokens.as_deref())
            && !numeric_equals(value, B10621_MTMD_BATCH_MAX_TOKENS)
        {
            return Some(reject(
                "--mtmd-batch-max-tokens",
                value,
                "mlxcel encodes each image in a single vision-tower forward and has no \
                 image-token batch to bound, so only b10621's own default of 1024 is inert",
                Some("`--max-image-payload-size` and `--max-images` bound image work on mlxcel"),
            ));
        }
        self.image_token_bounds().err()
    }

    /// The requested image token budget, or the rejection its values earn.
    ///
    /// Upstream reads both with `std::stoi` and only treats a **positive**
    /// value as a custom bound (`custom_image_min_tokens > 0`), so `0` and a
    /// negative number mean "use the model's own value" rather than being an
    /// error. It then throws when the resulting `image_max_pixels` is below
    /// `image_min_pixels`, which for a single checkpoint reduces to
    /// `max_tokens < min_tokens`.
    ///
    /// # Errors
    ///
    /// Returns a rejection for a non-integer value or for a maximum below the
    /// minimum.
    pub fn image_token_bounds(&self) -> Result<ImageTokenBounds, MultimodalCompatRejection> {
        let min_tokens =
            parse_image_tokens("--image-min-tokens", self.image_min_tokens.as_deref())?;
        let max_tokens =
            parse_image_tokens("--image-max-tokens", self.image_max_tokens.as_deref())?;
        if let (Some(min), Some(max)) = (min_tokens, max_tokens)
            && max < min
        {
            return Err(reject_owned(
                "--image-max-tokens",
                &max.to_string(),
                "b10621 refuses a maximum image-token budget below the minimum \
                 (`image_max_pixels` below `image_min_pixels`)",
                Some("raise `--image-max-tokens` above `--image-min-tokens`"),
            ));
        }
        Ok(ImageTokenBounds {
            min_tokens,
            max_tokens,
        })
    }

    /// Whether media parts are admitted, per `--mmproj-auto` / `--no-mmproj`.
    #[must_use]
    pub fn media_admission(&self) -> MediaAdmission {
        if self.no_mmproj || self.no_mmproj_auto {
            MediaAdmission::Disabled
        } else {
            MediaAdmission::Auto
        }
    }

    /// Resolve `--media-path` into a canonical confined root.
    ///
    /// b10621 requires the value to be an existing directory and appends the
    /// platform separator so its own `media_path + file_path` concatenation
    /// lands inside it. mlxcel canonicalizes instead, because the canonical
    /// root is what every later containment check compares against; a root that
    /// is itself a symlink would otherwise make every contained path look like
    /// an escape.
    ///
    /// # Errors
    ///
    /// Returns the same class of startup error upstream throws: the value is
    /// not a directory, or it cannot be canonicalized.
    pub fn resolve_media_root(&self) -> Result<Option<PathBuf>, String> {
        let Some(raw) = self.media_path.as_deref() else {
            return Ok(None);
        };
        resolve_media_root_path(raw).map(Some)
    }

    /// Resolve the b10621 environment bindings this group cannot leave to clap.
    ///
    /// The two `--x` / `--no-x` pairs bind `LLAMA_ARG_MMPROJ_AUTO` and
    /// `LLAMA_ARG_MMPROJ_OFFLOAD`, which upstream reads through
    /// `parse_bool_value` plus a `LLAMA_ARG_NO_*` alias meaning false. clap's
    /// own boolish parser accepts a wider vocabulary and errors outside it, so
    /// the shared [`env_bool_pair`] reproduces b10621's rules instead. Every
    /// value-taking option in this group is bound by clap directly.
    ///
    /// # Errors
    ///
    /// Returns the variable name and its value when b10621's `parse_bool_value`
    /// would throw on it, which is what b10621 does.
    pub fn apply_env_bindings(&mut self) -> Result<(), (&'static str, String)> {
        for (var, positive, negative) in [
            (
                "LLAMA_ARG_MMPROJ_AUTO",
                &mut self.mmproj_auto,
                &mut self.no_mmproj,
            ),
            (
                "LLAMA_ARG_MMPROJ_OFFLOAD",
                &mut self.mmproj_offload,
                &mut self.no_mmproj_offload,
            ),
        ] {
            // An explicit command-line occurrence of either half wins.
            if *positive || *negative {
                continue;
            }
            match env_bool_pair(var) {
                None => {}
                Some(Ok(true)) => *positive = true,
                Some(Ok(false)) => *negative = true,
                Some(Err(raw)) => return Err((var, raw)),
            }
        }
        // `--no-mmproj-auto` has no environment binding of its own upstream;
        // `LLAMA_ARG_NO_MMPROJ_AUTO` is the alias `env_bool_pair` already
        // resolves into the negative half above. The value-less positive-only
        // read is kept for symmetry with the GGML group, where a bare
        // `LLAMA_ARG_MMPROJ_AUTO=on` must fire the flag.
        if env_flag("LLAMA_ARG_MMPROJ_AUTO") {
            self.mmproj_auto = true;
        }
        Ok(())
    }
}

/// Canonicalize a `--media-path` value into a confined root.
///
/// Separated from [`MultimodalCompatArgs::resolve_media_root`] so tests and the
/// server can reach the same rule with a bare path.
///
/// # Errors
///
/// Returns a startup diagnostic when the path does not exist, is not a
/// directory, or cannot be canonicalized.
pub fn resolve_media_root_path(raw: &Path) -> Result<PathBuf, String> {
    let canonical = std::fs::canonicalize(raw)
        .map_err(|err| format!("--media-path {}: {err}", raw.display()))?;
    if !canonical.is_dir() {
        return Err(format!("--media-path not a directory: {}", raw.display()));
    }
    Ok(canonical)
}

/// Parse one image-token bound the way upstream's `std::stoi` handler does.
fn parse_image_tokens(
    option: &'static str,
    value: Option<&str>,
) -> Result<Option<u32>, MultimodalCompatRejection> {
    let Some(value) = present(value) else {
        return Ok(None);
    };
    let Ok(parsed) = value.parse::<i64>() else {
        return Err(reject(
            option,
            value,
            "b10621 reads this option as an integer number of tokens",
            Some("pass a positive token count, or drop the flag to use the checkpoint's own bound"),
        ));
    };
    // Upstream only treats a positive value as a custom bound; `0` and negative
    // numbers leave `custom_image_*_tokens` at its `-1` sentinel.
    Ok(u32::try_from(parsed).ok().filter(|n| *n > 0))
}

/// `Some(trimmed)` when a value is present and not whitespace-only.
///
/// An environment-bound option routinely arrives as an empty string from a
/// shell variable that was never set, and refusing to start over one would make
/// the compatibility surface worse than ignoring the flag.
fn present(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

/// True when `value` parses as an integer equal to `inert`.
fn numeric_equals(value: &str, inert: i64) -> bool {
    value.trim().parse::<i64>().is_ok_and(|n| n == inert)
}

fn reject(
    option: &'static str,
    value: &str,
    limitation: &'static str,
    alternative: Option<&'static str>,
) -> MultimodalCompatRejection {
    reject_owned(option, value, limitation, alternative)
}

fn reject_owned(
    option: &'static str,
    value: &str,
    limitation: &'static str,
    alternative: Option<&'static str>,
) -> MultimodalCompatRejection {
    MultimodalCompatRejection {
        option,
        value: value.to_owned(),
        limitation,
        alternative,
    }
}

#[cfg(test)]
#[path = "multimodal_compat_args_tests.rs"]
mod tests;
