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

//! Pooling strategies, `1_Pooling/config.json` parsing, L2 normalization and
//! `dimensions` truncation shared by every embedding family.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::Value;

/// Environment override for the pooling mode, applied after the checkpoint
/// config and the family default. Debugging aid; logged at startup.
pub const POOLING_ENV: &str = "MLXCEL_EMBEDDING_POOLING";

/// Epsilon guarding the mean-pooling denominator and the L2 norm.
pub const POOLING_EPS: f32 = 1e-9;

/// How the `[B, L, D]` hidden states collapse into one `[B, D]` vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PoolingMode {
    /// The first real token (`[CLS]`); supports left padding.
    Cls,
    /// Mask-weighted mean over real tokens.
    Mean,
    /// Element-wise max over real tokens.
    Max,
    /// The last real token; correct for left and right padding.
    LastToken,
}

impl PoolingMode {
    /// Stable lowercase identifier matching the sentence-transformers
    /// `pooling_mode` values.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            PoolingMode::Cls => "cls",
            PoolingMode::Mean => "mean",
            PoolingMode::Max => "max",
            PoolingMode::LastToken => "lasttoken",
        }
    }
}

impl fmt::Display for PoolingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PoolingMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cls" => Ok(PoolingMode::Cls),
            "mean" => Ok(PoolingMode::Mean),
            "max" => Ok(PoolingMode::Max),
            "lasttoken" | "last_token" | "last" => Ok(PoolingMode::LastToken),
            other => bail!(
                "unsupported pooling mode `{other}`; expected one of cls, mean, max, lasttoken"
            ),
        }
    }
}

/// Reader for the sentence-transformers `1_Pooling/config.json` module file.
pub struct PoolingConfig;

/// Legacy boolean flags and the mode each maps to. The last two are listed
/// so that a checkpoint asking for them fails with the mode's name instead
/// of a silent fallback.
const LEGACY_FLAGS: &[(&str, Option<PoolingMode>)] = &[
    ("pooling_mode_cls_token", Some(PoolingMode::Cls)),
    ("pooling_mode_mean_tokens", Some(PoolingMode::Mean)),
    ("pooling_mode_max_tokens", Some(PoolingMode::Max)),
    ("pooling_mode_lasttoken", Some(PoolingMode::LastToken)),
    ("pooling_mode_weightedmean_tokens", None),
    ("pooling_mode_mean_sqrt_len_tokens", None),
];

impl PoolingConfig {
    /// Relative path of the pooling module config inside a checkpoint.
    pub const RELATIVE_PATH: &'static str = "1_Pooling/config.json";

    /// Read `<model_dir>/1_Pooling/config.json`. `Ok(None)` when the file is
    /// absent; an error when it names a mode this runtime does not implement
    /// (`weightedmean`, `mean_sqrt_len_tokens`, `include_prompt: false`, or
    /// more than one legacy flag set).
    pub fn read(model_dir: &Path) -> Result<Option<PoolingMode>> {
        let path = model_dir.join(Self::RELATIVE_PATH);
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let value: Value = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Self::parse(&value)
            .with_context(|| format!("unsupported pooling config in {}", path.display()))
            .map(Some)
    }

    /// Parse a pooling config object. Accepts the new-style `"pooling_mode"`
    /// string and the legacy `pooling_mode_*` booleans; exactly one legacy
    /// flag may be true and none true means `mean`.
    pub fn parse(value: &Value) -> Result<PoolingMode> {
        if value
            .get("include_prompt")
            .and_then(Value::as_bool)
            .is_some_and(|include| !include)
        {
            bail!("`include_prompt: false` (prompt tokens excluded from pooling) is not supported");
        }

        if let Some(mode) = value.get("pooling_mode") {
            let Some(mode) = mode.as_str() else {
                bail!("`pooling_mode` must be a string");
            };
            return mode.parse();
        }

        let mut selected: Vec<(&str, Option<PoolingMode>)> = Vec::new();
        for (flag, mode) in LEGACY_FLAGS {
            if value.get(*flag).and_then(Value::as_bool).unwrap_or(false) {
                selected.push((flag, *mode));
            }
        }
        match selected.as_slice() {
            [] => Ok(PoolingMode::Mean),
            [(flag, None)] => bail!("pooling mode `{flag}` is not supported"),
            [(_, Some(mode))] => Ok(*mode),
            many => {
                let names: Vec<&str> = many.iter().map(|(flag, _)| *flag).collect();
                bail!(
                    "combined pooling modes are not supported (flags set: {})",
                    names.join(", ")
                )
            }
        }
    }
}

/// Sentinel for "no `--pooling` override installed" in [`POOLING_OVERRIDE`].
const NO_POOLING_OVERRIDE: u8 = u8::MAX;

/// The operator's `--pooling` choice, installed once at startup (#1452).
///
/// A process-wide cell rather than a parameter because the resolution happens
/// inside each family's constructor, which is reached through the weight
/// loader and carries no server context. That is the same shape
/// [`POOLING_ENV`] already had; the flag simply takes precedence over it.
/// Stored as the discriminant of [`PoolingMode`] so the cell is lock-free and
/// safe to read from the embedding worker thread.
static POOLING_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(NO_POOLING_OVERRIDE);

/// Install the `--pooling` override.
///
/// Called once from startup, before any embedding checkpoint is loaded. Passing
/// `None` clears it, which is what the server tests need between cases.
pub fn set_pooling_override(mode: Option<PoolingMode>) {
    let encoded = match mode {
        Some(PoolingMode::Cls) => 0,
        Some(PoolingMode::Mean) => 1,
        Some(PoolingMode::Max) => 2,
        Some(PoolingMode::LastToken) => 3,
        None => NO_POOLING_OVERRIDE,
    };
    POOLING_OVERRIDE.store(encoded, std::sync::atomic::Ordering::Relaxed);
}

/// The installed `--pooling` override, if any.
#[must_use]
pub fn pooling_override() -> Option<PoolingMode> {
    match POOLING_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => Some(PoolingMode::Cls),
        1 => Some(PoolingMode::Mean),
        2 => Some(PoolingMode::Max),
        3 => Some(PoolingMode::LastToken),
        _ => None,
    }
}

/// Resolve the effective pooling mode for a checkpoint, in order: the
/// `--pooling` flag ([`set_pooling_override`]), then the [`POOLING_ENV`]
/// override, then `1_Pooling/config.json`, then `family_default`. Whichever
/// wins is logged.
///
/// The flag outranks the variable so an operator who exports
/// `MLXCEL_EMBEDDING_POOLING` in a shell profile can still override it per
/// invocation, which is the usual precedence for a flag against its own
/// environment fallback.
///
/// Used by: every embedding family constructor and the test stub.
pub fn resolve_pooling_mode(model_dir: &Path, family_default: PoolingMode) -> Result<PoolingMode> {
    let from_config = PoolingConfig::read(model_dir)?;
    let resolved = from_config.unwrap_or(family_default);
    if let Some(forced) = pooling_override() {
        tracing::info!(
            target: "mlxcel::embeddings",
            "--pooling {forced} overrides the resolved pooling mode {resolved}"
        );
        return Ok(forced);
    }
    match std::env::var(POOLING_ENV) {
        Ok(raw) if !raw.trim().is_empty() => {
            let forced: PoolingMode = raw
                .parse()
                .with_context(|| format!("invalid {POOLING_ENV}={raw:?}"))?;
            tracing::info!(
                target: "mlxcel::embeddings",
                "{POOLING_ENV}={forced} overrides the resolved pooling mode {resolved}"
            );
            Ok(forced)
        }
        _ => {
            tracing::info!(
                target: "mlxcel::embeddings",
                source = if from_config.is_some() { "1_Pooling/config.json" } else { "family default" },
                "embedding pooling mode: {resolved}"
            );
            Ok(resolved)
        }
    }
}

fn scalar_f32(value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(&[value], &[1, 1, 1])
}

/// Gather `hidden[b, index[b], :]` for a `[B, 1]` int32 index array.
fn gather_rows(hidden: &MlxArray, index: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(hidden);
    let (b, d) = (shape[0], shape[2]);
    let index = mlxcel_core::reshape(index, &[b, 1, 1]);
    let picked = mlxcel_core::take_along_axis(hidden, &index, 1);
    mlxcel_core::reshape(&picked, &[b, d])
}

/// Pool `hidden: [B, L, D]` over the token axis with `mask: [B, L]` int32
/// (`1` = real token, `0` = padding), returning `[B, D]` f32.
///
/// The hidden states are cast to f32 first so the reductions run at full
/// precision regardless of the activation dtype.
///
/// Used by: every single-vector embedding family and the test stub.
pub fn pool(hidden: &MlxArray, mask: &MlxArray, mode: PoolingMode) -> UniquePtr<MlxArray> {
    let hidden = mlxcel_core::astype(hidden, dtype::FLOAT32);
    let shape = mlxcel_core::array_shape(&hidden);
    assert!(
        shape.len() == 3,
        "pool: hidden must be [B, L, D], got {shape:?}"
    );
    let (b, l) = (shape[0], shape[1]);
    let zero_i32 = mlxcel_core::from_slice_i32(&[0], &[1, 1]);
    let real = mlxcel_core::not_equal(mask, &zero_i32);
    let real_3d = mlxcel_core::reshape(&real, &[b, l, 1]);

    match mode {
        PoolingMode::Mean => {
            let mask_f = mlxcel_core::astype(&real, dtype::FLOAT32);
            let weighted =
                mlxcel_core::multiply(&hidden, &mlxcel_core::reshape(&mask_f, &[b, l, 1]));
            let summed = mlxcel_core::sum_axis(&weighted, 1, false);
            let counts = mlxcel_core::sum_axis(&mask_f, 1, true);
            let denom = mlxcel_core::maximum(
                &counts,
                &mlxcel_core::from_slice_f32(&[POOLING_EPS], &[1, 1]),
            );
            mlxcel_core::divide(&summed, &denom)
        }
        PoolingMode::Max => {
            let neg_inf = scalar_f32(f32::NEG_INFINITY);
            let masked = mlxcel_core::where_cond(&real_3d, &hidden, &neg_inf);
            mlxcel_core::max_axis(&masked, 1, false)
        }
        PoolingMode::Cls => {
            // argmax over the int mask returns the first real index; with
            // right padding that is 0, with left padding the first 1.
            let first = mlxcel_core::argmax(&mlxcel_core::astype(&real, dtype::INT32), 1, true);
            gather_rows(&hidden, &first)
        }
        PoolingMode::LastToken => {
            let positions = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, l, 1), &[1, l]);
            let none = mlxcel_core::from_slice_i32(&[-1], &[1, 1]);
            let candidates = mlxcel_core::where_cond(&real, &positions, &none);
            let last = mlxcel_core::max_axis(&candidates, 1, true);
            // An all-padding row has no real token; use the final index so the
            // gather stays in bounds (its output is never consumed).
            let last_index = mlxcel_core::from_slice_i32(&[l - 1], &[1, 1]);
            let last =
                mlxcel_core::where_cond(&mlxcel_core::less(&last, &zero_i32), &last_index, &last);
            gather_rows(&hidden, &last)
        }
    }
}

/// L2-normalize along the last axis: `v / max(||v||_2, 1e-9)`.
///
/// Used by: the embedding engine after pooling and after `dimensions`
/// truncation.
pub fn normalize_l2(x: &MlxArray) -> UniquePtr<MlxArray> {
    let x = mlxcel_core::astype(x, dtype::FLOAT32);
    let norm = mlxcel_core::linalg_norm(&x, -1, true);
    let eps = mlxcel_core::from_slice_f32(&[POOLING_EPS], &[1]);
    mlxcel_core::divide(&x, &mlxcel_core::maximum(&norm, &eps))
}

/// Keep the first `n` components of the last axis (Matryoshka-style
/// `dimensions` truncation). The caller re-normalizes when normalization is
/// on. `n` must satisfy `1 <= n <= D`; the HTTP layer validates the request
/// before it reaches here.
pub fn truncate_dimensions(x: &MlxArray, n: usize) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last = *shape.last().expect("truncate_dimensions: rank >= 1");
    assert!(
        n >= 1 && n as i32 <= last,
        "truncate_dimensions: n={n} out of range for width {last}"
    );
    if n as i32 == last {
        return mlxcel_core::astype(x, mlxcel_core::array_dtype(x));
    }
    mlxcel_core::utils::slice_axis(x, -1, 0, n as i32)
}
