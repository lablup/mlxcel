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

//! MLX reference policy shared by the OpenXLA reference-check examples
//! (issue #3212).
//!
//! `MLX_ENABLE_TF32` defaults to 1, which runs f32 MLX GEMMs at TF32-class
//! precision on CUDA Ampere and later; #932 showed that this alone decided a
//! family gate. The root and `mlxcel-core` test binaries pin it through a
//! `ctor`, but every example is its own binary, so each one calls
//! [`pin_mlx_tf32_disabled`] first thing in `main`, before anything can touch
//! MLX, which reads the variable once per process.
//!
//! Used by: `xla_mlx_micro_oracle` and every family reference-check example
//! (`xla_*_reference_check`, `xla_qwen25_vl_window_check`,
//! `xla_qwen3_vl_deepstack_check`, `xla_gemma3n_audio_*`,
//! `xla_gemma3n_sscp_conv_oracle`, `xla_phi4_audio_check`).

use std::ffi::OsStr;

/// The MLX environment variable this policy pins.
pub const MLX_TF32_VARIABLE: &str = "MLX_ENABLE_TF32";

/// The only value an MLX reference may run with.
pub const MLX_TF32_REQUIRED: &str = "0";

/// What [`pin_mlx_tf32_disabled`] has to do for one observed value.
#[derive(Debug, PartialEq, Eq)]
enum Tf32Decision {
    /// Unset: pin it to the required value.
    Pin,
    /// Already the required value.
    Keep,
}

fn decide(value: Option<&OsStr>) -> Result<Tf32Decision, String> {
    match value {
        None => Ok(Tf32Decision::Pin),
        Some(value) if value == OsStr::new(MLX_TF32_REQUIRED) => Ok(Tf32Decision::Keep),
        Some(value) => Err(format!(
            "refusing to run: {MLX_TF32_VARIABLE}={} would let MLX run f32 contractions at \
             TF32-class precision; an MLX reference requires {MLX_TF32_VARIABLE}=\
             {MLX_TF32_REQUIRED} (unset it or set it to {MLX_TF32_REQUIRED})",
            value.to_string_lossy()
        )),
    }
}

/// Pin `MLX_ENABLE_TF32=0` for this process, or refuse when the operator set
/// another value. Returns the recorded value and prints it, so every report
/// that follows names it.
///
/// Call it as the first statement of `main`, before any MLX call and before
/// any thread starts.
pub fn pin_mlx_tf32_disabled() -> Result<&'static str, String> {
    match decide(std::env::var_os(MLX_TF32_VARIABLE).as_deref())? {
        // SAFETY: called first in `main`, before this process starts any
        // thread or initializes MLX, so nothing reads the environment
        // concurrently.
        Tf32Decision::Pin => unsafe { std::env::set_var(MLX_TF32_VARIABLE, MLX_TF32_REQUIRED) },
        Tf32Decision::Keep => {}
    }
    eprintln!("mlx-reference-policy: {MLX_TF32_VARIABLE}={MLX_TF32_REQUIRED}");
    Ok(MLX_TF32_REQUIRED)
}

/// The `MLX_ENABLE_TF32` value this process runs with, for a JSON report.
/// After [`pin_mlx_tf32_disabled`] succeeded it is always `"0"`.
#[allow(dead_code)] // Only the examples that write a JSON report record it there.
pub fn recorded_mlx_tf32() -> String {
    std::env::var(MLX_TF32_VARIABLE).unwrap_or_else(|_| "unset (MLX default 1)".to_string())
}

/// [`pin_mlx_tf32_disabled`] for examples whose `main` returns `()`: print
/// the refusal and exit non-zero.
#[allow(dead_code)] // Not every example that includes this module uses both entry points.
pub fn pin_mlx_tf32_disabled_or_exit() -> &'static str {
    pin_mlx_tf32_disabled().unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    })
}

#[cfg(test)]
mod tf32_policy_tests {
    use super::*;

    #[test]
    fn unset_is_pinned_and_zero_is_kept() {
        assert_eq!(decide(None), Ok(Tf32Decision::Pin));
        assert_eq!(decide(Some(OsStr::new("0"))), Ok(Tf32Decision::Keep));
    }

    #[test]
    fn any_other_value_is_refused_and_named() {
        for value in ["1", "", "false", " 0"] {
            let error = decide(Some(OsStr::new(value))).unwrap_err();
            assert!(error.contains("refusing to run"), "{error}");
            assert!(
                error.contains(&format!("MLX_ENABLE_TF32={value}")),
                "{error}"
            );
        }
    }
}
