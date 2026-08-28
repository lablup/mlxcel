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

//! Multi-adapter LoRA specification (llama-server b10621 compatible,
//! issue #1439).
//!
//! b10621 loads any number of adapters from `--lora a,b` (scale 1.0 each) and
//! `--lora-scaled a:0.5,b:2.0`, keeps them as runtime-swappable layers, and
//! reports them by id on `GET /lora-adapters`. mlxcel fuses adapters into the
//! base weights at load time, one after another in the order the flags listed
//! them (fusion adds deltas, so the result is order-independent, but the
//! order is still fixed so logs and failure attribution are deterministic).
//! The parsed specification is what the inventory route reports and what the
//! per-request `lora` field is validated against.

use std::path::{Path, PathBuf};

/// One adapter the server was started with.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapterSpec {
    pub path: PathBuf,
    /// User scale from `--lora-scaled` (1.0 for `--lora`), multiplied into
    /// the adapter's own `alpha / r` at fusion time exactly as upstream
    /// multiplies its per-adapter scale.
    pub scale: f32,
    /// `false` under `--lora-init-without-apply`: the adapter is validated
    /// and reported at scale 0.0 but not fused.
    pub apply: bool,
}

impl LoraAdapterSpec {
    /// The scale `GET /lora-adapters` reports: b10621 reports 0.0 for a
    /// loaded-but-not-applied adapter.
    pub fn reported_scale(&self) -> f32 {
        if self.apply { self.scale } else { 0.0 }
    }
}

/// Parse the b10621 `--lora` / `--lora-scaled` / `--lora-init-without-apply`
/// surface into an ordered adapter list.
///
/// `--lora` is comma-separated paths at scale 1.0; `--lora-scaled` is
/// comma-separated `FNAME:SCALE` pairs, split on the LAST colon so absolute
/// paths keep their separators. A scale that does not parse, or is NaN or
/// infinite, is a startup error: fusing a non-finite delta would poison every
/// weight it touches and surface only as garbage output.
pub fn parse_lora_flags(
    lora: Option<&str>,
    lora_scaled: Option<&str>,
    init_without_apply: bool,
) -> anyhow::Result<Vec<LoraAdapterSpec>> {
    let mut specs = Vec::new();
    if let Some(list) = lora {
        for part in list.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            specs.push(LoraAdapterSpec {
                path: PathBuf::from(part),
                scale: 1.0,
                apply: !init_without_apply,
            });
        }
    }
    if let Some(list) = lora_scaled {
        for part in list.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let Some((path, scale_str)) = part.rsplit_once(':') else {
                anyhow::bail!(
                    "--lora-scaled entry '{part}' is missing its scale; the format is \
                     FNAME:SCALE (comma-separated for multiple adapters)"
                );
            };
            let scale: f32 = scale_str.trim().parse().map_err(|_| {
                anyhow::anyhow!(
                    "--lora-scaled entry '{part}' has a non-numeric scale '{scale_str}'"
                )
            })?;
            if !scale.is_finite() {
                anyhow::bail!(
                    "--lora-scaled entry '{part}' has a non-finite scale; fusing it would \
                     poison every weight the adapter touches"
                );
            }
            if path.trim().is_empty() {
                anyhow::bail!("--lora-scaled entry '{part}' has an empty adapter path");
            }
            specs.push(LoraAdapterSpec {
                path: PathBuf::from(path.trim()),
                scale,
                apply: !init_without_apply,
            });
        }
    }
    Ok(specs)
}

/// Startup validation: every adapter directory must exist and carry an
/// `adapter_config.json`, so a typo fails the command line rather than the
/// model load minutes later.
pub fn validate_adapter_paths(specs: &[LoraAdapterSpec]) -> anyhow::Result<()> {
    for spec in specs {
        let config = spec.path.join("adapter_config.json");
        if !config.is_file() {
            anyhow::bail!(
                "LoRA adapter {} is not a readable adapter directory (no adapter_config.json)",
                spec.path.display()
            );
        }
    }
    Ok(())
}

/// Whether the whole specification is the trivial single-adapter case the
/// pre-#1439 `adapter_path` plumbing already served byte-identically.
pub fn is_legacy_single(specs: &[LoraAdapterSpec]) -> Option<&Path> {
    match specs {
        [only] if only.apply && only.scale == 1.0 => Some(&only.path),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_splits_commas_at_scale_one() {
        let specs = parse_lora_flags(Some("a,b , c"), None, false).expect("parse");
        assert_eq!(specs.len(), 3);
        assert!(specs.iter().all(|s| s.scale == 1.0 && s.apply));
        assert_eq!(specs[0].path, PathBuf::from("a"));
        assert_eq!(specs[2].path, PathBuf::from("c"));
    }

    #[test]
    fn lora_scaled_parses_fname_colon_scale_pairs() {
        let specs =
            parse_lora_flags(None, Some("/abs/path/adapter:0.5,rel:2"), false).expect("parse");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].path, PathBuf::from("/abs/path/adapter"));
        assert_eq!(specs[0].scale, 0.5);
        assert_eq!(specs[1].scale, 2.0);
    }

    #[test]
    fn combined_flags_keep_listed_order() {
        let specs = parse_lora_flags(Some("plain"), Some("scaled:0.25"), false).expect("parse");
        assert_eq!(specs[0].path, PathBuf::from("plain"));
        assert_eq!(specs[1].path, PathBuf::from("scaled"));
        assert_eq!(specs[1].scale, 0.25);
    }

    #[test]
    fn init_without_apply_marks_every_adapter_unapplied() {
        let specs = parse_lora_flags(Some("a"), Some("b:2"), true).expect("parse");
        assert!(specs.iter().all(|s| !s.apply));
        assert_eq!(specs[0].reported_scale(), 0.0);
        assert_eq!(specs[1].reported_scale(), 0.0);
    }

    #[test]
    fn bad_scales_are_startup_errors() {
        assert!(
            parse_lora_flags(None, Some("a"), false).is_err(),
            "missing scale"
        );
        assert!(
            parse_lora_flags(None, Some("a:x"), false).is_err(),
            "non-numeric"
        );
        assert!(parse_lora_flags(None, Some("a:NaN"), false).is_err(), "NaN");
        assert!(parse_lora_flags(None, Some("a:inf"), false).is_err(), "inf");
        assert!(
            parse_lora_flags(None, Some(":1"), false).is_err(),
            "empty path"
        );
    }

    #[test]
    fn legacy_single_detection() {
        let single = parse_lora_flags(Some("a"), None, false).expect("parse");
        assert_eq!(is_legacy_single(&single), Some(Path::new("a")));
        let scaled = parse_lora_flags(None, Some("a:2"), false).expect("parse");
        assert!(is_legacy_single(&scaled).is_none());
        let noapply = parse_lora_flags(Some("a"), None, true).expect("parse");
        assert!(is_legacy_single(&noapply).is_none());
        let multi = parse_lora_flags(Some("a,b"), None, false).expect("parse");
        assert!(is_legacy_single(&multi).is_none());
    }

    #[test]
    fn missing_adapter_config_fails_validation() {
        let specs = vec![LoraAdapterSpec {
            path: PathBuf::from("/definitely/not/an/adapter"),
            scale: 1.0,
            apply: true,
        }];
        assert!(validate_adapter_paths(&specs).is_err());
    }
}
