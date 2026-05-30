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

//! YAML configuration schema for Axis A weight-load surgery.
//!
//! Schema overview (see `examples/surgery/*.yaml` for full samples):
//!
//! ```yaml
//! version: 1
//! operations:
//!   - op: scale
//!     pattern: "model.layers.*.self_attn.o_proj.weight"
//!     factor: 1.2
//!
//!   - op: add
//!     pattern: "model.layers.*.mlp.down_proj.weight"
//!     source: "./task_vectors/personality_friendly.safetensors"
//!     alpha: 0.5
//!
//!   - op: prune
//!     granularity: attention_head   # or: layer | mlp_channel
//!     pattern: "model.layers.12.self_attn.*"
//!     head_ids: [3, 7]
//!
//!   - op: replace
//!     pattern: "model.embed_tokens.weight"
//!     source: "./donors/other_model.safetensors"
//!     source_key: "model.embed_tokens.weight"
//!
//!   - op: interpolate
//!     pattern: "model.layers.*.*"
//!     source_a: "./model_a.safetensors"
//!     source_b: "./model_b.safetensors"
//!     ratio: 0.3
//!     method: slerp                 # or: lerp
//! ```
//!
//! The parser validates schema syntax, glob pattern syntax, schema
//! version, and the presence of external source files at parse time
//! (so the load fails before any weights are mutated). All five
//! op variants — `scale` (A5), `add` (A6), `prune` (A7), `replace`
//! (A8), and `interpolate` (A9) — materialize to real
//! [`crate::ops::ScaleOp`] / [`crate::ops::AddOp`] /
//! [`crate::ops::PruneOp`] / [`crate::ops::ReplaceOp`] /
//! [`crate::ops::InterpolateOp`] instances.
//!
//! ## Path resolution
//!
//! Relative `source*` paths in the YAML are resolved against the
//! parent directory of the YAML file. Absolute paths are taken
//! verbatim. The parser canonicalizes paths once at parse time, so
//! downstream ops can use the stored [`std::path::PathBuf`] without
//! re-resolving.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use globset::{Glob, GlobMatcher};
use serde::Deserialize;

use crate::ops::{InterpolateOp, ScaleOp};
#[cfg(test)]
use crate::WeightMap;
use crate::{SharedSurgeryOp, SurgeryError, SurgeryPipeline};

/// The only schema version this parser understands. Bump when the
/// YAML shape changes in a way that is not backwards-compatible; the
/// parser rejects YAML with an unknown version up front so users get
/// a clear error rather than a silent misinterpretation.
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Top-level YAML document.
///
/// Used by: [`parse_config_str`], [`parse_config_file`]
#[derive(Debug, Clone, Deserialize)]
pub struct SurgeryConfig {
    /// Schema version. Must equal [`SUPPORTED_SCHEMA_VERSION`].
    pub version: u32,
    /// Ordered list of surgery operations. An empty list is allowed
    /// — it produces an empty [`SurgeryPipeline`] whose `apply` is a
    /// no-op.
    #[serde(default)]
    pub operations: Vec<OpSpec>,
}

/// Tagged-union spec for the five supported operation types.
///
/// The `op` field selects the variant; remaining fields are
/// variant-specific. Unknown `op` values are rejected at parse time
/// with a descriptive error.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum OpSpec {
    /// `scale(W) = factor * W` over every tensor whose key matches
    /// `pattern`.
    Scale {
        /// Glob pattern matched against tensor keys.
        pattern: String,
        /// Scalar multiplier (any finite f32).
        factor: f32,
    },

    /// `add(W) = W + alpha * source[key]` (task-vector / arithmetic
    /// finetuning).
    Add {
        /// Glob pattern matched against tensor keys.
        pattern: String,
        /// Path to a safetensors file containing the task vector.
        source: PathBuf,
        /// Coefficient on the source term. Defaults to `1.0`.
        #[serde(default = "default_alpha_one")]
        alpha: f32,
    },

    /// Structural pruning at the granularity indicated by
    /// `granularity`.
    Prune {
        /// Which structural axis is being pruned.
        granularity: PruneGranularity,
        /// Glob pattern matched against tensor keys.
        pattern: String,
        /// Head indices (required when `granularity = attention_head`).
        #[serde(default)]
        head_ids: Option<Vec<usize>>,
        /// MLP channel indices (required when
        /// `granularity = mlp_channel`).
        #[serde(default)]
        channel_ids: Option<Vec<usize>>,
        /// Layer indices (required when `granularity = layer`).
        #[serde(default)]
        layer_ids: Option<Vec<usize>>,
    },

    /// Replace matching weights with the corresponding tensor from a
    /// donor safetensors file.
    Replace {
        /// Glob pattern matched against tensor keys.
        pattern: String,
        /// Donor file path.
        source: PathBuf,
        /// Donor tensor key to read.
        source_key: String,
    },

    /// Interpolate (lerp / slerp) between two donor checkpoints.
    Interpolate {
        /// Glob pattern matched against tensor keys.
        pattern: String,
        /// Donor checkpoint A.
        source_a: PathBuf,
        /// Donor checkpoint B.
        source_b: PathBuf,
        /// Mixing ratio in `[0.0, 1.0]`.
        ratio: f32,
        /// Interpolation method.
        method: InterpolateMethod,
    },
}

fn default_alpha_one() -> f32 {
    1.0
}

/// Structural-pruning granularity.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PruneGranularity {
    /// Drop entire attention heads.
    AttentionHead,
    /// Drop entire transformer blocks.
    Layer,
    /// Drop individual MLP channels.
    MlpChannel,
}

/// Tensor-space interpolation method.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InterpolateMethod {
    /// Spherical linear interpolation.
    Slerp,
    /// Linear interpolation.
    Lerp,
}

/// Parse a YAML config string and build a populated
/// [`SurgeryPipeline`].
///
/// `base_dir` is used to resolve any relative `source*` paths in the
/// YAML. Pass the directory containing the YAML file when calling
/// from [`parse_config_file`], or `None` to require absolute paths.
///
/// On success, all five operations (`scale` (A5), `add` (A6),
/// `prune` (A7), `replace` (A8), `interpolate` (A9)) are
/// materialized as real concrete [`SurgeryOp`] instances:
/// [`ScaleOp`] / [`crate::ops::AddOp`] / [`crate::ops::PruneOp`] /
/// [`crate::ops::ReplaceOp`] / [`crate::ops::InterpolateOp`].
///
/// Used by: [`parse_config_file`], CLI wiring from A4 (`--surgery`)
pub fn parse_config_str(
    yaml: &str,
    base_dir: Option<&Path>,
) -> Result<SurgeryPipeline, SurgeryError> {
    let config: SurgeryConfig = serde_yaml::from_str(yaml)
        .map_err(|e| SurgeryError::Other(anyhow::anyhow!("failed to parse surgery YAML: {e}")))?;

    if config.version != SUPPORTED_SCHEMA_VERSION {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "unsupported surgery schema version {}; this build understands version {}",
            config.version,
            SUPPORTED_SCHEMA_VERSION,
        )));
    }

    let mut ops: Vec<SharedSurgeryOp> = Vec::with_capacity(config.operations.len());
    for (idx, spec) in config.operations.into_iter().enumerate() {
        ops.push(materialize_op(idx, spec, base_dir)?);
    }
    Ok(SurgeryPipeline::from(ops))
}

/// Parse a YAML config from a file path and build a populated
/// [`SurgeryPipeline`]. Relative `source*` paths in the YAML are
/// resolved against the file's parent directory.
///
/// Used by: future CLI wiring (A4), `mlxcel surgery validate` (A4),
/// integration tests in this crate
pub fn parse_config_file<P: AsRef<Path>>(path: P) -> Result<SurgeryPipeline, SurgeryError> {
    let path = path.as_ref();
    let yaml = std::fs::read_to_string(path).map_err(|e| {
        SurgeryError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to read surgery config '{}': {e}", path.display()),
        ))
    })?;
    let base_dir = path.parent();
    parse_config_str(&yaml, base_dir)
}

/// Validate one `OpSpec`, resolve its paths, and construct the
/// matching concrete [`SurgeryOp`].
fn materialize_op(
    idx: usize,
    spec: OpSpec,
    base_dir: Option<&Path>,
) -> Result<SharedSurgeryOp, SurgeryError> {
    match spec {
        OpSpec::Scale { pattern, factor } => {
            // Compile the glob up-front so a malformed pattern still
            // fails at parse time with the standard
            // `surgery operation #N (scale): ...` prefix, even though
            // `ScaleOp::new` would compile the glob a second time on
            // its own. The cost is one transient `GlobMatcher`; the
            // benefit is identical error wording for every op kind.
            let _glob = compile_glob(idx, "scale", &pattern)?;
            if !factor.is_finite() {
                return Err(spec_error(idx, "scale", "factor must be finite"));
            }
            let op = ScaleOp::new(&pattern, factor).map_err(|e| {
                spec_error(idx, "scale", &format!("failed to construct ScaleOp: {e}"))
            })?;
            Ok(Arc::new(op))
        }
        OpSpec::Add {
            pattern,
            source,
            alpha,
        } => {
            // A6 — concrete factory hook for `op: add`.
            // The compiled glob is built once here for validation and
            // then handed off to `AddOp::from_compiled` so the
            // operation does not re-parse it.
            let glob = compile_glob(idx, "add", &pattern)?;
            if !alpha.is_finite() {
                return Err(spec_error(idx, "add", "alpha must be finite"));
            }
            let resolved = resolve_existing_source(idx, "add", "source", &source, base_dir)?;
            Ok(Arc::new(crate::ops::AddOp::from_compiled(
                glob, pattern, resolved, alpha,
            )))
        }
        OpSpec::Prune {
            granularity,
            pattern,
            head_ids,
            channel_ids,
            layer_ids,
        } => {
            // Validate glob syntax up-front so the YAML parser fails
            // cleanly before A7 builds a real PruneOp. Discard the
            // compiled glob — `PruneOp::new` re-compiles internally —
            // because matching is constructor-private.
            let _glob = compile_glob(idx, "prune", &pattern)?;
            validate_prune_ids(idx, granularity, &head_ids, &channel_ids, &layer_ids)?;
            // A7 dispatches granularity to the concrete `PruneOp`.
            // Errors here are user-facing (e.g. invalid id list), so
            // we route them through `spec_error` to keep the message
            // shape consistent with the rest of the parser.
            let op = crate::ops::prune::build_from_yaml(
                &pattern,
                granularity,
                head_ids,
                channel_ids,
                layer_ids,
            )
            .map_err(|e| spec_error(idx, "prune", &format!("{e}")))?;
            Ok(Arc::new(op))
        }
        OpSpec::Replace {
            pattern,
            source,
            source_key,
        } => {
            // Validate the glob syntax up front so users get the
            // same "malformed glob" diagnostic the other ops
            // produce. The `ReplaceOp` uses its own capture-aware
            // matcher (see `ops::replace::wildcard`) but pinning
            // syntax compatibility with `globset::Glob` here keeps
            // the YAML schema consistent across op kinds.
            let _glob_syntax_check = compile_glob(idx, "replace", &pattern)?;
            if source_key.is_empty() {
                return Err(spec_error(idx, "replace", "source_key must not be empty"));
            }
            let resolved = resolve_existing_source(idx, "replace", "source", &source, base_dir)?;
            let op = crate::ops::ReplaceOp::new(&pattern, &source_key, resolved)
                .map_err(|e| spec_error(idx, "replace", &format!("{e}")))?;
            Ok(Arc::new(op))
        }
        OpSpec::Interpolate {
            pattern,
            source_a,
            source_b,
            ratio,
            method,
        } => {
            // Validate the glob at the config layer so a malformed
            // pattern surfaces with a "surgery operation #N" error
            // path that mirrors the other op variants. The compiled
            // matcher itself is rebuilt inside `InterpolateOp::new`,
            // which is a cheap operation on the already-validated
            // string.
            let _ = compile_glob(idx, "interpolate", &pattern)?;
            if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
                return Err(spec_error(
                    idx,
                    "interpolate",
                    "ratio must be a finite number in [0.0, 1.0]",
                ));
            }
            let resolved_a =
                resolve_existing_source(idx, "interpolate", "source_a", &source_a, base_dir)?;
            let resolved_b =
                resolve_existing_source(idx, "interpolate", "source_b", &source_b, base_dir)?;
            let op = InterpolateOp::new(&pattern, resolved_a, resolved_b, ratio, method)
                .map_err(|e| spec_error(idx, "interpolate", &format!("{e}")))?;
            Ok(Arc::new(op))
        }
    }
}

fn compile_glob(idx: usize, op_name: &str, pattern: &str) -> Result<GlobMatcher, SurgeryError> {
    Glob::new(pattern)
        .map(|g| g.compile_matcher())
        .map_err(|e| {
            spec_error(
                idx,
                op_name,
                &format!("malformed glob pattern {pattern:?}: {e}"),
            )
        })
}

fn validate_prune_ids(
    idx: usize,
    granularity: PruneGranularity,
    head_ids: &Option<Vec<usize>>,
    channel_ids: &Option<Vec<usize>>,
    layer_ids: &Option<Vec<usize>>,
) -> Result<(), SurgeryError> {
    let chosen = match granularity {
        PruneGranularity::AttentionHead => ("head_ids", head_ids.as_deref()),
        PruneGranularity::MlpChannel => ("channel_ids", channel_ids.as_deref()),
        PruneGranularity::Layer => ("layer_ids", layer_ids.as_deref()),
    };
    let field = chosen.0;
    let ids = chosen.1.ok_or_else(|| {
        spec_error(
            idx,
            "prune",
            &format!("granularity={granularity:?} requires field `{field}`"),
        )
    })?;
    if ids.is_empty() {
        return Err(spec_error(
            idx,
            "prune",
            &format!("`{field}` must contain at least one id"),
        ));
    }
    Ok(())
}

fn resolve_existing_source(
    idx: usize,
    op_name: &str,
    field: &str,
    source: &Path,
    base_dir: Option<&Path>,
) -> Result<PathBuf, SurgeryError> {
    let resolved = if source.is_absolute() {
        source.to_path_buf()
    } else if let Some(base) = base_dir {
        base.join(source)
    } else {
        return Err(spec_error(
            idx,
            op_name,
            &format!(
                "{field} must be an absolute path when no config file directory is available; got {}",
                source.display()
            ),
        ));
    };

    if !resolved.exists() {
        return Err(spec_error(
            idx,
            op_name,
            &format!("{field} path does not exist: {}", resolved.display()),
        ));
    }
    Ok(resolved)
}

fn spec_error(idx: usize, op_name: &str, msg: &str) -> SurgeryError {
    SurgeryError::Other(anyhow::anyhow!(
        "surgery operation #{idx} ({op_name}): {msg}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel_core::weights::WeightTransform;
    use std::io::Write;

    /// Helper: write a YAML string into a tempfile and parse it via
    /// the file-based entry point so the test exercises relative
    /// path resolution too.
    fn parse_tempfile(yaml: &str) -> Result<SurgeryPipeline, SurgeryError> {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("surgery.yaml");
        let mut f = std::fs::File::create(&path).expect("write yaml");
        f.write_all(yaml.as_bytes()).expect("write yaml bytes");
        // Keep the tempdir alive by leaking it through std::mem::forget
        // so callers can re-stat the directory. The OS cleans up on
        // process exit; tests are short-lived.
        std::mem::forget(dir);
        parse_config_file(&path)
    }

    /// Helper: create a tempdir, write the YAML there, plus a list
    /// of dummy "donor" safetensors files. Returns the tempdir so
    /// the caller can keep it alive.
    fn write_yaml_with_donors(yaml: &str, donor_names: &[&str]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        for name in donor_names {
            let donor_path = dir.path().join(name);
            // Donors are validated by existence only at parse time;
            // any non-empty bytes are fine.
            std::fs::write(&donor_path, b"\x00\x00\x00\x00").expect("write donor");
        }
        let yaml_path = dir.path().join("surgery.yaml");
        std::fs::write(&yaml_path, yaml).expect("write yaml");
        (dir, yaml_path)
    }

    #[test]
    fn empty_operations_yields_empty_pipeline() {
        let yaml = "version: 1\noperations: []\n";
        let pipeline = parse_config_str(yaml, None).expect("parse should succeed");
        assert!(pipeline.is_empty());
        // Empty pipeline must be a no-op when applied via A1's hook.
        let mut weights = WeightMap::new();
        WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
            .expect("empty pipeline apply must succeed");
        assert!(weights.is_empty());
    }

    #[test]
    fn omitted_operations_field_defaults_to_empty() {
        let yaml = "version: 1\n";
        let pipeline = parse_config_str(yaml, None).expect("parse should succeed");
        assert!(pipeline.is_empty());
    }

    #[test]
    fn scale_op_parses_and_validates_factor() {
        let yaml = r#"
version: 1
operations:
  - op: scale
    pattern: "model.layers.*.self_attn.o_proj.weight"
    factor: 1.5
"#;
        let pipeline = parse_config_str(yaml, None).expect("parse");
        assert_eq!(pipeline.len(), 1);
        let names: Vec<_> = pipeline.ops().map(|op| op.name()).collect();
        assert_eq!(names, vec!["scale"]);
    }

    #[test]
    fn scale_rejects_non_finite_factor() {
        let yaml = r#"
version: 1
operations:
  - op: scale
    pattern: "*"
    factor: .nan
"#;
        let err = parse_config_str(yaml, None).expect_err("nan factor must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("scale") && msg.contains("finite"),
            "error must mention op and reason: {msg}"
        );
    }

    #[test]
    fn add_op_resolves_relative_source() {
        let yaml = r#"version: 1
operations:
  - op: add
    pattern: "model.layers.*.mlp.down_proj.weight"
    source: "./task_vec.safetensors"
    alpha: 0.5
"#;
        let (_dir, yaml_path) = write_yaml_with_donors(yaml, &["task_vec.safetensors"]);
        let pipeline = parse_config_file(&yaml_path).expect("parse with relative source");
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn add_op_alpha_defaults_to_one_when_omitted() {
        let yaml = r#"version: 1
operations:
  - op: add
    pattern: "*"
    source: "./donor.safetensors"
"#;
        let (_dir, yaml_path) = write_yaml_with_donors(yaml, &["donor.safetensors"]);
        // The pipeline parses successfully — alpha is implicit 1.0.
        let pipeline = parse_config_file(&yaml_path).expect("alpha is optional");
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn add_rejects_missing_source_file() {
        let yaml = r#"version: 1
operations:
  - op: add
    pattern: "*"
    source: "./does-not-exist.safetensors"
    alpha: 1.0
"#;
        let (_dir, yaml_path) = write_yaml_with_donors(yaml, &[]);
        let err = parse_config_file(&yaml_path).expect_err("missing donor must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("does-not-exist.safetensors") && msg.contains("does not exist"),
            "error must mention the missing path: {msg}"
        );
    }

    #[test]
    fn prune_attention_head_requires_head_ids() {
        let yaml = r#"version: 1
operations:
  - op: prune
    granularity: attention_head
    pattern: "model.layers.12.self_attn.*"
"#;
        let err = parse_config_str(yaml, None).expect_err("missing head_ids must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("head_ids") && msg.contains("requires"),
            "error must demand head_ids: {msg}"
        );
    }

    #[test]
    fn prune_attention_head_accepts_head_ids() {
        let yaml = r#"version: 1
operations:
  - op: prune
    granularity: attention_head
    pattern: "model.layers.12.self_attn.*"
    head_ids: [3, 7]
"#;
        let pipeline = parse_config_str(yaml, None).expect("parse");
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn prune_layer_requires_layer_ids() {
        let yaml = r#"version: 1
operations:
  - op: prune
    granularity: layer
    pattern: "model.layers.12.*"
"#;
        let err = parse_config_str(yaml, None).expect_err("missing layer_ids must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("layer_ids"),
            "error must mention layer_ids: {msg}"
        );
    }

    #[test]
    fn prune_mlp_channel_requires_channel_ids() {
        let yaml = r#"version: 1
operations:
  - op: prune
    granularity: mlp_channel
    pattern: "model.layers.*.mlp.gate_proj.weight"
"#;
        let err = parse_config_str(yaml, None).expect_err("missing channel_ids must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("channel_ids"),
            "error must mention channel_ids: {msg}"
        );
    }

    #[test]
    fn prune_rejects_empty_id_list() {
        let yaml = r#"version: 1
operations:
  - op: prune
    granularity: attention_head
    pattern: "*"
    head_ids: []
"#;
        let err = parse_config_str(yaml, None).expect_err("empty list must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("at least one id"),
            "error must reject empty: {msg}"
        );
    }

    #[test]
    fn replace_op_resolves_source() {
        let yaml = r#"version: 1
operations:
  - op: replace
    pattern: "model.embed_tokens.weight"
    source: "./donor.safetensors"
    source_key: "model.embed_tokens.weight"
"#;
        let (_dir, yaml_path) = write_yaml_with_donors(yaml, &["donor.safetensors"]);
        let pipeline = parse_config_file(&yaml_path).expect("replace parses");
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn replace_rejects_empty_source_key() {
        let yaml = r#"version: 1
operations:
  - op: replace
    pattern: "*"
    source: "./donor.safetensors"
    source_key: ""
"#;
        let (_dir, yaml_path) = write_yaml_with_donors(yaml, &["donor.safetensors"]);
        let err = parse_config_file(&yaml_path).expect_err("empty source_key must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("source_key") && msg.contains("not be empty"),
            "error must mention empty source_key: {msg}"
        );
    }

    #[test]
    fn interpolate_op_parses_with_slerp() {
        let yaml = r#"version: 1
operations:
  - op: interpolate
    pattern: "model.layers.*.*"
    source_a: "./a.safetensors"
    source_b: "./b.safetensors"
    ratio: 0.3
    method: slerp
"#;
        let (_dir, yaml_path) = write_yaml_with_donors(yaml, &["a.safetensors", "b.safetensors"]);
        let pipeline = parse_config_file(&yaml_path).expect("interpolate parses");
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn interpolate_rejects_out_of_range_ratio() {
        let yaml = r#"version: 1
operations:
  - op: interpolate
    pattern: "*"
    source_a: "./a.safetensors"
    source_b: "./b.safetensors"
    ratio: 1.5
    method: lerp
"#;
        let (_dir, yaml_path) = write_yaml_with_donors(yaml, &["a.safetensors", "b.safetensors"]);
        let err = parse_config_file(&yaml_path).expect_err("ratio>1 must fail");
        let msg = format!("{err}");
        assert!(msg.contains("ratio"), "error must mention ratio: {msg}");
    }

    #[test]
    fn unknown_op_is_rejected() {
        let yaml = r#"version: 1
operations:
  - op: teleport
    pattern: "*"
"#;
        let err = parse_config_str(yaml, None).expect_err("unknown op must fail");
        let msg = format!("{err}");
        assert!(msg.contains("teleport") || msg.contains("unknown"), "{msg}");
    }

    #[test]
    fn missing_required_field_is_rejected() {
        let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "*"
"#; // factor missing
        let err = parse_config_str(yaml, None).expect_err("missing field must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("factor"),
            "error must mention missing field: {msg}"
        );
    }

    #[test]
    fn wrong_field_type_is_rejected() {
        let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "*"
    factor: "not a number"
"#;
        let err = parse_config_str(yaml, None).expect_err("type mismatch must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("factor") || msg.contains("number"),
            "error must surface the type mismatch: {msg}",
        );
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let yaml = "version: 99\noperations: []\n";
        let err = parse_config_str(yaml, None).expect_err("future version must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("version") && msg.contains("99"),
            "error must mention the unknown version: {msg}",
        );
    }

    #[test]
    fn malformed_glob_pattern_is_rejected() {
        let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "model.layers.{0"
    factor: 1.0
"#;
        let err = parse_config_str(yaml, None).expect_err("malformed glob must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("glob") || msg.contains("pattern") || msg.contains("malformed"),
            "error must mention glob issue: {msg}"
        );
    }

    #[test]
    fn scale_op_materializes_to_real_implementation() {
        // A5 landing pin: `scale` is no longer a "not yet implemented"
        // stub. Parse a `scale` op, populate a synthetic WeightMap,
        // and confirm the pipeline mutates the matched tensor — i.e.
        // the placeholder is gone and `ScaleOp` is wired through the
        // factory.
        let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "model.layer.weight"
    factor: 2.0
"#;
        let pipeline = parse_config_str(yaml, None).expect("parse");
        let mut weights = WeightMap::new();
        weights.insert(
            "model.layer.weight".to_string(),
            mlxcel_core::from_slice_f32(&[3.0, -4.0], &[2]),
        );

        WeightTransform::apply(&pipeline, &mut weights, &serde_json::Value::Null)
            .expect("real ScaleOp must apply without error");

        let after = weights.get("model.layer.weight").unwrap();
        mlxcel_core::eval(after);
        let bytes = mlxcel_core::array_to_raw_bytes(after);
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![6.0, -8.0]);
    }

    #[test]
    fn full_schema_parses_with_all_op_types() {
        let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "model.layers.*.self_attn.o_proj.weight"
    factor: 1.2

  - op: add
    pattern: "model.layers.*.mlp.down_proj.weight"
    source: "./task_vec.safetensors"
    alpha: 0.5

  - op: prune
    granularity: attention_head
    pattern: "model.layers.12.self_attn.*"
    head_ids: [3, 7]

  - op: replace
    pattern: "model.embed_tokens.weight"
    source: "./donor.safetensors"
    source_key: "model.embed_tokens.weight"

  - op: interpolate
    pattern: "model.layers.*.*"
    source_a: "./a.safetensors"
    source_b: "./b.safetensors"
    ratio: 0.3
    method: slerp
"#;
        let (_dir, yaml_path) = write_yaml_with_donors(
            yaml,
            &[
                "task_vec.safetensors",
                "donor.safetensors",
                "a.safetensors",
                "b.safetensors",
            ],
        );
        let pipeline = parse_config_file(&yaml_path).expect("full schema parses");
        assert_eq!(pipeline.len(), 5);
        let names: Vec<&'static str> = pipeline.ops().map(|op| op.name()).collect();
        assert_eq!(
            names,
            vec!["scale", "add", "prune", "replace", "interpolate"]
        );
    }

    // Suppress the dead_code warning that fires because
    // `parse_tempfile` is wired up for potential future use, but
    // every existing test reaches the file-based entry point via
    // `write_yaml_with_donors` for clarity.
    #[allow(dead_code)]
    fn _keep_parse_tempfile_alive(yaml: &str) -> Result<SurgeryPipeline, SurgeryError> {
        parse_tempfile(yaml)
    }
}
