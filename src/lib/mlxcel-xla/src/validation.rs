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

//! Reusable per-architecture validation harness (issue #496).
//!
//! Adding an architecture to the OpenXLA backend needs two kinds of proof, and
//! this crate splits them into two tiers so the cheap one can gate every change
//! and the expensive one stays opt-in:
//!
//! 1. **Structural (byte-exact) - this module.** The Rust emitter must reproduce
//!    a frozen, trusted StableHLO graph for the architecture, byte for byte. It
//!    is pure Rust (no IREE, no GPU, no checkpoint), so it is the cheap
//!    regression gate that catches a graph which drifted from its validated form.
//!    [`check_arch`] parses a fixture's `config.json`, emits each declared graph,
//!    and diffs it against the committed golden `assets/<arch>/*.mlir`, localizing
//!    the first differing line on a mismatch.
//!
//! 2. **Execution (token-exact / reference-exact) - outside this crate.** The
//!    compiled graph must run token-exact against an HF fp32 oracle (single
//!    sequence) and reference-exact through the batched serve path. That needs
//!    real IREE execution and a checkpoint, so it lives as the `xla_oracle_check`
//!    and `xla_batch_bench` examples, driven end to end (oracle included) by
//!    `scripts/xla/validate_arch.sh`. See that script and the crate `README.md`.
//!
//! # Adding a family is turnkey
//!
//! Freezing goldens is a one-time authoring step, gated on the execution tier
//! having proven the emit is correct for a real checkpoint:
//!
//! 1. Prove token-exactness with `scripts/xla/validate_arch.sh --model <ckpt>`.
//! 2. Freeze the now-trusted graphs to `assets/<arch>/`: [`emit_graphs`] emits
//!    each graph from the checkpoint's `config.json` with no committed golden
//!    required, so it bootstraps a brand-new family; write each result to its
//!    `.mlir` file (the [`ArchFixture::emit_all`] / `freeze_goldens` path
//!    re-freezes an already-registered family).
//! 3. Register the fixture: add an [`ArchFixture`] pairing the config with its
//!    golden `.mlir` files and append it to [`REGISTERED`]. The data-driven test
//!    then guards it byte-for-byte forever.
//!
//! Not every family bundles goldens: an architecture whose graphs are emitted at
//! load (Qwen2.5, see `assets/qwen2.5-0.5b/README.md`) is covered by the
//! emitter's structural invariant tests and the execution tier instead. The
//! harness still drives its emit through [`emit_graphs`] (see the tests below).
//!
//! The emitter reads `MLXCEL_XLA_PRECISION` at emit time (default `f32`); the
//! committed goldens are the default-precision graphs, so [`check_arch`] rejects
//! a byte-exact run under a non-default precision (`f16` / `bf16`) with a clear
//! error rather than reporting a confusing diff.

use crate::emitter::{Config, emit_decode, emit_decode_batched, emit_decode_ragged, emit_prefill};

/// One emitted graph kind, matching the emitter's entry points. `sample = true`
/// ends the graph in an on-device argmax returning a token id (the single-sequence
/// session path); `sample = false` returns the raw logits (the batched serve path
/// samples on the host).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphKind {
    /// Bucketed prompt prefill ([`emit_prefill`]).
    Prefill { sample: bool },
    /// Single-token decode step ([`emit_decode`]).
    Decode { sample: bool },
    /// Ragged (continuous-batching) decode for `b_max` slots ([`emit_decode_ragged`]).
    DecodeRagged { b_max: usize, sample: bool },
    /// Uniform-`b_max` batched decode ([`emit_decode_batched`]).
    DecodeBatched { b_max: usize, sample: bool },
}

impl GraphKind {
    /// Emit this graph for `cfg`, at the ambient `MLXCEL_XLA_PRECISION`.
    fn emit(self, cfg: &Config) -> String {
        match self {
            GraphKind::Prefill { sample } => emit_prefill(cfg, sample),
            GraphKind::Decode { sample } => emit_decode(cfg, sample),
            GraphKind::DecodeRagged { b_max, sample } => emit_decode_ragged(cfg, b_max, sample),
            GraphKind::DecodeBatched { b_max, sample } => emit_decode_batched(cfg, b_max, sample),
        }
    }
}

impl core::fmt::Display for GraphKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GraphKind::Prefill { sample } => write!(f, "prefill(sample={sample})"),
            GraphKind::Decode { sample } => write!(f, "decode(sample={sample})"),
            GraphKind::DecodeRagged { b_max, sample } => {
                write!(f, "decode_ragged(b={b_max}, sample={sample})")
            }
            GraphKind::DecodeBatched { b_max, sample } => {
                write!(f, "decode_batched(b={b_max}, sample={sample})")
            }
        }
    }
}

/// One golden graph within an architecture fixture: which graph to emit and the
/// expected MLIR (the committed `assets/<arch>/<golden_name>`).
pub(crate) struct GraphFixture {
    /// The graph kind to emit and compare.
    pub kind: GraphKind,
    /// File name under `assets/<arch>/`, used in diagnostics and by the freeze path.
    pub golden_name: &'static str,
    /// The golden MLIR text (`include_str!` of the committed asset).
    pub golden: &'static str,
}

/// A per-architecture structural fixture: a checkpoint's `config.json` and the
/// set of graphs the emitter must reproduce byte-for-byte.
pub(crate) struct ArchFixture {
    /// Directory name under `assets/` and id in reports (e.g. `"llama-3.2-1b"`).
    pub arch: &'static str,
    /// The model's `config.json` text (`include_str!` of the committed asset).
    pub config_json: &'static str,
    /// The golden graphs this architecture pins.
    pub graphs: &'static [GraphFixture],
}

impl ArchFixture {
    /// Emit every declared graph, returning `(golden_name, mlir)` pairs. The
    /// re-freeze primitive for an already-registered family: write each pair to
    /// `assets/<arch>/<golden_name>` to refresh the goldens after an intentional,
    /// execution-tier-validated emitter change. Emits at the ambient
    /// `MLXCEL_XLA_PRECISION`.
    ///
    /// # Errors
    /// Returns the config-parse error if `config_json` is not a supported arch.
    pub fn emit_all(&self) -> Result<Vec<(&'static str, String)>, String> {
        let cfg =
            Config::from_json_str(self.config_json).map_err(|e| format!("{}: {e}", self.arch))?;
        Ok(self
            .graphs
            .iter()
            .map(|g| (g.golden_name, g.kind.emit(&cfg)))
            .collect())
    }
}

/// Emit each requested graph for the architecture in `config_json`, at the ambient
/// `MLXCEL_XLA_PRECISION`. The authoring/bootstrap primitive: it needs no committed
/// golden, so it can freeze a brand-new family before its [`ArchFixture`] exists.
///
/// # Errors
/// Returns the config-parse error if `config_json` is not a supported architecture.
pub(crate) fn emit_graphs(
    config_json: &str,
    kinds: &[GraphKind],
) -> Result<Vec<(GraphKind, String)>, String> {
    let cfg = Config::from_json_str(config_json)?;
    Ok(kinds.iter().map(|&k| (k, k.emit(&cfg))).collect())
}

/// The first line at which two MLIR texts differ (1-based), with both sides, so a
/// frozen-reference drift is easy to localize.
pub(crate) struct LineDiff {
    /// 1-based line number of the first difference.
    pub line: usize,
    /// The golden's line at `line` (`<end of file>` if the golden is shorter).
    pub expected: String,
    /// The emitted line at `line` (`<end of file>` if the emit is shorter).
    pub actual: String,
    /// Total lines in the golden.
    pub expected_lines: usize,
    /// Total lines in the emitted graph.
    pub actual_lines: usize,
}

/// The outcome of checking one golden graph.
pub(crate) struct GraphOutcome {
    /// The graph kind that was emitted.
    pub kind: GraphKind,
    /// The golden file name it was compared against.
    pub golden_name: &'static str,
    /// `None` on a byte-exact match; the first differing line otherwise.
    pub diff: Option<LineDiff>,
}

impl GraphOutcome {
    /// The emit matched the golden byte-for-byte.
    pub fn matched(&self) -> bool {
        self.diff.is_none()
    }
}

/// The outcome of checking one architecture fixture.
pub(crate) struct ArchReport {
    /// The fixture's architecture id.
    pub arch: &'static str,
    /// Per-graph outcomes, in fixture order.
    pub graphs: Vec<GraphOutcome>,
}

impl ArchReport {
    /// Every golden matched byte-for-byte.
    pub fn passed(&self) -> bool {
        self.graphs.iter().all(GraphOutcome::matched)
    }
}

impl core::fmt::Display for ArchReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "arch {}: {}",
            self.arch,
            if self.passed() { "PASS" } else { "FAIL" }
        )?;
        for g in &self.graphs {
            match &g.diff {
                None => writeln!(f, "  ok   {} ({})", g.golden_name, g.kind)?,
                Some(d) => writeln!(
                    f,
                    "  DIFF {} ({}): first mismatch at line {} ({} golden lines vs {} emitted)\n\
                     \x20   golden : {}\n\
                     \x20   emitted: {}",
                    g.golden_name,
                    g.kind,
                    d.line,
                    d.expected_lines,
                    d.actual_lines,
                    d.expected,
                    d.actual
                )?,
            }
        }
        Ok(())
    }
}

/// Whether an `MLXCEL_XLA_PRECISION` value selects the default (`f32`) emit that
/// the committed goldens were frozen at. Only `"f16"` / `"bf16"` are non-default,
/// mirroring `emitter::builder::precision_from_env`.
fn precision_is_default(value: Option<&str>) -> bool {
    !matches!(value, Some("f16") | Some("bf16"))
}

/// Reject a byte-exact run under a non-default precision: the goldens are the f32
/// graphs, so an `f16` / `bf16` emit would diff confusingly against them.
fn ambient_precision_is_default() -> Result<(), String> {
    let v = std::env::var("MLXCEL_XLA_PRECISION").ok();
    if precision_is_default(v.as_deref()) {
        Ok(())
    } else {
        Err(format!(
            "the structural goldens are the default-precision (f32) graphs; unset \
             MLXCEL_XLA_PRECISION (currently {:?}) to run the byte-exact check",
            v.unwrap_or_default()
        ))
    }
}

/// Parse a fixture's `config.json`, emit each declared graph, and compare it to the
/// committed golden byte-for-byte, localizing the first differing line on a
/// mismatch.
///
/// The committed goldens are the default-precision (`f32`) graphs, so this errors
/// if `MLXCEL_XLA_PRECISION` selects `f16` / `bf16` (which would emit a different
/// graph). Inspect [`ArchReport::passed`]; the `Display` renders a per-graph
/// pass/fail summary with the first-diff location.
///
/// # Errors
/// Returns an error if the ambient precision is not the default the goldens were
/// frozen at, or if `config.json` does not parse to a supported architecture.
pub(crate) fn check_arch(fx: &ArchFixture) -> Result<ArchReport, String> {
    ambient_precision_is_default()?;
    let cfg = Config::from_json_str(fx.config_json).map_err(|e| format!("{}: {e}", fx.arch))?;
    let graphs = fx
        .graphs
        .iter()
        .map(|g| {
            let emitted = g.kind.emit(&cfg);
            GraphOutcome {
                kind: g.kind,
                golden_name: g.golden_name,
                diff: first_line_diff(g.golden, &emitted),
            }
        })
        .collect();
    Ok(ArchReport {
        arch: fx.arch,
        graphs,
    })
}

/// The first line at which `golden` and `emitted` differ, or `None` if identical.
fn first_line_diff(golden: &str, emitted: &str) -> Option<LineDiff> {
    if golden == emitted {
        return None;
    }
    let (mut g, mut e) = (golden.lines(), emitted.lines());
    let mut line = 0usize;
    loop {
        line += 1;
        match (g.next(), e.next()) {
            (Some(a), Some(b)) if a == b => {}
            (a, b) => {
                return Some(LineDiff {
                    line,
                    expected: a.unwrap_or("<end of file>").to_string(),
                    actual: b.unwrap_or("<end of file>").to_string(),
                    expected_lines: golden.lines().count(),
                    actual_lines: emitted.lines().count(),
                });
            }
        }
    }
}

/// Llama-3.2-1B-Instruct: the reference architecture, with committed goldens. Its
/// graphs are the issue #451-emitted StableHLO the backend ships in
/// `assets/llama-3.2-1b/` (on-device-argmax `prefill` / `decode`, plus the
/// host-sampled `prefill_logits` and ragged `decode` serve graphs for B_max 4/8).
pub(crate) static LLAMA_3_2_1B: ArchFixture = ArchFixture {
    arch: "llama-3.2-1b",
    config_json: include_str!("../assets/llama-3.2-1b/config.json"),
    graphs: &[
        GraphFixture {
            kind: GraphKind::Prefill { sample: true },
            golden_name: "prefill.mlir",
            golden: include_str!("../assets/llama-3.2-1b/prefill.mlir"),
        },
        GraphFixture {
            kind: GraphKind::Decode { sample: true },
            golden_name: "decode.mlir",
            golden: include_str!("../assets/llama-3.2-1b/decode.mlir"),
        },
        GraphFixture {
            kind: GraphKind::Prefill { sample: false },
            golden_name: "prefill_logits.mlir",
            golden: include_str!("../assets/llama-3.2-1b/prefill_logits.mlir"),
        },
        GraphFixture {
            kind: GraphKind::DecodeRagged {
                b_max: 4,
                sample: false,
            },
            golden_name: "decode_ragged_logits_b4.mlir",
            golden: include_str!("../assets/llama-3.2-1b/decode_ragged_logits_b4.mlir"),
        },
        GraphFixture {
            kind: GraphKind::DecodeRagged {
                b_max: 8,
                sample: false,
            },
            golden_name: "decode_ragged_logits_b8.mlir",
            golden: include_str!("../assets/llama-3.2-1b/decode_ragged_logits_b8.mlir"),
        },
    ],
};

/// Every registered structural fixture. Append a family here to add it to the
/// byte-exact gate; see the module docs for the freeze workflow.
pub(crate) static REGISTERED: &[&ArchFixture] = &[&LLAMA_3_2_1B];

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_CONFIG_JSON: &str = include_str!("../assets/qwen2.5-0.5b/config.json");

    /// The structural gate: every registered fixture emits its committed goldens
    /// byte-for-byte. Demonstrates llama-3.2-1b passing (issue #496); this is the
    /// single source of truth the emitter's `from_json_reproduces_bundled_assets`
    /// test delegates to.
    #[test]
    fn registered_fixtures_are_byte_exact() {
        for fx in REGISTERED {
            let report = check_arch(fx).unwrap_or_else(|e| panic!("{}: {e}", fx.arch));
            assert!(report.passed(), "{report}");
        }
    }

    /// The guard actually catches drift: a wrong golden fails and the report
    /// localizes the first differing line, so a downstream emitter change that
    /// shifts a graph is caught with a pointer, not a wall of text.
    #[test]
    fn check_arch_detects_drift() {
        static DRIFT_GRAPHS: [GraphFixture; 1] = [GraphFixture {
            kind: GraphKind::Decode { sample: true },
            golden_name: "decode.mlir",
            golden: "// intentionally wrong first line\n",
        }];
        let drifted = ArchFixture {
            arch: "llama-3.2-1b (drift check)",
            config_json: include_str!("../assets/llama-3.2-1b/config.json"),
            graphs: &DRIFT_GRAPHS,
        };
        let report = check_arch(&drifted).expect("config parses");
        assert!(!report.passed(), "a wrong golden must fail");
        let d = report.graphs[0]
            .diff
            .as_ref()
            .expect("a mismatch is localized");
        assert_eq!(d.line, 1, "the first line differs");
    }

    /// The harness drives a second, golden-less architecture end to end through
    /// the emit path: Qwen2.5-0.5B parses and every graph kind emits non-empty
    /// StableHLO carrying the Qwen2 q/k/v bias args, proving the harness is
    /// arch-agnostic without bundling large goldens (Qwen graphs are emitted at
    /// load; see `assets/qwen2.5-0.5b/README.md`).
    #[test]
    fn emit_graphs_drives_a_golden_less_arch() {
        let kinds = [
            GraphKind::Prefill { sample: true },
            GraphKind::Decode { sample: false },
            GraphKind::DecodeRagged {
                b_max: 4,
                sample: false,
            },
            GraphKind::DecodeBatched {
                b_max: 4,
                sample: false,
            },
        ];
        let graphs = emit_graphs(QWEN_CONFIG_JSON, &kinds).expect("qwen2.5-0.5b config parses");
        assert_eq!(graphs.len(), kinds.len(), "one graph per requested kind");
        for (kind, mlir) in &graphs {
            assert!(!mlir.is_empty(), "{kind} emitted empty MLIR");
            assert!(mlir.contains("stablehlo."), "{kind} is not StableHLO");
            assert!(
                mlir.contains("['bq']"),
                "{kind} missing the Qwen2 q bias arg"
            );
        }
    }

    /// The precision guard is a pure predicate over the env value, so it needs no
    /// racy env mutation to test: only `f16` / `bf16` are non-default.
    #[test]
    fn precision_default_detection() {
        assert!(precision_is_default(None), "unset is the f32 default");
        assert!(precision_is_default(Some("f32")));
        assert!(precision_is_default(Some("anything-else")));
        assert!(!precision_is_default(Some("f16")));
        assert!(!precision_is_default(Some("bf16")));
    }

    /// `emit_all` reproduces the committed goldens for a registered fixture, so
    /// re-freezing is a no-op when the emitter is unchanged (the freeze path is
    /// safe to run) and the `(golden_name, mlir)` pairing lines up with the files.
    #[test]
    fn emit_all_reproduces_registered_goldens() {
        for fx in REGISTERED {
            let emitted = fx.emit_all().unwrap_or_else(|e| panic!("{}: {e}", fx.arch));
            assert_eq!(emitted.len(), fx.graphs.len());
            for ((name, mlir), gf) in emitted.iter().zip(fx.graphs) {
                assert_eq!(*name, gf.golden_name);
                assert_eq!(mlir, gf.golden, "{}/{name} re-emits its golden", fx.arch);
            }
        }
    }

    /// Re-freeze the committed goldens for every registered fixture. A no-op unless
    /// `MLXCEL_FREEZE_GOLDENS=1`; when set it rewrites each `assets/<arch>/*.mlir`
    /// from the current emitter output. Run it to refresh goldens after an
    /// intentional, execution-tier-validated emitter change; the byte-exact test
    /// then guards the refreshed files. A brand-new (unregistered) family freezes
    /// with [`emit_graphs`] before its fixture exists (see the module docs).
    #[test]
    fn freeze_goldens() {
        if std::env::var("MLXCEL_FREEZE_GOLDENS").as_deref() != Ok("1") {
            return;
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        for fx in REGISTERED {
            for (name, mlir) in fx.emit_all().unwrap_or_else(|e| panic!("{}: {e}", fx.arch)) {
                let p = root.join(fx.arch).join(name);
                std::fs::write(&p, mlir).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
                eprintln!("froze {}", p.display());
            }
        }
    }
}
