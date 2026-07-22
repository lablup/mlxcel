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

use crate::emitter::{
    Config, emit_decode, emit_decode_batched, emit_decode_ragged, emit_prefill,
    emit_prefill_embeddings,
};

/// One emitted graph kind, matching the emitter's entry points. `sample = true`
/// ends the graph in an on-device argmax returning a token id (the single-sequence
/// session path); `sample = false` returns the raw logits (the batched serve path
/// samples on the host).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphKind {
    /// Bucketed prompt prefill ([`emit_prefill`]).
    Prefill { sample: bool },
    /// Bucketed prefill from post-scale hidden states ([`emit_prefill_embeddings`]).
    PrefillEmbeddings { sample: bool },
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
            GraphKind::PrefillEmbeddings { sample } => emit_prefill_embeddings(cfg, sample),
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
            GraphKind::PrefillEmbeddings { sample } => {
                write!(f, "prefill_embeddings(sample={sample})")
            }
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

/// Qwen2-MoE (synthetic tiny): the structural fixture for the MoE FFN primitive
/// (issue #500). A small MoE config (4 experts, top-2, `norm_topk_prob`, a gated
/// shared expert, q/k/v bias, untied head) exercising the whole routing / dispatch
/// surface in a graph small enough to commit. The goldens are the frozen StableHLO
/// the emitter produces for `assets/qwen2-moe-tiny/config.json`: on-device-argmax
/// `prefill` / `decode`, plus the host-sampled token and embeddings prefill graphs
/// and ragged `decode` serve graph (B_max 4). The routing math itself is proven
/// token-exact against an HF MoE block by the out-of-crate execution check
/// (`spike/openxla/moe_oracle.py`); this fixture then guards the emit byte-for-byte
/// forever.
pub(crate) static QWEN2_MOE_TINY: ArchFixture = ArchFixture {
    arch: "qwen2-moe-tiny",
    config_json: include_str!("../assets/qwen2-moe-tiny/config.json"),
    graphs: &[
        GraphFixture {
            kind: GraphKind::Prefill { sample: true },
            golden_name: "prefill.mlir",
            golden: include_str!("../assets/qwen2-moe-tiny/prefill.mlir"),
        },
        GraphFixture {
            kind: GraphKind::Decode { sample: true },
            golden_name: "decode.mlir",
            golden: include_str!("../assets/qwen2-moe-tiny/decode.mlir"),
        },
        GraphFixture {
            kind: GraphKind::Prefill { sample: false },
            golden_name: "prefill_logits.mlir",
            golden: include_str!("../assets/qwen2-moe-tiny/prefill_logits.mlir"),
        },
        GraphFixture {
            kind: GraphKind::PrefillEmbeddings { sample: false },
            golden_name: "prefill_embeddings_logits.mlir",
            golden: include_str!("../assets/qwen2-moe-tiny/prefill_embeddings_logits.mlir"),
        },
        GraphFixture {
            kind: GraphKind::DecodeRagged {
                b_max: 4,
                sample: false,
            },
            golden_name: "decode_ragged_logits_b4.mlir",
            golden: include_str!("../assets/qwen2-moe-tiny/decode_ragged_logits_b4.mlir"),
        },
    ],
};

/// Qwen3-MoE (synthetic tiny): the #501 fixture composing the shared qk-norm attention
/// (#497: per-head q/k RMSNorm, no q/k/v bias) with the shared MoE FFN primitive (#500)
/// and NO shared expert. A small config (4 experts, top-2, `norm_topk_prob`, head_dim 4
/// with 3 q-heads so the per-head q-norm and non-square o_proj are genuinely distinct)
/// over 2 layers keeps the routed-expert graph small enough to commit. The routing math
/// is proven token-exact against an HF fp32 Qwen3-MoE block (max block diff ~1.9e-9,
/// `spike/openxla/moe_oracle.py`) and the qk-norm attention against OLMo2/Qwen3 (#497);
/// this fixture then guards the composed emit byte-for-byte forever.
pub(crate) static QWEN3_MOE_TINY: ArchFixture = ArchFixture {
    arch: "qwen3-moe-tiny",
    config_json: include_str!("../assets/qwen3-moe-tiny/config.json"),
    graphs: &[
        GraphFixture {
            kind: GraphKind::Prefill { sample: true },
            golden_name: "prefill.mlir",
            golden: include_str!("../assets/qwen3-moe-tiny/prefill.mlir"),
        },
        GraphFixture {
            kind: GraphKind::Decode { sample: true },
            golden_name: "decode.mlir",
            golden: include_str!("../assets/qwen3-moe-tiny/decode.mlir"),
        },
        GraphFixture {
            kind: GraphKind::Prefill { sample: false },
            golden_name: "prefill_logits.mlir",
            golden: include_str!("../assets/qwen3-moe-tiny/prefill_logits.mlir"),
        },
        GraphFixture {
            kind: GraphKind::DecodeRagged {
                b_max: 4,
                sample: false,
            },
            golden_name: "decode_ragged_logits_b4.mlir",
            golden: include_str!("../assets/qwen3-moe-tiny/decode_ragged_logits_b4.mlir"),
        },
    ],
};

/// OLMoE (synthetic tiny): the #501 fixture composing the FLAT q/k RMSNorm attention
/// (#497: raw weight over the whole projection, like OLMo2) on the STANDARD pre-norm
/// block with the shared MoE FFN primitive (#500) and NO shared expert. Unlike OLMo2,
/// OLMoE keeps `input_layernorm` (standard pre-norm) and its experts use
/// `intermediate_size` (no `moe_intermediate_size`). The routing math is proven
/// token-exact against an HF fp32 OLMoE block (max block diff ~2.2e-8,
/// `spike/openxla/moe_oracle.py`); this fixture guards the composed emit.
pub(crate) static OLMOE_TINY: ArchFixture = ArchFixture {
    arch: "olmoe-tiny",
    config_json: include_str!("../assets/olmoe-tiny/config.json"),
    graphs: &[
        GraphFixture {
            kind: GraphKind::Prefill { sample: true },
            golden_name: "prefill.mlir",
            golden: include_str!("../assets/olmoe-tiny/prefill.mlir"),
        },
        GraphFixture {
            kind: GraphKind::Decode { sample: true },
            golden_name: "decode.mlir",
            golden: include_str!("../assets/olmoe-tiny/decode.mlir"),
        },
        GraphFixture {
            kind: GraphKind::Prefill { sample: false },
            golden_name: "prefill_logits.mlir",
            golden: include_str!("../assets/olmoe-tiny/prefill_logits.mlir"),
        },
        GraphFixture {
            kind: GraphKind::DecodeRagged {
                b_max: 4,
                sample: false,
            },
            golden_name: "decode_ragged_logits_b4.mlir",
            golden: include_str!("../assets/olmoe-tiny/decode_ragged_logits_b4.mlir"),
        },
    ],
};

/// A dense-pack fixture: a small synthetic `config.json` and the frozen decode +
/// prefill goldens for a family. `$sample` selects the graph tail the goldens were
/// frozen with: `true` returns the on-device argmax token (the issue #499 pack:
/// Seed-OSS / MiMo / InternLM3 / ExaOne, each reusing an already-proven Llama /
/// Qwen2 forward up to a config / naming delta), `false` returns the raw logits
/// (the issue #498 pack: Cohere / Cohere2 / Phi3 / StableLM / StarCoder2 / Granite /
/// MiniCPM, each proven token-exact against an HF fp32 oracle on the synthetic model
/// via `spike/openxla/dense_arch_check.py`). Both are trusted goldens the byte-exact
/// gate then guards against drift; the graph order in the array does not matter (each
/// fixture is compared to its own golden).
macro_rules! dense_fixture {
    ($ident:ident, $arch:literal, $sample:expr) => {
        pub(crate) static $ident: ArchFixture = ArchFixture {
            arch: $arch,
            config_json: include_str!(concat!("../assets/", $arch, "/config.json")),
            graphs: &[
                GraphFixture {
                    kind: GraphKind::Decode { sample: $sample },
                    golden_name: "decode.mlir",
                    golden: include_str!(concat!("../assets/", $arch, "/decode.mlir")),
                },
                GraphFixture {
                    kind: GraphKind::Prefill { sample: $sample },
                    golden_name: "prefill.mlir",
                    golden: include_str!(concat!("../assets/", $arch, "/prefill.mlir")),
                },
            ],
        };
    };
}

// issue #498 dense pack (raw-logits goldens, sample = false): the parallel-block
// and norm-variant families, each proven token-exact on a synthetic model before
// freezing (`spike/openxla/dense_arch_check.py`).
dense_fixture!(COHERE, "cohere", false);
dense_fixture!(COHERE2, "cohere2", false);
dense_fixture!(PHI3, "phi3", false);
dense_fixture!(STABLELM, "stablelm", false);
dense_fixture!(STARCODER2, "starcoder2", false);
dense_fixture!(GRANITE, "granite", false);
dense_fixture!(MINICPM, "minicpm", false);

// issue #499 dense pack (argmax-token goldens, sample = true): each reuses an
// already-proven Llama / Qwen2 forward up to a config / naming delta.
// Seed-OSS: q/k/v projection bias (from `attention_bias`), untied, `default` rope
// type served as plain — the proven Qwen2 bias forward with standard names.
dense_fixture!(SEED_OSS, "seed_oss", true);
// MiMo: q/k/v projection bias, untied, plain RoPE; its config `sliding_window` is
// served globally (as for Qwen2), so it parses to `sliding_window = None`.
dense_fixture!(MIMO, "mimo", true);
// InternLM3: standard names, untied, `dynamic` rope served as plain (in-context).
dense_fixture!(INTERNLM3, "internlm3", true);
// ExaOne 3.x: llama3 RoPE, tied, GPT-2-style names (the `Exaone` weight scheme)
// and the `num_layers` / `layer_norm_epsilon` alternate config fields.
dense_fixture!(EXAONE, "exaone", true);

/// Every registered structural fixture. Append a family here to add it to the
/// byte-exact gate; see the module docs for the freeze workflow. The MoE fixtures
/// (issues #500 / #501) ride the same byte-exact gate as the dense families.
pub(crate) static REGISTERED: &[&ArchFixture] = &[
    &LLAMA_3_2_1B,
    &COHERE,
    &COHERE2,
    &PHI3,
    &STABLELM,
    &STARCODER2,
    &GRANITE,
    &MINICPM,
    &SEED_OSS,
    &MIMO,
    &INTERNLM3,
    &EXAONE,
    &QWEN2_MOE_TINY,
    &QWEN3_MOE_TINY,
    &OLMOE_TINY,
];

/// A golden-less structural fixture (issue #497): a small synthetic `config.json`
/// for a dense family plus the signature every emitted shared-core graph must
/// carry. It registers a family in the harness WITHOUT bundling byte-exact goldens,
/// which suits the new dense pack: their real checkpoints are large (Gemma /
/// SmolLM3 / OLMo2), post-cutoff-heavy, or need a follow-up (OLMo3 yarn RoPE), so
/// freezing real goldens would bloat the repo and pin an un-execution-proven graph.
/// The exact op deltas are locked by the emitter's with/without-diff tests, and the
/// execution tier (`spike/openxla/dense_arch_pack_check.py`) proves correctness on a
/// small synthetic model per family. Mirrors the Qwen2.5 golden-less emit test.
pub(crate) struct StructuralFixture {
    /// Family id (e.g. `"qwen3"`).
    pub arch: &'static str,
    /// A small synthetic `config.json` carrying the family's real arch flags.
    pub config_json: &'static str,
    /// Substrings every shared-core graph (prefill / decode / ragged) must contain.
    pub must_contain: &'static [&'static str],
    /// Substrings none of those graphs may contain (the family's absent features).
    pub must_not_contain: &'static [&'static str],
}

/// The shared-core graph kinds a structural fixture is checked against (the serve
/// path: host-sampled prefill / decode logits and ragged decode).
pub(crate) const STRUCTURAL_KINDS: &[GraphKind] = &[
    GraphKind::Prefill { sample: false },
    GraphKind::Decode { sample: false },
    GraphKind::DecodeRagged {
        b_max: 4,
        sample: false,
    },
];

/// The registered golden-less dense families (issue #497). Small synthetic dims
/// (hidden 8, head_dim 4, 4 layers) keep the emit tiny while exercising every
/// arch delta; `head_dim` differs from `hidden / n_q` and `n_q*head_dim` from
/// `hidden`, so the flat q-norm and non-square o_proj widths are genuinely distinct.
pub(crate) static STRUCTURAL_FAMILIES: &[StructuralFixture] = &[
    StructuralFixture {
        arch: "qwen3",
        config_json: r#"{"model_type":"qwen3","hidden_size":8,"num_attention_heads":3,
            "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"num_hidden_layers":4,
            "rms_norm_eps":1e-6,"rope_theta":1000000,"vocab_size":12,"attention_bias":false}"#,
        must_contain: &["['q_norm']", "['k_norm']", "['in_ln']"],
        must_not_contain: &["['bq']", "['pre_ff_ln']", "['post_ff_ln']"],
    },
    StructuralFixture {
        arch: "gemma1",
        config_json: r#"{"model_type":"gemma","hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"num_hidden_layers":4,
            "rms_norm_eps":1e-6,"rope_theta":10000.0,"vocab_size":12,
            "hidden_activation":"gelu_pytorch_tanh"}"#,
        must_contain: &["stablehlo.tanh", "['in_ln']"],
        must_not_contain: &["['pre_ff_ln']", "['post_ff_ln']", "['q_norm']"],
    },
    StructuralFixture {
        arch: "gemma3",
        config_json: r#"{"model_type":"gemma3_text","hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"num_hidden_layers":4,
            "rms_norm_eps":1e-6,"rope_theta":1000000,"rope_local_base_freq":10000,
            "sliding_window":2,"sliding_window_pattern":3,"query_pre_attn_scalar":4,
            "vocab_size":12,"hidden_activation":"gelu_pytorch_tanh"}"#,
        must_contain: &[
            "['q_norm']",
            "['pre_ff_ln']",
            "['post_ff_ln']",
            "stablehlo.tanh",
        ],
        must_not_contain: &["['bq']"],
    },
    StructuralFixture {
        arch: "smollm3",
        config_json: r#"{"model_type":"smollm3","hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"intermediate_size":16,"num_hidden_layers":4,
            "rms_norm_eps":1e-6,"rope_theta":5000000.0,"vocab_size":12,
            "no_rope_layers":[1,1,1,0]}"#,
        must_contain: &["['in_ln']"],
        must_not_contain: &["['q_norm']", "['pre_ff_ln']", "['bq']"],
    },
    StructuralFixture {
        arch: "olmo2",
        config_json: r#"{"model_type":"olmo2","hidden_size":8,"num_attention_heads":3,
            "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"num_hidden_layers":4,
            "rms_norm_eps":1e-6,"rope_theta":500000,"vocab_size":12,"tie_word_embeddings":false}"#,
        must_contain: &[
            "['q_norm']",
            "['k_norm']",
            "['post_ff_ln']",
            "params['lm_head']",
        ],
        must_not_contain: &["['in_ln']", "['pre_ff_ln']"],
    },
    StructuralFixture {
        arch: "olmo3",
        config_json: r#"{"model_type":"olmo3","hidden_size":8,"num_attention_heads":3,
            "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,"num_hidden_layers":4,
            "rms_norm_eps":1e-6,"rope_theta":500000,"vocab_size":12,"sliding_window":2,
            "sliding_window_pattern":4,"tie_word_embeddings":false}"#,
        must_contain: &["['q_norm']", "['post_ff_ln']", "params['lm_head']"],
        must_not_contain: &["['in_ln']", "['pre_ff_ln']"],
    },
];

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

    /// The dense pack (issue #499) reuses the proven forward: each family's config
    /// emits StableHLO byte-for-byte identical to a proven `llama` / `qwen2`
    /// reference carrying the SAME switches and dimensions, across the single,
    /// prefill, and ragged graph kinds. This is the correctness anchor for the
    /// families without a real-checkpoint execution run in-agent: their emit is
    /// literally the Llama / Qwen2 emit the backend already runs token-exact, so
    /// the delta is confined to config parsing (asserted in `config::tests`) and
    /// tensor naming (asserted in `weight_names::tests`). The references match each
    /// fixture's small synthetic dims / eps / theta exactly (rope tables and the
    /// eps constant must line up for a byte match).
    #[test]
    fn dense_pack_families_reuse_proven_graphs() {
        // (fixture, an equivalent config expressed via a proven model_type).
        let cases: &[(&ArchFixture, &str)] = &[
            // Seed-OSS == untied Qwen2 (q/k/v bias, plain RoPE).
            (
                &SEED_OSS,
                r#"{"model_type":"qwen2","hidden_size":8,"num_attention_heads":2,
                "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,
                "num_hidden_layers":2,"rms_norm_eps":1e-6,"rope_theta":10000000.0,
                "vocab_size":10,"tie_word_embeddings":false}"#,
            ),
            // MiMo == untied Qwen2 (q/k/v bias, plain RoPE), sliding_window ignored.
            (
                &MIMO,
                r#"{"model_type":"qwen2","hidden_size":8,"num_attention_heads":2,
                "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,
                "num_hidden_layers":2,"rms_norm_eps":1e-5,"rope_theta":640000.0,
                "vocab_size":10,"tie_word_embeddings":false}"#,
            ),
            // InternLM3 == untied plain-RoPE Llama (dynamic served as plain).
            (
                &INTERNLM3,
                r#"{"model_type":"llama","hidden_size":8,"num_attention_heads":2,
                "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,
                "num_hidden_layers":2,"rms_norm_eps":1e-5,"rope_theta":50000000.0,
                "vocab_size":10,"tie_word_embeddings":false}"#,
            ),
            // ExaOne 3.x == tied llama3-RoPE Llama (the weight scheme is loader-only).
            (
                &EXAONE,
                r#"{"model_type":"llama","hidden_size":8,"num_attention_heads":2,
                "num_key_value_heads":1,"head_dim":4,"intermediate_size":16,
                "num_hidden_layers":2,"rms_norm_eps":1e-5,"rope_theta":1000000.0,
                "vocab_size":10,"tie_word_embeddings":true,"rope_scaling":{"rope_type":"llama3",
                "factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,
                "original_max_position_embeddings":8192}}"#,
            ),
        ];
        for (fx, ref_json) in cases {
            let fam = Config::from_json_str(fx.config_json)
                .unwrap_or_else(|e| panic!("{}: {e}", fx.arch));
            let refc = Config::from_json_str(ref_json)
                .unwrap_or_else(|e| panic!("{} reference: {e}", fx.arch));
            let pairs = [
                ("decode", emit_decode(&fam, true), emit_decode(&refc, true)),
                (
                    "prefill",
                    emit_prefill(&fam, false),
                    emit_prefill(&refc, false),
                ),
                (
                    "ragged",
                    emit_decode_ragged(&fam, 4, false),
                    emit_decode_ragged(&refc, 4, false),
                ),
            ];
            for (name, got, want) in pairs {
                assert_eq!(
                    got, want,
                    "{}: emitted {name} is not byte-identical to its proven equivalent",
                    fx.arch
                );
            }
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
            GraphKind::PrefillEmbeddings { sample: false },
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

    /// Every registered golden-less dense family (issue #497) parses and emits each
    /// shared-core graph kind carrying its expected structural signature: the
    /// arch's signature args / ops are present and its absent features are absent,
    /// in prefill, single decode, and ragged decode alike. This is the harness
    /// registration for the new dense pack (Qwen3, Gemma1/3, SmolLM3, OLMo2/3),
    /// whose byte-exact op deltas are pinned by the emitter's with/without-diff
    /// tests and whose correctness is proven by the execution tier.
    #[test]
    fn structural_families_emit_expected_signature() {
        for fx in STRUCTURAL_FAMILIES {
            let graphs = emit_graphs(fx.config_json, STRUCTURAL_KINDS)
                .unwrap_or_else(|e| panic!("{}: {e}", fx.arch));
            assert_eq!(
                graphs.len(),
                STRUCTURAL_KINDS.len(),
                "{}: one graph/kind",
                fx.arch
            );
            for (kind, mlir) in &graphs {
                assert!(
                    mlir.contains("stablehlo."),
                    "{} {kind}: not StableHLO",
                    fx.arch
                );
                for needle in fx.must_contain {
                    assert!(
                        mlir.contains(needle),
                        "{} {kind}: missing signature {needle:?}",
                        fx.arch
                    );
                }
                for needle in fx.must_not_contain {
                    assert!(
                        !mlir.contains(needle),
                        "{} {kind}: unexpected {needle:?}",
                        fx.arch
                    );
                }
            }
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

    /// Freeze the MoE structural goldens (issues #500 / #501) for the synthetic tiny
    /// MoE fixtures: qwen2_moe (shared expert, #500), qwen3_moe and olmoe (no shared
    /// expert on the qk-norm attention core, #501). A no-op unless `MLXCEL_FREEZE_MOE=1`;
    /// when set it emits each graph from `assets/<arch>/config.json` and writes the
    /// `.mlir` goldens via the bootstrap [`emit_graphs`] path (no committed golden
    /// needed), so a brand-new family authors its goldens before its [`ArchFixture`]
    /// exists. Run once to author them; the byte-exact gate then guards them forever.
    #[test]
    fn freeze_moe_goldens() {
        if std::env::var("MLXCEL_FREEZE_MOE").as_deref() != Ok("1") {
            return;
        }
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        let targets: [(GraphKind, &str); 4] = [
            (GraphKind::Decode { sample: true }, "decode.mlir"),
            (GraphKind::Prefill { sample: true }, "prefill.mlir"),
            (GraphKind::Prefill { sample: false }, "prefill_logits.mlir"),
            (
                GraphKind::DecodeRagged {
                    b_max: 4,
                    sample: false,
                },
                "decode_ragged_logits_b4.mlir",
            ),
        ];
        for arch in ["qwen2-moe-tiny", "qwen3-moe-tiny", "olmoe-tiny"] {
            let root = assets.join(arch);
            let cfg_json =
                std::fs::read_to_string(root.join("config.json")).expect("read config.json");
            for (kind, name) in targets {
                let graphs = emit_graphs(&cfg_json, &[kind])
                    .unwrap_or_else(|e| panic!("{arch} config parses: {e}"));
                let (_, mlir) = &graphs[0];
                let p = root.join(name);
                std::fs::write(&p, mlir).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
                eprintln!("froze {}", p.display());
            }
            if arch == "qwen2-moe-tiny" {
                let kind = GraphKind::PrefillEmbeddings { sample: false };
                let graphs = emit_graphs(&cfg_json, &[kind])
                    .unwrap_or_else(|e| panic!("{arch} config parses: {e}"));
                let p = root.join("prefill_embeddings_logits.mlir");
                std::fs::write(&p, &graphs[0].1)
                    .unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
                eprintln!("froze {}", p.display());
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
