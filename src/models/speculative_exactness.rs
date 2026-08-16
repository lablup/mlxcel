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

//! Runtime block-vs-chain exactness gate for MTP speculative decoding.
//!
//! ## Why this exists
//!
//! MTP's temperature-0 contract is that a `T = K` verify block emits the
//! same tokens as `K` single-token decode steps. Issue #1165 shipped a
//! static gate for that contract: Metal available, plus a gated-delta
//! geometry check. Both conditions are necessary and neither is
//! sufficient, because the contract also depends on **which MLX kernel a
//! quantized projection dispatches to at `M = K` versus `M = 1`**, and
//! that is decided per GPU generation, per quantization mode, per operand
//! size and per block width:
//!
//! - `use_qmv_wide` in
//!   [`mlx/backend/metal/quantized.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/quantized.cpp)
//!   is `mode != "affine" || arch_gen >= 15`. When it holds, `M >= 2`
//!   takes `qmv_wide` while `M == 1` takes `qmv`, and the two reduce over
//!   K with different lane counts. Measured: an affine 4-bit projection is
//!   byte-equal below the batch limit on an M1 Ultra (generation 13) and
//!   diverges from `M = 2` on an M5 Max (generation 17+). An mxfp4
//!   projection diverges from `M = 2` on **both**, by 1 to 11 bytes of
//!   10240 rather than the 39% the affine row moves once it breaks.
//! - `get_qmv_batch_limit` decides where the matrix-matrix kernel takes
//!   over, and it is 10, 12, 18 or 32 depending on architecture size and
//!   generation, not one number. With both operands above 4096 it is 12
//!   on an M1 Ultra (`arch_size == 'd'`) and 10 on an M5 Max (the
//!   `default` branch), so a wide enough block forfeits byte-identity
//!   even on a generation whose `qmv` path is otherwise exact, and
//!   `--draft-block-size` is user-settable. This threshold is only
//!   observable where `use_qmv_wide` is false: from generation 15 the
//!   `M >= 2` split fires first and hides it.
//!
//! Encoding those rules on our side would be correct today and would go
//! stale the next time MLX retunes a threshold, which is the same failure
//! mode as the static gate this replaces. So the gate **measures** the
//! property on the loaded model instead of predicting it: run one verify
//! block and the equivalent single-token chain from the same state, and
//! compare the logits byte for byte.
//!
//! ## What a passing probe does and does not prove
//!
//! A diverging probe is conclusive: the arms provably differ, and no
//! amount of further sampling can make them agree.
//!
//! A passing probe is weaker than it first looks, and the earlier version
//! of this note overstated it. It said that byte-equal logits mean both
//! arms took the same kernels, since MLX dispatch depends on shape,
//! dtype, quantization mode and `M` and not on token values. The
//! dispatch half is true; the inference from it is not. Two *different*
//! kernels can still produce byte-identical output on a particular input
//! when their disagreement is small enough. Measured at op level on
//! 2026-08-17: mxfp4 group 32 at 5120 -> 5120 moves 1 to 11 bytes of
//! 10240 depending only on the operand draw, while the affine row at the
//! same shape moves about 39%. In that low-amplitude regime a single
//! input is a coin toss, and one M5 Max harness did read it as equal
//! before a seed sweep showed otherwise.
//!
//! So the probe compares several independent inputs
//! (`PROBE_DRAWS` in `models::qwen3_5`) and only reports equality when
//! every one of them agrees. Model-level amplification helps too: a
//! per-projection last-ulp difference propagates through 60-odd layers
//! before it reaches the logits, so the amplitude at the point of
//! comparison is far above the fragile regime. Neither argument is a
//! proof for every future input; together they are what this gate rests
//! on, and the failure they guard against is a false *pass*.
//!
//! ## Failure policy
//!
//! Fail closed. A model whose probe diverges declines MTP and runs
//! classic decode, because a silent loss of byte-identity is worse than a
//! lost speedup. `MLXCEL_MTP_ALLOW_INEXACT=1` overrides that for
//! throughput research on hardware where MTP pays but the probe fails
//! (Apple GPU generation 15 and newer today); it is loud, and it is the
//! only way to opt out.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Outcome of one block-vs-chain probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockChainExactness {
    /// Every verify position was byte-identical to its single-token step.
    Equal,
    /// At least one position differed. Carries the first one.
    Diverges {
        /// Zero-based position inside the verify block.
        position: usize,
        /// Bytes that differ at that position.
        differing_bytes: usize,
        /// Bytes compared at that position (one row of logits).
        total_bytes: usize,
    },
    /// The probe could not run (no Metal device, degenerate block width).
    /// Treated as a decline, same as a divergence, so an un-runnable probe
    /// never reads as a pass.
    NotRun(&'static str),
}

impl BlockChainExactness {
    pub fn is_equal(&self) -> bool {
        matches!(self, Self::Equal)
    }

    /// One-line reason, for the decline log and the CLI error.
    pub fn reason(&self) -> String {
        match self {
            Self::Equal => "verify block is byte-identical to the single-token chain".to_string(),
            Self::Diverges {
                position,
                differing_bytes,
                total_bytes,
            } => format!(
                "verify block position {position} differs from the single-token \
                 chain in {differing_bytes} of {total_bytes} logit bytes"
            ),
            Self::NotRun(why) => format!("exactness probe did not run: {why}"),
        }
    }
}

/// Identity of a probe result, so one process can hold verdicts for more
/// than one block width (the offline CLI can be re-run in-process by
/// tests) without re-measuring.
///
/// The model discriminator is coarse on purpose: a process serves one
/// target model, and the fields only need to distinguish "a different
/// model was loaded" from "the same one".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProbeKey {
    pub block_size: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
}

/// `MLXCEL_MTP_ALLOW_INEXACT`: engage MTP even when the probe says the
/// verify block is not byte-identical to classic decode.
///
/// Read once per process, mirroring the other `MLXCEL_*` switches. Off by
/// default: the exactness contract is the reason the gate exists.
pub fn allow_inexact() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("MLXCEL_MTP_ALLOW_INEXACT")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                matches!(v.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

fn verdict_cache() -> &'static Mutex<HashMap<ProbeKey, bool>> {
    static CACHE: OnceLock<Mutex<HashMap<ProbeKey, bool>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether MTP may engage for `key`, running `probe` at most once per key.
///
/// `probe` is only invoked on a cache miss, so the measurement cost (two
/// short prefills plus `K + 1` forwards; measured 4.9 s for the first
/// call and 1.3 s for a later one on a 27B target on an M1 Ultra, the
/// difference being MLX's one-time kernel compilation) is paid once per
/// process and never on the request path after that. The decision, not the raw verdict, is what gets cached: with
/// `MLXCEL_MTP_ALLOW_INEXACT` set a diverging probe still returns `true`,
/// and the log line says so.
pub fn mtp_exactness_gate<F>(key: ProbeKey, probe: F) -> bool
where
    F: FnOnce() -> BlockChainExactness,
{
    if let Ok(cache) = verdict_cache().lock()
        && let Some(decision) = cache.get(&key)
    {
        return *decision;
    }

    let verdict = probe();
    let decision = verdict.is_equal() || allow_inexact();

    if verdict.is_equal() {
        tracing::info!(
            block_size = key.block_size,
            "MTP exactness probe passed: {}",
            verdict.reason()
        );
    } else if decision {
        tracing::warn!(
            block_size = key.block_size,
            "MTP exactness probe FAILED but MLXCEL_MTP_ALLOW_INEXACT is set: {}. \
             Temperature-0 speculative output will NOT be byte-identical to \
             classic decode on this host.",
            verdict.reason()
        );
    } else {
        tracing::warn!(
            block_size = key.block_size,
            "MTP declined: {}. Falling back to classic decode. Set \
             MLXCEL_MTP_ALLOW_INEXACT=1 to engage anyway and forfeit the \
             temperature-0 byte-identity contract.",
            verdict.reason()
        );
    }

    if let Ok(mut cache) = verdict_cache().lock() {
        cache.insert(key, decision);
    }
    decision
}

/// Compare one verify block's per-position logits against the
/// single-token chain's, both supplied as raw little-endian bytes.
///
/// Split out from the model-driving code so the comparison itself is
/// unit-testable without a Metal device or a checkpoint.
pub fn compare_block_against_chain(
    block_positions: &[Vec<u8>],
    chain_positions: &[Vec<u8>],
) -> BlockChainExactness {
    if block_positions.len() != chain_positions.len() {
        return BlockChainExactness::NotRun("arm lengths differ");
    }
    if block_positions.is_empty() {
        return BlockChainExactness::NotRun("no positions compared");
    }
    for (position, (block, chain)) in block_positions.iter().zip(chain_positions).enumerate() {
        if block.len() != chain.len() {
            return BlockChainExactness::NotRun("arm logit widths differ");
        }
        let differing_bytes = block.iter().zip(chain).filter(|(a, b)| a != b).count();
        if differing_bytes > 0 {
            return BlockChainExactness::Diverges {
                position,
                differing_bytes,
                total_bytes: block.len(),
            };
        }
    }
    BlockChainExactness::Equal
}

#[cfg(test)]
#[path = "speculative_exactness_tests.rs"]
mod speculative_exactness_tests;
