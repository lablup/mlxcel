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

//! Tactics and the [`TunableOp`] contract (issue #906).
//!
//! A **tactic** is one concrete launch configuration for an op: a
//! `NumSplits` for the paged-decode kernel, a `(tile_m, tile_n, tile_k)` for
//! the Blackwell qmm, a row-window ceiling for multirow qmv. Tactics are
//! deliberately just an integer vector plus a label: the autotuner never
//! interprets them, it only times them and remembers which one was fastest.
//! The op that produced a tactic is the only code that knows what the numbers
//! mean.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::bucket::ShapeBucket;

/// One candidate launch configuration.
///
/// `label` is for humans (logs, cache files, reports). `params` is what the op
/// consumes. Equality is over both, so two tactics with the same parameters but
/// different labels are distinct; ops should therefore derive the label from
/// the parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tactic {
    /// Human-readable identity, e.g. `"num_splits=8"` or `"tile_m=128"`.
    pub label: String,
    /// Integer parameters, interpreted by the owning op.
    pub params: Vec<i64>,
}

impl Tactic {
    /// Build a tactic from a label and its parameters.
    #[must_use]
    pub fn new(label: impl Into<String>, params: Vec<i64>) -> Self {
        Self {
            label: label.into(),
            params,
        }
    }

    /// Single-parameter tactic labelled `"<name>=<value>"`. The common shape:
    /// most tunable launch knobs in this tree are one integer.
    #[must_use]
    pub fn scalar(name: &str, value: i64) -> Self {
        Self {
            label: format!("{name}={value}"),
            params: vec![value],
        }
    }

    /// Parameter `i`, or `None` when the tactic carries fewer. Ops read their
    /// parameters through this so a tactic deserialized from an older cache
    /// file with fewer parameters is rejected rather than mis-indexed.
    #[must_use]
    pub fn param(&self, i: usize) -> Option<i64> {
        self.params.get(i).copied()
    }
}

impl fmt::Display for Tactic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label)
    }
}

/// Failure of a single measured candidate run.
///
/// A candidate that fails is dropped from the sweep rather than aborting it:
/// a launch configuration can be infeasible on a given device (threadgroup
/// memory, register pressure, a JIT rejection) and the correct response is to
/// pick among the ones that did run.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TuneError {
    /// The candidate could not be launched on this device or shape.
    #[error("tactic {tactic} is infeasible: {reason}")]
    Infeasible { tactic: String, reason: String },
    /// The candidate launched but the run failed.
    #[error("tactic {tactic} failed: {reason}")]
    Failed { tactic: String, reason: String },
}

impl TuneError {
    #[must_use]
    pub fn infeasible(tactic: &Tactic, reason: impl Into<String>) -> Self {
        Self::Infeasible {
            tactic: tactic.label.clone(),
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn failed(tactic: &Tactic, reason: impl Into<String>) -> Self {
        Self::Failed {
            tactic: tactic.label.clone(),
            reason: reason.into(),
        }
    }
}

/// An op the autotuner can profile.
///
/// Implementors supply the key material (`op_name` / `runner_id` / `dtype_tag`),
/// the candidate space for a bucket, the default the runtime would have used
/// without tuning, and a `run` that performs exactly one invocation of the op
/// under a given tactic.
///
/// ## The `run` + `sync` contract
///
/// `run` must submit the work and force it to be evaluated (`eval` on the
/// produced arrays); `sync` must then block until the backend has actually
/// finished. The harness times `run` followed by `sync`, so an implementation
/// that forgets to eval measures graph construction rather than the kernel.
/// `sync` has a default implementation that calls MLX's default-stream
/// synchronize; test doubles override it.
pub trait TunableOp {
    /// Stable logical op name, e.g. `"paged_attention_decode"`.
    fn op_name(&self) -> &str;

    /// Kernel / launcher identity, e.g. `"metal"` or `"cuda"`. Distinguishes
    /// two implementations of the same logical op so they never share a cache
    /// entry.
    fn runner_id(&self) -> String;

    /// Extra key material, conventionally the dtype the op runs in. Ops whose
    /// tactic choice is dtype-invariant return a fixed tag (for example
    /// `"f32"` for the paged-decode kernel, which always accumulates in f32).
    fn dtype_tag(&self) -> String;

    /// Bucketed launch shape for this invocation. The harness uses it as part
    /// of the cache key and passes it to [`Self::candidates`].
    fn bucket(&self) -> ShapeBucket;

    /// Candidate tactics for `bucket`, cheapest-to-describe first. An empty
    /// list means the op has nothing to tune at this shape; the resolver then
    /// reports out-of-bucket and answers with [`Self::default_tactic`].
    fn candidates(&self, bucket: &ShapeBucket) -> Vec<Tactic>;

    /// What the runtime would use with the autotuner off. Returned on every
    /// fallback path, and identified inside the sweep so a report can state
    /// the tuned-vs-default delta.
    fn default_tactic(&self, bucket: &ShapeBucket) -> Tactic;

    /// An explicitly-set environment override, if this op has one.
    ///
    /// This is the single place the "an explicitly-set env var always wins over
    /// a tuned value" rule is enforced: when this returns `Some`, the resolver
    /// returns it without reading the cache and without profiling. Ops with no
    /// env knob keep the `None` default.
    fn env_override(&self) -> Option<Tactic> {
        None
    }

    /// Whether this op may be profiled lazily on first use inside a serving
    /// process (`MLXCEL_AUTOTUNE=1`).
    ///
    /// `false` restricts the op to the offline `mlxcel tune` path, which
    /// consumes cached entries at runtime but never profiles them. Ops whose
    /// `run` mutates process-wide state (an environment knob a C++ kernel
    /// reads) must return `false`: that mutation is only sound while the
    /// process is effectively single-threaded, which a server is not.
    fn lazy_tunable(&self) -> bool {
        true
    }

    /// Perform one invocation under `tactic`, evaluating its outputs.
    fn run(&self, tactic: &Tactic) -> Result<(), TuneError>;

    /// Block until the backend has finished the work `run` submitted.
    /// Overridden by test doubles that have no backend.
    fn sync(&self) {
        crate::synchronize_default();
    }
}
