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

//! Concrete [`crate::SurgeryOp`] implementations for Axis A
//! (issues A5–A9).
//!
//! The submodules in here each own a single operation. They share two
//! conventions:
//!
//! - Public struct is the bare op name in PascalCase
//!   ([`scale::ScaleOp`], [`add::AddOp`], [`prune::PruneOp`],
//!   [`replace::ReplaceOp`], [`interpolate::InterpolateOp`]). Each is
//!   `Send + Sync` and stateless across `apply` calls.
//! - [`split_mtp`] is the one whole-checkpoint transform rather than a
//!   pipeline op: it reads a raw GLM-4.7-Flash checkpoint and writes a
//!   standalone `glm4_moe_lite_mtp` drafter directory (issue #1326), so it
//!   exposes free functions instead of implementing [`crate::SurgeryOp`].
//! - Construction goes through a `from_spec` constructor that consumes
//!   the already-validated `OpSpec::*` variant from the YAML parser
//!   ([`crate::config`]). The factory in `crate::config` is the only
//!   public path that wraps these in `Arc<dyn SurgeryOp>`.
//!
//! See `docs_internal/architecture/structural-finetuning-overview-20260419.md`
//! §3.2 for the operation matrix.

pub mod add;
pub mod interpolate;
pub mod prune;
pub mod replace;
pub mod scale;
pub mod split_mtp;

#[cfg(test)]
mod add_apply_tests;
#[cfg(test)]
mod add_test_helpers;
#[cfg(test)]
mod add_tests;

pub use add::AddOp;
pub use interpolate::InterpolateOp;
pub use prune::{PruneOp, PruneSelector};
pub use replace::ReplaceOp;
pub use scale::ScaleOp;
pub use split_mtp::{SplitMtpOptions, SplitMtpReport, SplitMtpResult, split_mtp, split_mtp_dir};
