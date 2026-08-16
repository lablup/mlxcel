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

//! Shared helpers for `#[cfg(test)]` code paths in this crate.
//!
//! Items here are NOT part of the public API. The module is gated on
//! `cfg(test)` at the crate root so it compiles only inside the test binary;
//! production builds never see it.

pub(crate) mod env_lock;
pub(crate) mod pinned_checkpoint;
pub(crate) mod video_gate;
