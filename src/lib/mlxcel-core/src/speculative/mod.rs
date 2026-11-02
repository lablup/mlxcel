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

//! Speculative decoding: a cheap drafter proposes tokens and the target verifies
//! them in one forward.
//!
//! The decode loops live with their drafters:
//!
//! - [`mtp`]: Multi-Token Prediction round loop for the Gemma 4 assistant,
//!   Qwen 3.5 MTP, Inkling and GLM-4.7-Flash drafters. See
//!   [`mtp::MtpGenerator`].
//! - [`crate::drafter::dflash`]: the DFlash round loop.
//! - [`prompt_lookup`]: drafter-free speculation that proposes the tokens
//!   following an earlier occurrence of the sequence's tail, driven by
//!   [`crate::engine::Engine::verify_round`]. See
//!   [`prompt_lookup_drafter::PromptLookupDrafter`].
//! - [`full_model_drafter`]: an ordinary full model as a token-only drafter
//!   (#2757), driven by the same [`crate::engine::Engine::verify_round`] as
//!   prompt lookup on every surface that takes `--draft-model`. See
//!   [`full_model_drafter::FullModelDrafter`].
//!
//! `docs/speculative-acceptance.md` states which acceptance rule each path runs.

pub mod full_model_drafter;
pub mod mtp;
pub mod prompt_lookup;
pub mod prompt_lookup_drafter;
