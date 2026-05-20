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

//! Shared multimodal prompt and runtime helpers.
//!
//! This layer sits between model-specific vision implementations and the CLI /
//! server entry points. The goal is to keep prompt rewriting, image-token block
//! expansion, and prepared-embedding plumbing reusable across frontends.
//!
//! Modules:
//! - `batched_dispatch`: shared per-row batched dispatch helper used by every
//!   vision wrapper that needs `forward_batched_with_context_and_ids` to route
//!   each row through `forward_with_sequence_id` (PR #558 / issue #542).
//! - `gemma4_vl`: Gemma 4 mixed-length batching helpers (issue #542)
//! - `phi4mm_prompt`: Phi4MM `<|image_N|>` normalization and audio guard
//! - `phi4_siglip_prompt`: Phi4-SigLIP `<image>` placeholder handling
//! - `minicpmo_prompt`: MiniCPM-o image placeholder expansion and bounds
//! - `moondream3_prompt`: Moondream3 query/caption template shaping
//! - `qwen_vl`: Qwen-VL token insertion rules
//! - `phi3v_prompt`: Phi3V image-tag normalization
//! - `vlm_prompt`: generic image-token block expansion
//! - `vlm_runtime`: image preprocessing and embedding preparation shared by CLI/server
//! - `video`: generic VLM video frame extraction and FPS sampling (issue #553)

pub mod batched_dispatch;
pub mod gemma4_vl;
pub mod internvl_prompt;
pub mod minicpmo_prompt;
pub mod moondream3_prompt;
pub mod phi3v_prompt;
pub mod phi4_siglip_prompt;
pub mod phi4mm_prompt;
pub mod qwen_vl;
pub mod video;
pub mod vlm_prompt;
pub mod vlm_runtime;
pub mod youtu_vl_prompt;

#[cfg(test)]
#[path = "moondream3_prompt_tests.rs"]
mod moondream3_prompt_tests;
