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

//! Thin HTTP route adapters.
//!
//! These files should stay focused on request/response translation. Shared
//! policy belongs in `server/request_options.rs`, `server/chat_request.rs`,
//! `server/media.rs`, `server/streaming.rs`, and `server/model_worker.rs`.

pub mod anthropic;
pub mod audio;
pub mod cache;
pub mod chat;
pub mod completions;
pub mod detokenize;
pub mod health;
pub mod metrics;
pub mod models;
pub mod native_completion;
pub mod props;
pub mod responses;
pub mod slots;
pub mod tokenize;

pub use anthropic::{anthropic_count_tokens, anthropic_messages};
pub use audio::{audio_speech, audio_transcriptions, audio_translations};
pub use cache::{cache_reset, cache_stats};
pub use chat::chat_completions;
pub use completions::completions;
pub use detokenize::detokenize;
pub use health::health_check;
pub use metrics::metrics;
pub use models::list_models;
pub use native_completion::native_completion;
pub use props::props;
pub use responses::{cancel_response, create_response, delete_response, retrieve_response};
pub use slots::slots;
pub use tokenize::tokenize;
