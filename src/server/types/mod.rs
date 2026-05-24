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

//! OpenAI-compatible API types

pub mod anthropic_request;
pub mod anthropic_response;
pub mod anthropic_stream;
pub mod request;
pub mod response;
pub mod responses_request;
pub mod responses_response;
pub mod responses_stream;
pub mod stream;

// Anthropic types are accessed through their module paths (e.g.
// `types::anthropic_request::AnthropicRequest`) to avoid colliding with the
// OpenAI type glob below (both define `Tool`, `Role`, `MessageContent`, etc.).
pub use request::*;
pub use response::*;
pub use responses_request::*;
pub use responses_response::*;
pub use responses_stream::*;
pub use stream::*;
