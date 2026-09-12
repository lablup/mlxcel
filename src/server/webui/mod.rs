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

//! Static WebUI bundle routes.
//!
//! Issue #1836 intentionally exposes a reusable static router without wiring it
//! into server startup yet. The startup owner mounts this router under the
//! server's validated API prefix when `--webui` becomes active.

pub(crate) mod api;
pub(crate) mod assets;
pub(crate) mod catalog;
pub(crate) mod security;

pub use assets::{WEBUI_PREFIX, manifest_json, router};
