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

//! Classification of public reads, administrative mutations and event streams.

use super::{WebUiSecurityPolicy, api_relative_path, path_matches_prefix, query_has_parameter};
use axum::http::{Method, Uri};

#[derive(Debug, Clone, Copy)]
pub(super) struct Decision {
    pub(super) kind: RequestKind,
    pub(super) control: bool,
    pub(super) sse: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RequestKind {
    Public,
    Private,
    PrivatePreflight,
}

pub(super) fn classify_request(
    policy: &WebUiSecurityPolicy,
    method: &Method,
    uri: &Uri,
) -> Decision {
    let path = uri.path();
    let relative = api_relative_path(policy, path);
    let public_static = matches!(*method, Method::GET | Method::HEAD)
        && path_matches_prefix(path, &policy.public_webui_prefix);
    let public_health = matches!(*method, Method::GET | Method::HEAD)
        && matches!(relative, Some("/" | "/health" | "/v1/health"));
    let preflight = *method == Method::OPTIONS;
    let legacy_reload = *method == Method::GET
        && matches!(relative, Some("/models"))
        && query_has_parameter(uri.query(), "reload");
    let sse = matches!(*method, Method::GET | Method::HEAD)
        && matches!(relative, Some("/models/sse" | "/ui-api/v1/events"));
    let mutation = !matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS);
    // Default every versioned UI mutation to administrative limits. Legacy
    // inference/control/cancel requests intentionally keep data-plane limits.
    let control = !sse
        && (legacy_reload
            || mutation
                && relative.is_some_and(|path| {
                    path_matches_prefix(path, "/ui-api/v1")
                        || path_matches_prefix(path, "/models")
                        || path_matches_prefix(path, "/slots")
                        || matches!(
                            path,
                            "/props"
                                | "/settings"
                                | "/v1/settings"
                                | "/lora-adapters"
                                | "/v1/cache/reset"
                        )
                }));
    Decision {
        kind: if public_static || public_health {
            RequestKind::Public
        } else if preflight {
            RequestKind::PrivatePreflight
        } else {
            RequestKind::Private
        },
        control,
        sse,
    }
}
