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

//! Route and native request-field conformance for the llama-server b10621
//! compatibility manifest (`compat/llama-server/b10621/`, issue #1443).
//!
//! The manifest's route entries claim, via `mlxcel.route`, which b10621
//! method/path pairs the mlxcel server actually mounts; its native
//! request-field entries claim, via `mlxcel.field`, which b10621
//! `/completion` fields `NativeCompletionRequest` actually accepts. These
//! tests hold both claim sets against the code in BOTH directions:
//!
//! - a claim naming the b10621 identity itself (`mlxcel.route` equal to the
//!   entry id, `mlxcel.field` equal to the b10621 field) must resolve on the
//!   real router / exist on the struct;
//! - a claim naming a DIFFERENT identity is an `aliased` mapping: the alias
//!   must resolve / exist and the b10621 identity must NOT, which is what
//!   distinguishes `aliased` from `supported` mechanically;
//! - an UNCLAIMED route must NOT resolve and an unclaimed field must NOT
//!   exist, so implementing part of the surface without flipping the
//!   manifest entry fails CI and produces the reviewable diff the manifest
//!   exists for.
//!
//! It also carries the Rust-side copy of one structural rule: an entry whose
//! `divergence` list is non-empty may not be `supported`. That rule's full
//! enforcement (shape, wording, guidance) is in the Python validator, but
//! this file loads every shard anyway, so restating the rule here costs
//! nothing and keeps `cargo test` from passing a manifest `make verify`
//! would reject.
//!
//! The clap option surface half of the gate lives in
//! `tests/llama_compat_manifest.rs`; the structural half in
//! `scripts/ci/check_llama_compat_manifest.py`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

use super::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

/// Manifest document schema, independent of the pinned llama.cpp release.
/// Kept in lockstep with `scripts/ci/check_llama_compat_manifest.py`,
/// `scripts/compat/extract_b10621_manifest.py`, and
/// `tests/llama_compat_manifest.rs`; bump all four together. Issue #1443
/// follow-ups: 2 when pin.json's `shards` field changed from a bare name
/// list to a mapping of shard name to its owning-issue set, 3 when every
/// entry gained the structured `divergence` list.
const MANIFEST_SCHEMA_VERSION: i64 = 3;

fn manifest_entries() -> Vec<serde_json::Value> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("compat/llama-server/b10621");
    let mut entries = Vec::new();
    for path in std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {dir:?}: {e}"))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter(|p| p.file_name().is_some_and(|n| n != "pin.json"))
    {
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("shard readable"))
                .unwrap_or_else(|e| panic!("{path:?} is not valid JSON: {e}"));
        assert_eq!(
            doc["schema_version"], MANIFEST_SCHEMA_VERSION,
            "{path:?}: unsupported manifest schema_version"
        );
        entries.extend(
            doc["entries"]
                .as_array()
                .expect("entries array")
                .iter()
                .cloned(),
        );
    }
    assert!(!entries.is_empty(), "no manifest entries under {dir:?}");
    entries
}

/// Router with every conditional endpoint enabled, so the probe answers
/// "is this route mounted at all" rather than "is it enabled by default".
/// Default-enablement parity is a per-flag concern tracked on the option
/// entries (`--slots`, `--props`, `--metrics`).
fn probe_app() -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig {
            enable_slots_endpoint: true,
            enable_props_endpoint: true,
            enable_metrics_endpoint: true,
            ..Default::default()
        },
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    );
    create_app(state)
}

/// Router-mode probe app (issue #1438): the b10621 model-management routes
/// (`POST/DELETE /models`, `/models/load`, `/models/unload`, `/models/sse`)
/// are registered only by the router server, exactly as upstream registers
/// them only when `is_router_server`; claims for them resolve here.
fn router_probe_app() -> axum::Router {
    let dir =
        std::env::temp_dir().join(format!("mlxcel-compat-router-probe-{}", std::process::id()));
    let model_dir = dir.join("probe-model");
    std::fs::create_dir_all(&model_dir).expect("probe models dir");
    std::fs::write(model_dir.join("config.json"), "{}").expect("probe config");
    let pool = std::sync::Arc::new(
        crate::server::router_models::RouterPool::new(dir, ServerConfig::default(), 4, true)
            .expect("probe pool"),
    );
    crate::server::router_server::create_router_app(
        crate::server::router_server::RouterServerState {
            pool,
            config: Arc::new(ServerConfig::default()),
        },
    )
}

async fn probe_on(app: axum::Router, method: &str, path: &str) -> StatusCode {
    let method: Method = method.parse().expect("HTTP method");
    let needs_body = method == Method::POST;
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(if needs_body {
            // Name the probe pool's one model so a mounted handler answers
            // from its own logic (400/500) instead of the name-validation
            // 404 that would read as "route absent".
            Body::from(r#"{"model":"probe-model"}"#)
        } else {
            Body::empty()
        })
        .expect("probe request builds");
    app.oneshot(request)
        .await
        .expect("router responds")
        .status()
}

/// Whether either serving mode resolves the method/path pair: the
/// single-model app first, the router-mode app second, mirroring b10621's
/// two registration sets.
async fn probe_resolves(method: &str, path: &str) -> bool {
    let single = probe(method, path).await;
    if single != StatusCode::NOT_FOUND && single != StatusCode::METHOD_NOT_ALLOWED {
        return true;
    }
    // The router fallback dispatches unmatched paths, answering 400 for a
    // missing model rather than 404; only the routes the router itself
    // registers are meaningful here, so restrict the second probe to the
    // model-management set.
    const ROUTER_ONLY: [&str; 5] = [
        "/models",
        "/models/load",
        "/models/unload",
        "/models/sse",
        "/v1/models",
    ];
    if !ROUTER_ONLY.contains(&path) {
        return false;
    }
    let router = probe_on(router_probe_app(), method, path).await;
    router != StatusCode::NOT_FOUND && router != StatusCode::METHOD_NOT_ALLOWED
}

async fn probe(method: &str, path: &str) -> StatusCode {
    let method: Method = method.parse().expect("HTTP method");
    let needs_body = method == Method::POST;
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(if needs_body {
            Body::from("{}")
        } else {
            Body::empty()
        })
        .expect("probe request builds");
    probe_app()
        .oneshot(request)
        .await
        .expect("router responds")
        .status()
}

/// A route entry claiming the b10621 method/path itself must resolve on the
/// router (any status except 404/405 counts: a 400/401/422/501 from the
/// mounted handler still proves the method/path pair is served). A claim
/// naming a different method/path is an `aliased` mapping, so the alias must
/// resolve and the b10621 pair must not. A route entry WITHOUT a claim must
/// not resolve, so adding the route later forces the manifest entry to flip
/// in the same change.
#[tokio::test]
async fn manifest_route_claims_match_the_mounted_router() {
    for entry in manifest_entries() {
        if entry["kind"] != "route" {
            continue;
        }
        let id = entry["id"].as_str().expect("id");
        let path = entry["path"].as_str().expect("path");
        if path.contains("${") {
            // Synthetic entries for env-configured GCP routes and the Web
            // UI static mount have no fixed probeable path, so their claims
            // cannot be verified against the router here. A claim, when one
            // is recorded (#1456 mounts the GCP pair from `AIP_*`), must
            // name the synthetic id itself; behavior is covered by
            // `gcp_compat_tests.rs` against a router with the adapter
            // enabled.
            if !entry["mlxcel"].is_null() {
                assert_eq!(
                    entry["mlxcel"]["route"], entry["id"],
                    "{id}: a synthetic route claim must record the entry id"
                );
            }
            continue;
        }
        let method = entry["method"].as_str().expect("method");
        let status = probe(method, path).await;
        let resolved = probe_resolves(method, path).await;
        if entry["mlxcel"].is_null() {
            assert!(
                !resolved,
                "{id}: the manifest records this b10621 route as not served \
                 by mlxcel, but the router answered {status}. Flip the \
                 manifest entry (state, mlxcel.route, notes) in the same \
                 change that mounts the route."
            );
        } else {
            let claimed = entry["mlxcel"]["route"]
                .as_str()
                .unwrap_or_else(|| panic!("{id}: a route claim must record mlxcel.route"));
            if claimed == id {
                assert!(
                    resolved,
                    "{id}: the manifest claims this route is mounted, but the \
                     router answered {status}"
                );
            } else {
                // `aliased`: mlxcel serves the equivalent elsewhere. Both
                // halves are asserted so the state cannot drift into a
                // mislabelled `supported`.
                assert!(
                    !resolved,
                    "{id}: the manifest records an alias to {claimed:?}, but \
                     mlxcel also serves the b10621 route itself ({status}). \
                     Use state `supported` with mlxcel.route = the entry id."
                );
                let (alias_method, alias_path) = claimed.split_once(' ').unwrap_or_else(|| {
                    panic!("{id}: mlxcel.route must be \"METHOD /path\", got {claimed:?}")
                });
                let alias_status = probe(alias_method, alias_path).await;
                assert!(
                    alias_status != StatusCode::NOT_FOUND
                        && alias_status != StatusCode::METHOD_NOT_ALLOWED,
                    "{id}: the manifest maps this b10621 route onto \
                     {claimed:?}, but the router answered {alias_status} there"
                );
            }
        }
    }
}

/// A native request-field entry claiming the b10621 field name itself must
/// exist on `NativeCompletionRequest` (as a field or serde alias). A claim
/// naming a different field is an `aliased` mapping, so the alias must exist
/// and the b10621 name must not. An unclaimed entry must not exist at all,
/// so accepting a new b10621 field forces the manifest flip.
///
/// The struct is read as source text rather than probed by deserializing:
/// `NativeCompletionRequest` does not set `deny_unknown_fields`, so a
/// round-trip cannot distinguish "accepted" from "silently ignored".
///
/// An entry with a non-null `parent` is a SUBFIELD of a nested b10621 block
/// (`stream_options.include_usage` is the only one at b10621), so it is
/// modeled by a nested struct rather than by a key on
/// `NativeCompletionRequest` itself. Those claims are checked against the
/// whole of `request.rs`, and additionally require the parent block to be
/// declared on the request struct, so a subfield cannot be claimed without the
/// block that carries it. That is deliberately a weaker check than the
/// root-field one, because the module also carries the OpenAI routes'
/// `StreamOptions`: it is a ratchet against claiming a subfield nothing
/// models, not a proof that the native path reads it. The behavioral half
/// (the type diagnostic, and the value being inert on `/completion` exactly
/// as it is upstream) is asserted in `routes/native_route_tests.rs`.
#[test]
fn manifest_native_field_claims_match_native_completion_request() {
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/types/request.rs"),
    )
    .expect("request.rs readable");
    let start = source
        .find("pub struct NativeCompletionRequest {")
        .expect("NativeCompletionRequest struct present");
    let end = source[start..]
        .find("\n}")
        .map(|i| start + i)
        .expect("struct block terminated");
    let block = &source[start..end];

    let declared_in = |scope: &str, field: &str| {
        scope.contains(&format!("pub {field}:")) || scope.contains(&format!("alias = \"{field}\""))
    };

    for entry in manifest_entries() {
        if entry["kind"] != "native_request_field" {
            continue;
        }
        let id = entry["id"].as_str().expect("id");
        let field = entry["field"].as_str().expect("field");
        // A subfield lives on the nested struct that models its parent block,
        // so its scope is the whole module; a root field's scope is the
        // request struct alone.
        let parent = entry["parent"].as_str();
        let scope = if parent.is_some() {
            source.as_str()
        } else {
            block
        };
        let declares = |name: &str| declared_in(scope, name);
        if let Some(parent) = parent {
            assert!(
                declared_in(block, parent),
                "{id}: a subfield claim needs its parent block {parent:?} \
                 declared on NativeCompletionRequest"
            );
        }
        if entry["mlxcel"].is_null() {
            assert!(
                !declares(field),
                "{id}: NativeCompletionRequest accepts {field:?} but the \
                 manifest records it as not accepted. Flip the manifest \
                 entry in the same change that adds the field."
            );
        } else {
            let claimed = entry["mlxcel"]["field"]
                .as_str()
                .unwrap_or_else(|| panic!("{id}: a field claim must record mlxcel.field"));
            assert!(
                declares(claimed),
                "{id}: the manifest claims NativeCompletionRequest accepts \
                 {claimed:?}, but the struct no longer declares it"
            );
            if claimed != field {
                // `aliased`: mlxcel spells the same concept differently.
                assert!(
                    !declares(field),
                    "{id}: the manifest records {claimed:?} as the mlxcel \
                     spelling, but NativeCompletionRequest also declares the \
                     b10621 name {field:?}. Use state `supported` with \
                     mlxcel.field = the b10621 field name."
                );
            }
        }
    }
}

/// No entry may claim `supported` while recording an externally observable
/// divergence from b10621. Epic #1431 defines `supported` as "the spelling,
/// value domain, default, precedence, and externally observable behavior
/// match", so a non-empty `divergence` contradicts the state outright; the
/// honest states are `aliased`, `not_applicable` or `deferred`, each with an
/// owning issue. `scripts/ci/check_llama_compat_manifest.py` is the primary
/// gate and additionally checks the field's shape; this keeps a `cargo test`
/// run from passing a manifest that gate would reject.
#[test]
fn supported_entries_record_no_divergence_from_b10621() {
    let mut checked = 0usize;
    for entry in manifest_entries() {
        let id = entry["id"].as_str().expect("id");
        let divergence = entry["divergence"]
            .as_array()
            .unwrap_or_else(|| panic!("{id}: every entry must carry a divergence list"));
        checked += 1;
        if entry["state"] != "supported" {
            continue;
        }
        assert!(
            divergence.is_empty(),
            "{id}: state `supported` with {} recorded divergence(s) {divergence:?}.              An entry that differs from b10621 in externally observable behavior              is `aliased`, `not_applicable`, or `deferred` with the owning issue              named.",
            divergence.len()
        );
    }
    assert_eq!(
        checked, 376,
        "the manifest must hold all 376 b10621 entries (249 options, 53 routes,          74 native request fields)"
    );
}
