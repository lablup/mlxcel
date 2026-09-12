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

//! Standalone static WebUI route harness for relocated-binary tests.
//!
//! This is not the production `mlxcel-server --webui` startup path. It exists
//! so #1836 can prove that the embedded asset router works after the built
//! artifact is moved away from the repository, `webui/`, and `node_modules/`.

use axum::{Router, routing::get};
use std::{env, net::SocketAddr};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let requested_addr = env::var("MLXCEL_WEBUI_HARNESS_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:0".to_string())
        .parse::<SocketAddr>()?;
    let app = Router::new()
        .route("/", get(|| async { "ok" }))
        .route("/v1/models", get(|| async { "[]" }))
        .merge(mlxcel::server::webui::router());
    let listener = TcpListener::bind(requested_addr).await?;
    let local_addr = listener.local_addr()?;
    println!("MLXCEL_WEBUI_STATIC_HARNESS_ADDR={local_addr}");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}
