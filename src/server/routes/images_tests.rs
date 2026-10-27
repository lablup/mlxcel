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

//! `/v1/images/*` route tests over the stub image model, driven through the
//! real router so the request flow (JSON body, validation, worker dispatch,
//! PNG response) is exercised end to end.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use base64::Engine as _;
use image::GenericImageView;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{NO_IMAGE_MODEL_MESSAGE, image_error_response};
use crate::imagegen::testing::StubModel;
use crate::server::image_model::{ImageError, ImageModelProvider};
use crate::server::image_worker::ImageWorkerProvider;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const STUB_MODEL_ID: &str = "stub-image";

fn provider(channels: u8, edit: bool) -> Arc<dyn ImageModelProvider> {
    Arc::new(
        ImageWorkerProvider::from_loader(
            STUB_MODEL_ID.to_string(),
            4,
            Duration::from_secs(30),
            move || {
                Ok(Box::new(StubModel {
                    family: "flux2",
                    channels,
                    edit,
                }))
            },
        )
        .expect("stub image worker spawns"),
    )
}

fn app_with(provider: Option<Arc<dyn ImageModelProvider>>) -> axum::Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let model_provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = model_provider.batch_metrics().clone();
    let state = AppState::new(
        model_provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_image_model(provider);
    create_app(state)
}

fn rgb_app() -> axum::Router {
    app_with(Some(provider(3, true)))
}

async fn post(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("route responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body reads");
    let value: Value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("json body")
    };
    (status, value)
}

fn decode(b64: &Value) -> image::DynamicImage {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.as_str().expect("b64 string"))
        .expect("valid base64");
    image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).expect("valid png")
}

fn message(body: &Value) -> String {
    body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

/// A `data:` URL holding a `w`x`h` RGB PNG.
fn png_data_url(w: u32, h: u32) -> String {
    let mut bytes = Vec::new();
    image::DynamicImage::new_rgb8(w, h)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

#[tokio::test]
async fn without_an_image_model_both_routes_are_501_naming_the_flag() {
    for path in [
        "/v1/images/generations",
        "/images/generations",
        "/v1/images/edits",
        "/images/edits",
    ] {
        let (status, body) = post(
            app_with(None),
            path,
            json!({"prompt": "a cat", "image": png_data_url(32, 16)}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {body}");
        assert_eq!(message(&body), NO_IMAGE_MODEL_MESSAGE, "{path}");
        assert!(message(&body).contains("--image-model"), "{path}");
    }
}

#[tokio::test]
async fn generation_returns_base64_pngs_of_the_requested_size() {
    let (status, body) = post(
        rgb_app(),
        "/v1/images/generations",
        json!({"prompt": "a cat", "n": 2, "size": "64x32", "seed": 100}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["output_format"], "png");
    assert_eq!(body["size"], "64x32");
    assert!(body["created"].as_i64().unwrap() > 0);
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    for (i, item) in data.iter().enumerate() {
        assert_eq!(item["mime_type"], "image/png");
        assert_eq!(item["width"], 64);
        assert_eq!(item["height"], 32);
        assert_eq!(item["seed"], 100 + i as u64);
        assert_eq!(item["revised_prompt"], "revised: a cat");
        assert_eq!(decode(&item["b64_json"]).dimensions(), (64, 32));
    }
    assert_ne!(data[0]["b64_json"], data[1]["b64_json"], "seed + i differs");
}

#[tokio::test]
async fn the_unprefixed_alias_serves_the_same_route() {
    let (status, body) = post(
        rgb_app(),
        "/images/generations",
        json!({"prompt": "a cat", "size": "32x32"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn generation_defaults_to_512x512_and_a_random_seed() {
    let (status, body) = post(
        rgb_app(),
        "/v1/images/generations",
        json!({"prompt": "a cat"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["size"], "512x512");
    assert!(body["data"][0]["seed"].is_u64());
}

#[tokio::test]
async fn an_rgba_model_keeps_its_alpha_channel_on_the_wire() {
    let app = app_with(Some(provider(4, false)));
    let (status, body) = post(
        app,
        "/v1/images/generations",
        json!({"prompt": "a logo", "size": "32x16"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let png = decode(&body["data"][0]["b64_json"]);
    assert_eq!(png.color(), image::ColorType::Rgba8);
    assert!(png.to_rgba8().pixels().all(|p| p.0[3] == 128));
}

#[tokio::test]
async fn width_and_height_override_size_and_must_come_together() {
    let (status, body) = post(
        rgb_app(),
        "/v1/images/generations",
        json!({"prompt": "x", "size": "512x512", "width": 48, "height": 32}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["size"], "48x32");

    for partial in [json!({"width": 48}), json!({"height": 32})] {
        let mut request = json!({"prompt": "x"});
        request
            .as_object_mut()
            .unwrap()
            .extend(partial.as_object().unwrap().clone());
        let (status, body) = post(rgb_app(), "/v1/images/generations", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(message(&body).contains("together"), "{body}");
    }
}

#[tokio::test]
async fn the_server_never_writes_files_for_a_client() {
    for (field, value) in [
        ("response_format", json!("path")),
        ("response_format", json!("url")),
        ("output_path", json!("/tmp/x.png")),
        ("output_dir", json!("/tmp")),
    ] {
        let mut request = json!({"prompt": "x", "size": "32x32"});
        request[field] = value;
        let (status, body) = post(rgb_app(), "/v1/images/generations", request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{field}: {body}");
        assert!(
            message(&body).contains("does not write files"),
            "{field}: {body}"
        );
    }
}

#[tokio::test]
async fn b64_json_and_png_are_accepted_and_other_formats_are_not() {
    let (status, _) = post(
        rgb_app(),
        "/v1/images/generations",
        json!({"prompt": "x", "size": "32x32", "response_format": "b64_json", "output_format": "png"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = post(
        rgb_app(),
        "/v1/images/generations",
        json!({"prompt": "x", "size": "32x32", "response_format": "binary"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let (status, body) = post(
        rgb_app(),
        "/v1/images/generations",
        json!({"prompt": "x", "size": "32x32", "output_format": "jpeg"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn invalid_parameters_are_400_with_a_reason() {
    let cases = [
        (json!({"prompt": "x", "n": 0}), "between 1 and 10"),
        (json!({"prompt": "x", "n": 11}), "between 1 and 10"),
        (json!({"prompt": "x", "size": "30x32"}), "multiples of 16"),
        (json!({"prompt": "x", "size": "banana"}), "WIDTHxHEIGHT"),
        (json!({"prompt": "x", "size": "32x32", "steps": 0}), "steps"),
        (
            json!({"prompt": "x", "size": "32x32", "guidance": -1.0}),
            "guidance",
        ),
        (json!({"prompt": "   ", "size": "32x32"}), "prompt"),
        (
            json!({"prompt": "x", "model": "other"}),
            "not the served image model",
        ),
    ];
    for (request, needle) in cases {
        let (status, body) = post(rgb_app(), "/v1/images/generations", request.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{request}: {body}");
        assert!(message(&body).contains(needle), "{request}: {body}");
    }
}

#[tokio::test]
async fn the_served_model_id_is_accepted() {
    let (status, _) = post(
        rgb_app(),
        "/v1/images/generations",
        json!({"prompt": "x", "size": "32x32", "model": STUB_MODEL_ID}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_missing_prompt_is_a_400() {
    let (status, body) = post(
        rgb_app(),
        "/v1/images/generations",
        json!({"size": "32x32"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(message(&body).contains("invalid request body"), "{body}");
}

#[tokio::test]
async fn edit_takes_the_first_reference_size_by_default() {
    let (status, body) = post(
        rgb_app(),
        "/v1/images/edits",
        json!({"prompt": "make it blue", "image": png_data_url(48, 32)}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["size"], "48x32");
    assert_eq!(decode(&body["data"][0]["b64_json"]).dimensions(), (48, 32));
}

/// A stub family whose edit default doubles each side of the reference.
struct DoublingEditSize(StubModel);

impl crate::imagegen::ImageGenerationModel for DoublingEditSize {
    fn family(&self) -> &'static str {
        self.0.family()
    }
    fn supports(&self, task: crate::imagegen::ImageTask) -> bool {
        self.0.supports(task)
    }
    fn defaults(&self, task: crate::imagegen::ImageTask) -> crate::imagegen::ImageSamplingDefaults {
        self.0.defaults(task)
    }
    fn default_edit_size(&self) -> fn(u32, u32) -> (u32, u32) {
        |w, h| (2 * w, 2 * h)
    }
    fn generate(
        &self,
        req: &crate::imagegen::ImageRequest,
    ) -> anyhow::Result<Vec<crate::imagegen::GeneratedImage>> {
        self.0.generate(req)
    }
}

#[tokio::test]
async fn edit_without_a_size_uses_the_family_edit_size() {
    let provider: Arc<dyn ImageModelProvider> = Arc::new(
        ImageWorkerProvider::from_loader(
            STUB_MODEL_ID.to_string(),
            4,
            Duration::from_secs(30),
            || {
                Ok(Box::new(DoublingEditSize(StubModel {
                    family: "flux2",
                    channels: 3,
                    edit: true,
                })))
            },
        )
        .expect("stub image worker spawns"),
    );
    let (status, body) = post(
        app_with(Some(provider)),
        "/v1/images/edits",
        json!({"prompt": "make it blue", "image": png_data_url(48, 32)}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["size"], "96x64");
    assert_eq!(decode(&body["data"][0]["b64_json"]).dimensions(), (96, 64));
}

#[tokio::test]
async fn edit_accepts_a_list_of_references_and_an_explicit_size() {
    let (status, body) = post(
        rgb_app(),
        "/v1/images/edits",
        json!({
            "prompt": "merge",
            "image": [png_data_url(32, 32), png_data_url(16, 16)],
            "size": "64x64",
            "n": 2,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    assert_eq!(body["size"], "64x64");
}

#[tokio::test]
async fn edit_rejects_raw_filesystem_paths() {
    for path in ["/etc/passwd", "relative/cat.png", "~/cat.png"] {
        let (status, body) = post(
            rgb_app(),
            "/v1/images/edits",
            json!({"prompt": "x", "image": path}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
        assert!(message(&body).contains("raw file paths"), "{path}: {body}");
    }
}

#[tokio::test]
async fn edit_rejects_bad_image_payloads() {
    let (status, body) = post(
        rgb_app(),
        "/v1/images/edits",
        json!({"prompt": "x", "image": "data:image/png;base64,@@@not-base64"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let (status, body) = post(
        rgb_app(),
        "/v1/images/edits",
        json!({"prompt": "x", "image": []}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(message(&body).contains("at least one image"), "{body}");

    let (status, body) = post(rgb_app(), "/v1/images/edits", json!({"prompt": "x"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn edit_on_a_generate_only_family_is_400() {
    let app = app_with(Some(provider(3, false)));
    let (status, body) = post(
        app,
        "/v1/images/edits",
        json!({"prompt": "x", "image": png_data_url(32, 32)}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        message(&body).contains("does not support task `edit`"),
        "{body}"
    );
}

#[test]
fn provider_errors_map_to_the_shared_http_statuses() {
    assert_eq!(
        image_error_response(ImageError::QueueFull).status,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        image_error_response(ImageError::Timeout).status,
        StatusCode::GATEWAY_TIMEOUT
    );
    assert_eq!(
        image_error_response(ImageError::InvalidInput("bad".into())).status,
        StatusCode::BAD_REQUEST
    );
    let internal = image_error_response(ImageError::Internal("boom".into()));
    assert_eq!(internal.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(internal.error.message.contains("boom"));
}
