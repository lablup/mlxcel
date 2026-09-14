// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
use super::api::{WEBUI_JSON_BODY_BYTES, WebUiServerMode, media_limits};
use crate::server::media::ImageInputLimits;

#[test]
fn resolved_media_limits_preserve_configured_asymmetric_bounds() {
    let actual = serde_json::to_value(media_limits(
        ImageInputLimits {
            max_payload_bytes: 12345,
            max_images_per_request: 2,
            max_width: 320,
            max_height: 640,
            max_decode_alloc_bytes: 987654,
        },
        WebUiServerMode::RouterPool,
    ))
    .unwrap();
    assert_eq!(
        actual,
        serde_json::json!({
            "max_images": 2, "max_image_bytes": 12345,
            "max_width": 320, "max_height": 640,
            "max_decoded_bytes": 987654, "max_body_bytes": WEBUI_JSON_BODY_BYTES,
        })
    );
}

#[test]
fn media_body_budget_distinguishes_router_and_single_inference() {
    let limits = ImageInputLimits::default();
    assert_eq!(
        media_limits(limits, WebUiServerMode::RouterPool).max_body_bytes,
        crate::server::router_server::DISPATCH_BODY_CAP as u64
    );
    assert_eq!(
        media_limits(limits, WebUiServerMode::SingleModel).max_body_bytes,
        crate::server::app::main_json_body_limit_bytes() as u64
    );
    assert!(
        media_limits(limits, WebUiServerMode::SingleModel).max_body_bytes > WEBUI_JSON_BODY_BYTES
    );
}
