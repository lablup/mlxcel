// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;
#[test]
fn sam3d_body_box_partial_padding_and_rejection() {
    let image = DynamicImage::new_rgb8(4, 4);
    let parse = |v: Value| InputGeometry::parse(&image, v.as_object().expect("object"));
    assert_eq!(
        parse(serde_json::json!({"bbox":[-1,0,3,4]}))
            .expect("partial")
            .bbox,
        [-1.0, 0.0, 3.0, 4.0]
    );
    assert!(parse(serde_json::json!({"bbox":[4,0,6,4]})).is_err());
    assert!(parse(serde_json::json!({"bbox":[0,0,0,4]})).is_err());
    assert!(parse(serde_json::json!({"cam_int":[1,0,0,0,1,0,0,0,0]})).is_err());
}
#[test]
fn sam3d_body_crop_matches_authentic_opencv_bytes_and_normalization() -> Result<()> {
    let tensors = safetensors::SafeTensors::deserialize(include_bytes!(
        "../../../../tests/fixtures/sam3d_body/crop-tiny.safetensors"
    ))?;
    for case in ["tiny_full", "tiny_partial"] {
        let input = tensors.tensor(&format!("{case}.input"))?;
        let shape = input.shape();
        let image = DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(shape[1] as u32, shape[0] as u32, input.data().to_vec())
                .context("RGB input")?,
        );
        let box_tensor = tensors.tensor(&format!("{case}.bbox"))?;
        let values = box_tensor
            .data()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        let options = serde_json::json!({"bbox":values});
        let geometry = InputGeometry::parse(&image, options.as_object().context("options")?)?;
        let expected = tensors.tensor(&format!("{case}.crop"))?;
        let config = Config {
            image_size: [expected.shape()[0], expected.shape()[1]],
            ..Config::default()
        };
        let actual = crop(&image, &geometry, &config)?;
        assert_eq!(actual, expected.data(), "{case} OpenCV crop bytes");
        let normalized = tensors.tensor(&format!("{case}.normalized"))?;
        for (i, (byte, expected)) in actual
            .iter()
            .zip(normalized.data().chunks_exact(4))
            .enumerate()
        {
            let value =
                (f32::from(*byte) / 255.0 - config.image_mean[i % 3]) / config.image_std[i % 3];
            assert_eq!(value.to_le_bytes(), expected, "{case} normalized pixel {i}");
        }
        let cliff = tensors.tensor(&format!("{case}.cliff"))?;
        for (actual, expected) in geometry.cliff().iter().zip(cliff.data().chunks_exact(4)) {
            assert_eq!(actual.to_le_bytes(), expected, "{case} CLIFF");
        }
    }
    Ok(())
}
