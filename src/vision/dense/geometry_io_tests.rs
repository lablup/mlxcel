// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use serde::Deserialize;
#[derive(Deserialize)]
struct GaussianInput {
    positions: Vec<[f32; 3]>,
    sh_dc: Vec<[f32; 3]>,
    scales: Vec<[f32; 3]>,
    rotations: Vec<[f32; 4]>,
    opacities: Vec<[f32; 1]>,
}
#[derive(Deserialize)]
struct MeshInput {
    vertices: Vec<[f32; 3]>,
    faces: Vec<[u32; 3]>,
    vertex_colors: Vec<[f32; 6]>,
}
#[derive(Deserialize)]
struct Fixture {
    gaussian_input: GaussianInput,
    mesh_input: MeshInput,
    ply_header: String,
    ply_rows: Vec<[f32; 17]>,
    obj_text: String,
}
fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/sam3d_objects/writers.json"
    ))
    .expect("pinned writer fixture")
}
#[test]
fn geometry_io_obj_matches_upstream_colors_and_one_based_faces() -> Result<()> {
    let f = fixture();
    let input = f.mesh_input;
    let mesh = Mesh {
        vertices: input.vertices,
        faces: input.faces,
        vertex_colors: Some(input.vertex_colors),
    };
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("mesh.obj");
    write_obj(&path, &mesh)?;
    assert_eq!(std::fs::read_to_string(path)?, f.obj_text);
    Ok(())
}
#[test]
fn geometry_io_ply_matches_upstream_three_point_fixture() -> Result<()> {
    let f = fixture();
    let input = f.gaussian_input;
    let gaussian = Gaussians {
        positions: input.positions,
        sh_dc: input.sh_dc,
        scales: input.scales,
        rotations: input.rotations,
        opacities: input.opacities.into_iter().map(|p| p[0]).collect(),
    };
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("gaussian.ply");
    write_gaussian_ply(&path, &gaussian)?;
    let bytes = std::fs::read(path)?;
    assert_eq!(&bytes[..f.ply_header.len()], f.ply_header.as_bytes());
    let data = &bytes[f.ply_header.len()..];
    assert_eq!(data.len(), 3 * 17 * 4);
    for (actual, &expected) in data.chunks_exact(4).zip(f.ply_rows.iter().flatten()) {
        let actual = f32::from_le_bytes(actual.try_into()?);
        assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
    }
    Ok(())
}
#[test]
fn geometry_io_invalid_input_does_not_truncate_destination() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("existing");
    std::fs::write(&path, b"preserve")?;
    let mesh = Mesh {
        vertices: vec![[0.0; 3]],
        faces: vec![[0, 1, 0]],
        vertex_colors: Some(vec![[1.0; 6]]),
    };
    assert!(write_obj(&path, &mesh).is_err());
    let mut gaussian = Gaussians {
        positions: vec![[0.0; 3]],
        sh_dc: vec![[0.0; 3]],
        scales: vec![[1.0; 3]],
        rotations: vec![[1.0, 0.0, 0.0, 0.0]],
        opacities: vec![0.5],
    };
    gaussian.scales[0][1] = 0.0;
    assert!(write_gaussian_ply(&path, &gaussian).is_err());
    gaussian.scales[0][1] = f32::NAN;
    assert!(write_gaussian_ply(&path, &gaussian).is_err());
    gaussian.scales[0][1] = 1.0;
    gaussian.opacities[0] = f32::INFINITY;
    assert!(write_gaussian_ply(&path, &gaussian).is_err());
    assert_eq!(std::fs::read(path)?, b"preserve");
    Ok(())
}

#[test]
fn geometry_io_obj_matches_python_eight_significant_digits() -> Result<()> {
    // Pure-host Python oracle: struct.unpack('<f', struct.pack('<f', v))[0]
    // followed by '%.8g' % value, exactly the writer's f32-to-f64 promotion.
    let expected = [
        (0x3f9e0652, "1.2345679"),
        (0x38d1b717, "9.9999997e-05"),
        (0x38d1b716, "9.999999e-05"),
        (0x3727c5ac, "9.9999997e-06"),
        (0x4cbebc1f, "99999992"),
        (0x4cbebc20, "1e+08"),
        (0x4ceb7998, "1.234567e+08"),
        (0x80000000, "-0"),
        (0x007ffffd, "1.1754939e-38"),
        (0x7f7fffff, "3.4028235e+38"),
        (0x00000001, "1.4012985e-45"),
        (0x47f12010, "123456.12"),
        (0xc7f12010, "-123456.12"),
        (0x3f800000, "1"),
        (0x4b189680, "10000000"),
        (0xb8d1b717, "-9.9999997e-05"),
        (0x3a83126f, "0.001"),
    ];
    for (bits, expected) in expected {
        assert_eq!(obj_float(f32::from_bits(bits))?, expected);
    }
    let mesh = Mesh {
        vertices: vec![[f32::from_bits(0x3f9e0652), 1e-5, -0.0]],
        vertex_colors: Some(vec![[123456.125, 1e8, -1e-4, 0.0, 0.0, 0.0]]),
        faces: Vec::new(),
    };
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("precision.obj");
    write_obj(&path, &mesh)?;
    assert_eq!(
        std::fs::read_to_string(path)?,
        "v 1.2345679 9.9999997e-06 -0 123456.12 1e+08 -9.9999997e-05\n"
    );
    Ok(())
}

#[test]
fn geometry_io_sam3d_body_colorless_six_decimals() -> Result<()> {
    let mesh = Mesh {
        vertices: vec![[1.2345678, -0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        faces: vec![[0, 1, 2]],
        vertex_colors: None,
    };
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("body.obj");
    write_obj(&path, &mesh)?;
    assert_eq!(
        std::fs::read_to_string(path)?,
        "v 1.234568 -0.000000 2.000000\nv 0.000000 1.000000 0.000000\nv 0.000000 0.000000 1.000000\nf 1 2 3\n"
    );
    Ok(())
}
