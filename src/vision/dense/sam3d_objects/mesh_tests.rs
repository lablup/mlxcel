// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use serde::Deserialize;
#[derive(Deserialize)]
struct Expected {
    vertices: Vec<[f32; 3]>,
    faces: Vec<[u32; 3]>,
    vertex_colors: Vec<[f32; 6]>,
}
#[derive(Deserialize)]
struct Fixture {
    resolution: i32,
    coords: Vec<[i32; 4]>,
    features: Vec<Vec<f32>>,
    mesh: Expected,
}
#[test]
fn sam3d_objects_flexicubes_sphere_matches_pinned_upstream() -> Result<()> {
    let f: Fixture = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/sam3d_objects/sphere.json"
    ))?;
    let features: Vec<_> = f.features.into_iter().flatten().collect();
    let mesh = extract_host(&features, &f.coords, f.resolution)?;
    assert_eq!(mesh.vertices.len(), 128);
    assert_eq!(mesh.faces.len(), 252);
    assert_eq!(mesh.faces, f.mesh.faces, "stable topology and winding");
    for (&a, &b) in mesh
        .vertices
        .iter()
        .flatten()
        .zip(f.mesh.vertices.iter().flatten())
    {
        assert!((a - b).abs() < 2e-6, "vertex {a} != {b}");
    }
    for (&a, &b) in mesh
        .vertex_colors
        .iter()
        .flatten()
        .zip(f.mesh.vertex_colors.iter().flatten())
    {
        assert!((a - b).abs() < 2e-6, "color {a} != {b}");
    }
    // Every closed-sphere edge must occur twice, independent of the oracle.
    let mut edges = std::collections::BTreeMap::new();
    for &[a, b, c] in &mesh.faces {
        for (a, b) in [(a, b), (b, c), (c, a)] {
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    assert!(edges.values().all(|&n| n == 2));
    Ok(())
}
#[test]
fn sam3d_objects_flexicubes_validates_and_handles_empty_surface() -> Result<()> {
    assert!(extract_host(&[], &[], 0).is_err());
    assert_eq!(extract_host(&[], &[], 8)?, Mesh::default());
    let coords = [[0, 0, 0, 0]];
    let mut f = vec![0.0; 101];
    f[..8].fill(1.0);
    assert_eq!(extract_host(&f, &coords, 8)?, Mesh::default());
    f[55] = f32::NAN;
    assert!(extract_host(&f, &coords, 8).is_err());
    Ok(())
}
