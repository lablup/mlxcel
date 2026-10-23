// Copyright (c) Meta Platforms, Inc. and affiliates.
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//! Inference FlexiCubes, including learned alpha/beta/gamma and six color channels.
use super::sparse::Grid;
use crate::vision::dense::geometry_io::Mesh;
use anyhow::{Result, ensure};
use mlxcel_core::MlxArray;
#[path = "mesh_check_table.rs"]
mod check;
#[path = "mesh_dmc_table.rs"]
mod dmc;
#[path = "mesh_memory.rs"]
mod memory;
#[path = "mesh_num_vd_table.rs"]
mod number;
#[cfg(test)]
#[path = "mesh_tests.rs"]
mod tests;
#[path = "mesh_topology.rs"]
mod topology;
use memory::{bytes, phase, product, reserve, sum};

const CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 5],
    [4, 5],
    [0, 4],
    [2, 3],
    [3, 7],
    [6, 7],
    [2, 6],
    [2, 0],
    [3, 1],
    [7, 5],
    [6, 4],
];
fn key(p: [i32; 3], r: i32) -> u64 {
    (p[0] as u64 * r as u64 + p[1] as u64) * r as u64 + p[2] as u64
}
fn decode(k: u64, r: i32) -> [i32; 3] {
    let r = r as u64;
    [(k / (r * r)) as i32, (k / r % r) as i32, (k % r) as i32]
}
fn add(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
#[derive(Clone, Debug)]
struct Cube {
    key: u64,
    vertices: [usize; 8],
    case: usize,
    weights: [f32; 21],
}
struct Vertex {
    position: [f32; 3],
    sdf: f32,
    color: [f32; 6],
}

/// Extract the full inference FlexiCubes surface from `[N,101]` decoder output.
#[must_use = "mesh extraction errors must be handled"]
pub fn extract_mesh(features: &MlxArray, grid: &Grid) -> Result<Mesh> {
    let shape = mlxcel_core::array_shape(features);
    ensure!(
        shape == [grid.len() as i32, 101],
        "FlexiCubes features must be [N,101]"
    );
    let count = product(grid.len(), 101)?;
    // Include the caller's feature tensor and all retained grid arrays/cache
    // in every production phase.
    // Model weights, lazy decoder graphs, and allocator caches remain under the
    // external whole-process resource guard, not this workspace accounting.
    let external = sum(&[
        product(count, mlxcel_core::array_itemsize(features))?,
        grid.resident_bytes()?,
    ])?;
    // Chunk conversion avoids simultaneous full F32 native/raw/host copies.
    let chunk_rows = ((16 << 20) / (101 * 4)).max(1).min(grid.len());
    phase(
        "feature readback",
        external,
        &[
            bytes::<f32>(count)?,
            product(bytes::<f32>(product(chunk_rows, 101)?)?, 3)?,
        ],
    )?;
    let mut f = reserve::<f32>(count)?;
    for start in (0..grid.len()).step_by(chunk_rows.max(1)) {
        let end = (start + chunk_rows).min(grid.len());
        let chunk = mlxcel_core::slice(features, &[start as i32, 0], &[end as i32, 101]);
        let values = mlxcel_core::utils::array_to_vec_f32(&chunk);
        f.extend_from_slice(&values);
    }
    let (vertices, cubes) = prepare(
        &f,
        grid.host_coords(),
        grid.resolution,
        external,
        bytes::<f32>(f.capacity())?,
    )?;
    drop(f);
    topology::extract(vertices, cubes, external)
}
#[cfg(test)]
fn extract_host(f: &[f32], coords: &[[i32; 4]], r: i32) -> Result<Mesh> {
    let external = bytes::<[i32; 4]>(coords.len())?;
    let (vertices, cubes) = prepare(f, coords, r, external, bytes::<f32>(f.len())?)?;
    // Tests borrow the input; unlike production it remains live in topology.
    topology::extract(vertices, cubes, sum(&[external, bytes::<f32>(f.len())?])?)
}
fn prepare(
    f: &[f32],
    coords: &[[i32; 4]],
    r: i32,
    external: usize,
    feature_bytes: usize,
) -> Result<(Vec<Vertex>, Vec<Cube>)> {
    ensure!(r > 0 && r < i32::MAX, "invalid FlexiCubes resolution");
    let key_side = (r + 1) as u64;
    ensure!(
        key_side
            .checked_mul(key_side)
            .and_then(|v| v.checked_mul(key_side))
            .is_some(),
        "FlexiCubes coordinate key overflow"
    );
    ensure!(
        f.len()
            == coords
                .len()
                .checked_mul(101)
                .ok_or_else(|| anyhow::anyhow!("mesh size overflow"))?,
        "mesh features must have 101 channels"
    );
    ensure!(
        f.iter().all(|v| v.is_finite()),
        "FlexiCubes features must be finite"
    );
    ensure!(
        coords
            .iter()
            .all(|p| p[0] == 0 && p[1..].iter().all(|&v| v >= 0 && v < r)),
        "FlexiCubes requires one object and bounded coordinates"
    );
    if coords.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let inputs = sum(&[external, feature_bytes])?;
    let count = product(coords.len(), 8)?;
    phase(
        "corner contributions",
        inputs,
        &[
            bytes::<(u64, usize)>(coords.len())?,
            bytes::<(u64, usize)>(count)?,
        ],
    )?;
    let mut source = reserve::<(u64, usize)>(coords.len())?;
    source.extend(
        coords
            .iter()
            .enumerate()
            .map(|(i, p)| (key([p[1], p[2], p[3]], r), i)),
    );
    source.sort_unstable();
    ensure!(
        source.windows(2).all(|w| w[0].0 != w[1].0),
        "duplicate mesh coordinates"
    );
    let mut contributions = reserve::<(u64, usize)>(count)?;
    for (i, p) in coords.iter().enumerate() {
        for (j, c) in CORNERS.iter().enumerate() {
            contributions.push((key(add([p[1], p[2], p[3]], *c), r + 1), i * 8 + j));
        }
    }
    contributions.sort_unstable();
    let unique = 1 + contributions
        .windows(2)
        .filter(|w| w[0].0 != w[1].0)
        .count();
    let base = sum(&[inputs, bytes::<(u64, usize)>(source.capacity())?])?;
    phase(
        "corner attributes",
        base,
        &[
            bytes::<(u64, usize)>(contributions.capacity())?,
            bytes::<u64>(unique)?,
            bytes::<[f32; 10]>(unique)?,
        ],
    )?;
    let mut keys = reserve::<u64>(unique)?;
    let mut attrs = reserve::<[f32; 10]>(unique)?;
    let mut i = 0;
    while i < contributions.len() {
        let start = i;
        let k = contributions[i].0;
        let mut a = [0.0; 10];
        while i < contributions.len() && contributions[i].0 == k {
            let row = contributions[i].1 / 8;
            let corner = contributions[i].1 % 8;
            let f = &f[row * 101..(row + 1) * 101];
            a[0] += f[corner] - 1.0 / r as f32;
            for d in 0..3 {
                a[1 + d] += f[8 + corner * 3 + d];
            }
            for d in 0..6 {
                a[4 + d] += f[53 + corner * 6 + d];
            }
            i += 1;
        }
        for v in &mut a {
            *v /= (i - start) as f32;
        }
        keys.push(k);
        attrs.push(a);
    }
    drop(contributions);
    let count = product(keys.len(), 8)?;
    let attrs_live = sum(&[
        base,
        bytes::<u64>(keys.capacity())?,
        bytes::<[f32; 10]>(attrs.capacity())?,
    ])?;
    phase("candidate cubes", attrs_live, &[bytes::<u64>(count)?])?;
    let mut candidates = reserve::<u64>(count)?;
    for &k in &keys {
        let p = decode(k, r + 1);
        for c in CORNERS {
            let q = std::array::from_fn(|d| p[d] - c[d]);
            if q.iter().all(|&v| v >= 0 && v < r) {
                candidates.push(key(q, r));
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    let case_of = |k| {
        let p = decode(k, r);
        let mut case = 0;
        for (j, c) in CORNERS.iter().enumerate() {
            if keys
                .binary_search(&key(add(p, *c), r + 1))
                .ok()
                .is_some_and(|i| attrs[i][0] < 0.0)
            {
                case |= 1 << j;
            }
        }
        case
    };
    candidates.retain(|&k| {
        let case = case_of(k);
        case != 0 && case != 255
    });
    phase(
        "surface cases",
        attrs_live,
        &[
            bytes::<u64>(candidates.capacity())?,
            bytes::<(u64, usize)>(candidates.len())?,
        ],
    )?;
    let mut surface = reserve::<(u64, usize)>(candidates.len())?;
    surface.extend(candidates.iter().map(|&k| (k, case_of(k))));
    drop(candidates);
    if surface.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let count = product(surface.len(), 8)?;
    let surface_live = sum(&[attrs_live, bytes::<(u64, usize)>(surface.capacity())?])?;
    phase("surface corner keys", surface_live, &[bytes::<u64>(count)?])?;
    let mut all_keys = reserve::<u64>(count)?;
    for &(k, _) in &surface {
        for c in CORNERS {
            all_keys.push(key(add(decode(k, r), c), r + 1));
        }
    }
    all_keys.sort_unstable();
    all_keys.dedup();
    phase(
        "surface vertex attributes",
        surface_live,
        &[
            bytes::<u64>(all_keys.capacity())?,
            bytes::<Vertex>(all_keys.len())?,
        ],
    )?;
    let mut vertices = reserve::<Vertex>(all_keys.len())?;
    vertices.extend(all_keys.iter().map(|&k| {
        let mut a = [0.0; 10];
        a[0] = 1.0;
        if let Ok(i) = keys.binary_search(&k) {
            a = attrs[i];
        }
        let p = decode(k, r + 1);
        Vertex {
            position: std::array::from_fn(|d| {
                p[d] as f32 / r as f32 - 0.5 + a[d + 1].tanh() * ((1.0 - 1e-8) / (r as f32 * 2.0))
            }),
            sdf: a[0],
            color: std::array::from_fn(|d| sigmoid(a[d + 4])),
        }
    }));
    drop(keys);
    drop(attrs);
    phase(
        "weighted surface cubes",
        base,
        &[
            bytes::<(u64, usize)>(surface.capacity())?,
            bytes::<u64>(all_keys.capacity())?,
            bytes::<Vertex>(vertices.capacity())?,
            bytes::<Cube>(surface.len())?,
        ],
    )?;
    let mut cubes = reserve::<Cube>(surface.len())?;
    for &(k, original) in &surface {
        let p = decode(k, r);
        let check = check::CHECK_TABLE[original];
        let q = std::array::from_fn(|d| p[d] + i32::from(check[d + 1]));
        let invert = check[0] == 1
            && q.iter().all(|&v| v >= 0 && v < r)
            && surface
                .binary_search_by_key(&key(q, r), |v| v.0)
                .ok()
                .is_some_and(|i| check::CHECK_TABLE[surface[i].1][0] == 1);
        let case = if invert { check[4] as usize } else { original };
        let mut indices = [0; 8];
        for (i, c) in CORNERS.iter().enumerate() {
            indices[i] = all_keys
                .binary_search(&key(add(p, *c), r + 1))
                .map_err(|_| anyhow::anyhow!("missing surface corner"))?;
        }
        let mut weights = [0.0; 21];
        if let Ok(i) = source.binary_search_by_key(&k, |v| v.0) {
            weights.copy_from_slice(&f[source[i].1 * 101 + 32..source[i].1 * 101 + 53]);
        }
        cubes.push(Cube {
            key: k,
            vertices: indices,
            case,
            weights,
        });
    }
    // source, surface, and all_keys are released on return; production also
    // releases f before entering topology.
    Ok((vertices, cubes))
}
