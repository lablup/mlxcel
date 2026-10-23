// Copyright (c) Meta Platforms, Inc. and affiliates.
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use super::memory::{bytes, phase, product, reserve, sum};
use super::{Cube, EDGES, Vertex, dmc, number, sigmoid};
use crate::vision::dense::geometry_io::Mesh;
use anyhow::{Result, ensure};

pub(super) fn extract(vertices: Vec<Vertex>, cubes: Vec<Cube>, external: usize) -> Result<Mesh> {
    let total = cubes.iter().try_fold(0usize, |n, c| {
        n.checked_add(number::NUM_VD_TABLE[c.case] as usize)
            .ok_or_else(|| anyhow::anyhow!("FlexiCubes dual count overflow"))
    })?;
    ensure!(
        total <= u32::MAX as usize,
        "FlexiCubes vertex count exceeds u32"
    );
    let input = sum(&[
        external,
        bytes::<Vertex>(vertices.capacity())?,
        bytes::<Cube>(cubes.capacity())?,
    ])?;
    phase(
        "dual vertices",
        input,
        &[
            bytes::<[f32; 3]>(total)?,
            bytes::<[f32; 6]>(total)?,
            bytes::<f32>(total)?,
            bytes::<[u32; 12]>(cubes.len())?,
        ],
    )?;
    let mut mesh = Mesh {
        vertices: reserve(total)?,
        vertex_colors: reserve(total)?,
        faces: Vec::new(),
    };
    let mut gammas = reserve::<f32>(total)?;
    let mut dual_indices = reserve::<[u32; 12]>(cubes.len())?;
    dual_indices.resize(cubes.len(), [u32::MAX; 12]);
    // Reference groups cubes by dual-vertex count, then cube key, then group.
    for number in 1..=4 {
        for (ci, cube) in cubes.iter().enumerate() {
            if number::NUM_VD_TABLE[cube.case] as usize != number {
                continue;
            }
            for group in &dmc::DMC_TABLE[cube.case][..number] {
                let id = mesh.vertices.len() as u32;
                let mut position = [0.0; 3];
                let mut color = [0.0; 6];
                let mut denominator = 0.0;
                for &edge in group.iter().filter(|&&v| v >= 0) {
                    let edge = edge as usize;
                    let [a, b] = EDGES[edge];
                    let va = &vertices[cube.vertices[a]];
                    let vb = &vertices[cube.vertices[b]];
                    ensure!(
                        (va.sdf < 0.0) != (vb.sdf < 0.0),
                        "FlexiCubes table selected non-crossing edge in cube {}",
                        cube.key
                    );
                    let sa = va.sdf * (1.0 + 0.99 * cube.weights[12 + a].tanh());
                    let sb = vb.sdf * (1.0 + 0.99 * cube.weights[12 + b].tanh());
                    let t0 = sb / (sb - sa);
                    let t1 = -sa / (sb - sa);
                    let beta = 1.0 + 0.99 * cube.weights[edge].tanh();
                    for d in 0..3 {
                        position[d] += (va.position[d] * t0 + vb.position[d] * t1) * beta;
                    }
                    for d in 0..6 {
                        color[d] += (va.color[d] * t0 + vb.color[d] * t1) * beta;
                    }
                    denominator += beta;
                    dual_indices[ci][edge] = id;
                }
                ensure!(denominator > 0.0, "empty FlexiCubes dual group");
                mesh.vertices.push(position.map(|v| v / denominator));
                mesh.vertex_colors.push(color.map(|v| v / denominator));
                gammas.push(0.99 * sigmoid(cube.weights[20]) + 0.005);
            }
        }
    }
    // Count actual crossings instead of reserving all twelve edges per cube.
    let edge_count = cubes.iter().try_fold(0usize, |count, cube| {
        let crossing = EDGES
            .iter()
            .filter(|&&[a, b]| {
                (vertices[cube.vertices[a]].sdf < 0.0) != (vertices[cube.vertices[b]].sdf < 0.0)
            })
            .count();
        count
            .checked_add(crossing)
            .ok_or_else(|| anyhow::anyhow!("mesh edge count overflow"))
    })?;
    let duals_live = sum(&[
        bytes::<[f32; 3]>(mesh.vertices.capacity())?,
        bytes::<[f32; 6]>(mesh.vertex_colors.capacity())?,
        bytes::<f32>(gammas.capacity())?,
        bytes::<[u32; 12]>(dual_indices.capacity())?,
    ])?;
    phase(
        "crossing edges",
        input,
        &[
            duals_live,
            bytes::<(usize, usize, usize, usize)>(edge_count)?,
        ],
    )?;
    let mut edges = reserve::<(usize, usize, usize, usize)>(edge_count)?;
    for (ci, cube) in cubes.iter().enumerate() {
        for (ei, &[a, b]) in EDGES.iter().enumerate() {
            let a = cube.vertices[a];
            let b = cube.vertices[b];
            if (vertices[a].sdf < 0.0) != (vertices[b].sdf < 0.0) {
                edges.push((a, b, ci, ei));
            }
        }
    }
    drop(cubes);
    // Directed endpoint IDs intentionally match the pinned table's edge keys.
    edges.sort_unstable();
    let (mut flip_count, mut other_count) = (0usize, 0usize);
    let mut i = 0;
    while i < edges.len() {
        let start = i;
        let pair = (edges[i].0, edges[i].1);
        while i < edges.len() && (edges[i].0, edges[i].1) == pair {
            i += 1;
        }
        if i - start == 4 {
            if vertices[pair.0].sdf > 0.0 {
                flip_count += 1;
            } else {
                other_count += 1;
            }
        }
    }
    let quads = sum(&[flip_count, other_count])?;
    phase(
        "ordered quads",
        external,
        &[
            bytes::<Vertex>(vertices.capacity())?,
            duals_live,
            bytes::<(usize, usize, usize, usize)>(edges.capacity())?,
            bytes::<[u32; 4]>(quads)?,
        ],
    )?;
    let mut flipped = reserve::<[u32; 4]>(flip_count)?;
    let mut unflipped = reserve::<[u32; 4]>(other_count)?;
    let mut i = 0;
    while i < edges.len() {
        let start = i;
        let pair = (edges[i].0, edges[i].1);
        while i < edges.len() && (edges[i].0, edges[i].1) == pair {
            i += 1;
        }
        if i - start != 4 {
            continue;
        }
        let mut q = [0; 4];
        for (j, e) in edges[start..i].iter().enumerate() {
            q[j] = dual_indices[e.2][e.3];
        }
        ensure!(
            q.iter().all(|&v| v != u32::MAX),
            "missing FlexiCubes dual edge"
        );
        if vertices[pair.0].sdf > 0.0 {
            flipped.push([q[0], q[1], q[3], q[2]]);
        } else {
            unflipped.push([q[2], q[3], q[1], q[0]]);
        }
    }
    drop(edges);
    drop(dual_indices);
    drop(vertices);
    let face_count = product(quads, 2)?;
    phase(
        "triangle faces",
        external,
        &[
            bytes::<[f32; 3]>(mesh.vertices.capacity())?,
            bytes::<[f32; 6]>(mesh.vertex_colors.capacity())?,
            bytes::<f32>(gammas.capacity())?,
            bytes::<[u32; 4]>(sum(&[flipped.capacity(), unflipped.capacity()])?)?,
            bytes::<[u32; 3]>(face_count)?,
        ],
    )?;
    mesh.faces = reserve::<[u32; 3]>(face_count)?;
    for q in flipped.into_iter().chain(unflipped) {
        let g = q.map(|v| gammas[v as usize]);
        if g[0] * g[2] > g[1] * g[3] {
            mesh.faces.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
        } else {
            mesh.faces.extend([[q[0], q[1], q[3]], [q[3], q[1], q[2]]]);
        }
    }
    mesh.validate()?;
    Ok(mesh)
}
