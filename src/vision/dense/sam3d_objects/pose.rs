// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Object-local pose decoding (WXYZ quaternion and separate world transform).
use super::{flow::State, processing::Metadata};
use anyhow::{Context, Result, ensure};
use mlxcel_core::{self as mx, MlxArray, UniquePtr};
use std::collections::BTreeMap;
fn normalized(mut x: [f32; 3]) -> [f32; 3] {
    let n = (x.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-12);
    for v in &mut x {
        *v /= n;
    }
    x
}
// Preserve the pinned upstream pose-normalization decimal literals.
#[allow(clippy::excessive_precision)]
pub fn decode_pose(
    s: &State,
    m: &Metadata,
    factor: usize,
) -> Result<BTreeMap<String, UniquePtr<MlxArray>>> {
    let values = |n: &str| -> Result<Vec<f32>> {
        Ok(mx::utils::array_to_vec_f32(
            s.get(n)
                .with_context(|| format!("missing pose latent {n}"))?,
        ))
    };
    let mut rot = values("6drotation_normalized")?;
    let mean = [
        -0.06366085,
        0.008438225,
        0.00017084787,
        0.00071266104,
        -0.0030916727,
        0.5166094,
    ];
    let std = [
        0.6656972, 0.6787012, 0.3034501, 0.43945044, 0.39817974, 0.6176287,
    ];
    ensure!(rot.len() == 6, "rotation latent must have six elements");
    for i in 0..6 {
        rot[i] = rot[i] * std[i] + mean[i];
    }
    let a = normalized([rot[0], rot[1], rot[2]]);
    let dot = a[0] * rot[3] + a[1] * rot[4] + a[2] * rot[5];
    let b = normalized([
        rot[3] - dot * a[0],
        rot[4] - dot * a[1],
        rot[5] - dot * a[2],
    ]);
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let mat = [a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]];
    let qabs = [
        1.0 + mat[0] + mat[4] + mat[8],
        1.0 + mat[0] - mat[4] - mat[8],
        1.0 - mat[0] + mat[4] - mat[8],
        1.0 - mat[0] - mat[4] + mat[8],
    ]
    .map(|v| v.max(0.0).sqrt());
    let candidates = [
        [
            qabs[0] * qabs[0],
            mat[7] - mat[5],
            mat[2] - mat[6],
            mat[3] - mat[1],
        ],
        [
            mat[7] - mat[5],
            qabs[1] * qabs[1],
            mat[1] + mat[3],
            mat[2] + mat[6],
        ],
        [
            mat[2] - mat[6],
            mat[1] + mat[3],
            qabs[2] * qabs[2],
            mat[5] + mat[7],
        ],
        [
            mat[3] - mat[1],
            mat[2] + mat[6],
            mat[5] + mat[7],
            qabs[3] * qabs[3],
        ],
    ];
    let mut best = 0;
    for i in 1..4 {
        if qabs[i] > qabs[best] {
            best = i;
        }
    }
    let mut q = candidates[best].map(|v| v / (2.0 * qabs[best].max(0.1)));
    if q[0] < 0.0 {
        q = q.map(|v| -v);
    }
    let scale = values("scale")?;
    let translation = values("translation")?;
    ensure!(
        scale.len() == 3 && translation.len() == 3,
        "invalid pose latent shape"
    );
    let scale = scale.iter().map(|v| v.exp() * m.scale).sum::<f32>() / 3.0 * factor as f32;
    let t = [
        translation[0] * m.scale + m.shift[0],
        translation[1] * m.scale + m.shift[1],
        translation[2] * m.scale + m.shift[2],
    ];
    ensure!(
        q.iter().chain(&mat).chain(&t).all(|v| v.is_finite()) && scale.is_finite(),
        "non-finite object pose"
    );
    Ok(BTreeMap::from([
        ("rotation".into(), mx::from_slice_f32(&q, &[4])),
        ("rotation_matrix".into(), mx::from_slice_f32(&mat, &[3, 3])),
        ("translation".into(), mx::from_slice_f32(&t, &[3])),
        ("scale".into(), mx::from_slice_f32(&[scale; 3], &[3])),
    ]))
}
