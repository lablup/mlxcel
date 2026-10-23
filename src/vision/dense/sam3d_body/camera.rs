// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Camera ray conditioning and the upstream crop-relative keypoint projection.
use super::{
    config::Config,
    layers::Array,
    ops::{cat, scalar_add, slice_axis},
    preprocess::InputGeometry,
};
use mlxcel_core::{self as mx, MlxArray};
pub fn rays(c: &Config, g: &InputGeometry) -> Array {
    let [h, w] = c.image_size.map(|v| v as i32);
    let b = g.bbox;
    let m = g.cam_int;
    let y = mx::arange_f32(0.0, h as f32, 1.0);
    let x = mx::arange_f32(0.0, w as f32, 1.0);
    let y = mx::broadcast_to(&mx::reshape(&y, &[h, 1, 1]), &[h, w, 1]);
    let x = mx::broadcast_to(&mx::reshape(&x, &[1, w, 1]), &[h, w, 1]);
    let x = mx::divide_scalar(
        &scalar_add(
            &scalar_add(
                &mx::multiply_scalar(&x, ((b[2] - b[0]) / f64::from(w)) as f32),
                b[0] as f32,
            ),
            -m[2],
        ),
        m[0],
    );
    let y = mx::divide_scalar(
        &scalar_add(
            &scalar_add(
                &mx::multiply_scalar(&y, ((b[3] - b[1]) / f64::from(h)) as f32),
                b[1] as f32,
            ),
            -m[5],
        ),
        m[4],
    );
    mx::reshape(&cat(&[&x, &y], -1), &[1, h, w, 2])
}
pub fn encoded_rays(c: &Config, g: &InputGeometry) -> Array {
    let p = c.patch_size as i32;
    let [h, w] = c.image_size.map(|v| (v / c.patch_size) as i32);
    // nn.AvgPool2d's nonoverlapping path reduces patch axes, not a convolution.
    let windows = mx::transpose_axes(
        &mx::reshape(&rays(c, g), &[1, h, p, w, p, 2]),
        &[0, 1, 3, 2, 4, 5],
    );
    let down = mx::mean_axis(&mx::reshape(&windows, &[1, h, w, p * p, 2]), 3, false);
    let xyz = cat(&[&down, &mx::ones(&[1, h, w, 1], mx::dtype::FLOAT32)], -1);
    let freq = mx::linspace(1.0, 32.0, 16, mx::dtype::FLOAT32);
    let phase = mx::reshape(
        &mx::multiply(&mx::reshape(&xyz, &[1, h * w, 3, 1]), &freq),
        &[1, h * w, 48],
    );
    let phase = mx::multiply_scalar(&phase, std::f32::consts::PI);
    let encoded = cat(
        &[
            &mx::reshape(&xyz, &[1, h * w, 3]),
            &mx::sin(&phase),
            &mx::cos(&phase),
        ],
        -1,
    );
    mx::reshape(&encoded, &[1, h, w, 99])
}
pub fn project(kp: &MlxArray, cam: &MlxArray, g: &InputGeometry) -> Array {
    let b = g.bbox;
    let f = g.cam_int[0];
    let cx = (b[0] + b[2]) / 2.0;
    let cy = (b[1] + b[3]) / 2.0;
    let size = (b[2] - b[0]).max(b[3] - b[1]);
    let s = mx::negative(&slice_axis(cam, 1, 0, 1));
    let tx = slice_axis(cam, 1, 1, 2);
    let ty = mx::negative(&slice_axis(cam, 1, 2, 3));
    let bs = scalar_add(&mx::multiply_scalar(&s, size as f32), 1e-8);
    let divide = |v: f32| mx::divide(&mx::from_slice_f32(&[v], &[]), &bs);
    let offset = cat(
        &[
            &mx::add(&tx, &divide((2.0 * (cx - g.width as f64 / 2.0)) as f32)),
            &mx::add(&ty, &divide((2.0 * (cy - g.height as f64 / 2.0)) as f32)),
            &divide(2.0 * f),
        ],
        1,
    );
    let translated = mx::add(kp, &mx::reshape(&offset, &[1, 1, 3]));
    let norm = mx::divide(
        &translated,
        &scalar_add(&slice_axis(&translated, 2, 2, 3), 1e-8),
    );
    let x = scalar_add(
        &mx::multiply_scalar(&slice_axis(&norm, 2, 0, 1), f),
        g.width as f32 / 2.0,
    );
    let y = scalar_add(
        &mx::multiply_scalar(&slice_axis(&norm, 2, 1, 2), f),
        g.height as f32 / 2.0,
    );
    cat(
        &[
            &mx::divide_scalar(&scalar_add(&x, -cx as f32), (size * 1.2 / 2.0) as f32),
            &mx::divide_scalar(&scalar_add(&y, -cy as f32), (size * 1.2 / 2.0) as f32),
        ],
        -1,
    )
}
/// Upstream grid_sample_2d clamps corner addresses, retaining extrapolation weights.
pub fn sample(features: &MlxArray, coords: &MlxArray) -> Array {
    let s = mx::array_shape(features);
    let n = mx::array_shape(coords)[1];
    let x = mx::divide_scalar(
        &mx::multiply_scalar(
            &scalar_add(&slice_axis(coords, 2, 0, 1), 1.0),
            (s[2] - 1) as f32,
        ),
        2.0,
    );
    let y = mx::divide_scalar(
        &mx::multiply_scalar(
            &scalar_add(&slice_axis(coords, 2, 1, 2), 1.0),
            (s[1] - 1) as f32,
        ),
        2.0,
    );
    let x0 = mx::floor(&x);
    let y0 = mx::floor(&y);
    let x1 = scalar_add(&x0, 1.0);
    let y1 = scalar_add(&y0, 1.0);
    let fetch = |xx: &MlxArray, yy: &MlxArray| {
        let clip = |a: &MlxArray, max: i32| {
            mx::astype(
                &mx::clip(
                    a,
                    &mx::from_slice_f32(&[0.0], &[]),
                    &mx::from_slice_f32(&[max as f32], &[]),
                ),
                mx::dtype::INT32,
            )
        };
        let index = mx::add(
            &mx::multiply(&clip(yy, s[1] - 1), &mx::from_slice_i32(&[s[2]], &[])),
            &clip(xx, s[2] - 1),
        );
        mx::reshape(
            &mx::take(
                &mx::reshape(features, &[s[1] * s[2], s[3]]),
                &mx::reshape(&index, &[n]),
                0,
            ),
            &[1, n, s[3]],
        )
    };
    let wa = mx::multiply(&mx::subtract(&x1, &x), &mx::subtract(&y1, &y));
    let wb = mx::multiply(&mx::subtract(&x, &x0), &mx::subtract(&y1, &y));
    let wc = mx::multiply(&mx::subtract(&x1, &x), &mx::subtract(&y, &y0));
    let wd = mx::multiply(&mx::subtract(&x, &x0), &mx::subtract(&y, &y0));
    mx::add(
        &mx::add(
            &mx::add(
                &mx::multiply(&wa, &fetch(&x0, &y0)),
                &mx::multiply(&wb, &fetch(&x1, &y0)),
            ),
            &mx::multiply(&wc, &fetch(&x0, &y1)),
        ),
        &mx::multiply(&wd, &fetch(&x1, &y1)),
    )
}
