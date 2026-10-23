// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Convert mask logits into tight normalized query boxes.

use mlxcel_core::{MlxArray, UniquePtr};

pub fn mask_to_box_coordinate(mask_logits: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(mask_logits);
    if shape.len() != 4 || shape.iter().any(|&value| value <= 0) {
        return Err(format!(
            "PP-DocLayoutV3 mask logits must have positive [B,Q,H,W] shape, got {shape:?}"
        ));
    }
    let (batch, queries, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::array_dtype(mask_logits));
    let active = mlxcel_core::greater(mask_logits, &zero);
    let active = mlxcel_core::reshape(&active, &[batch, queries, height * width]);
    let active_f32 = mlxcel_core::astype(&active, mlxcel_core::dtype::FLOAT32);

    let mut xs = Vec::with_capacity((height * width) as usize);
    let mut ys = Vec::with_capacity((height * width) as usize);
    for y in 0..height {
        for x in 0..width {
            xs.push(x as f32);
            ys.push(y as f32);
        }
    }
    let xs = mlxcel_core::from_slice_f32(&xs, &[1, 1, height * width]);
    let ys = mlxcel_core::from_slice_f32(&ys, &[1, 1, height * width]);
    let xs = mlxcel_core::broadcast_to(&xs, &[batch, queries, height * width]);
    let ys = mlxcel_core::broadcast_to(&ys, &[batch, queries, height * width]);
    let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
    let big = mlxcel_core::full_f32(&[1], 16_777_216.0, mlxcel_core::dtype::FLOAT32);

    let x_max = mlxcel_core::add(
        &mlxcel_core::max_axis(&mlxcel_core::multiply(&xs, &active_f32), -1, false),
        &one,
    );
    let y_max = mlxcel_core::add(
        &mlxcel_core::max_axis(&mlxcel_core::multiply(&ys, &active_f32), -1, false),
        &one,
    );
    let x_min = mlxcel_core::min_axis(&mlxcel_core::where_cond(&active, &xs, &big), -1, false);
    let y_min = mlxcel_core::min_axis(&mlxcel_core::where_cond(&active, &ys, &big), -1, false);
    let nonempty = mlxcel_core::astype(
        &mlxcel_core::any_axis(&active, -1, false),
        mlxcel_core::dtype::FLOAT32,
    );
    let nonempty = mlxcel_core::expand_dims(&nonempty, -1);

    let corners = mlxcel_core::stack_owned(&[x_min, y_min, x_max, y_max], -1);
    let scale = mlxcel_core::from_slice_f32(
        &[width as f32, height as f32, width as f32, height as f32],
        &[4],
    );
    let corners = mlxcel_core::multiply(&mlxcel_core::divide(&corners, &scale), &nonempty);
    let component = |index: i32| {
        let sliced = mlxcel_core::slice(&corners, &[0, 0, index], &[batch, queries, index + 1]);
        mlxcel_core::reshape(&sliced, &[batch, queries])
    };
    let x0 = component(0);
    let y0 = component(1);
    let x1 = component(2);
    let y1 = component(3);
    let two = mlxcel_core::full_f32(&[1], 2.0, mlxcel_core::dtype::FLOAT32);
    Ok(mlxcel_core::stack_owned(
        &[
            mlxcel_core::divide(&mlxcel_core::add(&x0, &x1), &two),
            mlxcel_core::divide(&mlxcel_core::add(&y0, &y1), &two),
            mlxcel_core::subtract(&x1, &x0),
            mlxcel_core::subtract(&y1, &y0),
        ],
        -1,
    ))
}
