// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Validated object-local triangle meshes and Gaussian-splat exports.
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub vertex_colors: Option<Vec<[f32; 6]>>,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gaussians {
    pub positions: Vec<[f32; 3]>,
    pub sh_dc: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
}
impl Mesh {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.vertex_colors
                .as_ref()
                .is_none_or(|c| self.vertices.len() == c.len()),
            "mesh colors must match vertex count"
        );
        ensure!(
            self.vertices
                .iter()
                .flatten()
                .chain(self.vertex_colors.iter().flatten().flatten())
                .all(|v| v.is_finite()),
            "mesh attributes must be finite"
        );
        ensure!(
            self.faces
                .iter()
                .flatten()
                .all(|&v| (v as usize) < self.vertices.len()),
            "mesh face index out of bounds"
        );
        Ok(())
    }
}
impl Gaussians {
    pub fn validate(&self) -> Result<()> {
        let n = self.positions.len();
        ensure!(
            [
                self.sh_dc.len(),
                self.scales.len(),
                self.rotations.len(),
                self.opacities.len()
            ]
            .iter()
            .all(|&v| v == n),
            "Gaussian attributes must match position count"
        );
        ensure!(
            self.positions
                .iter()
                .flatten()
                .chain(self.sh_dc.iter().flatten())
                .chain(self.rotations.iter().flatten())
                .chain(self.opacities.iter())
                .all(|v| v.is_finite()),
            "Gaussian attributes must be finite"
        );
        ensure!(
            self.scales
                .iter()
                .flatten()
                .all(|v| v.is_finite() && *v > 0.0),
            "Gaussian scales must be finite and positive"
        );
        Ok(())
    }
}
/// Match NumPy's `%.8g`: promote f32 to double, round to eight significant
/// decimal digits, then choose fixed/scientific using the rounded exponent.
fn obj_float(value: f32) -> Result<String> {
    let scientific = format!("{:.7e}", f64::from(value));
    let (mantissa, exponent) = scientific.split_once('e').context("format OBJ exponent")?;
    let exponent: i32 = exponent.parse().context("parse OBJ exponent")?;
    if !(-4..8).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        return Ok(format!("{mantissa}e{exponent:+03}"));
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let mut fixed = if exponent < 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-exponent - 1) as usize))
    } else {
        let split = (exponent + 1) as usize;
        format!("{sign}{}.{}", &digits[..split], &digits[split..])
    };
    // Only fractional zeroes are removable: integer trailing zeroes carry place
    // value. Every branch above constructs a decimal point before trimming.
    while fixed.ends_with('0') {
        fixed.pop();
    }
    if fixed.ends_with('.') {
        fixed.pop();
    }
    Ok(fixed)
}

/// Write RGB vertices and one-based triangle indices after validating every row.
#[must_use = "geometry validation and I/O errors must be handled"]
pub fn write_obj(path: impl AsRef<Path>, mesh: &Mesh) -> Result<()> {
    mesh.validate()?;
    let path = path.as_ref();
    let mut out =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    for (i, v) in mesh.vertices.iter().enumerate() {
        if let Some(colors) = &mesh.vertex_colors {
            let c = colors[i];
            writeln!(
                out,
                "v {} {} {} {} {} {}",
                obj_float(v[0])?,
                obj_float(v[1])?,
                obj_float(v[2])?,
                obj_float(c[0])?,
                obj_float(c[1])?,
                obj_float(c[2])?
            )?;
        } else {
            writeln!(out, "v {:.6} {:.6} {:.6}", v[0], v[1], v[2])?;
        }
    }
    for f in &mesh.faces {
        writeln!(
            out,
            "f {} {} {}",
            u64::from(f[0]) + 1,
            u64::from(f[1]) + 1,
            u64::from(f[2]) + 1
        )?;
    }
    out.flush().context("flush OBJ")
}
/// Write standard binary little-endian Gaussian fields in upstream order.
#[must_use = "geometry validation and I/O errors must be handled"]
pub fn write_gaussian_ply(path: impl AsRef<Path>, gaussian: &Gaussians) -> Result<()> {
    gaussian.validate()?;
    let path = path.as_ref();
    let mut out =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    writeln!(
        out,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}",
        gaussian.positions.len()
    )?;
    for field in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
        "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        writeln!(out, "property float {field}")?;
    }
    writeln!(out, "end_header")?;
    for i in 0..gaussian.positions.len() {
        let p = gaussian.opacities[i].clamp(1e-7, 1.0 - 1e-7);
        let opacity = [(p / (1.0 - p)).ln()];
        let scale = gaussian.scales[i].map(f32::ln);
        for v in gaussian.positions[i]
            .iter()
            .chain([0.0; 3].iter())
            .chain(gaussian.sh_dc[i].iter())
            .chain(opacity.iter())
            .chain(scale.iter())
            .chain(gaussian.rotations[i].iter())
        {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    out.flush().context("flush Gaussian PLY")
}
#[cfg(test)]
#[path = "geometry_io_tests.rs"]
mod tests;
