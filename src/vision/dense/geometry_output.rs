// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::{
    geometry_io::{write_gaussian_ply, write_obj},
    predictor::DenseOutput,
};
use anyhow::{Result, ensure};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
pub(super) fn validate(output: &DenseOutput) -> Result<()> {
    let mut names = BTreeSet::new();
    for (name, mesh) in &output.meshes {
        ensure!(
            name == "mesh" && names.insert(name),
            "unsupported or duplicate mesh output name {name}"
        );
        mesh.validate()?;
    }
    for (name, gaussian) in &output.gaussians {
        ensure!(
            ["gaussian", "gaussian_4"].contains(&name.as_str()) && names.insert(name),
            "unsupported or duplicate Gaussian output name {name}"
        );
        gaussian.validate()?;
    }
    Ok(())
}
pub(super) fn write(root: &Path, output: &DenseOutput) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for (name, mesh) in &output.meshes {
        let path = root.join(format!("{name}.obj"));
        write_obj(&path, mesh)?;
        paths.push(path);
    }
    for (name, gaussian) in &output.gaussians {
        let path = root.join(format!("{name}.ply"));
        write_gaussian_ply(&path, gaussian)?;
        paths.push(path);
    }
    Ok(paths)
}
