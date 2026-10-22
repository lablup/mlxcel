// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Dense tensor, metadata, keypoint, and preview writers.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use image::{ImageBuffer, Luma, Rgb};
use safetensors::Dtype;
use serde_json::{Map, Value, json};

use super::output_instances::{SerializedMlxView, prepare_instances};
use super::predictor::DenseOutput;

#[derive(Debug, Clone)]
pub struct DenseOutputInfo<'a> {
    pub family: &'a str,
    pub task: &'a str,
    pub input_size: (u32, u32),
    pub elapsed_ms: f64,
    pub is_video: bool,
}

#[derive(Clone, Copy)]
struct MapShape {
    frames: usize,
    height: usize,
    width: usize,
    channels: usize,
    is_video: bool,
}

fn scalar_map_shape(shape: &[usize], is_video: bool) -> Result<MapShape> {
    let parsed = match shape {
        [height, width] if !is_video => MapShape {
            frames: 1,
            height: *height,
            width: *width,
            channels: 1,
            is_video,
        },
        [frames, height, width] if is_video => MapShape {
            frames: *frames,
            height: *height,
            width: *width,
            channels: 1,
            is_video,
        },
        [height, width, 1] if !is_video => MapShape {
            frames: 1,
            height: *height,
            width: *width,
            channels: 1,
            is_video,
        },
        [1, height, width] if !is_video => MapShape {
            frames: 1,
            height: *height,
            width: *width,
            channels: 1,
            is_video,
        },
        [frames, height, width, 1] if is_video => MapShape {
            frames: *frames,
            height: *height,
            width: *width,
            channels: 1,
            is_video,
        },
        _ => bail!("unsupported scalar-map shape {shape:?}"),
    };
    ensure!(
        parsed.frames > 0 && parsed.height > 0 && parsed.width > 0,
        "map dimensions must be positive"
    );
    Ok(parsed)
}

fn normal_map_shape(shape: &[usize], is_video: bool) -> Result<MapShape> {
    let parsed = match shape {
        [height, width, 3] if !is_video => MapShape {
            frames: 1,
            height: *height,
            width: *width,
            channels: 3,
            is_video,
        },
        [frames, height, width, 3] if is_video => MapShape {
            frames: *frames,
            height: *height,
            width: *width,
            channels: 3,
            is_video,
        },
        [1, height, width, 3] if !is_video => MapShape {
            frames: 1,
            height: *height,
            width: *width,
            channels: 3,
            is_video,
        },
        _ => bail!("unsupported normal-map shape {shape:?}"),
    };
    ensure!(
        parsed.frames > 0 && parsed.height > 0 && parsed.width > 0,
        "map dimensions must be positive"
    );
    Ok(parsed)
}

fn finite_range(values: &[f32]) -> Option<(f32, f32)> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(None, |range, value| match range {
            None => Some((value, value)),
            Some((min, max)) => Some((min.min(value), max.max(value))),
        })
}

fn remove_owned_file(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() || metadata.file_type().is_symlink() => {
            std::fs::remove_file(path)
                .with_context(|| format!("remove stale output {}", path.display()))?;
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("inspect {}", path.display())),
    }
    Ok(())
}

fn is_numbered_preview(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            let bytes = name.as_bytes();
            bytes.len() == 10 && &bytes[6..] == b".png" && bytes[..6].iter().all(u8::is_ascii_digit)
        })
}

fn remove_numbered_previews(path: &Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).with_context(|| format!("inspect {}", path.display())),
    };
    if metadata.file_type().is_symlink() {
        std::fs::remove_file(path)
            .with_context(|| format!("remove stale preview symlink {}", path.display()))?;
        return Ok(());
    }
    if !metadata.file_type().is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("read preview directory {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("read preview entry in {}", path.display()))?;
        if is_numbered_preview(&entry.path()) {
            remove_owned_file(&entry.path())?;
        }
    }
    if std::fs::read_dir(path)
        .with_context(|| format!("re-read preview directory {}", path.display()))?
        .next()
        .is_none()
    {
        std::fs::remove_dir(path)
            .with_context(|| format!("remove empty preview directory {}", path.display()))?;
    }
    Ok(())
}

fn remove_stale_owned_artifacts(output_dir: &Path) -> Result<()> {
    for name in [
        "result.safetensors",
        "meta.json",
        "keypoints.json",
        "depth.png",
        "normal.png",
        "mask.png",
        "segmentation.png",
    ] {
        remove_owned_file(&output_dir.join(name))?;
    }
    for name in ["depth", "normal", "normals", "mask", "segmentation"] {
        remove_numbered_previews(&output_dir.join(name))?;
    }
    Ok(())
}

fn preview_path(root: &Path, key: &str, frame: usize, shape: MapShape) -> Result<PathBuf> {
    if !shape.is_video {
        let name = match key {
            "normal" | "normals" => "normal.png",
            other => return Ok(root.join(format!("{other}.png"))),
        };
        return Ok(root.join(name));
    }
    let dir = root.join(key);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create preview directory {}", dir.display()))?;
    Ok(dir.join(format!("{frame:06}.png")))
}

fn write_depth_previews(
    root: &Path,
    key: &str,
    values: &[f32],
    shape: MapShape,
    range: Option<(f32, f32)>,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(shape.frames);
    let frame_len = shape.height * shape.width;
    let (min, max) = range.unwrap_or((0.0, 0.0));
    let span = f64::from(max) - f64::from(min);
    for frame in 0..shape.frames {
        let pixels = &values[frame * frame_len..(frame + 1) * frame_len];
        let pixels: Vec<u16> = pixels
            .iter()
            .map(|value| {
                if !value.is_finite() || span <= 0.0 {
                    0
                } else {
                    (((f64::from(*value) - f64::from(min)) / span).clamp(0.0, 1.0)
                        * f64::from(u16::MAX))
                    .round() as u16
                }
            })
            .collect();
        let image =
            ImageBuffer::<Luma<u16>, _>::from_raw(shape.width as u32, shape.height as u32, pixels)
                .ok_or_else(|| anyhow::anyhow!("depth preview shape does not match its tensor"))?;
        let path = preview_path(root, key, frame, shape)?;
        image
            .save(&path)
            .with_context(|| format!("write depth preview {}", path.display()))?;
        paths.push(path);
    }
    Ok(paths)
}

fn write_normal_previews(
    root: &Path,
    key: &str,
    values: &[f32],
    shape: MapShape,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(shape.frames);
    let frame_len = shape.height * shape.width * shape.channels;
    for frame in 0..shape.frames {
        let pixels = &values[frame * frame_len..(frame + 1) * frame_len];
        let pixels: Vec<u8> = pixels
            .iter()
            .map(|value| {
                if value.is_finite() {
                    (((value + 1.0) * 0.5).clamp(0.0, 1.0) * u8::MAX as f32).round() as u8
                } else {
                    0
                }
            })
            .collect();
        let image =
            ImageBuffer::<Rgb<u8>, _>::from_raw(shape.width as u32, shape.height as u32, pixels)
                .ok_or_else(|| anyhow::anyhow!("normal preview shape does not match its tensor"))?;
        let path = preview_path(root, key, frame, shape)?;
        image
            .save(&path)
            .with_context(|| format!("write normal preview {}", path.display()))?;
        paths.push(path);
    }
    Ok(paths)
}

fn write_u8_previews(
    root: &Path,
    key: &str,
    values: &[f32],
    shape: MapShape,
    binary: bool,
) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(shape.frames);
    let frame_len = shape.height * shape.width;
    for frame in 0..shape.frames {
        let pixels = &values[frame * frame_len..(frame + 1) * frame_len];
        let pixels: Vec<u8> = pixels
            .iter()
            .map(|value| {
                if !value.is_finite() {
                    0
                } else if binary {
                    u8::MAX * u8::from(*value > 0.5)
                } else {
                    value.round().clamp(0.0, u8::MAX as f32) as u8
                }
            })
            .collect();
        let image =
            ImageBuffer::<Luma<u8>, _>::from_raw(shape.width as u32, shape.height as u32, pixels)
                .ok_or_else(|| anyhow::anyhow!("map preview shape does not match its tensor"))?;
        let path = preview_path(root, key, frame, shape)?;
        image
            .save(&path)
            .with_context(|| format!("write map preview {}", path.display()))?;
        paths.push(path);
    }
    Ok(paths)
}

/// Write the stable dense-prediction file contract and return every written path.
pub fn write_dense_output(
    output_dir: &Path,
    output: &DenseOutput,
    info: &DenseOutputInfo<'_>,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create output directory {}", output_dir.display()))?;
    let instances = prepare_instances(output, info.input_size, info.is_video)?;

    for (name, array) in &output.tensors {
        let shape: Vec<usize> = mlxcel_core::array_shape(array)
            .into_iter()
            .map(|dim| dim as usize)
            .collect();
        match name.as_str() {
            "depth" | "mask" | "segmentation" => {
                scalar_map_shape(&shape, info.is_video)?;
            }
            "normal" | "normals" => {
                normal_map_shape(&shape, info.is_video)?;
            }
            _ => {}
        }
    }
    remove_stale_owned_artifacts(output_dir)?;
    instances.remove_stale(output_dir)?;

    let mut tensors = BTreeMap::new();
    let mut map_ranges = Map::new();
    let mut intrinsics = None;
    let mut written = Vec::new();
    for (name, array) in &output.tensors {
        let shape: Vec<usize> = mlxcel_core::array_shape(array)
            .into_iter()
            .map(|dim| dim as usize)
            .collect();
        let values = if name == "masks" {
            Vec::new()
        } else {
            mlxcel_core::utils::array_to_vec_f32(array)
        };
        if name != "masks" {
            ensure!(
                values.len() == shape.iter().product::<usize>(),
                "tensor {name} data length does not match shape {shape:?}"
            );
        }
        let dtype = if name == "masks" {
            Dtype::U8
        } else {
            Dtype::F32
        };
        let range = if name == "masks" {
            instances.mask_range()
        } else {
            finite_range(&values)
        };
        map_ranges.insert(
            name.clone(),
            range.map_or(Value::Null, |(min, max)| json!({ "min": min, "max": max })),
        );
        if name == "intrinsics" {
            intrinsics = Some(json!({ "shape": &shape, "values": &values }));
        }

        let mut preview_paths = match name.as_str() {
            "depth" => write_depth_previews(
                output_dir,
                name,
                &values,
                scalar_map_shape(&shape, info.is_video)?,
                range,
            )?,
            "normal" | "normals" => write_normal_previews(
                output_dir,
                name,
                &values,
                normal_map_shape(&shape, info.is_video)?,
            )?,
            "mask" => write_u8_previews(
                output_dir,
                name,
                &values,
                scalar_map_shape(&shape, info.is_video)?,
                true,
            )?,
            "segmentation" => write_u8_previews(
                output_dir,
                name,
                &values,
                scalar_map_shape(&shape, info.is_video)?,
                false,
            )?,
            _ => Vec::new(),
        };
        written.append(&mut preview_paths);

        drop(values);
        let array = array
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tensor {name} has a null MLX array"))?;
        tensors.insert(name.clone(), SerializedMlxView::new(array, shape, dtype)?);
    }

    for (name, array) in instances.supplemental_tensors() {
        let values = mlxcel_core::utils::array_to_vec_f32(array);
        map_ranges.insert(
            name.into(),
            finite_range(&values)
                .map_or(Value::Null, |(min, max)| json!({ "min": min, "max": max })),
        );
        let shape: Vec<usize> = mlxcel_core::array_shape(array)
            .into_iter()
            .map(|dim| dim as usize)
            .collect();
        tensors.insert(
            name.into(),
            SerializedMlxView::new(array, shape, Dtype::F32)?,
        );
    }

    let safetensors_path = output_dir.join("result.safetensors");
    safetensors::serialize_to_file(tensors, None::<HashMap<String, String>>, &safetensors_path)
        .with_context(|| format!("write {}", safetensors_path.display()))?;
    written.insert(0, safetensors_path);
    written.extend(instances.write(output_dir)?);

    if !output.keypoints.is_empty() {
        let path = output_dir.join("keypoints.json");
        let file =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        serde_json::to_writer_pretty(file, &output.keypoints)
            .with_context(|| format!("write {}", path.display()))?;
        written.push(path);
    }

    let mut meta = output.meta.clone();
    meta.insert("family".into(), Value::String(info.family.into()));
    meta.insert("task".into(), Value::String(info.task.into()));
    meta.insert(
        "input_size".into(),
        json!({ "width": info.input_size.0, "height": info.input_size.1 }),
    );
    meta.insert("maps".into(), Value::Object(map_ranges));
    meta.insert("timing_ms".into(), json!(info.elapsed_ms));
    if let Some(intrinsics) = intrinsics {
        meta.insert("intrinsics".into(), intrinsics);
    }
    let meta_path = output_dir.join("meta.json");
    let file = std::fs::File::create(&meta_path)
        .with_context(|| format!("create {}", meta_path.display()))?;
    serde_json::to_writer_pretty(file, &meta)
        .with_context(|| format!("write {}", meta_path.display()))?;
    written.push(meta_path);
    Ok(written)
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
