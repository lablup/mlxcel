// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Validation and file output for model-neutral scored instances and masks.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use image::{ImageBuffer, Luma};
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use safetensors::{Dtype, View};
use serde::Serialize;

use super::predictor::DenseOutput;

pub(super) struct SerializedMlxView<'a> {
    array: &'a MlxArray,
    shape: Vec<usize>,
    dtype: Dtype,
    byte_len: usize,
}

impl<'a> SerializedMlxView<'a> {
    pub(super) fn new(array: &'a MlxArray, shape: Vec<usize>, dtype: Dtype) -> Result<Self> {
        let item_size: usize = if dtype == Dtype::U8 { 1 } else { 4 };
        let byte_len = shape
            .iter()
            .try_fold(item_size, |size, dim| size.checked_mul(*dim))
            .ok_or_else(|| anyhow::anyhow!("tensor byte length overflowed"))?;
        Ok(Self {
            array,
            shape,
            dtype,
            byte_len,
        })
    }
}

impl View for SerializedMlxView<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Owned(if self.dtype == Dtype::U8 {
            mlxcel_core::array_to_raw_bytes(self.array)
        } else {
            mlxcel_core::utils::array_to_vec_f32(self.array)
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect()
        })
    }

    fn data_len(&self) -> usize {
        self.byte_len
    }
}

#[derive(Serialize)]
struct InstanceRecord {
    bbox: [f32; 4],
    score: f32,
    label: String,
    mask_index: Option<usize>,
    object_id: Option<u32>,
    frame: Option<u32>,
    mask: Option<String>,
}

pub(super) struct PreparedInstances {
    records: Vec<InstanceRecord>,
    mask_pixels: Vec<u8>,
    mask_files: Vec<(usize, String)>,
    mask_size: (u32, u32),
    boxes: Option<UniquePtr<MlxArray>>,
    scores: Option<UniquePtr<MlxArray>>,
}

fn tensor_shape(array: &MlxArray) -> Vec<usize> {
    mlxcel_core::array_shape(array)
        .into_iter()
        .map(|dim| dim as usize)
        .collect()
}

fn validate_instance_tensor(
    output: &DenseOutput,
    name: &str,
    expected_shape: &[usize],
    expected_values: &[f32],
) -> Result<()> {
    if let Some(array) = output.tensors.get(name) {
        let shape = tensor_shape(array);
        ensure!(
            shape == expected_shape,
            "instance tensor `{name}` has shape {shape:?}, expected {expected_shape:?}"
        );
        let actual_values = mlxcel_core::utils::array_to_vec_f32(array);
        ensure!(
            actual_values.as_slice() == expected_values,
            "instance tensor `{name}` values do not match DenseOutput.instances"
        );
    }
    Ok(())
}

pub(super) fn prepare_instances(
    output: &DenseOutput,
    input_size: (u32, u32),
    is_video: bool,
) -> Result<PreparedInstances> {
    let count = output.instances.len();
    let box_values: Vec<f32> = output
        .instances
        .iter()
        .flat_map(|instance| instance.bbox)
        .collect();
    let score_values: Vec<f32> = output
        .instances
        .iter()
        .map(|instance| instance.score)
        .collect();
    validate_instance_tensor(output, "boxes", &[count, 4], &box_values)?;
    validate_instance_tensor(output, "scores", &[count], &score_values)?;

    for (index, instance) in output.instances.iter().enumerate() {
        ensure!(
            instance.bbox.iter().all(|value| value.is_finite()) && instance.score.is_finite(),
            "instance {index} has a non-finite box or score"
        );
    }

    let masks = output.tensors.get("masks");
    let mut mask_pixels = Vec::new();
    let mask_count = if let Some(masks) = masks {
        ensure!(
            mlxcel_core::array_dtype(masks) == dtype::UINT8,
            "instance tensor `masks` must have uint8 dtype"
        );
        let shape = tensor_shape(masks);
        ensure!(
            shape.len() == 3
                && shape[1] == input_size.1 as usize
                && shape[2] == input_size.0 as usize,
            "instance tensor `masks` has shape {shape:?}, expected (N, {}, {})",
            input_size.1,
            input_size.0
        );
        let values = mlxcel_core::utils::array_to_vec_f32(masks);
        ensure!(
            values.iter().all(|value| *value == 0.0 || *value == 1.0),
            "instance tensor `masks` values must be 0 or 1"
        );
        mask_pixels = values.into_iter().map(|value| value as u8).collect();
        shape[0]
    } else {
        0
    };

    let mut referenced = vec![false; mask_count];
    let mut mask_files = Vec::with_capacity(mask_count);
    let mut records = Vec::with_capacity(count);
    for (index, instance) in output.instances.iter().enumerate() {
        let mask = if let Some(mask_index) = instance.mask_index {
            ensure!(
                masks.is_some(),
                "instance {index} references a missing masks tensor"
            );
            ensure!(
                mask_index < mask_count,
                "instance {index} mask_index {mask_index} is out of bounds for {mask_count} masks"
            );
            ensure!(
                !referenced[mask_index],
                "mask row {mask_index} is referenced by more than one instance"
            );
            referenced[mask_index] = true;
            let name = if is_video {
                let frame = instance.frame.ok_or_else(|| {
                    anyhow::anyhow!("video instance {index} with a mask has no frame")
                })?;
                let object_id = instance.object_id.ok_or_else(|| {
                    anyhow::anyhow!("video instance {index} with a mask has no object_id")
                })?;
                format!("f{frame:06}_o{object_id:03}.png")
            } else {
                format!("{mask_index:03}.png")
            };
            ensure!(
                !mask_files.iter().any(|(_, other)| other == &name),
                "instances produce duplicate mask path masks/{name}"
            );
            mask_files.push((mask_index, name.clone()));
            Some(format!("masks/{name}"))
        } else {
            None
        };
        records.push(InstanceRecord {
            bbox: instance.bbox,
            score: instance.score,
            label: instance.label.clone(),
            mask_index: instance.mask_index,
            object_id: instance.object_id,
            frame: instance.frame,
            mask,
        });
    }
    ensure!(
        referenced.iter().all(|value| *value),
        "every masks tensor row must be referenced by exactly one instance"
    );

    let boxes = (!output.instances.is_empty() && !output.tensors.contains_key("boxes"))
        .then(|| mlxcel_core::from_slice_f32(&box_values, &[count as i32, 4]));
    let scores = (!output.instances.is_empty() && !output.tensors.contains_key("scores"))
        .then(|| mlxcel_core::from_slice_f32(&score_values, &[count as i32]));

    Ok(PreparedInstances {
        records,
        mask_pixels,
        mask_files,
        mask_size: input_size,
        boxes,
        scores,
    })
}

impl PreparedInstances {
    pub(super) fn supplemental_tensors(&self) -> impl Iterator<Item = (&'static str, &MlxArray)> {
        [
            (
                "boxes",
                self.boxes.as_ref().and_then(|array| array.as_ref()),
            ),
            (
                "scores",
                self.scores.as_ref().and_then(|array| array.as_ref()),
            ),
        ]
        .into_iter()
        .filter_map(|(name, array)| array.map(|array| (name, array)))
    }

    pub(super) fn remove_stale(&self, output_dir: &Path) -> Result<()> {
        remove_file_if_present(&output_dir.join("instances.json"))?;
        remove_owned_masks(&output_dir.join("masks"))
    }

    pub(super) fn write(&self, output_dir: &Path) -> Result<Vec<PathBuf>> {
        let mut written = Vec::with_capacity(self.mask_files.len() + 1);
        if !self.mask_files.is_empty() {
            let dir = output_dir.join("masks");
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("create mask directory {}", dir.display()))?;
            let row_len = self.mask_size.0 as usize * self.mask_size.1 as usize;
            for (row, name) in &self.mask_files {
                let start = row * row_len;
                let pixels = self.mask_pixels[start..start + row_len]
                    .iter()
                    .map(|value| u8::MAX * u8::from(*value != 0))
                    .collect();
                let image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                    self.mask_size.0,
                    self.mask_size.1,
                    pixels,
                )
                .ok_or_else(|| anyhow::anyhow!("mask {row} data does not match source size"))?;
                let path = dir.join(name);
                image
                    .save(&path)
                    .with_context(|| format!("write mask {}", path.display()))?;
                written.push(path);
            }
        }

        let path = output_dir.join("instances.json");
        let file =
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        serde_json::to_writer_pretty(file, &self.records)
            .with_context(|| format!("write {}", path.display()))?;
        written.push(path);
        Ok(written)
    }
}

fn remove_file_if_present(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() || metadata.file_type().is_symlink() => {
            std::fs::remove_file(path).with_context(|| format!("remove stale {}", path.display()))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {}", path.display())),
    }
}

fn is_owned_mask(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(stem) = name.strip_suffix(".png") else {
        return false;
    };
    if stem.len() >= 3 && stem.bytes().all(|byte| byte.is_ascii_digit()) {
        return true;
    }
    stem.strip_prefix('f')
        .and_then(|value| value.split_once("_o"))
        .is_some_and(|(frame, object)| {
            frame.len() >= 6
                && object.len() >= 3
                && frame.bytes().all(|byte| byte.is_ascii_digit())
                && object.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn remove_owned_masks(path: &Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).with_context(|| format!("inspect {}", path.display())),
    };
    if metadata.file_type().is_symlink() || metadata.file_type().is_file() {
        std::fs::remove_file(path).with_context(|| format!("remove stale {}", path.display()))?;
        return Ok(());
    }
    if !metadata.file_type().is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path).with_context(|| format!("read {}", path.display()))? {
        let entry = entry.with_context(|| format!("read entry in {}", path.display()))?;
        if is_owned_mask(&entry.path()) {
            remove_file_if_present(&entry.path())?;
        }
    }
    if std::fs::read_dir(path)
        .with_context(|| format!("re-read {}", path.display()))?
        .next()
        .is_none()
    {
        std::fs::remove_dir(path).with_context(|| format!("remove empty {}", path.display()))?;
    }
    Ok(())
}
