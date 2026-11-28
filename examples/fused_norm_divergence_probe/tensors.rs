// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Byte dumps of the probe's per-layer tensors and the offline diff of two
//! arms' dumps (issue #3501).
//!
//! A dump directory holds one raw file per tensor plus `manifest.csv`
//! (`name,dtype,shape`), in the order the forward produced them, so the diff
//! reports the first tensor the two arms disagree on in forward order.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlxcel_core::{MlxArray, dtype};

/// One tensor read back from a dump.
pub struct Dumped {
    pub name: String,
    pub dtype: i32,
    pub shape: Vec<i32>,
    pub bytes: Vec<u8>,
}

/// Writes the tensors of one forward, in order, to a directory.
pub struct DumpWriter {
    dir: PathBuf,
    manifest: std::fs::File,
}

impl DumpWriter {
    pub fn create(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let mut manifest = std::fs::File::create(dir.join("manifest.csv"))?;
        writeln!(manifest, "name,dtype,shape")?;
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
        })
    }

    pub fn write(&mut self, name: &str, arr: &MlxArray) -> Result<()> {
        mlxcel_core::eval(arr);
        let bytes = mlxcel_core::array_to_raw_bytes(arr);
        let shape = mlxcel_core::array_shape(arr)
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join("x");
        std::fs::write(self.dir.join(format!("{name}.bin")), &bytes)?;
        writeln!(
            self.manifest,
            "{name},{},{shape}",
            mlxcel_core::array_dtype(arr)
        )?;
        Ok(())
    }
}

/// Bytes per element of the dtypes this probe writes and decodes.
fn dtype_size(dt: i32) -> Result<usize> {
    match dt {
        dtype::BFLOAT16 | dtype::FLOAT16 => Ok(2),
        dtype::FLOAT32 => Ok(4),
        other => bail!("dtype {other} is not a dtype this probe dumps"),
    }
}

/// A manifest name becomes a file name inside the dump directory, so it must
/// not be able to point anywhere else.
fn check_tensor_name(name: &str) -> Result<()> {
    if name.is_empty() || name.contains(['/', '\\']) || name.contains("..") {
        bail!("tensor name {name:?} is empty, holds a path separator or `..`");
    }
    Ok(())
}

/// Read every tensor of a dump, checking each manifest entry against its file
/// before anything is handed to `from_bytes`, which trusts the shape: the name
/// stays inside the directory, the dtype is known, no dimension is negative,
/// and the element count times the element size equals the file size.
pub fn read_dump(dir: &Path) -> Result<Vec<Dumped>> {
    let manifest = std::fs::read_to_string(dir.join("manifest.csv"))
        .with_context(|| format!("no manifest.csv in {}", dir.display()))?;
    let mut out = Vec::new();
    for line in manifest.lines().skip(1).filter(|l| !l.is_empty()) {
        let mut parts = line.split(',');
        let (Some(name), Some(dt), Some(shape)) = (parts.next(), parts.next(), parts.next()) else {
            bail!("malformed manifest line {line:?}");
        };
        check_tensor_name(name)?;
        let dt: i32 = dt.parse()?;
        let elem_size = dtype_size(dt).with_context(|| format!("tensor {name}"))?;
        let shape = shape
            .split('x')
            .map(str::parse)
            .collect::<Result<Vec<i32>, _>>()?;
        if let Some(dim) = shape.iter().find(|d| **d < 0) {
            bail!("tensor {name}: negative dimension {dim} in shape {shape:?}");
        }
        let expected = shape
            .iter()
            .try_fold(elem_size, |acc, &d| acc.checked_mul(d as usize))
            .with_context(|| format!("tensor {name}: shape {shape:?} overflows"))?;
        let bytes = std::fs::read(dir.join(format!("{name}.bin")))?;
        if bytes.len() != expected {
            bail!(
                "tensor {name}: shape {shape:?} needs {expected} bytes, {name}.bin holds {}",
                bytes.len()
            );
        }
        out.push(Dumped {
            name: name.to_string(),
            dtype: dt,
            shape,
            bytes,
        });
    }
    Ok(out)
}

/// Look a tensor up by name in an already read dump.
pub fn find<'a>(dump: &'a [Dumped], name: &str) -> Result<&'a Dumped> {
    dump.iter()
        .find(|d| d.name == name)
        .with_context(|| format!("{name} is not in the dump"))
}

impl Dumped {
    pub fn to_array(&self) -> mlxcel_core::UniquePtr<MlxArray> {
        mlxcel_core::from_bytes(&self.bytes, &self.shape, self.dtype)
    }
}

/// Values and order keys of a bf16 / f16 / f32 buffer. The key is monotone in
/// the value, so `|key_a - key_b|` is the distance in units in the last place.
pub fn decode(dt: i32, bytes: &[u8]) -> Result<(Vec<f32>, Vec<i64>)> {
    match dt {
        dtype::BFLOAT16 | dtype::FLOAT16 => {
            let bits: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let vals = bits
                .iter()
                .map(|&b| {
                    if dt == dtype::BFLOAT16 {
                        f32::from_bits(u32::from(b) << 16)
                    } else {
                        half_to_f32(b)
                    }
                })
                .collect();
            let keys = bits.iter().map(|&b| order_key(u64::from(b), 15)).collect();
            Ok((vals, keys))
        }
        dtype::FLOAT32 => {
            let bits: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let vals = bits.iter().map(|&b| f32::from_bits(b)).collect();
            let keys = bits.iter().map(|&b| order_key(u64::from(b), 31)).collect();
            Ok((vals, keys))
        }
        other => bail!("dtype {other} is not decoded by this probe"),
    }
}

fn order_key(bits: u64, sign_bit: u32) -> i64 {
    let mag = (bits & ((1u64 << sign_bit) - 1)) as i64;
    if (bits >> sign_bit) & 1 == 1 {
        -mag
    } else {
        mag
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Element-wise comparison of two equally shaped buffers.
#[derive(Debug, Default, Clone, Copy)]
pub struct Deviation {
    pub elements: usize,
    pub differing: usize,
    pub max_ulp: i64,
    pub max_abs: f64,
    pub rms: f64,
    /// RMS of the reference, for reading `max_abs` and `rms` relatively.
    pub ref_rms: f64,
}

impl Deviation {
    pub fn between(dt: i32, a: &[u8], b: &[u8]) -> Result<Self> {
        let (va, ka) = decode(dt, a)?;
        let (vb, kb) = decode(dt, b)?;
        if va.len() != vb.len() {
            bail!("length mismatch: {} vs {}", va.len(), vb.len());
        }
        let mut dev = Self {
            elements: va.len(),
            ..Self::default()
        };
        let (mut sq, mut ref_sq) = (0f64, 0f64);
        for i in 0..va.len() {
            let d = f64::from(va[i]) - f64::from(vb[i]);
            if ka[i] != kb[i] {
                dev.differing += 1;
                dev.max_ulp = dev.max_ulp.max((ka[i] - kb[i]).abs());
            }
            dev.max_abs = dev.max_abs.max(d.abs());
            sq += d * d;
            ref_sq += f64::from(vb[i]) * f64::from(vb[i]);
        }
        let n = va.len().max(1) as f64;
        dev.rms = (sq / n).sqrt();
        dev.ref_rms = (ref_sq / n).sqrt();
        Ok(dev)
    }

    pub fn summary(&self) -> String {
        format!(
            "differing={}/{} max_ulp={} max_abs={:.6e} rms={:.6e} ref_rms={:.6e}",
            self.differing, self.elements, self.max_ulp, self.max_abs, self.rms, self.ref_rms
        )
    }
}

/// Diff two dumps tensor by tensor and print every tensor, marking the first
/// that is not byte-identical.
pub fn compare_dumps(a_dir: &Path, b_dir: &Path) -> Result<()> {
    let a = read_dump(a_dir)?;
    let b = read_dump(b_dir)?;
    if a.len() != b.len() {
        bail!("the dumps hold {} and {} tensors", a.len(), b.len());
    }
    let mut first = None;
    for (ta, tb) in a.iter().zip(&b) {
        if ta.name != tb.name || ta.dtype != tb.dtype || ta.shape != tb.shape {
            bail!("tensor {} does not line up with {}", ta.name, tb.name);
        }
        let identical = ta.bytes == tb.bytes;
        let dev = Deviation::between(ta.dtype, &ta.bytes, &tb.bytes)?;
        if !identical && first.is_none() {
            first = Some(ta.name.clone());
        }
        let mark = if identical { "same" } else { "DIFF" };
        println!("[compare] {:<16} {mark} {}", ta.name, dev.summary());
    }
    match first {
        Some(name) => println!("[compare] first divergent tensor: {name}"),
        None => println!("[compare] the dumps are byte-identical"),
    }
    Ok(())
}
