// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Single-layer SSM mixer probe runner (issue #3337).
//!
//! Emits one Mamba2, Mamba1, or Falcon-Mamba mixer layer
//! ([`crate::emitter::emit_ssm_probe_with`]), compiles it for a device, and runs
//! it through the auxiliary IREE ABI with the mixer weights resident and the
//! activations and state as inputs. The `xla_ssm_probe` example feeds it the
//! inputs `spike/openxla/ssm_oracle.py` wrote and hands the outputs back to the
//! oracle for comparison.

#[cfg(feature = "iree")]
use std::process::Command;
#[cfg(feature = "iree")]
use std::time::Instant;

#[cfg(feature = "iree")]
use crate::aux_manifest::{AuxiliaryArtifactContract, write_auxiliary_manifest};
#[cfg(feature = "iree")]
use crate::auxiliary::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    AuxiliaryWeightStorage, IreeAuxiliaryModule,
};
use crate::emitter::{
    Precision, ProbeMode, SsmConfig, SsmKind, SsmPrefillForm, emit_ssm_probe_with,
    precision_env_override,
};
#[cfg(feature = "iree")]
use crate::iree::{compile_one, iree_compile_bin, target_flags};

/// The probe's dimensions and mode, as `ssm_oracle.py` records them.
#[derive(Clone, Debug, PartialEq)]
pub struct SsmProbeSpec {
    /// `mamba2` or `mamba1`.
    pub kind: String,
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub state_size: usize,
    pub n_groups: usize,
    pub conv_kernel: usize,
    pub use_conv_bias: bool,
    pub use_bias: bool,
    pub time_step_limit: (f32, f32),
    pub norm_eps: f32,
    pub norm_group_size: usize,
    pub chunk_size: usize,
    /// Mamba1 low-rank dt projection width (ignored by Mamba2).
    pub dt_rank: usize,
    /// Falcon-Mamba's weightless RMSNorm on dt, B, and C.
    pub bcdt_norm: bool,
    /// Prefill: the padded capacity `T`. Decode: the batch rows `B`.
    pub rows: usize,
    /// `true` for one decode step, `false` for a padded prefill.
    pub decode: bool,
    /// `chunked`, `associative`, or `sequential` (prefill only).
    pub form: String,
}

impl SsmProbeSpec {
    fn config(&self) -> Result<SsmConfig, String> {
        // `meta.json` is read from disk: reject what would panic the emitter
        // (zero rows divide the chunk count by zero) or wrap a width.
        if self.hidden == 0 || self.rows == 0 {
            return Err(format!(
                "SSM probe needs positive hidden and rows, got hidden={} rows={}",
                self.hidden, self.rows
            ));
        }
        if i32::try_from(self.rows).is_err() {
            return Err(format!("SSM probe rows={} do not fit i32", self.rows));
        }
        let d_inner = self
            .n_heads
            .checked_mul(self.head_dim)
            .ok_or("SSM probe d_inner overflows usize")?;
        let (kind, conv_dim) = match self.kind.as_str() {
            "mamba2" => (
                SsmKind::Mamba2,
                self.n_groups
                    .checked_mul(self.state_size)
                    .and_then(|gn| gn.checked_mul(2))
                    .and_then(|bc| bc.checked_add(d_inner))
                    .ok_or("SSM probe conv_dim overflows usize")?,
            ),
            "mamba1" => (
                SsmKind::Mamba1 {
                    dt_rank: self.dt_rank,
                    bcdt_norm: self.bcdt_norm,
                },
                d_inner,
            ),
            other => return Err(format!("unknown SSM probe kind {other:?}")),
        };
        let config = SsmConfig {
            kind,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            state_size: self.state_size,
            n_groups: self.n_groups,
            conv_kernel: self.conv_kernel,
            conv_dim,
            use_conv_bias: self.use_conv_bias,
            use_bias: self.use_bias,
            time_step_limit: self.time_step_limit,
            norm_eps: self.norm_eps,
            norm_group_size: self.norm_group_size,
            chunk_size: self.chunk_size,
        };
        config.validate()?;
        Ok(config)
    }

    /// The resident weight names and shapes, in probe argument order.
    pub fn weight_layout(&self) -> Result<Vec<(&'static str, Vec<usize>)>, String> {
        let c = self.config()?;
        let (cd, h) = (c.conv_dim, c.n_heads);
        let mut layout = vec![("in_proj", vec![c.in_proj_dim(), self.hidden])];
        if c.use_bias {
            layout.push(("in_proj_b", vec![c.in_proj_dim()]));
        }
        layout.push(("conv_w", vec![cd, 1, c.conv_kernel]));
        if c.use_conv_bias {
            layout.push(("conv_b", vec![cd]));
        }
        match c.kind {
            SsmKind::Mamba2 => layout.extend([
                ("dt_bias", vec![h]),
                ("a_log", vec![h]),
                ("d", vec![h]),
                ("norm_w", vec![c.d_inner()]),
            ]),
            SsmKind::Mamba1 { dt_rank, .. } => layout.extend([
                ("x_proj", vec![dt_rank + 2 * c.state_size, c.d_inner()]),
                ("dt_proj", vec![c.d_inner(), dt_rank]),
                ("dt_proj_b", vec![c.d_inner()]),
                ("a_log", vec![c.d_inner(), c.state_size]),
                ("d", vec![c.d_inner()]),
            ]),
            SsmKind::Plamo2 { dt_rank } => layout.extend([
                ("bcdt_proj", vec![2 * c.state_size + dt_rank, c.d_inner()]),
                ("dt_proj", vec![h, dt_rank]),
                ("dt_norm", vec![dt_rank]),
                ("b_norm", vec![c.state_size]),
                ("c_norm", vec![c.state_size]),
                ("a_log", vec![h]),
                ("d", vec![h]),
                ("dt_bias", vec![h]),
            ]),
        }
        layout.push(("out_proj", vec![self.hidden, c.d_inner()]));
        if c.use_bias {
            layout.push(("out_proj_b", vec![self.hidden]));
        }
        Ok(layout)
    }

    /// Shapes of `(x, h0, conv0)`; the outputs `(y, h_T, conv_tail)` match them.
    pub fn io_shapes(&self) -> Result<[Vec<usize>; 3], String> {
        let c = self.config()?;
        let (h, p, n) = (c.n_heads, c.head_dim, c.state_size);
        let tail = c.conv_kernel - 1;
        let h_shape = match (c.kind, self.decode) {
            (SsmKind::Mamba2 | SsmKind::Plamo2 { .. }, true) => vec![self.rows, h, p, n],
            (SsmKind::Mamba2 | SsmKind::Plamo2 { .. }, false) => vec![h, p, n],
            (SsmKind::Mamba1 { .. }, true) => vec![self.rows, c.d_inner(), n],
            (SsmKind::Mamba1 { .. }, false) => vec![c.d_inner(), n],
        };
        let conv_shape = if self.decode {
            vec![self.rows, tail, c.conv_dim]
        } else {
            vec![tail, c.conv_dim]
        };
        Ok([vec![self.rows, self.hidden], h_shape, conv_shape])
    }

    /// The probe's StableHLO text. f32 contractions (the HF fp32 gate) unless
    /// `MLXCEL_XLA_PRECISION` asks for the f16 / bf16 projection path.
    pub fn emit(&self) -> Result<String, String> {
        let c = self.config()?;
        let mode = if self.decode {
            ProbeMode::Decode
        } else {
            ProbeMode::Prefill
        };
        let form = SsmPrefillForm::parse(&self.form)?;
        let precision = precision_env_override().unwrap_or(Precision::F32);
        Ok(emit_ssm_probe_with(
            &c,
            self.hidden,
            self.rows,
            mode,
            form,
            precision,
        ))
    }
}

/// The probe's activations and state, flat row-major f32.
#[derive(Clone, Debug, Default)]
pub struct SsmProbeInputs {
    /// One entry per [`SsmProbeSpec::weight_layout`] row, in order.
    pub weights: Vec<Vec<f32>>,
    pub x: Vec<f32>,
    /// Prefill only.
    pub real_len: i32,
    pub h0: Vec<f32>,
    pub conv0: Vec<f32>,
}

/// The probe results, flat row-major f32.
#[derive(Clone, Debug, Default)]
pub struct SsmProbeOutputs {
    pub y: Vec<f32>,
    pub h: Vec<f32>,
    pub conv: Vec<f32>,
    pub compile_seconds: f64,
    pub run_seconds: f64,
}

#[cfg(feature = "iree")]
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

#[cfg(feature = "iree")]
fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(feature = "iree")]
fn compiler_version(compiler: &std::path::Path) -> Result<String, String> {
    let out = Command::new(compiler)
        .arg("--version")
        .output()
        .map_err(|error| format!("run {} --version: {error}", compiler.display()))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Element count of `shape`, or an error when it overflows `usize`.
fn element_count(label: &str, shape: &[usize]) -> Result<usize, String> {
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| format!("{label} shape {shape:?} overflows usize"))
}

fn check_len(label: &str, values: &[f32], shape: &[usize]) -> Result<(), String> {
    let expected = element_count(label, shape)?;
    if values.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "{label} has {} values, expected {expected} for {shape:?}",
            values.len()
        ))
    }
}

/// Parse a little-endian `<f4` or `<i4` C-order `.npy` image (format 1.x or
/// 2.x) into f32 values and the declared shape. `ssm_oracle.py` writes these;
/// every length is bounds-checked so a truncated or hostile file is an error,
/// never a panic or a short buffer.
pub fn parse_npy_f32(bytes: &[u8]) -> Result<(Vec<f32>, Vec<usize>), String> {
    if bytes.get(..6) != Some(b"\x93NUMPY".as_slice()) {
        return Err("not a .npy file".to_string());
    }
    let short = || "truncated .npy header".to_string();
    let (header_len, start): (usize, usize) = match bytes.get(6).copied().ok_or_else(short)? {
        1 => {
            let raw = bytes.get(8..10).ok_or_else(short)?;
            (usize::from(u16::from_le_bytes([raw[0], raw[1]])), 10)
        }
        2 | 3 => {
            let raw = bytes.get(8..12).ok_or_else(short)?;
            let len = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            (
                usize::try_from(len).map_err(|_| "npy header too long".to_string())?,
                12,
            )
        }
        version => return Err(format!("unsupported .npy version {version}")),
    };
    let end = start.checked_add(header_len).ok_or_else(short)?;
    let header = bytes.get(start..end).ok_or_else(short)?;
    let header = std::str::from_utf8(header).map_err(|_| "npy header is not UTF-8".to_string())?;
    if !header.contains("'fortran_order': False") {
        return Err("only C-order .npy files are supported".to_string());
    }
    let int = header.contains("'descr': '<i4'");
    if !int && !header.contains("'descr': '<f4'") {
        return Err("only <f4 and <i4 .npy files are supported".to_string());
    }
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or("no shape in the .npy header")?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| format!("bad .npy dim {part:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = element_count("npy", &shape)?;
    let data = &bytes[end..];
    if count.checked_mul(4) != Some(data.len()) {
        return Err(format!(
            ".npy data is {} bytes, expected {count} x 4 for {shape:?}",
            data.len()
        ));
    }
    let values = data
        .chunks_exact(4)
        .map(|c| {
            let raw = [c[0], c[1], c[2], c[3]];
            if int {
                i32::from_le_bytes(raw) as f32
            } else {
                f32::from_le_bytes(raw)
            }
        })
        .collect();
    Ok((values, shape))
}

/// Serialize f32 values as a format 1.0 little-endian C-order `.npy` image.
pub fn npy_f32_bytes(values: &[f32], shape: &[usize]) -> Result<Vec<u8>, String> {
    if element_count("npy", shape)? != values.len() {
        return Err(format!(
            "{} values do not fill .npy shape {shape:?}",
            values.len()
        ));
    }
    let dims = match shape.len() {
        0 => String::new(),
        1 => format!("{},", shape[0]),
        _ => shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", "),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let header_len = u16::try_from(header.len()).map_err(|_| "npy header too long".to_string())?;
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    Ok(bytes)
}

#[cfg(feature = "iree")]
/// Compile the probe for `device` and run it once on `inputs`.
pub fn run_ssm_probe(
    device: &str,
    spec: &SsmProbeSpec,
    inputs: &SsmProbeInputs,
) -> Result<SsmProbeOutputs, String> {
    let layout = spec.weight_layout()?;
    if layout.len() != inputs.weights.len() {
        return Err(format!(
            "probe takes {} weights, got {}",
            layout.len(),
            inputs.weights.len()
        ));
    }
    let [x_shape, h_shape, conv_shape] = spec.io_shapes()?;
    check_len("x", &inputs.x, &x_shape)?;
    check_len("h0", &inputs.h0, &h_shape)?;
    check_len("conv0", &inputs.conv0, &conv_shape)?;
    let mut weights = Vec::with_capacity(layout.len());
    for ((name, shape), values) in layout.iter().zip(&inputs.weights) {
        check_len(name, values, shape)?;
        weights.push(AuxiliaryWeight {
            name: format!("ssm.{name}"),
            storage: AuxiliaryWeightStorage::Float32(values.clone()),
            dtype: AuxiliaryWeightDType::Float32,
            shape: shape.clone(),
        });
    }

    let mlir = spec.emit()?;
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    let cache = crate::cache_dir::xla_cache_dir(crate::cache_dir::XlaCache::SsmProbe)?;
    let started = Instant::now();
    let vmfb = compile_one(&compiler, &mlir, &flags, &cache, "ssm-probe", spec.rows)?;
    let compile_seconds = started.elapsed().as_secs_f64();
    let contract = AuxiliaryArtifactContract::new_legacy_unqualified(
        "ssm_probe.main",
        format!("{spec:?}"),
        format!(
            "compiler={};version={};flags={flags:?}",
            compiler.display(),
            compiler_version(&compiler)?
        ),
    )?;
    write_auxiliary_manifest(&vmfb, &contract, &weights)?;
    let mut module = IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)?;

    let x = f32_bytes(&inputs.x);
    let real_len = inputs.real_len.to_ne_bytes();
    let h0 = f32_bytes(&inputs.h0);
    let conv0 = f32_bytes(&inputs.conv0);
    let mut args = vec![AuxiliaryInput {
        bytes: &x,
        dtype: AuxiliaryTensorDType::Float32,
        shape: &x_shape,
    }];
    if !spec.decode {
        args.push(AuxiliaryInput {
            bytes: &real_len,
            dtype: AuxiliaryTensorDType::Int32,
            shape: &[],
        });
    }
    args.push(AuxiliaryInput {
        bytes: &h0,
        dtype: AuxiliaryTensorDType::Float32,
        shape: &h_shape,
    });
    args.push(AuxiliaryInput {
        bytes: &conv0,
        dtype: AuxiliaryTensorDType::Float32,
        shape: &conv_shape,
    });
    let mut y = vec![0u8; x.len()];
    let mut h = vec![0u8; h0.len()];
    let mut conv = vec![0u8; conv0.len()];
    let started = Instant::now();
    module.invoke(
        &args,
        &mut [
            AuxiliaryOutput {
                bytes: &mut y,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &x_shape,
            },
            AuxiliaryOutput {
                bytes: &mut h,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &h_shape,
            },
            AuxiliaryOutput {
                bytes: &mut conv,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &conv_shape,
            },
        ],
    )?;
    Ok(SsmProbeOutputs {
        y: decode_f32(&y),
        h: decode_f32(&h),
        conv: decode_f32(&conv),
        compile_seconds,
        run_seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_spec() -> SsmProbeSpec {
        SsmProbeSpec {
            kind: "mamba2".to_string(),
            hidden: 16,
            n_heads: 4,
            head_dim: 8,
            state_size: 16,
            n_groups: 2,
            conv_kernel: 4,
            use_conv_bias: true,
            use_bias: false,
            time_step_limit: (0.0, f32::INFINITY),
            norm_eps: 1e-5,
            norm_group_size: 32,
            chunk_size: 8,
            dt_rank: 0,
            bcdt_norm: false,
            rows: 16,
            decode: false,
            form: "chunked".to_string(),
        }
    }

    #[test]
    fn spec_rejects_zero_rows_and_hidden_before_emitting() {
        tiny_spec().emit().expect("tiny spec emits");
        for spec in [
            SsmProbeSpec {
                rows: 0,
                ..tiny_spec()
            },
            SsmProbeSpec {
                hidden: 0,
                ..tiny_spec()
            },
            SsmProbeSpec {
                rows: 0,
                decode: true,
                ..tiny_spec()
            },
        ] {
            let error = spec.emit().expect_err("zero rows/hidden must be an error");
            assert!(error.contains("positive"), "{error}");
        }
    }

    #[test]
    fn spec_rejects_wrapping_widths() {
        let spec = SsmProbeSpec {
            n_heads: usize::MAX / 2 + 1,
            n_groups: 1,
            ..tiny_spec()
        };
        assert!(spec.weight_layout().unwrap_err().contains("overflow"));
        let spec = SsmProbeSpec {
            rows: usize::from(u16::MAX) << 32,
            ..tiny_spec()
        };
        assert!(spec.io_shapes().unwrap_err().contains("i32"));
    }

    #[test]
    fn check_len_rejects_a_wrapping_shape() {
        // 2^63 * 2 wraps to 0 in release arithmetic; an empty buffer must not
        // pass as a match for it.
        let error = check_len("x", &[], &[1usize << 63, 2]).unwrap_err();
        assert!(error.contains("overflow"), "{error}");
        check_len("x", &[0.0; 6], &[2, 3]).expect("exact length passes");
        assert!(check_len("x", &[0.0; 5], &[2, 3]).is_err());
    }

    #[test]
    fn npy_round_trips_and_rejects_bad_images() {
        let bytes = npy_f32_bytes(&[1.0, -2.0, 3.5, 0.25, 8.0, 9.0], &[2, 3]).unwrap();
        let (values, shape) = parse_npy_f32(&bytes).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(values, vec![1.0, -2.0, 3.5, 0.25, 8.0, 9.0]);
        assert!(npy_f32_bytes(&[1.0], &[2]).is_err());

        // Truncated at every prefix length: an error, never a panic.
        for cut in 0..bytes.len() {
            assert!(parse_npy_f32(&bytes[..cut]).is_err(), "prefix {cut}");
        }
        // Version 2 magic with no room for the 4-byte header length.
        assert!(parse_npy_f32(b"\x93NUMPY\x02\x00\x10\x00").is_err());
        // A header length past the end of the file.
        assert!(parse_npy_f32(b"\x93NUMPY\x01\x00\xff\xff{}").is_err());
        // Trailing data beyond the declared shape.
        let mut long = bytes.clone();
        long.extend_from_slice(&[0; 4]);
        assert!(parse_npy_f32(&long).is_err());
        // A declared shape whose element count wraps.
        let mut header =
            "{'descr': '<f4', 'fortran_order': False, 'shape': (9223372036854775808, 2), }"
                .to_string();
        header.push('\n');
        let mut wrapped = b"\x93NUMPY\x01\x00".to_vec();
        wrapped.extend_from_slice(&(header.len() as u16).to_le_bytes());
        wrapped.extend_from_slice(header.as_bytes());
        assert!(parse_npy_f32(&wrapped).unwrap_err().contains("overflow"));
    }
}
