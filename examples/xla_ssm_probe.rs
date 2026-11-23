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

//! Single-layer Mamba2 mixer execution check (issue #3337).
//!
//! Reads the inputs `spike/openxla/ssm_oracle.py --emit-inputs DIR` wrote,
//! compiles the emitter's SSM probe for `--device`, runs it through the IREE
//! shim, and writes `y.npy`, `h.npy`, `conv.npy` and `run.json` to `--out` for
//! `ssm_oracle.py --check`. `--emit-only FILE` writes the StableHLO instead of
//! running it (the lowering measurements compile that file directly).
//!
//! ```text
//! cargo run --release --features xla-iree --example xla_ssm_probe -- \
//!     --inputs DIR --device hip --out DIR [--form chunked] [--chunk-size 128]
//! ```

use std::path::{Path, PathBuf};

use mlxcel_xla::{SsmProbeInputs, SsmProbeSpec, run_ssm_probe};
use serde_json::Value;

struct Args {
    inputs: PathBuf,
    device: String,
    out: Option<PathBuf>,
    form: Option<String>,
    chunk_size: Option<usize>,
    emit_only: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        inputs: PathBuf::new(),
        device: "local-task".to_string(),
        out: None,
        form: None,
        chunk_size: None,
        emit_only: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--inputs" => args.inputs = PathBuf::from(value()?),
            "--device" => args.device = value()?,
            "--out" => args.out = Some(PathBuf::from(value()?)),
            "--form" => args.form = Some(value()?),
            "--chunk-size" => {
                let text = value()?;
                args.chunk_size = Some(
                    text.parse()
                        .map_err(|_| format!("bad --chunk-size {text}"))?,
                );
            }
            "--emit-only" => args.emit_only = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.inputs.as_os_str().is_empty() {
        return Err("--inputs DIR is required".to_string());
    }
    if args.out.is_none() && args.emit_only.is_none() {
        return Err("--out DIR (or --emit-only FILE) is required".to_string());
    }
    Ok(args)
}

/// Read a little-endian f4 / i4 C-order `.npy` file as f32 values and shape.
fn read_npy(path: &Path) -> Result<(Vec<f32>, Vec<usize>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a .npy file", path.display()));
    }
    let (header_len, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        _ => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len])
        .map_err(|_| format!("{}: header is not UTF-8", path.display()))?;
    if header.contains("'fortran_order': True") {
        return Err(format!("{}: Fortran order is unsupported", path.display()));
    }
    let int = header.contains("'<i4'");
    if !int && !header.contains("'<f4'") {
        return Err(format!(
            "{}: only <f4 and <i4 are supported",
            path.display()
        ));
    }
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .ok_or_else(|| format!("{}: no shape in header", path.display()))?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>().map_err(|_| format!("bad dim {part}")))
        .collect::<Result<Vec<_>, _>>()?;
    let data = &bytes[start + header_len..];
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
        .collect::<Vec<_>>();
    if values.len() != shape.iter().product::<usize>() {
        return Err(format!("{}: data does not match shape", path.display()));
    }
    Ok((values, shape))
}

fn write_npy(path: &Path, values: &[f32], shape: &[usize]) -> Result<(), String> {
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
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn field<'a>(meta: &'a Value, key: &str) -> Result<&'a Value, String> {
    meta.get(key)
        .ok_or_else(|| format!("meta.json has no {key}"))
}

fn usize_field(meta: &Value, key: &str) -> Result<usize, String> {
    field(meta, key)?
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("meta.json {key} is not an integer"))
}

fn f32_field(value: &Value) -> f32 {
    value.as_f64().map_or(f32::INFINITY, |v| v as f32)
}

fn spec_from_meta(meta: &Value, args: &Args) -> Result<SsmProbeSpec, String> {
    let limit = field(meta, "time_step_limit")?
        .as_array()
        .ok_or("time_step_limit is not a pair")?;
    let flag = |key: &str| field(meta, key).map(|v| v.as_bool().unwrap_or(false));
    Ok(SsmProbeSpec {
        hidden: usize_field(meta, "hidden")?,
        n_heads: usize_field(meta, "n_heads")?,
        head_dim: usize_field(meta, "head_dim")?,
        state_size: usize_field(meta, "state_size")?,
        n_groups: usize_field(meta, "n_groups")?,
        conv_kernel: usize_field(meta, "conv_kernel")?,
        use_conv_bias: flag("use_conv_bias")?,
        use_bias: flag("use_bias")?,
        time_step_limit: (f32_field(&limit[0]), f32_field(&limit[1])),
        norm_eps: f32_field(field(meta, "norm_eps")?),
        norm_group_size: usize_field(meta, "norm_group_size")?,
        chunk_size: match args.chunk_size {
            Some(size) => size,
            None => usize_field(meta, "chunk_size")?,
        },
        rows: usize_field(meta, "rows")?,
        decode: field(meta, "mode")?.as_str() == Some("decode"),
        form: args.form.clone().unwrap_or_else(|| "chunked".to_string()),
    })
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let meta_path = args.inputs.join("meta.json");
    let meta: Value = serde_json::from_str(
        &std::fs::read_to_string(&meta_path)
            .map_err(|e| format!("read {}: {e}", meta_path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", meta_path.display()))?;
    let spec = spec_from_meta(&meta, &args)?;
    if let Some(path) = &args.emit_only {
        return std::fs::write(path, spec.emit()?)
            .map_err(|e| format!("write {}: {e}", path.display()));
    }
    let out = args.out.clone().expect("checked in parse_args");

    let load = |name: &str| read_npy(&args.inputs.join(format!("{name}.npy"))).map(|(v, _)| v);
    let weights = spec
        .weight_layout()?
        .iter()
        .map(|(name, _)| load(name))
        .collect::<Result<Vec<_>, _>>()?;
    let real_len = if spec.decode {
        0
    } else {
        usize_field(&meta, "real_len")? as i32
    };
    let inputs = SsmProbeInputs {
        weights,
        x: load("x")?,
        real_len,
        h0: load("h0")?,
        conv0: load("conv0")?,
    };
    let result = run_ssm_probe(&args.device, &spec, &inputs)?;

    std::fs::create_dir_all(&out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let [x_shape, h_shape, conv_shape] = spec.io_shapes()?;
    write_npy(&out.join("y.npy"), &result.y, &x_shape)?;
    write_npy(&out.join("h.npy"), &result.h, &h_shape)?;
    write_npy(&out.join("conv.npy"), &result.conv, &conv_shape)?;
    let report = serde_json::json!({
        "inputs": args.inputs.canonicalize().unwrap_or(args.inputs.clone()),
        "device": args.device,
        "form": spec.form,
        "chunk_size": spec.chunk_size,
        "rows": spec.rows,
        "compile_seconds": result.compile_seconds,
        "run_seconds": result.run_seconds,
    });
    std::fs::write(out.join("run.json"), report.to_string())
        .map_err(|e| format!("write run.json: {e}"))?;
    println!(
        "[xla_ssm_probe] device={} form={} rows={} compile={:.2}s run={:.3}s -> {}",
        args.device,
        spec.form,
        spec.rows,
        result.compile_seconds,
        result.run_seconds,
        out.display()
    );
    Ok(())
}
