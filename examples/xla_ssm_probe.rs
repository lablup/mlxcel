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

//! Single-layer Mamba2/Mamba1/Falcon-Mamba execution check (issues #3337, #3348).
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

use mlxcel_xla::{SsmProbeInputs, SsmProbeSpec, npy_f32_bytes, parse_npy_f32, run_ssm_probe};
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

/// Read a `.npy` file and require the shape the probe declares for it.
fn read_npy(path: &Path, want: &[usize]) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let (values, shape) = parse_npy_f32(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    if shape != want {
        return Err(format!(
            "{} has shape {shape:?}, the probe expects {want:?}",
            path.display()
        ));
    }
    Ok(values)
}

fn write_npy(path: &Path, values: &[f32], shape: &[usize]) -> Result<(), String> {
    std::fs::write(path, npy_f32_bytes(values, shape)?)
        .map_err(|e| format!("write {}: {e}", path.display()))
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

/// A number, or `null` for +inf (JSON has no infinity; the oracle writes the
/// unbounded `time_step_limit` ceiling as `null`).
fn f32_or_inf(value: &Value, key: &str) -> Result<f32, String> {
    match value {
        Value::Null => Ok(f32::INFINITY),
        other => other
            .as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| format!("meta.json {key} is not a number")),
    }
}

fn spec_from_meta(meta: &Value, args: &Args) -> Result<SsmProbeSpec, String> {
    let limit = match field(meta, "time_step_limit")?
        .as_array()
        .map(Vec::as_slice)
    {
        Some([low, high]) => (
            f32_or_inf(low, "time_step_limit[0]")?,
            f32_or_inf(high, "time_step_limit[1]")?,
        ),
        _ => return Err("meta.json time_step_limit is not a pair".to_string()),
    };
    let norm_eps = field(meta, "norm_eps")?
        .as_f64()
        .ok_or("meta.json norm_eps is not a number")? as f32;
    let flag = |key: &str| field(meta, key).map(|v| v.as_bool().unwrap_or(false));
    Ok(SsmProbeSpec {
        kind: meta
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("mamba2")
            .to_string(),
        hidden: usize_field(meta, "hidden")?,
        n_heads: usize_field(meta, "n_heads")?,
        head_dim: usize_field(meta, "head_dim")?,
        state_size: usize_field(meta, "state_size")?,
        n_groups: usize_field(meta, "n_groups")?,
        conv_kernel: usize_field(meta, "conv_kernel")?,
        use_conv_bias: flag("use_conv_bias")?,
        use_bias: flag("use_bias")?,
        time_step_limit: limit,
        norm_eps,
        norm_group_size: usize_field(meta, "norm_group_size")?,
        chunk_size: match args.chunk_size {
            Some(size) => size,
            None => usize_field(meta, "chunk_size")?,
        },
        dt_rank: meta.get("dt_rank").and_then(Value::as_u64).unwrap_or(0) as usize,
        bcdt_norm: meta
            .get("bcdt_norm")
            .and_then(Value::as_bool)
            .unwrap_or(false),
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

    let load =
        |name: &str, shape: &[usize]| read_npy(&args.inputs.join(format!("{name}.npy")), shape);
    let weights = spec
        .weight_layout()?
        .iter()
        .map(|(name, shape)| load(name, shape))
        .collect::<Result<Vec<_>, _>>()?;
    let real_len = if spec.decode {
        0
    } else {
        let real_len = usize_field(&meta, "real_len")?;
        if real_len > spec.rows {
            return Err(format!(
                "meta.json real_len={real_len} exceeds rows={}",
                spec.rows
            ));
        }
        i32::try_from(real_len).map_err(|_| format!("real_len={real_len} does not fit i32"))?
    };
    let [x_shape, h_shape, conv_shape] = spec.io_shapes()?;
    let inputs = SsmProbeInputs {
        weights,
        x: load("x", &x_shape)?,
        real_len,
        h0: load("h0", &h_shape)?,
        conv0: load("conv0", &conv_shape)?,
    };
    let result = run_ssm_probe(&args.device, &spec, &inputs)?;

    std::fs::create_dir_all(&out).map_err(|e| format!("create {}: {e}", out.display()))?;
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
