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

//! Oracle dump for the standalone cuda-oxide fused residual-add + RMSNorm
//! kernel (`spike/cuda-oxide-fused-norm`, issue #2832).
//!
//! The standalone harness writes its deterministic fixtures (inputs already
//! rounded to the case dtypes, stored as f32) and its device outputs under a
//! directory. This example replays every case through the two references the
//! issue names, the existing fused CUDA kernel (`fused_add_rms_norm`) and the
//! graph path (`add` + `fast_rms_norm`, via `layers::graph_add_rms_norm`),
//! and writes their raw output bits next to a `meta.tsv`. The harness's
//! `compare` subcommand then scores the standalone kernel against both.
//!
//!     cargo run --profile test-fast -p mlxcel-core --features cuda \
//!         --example oxide_fused_norm_oracle -- --fixtures DIR --out DIR
//!
//! The oracle is the backend the build selects; the meta file records which.
//! On a build without a fused port, or with the default device on the CPU,
//! the example fails rather than writing a graph-only dump, so a comparison
//! cannot pass against a reference that never ran the kernel.
//! Numerical and I/O failures exit with status 1; invalid arguments exit 2.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mlxcel_core::layers::{GemmaRMSNorm, RMSNorm, graph_add_rms_norm};
use mlxcel_core::{
    MlxArray, UniquePtr, array_to_raw_bytes, astype, copy, dtype, eval, from_slice_f32,
    fused_add_rms_norm, fused_add_rms_norm_available, hardware,
};

const CASES_TSV_HEADER: &str = "name\tact\tw\trows\tdim\teps\tweight_bias\tpattern\tthreads\tgrid";

struct Case {
    name: String,
    act: i32,
    w: i32,
    rows: i32,
    dim: i32,
    eps: f32,
    weight_bias: f32,
}

fn parse_dtype(s: &str) -> Option<i32> {
    match s {
        "f32" => Some(dtype::FLOAT32),
        "f16" => Some(dtype::FLOAT16),
        "bf16" => Some(dtype::BFLOAT16),
        _ => None,
    }
}

fn read_cases(path: &Path) -> Result<Vec<Case>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut cases = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            if line != CASES_TSV_HEADER {
                return Err(format!("{}: unexpected header {line:?}", path.display()));
            }
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 10 {
            return Err(format!(
                "{}: line {}: expected 10 fields",
                path.display(),
                i + 1
            ));
        }
        let bad = |what: &str| format!("{}: line {}: bad {what}", path.display(), i + 1);
        cases.push(Case {
            name: f[0].to_string(),
            act: parse_dtype(f[1]).ok_or_else(|| bad("act"))?,
            w: parse_dtype(f[2]).ok_or_else(|| bad("w"))?,
            rows: f[3].parse().map_err(|_| bad("rows"))?,
            dim: f[4].parse().map_err(|_| bad("dim"))?,
            eps: f[5].parse().map_err(|_| bad("eps"))?,
            weight_bias: f[6].parse().map_err(|_| bad("weight_bias"))?,
        });
    }
    Ok(cases)
}

fn read_f32(path: &Path, expect: usize) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() != expect * 4 {
        return Err(format!(
            "{}: {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expect * 4
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// An f32 fixture whose values are exactly representable in `dt`, cast to
/// `dt`, so the array holds the very bits the standalone kernel read.
fn fixture_array(values: &[f32], shape: &[i32], dt: i32) -> UniquePtr<MlxArray> {
    let f = from_slice_f32(values, shape);
    let a = astype(&f, dt);
    eval(&a);
    a
}

fn write_array(path: &Path, arr: &MlxArray, expect_elements: usize, dt: i32) -> Result<(), String> {
    eval(arr);
    let bytes = array_to_raw_bytes(arr);
    let width = if dt == dtype::FLOAT32 { 4 } else { 2 };
    if bytes.len() != expect_elements * width {
        return Err(format!(
            "{}: array has {} bytes, expected {} ({} elements x {} bytes)",
            path.display(),
            bytes.len(),
            expect_elements * width,
            expect_elements,
            width
        ));
    }
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn run(fixtures: &Path, out: &Path) -> Result<(), String> {
    let backend = hardware::gpu_backend_kind();
    if !fused_add_rms_norm_available() {
        return Err(format!(
            "fused_add_rms_norm is unavailable on this build (backend {backend:?}); refusing to write a graph-only oracle"
        ));
    }
    let cases = read_cases(&fixtures.join("cases.tsv"))?;
    fs::create_dir_all(out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;

    let mut elements = 0usize;
    for case in &cases {
        let n = (case.rows as usize) * (case.dim as usize);
        let dir = fixtures.join(&case.name);
        let x = read_f32(&dir.join("x.f32"), n)?;
        let residual = read_f32(&dir.join("residual.f32"), n)?;
        let weight = read_f32(&dir.join("weight.f32"), case.dim as usize)?;

        let delta = fixture_array(&x, &[case.rows, case.dim], case.act);
        let residual = fixture_array(&residual, &[case.rows, case.dim], case.act);
        let weight = fixture_array(&weight, &[case.dim], case.w);

        // The existing fused kernel, called directly: the comparison is
        // always against the kernel, never against a kill-switched fallback.
        let mut fused_normed = UniquePtr::null();
        let mut fused_residual = UniquePtr::null();
        fused_add_rms_norm(
            &delta,
            &residual,
            &weight,
            case.eps,
            case.weight_bias,
            &mut fused_normed,
            &mut fused_residual,
        )
        .map_err(|e| format!("{}: fused_add_rms_norm refused: {e}", case.name))?;

        // The graph path the blocks used before the fused kernel. The Gemma
        // `(1 + w)` convention is a layer, not a scalar, on this side.
        let (graph_normed, graph_residual) = if case.weight_bias == 0.0 {
            let norm = RMSNorm::new(copy(&weight), case.eps);
            graph_add_rms_norm(&norm, &delta, &residual)
        } else if case.weight_bias == 1.0 {
            let norm = GemmaRMSNorm::new(copy(&weight), case.eps);
            graph_add_rms_norm(&norm, &delta, &residual)
        } else {
            return Err(format!(
                "{}: weight_bias {} has no graph-path layer equivalent",
                case.name, case.weight_bias
            ));
        };

        // MLX's `rms_norm` promotes to `result_type(x, weight)`, so a 16-bit
        // activation with an f32 weight comes back as f32 on the graph side
        // while the fused kernel stores the activation dtype. Round the graph
        // output to the activation dtype so both references are compared in
        // the dtype the kernel writes; the fused dump is never converted.
        let graph_normed = astype(&graph_normed, case.act);
        let graph_residual = astype(&graph_residual, case.act);

        let odir = out.join(&case.name);
        fs::create_dir_all(&odir).map_err(|e| format!("mkdir {}: {e}", odir.display()))?;
        write_array(&odir.join("fused_normed.bits"), &fused_normed, n, case.act)?;
        write_array(
            &odir.join("fused_new_residual.bits"),
            &fused_residual,
            n,
            case.act,
        )?;
        write_array(&odir.join("graph_normed.bits"), &graph_normed, n, case.act)?;
        write_array(
            &odir.join("graph_new_residual.bits"),
            &graph_residual,
            n,
            case.act,
        )?;
        elements += n;
        println!(
            "ORACLE case={} rows={} dim={} elements={}",
            case.name, case.rows, case.dim, n
        );
    }

    let meta = format!(
        "backend\t{backend:?}\nfused_add_rms_norm_available\ttrue\ncases\t{}\nelements\t{}\nfused_reference\tmlxcel_core::fused_add_rms_norm (kernels/fused_norm.cpp CUDA port)\ngraph_reference\tlayers::graph_add_rms_norm (add + fast_rms_norm; GemmaRMSNorm for weight_bias=1)\n",
        cases.len(),
        elements
    );
    fs::write(out.join("meta.tsv"), meta).map_err(|e| format!("write meta.tsv: {e}"))?;
    println!(
        "RESULT oracle backend={backend:?} cases={} elements={} out={}",
        cases.len(),
        elements,
        out.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut fixtures, mut out) = (None::<PathBuf>, None::<PathBuf>);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--fixtures" => fixtures = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "-h" | "--help" => {
                println!("usage: oxide_fused_norm_oracle --fixtures DIR --out DIR");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unexpected argument: {other}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(fixtures), Some(out)) = (fixtures, out) else {
        eprintln!("usage: oxide_fused_norm_oracle --fixtures DIR --out DIR");
        return ExitCode::from(2);
    };
    match run(&fixtures, &out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("FAIL error={error}");
            ExitCode::FAILURE
        }
    }
}
