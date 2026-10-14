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

//! ROCm custom-kernel JIT modules are keyed by their generated source
//! (issue #2149).
//!
//! `fast::hip_kernel` writes each input's dtype, each output's dtype, and
//! whether each input is 0-d into the kernel source, but the ROCm overlay used
//! to cache the compiled module under the kernel name alone (the name covers
//! only `template_args`). A second call that differed only there reused the
//! first module. The probe (`mlxcel_core::rocm_faults::jit_key_probe_array`)
//! launches one kernel name with no template args, so these tests see exactly
//! that case:
//!
//! - f32, then f16, then bf16 inputs: with a name-only key the f16 and bf16
//!   buffers are read through `const float*` and the output is unrelated to
//!   the input;
//! - an f32 output, then an f16 output, at one f32 input: with a name-only
//!   key the f16 buffer is written through `float*`;
//! - a 1-d input, then a 0-d input, with the kernel reading `inp_shape`: with
//!   a name-only key the 0-d launch passes no shape argument to a module that
//!   declares one, so the output pointer is read from the wrong slot. That can
//!   fault the queue, which kills the device for the whole process, so it
//!   runs in a child process (this binary, re-invoked on an ignored test), as
//!   `tests/rocm_gpu_faults.rs` does.
//!
//! All three fail on the name-keyed overlay (checked by reverting the fix) and pass
//! with the source-hash key. Skips on any other backend. Run on a ROCm host
//! with:
//!
//! ```sh
//! cargo test --features rocm --test rocm_custom_kernel_jit_key -- --test-threads=1
//! ```

#![cfg(feature = "rocm")]

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use mlxcel_core::hardware::{GpuBackendKind, gpu_backend_kind};
use mlxcel_core::rocm_faults::jit_key_probe_array;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Set by the parent test on the child it spawns; the child body runs only
/// when it is present, so `--ignored` sweeps do not run it in a shared
/// process.
const CHILD_ENV: &str = "MLXCEL_ROCM_JIT_KEY_CHILD";
const CHILD_TEST: &str = "child_scalar_after_vector";
/// Two cold hiprtc compiles of the probe, with a wide margin; a child that
/// takes longer is hung on a faulted queue.
const CHILD_BUDGET: Duration = Duration::from_secs(120);

/// Exactly representable in f16 and bf16, so every dtype's output must equal
/// these values bit for bit after the kernel's conversion to f32.
const VALUES: [f32; 8] = [0.5, -1.25, 3.0, 8.0, -0.75, 96.0, 0.0, 2.5];

fn on_rocm() -> bool {
    gpu_backend_kind() == GpuBackendKind::Rocm
}

fn input(values: &[f32], shape: &[i32], dtype: i32) -> UniquePtr<MlxArray> {
    let f32_array = mlxcel_core::from_slice_f32(values, shape);
    mlxcel_core::astype(&f32_array, dtype)
}

/// Evaluate the probe on `inp` (f16 output when `f16_output`) and return its
/// output as f32.
fn probe(inp: &MlxArray, f16_output: bool) -> Result<Vec<f32>, String> {
    let raw = jit_key_probe_array(inp, f16_output).map_err(|e| e.to_string())?;
    // Evaluate the probe on its own first, so the module lookup under test
    // happens before the conversion below is part of the graph.
    mlxcel_core::try_eval(&raw).map_err(|e| e.to_string())?;
    let out = mlxcel_core::astype(&raw, dtype::FLOAT32);
    mlxcel_core::try_eval(&out).map_err(|e| e.to_string())?;
    let bytes = mlxcel_core::array_to_raw_bytes(&out);
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn expected(shape0: f32, values: &[f32]) -> Vec<f32> {
    std::iter::once(shape0)
        .chain(values.iter().copied())
        .collect()
}

#[test]
fn input_dtype_selects_its_own_module() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let n = VALUES.len() as i32;
    // f32 first, so a name-keyed cache holds the `const float*` module when
    // the narrower dtypes arrive.
    for (label, dt) in [
        ("float32", dtype::FLOAT32),
        ("float16", dtype::FLOAT16),
        ("bfloat16", dtype::BFLOAT16),
    ] {
        let inp = input(&VALUES, &[n], dt);
        let got = probe(&inp, false).unwrap_or_else(|e| panic!("{label} probe failed: {e}"));
        assert_eq!(
            got,
            expected(n as f32, &VALUES),
            "a {label} input must be read as {label}, not through the module compiled for an earlier dtype"
        );
    }
}

#[test]
fn output_dtype_selects_its_own_module() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let n = VALUES.len() as i32;
    let inp = input(&VALUES, &[n], dtype::FLOAT32);
    // f32 output first, so a name-keyed cache holds the `float*` module when
    // the f16 output arrives.
    for (label, f16_output) in [("float32", false), ("float16", true)] {
        let got =
            probe(&inp, f16_output).unwrap_or_else(|e| panic!("{label}-output probe failed: {e}"));
        assert_eq!(
            got,
            expected(n as f32, &VALUES),
            "a {label} output must be written as {label}, not through the module compiled for an earlier output dtype"
        );
    }
}

fn parse_line(stdout: &str, label: &str) -> Result<Vec<f32>, String> {
    let prefix = format!("JIT_KEY_PROBE {label} ");
    // The harness prints `test <name> ... ` without a newline before the
    // child's first line, so the marker is searched for anywhere in a line.
    let line = stdout
        .lines()
        .find(|line| line.contains(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in the child's stdout:\n{stdout}"));
    let rest = &line[line.find(&prefix).expect("prefix located") + prefix.len()..];
    if let Some(err) = rest.strip_prefix("err=") {
        return Err(err.to_string());
    }
    let values = rest
        .strip_prefix("ok=")
        .unwrap_or_else(|| panic!("unexpected child line: {line}"));
    Ok(values
        .split(',')
        .map(|v| v.parse().expect("an f32 in the child's output"))
        .collect())
}

#[test]
fn scalar_input_after_vector_input_selects_its_own_module() {
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }
    let exe = std::env::current_exe().expect("path of this test binary");
    let mut child = Command::new(exe)
        .args([
            CHILD_TEST,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child test process");

    // Drain both pipes on their own threads so a chatty child (the HIP
    // runtime prints a queue dump on a fault) can never block on a full pipe.
    let mut stdout_pipe = child.stdout.take().expect("child stdout");
    let mut stderr_pipe = child.stderr.take().expect("child stderr");
    let stdout_reader = thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout_pipe.read_to_string(&mut text);
        text
    });
    let stderr_reader = thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr_pipe.read_to_string(&mut text);
        text
    });

    let deadline = Instant::now() + CHILD_BUDGET;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the child") {
            break Some(status);
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(50));
    };
    let stdout = stdout_reader.join().expect("stdout reader thread");
    let stderr = stderr_reader.join().expect("stderr reader thread");
    let context = format!("--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}");

    let status = status
        .unwrap_or_else(|| panic!("the child did not finish within {CHILD_BUDGET:?}\n{context}"));
    assert!(status.success(), "child exited with {status}\n{context}");

    let vector = parse_line(&stdout, "vector")
        .unwrap_or_else(|e| panic!("the 1-d probe failed: {e}\n{context}"));
    assert_eq!(
        vector,
        expected(4.0, &VALUES[..4]),
        "the 1-d probe must read its own `inp_shape`\n{context}"
    );
    let scalar = parse_line(&stdout, "scalar")
        .unwrap_or_else(|e| panic!("the 0-d probe after a 1-d one failed: {e}\n{context}"));
    assert_eq!(
        scalar,
        expected(-1.0, &VALUES[..1]),
        "a 0-d input has no `inp_shape` parameter and must not launch the module compiled for a 1-d input\n{context}"
    );
}

/// The body of `scalar_input_after_vector_input_selects_its_own_module`, run
/// in the child process it spawns. Prints one `JIT_KEY_PROBE <label>
/// <ok=v,v,...|err=...>` line per probe, then leaves through `_exit`.
#[test]
#[ignore = "spawned by scalar_input_after_vector_input_selects_its_own_module; a regression can fault the device for the whole process"]
fn child_scalar_after_vector() {
    if std::env::var_os(CHILD_ENV).is_none() {
        eprintln!("skipping: run through scalar_input_after_vector_input_selects_its_own_module");
        return;
    }
    if !on_rocm() {
        eprintln!("skipping: not running on a ROCm device");
        return;
    }

    fn report(label: &str, outcome: Result<Vec<f32>, String>) {
        match outcome {
            Ok(values) => {
                let joined: Vec<String> = values.iter().map(f32::to_string).collect();
                println!("JIT_KEY_PROBE {label} ok={}", joined.join(","));
            }
            Err(err) => println!("JIT_KEY_PROBE {label} err={}", err.replace('\n', " ")),
        }
        let _ = std::io::stdout().flush();
    }

    let vector = input(&VALUES[..4], &[4], dtype::FLOAT32);
    report("vector", probe(&vector, false));
    let scalar = input(&VALUES[..1], &[], dtype::FLOAT32);
    report("scalar", probe(&scalar, false));

    // A regression faults the queue, and HIP's teardown then waits on host
    // callbacks that never run, so leave without running it, as
    // `tests/rocm_gpu_faults.rs` does. The report lines were flushed above.
    // SAFETY: `_exit` only ends the process; nothing after it runs, and the
    // only state worth flushing (stdout) was flushed in `report`.
    unsafe { libc::_exit(0) }
}
