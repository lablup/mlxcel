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

//! Arms, shapes, modes and argument parsing of `oxide_fused_norm_bench`.

use mlxcel_core::dtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Graph,
    Fused,
    Oxide,
}

impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Arm::Graph => "graph",
            Arm::Fused => "fused",
            Arm::Oxide => "oxide",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "graph" => Some(Arm::Graph),
            "fused" => Some(Arm::Fused),
            "oxide" => Some(Arm::Oxide),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub act: i32,
    pub weight: i32,
    pub rows: i32,
    pub dim: i32,
}

pub fn dtype_name(dt: i32) -> &'static str {
    match dt {
        dtype::FLOAT32 => "f32",
        dtype::FLOAT16 => "f16",
        dtype::BFLOAT16 => "bf16",
        _ => "other",
    }
}

pub fn parse_dtype(s: &str) -> Option<i32> {
    match s {
        "f32" => Some(dtype::FLOAT32),
        "f16" => Some(dtype::FLOAT16),
        "bf16" => Some(dtype::BFLOAT16),
        _ => None,
    }
}

/// The residual joins of the #2834 checkpoint (hidden size 2048, bf16
/// activations and weight): one decode step at B=1 and B=4, and the prefill of
/// a 128-token and a 2048-token prompt at B=1 plus the 128-token one at B=4.
pub fn model_shapes() -> Vec<Shape> {
    [1, 4, 128, 512, 2048]
        .into_iter()
        .map(|rows| Shape {
            act: dtype::BFLOAT16,
            weight: dtype::BFLOAT16,
            rows,
            dim: 2048,
        })
        .collect()
}

/// The shape list of the #2832 standalone benchmark
/// (`spike/cuda-oxide-fused-norm/results/bench-sm_121.json`).
pub fn bench2832_shapes() -> Vec<Shape> {
    let mut shapes = Vec::new();
    for dt in [dtype::FLOAT32, dtype::BFLOAT16, dtype::FLOAT16] {
        for (rows, dim) in [
            (1, 1024),
            (1, 4096),
            (1, 8192),
            (5, 4096),
            (33, 2048),
            (64, 8192),
        ] {
            shapes.push(Shape {
                act: dt,
                weight: dt,
                rows,
                dim,
            });
        }
    }
    shapes
}

pub fn parse_shapes(spec: &str) -> Result<Vec<Shape>, String> {
    let mut shapes = Vec::new();
    for item in spec.split(',').filter(|s| !s.is_empty()) {
        match item {
            "model" => shapes.extend(model_shapes()),
            "bench2832" => shapes.extend(bench2832_shapes()),
            custom => {
                // act:weight:ROWSxDIM
                let parts: Vec<&str> = custom.split(':').collect();
                let bad = || format!("shape {custom:?}: expected act:weight:ROWSxDIM");
                if parts.len() != 3 {
                    return Err(bad());
                }
                let (rows, dim) = parts[2].split_once('x').ok_or_else(bad)?;
                shapes.push(Shape {
                    act: parse_dtype(parts[0]).ok_or_else(bad)?,
                    weight: parse_dtype(parts[1]).ok_or_else(bad)?,
                    rows: rows.parse().map_err(|_| bad())?,
                    dim: dim.parse().map_err(|_| bad())?,
                });
            }
        }
    }
    if shapes.is_empty() {
        return Err("no shapes selected".to_string());
    }
    Ok(shapes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Build,
    Iso,
    Pipe,
    Cold,
}

pub struct Args {
    pub mode: Mode,
    pub arms: Vec<Arm>,
    pub shapes: Vec<Shape>,
    pub iters: usize,
    pub warmup: usize,
    pub rounds: usize,
    pub csv: Option<String>,
    pub label: String,
}

pub fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        mode: Mode::Iso,
        arms: vec![Arm::Graph, Arm::Fused, Arm::Oxide],
        shapes: model_shapes(),
        iters: 200,
        warmup: 20,
        rounds: 5,
        csv: None,
        label: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--mode" => {
                args.mode = match value()?.as_str() {
                    "build" => Mode::Build,
                    "iso" => Mode::Iso,
                    "pipe" => Mode::Pipe,
                    "cold" => Mode::Cold,
                    other => return Err(format!("unknown mode {other:?}")),
                }
            }
            "--arms" => {
                args.arms = value()?
                    .split(',')
                    .map(|s| Arm::parse(s).ok_or_else(|| format!("unknown arm {s:?}")))
                    .collect::<Result<_, _>>()?;
            }
            "--shapes" => args.shapes = parse_shapes(&value()?)?,
            "--iters" => args.iters = value()?.parse().map_err(|e| format!("--iters: {e}"))?,
            "--warmup" => args.warmup = value()?.parse().map_err(|e| format!("--warmup: {e}"))?,
            "--rounds" => args.rounds = value()?.parse().map_err(|e| format!("--rounds: {e}"))?,
            "--csv" => args.csv = Some(value()?),
            "--label" => args.label = value()?,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if args.arms.is_empty() || args.iters == 0 || args.rounds == 0 {
        return Err("--arms, --iters and --rounds must be non-empty".to_string());
    }
    if args.mode == Mode::Cold && args.arms.len() != 1 {
        return Err("--mode cold measures one arm per process; pass a single --arms".to_string());
    }
    Ok(args)
}
