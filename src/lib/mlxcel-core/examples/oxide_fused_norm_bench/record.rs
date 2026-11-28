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

//! Timing statistics and CSV output of `oxide_fused_norm_bench`.

use std::fmt::Write as _;

use super::args::{Args, Arm, Shape, dtype_name};

pub struct Stats {
    pub mean_us: f64,
    pub median_us: f64,
    pub p90_us: f64,
    pub min_us: f64,
}

pub fn stats(samples_us: &mut [f64]) -> Stats {
    samples_us.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    let n = samples_us.len();
    let median_us = if n.is_multiple_of(2) {
        (samples_us[n / 2 - 1] + samples_us[n / 2]) / 2.0
    } else {
        samples_us[n / 2]
    };
    Stats {
        mean_us: samples_us.iter().sum::<f64>() / n as f64,
        median_us,
        p90_us: samples_us[((n as f64 * 0.9).ceil() as usize).clamp(1, n) - 1],
        min_us: samples_us[0],
    }
}

pub const CSV_HEADER: &str = "label,mode,arm,act,weight,rows,dim,round,iters,warmup,mean_us,median_us,\
p90_us,min_us,oxide_routes,native_routes,launches,route_valid";

pub struct Recorder {
    pub csv: String,
    pub invalid: usize,
}

impl Recorder {
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        args: &Args,
        mode: &str,
        arm: Arm,
        shape: Shape,
        round: usize,
        iters: usize,
        s: &Stats,
        delta: (u64, u64, u64),
        valid: bool,
    ) {
        if !valid {
            self.invalid += 1;
        }
        let _ = writeln!(
            self.csv,
            "{},{mode},{},{},{},{},{},{round},{iters},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{valid}",
            args.label,
            arm.name(),
            dtype_name(shape.act),
            dtype_name(shape.weight),
            shape.rows,
            shape.dim,
            args.warmup,
            s.mean_us,
            s.median_us,
            s.p90_us,
            s.min_us,
            delta.0,
            delta.1,
            delta.2,
        );
        println!(
            "{mode:<5} {:<5} {:>4}/{:<4} {:>5} x {:<5} round {round}  mean {:>9.3} us  median {:>9.3}  p90 {:>9.3}  min {:>9.3}  routes {delta:?}{}",
            arm.name(),
            dtype_name(shape.act),
            dtype_name(shape.weight),
            shape.rows,
            shape.dim,
            s.mean_us,
            s.median_us,
            s.p90_us,
            s.min_us,
            if valid { "" } else { "  INVALID ROUTE" },
        );
    }
}

pub fn append(path: &str, header: &str, body: &str) -> Result<(), String> {
    use std::io::Write as _;
    let exists = std::path::Path::new(path).exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{path}: {e}"))?;
    if !exists {
        file.write_all(header.as_bytes())
            .map_err(|e| format!("{path}: {e}"))?;
    }
    file.write_all(body.as_bytes())
        .map_err(|e| format!("{path}: {e}"))
}
