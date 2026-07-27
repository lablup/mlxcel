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

fn main() -> Result<(), String> {
    let device = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "local-task".to_string());
    let report = mlxcel_xla::run_dense_matmul_probe(&device)?;
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("serialize numeric probe report: {error}"))?;
    println!("{json}");
    if !report.passed() {
        return Err("dense matmul canonical-decomposition probe diverged".to_string());
    }
    if report.production_qualified() {
        return Err(
            "canonical-decomposition probe must never claim production qualification".to_string(),
        );
    }
    Ok(())
}
