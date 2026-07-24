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
    let report = mlxcel_xla::run_auxiliary_abi_smoke(&device)?;
    println!("device={device}");
    println!("float_output={:?}", report.float_output);
    println!("integer_output={:?}", report.integer_output);
    println!("bool_output={:?}", report.bool_output);
    println!(
        "negative_gates=few:{} many:{} type:{} shape:{} identity:{}",
        report.too_few_outputs_rejected,
        report.too_many_outputs_rejected,
        report.output_type_mismatch_rejected,
        report.output_shape_mismatch_rejected,
        report.config_identity_mismatch_rejected,
    );
    Ok(())
}
