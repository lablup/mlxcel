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

//! Production-MLX operator micro-oracles on CUDA (issue #3212).
//!
//! Runs each operator's identical operands through the production MLX
//! operation on the MLX CUDA stream and through the emitted IREE operation on
//! the IREE `cuda` (or `hip`, or `local-*`) target, and prints one
//! `NumericOracleClaim::ProductionMlx` report per operator with the full
//! backend identity and the first divergent coordinate. Q4 reports are kept in
//! their own section, tagged with the post-#923 kernel identity.
//!
//! ```text
//! eval "$(bash scripts/iree/setup-cuda.sh --env)"
//! gpu-lock run --tag micro-oracle -- cargo run --release \
//!   --features cuda,xla-micro-oracle --example xla_mlx_micro_oracle -- \
//!   [--device cuda] [--only q4-qmv,q4-qmm] [--output reports.json]
//! ```
//!
//! Exits non-zero when any operator diverges. A divergence is a finding to
//! record, not a threshold to relax.

#[path = "xla_mlx_micro_oracle/identity.rs"]
mod identity;
#[path = "xla_common/mlx_reference_policy.rs"]
mod mlx_reference_policy;
#[path = "xla_mlx_micro_oracle/references.rs"]
mod references;

use mlxcel_xla::{
    AlgorithmIdentity, NumericBackendIdentity, NumericOracleReport, ProductionProbeOperator,
    production_probe_spec, run_production_probe,
};
use serde_json::json;

use identity::{MlxIdentity, mlx_identity};

const RUN_SCHEMA: &str = "mlxcel-xla-production-mlx-micro-oracle-run-v1";

struct Options {
    device: String,
    operators: Vec<ProductionProbeOperator>,
    output: Option<std::path::PathBuf>,
}

fn parse_options() -> Result<Options, String> {
    let mut device = "cuda".to_string();
    let mut operators = ProductionProbeOperator::ALL.to_vec();
    let mut output = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))
        };
        match argument.as_str() {
            "--device" => device = value()?,
            "--output" => output = Some(value()?.into()),
            "--only" => {
                operators = value()?
                    .split(',')
                    .map(|name| {
                        ProductionProbeOperator::from_name(name.trim()).ok_or_else(|| {
                            let known =
                                ProductionProbeOperator::ALL.map(|operator| operator.name());
                            format!("unknown operator {name:?}; known: {}", known.join(", "))
                        })
                    })
                    .collect::<Result<_, _>>()?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Options {
        device,
        operators,
        output,
    })
}

fn reference_backend(
    identity: &MlxIdentity,
    operator: ProductionProbeOperator,
) -> Result<NumericBackendIdentity, String> {
    NumericBackendIdentity::new(
        "mlx",
        identity.mlx_commit,
        identity.patch(),
        format!("mlxcel-core {} (cmake, nvcc)", identity.mlxcel_version),
        vec![
            format!("MLX_CUDA_ARCHITECTURES={}", identity.cuda_architectures),
            format!("MLX_ENABLE_TF32={}", identity.mlx_enable_tf32),
        ],
        "cuda",
        format!("{} ({})", identity.device, identity.compute_capability),
        format!("libmlx-{}", identity.mlx_commit),
        AlgorithmIdentity::observed(references::mlx_operation(operator))?,
    )
}

fn run_operator(
    device: &str,
    identity: &MlxIdentity,
    operator: ProductionProbeOperator,
) -> Result<NumericOracleReport, String> {
    let spec = production_probe_spec(operator)?;
    run_production_probe(
        device,
        operator,
        reference_backend(identity, operator)?,
        |operands| references::run_mlx_reference(&spec, operands),
    )
}

fn summary(report: &NumericOracleReport) -> serde_json::Value {
    json!({
        "operation": report.operation(),
        "passed": report.passed(),
        "production_qualified": report.production_qualified(),
        "failures": report.comparison().failures(),
        "max_absolute": report.comparison().max_absolute(),
        "max_relative": report.comparison().max_relative(),
        "first_divergence": report.first_divergence(),
    })
}

fn main() -> Result<(), String> {
    let mlx_enable_tf32 = mlx_reference_policy::pin_mlx_tf32_disabled()?;
    let options = parse_options()?;
    mlxcel_core::set_default_device(true);
    if !mlxcel_core::default_device_is_gpu() {
        return Err("the MLX reference must run on the MLX GPU stream".to_string());
    }
    let identity = mlx_identity(mlx_enable_tf32)?;
    let mut dense = Vec::new();
    let mut q4 = Vec::new();
    let mut errors = Vec::new();
    for operator in options.operators {
        eprintln!(
            "micro-oracle: running {} on {}",
            operator.name(),
            options.device
        );
        match run_operator(&options.device, &identity, operator) {
            Ok(report) if operator.is_affine_q4() => q4.push(report),
            Ok(report) => dense.push(report),
            Err(error) => errors.push(format!("{}: {error}", operator.name())),
        }
    }
    let reports = dense.iter().chain(&q4).collect::<Vec<_>>();
    let run = json!({
        "schema": RUN_SCHEMA,
        "iree_device": options.device,
        "mlx": identity,
        "summary": reports.iter().map(|report| summary(report)).collect::<Vec<_>>(),
        "reports": dense,
        "q4_post_923_reports": q4,
        "errors": errors,
    });
    let text = serde_json::to_string_pretty(&run)
        .map_err(|error| format!("serialize micro-oracle run: {error}"))?;
    println!("{text}");
    if let Some(path) = options.output {
        std::fs::write(&path, &text)
            .map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    let diverged = reports
        .iter()
        .filter(|report| !report.passed())
        .map(|report| report.operation())
        .collect::<Vec<_>>();
    if !errors.is_empty() || !diverged.is_empty() {
        return Err(format!(
            "production-MLX micro-oracle: {} diverged [{}], {} failed to run",
            diverged.len(),
            diverged.join(", "),
            errors.len()
        ));
    }
    Ok(())
}
