#!/usr/bin/env bash
set -euo pipefail
cd /home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2227
git rev-parse HEAD > /tmp/mlxcel-auto-plan/2227-final-build-source.txt
echo release > /tmp/mlxcel-auto-plan/2227-final-build-phase.txt
cargo build --release --features cuda --bin mlxcel-bench-engine --bin mlxcel-engine-parity --bin mlxcel > /tmp/mlxcel-auto-plan/2227-final-release.log 2>&1
echo test-build > /tmp/mlxcel-auto-plan/2227-final-build-phase.txt
cargo test --workspace --profile test-fast --features cuda --no-run > /tmp/mlxcel-auto-plan/2227-final-test-build.log 2>&1
echo clippy > /tmp/mlxcel-auto-plan/2227-final-build-phase.txt
cargo clippy --workspace --all-targets --features cuda -- -D warnings > /tmp/mlxcel-auto-plan/2227-final-clippy.log 2>&1
echo fmt > /tmp/mlxcel-auto-plan/2227-final-build-phase.txt
cargo fmt --all -- --check > /tmp/mlxcel-auto-plan/2227-final-fmt.log 2>&1
echo contracts > /tmp/mlxcel-auto-plan/2227-final-build-phase.txt
make verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat > /tmp/mlxcel-auto-plan/2227-final-contracts.log 2>&1
echo complete > /tmp/mlxcel-auto-plan/2227-final-build-phase.txt
