#!/usr/bin/env bash
#
# 2-host logical pipeline-parallel job: simulates two PP stages as two
# independent processes on the same runner. The test exercises the full
# production code path under `src/distributed/pipeline/` — coordinator,
# activation transport, remote stage service, and scheduler — using the
# loopback TCP transport. This is the minimum viable guardrail for every
# PR that touches PP-related code.
#
# The underlying production code is exercised by the integration test
# `pipeline_multi_stage_two_host_logical_smoke` in
# `tests/pipeline_ci_multi_stage_real_models.rs`. The test is gated with
# `#[ignore]` because it needs a local model checkout, so it is driven
# from this script with `--ignored`.
#
# When model weights are present the test runs a real forward pass across
# two stages; when they are absent the test skips cleanly (exit 0), which
# keeps the GitHub-hosted runner job green while still catching regressions
# in everything that can be linked and compiled without model weights.
#
# Optional environment:
#   TEST_MODEL       Model directory name. Default: `llama-3.2-1b-4bit`.
#   CARGO_PROFILE    `debug` (default) or `release`.
#
# Usage:
#   scripts/ci/run-pp-two-host-logical.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/ci/_common.sh
source "$SCRIPT_DIR/_common.sh"

export PP_CI_SCRIPT_TAG="two-host-logical"
ROOT_DIR="$(pp_ci_root)"
cd "$ROOT_DIR"

pp_ci_require cargo

CARGO_PROFILE="${CARGO_PROFILE:-debug}"
TEST_MODEL="${TEST_MODEL:-llama-3.2-1b-4bit}"

PROFILE_FLAGS=()
if [[ "$CARGO_PROFILE" == "release" ]]; then
  PROFILE_FLAGS+=("--release")
fi

pp_ci_log "running 2-host logical pipeline-parallel smoke"
pp_ci_log "cargo profile: $CARGO_PROFILE"
pp_ci_log "target model: $TEST_MODEL"

# Non-ignored unit suites that exercise the production pipeline modules.
# Runs on every PR touching PP paths and does not need model weights.
cargo test "${PROFILE_FLAGS[@]}" \
  -p mlxcel \
  --lib \
  -- \
  distributed::pipeline:: \
  distributed::cluster_init:: \
  distributed::cluster_auth:: \
  distributed::cluster_handshake:: \
  distributed::discovery_beacon:: \
  distributed::peer_identity:: \
  distributed::tcp_transport:: \
  distributed::rdma_transport:: \
  models::gemma3::gemma3_stage_tests:: \
  models::gemma4::gemma4_stage_tests:: \
  --nocapture

# The two `*_stage_tests` filters above are the weight-free stand-in for the
# real-model remote parity tests (`tests/pipeline_remote_real_models.rs`,
# `tests/pipeline_server_remote_real_models.rs`): they split a synthetic
# Gemma 3 / Gemma 4 model into two stages and require the dense logits bit
# for bit (issue #2654), and `distributed::pipeline::wire_tensor::tests`
# pins the dtype-preserving stage boundary. The real-model tests stay out of
# this job: this runner has only the qwen3 fixture, and the families they
# cover need checkpoints it does not carry.

# Heterogeneous-memory partition regression: runs without model weights and
# ensures the partitioner plus admission control keep honouring memory
# imbalances (small stage 0, large stage 1).
cargo test "${PROFILE_FLAGS[@]}" \
  --test pipeline_ci_multi_stage_real_models \
  pipeline_heterogeneous_memory_partition_is_stable \
  -- --nocapture

# Multi-stage loopback smoke — gated by `#[ignore]` so it runs only here and
# self-skips when model weights are absent.
cargo test "${PROFILE_FLAGS[@]}" \
  --test pipeline_ci_multi_stage_real_models \
  pipeline_multi_stage_two_host_logical_smoke \
  -- --ignored --nocapture --test-threads=1

pp_ci_log "2-host logical job completed"
