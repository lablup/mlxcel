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

//! Platform gate for distributed inference.
//!
//! Native Windows runs single-node only (#58): tensor parallelism, pipeline
//! parallelism, disaggregated prefill/decode and zero-config cluster bring-up
//! are refused there at startup instead of being attempted. The transport code
//! itself compiles on Windows; what is missing is any validation of
//! multi-host transport, discovery and RDMA on that platform, so the CLI and
//! the server stop before reaching it.

/// The refusal for a distributed request, or `None` when the platform allows
/// it. `what` names the flag or mode that was requested.
pub fn distributed_refusal(is_windows: bool, what: &str) -> Option<String> {
    is_windows.then(|| {
        format!(
            "{what} is not supported on native Windows: mlxcel runs single-node only there, with \
             tensor parallelism, pipeline parallelism and multi-node serving disabled. Run \
             distributed inference from Linux, macOS or WSL2."
        )
    })
}

/// Fail with [`distributed_refusal`]'s message when this build's platform does
/// not support distributed inference.
pub fn ensure_distributed_supported(what: &str) -> anyhow::Result<()> {
    match distributed_refusal(cfg!(windows), what) {
        Some(message) => anyhow::bail!(message),
        None => Ok(()),
    }
}

#[cfg(test)]
#[path = "platform_tests.rs"]
mod tests;
