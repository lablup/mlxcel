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

//! Worker count of the programmatic IREE `local-task` device (issue #2790).
//!
//! The C shim builds the local-task device itself (`csrc/xla_local_task.c`)
//! with 8 MiB worker stacks, because IREE's flag-driven default (128 KiB)
//! cannot hold the static TLS of a process that has loaded CUDA. The worker
//! count is the one knob left: [`LOCAL_TASK_WORKERS_ENV`] when set, else
//! IREE's physical-core query (passed to C as `0`).
//!
//! Used by: `iree::load_ragged` and `auxiliary::AuxiliaryModule` (the two
//! device-creating FFI calls) and the test-support executor probe.

/// Environment variable that sets the local-task worker count.
pub const LOCAL_TASK_WORKERS_ENV: &str = "MLXCEL_XLA_LOCAL_TASK_WORKERS";

/// Whether `device` names IREE's local-task driver, matching
/// `xla_device_uri_is_local_task` in the shim.
#[must_use]
pub fn is_local_task_device(device: &str) -> bool {
    device == "local-task" || device.starts_with("local-task://")
}

/// IREE's `IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT`: the most workers a task
/// topology can hold. A larger request is rejected here instead of being
/// clamped silently by IREE.
pub const MAX_LOCAL_TASK_WORKERS: i32 = 256;

/// Parse a [`LOCAL_TASK_WORKERS_ENV`] value: unset is `0` (one worker per
/// physical core), anything else must be an integer in
/// `1..=`[`MAX_LOCAL_TASK_WORKERS`].
///
/// # Errors
///
/// Returns a message naming the variable for `0`, a negative, an empty, a
/// non-integer or an above-range value.
pub fn parse_local_task_workers(value: Option<&str>) -> Result<i32, String> {
    let Some(value) = value else {
        return Ok(0);
    };
    match value.trim().parse::<i32>() {
        Ok(workers) if (1..=MAX_LOCAL_TASK_WORKERS).contains(&workers) => Ok(workers),
        _ => Err(format!(
            "{LOCAL_TASK_WORKERS_ENV} must be an integer worker count between 1 and \
             {MAX_LOCAL_TASK_WORKERS}, got {value:?}"
        )),
    }
}

/// The worker count to hand the shim for `device`: `0` for any device but
/// local-task (the shim ignores it there), else the parsed
/// [`LOCAL_TASK_WORKERS_ENV`].
///
/// # Errors
///
/// See [`parse_local_task_workers`]; a non-UTF-8 value is rejected too.
pub fn local_task_workers_for(device: &str) -> Result<i32, String> {
    if !is_local_task_device(device) {
        return Ok(0);
    }
    match std::env::var(LOCAL_TASK_WORKERS_ENV) {
        Ok(value) => parse_local_task_workers(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{LOCAL_TASK_WORKERS_ENV} must be an integer worker count between 1 and {MAX_LOCAL_TASK_WORKERS}, got a non-UTF-8 value"
        )),
    }
}

/// What the shim's local-task executor was built with (test support).
#[cfg(all(
    feature = "iree",
    any(feature = "test-support", feature = "diagnostics")
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalTaskProbe {
    /// The worker stack size the executor was given, in bytes.
    pub worker_stack_size: u64,
    /// Live workers of the executor.
    pub worker_count: u64,
    /// IREE's global `--task_worker_stack_size` flag after the executor was
    /// built, or `None` when this IREE build has no flag registry.
    pub flag_worker_stack_size: Option<i64>,
}

#[cfg(all(
    feature = "iree",
    any(feature = "test-support", feature = "diagnostics")
))]
unsafe extern "C" {
    fn xla_local_task_probe(
        local_task_workers: i32,
        out_stack_size: *mut u64,
        out_worker_count: *mut u64,
        out_flag_stack_size: *mut i64,
    ) -> i32;
}

/// Build (and release) the local-task executor exactly as device creation
/// does, with `workers` workers (`0` = one per physical core), and report
/// how it was configured.
///
/// # Errors
///
/// Returns the shim's status code when the executor cannot be created, for
/// example when worker threads cannot be spawned.
#[cfg(all(
    feature = "iree",
    any(feature = "test-support", feature = "diagnostics")
))]
pub fn local_task_executor_probe(workers: i32) -> Result<LocalTaskProbe, String> {
    let mut stack = 0_u64;
    let mut count = 0_u64;
    let mut flag = -1_i64;
    // Safety: the out-pointers are valid for writes for the whole call.
    let rc = unsafe { xla_local_task_probe(workers, &mut stack, &mut count, &mut flag) };
    if rc != 0 {
        return Err(format!(
            "xla_local_task_probe failed (status {rc}); see the IREE status on stderr"
        ));
    }
    Ok(LocalTaskProbe {
        worker_stack_size: stack,
        worker_count: count,
        flag_worker_stack_size: (flag >= 0).then_some(flag),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_task_uris_match_the_shim() {
        assert!(is_local_task_device("local-task"));
        assert!(is_local_task_device("local-task://0"));
        assert!(!is_local_task_device("local-sync"));
        assert!(!is_local_task_device("cuda"));
        assert!(!is_local_task_device("local-tasks"));
    }

    #[test]
    fn unset_means_one_worker_per_physical_core() {
        assert_eq!(parse_local_task_workers(None), Ok(0));
    }

    #[test]
    fn a_positive_count_is_passed_through() {
        assert_eq!(parse_local_task_workers(Some("1")), Ok(1));
        assert_eq!(parse_local_task_workers(Some(" 12 ")), Ok(12));
        assert_eq!(
            parse_local_task_workers(Some("256")),
            Ok(MAX_LOCAL_TASK_WORKERS)
        );
    }

    #[test]
    fn a_count_above_the_iree_topology_limit_is_a_load_error() {
        for value in ["257", "1000"] {
            let err =
                parse_local_task_workers(Some(value)).expect_err("IREE would clamp this silently");
            assert!(err.contains(LOCAL_TASK_WORKERS_ENV), "{value:?}: {err}");
            assert!(err.contains("256"), "{value:?}: {err}");
        }
    }

    #[test]
    fn zero_negative_and_non_numeric_values_name_the_variable() {
        for value in ["0", "-2", "", "four", "1.5", "99999999999"] {
            let err = parse_local_task_workers(Some(value))
                .expect_err("only a positive i32 is a worker count");
            assert!(err.contains(LOCAL_TASK_WORKERS_ENV), "{value:?}: {err}");
        }
    }

    #[test]
    fn other_devices_never_read_the_variable() {
        // Whatever the environment holds, a non-local-task device gets 0.
        assert_eq!(local_task_workers_for("cuda"), Ok(0));
        assert_eq!(local_task_workers_for("hip"), Ok(0));
    }
}
