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

#[cfg(feature = "diagnostics")]
use std::ffi::c_int;
#[cfg(feature = "diagnostics")]
use std::sync::OnceLock;

#[cfg(feature = "diagnostics")]
static LOCAL_TASK_THREADS: OnceLock<Result<(), String>> = OnceLock::new();

#[cfg(feature = "diagnostics")]
unsafe extern "C" {
    fn xla_diagnostics_configure_local_task_threads() -> c_int;
}

/// Configure bounded local-task threading before diagnostics create IREE.
///
/// This selects one topology group and lets pthreads choose the host-default
/// worker stack instead of IREE's exact `PTHREAD_STACK_MIN` request. The IREE
/// flag registry is process-global, so cache both success and failure and make
/// repeated diagnostic calls observe one immutable parse result.
#[cfg(feature = "diagnostics")]
pub fn configure_diagnostic_local_task_threads() -> Result<(), String> {
    LOCAL_TASK_THREADS
        .get_or_init(|| {
            // SAFETY: the diagnostics-only C helper takes no pointers, owns its
            // synthetic argv for the duration of the call, and returns a status.
            let status = unsafe { xla_diagnostics_configure_local_task_threads() };
            if status == 0 {
                Ok(())
            } else {
                Err(format!(
                    "failed to configure diagnostics-only IREE local-task threads \
                     (status {status})"
                ))
            }
        })
        .clone()
}

/// Whether the diagnostics-only threading override completed successfully.
#[doc(hidden)]
#[cfg(feature = "diagnostics")]
pub fn diagnostic_local_task_threads_are_configured() -> bool {
    LOCAL_TASK_THREADS.get().is_some_and(Result::is_ok)
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_helper_pins_bounded_host_thread_flags() {
        let source = include_str!("../csrc/xla_diagnostic_flags.c");
        assert!(source.contains("--task_topology_group_count=1"));
        assert!(source.contains("--task_worker_stack_size=0"));
        assert!(source.contains("IREE_FLAGS_PARSE_MODE_DEFAULT"));
    }

    #[test]
    fn rust_wrapper_caches_the_process_global_parse_result() {
        let source = include_str!("diagnostic_flags.rs");
        assert!(source.contains("OnceLock<Result<(), String>>"));
        assert!(source.contains(".get_or_init(||"));
    }
}
