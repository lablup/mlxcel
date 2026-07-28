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

// Compiled only by mlxcel-xla's `diagnostics` feature. Production IREE startup
// does not parse or override global flags.

#include <stdio.h>

#include "iree/base/tooling/flags.h"

int xla_diagnostics_configure_local_task_threads(void) {
  char program[] = "mlxcel-xla-diagnostics";
  char topology_flag[] = "--task_topology_group_count=1";
  char worker_stack_flag[] = "--task_worker_stack_size=0";
  char* argv_storage[] = {program, topology_flag, worker_stack_flag, NULL};
  char** argv = argv_storage;
  int argc = 3;

  iree_status_t status =
      iree_flags_parse(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (!iree_status_is_ok(status)) {
    int status_code = (int)iree_status_code(status);
    fputs("failed to configure diagnostics-only IREE task threads: ", stderr);
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return status_code == 0 ? 1 : status_code;
  }
  if (argc != 1) {
    fputs("diagnostics-only IREE task thread flags were not consumed\n", stderr);
    return 1;
  }
  return 0;
}
