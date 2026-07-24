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

// Standalone real-runtime acceptance probe for csrc/xla_aux.c.

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IREE_ALLOCATOR_SYSTEM_CTL iree_xla_libc_ctl
#include <iree/runtime/api.h>

iree_status_t iree_xla_libc_ctl(void* self, iree_allocator_command_t command,
                                const void* params, void** inout_ptr) {
  (void)self;
  const iree_allocator_alloc_params_t* p =
      (const iree_allocator_alloc_params_t*)params;
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
      *inout_ptr = malloc(p->byte_length ? p->byte_length : 1);
      break;
    case IREE_ALLOCATOR_COMMAND_CALLOC:
      *inout_ptr = calloc(1, p->byte_length ? p->byte_length : 1);
      break;
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      *inout_ptr = realloc(*inout_ptr, p->byte_length ? p->byte_length : 1);
      break;
    case IREE_ALLOCATOR_COMMAND_FREE:
      free(*inout_ptr);
      *inout_ptr = NULL;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
  }
  return *inout_ptr ? iree_ok_status()
                    : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
}

typedef struct xla_aux_ctx xla_aux_ctx;
typedef struct xla_tensor_desc {
  const void* data;
  size_t byte_length;
  int32_t dtype;
  int32_t rank;
  int64_t dims[4];
} xla_tensor_desc;
typedef struct xla_mut_tensor_desc {
  void* data;
  size_t byte_length;
  int32_t dtype;
  int32_t rank;
  int64_t dims[4];
} xla_mut_tensor_desc;

xla_aux_ctx* xla_aux_create(
    const char* device_uri, const char* module_vmfb, const char* entry_name,
    uint64_t compatibility_fingerprint, int32_t n_weights,
    const void* const* weight_data, const int32_t* weight_dtypes,
    const int32_t* weight_ranks, const int64_t* weight_dims);
int xla_aux_invoke(xla_aux_ctx* context, int32_t n_inputs,
                   const xla_tensor_desc* inputs, int32_t n_outputs,
                   xla_mut_tensor_desc* outputs);
void xla_aux_free(xla_aux_ctx* context);

static xla_mut_tensor_desc output(void* data, size_t bytes, int32_t dtype,
                                  int32_t rank, int64_t d0, int64_t d1) {
  xla_mut_tensor_desc result = {
      .data = data,
      .byte_length = bytes,
      .dtype = dtype,
      .rank = rank,
      .dims = {d0, d1, 0, 0},
  };
  return result;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s DEVICE MODULE_VMFB\n", argv[0]);
    return 2;
  }
  float weight[] = {1.5f, -2.0f};
  const void* weight_data[] = {weight};
  int32_t weight_dtypes[] = {0};
  int32_t weight_ranks[] = {1};
  int64_t weight_dims[] = {2, 0, 0, 0};
  xla_aux_ctx* context =
      xla_aux_create(argv[1], argv[2], "aux_smoke.main", 1, 1, weight_data,
                     weight_dtypes, weight_ranks, weight_dims);
  if (!context) return 1;

  float floats[] = {2.5f, 8.0f};
  int32_t integers[] = {7, -3};
  uint8_t mask[] = {1, 0};
  xla_tensor_desc inputs[] = {
      {.data = floats,
       .byte_length = sizeof(floats),
       .dtype = 0,
       .rank = 1,
       .dims = {2, 0, 0, 0}},
      {.data = integers,
       .byte_length = sizeof(integers),
       .dtype = 1,
       .rank = 1,
       .dims = {2, 0, 0, 0}},
      {.data = mask,
       .byte_length = sizeof(mask),
       .dtype = 2,
       .rank = 1,
       .dims = {2, 0, 0, 0}},
  };
  float float_output[2] = {0};
  int32_t integer_output[2] = {0};
  uint8_t bool_output[2] = {0};
  xla_mut_tensor_desc outputs[] = {
      output(float_output, sizeof(float_output), 0, 1, 2, 0),
      output(integer_output, sizeof(integer_output), 1, 1, 2, 0),
      output(bool_output, sizeof(bool_output), 2, 1, 2, 0),
  };
  if (xla_aux_invoke(context, 3, inputs, 3, outputs) != 0 ||
      fabsf(float_output[0] - 4.0f) > 1e-6f ||
      fabsf(float_output[1] - 6.0f) > 1e-6f ||
      integer_output[0] != 7 || integer_output[1] != -3 ||
      bool_output[0] != 1 || bool_output[1] != 0) {
    fprintf(stderr, "positive typed multi-I/O invocation failed\n");
    xla_aux_free(context);
    return 1;
  }

  bool too_few = xla_aux_invoke(context, 3, inputs, 2, outputs) != 0;
  xla_mut_tensor_desc extra_outputs[] = {
      outputs[0], outputs[1], outputs[2],
      output(bool_output, sizeof(bool_output), 2, 1, 2, 0),
  };
  bool too_many =
      xla_aux_invoke(context, 3, inputs, 4, extra_outputs) != 0;
  xla_mut_tensor_desc wrong_type[] = {outputs[0], outputs[1], outputs[2]};
  wrong_type[0].dtype = 1;
  bool type_mismatch =
      xla_aux_invoke(context, 3, inputs, 3, wrong_type) != 0;
  xla_mut_tensor_desc wrong_shape[] = {outputs[0], outputs[1], outputs[2]};
  wrong_shape[1].rank = 2;
  wrong_shape[1].dims[0] = 1;
  wrong_shape[1].dims[1] = 2;
  bool shape_mismatch =
      xla_aux_invoke(context, 3, inputs, 3, wrong_shape) != 0;
  xla_aux_free(context);

  printf("float_output=[%.1f,%.1f]\n", float_output[0], float_output[1]);
  printf("integer_output=[%d,%d]\n", integer_output[0], integer_output[1]);
  printf("bool8_output=[%u,%u]\n", bool_output[0], bool_output[1]);
  printf("negative_gates=few:%d many:%d type:%d shape:%d\n", too_few,
         too_many, type_mismatch, shape_mismatch);
  return too_few && too_many && type_mismatch && shape_mismatch ? 0 : 1;
}
