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

// Generic resident-weight auxiliary IREE module used by vision/audio
// front-ends. It deliberately does not share or change the language-model ABI.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The host executable provides this libc-backed allocator implementation from
// xla_iree.c. Defining the control symbol before the IREE headers exposes
// iree_allocator_system() for this translation unit too.
#define IREE_ALLOCATOR_SYSTEM_CTL iree_xla_libc_ctl

#include <iree/runtime/api.h>

#define AUX_CHECK_GOTO(expr, label, rc_var)                              \
  do {                                                                   \
    iree_status_t _status = (expr);                                      \
    if (!iree_status_is_ok(_status)) {                                   \
      int _code = (int)iree_status_code(_status);                        \
      fprintf(stderr, "xla_aux: %s failed (status %d):\n", #expr,       \
              _code);                                                    \
      iree_status_fprint(stderr, _status);                               \
      iree_status_ignore(_status);                                       \
      rc_var = _code ? _code : 1;                                       \
      goto label;                                                        \
    }                                                                    \
  } while (0)

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

typedef struct xla_aux_ctx {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;
  iree_hal_allocator_t* allocator;
  int32_t n_weights;
  iree_hal_buffer_view_t** weights;
  char* entry_name;
} xla_aux_ctx;

void xla_aux_free(xla_aux_ctx* context);

static const iree_hal_buffer_params_t kDeviceLocalParams = {
    .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    .access = IREE_HAL_MEMORY_ACCESS_ALL,
    .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
};

// Auxiliary invocation dtype: 0=f32, 1=i32, 2=bool8.
static bool descriptor_dtype(int32_t dtype,
                             iree_hal_element_type_t* element_type,
                             iree_host_size_t* element_size) {
  switch (dtype) {
    case 0:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      *element_size = sizeof(float);
      return true;
    case 1:
      *element_type = IREE_HAL_ELEMENT_TYPE_INT_32;
      *element_size = sizeof(int32_t);
      return true;
    case 2:
      *element_type = IREE_HAL_ELEMENT_TYPE_BOOL_8;
      *element_size = sizeof(uint8_t);
      return true;
    default:
      return false;
  }
}

static int validate_descriptor(const char* kind, int32_t index,
                               const void* data, size_t byte_length,
                               int32_t dtype, int32_t rank,
                               const int64_t dims[4],
                               iree_host_size_t* element_count,
                               iree_hal_element_type_t* element_type) {
  iree_host_size_t element_size = 0;
  if (!data || rank < 0 || rank > 4 ||
      !descriptor_dtype(dtype, element_type, &element_size)) {
    fprintf(stderr,
            "xla_aux: invalid %s %d pointer/dtype/rank (dtype=%d rank=%d)\n",
            kind, index, dtype, rank);
    return 1;
  }
  iree_host_size_t count = 1;
  for (int32_t axis = 0; axis < rank; ++axis) {
    int64_t dimension = dims[axis];
    if (dimension <= 0 || (uint64_t)dimension > SIZE_MAX / count) {
      fprintf(stderr, "xla_aux: invalid %s %d axis %d=%lld\n", kind, index,
              axis, (long long)dimension);
      return 1;
    }
    count *= (iree_host_size_t)dimension;
  }
  if (count > SIZE_MAX / element_size ||
      byte_length != (size_t)(count * element_size)) {
    fprintf(stderr,
            "xla_aux: %s %d byte count %zu disagrees with dtype/shape\n",
            kind, index, byte_length);
    return 1;
  }
  *element_count = count;
  return 0;
}

static iree_status_t allocate_input(xla_aux_ctx* context, int32_t index,
                                    const xla_tensor_desc* descriptor,
                                    iree_hal_buffer_view_t** output) {
  iree_host_size_t count = 0;
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  if (validate_descriptor("input", index, descriptor->data,
                          descriptor->byte_length, descriptor->dtype,
                          descriptor->rank, descriptor->dims, &count,
                          &element_type) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid auxiliary input descriptor");
  }
  (void)count;
  iree_hal_dim_t shape[4];
  for (int32_t axis = 0; axis < descriptor->rank; ++axis) {
    shape[axis] = (iree_hal_dim_t)descriptor->dims[axis];
  }
  return iree_hal_buffer_view_allocate_buffer_copy(
      context->device, context->allocator,
      (iree_host_size_t)descriptor->rank, shape, element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, kDeviceLocalParams,
      iree_make_const_byte_span(descriptor->data,
                                (iree_host_size_t)descriptor->byte_length),
      output);
}

static iree_status_t create_impl(
    xla_aux_ctx* context, const char* device_uri, const char* module_vmfb,
    const char* entry_name, uint64_t compatibility_fingerprint,
    int32_t n_weights, const void* const* weight_data,
    const int32_t* weight_dtypes, const int32_t* weight_ranks,
    const int64_t* weight_dims) {
  if (!device_uri || !module_vmfb || !entry_name || entry_name[0] == '\0' ||
      compatibility_fingerprint == 0 || n_weights <= 0 || !weight_data ||
      !weight_dtypes || !weight_ranks || !weight_dims) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid auxiliary module creation contract");
  }
  context->n_weights = n_weights;
  size_t entry_length = strlen(entry_name);
  context->entry_name = (char*)malloc(entry_length + 1);
  if (!context->entry_name) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  memcpy(context->entry_name, entry_name, entry_length + 1);

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &context->instance));
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      context->instance, iree_make_cstring_view(device_uri),
      &context->device));
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      context->instance, &session_options, context->device,
      iree_runtime_instance_host_allocator(context->instance),
      &context->session));
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      context->session, module_vmfb));
  iree_runtime_call_t probe;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      context->session, iree_make_cstring_view(context->entry_name), &probe));
  iree_runtime_call_deinitialize(&probe);
  context->allocator =
      iree_runtime_session_device_allocator(context->session);

  context->weights = (iree_hal_buffer_view_t**)calloc(
      (size_t)n_weights, sizeof(iree_hal_buffer_view_t*));
  if (!context->weights) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  for (int32_t index = 0; index < n_weights; ++index) {
    int32_t rank = weight_ranks[index];
    if (rank < 0 || rank > 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "auxiliary weight rank is outside 0..=4");
    }
    iree_hal_dim_t shape[4];
    iree_host_size_t count = 1;
    for (int32_t axis = 0; axis < rank; ++axis) {
      int64_t dimension = weight_dims[index * 4 + axis];
      if (dimension <= 0 || (uint64_t)dimension > SIZE_MAX / count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid auxiliary weight shape");
      }
      shape[axis] = (iree_hal_dim_t)dimension;
      count *= (iree_host_size_t)dimension;
    }
    iree_hal_element_type_t element_type;
    iree_host_size_t element_size;
    switch (weight_dtypes[index]) {
      case 0:
        element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
        element_size = sizeof(float);
        break;
      case 1:
        element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
        element_size = sizeof(uint16_t);
        break;
      case 2:
        element_type = IREE_HAL_ELEMENT_TYPE_UINT_32;
        element_size = sizeof(uint32_t);
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown auxiliary weight dtype");
    }
    if (count > SIZE_MAX / element_size) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "auxiliary weight byte count overflowed");
    }
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
        context->device, context->allocator, (iree_host_size_t)rank, shape,
        element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        kDeviceLocalParams,
        iree_make_const_byte_span(weight_data[index], count * element_size),
        &context->weights[index]));
  }
  return iree_ok_status();
}

xla_aux_ctx* xla_aux_create(
    const char* device_uri, const char* module_vmfb, const char* entry_name,
    uint64_t compatibility_fingerprint, int32_t n_weights,
    const void* const* weight_data, const int32_t* weight_dtypes,
    const int32_t* weight_ranks, const int64_t* weight_dims) {
  xla_aux_ctx* context = (xla_aux_ctx*)calloc(1, sizeof(xla_aux_ctx));
  if (!context) return NULL;
  iree_status_t status = create_impl(
      context, device_uri, module_vmfb, entry_name,
      compatibility_fingerprint, n_weights, weight_data, weight_dtypes,
      weight_ranks, weight_dims);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "xla_aux_create failed (status %d):\n",
            (int)iree_status_code(status));
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    xla_aux_free(context);
    return NULL;
  }
  return context;
}

int xla_aux_invoke(xla_aux_ctx* context, int32_t n_inputs,
                   const xla_tensor_desc* inputs, int32_t n_outputs,
                   xla_mut_tensor_desc* outputs) {
  if (!context || n_inputs < 0 || n_outputs <= 0 ||
      (n_inputs > 0 && !inputs) || !outputs) {
    return 1;
  }
  int rc = 0;
  bool call_initialized = false;
  iree_runtime_call_t call;
  iree_hal_buffer_view_t** input_views = n_inputs == 0
      ? NULL
      : (iree_hal_buffer_view_t**)calloc(
            (size_t)n_inputs, sizeof(iree_hal_buffer_view_t*));
  iree_hal_buffer_view_t** output_views =
      (iree_hal_buffer_view_t**)calloc(
          (size_t)n_outputs, sizeof(iree_hal_buffer_view_t*));
  if ((n_inputs > 0 && !input_views) || !output_views) {
    free(input_views);
    free(output_views);
    return 1;
  }
  AUX_CHECK_GOTO(iree_runtime_call_initialize_by_name(
                     context->session,
                     iree_make_cstring_view(context->entry_name), &call),
                 cleanup, rc);
  call_initialized = true;
  for (int32_t index = 0; index < context->n_weights; ++index) {
    AUX_CHECK_GOTO(iree_runtime_call_inputs_push_back_buffer_view(
                       &call, context->weights[index]),
                   cleanup, rc);
  }
  for (int32_t index = 0; index < n_inputs; ++index) {
    AUX_CHECK_GOTO(allocate_input(context, index, &inputs[index],
                                  &input_views[index]),
                   cleanup, rc);
    AUX_CHECK_GOTO(iree_runtime_call_inputs_push_back_buffer_view(
                       &call, input_views[index]),
                   cleanup, rc);
  }
  AUX_CHECK_GOTO(iree_runtime_call_invoke(&call, 0), cleanup, rc);
  if (iree_vm_list_size(iree_runtime_call_outputs(&call)) !=
      (iree_host_size_t)n_outputs) {
    fprintf(stderr,
            "xla_aux: function returned %zu outputs, caller declared %d\n",
            (size_t)iree_vm_list_size(iree_runtime_call_outputs(&call)),
            n_outputs);
    rc = 1;
    goto cleanup;
  }
  for (int32_t index = 0; index < n_outputs; ++index) {
    AUX_CHECK_GOTO(iree_runtime_call_outputs_pop_front_buffer_view(
                       &call, &output_views[index]),
                   cleanup, rc);
    iree_host_size_t count = 0;
    iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
    if (validate_descriptor(
            "output", index, outputs[index].data, outputs[index].byte_length,
            outputs[index].dtype, outputs[index].rank, outputs[index].dims,
            &count, &element_type) != 0) {
      rc = 1;
      goto cleanup;
    }
    iree_host_size_t actual_rank =
        iree_hal_buffer_view_shape_rank(output_views[index]);
    iree_hal_element_type_t actual_type =
        iree_hal_buffer_view_element_type(output_views[index]);
    iree_host_size_t actual_count =
        iree_hal_buffer_view_element_count(output_views[index]);
    if (actual_rank != (iree_host_size_t)outputs[index].rank ||
        actual_type != element_type ||
        iree_hal_buffer_view_encoding_type(output_views[index]) !=
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR ||
        actual_count != count) {
      fprintf(stderr,
              "xla_aux: output %d type/shape mismatch "
              "(rank=%zu/%d type=%u/%u elements=%zu/%zu)\n",
              index, (size_t)actual_rank, outputs[index].rank,
              (unsigned)actual_type, (unsigned)element_type,
              (size_t)actual_count, (size_t)count);
      rc = 1;
      goto cleanup;
    }
    for (int32_t axis = 0; axis < outputs[index].rank; ++axis) {
      if (iree_hal_buffer_view_shape_dim(output_views[index], axis) !=
          (iree_hal_dim_t)outputs[index].dims[axis]) {
        fprintf(stderr,
                "xla_aux: output %d axis %d mismatch (%lld/%lld)\n",
                index, axis,
                (long long)iree_hal_buffer_view_shape_dim(
                    output_views[index], axis),
                (long long)outputs[index].dims[axis]);
        rc = 1;
        goto cleanup;
      }
    }
    AUX_CHECK_GOTO(iree_hal_device_transfer_d2h(
                       context->device,
                       iree_hal_buffer_view_buffer(output_views[index]), 0,
                       outputs[index].data,
                       (iree_device_size_t)outputs[index].byte_length,
                       IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                       iree_infinite_timeout()),
                   cleanup, rc);
  }

cleanup:
  for (int32_t index = 0; index < n_outputs; ++index) {
    if (output_views[index]) {
      iree_hal_buffer_view_release(output_views[index]);
    }
  }
  for (int32_t index = 0; index < n_inputs; ++index) {
    if (input_views[index]) iree_hal_buffer_view_release(input_views[index]);
  }
  free(output_views);
  free(input_views);
  if (call_initialized) iree_runtime_call_deinitialize(&call);
  return rc;
}

void xla_aux_free(xla_aux_ctx* context) {
  if (!context) return;
  if (context->weights) {
    for (int32_t index = 0; index < context->n_weights; ++index) {
      if (context->weights[index]) {
        iree_hal_buffer_view_release(context->weights[index]);
      }
    }
    free(context->weights);
  }
  free(context->entry_name);
  if (context->session) iree_runtime_session_release(context->session);
  if (context->device) iree_hal_device_release(context->device);
  if (context->instance) iree_runtime_instance_release(context->instance);
  free(context);
}
