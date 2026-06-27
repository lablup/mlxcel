// Minimal C shim over the IREE runtime C API: load a vmfb and invoke
// module.main with two [n]f32 inputs, returning the [n]f32 output. This is the
// FFI-gate proof (issue #449 Phase 3 M2) and the shape the mlxcel-xla backend
// will use: a thin C shim over the prebuilt IREE runtime, with Rust calling a
// flat C ABI rather than binding the runtime structs directly.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// The iree-dist build leaves the system allocator to the application (its
// iree_allocator_system() is gated on IREE_ALLOCATOR_SYSTEM_CTL). Point it at a
// libc malloc/free control function, defined below, before the IREE headers.
#define IREE_ALLOCATOR_SYSTEM_CTL iree_gate_libc_ctl

#include <iree/runtime/api.h>
#include <iree/hal/buffer_view_util.h>
#include <iree/hal/buffer_transfer.h>

// libc-backed implementation of the system allocator control function.
iree_status_t iree_gate_libc_ctl(void* self, iree_allocator_command_t command,
                                 const void* params, void** inout_ptr) {
  (void)self;
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC: {
      const iree_allocator_alloc_params_t* p =
          (const iree_allocator_alloc_params_t*)params;
      iree_host_size_t len = p->byte_length ? p->byte_length : 1;
      void* ptr = NULL;
      if (command == IREE_ALLOCATOR_COMMAND_CALLOC) {
        ptr = calloc(1, len);
      } else if (command == IREE_ALLOCATOR_COMMAND_REALLOC) {
        ptr = realloc(*inout_ptr, len);
      } else {
        ptr = malloc(len);
      }
      if (!ptr) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
      *inout_ptr = ptr;
      return iree_ok_status();
    }
    case IREE_ALLOCATOR_COMMAND_FREE:
      free(*inout_ptr);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
  }
}

#define GATE_CHECK(expr)                                                 \
  do {                                                                   \
    iree_status_t _s = (expr);                                          \
    if (!iree_status_is_ok(_s)) {                                       \
      int _c = (int)iree_status_code(_s);                               \
      fprintf(stderr, "iree_gate: %s failed (status %d)\n", #expr, _c); \
      iree_status_ignore(_s);                                           \
      return _c ? _c : 1;                                               \
    }                                                                    \
  } while (0)

// Returns 0 on success; a,b and out are host arrays of n f32.
int iree_gate_run_add(const char* vmfb_path, const float* a, const float* b,
                      int32_t n, float* out) {
  iree_runtime_instance_options_t inst_opts;
  iree_runtime_instance_options_initialize(&inst_opts);
  iree_runtime_instance_options_use_all_available_drivers(&inst_opts);
  iree_runtime_instance_t* instance = NULL;
  GATE_CHECK(iree_runtime_instance_create(&inst_opts, iree_allocator_system(),
                                          &instance));

  iree_hal_device_t* device = NULL;
  GATE_CHECK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("local-task"), &device));

  iree_runtime_session_options_t sess_opts;
  iree_runtime_session_options_initialize(&sess_opts);
  iree_runtime_session_t* session = NULL;
  GATE_CHECK(iree_runtime_session_create_with_device(
      instance, &sess_opts, device,
      iree_runtime_instance_host_allocator(instance), &session));

  GATE_CHECK(
      iree_runtime_session_append_bytecode_module_from_file(session, vmfb_path));

  iree_runtime_call_t call;
  GATE_CHECK(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.main"), &call));

  iree_hal_allocator_t* allocator =
      iree_runtime_session_device_allocator(session);
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  iree_hal_dim_t shape[1] = {(iree_hal_dim_t)n};
  iree_host_size_t bytes = (iree_host_size_t)n * sizeof(float);

  iree_hal_buffer_view_t* arg0 = NULL;
  GATE_CHECK(iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, 1, shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(a, bytes), &arg0));
  iree_hal_buffer_view_t* arg1 = NULL;
  GATE_CHECK(iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, 1, shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(b, bytes), &arg1));

  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, arg0));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, arg1));

  GATE_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* ret = NULL;
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret));
  GATE_CHECK(iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret), 0, out, bytes,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  iree_hal_buffer_view_release(arg0);
  iree_hal_buffer_view_release(arg1);
  iree_hal_buffer_view_release(ret);
  iree_runtime_call_deinitialize(&call);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  return 0;
}
