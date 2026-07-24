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

// Thin C shim over the prebuilt IREE runtime C API for the OpenXLA backend
// (issue #449 Phase 3). It loads the #451-emitted `prefill` and `decode_step`
// vmfbs into one IREE session, keeps the model weights resident as device
// buffers (uploaded once), threads the KV cache across decode steps, and returns
// a token id per step. The grown shape of the FFI gate (spike/iree-ffi), proven
// token-exact from Rust before being vendored here. Rust calls the flat C ABI
// (`xla_llama_*`) and never binds the IREE runtime structs directly.
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// The iree-dist build leaves the system allocator to the application (its
// iree_allocator_system() is gated on IREE_ALLOCATOR_SYSTEM_CTL). Point it at a
// libc malloc/free control function, defined below, before the IREE headers.
#define IREE_ALLOCATOR_SYSTEM_CTL iree_xla_libc_ctl

#include <iree/runtime/api.h>
#include <iree/hal/buffer_view_util.h>
#include <iree/hal/buffer_transfer.h>
// libc-backed implementation of the system allocator control function.
iree_status_t iree_xla_libc_ctl(void* self, iree_allocator_command_t command,
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

// Print the FULL annotated IREE status (op / dispatch location, message), not
// just the numeric code, so a runtime fault names its cause instead of an opaque
// "status 13" (issue #613: the metal-spirv packed-dequant prefill invoke).
#define XLA_CHECK(expr)                                                  \
  do {                                                                   \
    iree_status_t _s = (expr);                                          \
    if (!iree_status_is_ok(_s)) {                                       \
      int _c = (int)iree_status_code(_s);                               \
      fprintf(stderr, "xla_iree: %s failed (status %d):\n", #expr, _c); \
      iree_status_fprint(stderr, _s);                                   \
      iree_status_ignore(_s);                                           \
      return _c ? _c : 1;                                               \
    }                                                                    \
  } while (0)

#define XLA_CHECK_GOTO(expr, label, rc_var)                              \
  do {                                                                   \
    iree_status_t _s = (expr);                                           \
    if (!iree_status_is_ok(_s)) {                                        \
      int _c = (int)iree_status_code(_s);                                \
      fprintf(stderr, "xla_iree: %s failed (status %d):\n", #expr, _c); \
      iree_status_fprint(stderr, _s);                                    \
      iree_status_ignore(_s);                                            \
      rc_var = _c ? _c : 1;                                             \
      goto label;                                                        \
    }                                                                    \
  } while (0)

typedef struct xla_ctx {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;   // holds the compatible prefill/decode bundle
  iree_hal_allocator_t* allocator;   // device allocator (shared by all calls)
  uint64_t compatibility_fingerprint;
  int32_t n_weights;
  int32_t context_capacity;
  int32_t hidden_size;
  // Explicit position ABI selected when the bundle is created:
  // 0 = scalar/1D RoPE, 1 = M-RoPE temporal/height/width coordinates.
  int32_t position_mode;
  // 0 = ordinary `prefill_embeddings.main`, 1 = Gemma3n's distinct
  // `prefill_embeddings_ple.main` with a rank-3 dense PLE argument.
  int32_t prefill_embeddings_kind;
  int32_t dense_ple_layers;
  int32_t dense_ple_hidden;
  int32_t model_layers;
  int32_t deepstack_layers;
  int32_t deepstack_visual_positions;
  int32_t* deepstack_target_layers;
  int32_t has_deepstack;
  int32_t has_prefill_diagnostics;
  iree_hal_buffer_view_t** weights;  // resident weights, uploaded once
  iree_hal_buffer_view_t* kcache;    // threaded KV (set by prefill, advanced by decode)
  iree_hal_buffer_view_t* vcache;
  // Batched (continuous-batching) state, used by the ragged decode path
  // (#449 M3 Stage 2b). kcache_b/vcache_b are the rank-5 per-slot KV
  // [B,L,S,nkv,d] the ragged decode threads; a slot is seeded by a single-seq
  // prefill whose rank-4 KV is copied DEVICE-SIDE into the slot's region (no host
  // round-trip). rg_per is one slot's element count; rg_dims is its [L,S,nkv,d].
  iree_hal_buffer_view_t* kcache_b;
  iree_hal_buffer_view_t* vcache_b;
  int32_t rg_bsz;
  iree_host_size_t rg_per;
  iree_hal_dim_t rg_dims[4];
} xla_ctx;

// Explicit descriptor for host tensors entering the embeddings ABI. `dtype` is
// 0=f32 or 1=i32. Rust supplies every byte/rank/shape field and this shim repeats
// all safety-critical checks before constructing an IREE buffer view.
typedef struct xla_tensor_desc {
  const void* data;
  size_t byte_length;
  int32_t dtype;
  int32_t rank;
  int64_t dims[4];
} xla_tensor_desc;

// Forward declaration: xla_llama_create's error path frees a partial context.
void xla_llama_free(xla_ctx* c);

static const iree_hal_buffer_params_t kDeviceLocalParams = {
    .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    .access = IREE_HAL_MEMORY_ACCESS_ALL,
    .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
};

static int32_t xla_argmax_f32(const float* v, int32_t n) {
  int32_t best = 0;
  float bv = v[0];
  for (int32_t i = 1; i < n; i++) {
    if (v[i] > bv) {
      bv = v[i];
      best = i;
    }
  }
  return best;
}

// Read the next token from a model output buffer view, auto-detecting the
// sampling mode: a scalar i32 is an on-device argmax result (the Phase 2b
// pattern, 4-byte readback); a [V] f32 vector is raw logits, argmaxed on the
// host. The same shim therefore drives both the logits-returning and the
// on-device-argmax vmfbs.
static iree_status_t xla_read_token(xla_ctx* c, iree_hal_buffer_view_t* out,
                                    int32_t* out_token) {
  iree_host_size_t n = iree_hal_buffer_view_element_count(out);
  iree_hal_element_type_t et = iree_hal_buffer_view_element_type(out);
  if (n == 1 && et == IREE_HAL_ELEMENT_TYPE_INT_32) {
    int32_t tok = 0;
    IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
        c->device, iree_hal_buffer_view_buffer(out), 0, &tok, sizeof(int32_t),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    *out_token = tok;
    return iree_ok_status();
  }
  float* host = (float*)malloc(n * sizeof(float));
  if (!host) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_status_t s = iree_hal_device_transfer_d2h(
      c->device, iree_hal_buffer_view_buffer(out), 0, host, n * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  if (iree_status_is_ok(s)) *out_token = xla_argmax_f32(host, (int32_t)n);
  free(host);
  return s;
}

// Batched output readback (ragged decode): a [B] i32 buffer is an on-device
// per-row argmax (4*B-byte readback); a [B, V] f32 buffer is raw logits,
// argmaxed per row on the host. Fills out_tokens[0..bsz].
// Copy a model output's f32 logits to a host buffer (#449 M3 Stage 2d sampling).
// The ragged engine reads back the full per-row logit distribution and samples on
// the host (temperature / top-k / top-p), versus the on-device argmax variant
// that returns token ids. `count` is the expected element count (`[V]` for prefill,
// `[B*V]` for ragged decode).
static iree_status_t xla_read_logits(xla_ctx* c, iree_hal_buffer_view_t* out,
                                     float* host, iree_host_size_t count) {
  iree_host_size_t n = iree_hal_buffer_view_element_count(out);
  if (n != count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "logits element count mismatch");
  }
  return iree_hal_device_transfer_d2h(
      c->device, iree_hal_buffer_view_buffer(out), 0, host, n * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
}

// Diagnostics-only readback of one compact K/V slice per decoder layer.
//
// The normal prefill graph already returns the complete rank-4 caches
// [layers, sequence, kv_heads, head_dim]. Reading the final effective prompt
// position, head zero, and a caller-bounded prefix of head_dim lets the LLaVA
// reference harness compare every layer without compiling a second decoder or
// copying the full caches to host. Production callers pass kv_width=0 and never
// enter this branch.
static int xla_read_selected_kv(xla_ctx* c, iree_hal_buffer_view_t* kc,
                                iree_hal_buffer_view_t* vc, int32_t real_len,
                                int32_t kv_width, float* out_kv) {
  if (kv_width == 0 && out_kv == NULL) return 0;
  if (kv_width <= 0 || !out_kv || real_len <= 0 ||
      real_len > c->context_capacity) {
    fprintf(stderr, "xla_iree: invalid selected KV diagnostic contract\n");
    return 1;
  }
  iree_hal_dim_t layers = iree_hal_buffer_view_shape_dim(kc, 0);
  iree_hal_dim_t sequence = iree_hal_buffer_view_shape_dim(kc, 1);
  iree_hal_dim_t kv_heads = iree_hal_buffer_view_shape_dim(kc, 2);
  iree_hal_dim_t head_dim = iree_hal_buffer_view_shape_dim(kc, 3);
  if (layers != (iree_hal_dim_t)c->model_layers ||
      sequence != (iree_hal_dim_t)c->context_capacity || kv_heads <= 0 ||
      (iree_hal_dim_t)kv_width > head_dim) {
    fprintf(stderr,
            "xla_iree: selected KV dimensions disagree with runtime "
            "(layers=%lld sequence=%lld heads=%lld head_dim=%lld width=%d)\n",
            (long long)layers, (long long)sequence, (long long)kv_heads,
            (long long)head_dim, kv_width);
    return 1;
  }

  iree_hal_buffer_t* buffers[2] = {
      iree_hal_buffer_view_buffer(kc),
      iree_hal_buffer_view_buffer(vc),
  };
  for (iree_hal_dim_t layer = 0; layer < layers; ++layer) {
    uint64_t element_offset = (uint64_t)layer;
    if (element_offset > UINT64_MAX / (uint64_t)sequence) {
      fprintf(stderr, "xla_iree: selected KV layer offset overflowed\n");
      return 1;
    }
    element_offset *= (uint64_t)sequence;
    if (element_offset > UINT64_MAX - (uint64_t)(real_len - 1)) {
      fprintf(stderr, "xla_iree: selected KV position offset overflowed\n");
      return 1;
    }
    element_offset += (uint64_t)(real_len - 1);
    if (element_offset > UINT64_MAX / (uint64_t)kv_heads) {
      fprintf(stderr, "xla_iree: selected KV head offset overflowed\n");
      return 1;
    }
    element_offset *= (uint64_t)kv_heads;
    if (element_offset > UINT64_MAX / (uint64_t)head_dim) {
      fprintf(stderr, "xla_iree: selected KV dimension offset overflowed\n");
      return 1;
    }
    element_offset *= (uint64_t)head_dim;
    if (element_offset > UINT64_MAX / sizeof(float)) {
      fprintf(stderr, "xla_iree: selected KV byte offset overflowed\n");
      return 1;
    }
    iree_device_size_t byte_offset =
        (iree_device_size_t)(element_offset * sizeof(float));
    for (int32_t kind = 0; kind < 2; ++kind) {
      float* destination =
          out_kv + (((size_t)layer * 2u + (size_t)kind) * (size_t)kv_width);
      iree_status_t status = iree_hal_device_transfer_d2h(
          c->device, buffers[kind], byte_offset, destination,
          (iree_device_size_t)kv_width * sizeof(float),
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
      if (!iree_status_is_ok(status)) {
        int code = (int)iree_status_code(status);
        fprintf(stderr,
                "xla_iree: selected KV d2h failed at layer=%lld kind=%d "
                "(status %d):\n",
                (long long)layer, kind, code);
        iree_status_fprint(stderr, status);
        iree_status_ignore(status);
        return code ? code : 1;
      }
    }
  }
  return 0;
}

// Allocate a resident device buffer view of the given shape/elt and copy bytes.
static iree_status_t xla_alloc_bv(xla_ctx* c, iree_host_size_t rank,
                                  const iree_hal_dim_t* shape,
                                  iree_hal_element_type_t elt, const void* data,
                                  iree_host_size_t nbytes,
                                  iree_hal_buffer_view_t** out) {
  return iree_hal_buffer_view_allocate_buffer_copy(
      c->device, c->allocator, rank, shape, elt,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, kDeviceLocalParams,
      iree_make_const_byte_span(data, nbytes), out);
}

static int xla_validate_tensor_desc(const char* name,
                                    const xla_tensor_desc* desc,
                                    int32_t expected_dtype,
                                    int32_t expected_rank,
                                    const int64_t* expected_dims) {
  if (!desc || !desc->data || desc->dtype != expected_dtype ||
      desc->rank != expected_rank || expected_rank < 0 || expected_rank > 4) {
    fprintf(stderr,
            "xla_iree: invalid %s descriptor pointer/dtype/rank (want dtype=%d rank=%d)\n",
            name, expected_dtype, expected_rank);
    return 1;
  }
  size_t element_size = expected_dtype == 0 ? sizeof(float) : sizeof(int32_t);
  size_t elements = 1;
  for (int32_t axis = 0; axis < expected_rank; ++axis) {
    int64_t dim = desc->dims[axis];
    if (dim <= 0 || dim != expected_dims[axis] ||
        (size_t)dim > SIZE_MAX / elements) {
      fprintf(stderr,
              "xla_iree: invalid %s descriptor axis %d=%lld (want %lld)\n",
              name, axis, (long long)dim, (long long)expected_dims[axis]);
      return 1;
    }
    elements *= (size_t)dim;
  }
  if (elements > SIZE_MAX / element_size) {
    fprintf(stderr, "xla_iree: %s descriptor byte count overflowed\n", name);
    return 1;
  }
  size_t expected_bytes = elements * element_size;
  if (desc->byte_length != expected_bytes) {
    fprintf(stderr,
            "xla_iree: invalid %s descriptor byte_length=%zu (want %zu)\n",
            name, desc->byte_length, expected_bytes);
    return 1;
  }
  return 0;
}

static iree_status_t xla_alloc_desc_bv(xla_ctx* c,
                                       const xla_tensor_desc* desc,
                                       iree_hal_buffer_view_t** out) {
  iree_hal_dim_t shape[4];
  for (int32_t axis = 0; axis < desc->rank; ++axis) {
    shape[axis] = (iree_hal_dim_t)desc->dims[axis];
  }
  iree_hal_element_type_t element_type =
      desc->dtype == 0 ? IREE_HAL_ELEMENT_TYPE_FLOAT_32
                       : IREE_HAL_ELEMENT_TYPE_INT_32;
  return xla_alloc_bv(c, (iree_host_size_t)desc->rank, shape, element_type,
                      desc->data, (iree_host_size_t)desc->byte_length, out);
}

// Validate the static KV contract shared by the paired prefill/decode modules.
// `context_axis` is 1 for rank-4 [L,S,nkv,d] and 2 for rank-5 [B,L,S,nkv,d].
static int xla_validate_kv_pair(xla_ctx* c, iree_hal_buffer_view_t* kc,
                                iree_hal_buffer_view_t* vc,
                                iree_host_size_t rank,
                                iree_host_size_t context_axis) {
  if (!kc || !vc || iree_hal_buffer_view_shape_rank(kc) != rank ||
      iree_hal_buffer_view_shape_rank(vc) != rank ||
      iree_hal_buffer_view_element_type(kc) !=
          IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
      iree_hal_buffer_view_element_type(vc) !=
          IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
      iree_hal_buffer_view_encoding_type(kc) !=
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR ||
      iree_hal_buffer_view_encoding_type(vc) !=
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    fprintf(stderr,
            "xla_iree: expected matching dense-row-major rank-%zu f32 K/V buffers\n",
            (size_t)rank);
    return 1;
  }
  for (iree_host_size_t i = 0; i < rank; ++i) {
    iree_hal_dim_t kd = iree_hal_buffer_view_shape_dim(kc, i);
    iree_hal_dim_t vd = iree_hal_buffer_view_shape_dim(vc, i);
    if (kd != vd) {
      fprintf(stderr, "xla_iree: K/V shape mismatch at axis %zu (%lld vs %lld)\n",
              (size_t)i, (long long)kd, (long long)vd);
      return 1;
    }
  }
  iree_hal_dim_t actual =
      iree_hal_buffer_view_shape_dim(kc, context_axis);
  if (actual != (iree_hal_dim_t)c->context_capacity) {
    fprintf(stderr,
            "xla_iree: module KV context dimension %lld disagrees with configured context_capacity=%d\n",
            (long long)actual, c->context_capacity);
    return 1;
  }
  return 0;
}

static iree_status_t xla_llama_create_impl(
    xla_ctx* c, const char* device_uri, const char* prefill_vmfb,
    const char* prefill_embeddings_vmfb,
    const char* prefill_embeddings_deepstack_vmfb, const char* decode_vmfb,
    const char* prefill_diagnostics_vmfb,
    uint64_t compatibility_fingerprint, int32_t n_weights,
    const void* const* weight_data,
    const int32_t* weight_dtypes, const int32_t* weight_ranks,
    const int64_t* weight_dims, int32_t context_capacity,
    int32_t hidden_size, int32_t position_mode,
    int32_t prefill_embeddings_kind,
    int32_t dense_ple_layers, int32_t dense_ple_hidden, int32_t model_layers,
    int32_t deepstack_layers, int32_t deepstack_visual_positions,
    const int32_t* deepstack_target_layers) {
  if (context_capacity <= 0 || hidden_size <= 0 ||
      compatibility_fingerprint == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bundle fingerprint, context_capacity, and hidden_size must be positive");
  }
  if (position_mode != 0 && position_mode != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "position_mode must be 0 (1D) or 1 (M-RoPE 3D)");
  }
  if ((prefill_embeddings_kind != 0 && prefill_embeddings_kind != 1) ||
      (prefill_embeddings_kind == 0 &&
       (dense_ple_layers != 0 || dense_ple_hidden != 0)) ||
      (prefill_embeddings_kind == 1 &&
       (dense_ple_layers <= 0 || dense_ple_hidden <= 0))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid embeddings entry kind or dense PLE dimensions");
  }
  bool has_deepstack = prefill_embeddings_deepstack_vmfb &&
                       prefill_embeddings_deepstack_vmfb[0] != '\0';
  if (model_layers <= 0 ||
      (has_deepstack &&
       (prefill_embeddings_kind != 0 || deepstack_layers <= 0 ||
        deepstack_visual_positions <= 0 || !deepstack_target_layers)) ||
      (!has_deepstack &&
       (deepstack_layers != 0 || deepstack_visual_positions != 0 ||
        deepstack_target_layers != NULL))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid optional DeepStack bundle contract");
  }
  for (int32_t i = 0; i < deepstack_layers; ++i) {
    int32_t layer = deepstack_target_layers[i];
    if (layer < 0 || layer >= model_layers ||
        (i > 0 && layer <= deepstack_target_layers[i - 1])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "DeepStack target layers must be sorted, unique, and in model bounds");
    }
  }
  c->n_weights = n_weights;
  c->context_capacity = context_capacity;
  c->hidden_size = hidden_size;
  c->position_mode = position_mode;
  c->compatibility_fingerprint = compatibility_fingerprint;
  c->prefill_embeddings_kind = prefill_embeddings_kind;
  c->dense_ple_layers = dense_ple_layers;
  c->dense_ple_hidden = dense_ple_hidden;
  c->model_layers = model_layers;
  c->deepstack_layers = deepstack_layers;
  c->deepstack_visual_positions = deepstack_visual_positions;
  c->has_deepstack = has_deepstack;
  if (has_deepstack) {
    c->deepstack_target_layers =
        (int32_t*)malloc((size_t)deepstack_layers * sizeof(int32_t));
    if (!c->deepstack_target_layers) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    memcpy(c->deepstack_target_layers, deepstack_target_layers,
           (size_t)deepstack_layers * sizeof(int32_t));
  }
  c->has_prefill_diagnostics =
      prefill_diagnostics_vmfb && prefill_diagnostics_vmfb[0] != '\0';

  iree_runtime_instance_options_t inst_opts;
  iree_runtime_instance_options_initialize(&inst_opts);
  // The unified runtime's register-all module owns both CUDA and local-task.
  // Pre-registering CUDA here makes register-all stop at the duplicate entry
  // before local-task is reached, which silently breaks CPU parity runs in a
  // CUDA-enabled build.
  iree_runtime_instance_options_use_all_available_drivers(&inst_opts);
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
      &inst_opts, iree_allocator_system(), &c->instance));

  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      c->instance, iree_make_cstring_view(device_uri), &c->device));

  iree_runtime_session_options_t sess_opts;
  iree_runtime_session_options_initialize(&sess_opts);
  IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
      c->instance, &sess_opts, c->device,
      iree_runtime_instance_host_allocator(c->instance), &c->session));

  // All bundle members enter one session atomically. Their distinct names make
  // the calls "prefill.main", "prefill_embeddings.main", and
  // "decode_step.main".
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      c->session, prefill_vmfb));
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      c->session, prefill_embeddings_vmfb));
  if (c->has_deepstack) {
    IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
        c->session, prefill_embeddings_deepstack_vmfb));
  }
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      c->session, decode_vmfb));
  if (c->has_prefill_diagnostics) {
    IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
        c->session, prefill_diagnostics_vmfb));
  }

  const char* embeddings_entry = prefill_embeddings_kind == 1
                                      ? "prefill_embeddings_ple.main"
                                      : "prefill_embeddings.main";
  const char* required_entries[] = {"prefill.main", embeddings_entry,
                                    "decode_step.main"};
  for (size_t i = 0; i < 3; ++i) {
    iree_runtime_call_t probe;
    IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
        c->session, iree_make_cstring_view(required_entries[i]), &probe));
    iree_runtime_call_deinitialize(&probe);
  }
  if (c->has_deepstack) {
    iree_runtime_call_t probe;
    IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
        c->session,
        iree_make_cstring_view("prefill_embeddings_deepstack.main"), &probe));
    iree_runtime_call_deinitialize(&probe);
  }
  if (c->has_prefill_diagnostics) {
    iree_runtime_call_t probe;
    IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
        c->session, iree_make_cstring_view("prefill_diagnostics.main"), &probe));
    iree_runtime_call_deinitialize(&probe);
  }

  c->allocator = iree_runtime_session_device_allocator(c->session);

  c->weights = (iree_hal_buffer_view_t**)calloc(
      (size_t)n_weights, sizeof(iree_hal_buffer_view_t*));
  if (!c->weights) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  for (int32_t i = 0; i < n_weights; i++) {
    int32_t rank = weight_ranks[i];
    iree_hal_dim_t shape[4];
    iree_host_size_t nelem = 1;
    for (int32_t d = 0; d < rank; d++) {
      iree_hal_dim_t dim = (iree_hal_dim_t)weight_dims[i * 4 + d];
      shape[d] = dim;
      nelem *= (iree_host_size_t)dim;
    }
    // issue #516 per-weight dtype: f32 (0, a widened / dequantized weight) or the
    // raw parts of an MLX affine-quantized projection, f16 (1) scales / biases or a
    // packed U32 (2) weight the graph dequantizes in place.
    iree_hal_element_type_t elt;
    iree_host_size_t esize;
    switch (weight_dtypes[i]) {
      case 0:
        elt = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
        esize = 4;
        break;
      case 1:
        elt = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
        esize = 2;
        break;
      case 2:
        elt = IREE_HAL_ELEMENT_TYPE_UINT_32;
        esize = 4;
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown weight dtype %d for weight %d",
                                weight_dtypes[i], i);
    }
    IREE_RETURN_IF_ERROR(xla_alloc_bv(c, (iree_host_size_t)rank, shape, elt,
                                      weight_data[i], nelem * esize,
                                      &c->weights[i]));
  }
  return iree_ok_status();
}

// Create the execution context. `device_uri` is "local-task" (CPU) or another
// registered HAL driver. `weight_data[i]` points at `prod(weight_dims[i*4..])`
// elements of dtype `weight_dtypes[i]` (0 f32, 1 f16, 2 packed U32; issue #516)
// laid out row-major; the shim copies them into resident device buffers, so the
// caller may free them after this returns. Returns NULL on failure (after printing
// a diagnostic).
xla_ctx* xla_llama_create(const char* device_uri, const char* prefill_vmfb,
                          const char* prefill_embeddings_vmfb,
                          const char* prefill_embeddings_deepstack_vmfb,
                          const char* decode_vmfb,
                          const char* prefill_diagnostics_vmfb,
                          uint64_t compatibility_fingerprint,
                          int32_t n_weights,
                          const void* const* weight_data,
                          const int32_t* weight_dtypes,
                          const int32_t* weight_ranks,
                          const int64_t* weight_dims,
                          int32_t context_capacity, int32_t hidden_size,
                          int32_t position_mode,
                          int32_t prefill_embeddings_kind,
                          int32_t dense_ple_layers,
                          int32_t dense_ple_hidden, int32_t model_layers,
                          int32_t deepstack_layers,
                          int32_t deepstack_visual_positions,
                          const int32_t* deepstack_target_layers) {
  xla_ctx* c = (xla_ctx*)calloc(1, sizeof(xla_ctx));
  if (!c) return NULL;
  iree_status_t s =
      xla_llama_create_impl(c, device_uri, prefill_vmfb,
                            prefill_embeddings_vmfb,
                            prefill_embeddings_deepstack_vmfb, decode_vmfb,
                            prefill_diagnostics_vmfb,
                            compatibility_fingerprint, n_weights, weight_data,
                            weight_dtypes, weight_ranks, weight_dims,
                            context_capacity, hidden_size,
                            position_mode,
                            prefill_embeddings_kind, dense_ple_layers,
                            dense_ple_hidden, model_layers, deepstack_layers,
                            deepstack_visual_positions,
                            deepstack_target_layers);
  if (!iree_status_is_ok(s)) {
    char buf[512];
    iree_host_size_t got = 0;
    iree_status_format(s, sizeof(buf), buf, &got);
    fprintf(stderr, "xla_llama_create failed: %.*s (code %d)\n", (int)got, buf,
            (int)iree_status_code(s));
    iree_status_ignore(s);
    xla_llama_free(c);
    return NULL;
  }
  return c;
}

// prefill: tokens[lp], positions[lp] or positions[3, lp], real_len -> first token
// id; the returned
// KV cache becomes the resident cache decode threads forward.
int xla_llama_prefill(xla_ctx* c, const int32_t* tokens, int32_t lp,
                      const int32_t* positions, int32_t real_len,
                      int32_t* out_token) {
  if (lp != c->context_capacity || real_len < 0 || real_len > lp) {
    fprintf(stderr,
            "xla_llama_prefill: lp=%d real_len=%d context_capacity=%d\n", lp,
            real_len, c->context_capacity);
    return 1;
  }
  iree_runtime_call_t call;
  XLA_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("prefill.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    XLA_CHECK(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]));
  }
  iree_hal_dim_t seq_shape[1] = {(iree_hal_dim_t)lp};
  iree_hal_dim_t position_shape[2] = {3, (iree_hal_dim_t)lp};
  iree_host_size_t seq_bytes = (iree_host_size_t)lp * sizeof(int32_t);
  iree_host_size_t position_bytes =
      seq_bytes * (c->position_mode == 1 ? 3 : 1);
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  XLA_CHECK(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                         seq_bytes, &tok_bv));
  XLA_CHECK(xla_alloc_bv(c, c->position_mode == 1 ? 2 : 1,
                         c->position_mode == 1 ? position_shape : seq_shape,
                         IREE_HAL_ELEMENT_TYPE_INT_32, positions,
                         position_bytes, &pos_bv));
  XLA_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &real_len,
                         sizeof(int32_t), &len_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv));

  XLA_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* token_out = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &token_out));
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc));
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc));
  if (xla_validate_kv_pair(c, kc, vc, 4, 1) != 0) {
    iree_hal_buffer_view_release(token_out);
    iree_hal_buffer_view_release(kc);
    iree_hal_buffer_view_release(vc);
    iree_hal_buffer_view_release(tok_bv);
    iree_hal_buffer_view_release(pos_bv);
    iree_hal_buffer_view_release(len_bv);
    iree_runtime_call_deinitialize(&call);
    return 1;
  }
  if (c->kcache) iree_hal_buffer_view_release(c->kcache);
  if (c->vcache) iree_hal_buffer_view_release(c->vcache);
  c->kcache = kc;
  c->vcache = vc;

  XLA_CHECK(xla_read_token(c, token_out, out_token));

  iree_hal_buffer_view_release(token_out);
  iree_hal_buffer_view_release(tok_bv);
  iree_hal_buffer_view_release(pos_bv);
  iree_hal_buffer_view_release(len_bv);
  iree_runtime_call_deinitialize(&call);
  return 0;
}

// Shared decode implementation. `position_mode` is an explicit ABI contract,
// never inferred from the position buffer rank.
static int xla_llama_decode_impl(xla_ctx* c, int32_t token,
                                 const int32_t* positions,
                                 int32_t position_mode, int32_t cache_len,
                                 int32_t* out_token) {
  if (!c->kcache || !c->vcache || c->position_mode != position_mode ||
      cache_len < 0 || cache_len >= c->context_capacity) {
    fprintf(stderr,
            "xla_llama_decode: mode=%d bundle_mode=%d cache_len=%d context_capacity=%d or KV is uninitialized\n",
            position_mode, c->position_mode, cache_len, c->context_capacity);
    return 1;
  }
  int32_t position_count = position_mode == 1 ? 3 : 1;
  for (int32_t axis = 0; axis < position_count; ++axis) {
    if (positions[axis] < 0) {
      fprintf(stderr, "xla_llama_decode: position axis %d is negative (%d)\n",
              axis, positions[axis]);
      return 1;
    }
  }
  iree_runtime_call_t call;
  XLA_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("decode_step.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    XLA_CHECK(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]));
  }
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  XLA_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &token,
                         sizeof(int32_t), &tok_bv));
  iree_hal_dim_t position_shape[1] = {3};
  XLA_CHECK(xla_alloc_bv(c, position_mode == 1 ? 1 : 0,
                         position_mode == 1 ? position_shape : NULL,
                         IREE_HAL_ELEMENT_TYPE_INT_32, positions,
                         (iree_host_size_t)position_count * sizeof(int32_t),
                         &pos_bv));
  XLA_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &cache_len,
                         sizeof(int32_t), &len_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->kcache));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->vcache));

  XLA_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* token_out = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &token_out));
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc));
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc));
  if (xla_validate_kv_pair(c, kc, vc, 4, 1) != 0) {
    iree_hal_buffer_view_release(token_out);
    iree_hal_buffer_view_release(kc);
    iree_hal_buffer_view_release(vc);
    iree_hal_buffer_view_release(tok_bv);
    iree_hal_buffer_view_release(pos_bv);
    iree_hal_buffer_view_release(len_bv);
    iree_runtime_call_deinitialize(&call);
    return 1;
  }
  iree_hal_buffer_view_t* old_k = c->kcache;
  iree_hal_buffer_view_t* old_v = c->vcache;
  c->kcache = kc;
  c->vcache = vc;

  XLA_CHECK(xla_read_token(c, token_out, out_token));

  iree_hal_buffer_view_release(token_out);
  iree_hal_buffer_view_release(tok_bv);
  iree_hal_buffer_view_release(pos_bv);
  iree_hal_buffer_view_release(len_bv);
  iree_runtime_call_deinitialize(&call);
  // The old KV was an input to this call; the call held a ref during invoke and
  // dropped it at deinitialize, so release our own ref now.
  iree_hal_buffer_view_release(old_k);
  iree_hal_buffer_view_release(old_v);
  return 0;
}

int xla_llama_decode(xla_ctx* c, int32_t token, int32_t pos,
                     int32_t cache_len, int32_t* out_token) {
  return xla_llama_decode_impl(c, token, &pos, 0, cache_len, out_token);
}

int xla_llama_decode_mrope(xla_ctx* c, int32_t token,
                           const int32_t positions[3], int32_t cache_len,
                           int32_t* out_token) {
  return xla_llama_decode_impl(c, token, positions, 1, cache_len, out_token);
}

// ===========================================================================
// Ragged continuous-batching decode (#449 M3 Stage 2b). `B_max` slots share one
// rank-5 KV [B,L,S,nkv,d]; each slot carries its OWN token/pos/cache_len, so
// sequences of different lengths decode together. A request is seeded into a
// slot by the single-seq prefill, whose rank-4 KV is copied DEVICE-SIDE into the
// slot's region of the rank-5 cache (no host round-trip; only that slot's bytes
// move, so live slots are untouched). This is the productized engine path; the
// spike (spike/iree-ffi) proved the graph + scheduler with a host-mirror admit
// first, which 2b replaces with this device-side slot write.
// ===========================================================================

// Ensure the rank-5 per-slot KV exists, allocating it zeroed on first use from
// the rank-4 dims the first prefill_slot captured. Inactive slots stay zeroed
// until a prefill_slot overwrites their region; the ragged decode's per-row mask
// makes a zeroed (cache_len 0) slot a harmless no-op whose output is discarded.
static iree_status_t xla_ensure_batch_kv(xla_ctx* c) {
  if (c->kcache_b && c->vcache_b) return iree_ok_status();
  if (c->rg_bsz <= 0 || c->rg_per == 0 ||
      (iree_host_size_t)c->rg_bsz > SIZE_MAX / c->rg_per) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "batch KV element count overflowed");
  }
  iree_host_size_t count = (iree_host_size_t)c->rg_bsz * c->rg_per;
  if (count > SIZE_MAX / sizeof(float)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "batch KV byte count overflowed");
  }
  iree_host_size_t bytes = count * sizeof(float);
  float* zeros = (float*)calloc((size_t)count, sizeof(float));
  if (!zeros) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_hal_dim_t shape5[5] = {(iree_hal_dim_t)c->rg_bsz, c->rg_dims[0],
                              c->rg_dims[1], c->rg_dims[2], c->rg_dims[3]};
  iree_status_t s = xla_alloc_bv(c, 5, shape5, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                 zeros, bytes, &c->kcache_b);
  if (iree_status_is_ok(s)) {
    s = xla_alloc_bv(c, 5, shape5, IREE_HAL_ELEMENT_TYPE_FLOAT_32, zeros, bytes,
                     &c->vcache_b);
  }
  free(zeros);
  if (!iree_status_is_ok(s)) {
    if (c->kcache_b) {
      iree_hal_buffer_view_release(c->kcache_b);
      c->kcache_b = NULL;
    }
    if (c->vcache_b) {
      iree_hal_buffer_view_release(c->vcache_b);
      c->vcache_b = NULL;
    }
  }
  return s;
}

// Copy one rank-4 prefill KV pair directly into the selected region of the
// resident rank-5 batch cache. Token and embeddings prefill both use this exact
// device-side path; no KV bytes cross host memory.
static int xla_store_prefill_kv_slot(xla_ctx* c, int32_t slot,
                                     iree_hal_buffer_view_t* kc,
                                     iree_hal_buffer_view_t* vc) {
  if (slot < 0 || slot >= c->rg_bsz ||
      xla_validate_kv_pair(c, kc, vc, 4, 1) != 0) {
    return 1;
  }
  iree_host_size_t per = iree_hal_buffer_view_element_count(kc);
  if (c->rg_per == 0) {
    c->rg_per = per;
    for (int32_t i = 0; i < 4; ++i) {
      c->rg_dims[i] = iree_hal_buffer_view_shape_dim(kc, i);
    }
  } else {
    if (per != c->rg_per) {
      fprintf(stderr, "xla_iree: prefill KV element count changed\n");
      return 1;
    }
    for (int32_t i = 0; i < 4; ++i) {
      if (c->rg_dims[i] != iree_hal_buffer_view_shape_dim(kc, i)) {
        fprintf(stderr, "xla_iree: prefill KV shape changed at axis %d\n", i);
        return 1;
      }
    }
  }
  XLA_CHECK(xla_ensure_batch_kv(c));
  if (per > IREE_DEVICE_SIZE_MAX / sizeof(float)) {
    fprintf(stderr, "xla_iree: prefill KV byte count overflowed\n");
    return 1;
  }
  iree_device_size_t len = (iree_device_size_t)per * sizeof(float);
  if ((iree_device_size_t)slot > IREE_DEVICE_SIZE_MAX / len) {
    fprintf(stderr, "xla_iree: prefill KV slot offset overflowed\n");
    return 1;
  }
  iree_device_size_t off = (iree_device_size_t)slot * len;
  XLA_CHECK(iree_hal_device_transfer_d2d(
      c->device, iree_hal_buffer_view_buffer(kc), 0,
      iree_hal_buffer_view_buffer(c->kcache_b), off, len,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  XLA_CHECK(iree_hal_device_transfer_d2d(
      c->device, iree_hal_buffer_view_buffer(vc), 0,
      iree_hal_buffer_view_buffer(c->vcache_b), off, len,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  return 0;
}

// Reset the batched state for `bsz` slots: drop any rank-5 KV and forget the
// per-slot dims (re-captured by the next prefill_slot, which lazily reallocates
// the rank-5 KV once the dims are known). Call once before seeding slots.
int xla_llama_ragged_reset(xla_ctx* c, int32_t bsz) {
  if (bsz <= 0) {
    fprintf(stderr, "xla_llama_ragged_reset: bsz must be positive\n");
    return 1;
  }
  c->rg_bsz = bsz;
  c->rg_per = 0;
  if (c->kcache_b) {
    iree_hal_buffer_view_release(c->kcache_b);
    c->kcache_b = NULL;
  }
  if (c->vcache_b) {
    iree_hal_buffer_view_release(c->vcache_b);
    c->vcache_b = NULL;
  }
  return 0;
}

// Seed slot `slot` with a prompt and return its first-token LOGITS (#449 M3
// Stage 2d sampling). Runs the logits prefill graph (returns `[vocab]` logits +
// rank-4 KV), copies the logits to host `out_logits`, then copies the prompt's KV
// DEVICE-SIDE into the slot's region of the rank-5 cache: only this slot's bytes
// move (offset slot*per), so live slots are not disturbed and there is no host
// round-trip. The caller (engine) samples the first token from `out_logits`.
int xla_llama_prefill_slot_logits(xla_ctx* c, int32_t slot, const int32_t* tokens,
                                  int32_t lp, const int32_t* positions,
                                  int32_t real_len, int32_t vocab,
                                  float* out_logits) {
  if (slot < 0 || slot >= c->rg_bsz) {
    fprintf(stderr, "xla_llama_prefill_slot_logits: slot %d out of range [0,%d)\n",
            slot, c->rg_bsz);
    return 1;
  }
  if (lp != c->context_capacity || real_len <= 0 || real_len > lp) {
    fprintf(stderr,
            "xla_llama_prefill_slot_logits: lp=%d real_len=%d context_capacity=%d\n",
            lp, real_len, c->context_capacity);
    return 1;
  }
  iree_runtime_call_t call;
  XLA_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("prefill.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    XLA_CHECK(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]));
  }
  iree_hal_dim_t seq_shape[1] = {(iree_hal_dim_t)lp};
  iree_hal_dim_t position_shape[2] = {3, (iree_hal_dim_t)lp};
  iree_host_size_t seq_bytes = (iree_host_size_t)lp * sizeof(int32_t);
  iree_host_size_t position_bytes =
      seq_bytes * (c->position_mode == 1 ? 3 : 1);
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  XLA_CHECK(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                         seq_bytes, &tok_bv));
  XLA_CHECK(xla_alloc_bv(c, c->position_mode == 1 ? 2 : 1,
                         c->position_mode == 1 ? position_shape : seq_shape,
                         IREE_HAL_ELEMENT_TYPE_INT_32, positions,
                         position_bytes, &pos_bv));
  XLA_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &real_len,
                         sizeof(int32_t), &len_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv));

  XLA_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* logits = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &logits));
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc));
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc));
  if (xla_validate_kv_pair(c, kc, vc, 4, 1) != 0) {
    iree_hal_buffer_view_release(logits);
    iree_hal_buffer_view_release(kc);
    iree_hal_buffer_view_release(vc);
    iree_hal_buffer_view_release(tok_bv);
    iree_hal_buffer_view_release(pos_bv);
    iree_hal_buffer_view_release(len_bv);
    iree_runtime_call_deinitialize(&call);
    return 1;
  }
  iree_status_t rs =
      xla_read_logits(c, logits, out_logits, (iree_host_size_t)vocab);
  iree_hal_buffer_view_release(logits);
  iree_hal_buffer_view_release(tok_bv);
  iree_hal_buffer_view_release(pos_bv);
  iree_hal_buffer_view_release(len_bv);
  iree_runtime_call_deinitialize(&call);
  if (!iree_status_is_ok(rs)) {
    int code = (int)iree_status_code(rs);
    iree_status_ignore(rs);
    iree_hal_buffer_view_release(kc);
    iree_hal_buffer_view_release(vc);
    return code ? code : 1;
  }
  int copy_rc = xla_store_prefill_kv_slot(c, slot, kc, vc);
  iree_hal_buffer_view_release(kc);
  iree_hal_buffer_view_release(vc);
  return copy_rc;
}

// Test-only Gemma3n intermediate-oracle entry. The optional diagnostic module
// is never loaded by normal bundles. Its first output is one statically laid
// out f32 vector; K/V are handled exactly like ordinary slot prefill so the
// same loaded context can continue through decode and greedy validation.
int xla_llama_prefill_diagnostics_slot(
    xla_ctx* c, int32_t slot, const int32_t* tokens, int32_t lp,
    const int32_t* positions, int32_t real_len, int32_t diagnostic_len,
    float* out_diagnostics) {
  if (!c || !c->has_prefill_diagnostics || slot < 0 || slot >= c->rg_bsz ||
      lp != c->context_capacity || real_len <= 0 || real_len > lp ||
      diagnostic_len <= 0 || !out_diagnostics) {
    fprintf(stderr,
            "xla_llama_prefill_diagnostics_slot: invalid diagnostic bundle/input contract\n");
    return 1;
  }
  int rc = 0;
  bool call_initialized = false;
  iree_runtime_call_t call;
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  iree_hal_buffer_view_t* output = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  XLA_CHECK_GOTO(iree_runtime_call_initialize_by_name(
                     c->session,
                     iree_make_cstring_view("prefill_diagnostics.main"), &call),
                 cleanup, rc);
  call_initialized = true;
  for (int32_t i = 0; i < c->n_weights; ++i) {
    XLA_CHECK_GOTO(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]),
        cleanup, rc);
  }
  iree_hal_dim_t seq_shape[1] = {(iree_hal_dim_t)lp};
  iree_hal_dim_t position_shape[2] = {3, (iree_hal_dim_t)lp};
  iree_host_size_t seq_bytes = (iree_host_size_t)lp * sizeof(int32_t);
  iree_host_size_t position_bytes =
      seq_bytes * (c->position_mode == 1 ? 3 : 1);
  XLA_CHECK_GOTO(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32,
                             tokens, seq_bytes, &tok_bv),
                 cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_bv(c, c->position_mode == 1 ? 2 : 1,
                             c->position_mode == 1 ? position_shape : seq_shape,
                             IREE_HAL_ELEMENT_TYPE_INT_32, positions,
                             position_bytes, &pos_bv),
                 cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32,
                             &real_len, sizeof(int32_t), &len_bv),
                 cleanup, rc);
  XLA_CHECK_GOTO(
      iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv), cleanup,
      rc);
  XLA_CHECK_GOTO(
      iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv), cleanup,
      rc);
  XLA_CHECK_GOTO(
      iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv), cleanup,
      rc);
  XLA_CHECK_GOTO(iree_runtime_call_invoke(&call, 0), cleanup, rc);
  XLA_CHECK_GOTO(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &output), cleanup,
      rc);
  XLA_CHECK_GOTO(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc),
                 cleanup, rc);
  XLA_CHECK_GOTO(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc),
                 cleanup, rc);
  if (xla_validate_kv_pair(c, kc, vc, 4, 1) != 0) {
    rc = 1;
    goto cleanup;
  }
  XLA_CHECK_GOTO(
      xla_read_logits(c, output, out_diagnostics,
                      (iree_host_size_t)diagnostic_len),
      cleanup, rc);
  rc = xla_store_prefill_kv_slot(c, slot, kc, vc);

cleanup:
  if (output) iree_hal_buffer_view_release(output);
  if (kc) iree_hal_buffer_view_release(kc);
  if (vc) iree_hal_buffer_view_release(vc);
  if (tok_bv) iree_hal_buffer_view_release(tok_bv);
  if (pos_bv) iree_hal_buffer_view_release(pos_bv);
  if (len_bv) iree_hal_buffer_view_release(len_bv);
  if (call_initialized) iree_runtime_call_deinitialize(&call);
  return rc;
}

static int xla_llama_prefill_embeddings_impl(
    xla_ctx* c, int32_t slot, int32_t position_mode,
    const xla_tensor_desc* embeddings,
    const xla_tensor_desc* dense_ple,
    const xla_tensor_desc* positions, const xla_tensor_desc* attention_bias,
    int32_t real_len, int32_t vocab, int32_t* out_token, float* out_logits,
    int32_t kv_width, float* out_kv) {
  if (!c || real_len <= 0 || real_len > c->context_capacity ||
      position_mode != c->position_mode) {
    fprintf(stderr,
            "xla_llama_prefill_embeddings: real_len=%d context_capacity=%d mode=%d bundle_mode=%d\n",
            real_len, c ? c->context_capacity : 0, position_mode,
            c ? c->position_mode : -1);
    return 1;
  }
  int64_t embeddings_dims[2] = {c->context_capacity, c->hidden_size};
  int64_t positions_1d_dims[1] = {c->context_capacity};
  int64_t positions_mrope_dims[2] = {3, c->context_capacity};
  int64_t bias_dims[2] = {c->context_capacity, c->context_capacity};
  int64_t dense_ple_dims[3] = {
      c->context_capacity, c->dense_ple_layers, c->dense_ple_hidden};
  if (xla_validate_tensor_desc("embeddings", embeddings, 0, 2,
                               embeddings_dims) != 0 ||
      xla_validate_tensor_desc(
          "positions", positions, 1, position_mode == 1 ? 2 : 1,
          position_mode == 1 ? positions_mrope_dims : positions_1d_dims) !=
          0 ||
      xla_validate_tensor_desc("attention_bias", attention_bias, 0, 2,
                               bias_dims) != 0) {
    return 1;
  }
  if ((c->prefill_embeddings_kind == 0 && dense_ple != NULL) ||
      (c->prefill_embeddings_kind == 1 &&
       xla_validate_tensor_desc("dense_ple", dense_ple, 0, 3,
                                dense_ple_dims) != 0)) {
    fprintf(stderr,
            "xla_llama_prefill_embeddings: dense PLE does not match bundle kind\n");
    return 1;
  }
  bool batched = slot >= 0;
  if ((!batched && !out_token) ||
      (batched && (slot >= c->rg_bsz || vocab <= 0 || !out_logits))) {
    fprintf(stderr,
            "xla_llama_prefill_embeddings: invalid slot/output/vocab contract\n");
    return 1;
  }

  int rc = 0;
  bool call_initialized = false;
  iree_runtime_call_t call;
  iree_hal_buffer_view_t* embeddings_bv = NULL;
  iree_hal_buffer_view_t* dense_ple_bv = NULL;
  iree_hal_buffer_view_t* positions_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  iree_hal_buffer_view_t* bias_bv = NULL;
  iree_hal_buffer_view_t* output = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;

  XLA_CHECK_GOTO(iree_runtime_call_initialize_by_name(
                     c->session,
                     iree_make_cstring_view(
                         c->prefill_embeddings_kind == 1
                             ? "prefill_embeddings_ple.main"
                             : "prefill_embeddings.main"),
                     &call),
                 cleanup, rc);
  call_initialized = true;
  for (int32_t i = 0; i < c->n_weights; ++i) {
    XLA_CHECK_GOTO(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]),
        cleanup, rc);
  }
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, embeddings, &embeddings_bv), cleanup, rc);
  if (dense_ple) {
    XLA_CHECK_GOTO(xla_alloc_desc_bv(c, dense_ple, &dense_ple_bv), cleanup, rc);
  }
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, positions, &positions_bv), cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32,
                             &real_len, sizeof(int32_t), &len_bv),
                 cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, attention_bias, &bias_bv), cleanup, rc);
  XLA_CHECK_GOTO(
      iree_runtime_call_inputs_push_back_buffer_view(&call, embeddings_bv),
      cleanup, rc);
  if (dense_ple_bv) {
    XLA_CHECK_GOTO(
        iree_runtime_call_inputs_push_back_buffer_view(&call, dense_ple_bv),
        cleanup, rc);
  }
  XLA_CHECK_GOTO(
      iree_runtime_call_inputs_push_back_buffer_view(&call, positions_bv),
      cleanup, rc);
  XLA_CHECK_GOTO(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv),
                 cleanup, rc);
  XLA_CHECK_GOTO(iree_runtime_call_inputs_push_back_buffer_view(&call, bias_bv),
                 cleanup, rc);
  XLA_CHECK_GOTO(iree_runtime_call_invoke(&call, 0), cleanup, rc);
  XLA_CHECK_GOTO(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &output), cleanup,
      rc);
  XLA_CHECK_GOTO(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc),
                 cleanup, rc);
  XLA_CHECK_GOTO(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc),
                 cleanup, rc);
  if (xla_validate_kv_pair(c, kc, vc, 4, 1) != 0) {
    rc = 1;
    goto cleanup;
  }

  if (batched) {
    XLA_CHECK_GOTO(
        xla_read_logits(c, output, out_logits, (iree_host_size_t)vocab),
        cleanup, rc);
    rc = xla_read_selected_kv(c, kc, vc, real_len, kv_width, out_kv);
    if (rc != 0) goto cleanup;
    rc = xla_store_prefill_kv_slot(c, slot, kc, vc);
  } else {
    XLA_CHECK_GOTO(xla_read_token(c, output, out_token), cleanup, rc);
    iree_hal_buffer_view_t* old_k = c->kcache;
    iree_hal_buffer_view_t* old_v = c->vcache;
    c->kcache = kc;
    c->vcache = vc;
    kc = NULL;
    vc = NULL;
    if (old_k) iree_hal_buffer_view_release(old_k);
    if (old_v) iree_hal_buffer_view_release(old_v);
  }

cleanup:
  if (output) iree_hal_buffer_view_release(output);
  if (kc) iree_hal_buffer_view_release(kc);
  if (vc) iree_hal_buffer_view_release(vc);
  if (embeddings_bv) iree_hal_buffer_view_release(embeddings_bv);
  if (dense_ple_bv) iree_hal_buffer_view_release(dense_ple_bv);
  if (positions_bv) iree_hal_buffer_view_release(positions_bv);
  if (len_bv) iree_hal_buffer_view_release(len_bv);
  if (bias_bv) iree_hal_buffer_view_release(bias_bv);
  if (call_initialized) iree_runtime_call_deinitialize(&call);
  return rc;
}

int xla_llama_prefill_embeddings(
    xla_ctx* c, int32_t position_mode, const xla_tensor_desc* embeddings,
    const xla_tensor_desc* positions, const xla_tensor_desc* attention_bias,
    int32_t real_len, int32_t* out_token) {
  return xla_llama_prefill_embeddings_impl(
      c, -1, position_mode, embeddings, NULL, positions, attention_bias,
      real_len, 0, out_token, NULL, 0, NULL);
}

int xla_llama_prefill_embeddings_slot_logits(
    xla_ctx* c, int32_t slot, int32_t position_mode,
    const xla_tensor_desc* embeddings,
    const xla_tensor_desc* positions, const xla_tensor_desc* attention_bias,
    int32_t real_len, int32_t vocab, float* out_logits) {
  return xla_llama_prefill_embeddings_impl(
      c, slot, position_mode, embeddings, NULL, positions, attention_bias,
      real_len, vocab, NULL, out_logits, 0, NULL);
}

int xla_llama_prefill_embeddings_slot_diagnostics(
    xla_ctx* c, int32_t slot, int32_t position_mode,
    const xla_tensor_desc* embeddings,
    const xla_tensor_desc* positions, const xla_tensor_desc* attention_bias,
    int32_t real_len, int32_t vocab, int32_t kv_width, float* out_logits,
    float* out_kv) {
  return xla_llama_prefill_embeddings_impl(
      c, slot, position_mode, embeddings, NULL, positions, attention_bias,
      real_len, vocab, NULL, out_logits, kv_width, out_kv);
}

int xla_llama_prefill_embeddings_ple(
    xla_ctx* c, int32_t position_mode, const xla_tensor_desc* embeddings,
    const xla_tensor_desc* dense_ple, const xla_tensor_desc* positions,
    const xla_tensor_desc* attention_bias, int32_t real_len,
    int32_t* out_token) {
  return xla_llama_prefill_embeddings_impl(
      c, -1, position_mode, embeddings, dense_ple, positions, attention_bias,
      real_len, 0, out_token, NULL, 0, NULL);
}

int xla_llama_prefill_embeddings_ple_slot_logits(
    xla_ctx* c, int32_t slot, int32_t position_mode,
    const xla_tensor_desc* embeddings,
    const xla_tensor_desc* dense_ple, const xla_tensor_desc* positions,
    const xla_tensor_desc* attention_bias, int32_t real_len, int32_t vocab,
    float* out_logits) {
  return xla_llama_prefill_embeddings_impl(
      c, slot, position_mode, embeddings, dense_ple, positions, attention_bias,
      real_len, vocab, NULL, out_logits, 0, NULL);
}

static int xla_llama_prefill_embeddings_deepstack_impl(
    xla_ctx* c, int32_t slot, int32_t position_mode,
    const xla_tensor_desc* embeddings,
    const xla_tensor_desc* positions, const xla_tensor_desc* attention_bias,
    const xla_tensor_desc* visual_positions,
    const xla_tensor_desc* layer_features,
    const xla_tensor_desc* layer_indices, int32_t actual_layer_count,
    int32_t actual_visual_count, int32_t real_len, int32_t vocab,
    int32_t* out_token, float* out_logits) {
  if (!c || !c->has_deepstack || position_mode != c->position_mode ||
      real_len <= 0 ||
      real_len > c->context_capacity ||
      actual_layer_count != c->deepstack_layers ||
      actual_visual_count < 0 ||
      actual_visual_count > c->deepstack_visual_positions) {
    fprintf(stderr,
            "xla_llama_prefill_embeddings_deepstack: invalid bundle/mode/count/length contract\n");
    return 1;
  }
  int64_t embeddings_dims[2] = {c->context_capacity, c->hidden_size};
  int64_t positions_1d_dims[1] = {c->context_capacity};
  int64_t positions_mrope_dims[2] = {3, c->context_capacity};
  int64_t bias_dims[2] = {c->context_capacity, c->context_capacity};
  int64_t visual_position_dims[1] = {c->deepstack_visual_positions};
  int64_t layer_feature_dims[3] = {
      c->deepstack_layers, c->deepstack_visual_positions, c->hidden_size};
  int64_t layer_index_dims[1] = {c->deepstack_layers};
  if (xla_validate_tensor_desc("embeddings", embeddings, 0, 2,
                               embeddings_dims) != 0 ||
      xla_validate_tensor_desc(
          "positions", positions, 1, position_mode == 1 ? 2 : 1,
          position_mode == 1 ? positions_mrope_dims : positions_1d_dims) != 0 ||
      xla_validate_tensor_desc("attention_bias", attention_bias, 0, 2,
                               bias_dims) != 0 ||
      xla_validate_tensor_desc("deepstack.visual_positions", visual_positions,
                               1, 1, visual_position_dims) != 0 ||
      xla_validate_tensor_desc("deepstack.layer_features", layer_features, 0,
                               3, layer_feature_dims) != 0 ||
      xla_validate_tensor_desc("deepstack.layer_indices", layer_indices, 1, 1,
                               layer_index_dims) != 0) {
    return 1;
  }

  const int32_t* visual = (const int32_t*)visual_positions->data;
  const int32_t* layers = (const int32_t*)layer_indices->data;
  const float* features = (const float*)layer_features->data;
  for (int32_t i = 0; i < actual_visual_count; ++i) {
    if (visual[i] < 0 || visual[i] >= real_len ||
        (i > 0 && visual[i] <= visual[i - 1])) {
      fprintf(stderr,
              "xla_llama_prefill_embeddings_deepstack: visual positions must be sorted, unique, and in range\n");
      return 1;
    }
  }
  for (int32_t i = actual_visual_count;
       i < c->deepstack_visual_positions; ++i) {
    if (visual[i] != -1) {
      fprintf(stderr,
              "xla_llama_prefill_embeddings_deepstack: padded visual positions must be -1\n");
      return 1;
    }
  }
  for (int32_t i = 0; i < actual_layer_count; ++i) {
    if (layers[i] != c->deepstack_target_layers[i] || layers[i] < 0 ||
        layers[i] >= c->model_layers) {
      fprintf(stderr,
              "xla_llama_prefill_embeddings_deepstack: layer indices disagree with compiled targets\n");
      return 1;
    }
  }
  size_t feature_count =
      (size_t)c->deepstack_layers *
      (size_t)c->deepstack_visual_positions * (size_t)c->hidden_size;
  for (size_t i = 0; i < feature_count; ++i) {
    if (!isfinite(features[i])) {
      fprintf(stderr,
              "xla_llama_prefill_embeddings_deepstack: non-finite layer feature\n");
      return 1;
    }
  }
  for (int32_t layer = 0; layer < c->deepstack_layers; ++layer) {
    for (int32_t visual_index = actual_visual_count;
         visual_index < c->deepstack_visual_positions; ++visual_index) {
      size_t offset =
          ((size_t)layer * (size_t)c->deepstack_visual_positions +
           (size_t)visual_index) *
          (size_t)c->hidden_size;
      for (int32_t hidden = 0; hidden < c->hidden_size; ++hidden) {
        if (features[offset + (size_t)hidden] != 0.0f) {
          fprintf(stderr,
                  "xla_llama_prefill_embeddings_deepstack: padded feature rows must be zero\n");
          return 1;
        }
      }
    }
  }

  bool batched = slot >= 0;
  if ((!batched && !out_token) ||
      (batched && (slot >= c->rg_bsz || vocab <= 0 || !out_logits))) {
    fprintf(stderr,
            "xla_llama_prefill_embeddings_deepstack: invalid slot/output/vocab contract\n");
    return 1;
  }

  int rc = 0;
  bool call_initialized = false;
  iree_runtime_call_t call;
  iree_hal_buffer_view_t* embeddings_bv = NULL;
  iree_hal_buffer_view_t* positions_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  iree_hal_buffer_view_t* bias_bv = NULL;
  iree_hal_buffer_view_t* visual_bv = NULL;
  iree_hal_buffer_view_t* features_bv = NULL;
  iree_hal_buffer_view_t* layers_bv = NULL;
  iree_hal_buffer_view_t* actual_layers_bv = NULL;
  iree_hal_buffer_view_t* actual_visual_bv = NULL;
  iree_hal_buffer_view_t* output = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;

  XLA_CHECK_GOTO(iree_runtime_call_initialize_by_name(
                     c->session,
                     iree_make_cstring_view(
                         "prefill_embeddings_deepstack.main"),
                     &call),
                 cleanup, rc);
  call_initialized = true;
  for (int32_t i = 0; i < c->n_weights; ++i) {
    XLA_CHECK_GOTO(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]),
        cleanup, rc);
  }
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, embeddings, &embeddings_bv), cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, positions, &positions_bv), cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32,
                             &real_len, sizeof(int32_t), &len_bv),
                 cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, attention_bias, &bias_bv), cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, visual_positions, &visual_bv), cleanup,
                 rc);
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, layer_features, &features_bv), cleanup,
                 rc);
  XLA_CHECK_GOTO(xla_alloc_desc_bv(c, layer_indices, &layers_bv), cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32,
                             &actual_layer_count, sizeof(int32_t),
                             &actual_layers_bv),
                 cleanup, rc);
  XLA_CHECK_GOTO(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32,
                             &actual_visual_count, sizeof(int32_t),
                             &actual_visual_bv),
                 cleanup, rc);
  iree_hal_buffer_view_t* inputs[] = {
      embeddings_bv, positions_bv, len_bv,          bias_bv, visual_bv,
      features_bv,   layers_bv,    actual_layers_bv, actual_visual_bv};
  for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    XLA_CHECK_GOTO(
        iree_runtime_call_inputs_push_back_buffer_view(&call, inputs[i]),
        cleanup, rc);
  }
  XLA_CHECK_GOTO(iree_runtime_call_invoke(&call, 0), cleanup, rc);
  XLA_CHECK_GOTO(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &output), cleanup,
      rc);
  XLA_CHECK_GOTO(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc),
                 cleanup, rc);
  XLA_CHECK_GOTO(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc),
                 cleanup, rc);
  if (xla_validate_kv_pair(c, kc, vc, 4, 1) != 0) {
    rc = 1;
    goto cleanup;
  }
  if (batched) {
    XLA_CHECK_GOTO(
        xla_read_logits(c, output, out_logits, (iree_host_size_t)vocab),
        cleanup, rc);
    rc = xla_store_prefill_kv_slot(c, slot, kc, vc);
  } else {
    XLA_CHECK_GOTO(xla_read_token(c, output, out_token), cleanup, rc);
    iree_hal_buffer_view_t* old_k = c->kcache;
    iree_hal_buffer_view_t* old_v = c->vcache;
    c->kcache = kc;
    c->vcache = vc;
    kc = NULL;
    vc = NULL;
    if (old_k) iree_hal_buffer_view_release(old_k);
    if (old_v) iree_hal_buffer_view_release(old_v);
  }

cleanup:
  if (output) iree_hal_buffer_view_release(output);
  if (kc) iree_hal_buffer_view_release(kc);
  if (vc) iree_hal_buffer_view_release(vc);
  if (embeddings_bv) iree_hal_buffer_view_release(embeddings_bv);
  if (positions_bv) iree_hal_buffer_view_release(positions_bv);
  if (len_bv) iree_hal_buffer_view_release(len_bv);
  if (bias_bv) iree_hal_buffer_view_release(bias_bv);
  if (visual_bv) iree_hal_buffer_view_release(visual_bv);
  if (features_bv) iree_hal_buffer_view_release(features_bv);
  if (layers_bv) iree_hal_buffer_view_release(layers_bv);
  if (actual_layers_bv) iree_hal_buffer_view_release(actual_layers_bv);
  if (actual_visual_bv) iree_hal_buffer_view_release(actual_visual_bv);
  if (call_initialized) iree_runtime_call_deinitialize(&call);
  return rc;
}

int xla_llama_prefill_embeddings_deepstack(
    xla_ctx* c, int32_t position_mode, const xla_tensor_desc* embeddings,
    const xla_tensor_desc* positions, const xla_tensor_desc* attention_bias,
    const xla_tensor_desc* visual_positions,
    const xla_tensor_desc* layer_features,
    const xla_tensor_desc* layer_indices, int32_t actual_layer_count,
    int32_t actual_visual_count, int32_t real_len, int32_t* out_token) {
  return xla_llama_prefill_embeddings_deepstack_impl(
      c, -1, position_mode, embeddings, positions, attention_bias, visual_positions,
      layer_features, layer_indices, actual_layer_count, actual_visual_count,
      real_len, 0, out_token, NULL);
}

int xla_llama_prefill_embeddings_deepstack_slot_logits(
    xla_ctx* c, int32_t slot, int32_t position_mode,
    const xla_tensor_desc* embeddings,
    const xla_tensor_desc* positions, const xla_tensor_desc* attention_bias,
    const xla_tensor_desc* visual_positions,
    const xla_tensor_desc* layer_features,
    const xla_tensor_desc* layer_indices, int32_t actual_layer_count,
    int32_t actual_visual_count, int32_t real_len, int32_t vocab,
    float* out_logits) {
  return xla_llama_prefill_embeddings_deepstack_impl(
      c, slot, position_mode, embeddings, positions, attention_bias, visual_positions,
      layer_features, layer_indices, actual_layer_count, actual_visual_count,
      real_len, vocab, NULL, out_logits);
}

// Ragged decode_step: token[B], pos[B], cache_len[B] (per row), rank-5 KV ->
// `[B, vocab]` LOGITS (copied to host `out_logits`); advances the resident rank-5
// KV in place. The caller (engine) samples a token per row. Inactive rows
// (token/pos/cache_len 0) are masked no-ops whose logits the caller discards.
static int xla_llama_decode_ragged_impl(
    xla_ctx* c, int32_t bsz, const int32_t* tokens,
    const int32_t* positions, int32_t position_mode,
    const int32_t* cache_len, int32_t vocab, float* out_logits) {
  if (bsz != c->rg_bsz || !c->kcache_b || !c->vcache_b ||
      c->position_mode != position_mode) {
    fprintf(stderr,
            "xla_llama_decode_ragged_logits: bsz=%d configured=%d mode=%d bundle_mode=%d or KV is uninitialized\n",
            bsz, c->rg_bsz, position_mode, c->position_mode);
    return 1;
  }
  for (int32_t i = 0; i < bsz; ++i) {
    if (cache_len[i] < 0 || cache_len[i] >= c->context_capacity) {
      fprintf(stderr,
              "xla_llama_decode_ragged_logits: row=%d cache_len=%d context_capacity=%d\n",
              i, cache_len[i], c->context_capacity);
      return 1;
    }
    int32_t position_count = position_mode == 1 ? 3 : 1;
    for (int32_t axis = 0; axis < position_count; ++axis) {
      int32_t coordinate = positions[i * position_count + axis];
      if (coordinate < 0 ||
          (position_mode == 0 && coordinate != cache_len[i])) {
        fprintf(stderr,
                "xla_llama_decode_ragged_logits: row=%d axis=%d coordinate=%d cache_len=%d\n",
                i, axis, coordinate, cache_len[i]);
        return 1;
      }
    }
  }
  iree_runtime_call_t call;
  XLA_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("decode_step.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    XLA_CHECK(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]));
  }
  iree_hal_dim_t vshape[1] = {(iree_hal_dim_t)bsz};
  iree_hal_dim_t position_shape[2] = {(iree_hal_dim_t)bsz, 3};
  iree_host_size_t vbytes = (iree_host_size_t)bsz * sizeof(int32_t);
  iree_host_size_t position_bytes =
      vbytes * (position_mode == 1 ? 3 : 1);
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  XLA_CHECK(xla_alloc_bv(c, 1, vshape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                         vbytes, &tok_bv));
  XLA_CHECK(xla_alloc_bv(c, position_mode == 1 ? 2 : 1,
                         position_mode == 1 ? position_shape : vshape,
                         IREE_HAL_ELEMENT_TYPE_INT_32, positions,
                         position_bytes, &pos_bv));
  XLA_CHECK(xla_alloc_bv(c, 1, vshape, IREE_HAL_ELEMENT_TYPE_INT_32, cache_len,
                         vbytes, &len_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->kcache_b));
  XLA_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->vcache_b));

  XLA_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* logits_out = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &logits_out));
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc));
  XLA_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc));
  if (xla_validate_kv_pair(c, kc, vc, 5, 2) != 0 ||
      iree_hal_buffer_view_shape_dim(kc, 0) != (iree_hal_dim_t)c->rg_bsz) {
    fprintf(stderr, "xla_llama_decode_ragged_logits: output batch shape mismatch\n");
    iree_hal_buffer_view_release(logits_out);
    iree_hal_buffer_view_release(kc);
    iree_hal_buffer_view_release(vc);
    iree_hal_buffer_view_release(tok_bv);
    iree_hal_buffer_view_release(pos_bv);
    iree_hal_buffer_view_release(len_bv);
    iree_runtime_call_deinitialize(&call);
    return 1;
  }
  iree_hal_buffer_view_t* old_k = c->kcache_b;
  iree_hal_buffer_view_t* old_v = c->vcache_b;
  c->kcache_b = kc;
  c->vcache_b = vc;

  XLA_CHECK(xla_read_logits(c, logits_out, out_logits,
                            (iree_host_size_t)bsz * (iree_host_size_t)vocab));
  iree_hal_buffer_view_t* token_out = logits_out;

  iree_hal_buffer_view_release(token_out);
  iree_hal_buffer_view_release(tok_bv);
  iree_hal_buffer_view_release(pos_bv);
  iree_hal_buffer_view_release(len_bv);
  iree_runtime_call_deinitialize(&call);
  iree_hal_buffer_view_release(old_k);
  iree_hal_buffer_view_release(old_v);
  return 0;
}

int xla_llama_decode_ragged_logits(xla_ctx* c, int32_t bsz,
                                   const int32_t* tokens,
                                   const int32_t* positions,
                                   const int32_t* cache_len, int32_t vocab,
                                   float* out_logits) {
  return xla_llama_decode_ragged_impl(c, bsz, tokens, positions, 0, cache_len,
                                      vocab, out_logits);
}

int xla_llama_decode_ragged_mrope_logits(
    xla_ctx* c, int32_t bsz, const int32_t* tokens,
    const int32_t* positions, const int32_t* cache_len, int32_t vocab,
    float* out_logits) {
  return xla_llama_decode_ragged_impl(c, bsz, tokens, positions, 1, cache_len,
                                      vocab, out_logits);
}

void xla_llama_free(xla_ctx* c) {
  if (!c) return;
  if (c->weights) {
    for (int32_t i = 0; i < c->n_weights; i++) {
      if (c->weights[i]) iree_hal_buffer_view_release(c->weights[i]);
    }
    free(c->weights);
  }
  if (c->kcache) iree_hal_buffer_view_release(c->kcache);
  if (c->vcache) iree_hal_buffer_view_release(c->vcache);
  if (c->kcache_b) iree_hal_buffer_view_release(c->kcache_b);
  if (c->vcache_b) iree_hal_buffer_view_release(c->vcache_b);
  free(c->deepstack_target_layers);
  if (c->session) iree_runtime_session_release(c->session);
  if (c->device) iree_hal_device_release(c->device);
  if (c->instance) iree_runtime_instance_release(c->instance);
  free(c);
}
