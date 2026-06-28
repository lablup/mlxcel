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
#include <stdio.h>
#include <stdlib.h>

// The iree-dist build leaves the system allocator to the application (its
// iree_allocator_system() is gated on IREE_ALLOCATOR_SYSTEM_CTL). Point it at a
// libc malloc/free control function, defined below, before the IREE headers.
#define IREE_ALLOCATOR_SYSTEM_CTL iree_xla_libc_ctl

#include <iree/runtime/api.h>
#include <iree/hal/buffer_view_util.h>
#include <iree/hal/buffer_transfer.h>
#ifdef XLA_GATE_CUDA
// CUDA mode (GB10): built against a source-built cuda-enabled IREE runtime
// (build.rs cuda mode defines XLA_GATE_CUDA). The unified runtime bundles the
// cuda driver impl + local-task, but the registration wrapper is separate, so
// the driver is registered explicitly below.
#include <iree/hal/drivers/cuda/registration/driver_module.h>
#endif

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

#define XLA_CHECK(expr)                                                  \
  do {                                                                   \
    iree_status_t _s = (expr);                                          \
    if (!iree_status_is_ok(_s)) {                                       \
      int _c = (int)iree_status_code(_s);                               \
      fprintf(stderr, "xla_iree: %s failed (status %d)\n", #expr, _c);  \
      iree_status_ignore(_s);                                           \
      return _c ? _c : 1;                                               \
    }                                                                    \
  } while (0)

typedef struct xla_ctx {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;   // holds both @prefill and @decode_step
  iree_hal_allocator_t* allocator;   // device allocator (shared by both calls)
  int32_t n_weights;
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

static iree_status_t xla_llama_create_impl(
    xla_ctx* c, const char* device_uri, const char* prefill_vmfb,
    const char* decode_vmfb, int32_t n_weights, const float* const* weight_data,
    const int32_t* weight_ranks, const int64_t* weight_dims) {
  c->n_weights = n_weights;

#ifdef XLA_GATE_CUDA
  // Register the CUDA driver so use_all_available_drivers exposes it for a
  // device_uri of "cuda" (GB10). Non-fatal; CUDA is initialized only when a
  // cuda device is actually created.
  iree_status_t cu_reg =
      iree_hal_cuda_driver_module_register(iree_hal_driver_registry_default());
  if (!iree_status_is_ok(cu_reg)) iree_status_ignore(cu_reg);
#endif

  iree_runtime_instance_options_t inst_opts;
  iree_runtime_instance_options_initialize(&inst_opts);
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

  // Both modules in one session; their distinct names (@prefill, @decode_step)
  // make the calls "prefill.main" and "decode_step.main".
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      c->session, prefill_vmfb));
  IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      c->session, decode_vmfb));

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
    IREE_RETURN_IF_ERROR(xla_alloc_bv(c, (iree_host_size_t)rank, shape,
                                      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                      weight_data[i], nelem * sizeof(float),
                                      &c->weights[i]));
  }
  return iree_ok_status();
}

// Create the execution context. `device_uri` is "local-task" (CPU) or another
// registered HAL driver. `weight_data[i]` points at `prod(weight_dims[i*4..])`
// f32 values laid out row-major; the shim copies them into resident device
// buffers, so the caller may free them after this returns. Returns NULL on
// failure (after printing a diagnostic).
xla_ctx* xla_llama_create(const char* device_uri, const char* prefill_vmfb,
                          const char* decode_vmfb, int32_t n_weights,
                          const float* const* weight_data,
                          const int32_t* weight_ranks,
                          const int64_t* weight_dims) {
  xla_ctx* c = (xla_ctx*)calloc(1, sizeof(xla_ctx));
  if (!c) return NULL;
  iree_status_t s =
      xla_llama_create_impl(c, device_uri, prefill_vmfb, decode_vmfb, n_weights,
                            weight_data, weight_ranks, weight_dims);
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

// prefill: tokens[lp], positions[lp], real_len -> first token id; the returned
// KV cache becomes the resident cache decode threads forward.
int xla_llama_prefill(xla_ctx* c, const int32_t* tokens, int32_t lp,
                      const int32_t* positions, int32_t real_len,
                      int32_t* out_token) {
  iree_runtime_call_t call;
  XLA_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("prefill.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    XLA_CHECK(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]));
  }
  iree_hal_dim_t seq_shape[1] = {(iree_hal_dim_t)lp};
  iree_host_size_t seq_bytes = (iree_host_size_t)lp * sizeof(int32_t);
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  XLA_CHECK(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                         seq_bytes, &tok_bv));
  XLA_CHECK(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32,
                         positions, seq_bytes, &pos_bv));
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

// decode_step: token, pos, cache_len + resident KV -> next token id; advances
// the resident KV cache in place.
int xla_llama_decode(xla_ctx* c, int32_t token, int32_t pos, int32_t cache_len,
                     int32_t* out_token) {
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
  XLA_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &pos,
                         sizeof(int32_t), &pos_bv));
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
  iree_host_size_t count = (iree_host_size_t)c->rg_bsz * c->rg_per;
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
  return s;
}

// Reset the batched state for `bsz` slots: drop any rank-5 KV and forget the
// per-slot dims (re-captured by the next prefill_slot, which lazily reallocates
// the rank-5 KV once the dims are known). Call once before seeding slots.
int xla_llama_ragged_reset(xla_ctx* c, int32_t bsz) {
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
  iree_runtime_call_t call;
  XLA_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("prefill.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    XLA_CHECK(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]));
  }
  iree_hal_dim_t seq_shape[1] = {(iree_hal_dim_t)lp};
  iree_host_size_t seq_bytes = (iree_host_size_t)lp * sizeof(int32_t);
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  XLA_CHECK(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                         seq_bytes, &tok_bv));
  XLA_CHECK(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32, positions,
                         seq_bytes, &pos_bv));
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
  if (c->kcache) iree_hal_buffer_view_release(c->kcache);
  if (c->vcache) iree_hal_buffer_view_release(c->vcache);
  c->kcache = kc;
  c->vcache = vc;

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
    return code ? code : 1;
  }

  if (iree_hal_buffer_view_shape_rank(c->kcache) != 4) {
    fprintf(stderr, "xla_llama_prefill_slot_logits: expected rank-4 single-seq KV\n");
    return 1;
  }
  iree_host_size_t per = iree_hal_buffer_view_element_count(c->kcache);
  if (c->rg_per == 0) {
    c->rg_per = per;
    for (int32_t i = 0; i < 4; i++) {
      c->rg_dims[i] = iree_hal_buffer_view_shape_dim(c->kcache, i);
    }
  } else if (per != c->rg_per) {
    fprintf(stderr, "xla_llama_prefill_slot_logits: KV element count changed\n");
    return 1;
  }
  XLA_CHECK(xla_ensure_batch_kv(c));
  iree_device_size_t off = (iree_device_size_t)slot * per * sizeof(float);
  iree_device_size_t len = (iree_device_size_t)per * sizeof(float);
  XLA_CHECK(iree_hal_device_transfer_d2d(
      c->device, iree_hal_buffer_view_buffer(c->kcache), 0,
      iree_hal_buffer_view_buffer(c->kcache_b), off, len,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  XLA_CHECK(iree_hal_device_transfer_d2d(
      c->device, iree_hal_buffer_view_buffer(c->vcache), 0,
      iree_hal_buffer_view_buffer(c->vcache_b), off, len,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  iree_hal_buffer_view_release(c->kcache);
  c->kcache = NULL;
  iree_hal_buffer_view_release(c->vcache);
  c->vcache = NULL;
  return 0;
}

// Ragged decode_step: token[B], pos[B], cache_len[B] (per row), rank-5 KV ->
// `[B, vocab]` LOGITS (copied to host `out_logits`); advances the resident rank-5
// KV in place. The caller (engine) samples a token per row. Inactive rows
// (token/pos/cache_len 0) are masked no-ops whose logits the caller discards.
int xla_llama_decode_ragged_logits(xla_ctx* c, int32_t bsz, const int32_t* tokens,
                                   const int32_t* pos, const int32_t* cache_len,
                                   int32_t vocab, float* out_logits) {
  iree_runtime_call_t call;
  XLA_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("decode_step.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    XLA_CHECK(
        iree_runtime_call_inputs_push_back_buffer_view(&call, c->weights[i]));
  }
  iree_hal_dim_t vshape[1] = {(iree_hal_dim_t)bsz};
  iree_host_size_t vbytes = (iree_host_size_t)bsz * sizeof(int32_t);
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  XLA_CHECK(xla_alloc_bv(c, 1, vshape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                         vbytes, &tok_bv));
  XLA_CHECK(xla_alloc_bv(c, 1, vshape, IREE_HAL_ELEMENT_TYPE_INT_32, pos, vbytes,
                         &pos_bv));
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
  if (c->session) iree_runtime_session_release(c->session);
  if (c->device) iree_hal_device_release(c->device);
  if (c->instance) iree_runtime_instance_release(c->instance);
  free(c);
}
