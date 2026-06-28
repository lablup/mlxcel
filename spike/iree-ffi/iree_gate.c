// Minimal C shim over the IREE runtime C API: load a vmfb and invoke
// module.main with two [n]f32 inputs, returning the [n]f32 output. This is the
// FFI-gate proof (issue #449 Phase 3 M2) and the shape the mlxcel-xla backend
// will use: a thin C shim over the prebuilt IREE runtime, with Rust calling a
// flat C ABI rather than binding the runtime structs directly.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The iree-dist build leaves the system allocator to the application (its
// iree_allocator_system() is gated on IREE_ALLOCATOR_SYSTEM_CTL). Point it at a
// libc malloc/free control function, defined below, before the IREE headers.
#define IREE_ALLOCATOR_SYSTEM_CTL iree_gate_libc_ctl

#include <iree/runtime/api.h>
#include <iree/hal/buffer_view_util.h>
#include <iree/hal/buffer_transfer.h>
#ifdef XLA_GATE_CUDA
// CUDA experiment (GB10): the prebuilt dist has no cuda driver, so this builds
// against a source-built cuda-enabled IREE runtime (build.rs cuda mode defines
// XLA_GATE_CUDA). The unified runtime bundles only local-task; the cuda driver
// is a separate lib, registered explicitly below.
#include <iree/hal/drivers/cuda/registration/driver_module.h>
#endif

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

// ===========================================================================
// Llama-3.2-1B execution: load the #451-emitted prefill + decode vmfbs into one
// session, keep the 146 weights resident as device buffers, thread the KV cache
// across steps, and return a token id per step (issue #449 Phase 3 M2). This is
// the shape mlxcel-xla's prefill / decode_step will take. Argmax is on the host
// here (read [V] logits back, pick the max); the Phase 2b on-device argmax
// variant returns a scalar token id and is wired once the emitter emits it.
// ===========================================================================

typedef struct xla_ctx {
  iree_runtime_instance_t* instance;
  iree_hal_device_t* device;
  iree_runtime_session_t* session;   // holds both @prefill and @decode_step
  iree_hal_allocator_t* allocator;   // device allocator (shared by both calls)
  int32_t n_weights;
  iree_hal_buffer_view_t** weights;  // resident weights, uploaded once
  iree_hal_buffer_view_t* kcache;    // threaded KV (set by prefill, advanced by decode)
  iree_hal_buffer_view_t* vcache;
  iree_hal_buffer_view_t* kcache_b;  // rank-5 batched KV [B,L,S,nkv,d] (uniform-B/ragged decode)
  iree_hal_buffer_view_t* vcache_b;
  // Ragged (continuous-batching) host KV mirror: per-slot single-seq prefills are
  // assembled here, then committed (h2d) into kcache_b/vcache_b. rg_dims is the
  // rank-4 single-seq KV shape [L,S,nkv,d]; rg_per = its element count.
  int32_t rg_bsz;
  float* rg_mk;
  float* rg_mv;
  iree_host_size_t rg_per;
  iree_hal_dim_t rg_dims[4];
  int32_t vocab;
} xla_ctx;

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

// Batched output readback (uniform-B decode): a [B] i32 buffer is an on-device
// per-row argmax (4*B-byte readback); a [B, V] f32 buffer is raw logits, argmaxed
// per row on the host. Fills out_tokens[0..bsz].
static iree_status_t xla_read_tokens_batch(xla_ctx* c, int32_t bsz,
                                           iree_hal_buffer_view_t* out,
                                           int32_t* out_tokens) {
  iree_host_size_t n = iree_hal_buffer_view_element_count(out);
  iree_hal_element_type_t et = iree_hal_buffer_view_element_type(out);
  if (et == IREE_HAL_ELEMENT_TYPE_INT_32 && n == (iree_host_size_t)bsz) {
    return iree_hal_device_transfer_d2h(
        c->device, iree_hal_buffer_view_buffer(out), 0, out_tokens,
        (iree_host_size_t)bsz * sizeof(int32_t),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  iree_host_size_t v = n / (iree_host_size_t)bsz;
  float* host = (float*)malloc(n * sizeof(float));
  if (!host) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_status_t s = iree_hal_device_transfer_d2h(
      c->device, iree_hal_buffer_view_buffer(out), 0, host, n * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  if (iree_status_is_ok(s)) {
    for (int32_t r = 0; r < bsz; r++) {
      out_tokens[r] = xla_argmax_f32(host + (iree_host_size_t)r * v, (int32_t)v);
    }
  }
  free(host);
  return s;
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
    const int32_t* weight_ranks, const int64_t* weight_dims, int32_t vocab) {
  c->vocab = vocab;
  c->n_weights = n_weights;

#ifdef XLA_GATE_CUDA
  // Register the CUDA driver so use_all_available_drivers exposes it for a
  // device_uri of "cuda" (GB10). Non-fatal; CUDA is initialized only when a
  // cuda device is created.
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

  c->weights = (iree_hal_buffer_view_t**)calloc((size_t)n_weights,
                                                sizeof(iree_hal_buffer_view_t*));
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

// Create the execution context. Returns NULL on failure (after printing).
xla_ctx* xla_llama_create(const char* device_uri, const char* prefill_vmfb,
                          const char* decode_vmfb, int32_t n_weights,
                          const float* const* weight_data,
                          const int32_t* weight_ranks,
                          const int64_t* weight_dims, int32_t vocab) {
  xla_ctx* c = (xla_ctx*)calloc(1, sizeof(xla_ctx));
  if (!c) return NULL;
  iree_status_t s =
      xla_llama_create_impl(c, device_uri, prefill_vmfb, decode_vmfb, n_weights,
                            weight_data, weight_ranks, weight_dims, vocab);
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
  GATE_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("prefill.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call,
                                                              c->weights[i]));
  }
  iree_hal_dim_t seq_shape[1] = {(iree_hal_dim_t)lp};
  iree_host_size_t seq_bytes = (iree_host_size_t)lp * sizeof(int32_t);
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  GATE_CHECK(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                          seq_bytes, &tok_bv));
  GATE_CHECK(xla_alloc_bv(c, 1, seq_shape, IREE_HAL_ELEMENT_TYPE_INT_32,
                          positions, seq_bytes, &pos_bv));
  GATE_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &real_len,
                          sizeof(int32_t), &len_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv));

  GATE_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* logits = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &logits));
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc));
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc));
  if (c->kcache) iree_hal_buffer_view_release(c->kcache);
  if (c->vcache) iree_hal_buffer_view_release(c->vcache);
  c->kcache = kc;
  c->vcache = vc;

  GATE_CHECK(xla_read_token(c, logits, out_token));

  iree_hal_buffer_view_release(logits);
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
  GATE_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("decode_step.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call,
                                                              c->weights[i]));
  }
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  GATE_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &token,
                          sizeof(int32_t), &tok_bv));
  GATE_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &pos,
                          sizeof(int32_t), &pos_bv));
  GATE_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &cache_len,
                          sizeof(int32_t), &len_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->kcache));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->vcache));

  GATE_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* logits = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &logits));
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc));
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc));
  iree_hal_buffer_view_t* old_k = c->kcache;
  iree_hal_buffer_view_t* old_v = c->vcache;
  c->kcache = kc;
  c->vcache = vc;

  GATE_CHECK(xla_read_token(c, logits, out_token));

  iree_hal_buffer_view_release(logits);
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
// Uniform-B batched decode (#449 M3 Stage 1). All B sequences advance in
// lockstep (shared pos/cache_len), so the single-seq prefill runs once and its
// rank-4 KV is tiled across B rows into the rank-5 cache the batched decode
// threads. The batched decode vmfb (emitter `decode-batch-argmax <B>`) takes
// token[B] + the rank-5 KV and returns token[B], reusing each weight across the
// batch (GEMV -> GEMM). prefill_vmfb is the same single-seq graph as the scalar
// path; only the decode module is the batched one.
// ===========================================================================

// Tile a rank-4 KV buffer [d0,d1,d2,d3] B times into a resident rank-5 buffer
// [B,d0,d1,d2,d3] (every row a copy). One-time prefill cost.
static iree_status_t xla_tile_one(xla_ctx* c, int32_t bsz,
                                  iree_hal_buffer_view_t* src,
                                  iree_hal_buffer_view_t** dst) {
  if (iree_hal_buffer_view_shape_rank(src) != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected rank-4 KV to tile");
  }
  iree_hal_dim_t d0 = iree_hal_buffer_view_shape_dim(src, 0);
  iree_hal_dim_t d1 = iree_hal_buffer_view_shape_dim(src, 1);
  iree_hal_dim_t d2 = iree_hal_buffer_view_shape_dim(src, 2);
  iree_hal_dim_t d3 = iree_hal_buffer_view_shape_dim(src, 3);
  iree_host_size_t per =
      (iree_host_size_t)d0 * (iree_host_size_t)d1 * (iree_host_size_t)d2 *
      (iree_host_size_t)d3;
  iree_host_size_t pbytes = per * sizeof(float);

  float* host = (float*)malloc(pbytes);
  if (!host) return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_status_t s = iree_hal_device_transfer_d2h(
      c->device, iree_hal_buffer_view_buffer(src), 0, host, pbytes,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  if (!iree_status_is_ok(s)) {
    free(host);
    return s;
  }
  float* hostb = (float*)malloc((iree_host_size_t)bsz * pbytes);
  if (!hostb) {
    free(host);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  for (int32_t r = 0; r < bsz; r++) {
    memcpy(hostb + (iree_host_size_t)r * per, host, pbytes);
  }
  free(host);
  iree_hal_dim_t shape5[5] = {(iree_hal_dim_t)bsz, d0, d1, d2, d3};
  s = xla_alloc_bv(c, 5, shape5, IREE_HAL_ELEMENT_TYPE_FLOAT_32, hostb,
                   (iree_host_size_t)bsz * pbytes, dst);
  free(hostb);
  return s;
}

// Tile the single-seq KV (c->kcache/vcache, set by prefill) into the rank-5
// batched cache and drop the single-seq buffers.
static iree_status_t xla_tile_kv_to_batch(xla_ctx* c, int32_t bsz) {
  if (c->kcache_b) {
    iree_hal_buffer_view_release(c->kcache_b);
    c->kcache_b = NULL;
  }
  if (c->vcache_b) {
    iree_hal_buffer_view_release(c->vcache_b);
    c->vcache_b = NULL;
  }
  IREE_RETURN_IF_ERROR(xla_tile_one(c, bsz, c->kcache, &c->kcache_b));
  IREE_RETURN_IF_ERROR(xla_tile_one(c, bsz, c->vcache, &c->vcache_b));
  iree_hal_buffer_view_release(c->kcache);
  c->kcache = NULL;
  iree_hal_buffer_view_release(c->vcache);
  c->vcache = NULL;
  return iree_ok_status();
}

// Batched prefill: run the single-seq prefill once for the prompt, tile its KV
// across B rows, and report the first token (identical for every row). The
// batched decode then advances each row independently.
int xla_llama_prefill_batch(xla_ctx* c, int32_t bsz, const int32_t* tokens,
                            int32_t lp, const int32_t* positions,
                            int32_t real_len, int32_t* out_tokens) {
  int32_t first = 0;
  int rc = xla_llama_prefill(c, tokens, lp, positions, real_len, &first);
  if (rc != 0) return rc;
  GATE_CHECK(xla_tile_kv_to_batch(c, bsz));
  for (int32_t r = 0; r < bsz; r++) out_tokens[r] = first;
  return 0;
}

// Batched decode_step: token[B], shared pos + cache_len, rank-5 KV -> token[B];
// advances the resident rank-5 KV in place.
int xla_llama_decode_batch(xla_ctx* c, int32_t bsz, const int32_t* tokens,
                           int32_t pos, int32_t cache_len, int32_t* out_tokens) {
  iree_runtime_call_t call;
  GATE_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("decode_step.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call,
                                                              c->weights[i]));
  }
  iree_hal_dim_t tok_shape[1] = {(iree_hal_dim_t)bsz};
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  GATE_CHECK(xla_alloc_bv(c, 1, tok_shape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                          (iree_host_size_t)bsz * sizeof(int32_t), &tok_bv));
  GATE_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &pos,
                          sizeof(int32_t), &pos_bv));
  GATE_CHECK(xla_alloc_bv(c, 0, NULL, IREE_HAL_ELEMENT_TYPE_INT_32, &cache_len,
                          sizeof(int32_t), &len_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->kcache_b));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->vcache_b));

  GATE_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* logits = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &logits));
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc));
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc));
  iree_hal_buffer_view_t* old_k = c->kcache_b;
  iree_hal_buffer_view_t* old_v = c->vcache_b;
  c->kcache_b = kc;
  c->vcache_b = vc;

  GATE_CHECK(xla_read_tokens_batch(c, bsz, logits, out_tokens));

  iree_hal_buffer_view_release(logits);
  iree_hal_buffer_view_release(tok_bv);
  iree_hal_buffer_view_release(pos_bv);
  iree_hal_buffer_view_release(len_bv);
  iree_runtime_call_deinitialize(&call);
  iree_hal_buffer_view_release(old_k);
  iree_hal_buffer_view_release(old_v);
  return 0;
}

// ===========================================================================
// Ragged continuous-batching decode (#449 M3 Stage 2a). Each slot carries its
// OWN position/length, so sequences of different lengths share the batch. The
// per-slot prompt KV is built by running the single-seq prefill into a host
// mirror (one slot at a time), then committed (h2d) into the resident rank-5 KV
// the ragged decode threads. decode_ragged takes token/pos/cache_len as [B]
// arrays (per row) instead of the uniform-B scalars.
// ===========================================================================

// Reset the ragged state for a batch of `bsz` slots (drops any mirror + KV).
int xla_llama_ragged_reset(xla_ctx* c, int32_t bsz) {
  c->rg_bsz = bsz;
  free(c->rg_mk);
  c->rg_mk = NULL;
  free(c->rg_mv);
  c->rg_mv = NULL;
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

// Prefill one prompt into mirror slot `slot` and report its first token. Runs the
// single-seq prefill (reusing the scalar prefill vmfb) and copies its rank-4 KV
// into the host mirror; call xla_llama_commit once all slots are filled.
int xla_llama_prefill_slot(xla_ctx* c, int32_t slot, const int32_t* tokens,
                           int32_t lp, const int32_t* positions, int32_t real_len,
                           int32_t* out_first) {
  int rc = xla_llama_prefill(c, tokens, lp, positions, real_len, out_first);
  if (rc != 0) return rc;
  if (iree_hal_buffer_view_shape_rank(c->kcache) != 4) {
    fprintf(stderr, "xla_llama_prefill_slot: expected rank-4 single-seq KV\n");
    return 1;
  }
  iree_host_size_t per = iree_hal_buffer_view_element_count(c->kcache);
  if (!c->rg_mk) {
    c->rg_per = per;
    for (int32_t i = 0; i < 4; i++) {
      c->rg_dims[i] = iree_hal_buffer_view_shape_dim(c->kcache, i);
    }
    c->rg_mk = (float*)calloc((size_t)c->rg_bsz * per, sizeof(float));
    c->rg_mv = (float*)calloc((size_t)c->rg_bsz * per, sizeof(float));
    if (!c->rg_mk || !c->rg_mv) return 1;
  }
  GATE_CHECK(iree_hal_device_transfer_d2h(
      c->device, iree_hal_buffer_view_buffer(c->kcache), 0,
      c->rg_mk + (size_t)slot * per, per * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  GATE_CHECK(iree_hal_device_transfer_d2h(
      c->device, iree_hal_buffer_view_buffer(c->vcache), 0,
      c->rg_mv + (size_t)slot * per, per * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  iree_hal_buffer_view_release(c->kcache);
  c->kcache = NULL;
  iree_hal_buffer_view_release(c->vcache);
  c->vcache = NULL;
  return 0;
}

// Upload the host mirror into the resident rank-5 KV the ragged decode threads.
int xla_llama_commit(xla_ctx* c) {
  if (!c->rg_mk) {
    fprintf(stderr, "xla_llama_commit: no slots prefilled\n");
    return 1;
  }
  iree_hal_dim_t shape5[5] = {(iree_hal_dim_t)c->rg_bsz, c->rg_dims[0],
                              c->rg_dims[1], c->rg_dims[2], c->rg_dims[3]};
  if (c->kcache_b) {
    iree_hal_buffer_view_release(c->kcache_b);
    c->kcache_b = NULL;
  }
  if (c->vcache_b) {
    iree_hal_buffer_view_release(c->vcache_b);
    c->vcache_b = NULL;
  }
  iree_host_size_t bytes = (iree_host_size_t)c->rg_bsz * c->rg_per * sizeof(float);
  GATE_CHECK(xla_alloc_bv(c, 5, shape5, IREE_HAL_ELEMENT_TYPE_FLOAT_32, c->rg_mk,
                          bytes, &c->kcache_b));
  GATE_CHECK(xla_alloc_bv(c, 5, shape5, IREE_HAL_ELEMENT_TYPE_FLOAT_32, c->rg_mv,
                          bytes, &c->vcache_b));
  return 0;
}

// Ragged decode_step: token[B], pos[B], cache_len[B] (per row), rank-5 KV ->
// token[B]; advances the resident rank-5 KV in place.
int xla_llama_decode_ragged(xla_ctx* c, int32_t bsz, const int32_t* tokens,
                            const int32_t* pos, const int32_t* cache_len,
                            int32_t* out_tokens) {
  iree_runtime_call_t call;
  GATE_CHECK(iree_runtime_call_initialize_by_name(
      c->session, iree_make_cstring_view("decode_step.main"), &call));
  for (int32_t i = 0; i < c->n_weights; i++) {
    GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call,
                                                              c->weights[i]));
  }
  iree_hal_dim_t vshape[1] = {(iree_hal_dim_t)bsz};
  iree_host_size_t vbytes = (iree_host_size_t)bsz * sizeof(int32_t);
  iree_hal_buffer_view_t* tok_bv = NULL;
  iree_hal_buffer_view_t* pos_bv = NULL;
  iree_hal_buffer_view_t* len_bv = NULL;
  GATE_CHECK(xla_alloc_bv(c, 1, vshape, IREE_HAL_ELEMENT_TYPE_INT_32, tokens,
                          vbytes, &tok_bv));
  GATE_CHECK(xla_alloc_bv(c, 1, vshape, IREE_HAL_ELEMENT_TYPE_INT_32, pos, vbytes,
                          &pos_bv));
  GATE_CHECK(xla_alloc_bv(c, 1, vshape, IREE_HAL_ELEMENT_TYPE_INT_32, cache_len,
                          vbytes, &len_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, tok_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, pos_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, len_bv));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->kcache_b));
  GATE_CHECK(iree_runtime_call_inputs_push_back_buffer_view(&call, c->vcache_b));

  GATE_CHECK(iree_runtime_call_invoke(&call, 0));

  iree_hal_buffer_view_t* logits = NULL;
  iree_hal_buffer_view_t* kc = NULL;
  iree_hal_buffer_view_t* vc = NULL;
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &logits));
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &kc));
  GATE_CHECK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &vc));
  iree_hal_buffer_view_t* old_k = c->kcache_b;
  iree_hal_buffer_view_t* old_v = c->vcache_b;
  c->kcache_b = kc;
  c->vcache_b = vc;

  GATE_CHECK(xla_read_tokens_batch(c, bsz, logits, out_tokens));

  iree_hal_buffer_view_release(logits);
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
  free(c->rg_mk);
  free(c->rg_mv);
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
