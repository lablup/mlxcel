// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/lru_cache.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/error.h"
#include "mlx/stream.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// Only include thrust headers when compiling with HIP compiler
// (thrust headers have dependencies on CUDA/HIP-specific headers)
#ifdef __HIPCC__
#include <thrust/execution_policy.h>
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlx::core::rocm {

// Forward declaration
class Device;
class Worker;

// Gate for the automatic HIP-graph batching path. Default OFF so the legacy
// immediate-launch path is unaffected unless MLX_USE_HIP_GRAPHS is set.
bool use_hip_graphs();

// Inline (graph-splitting) launch counter; see device.cpp diagnostics.
extern std::atomic<long> g_inline_launches_;

// --- GPU failure reporting (ml-explore/mlx#3742, lablup/mlxcel#1804) ---
//
// The HIP runtime on ROCm 7 does not abort the process on a GPU memory fault:
// it puts the whole device context into an error state, every later HIP call
// on any thread returns the fault (hipErrorIllegalAddress), and no queued
// hipLaunchHostFunc callback ever runs. This backend signals completion with
// exactly those callbacks, so without care a waiter spins forever. Each
// CommandEncoder therefore owns an Error, as the Metal encoder does; a wait or
// query that finds the stream failed records the status there and attaches it
// to the event, and Event::wait() throws it through Event::check_error().

// One line describing |status| observed while |where|; names the HIP error
// and, when the device context is gone, says so.
std::string describe_device_error(hipError_t status, const char* where);

// Record |status| as the failure of stream |s| (keeping an earlier one) and
// return the Error events signaled on |s| should point at. Does not bind the
// device and makes no HIP call that could fail, so it is safe on a dead device
// and from const queries. A CPU stream, or a stream whose encoder does not
// exist, gets a process-wide fallback Error.
Error& record_stream_error(Stream s, hipError_t status, const char* where);

// MLX_ROCM_GPU_WATCHDOG_SECS: longest a single host wait for GPU work may block
// before it fails with kWatchdogExpired. 0 (the default) disables it. A kernel
// that never finishes cannot be told from a slow one, so this is an opt-in
// bound for servers that prefer a failed request to a stuck one; the stream
// stays wedged behind the kernel either way.
int gpu_watchdog_seconds();
constexpr hipError_t kWatchdogExpired = hipErrorLaunchTimeOut;
// Diagnostics: tag inline launches by the primitive currently in eval_gpu.
void set_current_prim(const char* name);
void record_inline_launch();

// mlxcelverse: add_kernel_node hands HIP the address of every argument, so
// each argument must have the same size as the kernel parameter it lands in.
// A narrower argument is read with garbage high bytes (Scatter once passed an
// int32_t for an int64_t parameter and every update collapsed onto the first
// index), so the sizes are checked at compile time.
template <typename Func>
struct kernel_param_types;
template <typename... Args>
struct kernel_param_types<void(Args...)> {
  using type = std::tuple<Args...>;
};
template <typename KernelArgs, typename Passed, std::size_t... I>
constexpr bool kernel_arg_sizes_match(std::index_sequence<I...>) {
  return (
      (sizeof(std::tuple_element_t<I, KernelArgs>) ==
       sizeof(std::decay_t<std::tuple_element_t<I, Passed>>)) &&
      ...);
}

class CommandEncoder {
 public:
  explicit CommandEncoder(Device& d);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  void set_input_array(const array& arr);
  void set_output_array(const array& arr);

  template <typename F>
  void launch_kernel(F&& func);

  template <typename Func, typename... Params>
  void add_kernel_node(
      Func* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      Params&&... params) {
    add_kernel_node_ex(func, grid_dim, block_dim, smem_bytes, params...);
  }

  template <typename Func, typename... Params>
  void add_kernel_node_ex(
      Func* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      Params&&... params) {
    constexpr size_t num = sizeof...(Params);
    using KernelArgs = typename kernel_param_types<Func>::type;
    static_assert(
        std::tuple_size_v<KernelArgs> == num,
        "add_kernel_node: argument count differs from the kernel's parameters");
    static_assert(
        kernel_arg_sizes_match<KernelArgs, std::tuple<Params...>>(
            std::index_sequence_for<Params...>{}),
        "add_kernel_node: an argument's size differs from its kernel "
        "parameter; pass the parameter's exact type");
    if (!use_hip_graphs()) {
      // Immediate launch: kernelParams are consumed synchronously, so
      // addresses of the caller's locals are fine.
      void* ptrs[num > 0 ? num : 1];
      size_t i = 0;
      (
          [&](auto&& p) {
            ptrs[i++] =
                const_cast<void*>(static_cast<const void*>(std::addressof(p)));
          }(std::forward<Params>(params)),
          ...);
      add_kernel_node_raw(
          reinterpret_cast<void*>(func), grid_dim, block_dim, smem_bytes, ptrs);
      return;
    }
    // Graph build: a HIP graph kernel node references its kernelParams until
    // the node is instantiated/updated into the exec graph, which happens later
    // in commit(). The caller's argument locals are gone by then, so copy the
    // argument VALUES (and the pointer array) into a heap pack kept alive until
    // commit() finishes (cleared there).
    struct Pack {
      std::tuple<std::decay_t<Params>...> vals;
      std::array<void*, (num > 0 ? num : 1)> ptrs;
    };
    // Construct the value tuple in place — default-constructing Pack (and thus
    // the tuple) requires every decayed arg type to be default-constructible,
    // which some kernels' arg types are not (deleted default ctor on a clean
    // build).
    auto pack = std::make_shared<Pack>(Pack{
        std::tuple<std::decay_t<Params>...>(std::forward<Params>(params)...),
        {}});
    fill_param_ptrs(
        pack->vals, pack->ptrs, std::index_sequence_for<Params...>{});
    graph_node_args_.push_back(pack);
    add_kernel_node_raw(
        reinterpret_cast<void*>(func),
        grid_dim,
        block_dim,
        smem_bytes,
        pack->ptrs.data());
  }

  template <typename Tuple, typename Arr, size_t... I>
  static void
  fill_param_ptrs(Tuple& vals, Arr& ptrs, std::index_sequence<I...>) {
    ((ptrs[I] = const_cast<void*>(
          static_cast<const void*>(std::addressof(std::get<I>(vals))))),
     ...);
  }

  void add_kernel_node_raw(
      void* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      void** params);

  // Add a kernel node from a fully-formed params struct (preserves both
  // kernelParams and extra). Used by add_kernel_node_raw and by the child-graph
  // flattening path. Does all bookkeeping + dependency wiring.
  void add_kernel_node_kp(const hipKernelNodeParams& kp);

  // Add a MODULE-function (hiprtc/JIT or CustomKernel) kernel as a graph node.
  // ROCm 7.13's hipGraphAddKernelNode accepts a hipFunction_t in
  // hipKernelNodeParams.func, so JIT-fused kernels need not graph-split. In the
  // eager path this launches immediately. `args_keepalive` owns the storage
  // that `params` points into; it is held until the graph is instantiated
  // (commit).
  void add_module_kernel_node(
      void* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      void** params,
      std::shared_ptr<void> args_keepalive);

  void add_temporary(const array& arr);

  // Record |bytes| of new allocations made by the primitive just encoded (its
  // outputs and scratch). They stay pinned until this batch completes, so the
  // eager path commits once a batch has pinned enough, and waits for older
  // batches once too much is pinned in flight (lablup/mlxcel#2062).
  void add_batch_bytes(size_t bytes) {
    batch_bytes_ += bytes;
  }

  void add_completed_handler(std::function<void()> task);
  void maybe_commit();
  // commit(), then apply the in-flight bound to the batch just committed.
  // Used by maybe_commit and by gpu::finalize, so the last batch of every
  // eval (and every async_eval) is counted too.
  void commit_and_throttle();
  bool needs_commit();
  void commit();

  Device& device() {
    return device_;
  }

  HipStream& stream() {
    return stream_;
  }

  // Wait until kernels and completion handlers are finished. Throws the
  // stream's error if it has failed instead of waiting for handlers that a
  // failed stream never runs.
  void synchronize();

  // Errors observed on this stream. Events signaled on this stream point at
  // this object once the stream fails, so it has to outlive them: it lives as
  // long as the encoder, which its Device keeps for the life of the process.
  Error& error() {
    return error_;
  }

  // Record |status| as this stream's failure, keeping an earlier one. Does
  // not throw; callers throw through error().check() or Event::check_error().
  void set_device_error(hipError_t status, const char* where);

  // Read and clear the calling thread's pending HIP error after |primitive|
  // launched its kernels, and throw it as that primitive's launch failure.
  // Kernels launched with hipLaunchKernelGGL or <<<>>> report a synchronous
  // failure no other way. If the pending error is the device's fault state
  // rather than this launch, the stream is poisoned and that is thrown.
  void check_launch(const char* primitive);

 private:
  struct GraphNode {
    hipGraphNode_t node;
    // K = kernel, E = empty, () = subgraph
    std::string node_type;
    std::string id;
  };

  void insert_graph_dependencies(GraphNode node);
  void insert_graph_dependencies(std::vector<GraphNode> nodes);
  void add_child_graph_node(hipGraph_t child, const std::string& key);

  // In-flight memory bound for the eager path (lablup/mlxcel#2062).
  // Raw events rather than HipEvent: HipEvent returns its handle to a
  // function-local static pool on destruction, and an encoder can outlive
  // that pool at process exit.
  struct InflightBatch {
    hipEvent_t done; // recorded after the batch's kernels
    size_t bytes;
  };
  void throttle_inflight(size_t committed_bytes);
  void release_inflight();

  Device& device_;
  HipStream stream_;
  std::unique_ptr<Worker> worker_;
  Error error_;
  int node_count_{0};
  std::vector<std::shared_ptr<array::Data>> temporaries_;
  std::unordered_set<const array::Data*> temporary_ptrs_;
  // Bytes newly allocated by the primitives of the open batch.
  size_t batch_bytes_{0};
  // MLX_ROCM_MAX_INFLIGHT_MB in bytes; 0 disables the bound.
  size_t inflight_budget_{0};
  // Batches committed by the byte or op bound and not yet known complete.
  std::deque<InflightBatch> inflight_;
  size_t inflight_bytes_{0};
  std::vector<hipEvent_t> spare_events_;

  // --- Automatic graph-batching state (mirrors CUDA CommandEncoder) ---
  hipGraph_t build_graph_{nullptr};
  std::vector<hipGraphNode_t> from_nodes_;
  std::vector<hipGraphNode_t> to_nodes_;
  hipGraphNode_t last_node_{nullptr};
  std::string graph_nodes_key_;
  std::string graph_deps_key_;
  std::vector<std::uintptr_t> active_deps_;
  std::vector<std::uintptr_t> active_outputs_;
  std::unordered_map<std::uintptr_t, GraphNode> node_map_;
  size_t bytes_in_graph_{0};
  int max_ops_per_graph_{50};
  int max_mb_per_graph_{200};
  LRUCache<hipGraphExec_t> graph_cache_{400};
  // Per-topology exec pool for hipGraphExecUpdate reuse. The same kernel
  // sequence (e.g. one GDN layer's graph) recurs many times per token, all
  // launched async on one stream with NO sync between them. Updating a single
  // shared exec in-flight corrupts (a queued launch then reads another layer's
  // params). So we keep a pool per topology key and only ExecUpdate+reuse a
  // slot whose prior launch has drained (inflight==0); otherwise we grow the
  // pool. Across tokens (synced) every slot is free, so slot 0 is reused.
  // ROCm CLR (clr#138) stores kernel-node kernelParams BY POINTER, not a deep
  // copy (CUDA deep-copies). So a cached exec keeps reading the source graph's
  // nodes and our arg Packs by address — they must outlive the exec, not be
  // freed per-token. Each slot OWNS the graph + Packs its exec currently points
  // at; on reuse (the slot has drained, inflight==0) the old ones are released
  // and the new build's graph + Packs take their place.
  struct ExecSlot {
    hipGraphExec_t exec;
    std::shared_ptr<std::atomic<int>> inflight;
    hipGraph_t source_graph{nullptr};
    std::vector<std::shared_ptr<void>> packs;
    // Source-graph kernel nodes in insertion order (kept for the exec's life).
    std::vector<hipGraphNode_t> src_nodes;
  };

  std::unordered_map<std::string, std::vector<ExecSlot>> exec_pool_;
  // Per-build kernel-arg packs: keep the kernelParams values alive while the
  // (async) exec may reference them. Held one extra commit via _prev_.
  std::vector<std::shared_ptr<void>> graph_node_args_;
  std::vector<std::shared_ptr<void>> graph_node_args_prev_;
  // Current build's kernel nodes + their params, in insertion order.
  std::vector<hipGraphNode_t> build_nodes_;
  std::vector<hipKernelNodeParams> build_node_params_;

  // --- Build-once decode-graph replay ---
  // In decode mode the single-token forward has ONE stable topology (proven by
  // ExecUpdate succeeding every token). After the first build caches an exec,
  // we stop rebuilding the graph node-by-node: subsequent tokens buffer the
  // kernel params and only re-point the cached exec's nodes
  // (hipGraphExecKernelNodeSetParams) then relaunch. Eliminates ~1000
  // hipGraphAddKernelNode/token. Build-once decode replay via a PERSISTENT
  // source graph: the decode topology is stable, so we build its source graph
  // (build_graph_/build_nodes_) ONCE, then each token update the source nodes'
  // params with hipGraphKernelNodeSetParams (no AddKernelNode rebuild) and
  // hipGraphExecUpdate a drained pooled exec from it (the proven path).
  std::string decode_key_; // confirmed-stable decode topology key
  std::string pending_decode_key_; // last decode build's key (stability check)
  std::string prefill_key_; // confirmed-stable prefill-chunk topology key
  std::string
      pending_prefill_key_; // last prefill chunk's key (stability check)
  static std::string kernel_node_key(const hipKernelNodeParams& kp);

 public:
  // Full decode-step stream capture (build-once / replay). With graphs OFF the
  // whole call_fn launches eagerly on stream_; we capture it into ONE graph,
  // instantiate ONE exec, then relaunch that exec verbatim every token.
  // Captures EVERYTHING (manual kernels + library/JIT/memset), unlike the
  // manual-node chain which drops the ~1.4k inline ops. Returns false on any
  // failure so the engine falls back to eager.
  bool decode_capture_begin(); // begin stream capture (graphs must be off)
  bool decode_capture_end_record(
      int slot); // end capture, instantiate slot, launch
  bool decode_capture_replay(int slot); // relaunch the captured exec in slot
  void decode_capture_destroy(); // free both exec/graph slots
  bool decode_captured(int slot) const {
    return decode_cap_exec_[slot & 1] != nullptr;
  }

 private:
  hipGraphExec_t decode_cap_exec_[2]{
      nullptr,
      nullptr}; // per-parity captured exec
  hipGraph_t decode_cap_graph_[2]{
      nullptr,
      nullptr}; // their source graphs (owned)
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current HIP device, required by some HIP calls.
  void make_current();

  CommandEncoder& get_command_encoder(Stream s);
  void clear_encoders();

  // The encoder for |s| if one exists, without binding the device or creating
  // one. For the error path, which must not make a HIP call that can fail.
  CommandEncoder* find_encoder(Stream s);

  int hip_device() const {
    return device_;
  }

  rocblas_handle get_rocblas_handle();
  void set_rocblas_stream(hipStream_t stream);

  // Check if rocBLAS is available for the current GPU architecture
  bool is_rocblas_available();

  // Check if rocBLAS bf16 GEMM works on this device (probed at init)
  bool is_rocblas_bf16_available();

  // True iff this device's gcnArchName is on the rocWMMA arch allowlist
  // (CDNA1/2/3 + RDNA3 dGPU + RDNA3.5 gfx1150–1152 + RDNA4). Lazy-cached.
  bool has_native_wmma();

  // True iff CDNA2 (gfx90a) or CDNA3 (gfx942): clean bf16 MFMA
  // (v_mfma_f32_16x16x16bf16) AND a working hipBLASLt pointer-offset GEMM.
  // gfx1151 pegs on offset GEMMs, gfx908 bf16 MFMA differs, RDNA uses WMMA —
  // all fall back. Lazy-cached.
  bool supports_cdna_mfma_gemm();

  // Max shared memory (LDS) a single block may use on this device, in bytes,
  // queried from hipDeviceProp at construction. RDNA3/3.5 report 64 KB; RDNA4
  // and CDNA may report more. Kernels that size LDS tiles must read this from
  // the device actually running the op rather than assume a fixed budget.
  int max_shared_memory_per_block() const {
    return max_shared_memory_per_block_;
  }

  // Wavefront (warp) size of this device, queried from hipDeviceProp at
  // construction (CDNA/gfx9xx=64, RDNA=32). Host-side launch math (block dims,
  // warp counts) must use THIS runtime value, not a compile-time constant, or a
  // 32-vs-64 mismatch mis-reduces the warp kernels -> NaN.
  int warp_size() const {
    return warp_size_;
  }

  // Compute units (hipDeviceProp::multiProcessorCount). Reduced-CU RDNA3.5
  // partitions (e.g. gfx1152 VF) report far fewer CUs than desktop parts —
  // kernels should prefer larger tiles / fewer blocks when num_cus is low.
  int num_cus() const {
    return num_cus_;
  }

  int max_threads_per_block() const {
    return max_threads_per_block_;
  }

  // Max work-per-thread for fused Compiled kernels (vectorized loads).
  // Wave32 / low-CU iGPUs (e.g. gfx1152) are more fragile with 32B vectors
  // (WPT=16 for bf16); prefer 8 so we stay within always-instantiated, low
  // register-pressure specializations. CDNA/dGPU can use 16.
  int max_compiled_work_per_thread() const {
    int w = warp_size_ > 0 ? warp_size_ : 32;
    if (w <= 32 && num_cus_ > 0 && num_cus_ <= 16) {
      return 8;
    }
    return 16;
  }

  // Prefer block sizes that pack full wavefronts and stay under device limits.
  // cu-aware: on very small GPUs keep occupancy high with smaller blocks when
  // the work is tiny; still multiple of warp_size.
  int preferred_block_size(int work_items = -1) const {
    int w = warp_size_ > 0 ? warp_size_ : 32;
    int cap = max_threads_per_block_ > w ? max_threads_per_block_ : w;
    int bs = 256;
    if (num_cus_ > 0 && num_cus_ <= 8) {
      bs = 128; // low-CU: smaller blocks → more concurrent CUs for tiny grids
    }
    if (work_items > 0 && work_items < bs) {
      bs = ((work_items + w - 1) / w) * w;
      if (bs < w)
        bs = w;
    }
    if (bs > cap)
      bs = (cap / w) * w;
    return bs > w ? bs : w;
  }

 private:
  int device_;
  rocblas_handle rocblas_{nullptr};
  hipStream_t rocblas_stream_{nullptr};
  bool rocblas_initialized_{false};
  bool rocblas_available_{true};
  bool rocblas_bf16_probed_{false};
  bool rocblas_bf16_available_{false};
  bool wmma_probed_{false};
  bool has_native_wmma_{false};
  bool cdna_mfma_probed_{false};
  bool cdna_mfma_ok_{false};
  int max_shared_memory_per_block_{65536};
  int warp_size_{64};
  int num_cus_{0};
  int max_threads_per_block_{1024};
  std::unordered_map<int, std::unique_ptr<CommandEncoder>> encoders_;
  // MLX's scheduler runs a thread per stream, so get_command_encoder() can be
  // called concurrently (incl. cross-stream AtomicEvent::signal during weight
  // materialization). The map's find/emplace/rehash is not thread-safe — a
  // concurrent insert hands back a garbage encoder and crashes. Serialize it.
  std::mutex encoders_mtx_;
};

// Put `device_index` in blocking-sync wait mode (hipDeviceScheduleBlockingSync)
// once per process, before MLX creates the first HIP queue on it. Thread-safe;
// touches only that device, and leaves the calling thread's current device as
// it found it. Call it before any HIP call that can create a queue on a device
// (a null-stream query, sync, copy or launch, or a stream creation) on a path
// that can run before rocm::device(): the order is what matters. CLR types a
// queue's completion signals by the wait mode in force when the queue is
// created and never re-types them, so a queue created before the flag keeps
// signals that cannot carry the completion handlers a later hipFree or device
// sync attaches, and that wait never returns (lablup/mlxcel#1876).
void ensure_device_flags(int device_index);
// ensure_device_flags() for the calling thread's current HIP device.
void ensure_current_device_flags();

Device& device(mlx::core::Device device);
CommandEncoder& get_command_encoder(Stream s);
void clear_all_encoders();

// True while a HIP graph capture is in progress on any stream. Lazy library
// inits that abort under capture (e.g. hipblasLtCreate) check this.
bool stream_capturing();
void set_stream_capturing(bool v);
void set_graph_active(bool v);

// True from capture start until the captured graph is destroyed. The allocator
// defers all frees while set so graph-referenced buffers stay valid through
// replay.
bool graph_active();

// Decode-mode: single-token forward = one graph, refreshed via ExecUpdate and
// launched once/token. Set per-step by the generation loop; MLX_GRAPH_DECODE=0
// disables. See device.cpp.
bool graph_decode_mode();
void set_graph_decode_mode(bool v);
void flush_graph_deferred_frees();
// Per-generation deferred-free lifetime: each graph chunk (commit) frees its
// own buffers once its launch completes, instead of hoarding until synchronize.
uint64_t graph_current_gen();
void graph_advance_gen();
void free_graph_generation(uint64_t gen);
// Reclaim a generation's stream-ordered pool buffers via hipFreeAsync on the
// generation stream (no blocking drain); see allocator.cpp.
void free_graph_generation_async(uint64_t gen);
// Bytes currently held in the graph deferred-free backlog. commit() drains it
// (sync + flush) once it exceeds a cap, bounding graph-mode peak memory.
size_t graph_deferred_bytes();

// Return an execution policy that does not sync for result.
// Only available when compiling with HIP compiler
#ifdef __HIPCC__
inline auto thrust_policy(hipStream_t stream) {
  return thrust::hip::par.on(stream);
}
#endif

// Template implementation (must be after Device is defined)
template <typename F>
void CommandEncoder::launch_kernel(F&& func) {
  device_.make_current();
  // Residual ops not migrated to add_kernel_node (library GEMM, JIT module
  // kernels, memsets) can't be HIP graph kernel nodes. Graph-split:
  // flush+launch the accumulated graph, run this op immediately on the same
  // stream (ordered after the graph), and start the next graph fresh.
  if (use_hip_graphs()) {
    static const bool capture_lib = [] {
      const char* e = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
      return e && e[0] == '1';
    }();
    auto hstream = static_cast<hipStream_t>(stream_);
    if (capture_lib && !graph_decode_mode()) {
      hipError_t be =
          hipStreamBeginCapture(hstream, hipStreamCaptureModeThreadLocal);
      if (be == hipSuccess) {
        // A launch that throws (a rejected hipModuleLaunchKernel) must not
        // leave the stream capturing, or every later operation on it fails
        // too. End the capture and retry eagerly below, which rethrows a
        // genuine launch failure outside the capture.
        bool launched = true;
        try {
          func(hstream);
        } catch (...) {
          launched = false;
        }
        hipGraph_t child = nullptr;
        hipError_t ee = hipStreamEndCapture(hstream, &child);
        if (!launched) {
          ee = hipErrorStreamCaptureInvalidated;
        }
        // Failures here fall back to the eager launch below; the graph
        // handles are released without checking (an already-failed capture
        // has nothing better to report) and the pending error is cleared.
        if (ee == hipSuccess && child) {
          size_t nn = 0;
          (void)hipGraphGetNodes(child, nullptr, &nn);
          if (nn > 0) {
            add_child_graph_node(child, "lib");
            (void)hipGraphDestroy(child);
            return;
          }
          (void)hipGraphDestroy(child);
        } else if (child) {
          (void)hipGraphDestroy(child);
        }
        (void)hipGetLastError();
      } else {
        if (std::getenv("MLX_GRAPH_SPLIT_LOG"))
          fprintf(stderr, "[cap-fail] begin err=%s\n", hipGetErrorString(be));
        (void)hipGetLastError();
      }
    }
    commit();
    func(hstream);
    record_inline_launch();
    return;
  }
  func(static_cast<hipStream_t>(stream_));
  node_count_++;
}

} // namespace mlx::core::rocm
