// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include <algorithm>
#include <atomic>
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/worker.h"
#include "mlx/utils.h"

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace mlx::core::rocm {

namespace {

// Can be tuned with MLX_MAX_OPS_PER_BUFFER
constexpr int default_max_ops_per_buffer = 2000;

inline bool is_empty_dim(dim3 dim) {
  return (dim.x == 0 && dim.y == 0 && dim.z == 0) ||
      (dim.x == 1 && dim.y == 1 && dim.z == 1);
}

} // namespace

// True while the engine is in a single-token decode step; false during prefill
// (multi-token) and outside generation. Set by set_graph_decode_mode().
static std::atomic<bool> g_graph_decode_mode{false};

bool use_hip_graphs() {
  // The rebuild-every-eval graph-build path is a net loss vs eager (and the
  // prefill variant segfaults on RDNA3.5). Decode uses build-once
  // capture/replay (decode_capture_*) and prefill uses the WMMA GEMM — neither
  // goes through here — so this path is permanently off.
  //
  // Train graphs: also hard-off until a TrainArena (see RocmAllocator
  // train_arena_*) wraps the step tape with stable addresses. Flipping this
  // without an arena reintroduces aperture violations / wrong loss.
  return false;
}

// Count of inline (graph-splitting) launches — library GEMM / JIT / memset run
// outside the graph via launch_kernel. Declared extern in device.h.
std::atomic<long> g_inline_launches_{0};
// True while a full decode-step stream capture is in progress. The worker's
// completion signaling uses hipLaunchHostFunc, which is NOT stream-capturable
// (and a captured host node would re-fire on every replay). While capturing we
// skip the host callback and signal completion inline instead (see Worker).
std::atomic<bool> g_decode_capturing{false};
void set_current_prim(const char*) {}
void record_inline_launch() {
  g_inline_launches_.fetch_add(1, std::memory_order_relaxed);
}

// MLX_ROCM_MAX_INFLIGHT_MB: how much newly allocated memory (outputs and
// scratch of encoded primitives) the eager path lets committed-but-unfinished
// command batches pin, in MiB (lablup/mlxcel#2062). A batch's buffers are only
// released by its completion handler, and the host encodes much faster than
// the GPU runs a prefill, so without a bound the temporaries of many batches
// are live at once: 15.85 GB of them, mostly the f16 weight copies of the
// dequantize-and-GEMM qmm path, on top of 4.75 GB of 4-bit weights for
// Llama-3.1-8B at 512 prompt tokens on gfx1151.
// A batch is committed once it has pinned a quarter of the budget, and the
// host waits for the oldest committed batch while more than the budget is in
// flight. 0 turns the bound off (the previous behavior: commit every
// MLX_MAX_OPS_PER_BUFFER ops, never wait).
constexpr size_t default_max_inflight_mb = 1024;

size_t max_inflight_bytes() {
  static const size_t bytes = [] {
    size_t mb = default_max_inflight_mb;
    if (const char* e = std::getenv("MLX_ROCM_MAX_INFLIGHT_MB"); e && *e) {
      // Whole string a non-negative decimal integer small enough to scale to
      // bytes; anything else keeps the default rather than guessing.
      char* end = nullptr;
      errno = 0;
      unsigned long long v = std::strtoull(e, &end, 10);
      if (end == e || *end != '\0' || errno == ERANGE || e[0] == '-' ||
          v > (std::numeric_limits<size_t>::max() >> 20)) {
        std::fprintf(
            stderr,
            "[ROCm] ignoring invalid MLX_ROCM_MAX_INFLIGHT_MB=\"%s\" "
            "(expected a non-negative integer); using %zu\n",
            e,
            default_max_inflight_mb);
      } else {
        mb = static_cast<size_t>(v);
      }
    }
    return mb << 20;
  }();
  return bytes;
}

// Per-arch op/MB caps for the build graph. Tunable via env.
// The earlier "corrupts at >3 nodes" was actually one bad op (Concatenate,
// whose multi-copy kernels corrupt when co-grouped); it is now graph-split in
// gpu::eval (is_graph_split_op), so large graphs are correct again.
static std::pair<int, int> get_graph_limits() {
  int ops = env::max_ops_per_buffer(50);
  int mb = env::max_mb_per_buffer(200);
  return {ops, mb};
}

Device::Device(int device) : device_(device) {
  make_current();
  {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, device_) == hipSuccess) {
      fprintf(
          stderr,
          "[mlx-rocm] bound HIP device %d: %s (%s) cus=%d warp=%d lds=%dKB\n",
          device_,
          p.gcnArchName,
          p.name,
          p.multiProcessorCount,
          p.warpSize,
          p.sharedMemPerBlock / 1024);
      if (p.sharedMemPerBlock > 0) {
        max_shared_memory_per_block_ = static_cast<int>(p.sharedMemPerBlock);
      }
      if (p.warpSize > 0) {
        warp_size_ = p.warpSize;
      }
      if (p.multiProcessorCount > 0) {
        num_cus_ = p.multiProcessorCount;
      }
      if (p.maxThreadsPerBlock > 0) {
        max_threads_per_block_ = p.maxThreadsPerBlock;
      }
      // Optional debug/experiment override (RDNA is hardware wave32 including
      // RDNA4; only set this if you know you need a different launch width).
      // Wrong value vs device kernels → garbage QMV / decode.
      if (const char* fw = std::getenv("MLX_ROCM_FORCE_WARP_SIZE")) {
        int forced = std::atoi(fw);
        if (forced == 32 || forced == 64) {
          fprintf(
              stderr,
              "[mlx-rocm] MLX_ROCM_FORCE_WARP_SIZE=%d overrides device warp=%d\n",
              forced,
              warp_size_);
          warp_size_ = forced;
        } else {
          fprintf(
              stderr,
              "[mlx-rocm] ignoring MLX_ROCM_FORCE_WARP_SIZE=%s (want 32 or 64)\n",
              fw);
        }
      }
    }
  }
  // rocBLAS initialization is now lazy - done in get_rocblas_handle()
}

Device::~Device() {
  if (rocblas_) {
    rocblas_destroy_handle(rocblas_);
  }
}

rocblas_handle Device::get_rocblas_handle() {
  if (!rocblas_initialized_) {
    rocblas_initialized_ = true;
    make_current();

    // Check if the GPU architecture is supported by rocBLAS
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_);
    std::string arch_name = props.gcnArchName;

    // List of architectures supported by rocBLAS (based on TensileLibrary
    // files). These are the architectures that have TensileLibrary_lazy_*.dat.
    static const std::vector<std::string> supported_archs = {
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1200",
        "gfx1201"};

    // Extract base architecture name (remove any suffix like :sramecc+:xnack-)
    std::string base_arch = arch_name;
    size_t colon_pos = base_arch.find(':');
    if (colon_pos != std::string::npos) {
      base_arch = base_arch.substr(0, colon_pos);
    }

    bool arch_supported = false;
    for (const auto& supported : supported_archs) {
      if (base_arch == supported) {
        arch_supported = true;
        break;
      }
    }

    if (!arch_supported) {
      rocblas_available_ = false;
      rocblas_ = nullptr;
      std::cerr << "Warning: rocBLAS does not support GPU architecture '"
                << arch_name << "'. "
                << "Matrix multiplication operations will not be available. "
                << "Supported architectures: gfx908, gfx90a, gfx942, gfx950, "
                << "gfx1030, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, "
                << "gfx1152, gfx1200, gfx1201." << std::endl;
    } else {
      rocblas_status status = rocblas_create_handle(&rocblas_);
      if (status != rocblas_status_success) {
        rocblas_available_ = false;
        rocblas_ = nullptr;
        std::cerr
            << "Warning: rocBLAS initialization failed (status "
            << static_cast<int>(status)
            << "). Matrix multiplication operations will not be available."
            << std::endl;
      }
    }
  }
  if (!rocblas_available_) {
    throw std::runtime_error(
        "rocBLAS is not available on this GPU architecture. "
        "Matrix multiplication operations are not supported.");
  }
  return rocblas_;
}

bool Device::is_rocblas_available() {
  if (!rocblas_initialized_) {
    try {
      get_rocblas_handle();
    } catch (...) {
    }
  }
  return rocblas_available_;
}

bool Device::is_rocblas_bf16_available() {
  if (!rocblas_bf16_probed_) {
    rocblas_bf16_probed_ = true;
    rocblas_bf16_available_ = false;

    if (!is_rocblas_available()) {
      return false;
    }

    // Probe: run a tiny bf16 GEMM and check if the GPU survives.
    // rocBLAS may claim support but crash if the Tensile .co files
    // are corrupt or missing specific kernel variants.
    make_current();
    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    void* c_ptr = nullptr;
    hipError_t err;

    err = hipMalloc(&a_ptr, 4 * 4 * 2); // 4x4 bf16
    if (err != hipSuccess)
      return false;
    err = hipMalloc(&b_ptr, 4 * 4 * 2);
    if (err != hipSuccess) {
      hipFree(a_ptr);
      return false;
    }
    err = hipMalloc(&c_ptr, 4 * 4 * 2);
    if (err != hipSuccess) {
      hipFree(a_ptr);
      hipFree(b_ptr);
      return false;
    }

    (void)hipMemset(a_ptr, 0, 4 * 4 * 2);
    (void)hipMemset(b_ptr, 0, 4 * 4 * 2);
    (void)hipMemset(c_ptr, 0, 4 * 4 * 2);

    float alpha = 1.0f, beta = 0.0f;
    rocblas_status status = rocblas_gemm_ex(
        rocblas_,
        rocblas_operation_none,
        rocblas_operation_none,
        4,
        4,
        4,
        &alpha,
        a_ptr,
        rocblas_datatype_bf16_r,
        4,
        b_ptr,
        rocblas_datatype_bf16_r,
        4,
        &beta,
        c_ptr,
        rocblas_datatype_bf16_r,
        4,
        c_ptr,
        rocblas_datatype_bf16_r,
        4,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0,
        0);

    // Sync and check if the GPU is still alive
    hipError_t sync_err = hipDeviceSynchronize();
    // Clear any lingering error
    (void)hipGetLastError();

    hipFree(a_ptr);
    hipFree(b_ptr);
    hipFree(c_ptr);

    if (status == rocblas_status_success && sync_err == hipSuccess) {
      rocblas_bf16_available_ = true;
    } else {
      // GPU may be in a bad state — need to reset
      (void)hipDeviceReset();
      // Re-initialize device
      make_current();
      // Re-create rocBLAS handle
      if (rocblas_) {
        rocblas_destroy_handle(rocblas_);
        rocblas_ = nullptr;
      }
      rocblas_status rs = rocblas_create_handle(&rocblas_);
      if (rs != rocblas_status_success) {
        rocblas_available_ = false;
      }
      std::cerr << "Warning: rocBLAS bfloat16 GEMM probe failed on this GPU. "
                << "Using fallback kernels for bf16 matmul." << std::endl;
    }
  }
  return rocblas_bf16_available_;
}

bool Device::has_native_wmma() {
  if (!wmma_probed_) {
    wmma_probed_ = true;

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_) != hipSuccess) {
      has_native_wmma_ = false;
      return has_native_wmma_;
    }

    // Strip any ":sramecc+:xnack-" style suffix from gcnArchName.
    std::string base_arch = props.gcnArchName;
    size_t colon_pos = base_arch.find(':');
    if (colon_pos != std::string::npos) {
      base_arch = base_arch.substr(0, colon_pos);
    }

    // rocWMMA arch allowlist. Keep in sync with detect_rocm_hw_info() in
    // mlx/backend/rocm/quantized/qmm.hip. RDNA3.5: gfx1150/1151/1152 all have
    // WMMA; Device used to omit 1150/1152 which forced flash/qmm off on those
    // parts (incl. reduced-CU gfx1152 instances).
    static const std::vector<std::string> rocwmma_archs = {
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
        "gfx1200",
        "gfx1201",
    };
    for (const auto& a : rocwmma_archs) {
      if (base_arch == a) {
        has_native_wmma_ = true;
        break;
      }
    }
  }
  return has_native_wmma_;
}

bool Device::supports_cdna_mfma_gemm() {
  if (!cdna_mfma_probed_) {
    cdna_mfma_probed_ = true;
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_) != hipSuccess) {
      cdna_mfma_ok_ = false;
      return cdna_mfma_ok_;
    }
    std::string base_arch = props.gcnArchName;
    size_t colon_pos = base_arch.find(':');
    if (colon_pos != std::string::npos) {
      base_arch = base_arch.substr(0, colon_pos);
    }
    // CDNA2 (gfx90a) + CDNA3 (gfx942) only: clean v_mfma_f32_16x16x16bf16 and a
    // working hipBLASLt pointer-offset GEMM. gfx1151 (RDNA3.5) pegs on the
    // offset path; gfx908 (CDNA1) bf16 MFMA differs; RDNA/others use WMMA.
    cdna_mfma_ok_ = (base_arch == "gfx90a" || base_arch == "gfx942");
  }
  return cdna_mfma_ok_;
}

void Device::make_current() {
  // HIP's current device is per-thread, so the cache must be too — a process
  // global lets one thread's binding suppress another's, stranding allocations
  // on the wrong device in a multi-GPU / multi-stream-thread run.
  thread_local int current = -1;
  if (current != device_) {
    CHECK_HIP_ERROR(hipSetDevice(device_));
    current = device_;
  }
}

void Device::set_rocblas_stream(hipStream_t stream) {
  if (rocblas_stream_ != stream) {
    rocblas_set_stream(get_rocblas_handle(), stream);
    rocblas_stream_ = stream;
  }
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  // Bind this device current before constructing/returning the encoder. Callers
  // reach this member directly (e.g. QuantizedMatmul::eval_gpu), and the
  // encoder's stream + the kernels launched on it must land on this device, not
  // whatever was current on the calling thread.
  make_current();
  std::lock_guard<std::mutex> lk(encoders_mtx_);
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    auto [inserted_it, success] =
        encoders_.emplace(s.index, std::make_unique<CommandEncoder>(*this));
    it = inserted_it;
  }
  return *it->second;
}

void Device::clear_encoders() {
  encoders_.clear();
}

CommandEncoder* Device::find_encoder(Stream s) {
  std::lock_guard<std::mutex> lk(encoders_mtx_);
  auto it = encoders_.find(s.index);
  return it == encoders_.end() ? nullptr : it->second.get();
}

CommandEncoder::CommandEncoder(Device& d)
    : device_(d),
      stream_(d),
      worker_(std::make_unique<Worker>(d.hip_device())) {
  std::tie(max_ops_per_graph_, max_mb_per_graph_) = get_graph_limits();
  inflight_budget_ = max_inflight_bytes();
  if (use_hip_graphs()) {
    device_.make_current();
    CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
    set_graph_active(true);
  }
}

CommandEncoder::~CommandEncoder() {
  // Destructor path: a failed destroy has nowhere to go, and on a device that
  // has faulted every one of these returns the fault.
  release_inflight();
  for (hipEvent_t ev : spare_events_) {
    (void)hipEventDestroy(ev);
  }
  for (auto& [key, pool] : exec_pool_) {
    for (auto& slot : pool) {
      (void)hipGraphExecDestroy(slot.exec);
      if (slot.source_graph) {
        (void)hipGraphDestroy(slot.source_graph);
      }
    }
  }
  if (build_graph_) {
    (void)hipGraphDestroy(build_graph_);
    build_graph_ = nullptr;
  }
}

// --- GPU failure reporting (lablup/mlxcel#1804) -----------------------------

int gpu_watchdog_seconds() {
  static const int secs = [] {
    const char* e = std::getenv("MLX_ROCM_GPU_WATCHDOG_SECS");
    if (!e || !*e) {
      return 0;
    }
    // The whole string must be a non-negative decimal integer that fits an
    // int. Anything else (trailing junk such as "10s", overflow, a negative
    // value) leaves the watchdog off rather than guessing what was meant.
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE || v < 0 ||
        v > std::numeric_limits<int>::max()) {
      std::fprintf(
          stderr,
          "[ROCm] ignoring invalid MLX_ROCM_GPU_WATCHDOG_SECS=\"%s\" "
          "(expected a non-negative integer); the GPU watchdog stays off\n",
          e);
      return 0;
    }
    return static_cast<int>(v);
  }();
  return secs;
}

std::string describe_device_error(hipError_t status, const char* where) {
  std::ostringstream oss;
  if (status == kWatchdogExpired) {
    oss << "[ROCm] GPU watchdog: a host wait (" << where << ") exceeded "
        << "MLX_ROCM_GPU_WATCHDOG_SECS=" << gpu_watchdog_seconds()
        << " seconds. The GPU stream is still busy or stuck; work queued "
        << "behind it will not complete, so restart the process to use the "
        << "GPU again.";
    return oss.str();
  }
  oss << "[ROCm] GPU stream failed while " << where << ": "
      << hipGetErrorString(status) << " (" << hipGetErrorName(status) << ", "
      << static_cast<int>(status) << ").";
  // After a queue fault the runtime fails every HIP call on every thread,
  // hipGetDevice included; say so rather than let the caller retry.
  int dev = 0;
  bool context_gone = hipGetDevice(&dev) != hipSuccess;
  (void)hipGetLastError();
  if (context_gone) {
    oss << " The HIP runtime reported the faulting kernel on stderr. The "
        << "device context is unusable for the rest of the process (every "
        << "HIP call now returns this error), so restart the process to use "
        << "the GPU again.";
  }
  return oss.str();
}

void CommandEncoder::set_device_error(hipError_t status, const char* where) {
  // Keep the earliest error, as the Metal completion handler does.
  if (error_.valid()) {
    return;
  }
  error_.set_message(
      std::make_shared<std::string>(describe_device_error(status, where)));
}

void CommandEncoder::check_launch(const char* primitive) {
  hipError_t status = hipGetLastError();
  if (status == hipSuccess) {
    return;
  }
  // A failed allocation that was recovered from leaves hipErrorOutOfMemory
  // pending too: the allocator retries after releasing its cache or falls
  // back to managed memory, and hipBLASLt runs without a workspace when its
  // hipMalloc fails. An allocation that was not recovered from has already
  // thrown, and a kernel launch does not report this status, so it is not
  // this primitive's launch failure.
  if (status == hipErrorOutOfMemory) {
    return;
  }
  int dev = 0;
  bool context_gone = hipGetDevice(&dev) != hipSuccess;
  (void)hipGetLastError();
  if (context_gone) {
    std::string where = std::string("launching ") + primitive;
    set_device_error(status, where.c_str());
    error_.check();
  }
  std::ostringstream oss;
  oss << "[ROCm] launching " << primitive
      << " failed: " << hipGetErrorString(status) << " ("
      << hipGetErrorName(status) << ", " << static_cast<int>(status) << ").";
  throw std::runtime_error(oss.str());
}

std::unordered_map<int, Device>& get_devices();

Error& record_stream_error(Stream s, hipError_t status, const char* where) {
  if (s.device.type == mlx::core::Device::gpu) {
    auto& devices = get_devices();
    if (auto it = devices.find(s.device.index); it != devices.end()) {
      if (auto* encoder = it->second.find_encoder(s)) {
        encoder->set_device_error(status, where);
        return encoder->error();
      }
    }
  }
  // Leaked on purpose: Event::set_error stores a raw pointer, so the object
  // must outlive every event that could point at it.
  static Error* fallback = new Error();
  if (!fallback->valid()) {
    fallback->set_message(
        std::make_shared<std::string>(describe_device_error(status, where)));
  }
  return *fallback;
}

void CommandEncoder::add_temporary(const array& arr) {
  auto data = arr.data_shared_ptr();
  const array::Data* ptr = data.get();
  if (temporary_ptrs_.insert(ptr).second) {
    temporaries_.push_back(std::move(data));
  }
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_->add_task(std::move(task));
}

void CommandEncoder::set_input_array(const array& arr) {
  if (!use_hip_graphs()) {
    return;
  }
  bytes_in_graph_ += arr.data_size();
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
}

void CommandEncoder::set_output_array(const array& arr) {
  if (!use_hip_graphs()) {
    return;
  }
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
  active_outputs_.push_back(id);
}

void CommandEncoder::insert_graph_dependencies(GraphNode node) {
  node.id = std::to_string(node_count_++);
  std::vector<GraphNode> nodes;
  nodes.push_back(std::move(node));
  insert_graph_dependencies(std::move(nodes));
}

void CommandEncoder::insert_graph_dependencies(std::vector<GraphNode> nodes) {
  // Two edge sets combined:
  //  (1) REAL data-dependency edges (node_map_: buffer ptr -> producing node).
  //      These give each node a unique data-flow identity, which
  //      hipGraphExecUpdate relies on to re-map kernelParams correctly across
  //      reuse — a bare submission chain leaves same-signature nodes ambiguous
  //      and corrupts under ExecUpdate.
  //  (2) A submission-order CHAIN edge (last_node_ -> node) as a backstop. Not
  //      every migrated kernel registers all of its I/O (the dense GEMM path
  //      and a few others register nothing), so the real-dep graph alone has
  //      missing edges -> races -> coherent-but-wrong output. The chain is a
  //      superset of the true partial order (eager runs serial on one stream),
  //      so it fills every gap and makes the graph bit-correct vs eager, while
  //      the real edges still uniquely identify nodes for ExecUpdate.
  for (auto& node : nodes) {
    graph_nodes_key_ += node.node_type;
    graph_nodes_key_ += "-";
  }
  std::vector<GraphNode> deps;
  std::unordered_set<hipGraphNode_t> set_deps;
  for (auto d : active_deps_) {
    if (auto it = node_map_.find(d); it != node_map_.end()) {
      if (set_deps.insert(it->second.node).second) {
        deps.push_back(it->second);
      }
    }
  }
  active_deps_.clear();

  for (auto o : active_outputs_) {
    for (auto& node : nodes) {
      node_map_.emplace(o, node).first->second = node;
    }
  }
  active_outputs_.clear();

  for (auto& from : deps) {
    for (auto& to : nodes) {
      from_nodes_.push_back(from.node);
      to_nodes_.push_back(to.node);
      graph_deps_key_ += from.id;
      graph_deps_key_ += "-";
      graph_deps_key_ += to.id;
      graph_deps_key_ += "-";
    }
  }

  // (2) Submission-order CHAIN edge. Serialize every node behind the previously
  // inserted node so the graph's execution order is a superset of eager's
  // serial-on-one-stream order — even for kernels that register no I/O (dense
  // GEMM path and a few others), whose nodes would otherwise have no edges and
  // race their true producers/consumers at any cap > 1. The chain is
  // deterministic for a given node sequence, so identical kernel sequences
  // still produce identical topology and hipGraphExecUpdate stays valid. Skip
  // an edge already present as a real data dep (avoid an exact duplicate edge).
  hipGraphNode_t prev = last_node_;
  for (auto& to : nodes) {
    // A real-dep edge prev->to was already added iff prev is in set_deps; skip
    // then so we never add an exact duplicate edge (HIP rejects duplicates).
    // Internal batch nodes are never external producers, so they always chain.
    if (prev && !set_deps.count(prev)) {
      from_nodes_.push_back(prev);
      to_nodes_.push_back(to.node);
      graph_deps_key_ += "c-";
    }
    prev = to.node;
  }
  last_node_ = prev;
}

void CommandEncoder::add_kernel_node_raw(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params) {
  if (!use_hip_graphs()) {
    device_.make_current();
    CHECK_HIP_ERROR(hipLaunchKernel(
        func, grid_dim, block_dim, params, smem_bytes, stream_));
    node_count_++;
    return;
  }

  hipKernelNodeParams kernel_params = {};
  kernel_params.func = func;
  kernel_params.gridDim = grid_dim;
  kernel_params.blockDim = block_dim;
  kernel_params.kernelParams = params;
  kernel_params.sharedMemBytes = smem_bytes;
  add_kernel_node_kp(kernel_params);
}

// Key the node by its kernel FUNCTION + FULL launch dims so a reused exec is
// re-pointed only onto a structurally identical graph. The grid/block PRODUCT
// collided distinct shapes (e.g. 2097152x1x1 vs 1024x2048x1); encode each dim.
std::string CommandEncoder::kernel_node_key(const hipKernelNodeParams& kp) {
  std::string key = "K";
  key += std::to_string(reinterpret_cast<std::uintptr_t>(kp.func));
  key += "_";
  key += std::to_string(kp.gridDim.x);
  key += ",";
  key += std::to_string(kp.gridDim.y);
  key += ",";
  key += std::to_string(kp.gridDim.z);
  key += "x";
  key += std::to_string(kp.blockDim.x);
  key += ",";
  key += std::to_string(kp.blockDim.y);
  key += ",";
  key += std::to_string(kp.blockDim.z);
  key += "s";
  key += std::to_string(kp.sharedMemBytes);
  return key;
}

void CommandEncoder::add_kernel_node_kp(const hipKernelNodeParams& kp) {
  // A launch with any zero grid/block dim is a no-op: hipLaunchKernel tolerates
  // it (eager path), but hipGraphAddKernelNode rejects it with "invalid
  // argument". Skip it entirely — it does no work, and skipping is consistent
  // across tokens so the decode topology stays stable for replay.
  if (kp.func == nullptr || kp.gridDim.x == 0 || kp.gridDim.y == 0 ||
      kp.gridDim.z == 0 || kp.blockDim.x == 0 || kp.blockDim.y == 0 ||
      kp.blockDim.z == 0) {
    return;
  }
  std::string key = kernel_node_key(kp);

  hipGraphNode_t node;
  hipError_t kn_err =
      hipGraphAddKernelNode(&node, build_graph_, nullptr, 0, &kp);
  if (kn_err != hipSuccess) {
    // ROCm's hipGraphAddKernelNode rejects some kernels with invalid-argument
    // (certain WMMA / single-block / JIT kernels). In replay mode, don't split
    // — CAPTURE the launch into a child graph node (CUDA's model: capture
    // records any launched kernel that the manual node API won't accept), so
    // the chunk stays ~1 fragment. Else fall back to the eager graph-split.
    (void)hipGetLastError();
    static const bool cap_reject = [] {
      const char* e = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
      return e && e[0] == '1';
    }();
    if (cap_reject && !graph_decode_mode()) {
      device_.make_current();
      hipError_t be =
          hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
      if (be == hipSuccess) {
        (void)hipLaunchKernel(
            kp.func,
            kp.gridDim,
            kp.blockDim,
            kp.kernelParams,
            kp.sharedMemBytes,
            stream_);
        hipGraph_t child = nullptr;
        hipError_t ee = hipStreamEndCapture(stream_, &child);
        if (ee == hipSuccess && child) {
          size_t nn = 0;
          hipGraphGetNodes(child, nullptr, &nn);
          if (nn > 0) {
            add_child_graph_node(child, key);
            hipGraphDestroy(child);
            return;
          }
          hipGraphDestroy(child);
        } else if (child) {
          hipGraphDestroy(child);
        }
        (void)hipGetLastError();
      }
    }
    commit();
    device_.make_current();
    (void)hipLaunchKernel(
        kp.func,
        kp.gridDim,
        kp.blockDim,
        kp.kernelParams,
        kp.sharedMemBytes,
        stream_);
    record_inline_launch();
    return;
  }
  build_nodes_.push_back(node);
  build_node_params_.push_back(kp);
  insert_graph_dependencies(GraphNode{node, key});
}

void CommandEncoder::add_module_kernel_node(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params,
    std::shared_ptr<void> args_keepalive) {
  if (!use_hip_graphs()) {
    device_.make_current();
    CHECK_HIP_ERROR(hipModuleLaunchKernel(
        reinterpret_cast<hipFunction_t>(func),
        grid_dim.x,
        grid_dim.y,
        grid_dim.z,
        block_dim.x,
        block_dim.y,
        block_dim.z,
        smem_bytes,
        stream_,
        params,
        nullptr));
    node_count_++;
    return;
  }
  // Graph path: the node references `params` (which point into the kept-alive
  // KernelArgs) until commit instantiates the graph. A module hipFunction_t is
  // a valid hipKernelNodeParams.func on ROCm 7.13 (see device.h note).
  if (args_keepalive) {
    graph_node_args_.push_back(std::move(args_keepalive));
  }
  add_kernel_node_raw(func, grid_dim, block_dim, smem_bytes, params);
}

void CommandEncoder::add_child_graph_node(
    hipGraph_t child,
    const std::string& key) {
  // hipGraphExecUpdate does NOT refresh kernel params nested inside child-graph
  // nodes (confirmed: it returns success but the child keeps stale kernargs),
  // so embedding a child node silently breaks graph reuse across tokens.
  // Flatten the child's kernels into build_graph_ as TOP-LEVEL kernel nodes via
  // the normal add path so ExecUpdate refreshes them. The child's kernelParams
  // point at the caller's arg storage (kept alive like any other op), and
  // chain-edge ordering serializes the flattened nodes correctly. Fall back to
  // embedding the child as-is only if it contains non-kernel nodes (none
  // observed in practice).
  size_t n = 0;
  hipGraphGetNodes(child, nullptr, &n);
  std::vector<hipGraphNode_t> cnodes(n);
  if (n) {
    hipGraphGetNodes(child, cnodes.data(), &n);
  }
  // Topologically order the child's kernels (Kahn) so chain-edge serialization
  // never places a consumer before its producer.
  size_t ne = 0;
  hipGraphGetEdges(child, nullptr, nullptr, &ne);
  std::vector<hipGraphNode_t> cfrom(ne), cto(ne);
  if (ne) {
    hipGraphGetEdges(child, cfrom.data(), cto.data(), &ne);
  }
  bool all_kernels = n > 0;
  for (size_t i = 0; i < n; i++) {
    hipGraphNodeType t;
    if (hipGraphNodeGetType(cnodes[i], &t) != hipSuccess ||
        t != hipGraphNodeTypeKernel) {
      all_kernels = false;
      break;
    }
  }
  if (all_kernels) {
    std::unordered_map<hipGraphNode_t, int> indeg;
    std::unordered_map<hipGraphNode_t, std::vector<hipGraphNode_t>> succ;
    for (auto nd : cnodes)
      indeg[nd] = 0;
    for (size_t i = 0; i < ne; i++) {
      succ[cfrom[i]].push_back(cto[i]);
      indeg[cto[i]]++;
    }
    std::vector<hipGraphNode_t> order;
    order.reserve(n);
    for (auto nd : cnodes)
      if (indeg[nd] == 0)
        order.push_back(nd);
    for (size_t h = 0; h < order.size(); h++)
      for (auto s : succ[order[h]])
        if (--indeg[s] == 0)
          order.push_back(s);
    if (order.size() == n) {
      bool ok = true;
      std::vector<hipKernelNodeParams> kps(n);
      for (size_t i = 0; i < n && ok; i++) {
        kps[i] = {};
        if (hipGraphKernelNodeGetParams(order[i], &kps[i]) != hipSuccess)
          ok = false;
      }
      if (ok) {
        for (size_t i = 0; i < n; i++) {
          add_kernel_node_kp(
              kps[i]); // preserve full params (kernelParams+extra)
        }
        return;
      }
    }
  }
  // Fallback: embed the child as-is (non-kernel nodes or topo failure).
  hipGraphNode_t node;
  CHECK_HIP_ERROR(
      hipGraphAddChildGraphNode(&node, build_graph_, nullptr, 0, child));
  insert_graph_dependencies(GraphNode{node, key});
}

void CommandEncoder::maybe_commit() {
  if (use_hip_graphs()) {
    if (needs_commit()) {
      commit();
    }
    return;
  }
  if (needs_commit()) {
    commit_and_throttle();
  }
}

void CommandEncoder::commit_and_throttle() {
  const size_t bytes = batch_bytes_;
  commit();
  throttle_inflight(bytes);
}

bool CommandEncoder::needs_commit() {
  if (!use_hip_graphs()) {
    return node_count_ >= env::max_ops_per_buffer(default_max_ops_per_buffer) ||
        (inflight_budget_ > 0 && batch_bytes_ >= inflight_budget_ / 4);
  }
  // Decode-mode: never split mid-forward — the whole single-token forward
  // becomes one graph, committed once at finalize, refreshed via ExecUpdate.
  // Decode's live intermediates are small, so the per-graph caps (which bound
  // prefill) aren't needed here.
  if (graph_decode_mode()) {
    return false;
  }
  return (node_count_ > max_ops_per_graph_) ||
      ((bytes_in_graph_ >> 20) > static_cast<size_t>(max_mb_per_graph_));
}

// --- Full decode-step stream capture (build-once / replay) ------------------
// The manual-node chain can't capture the ~1.4k library/JIT/memset ops that go
// through launch_kernel during decode. Instead, run the whole forward with
// graphs OFF (every kernel launches on stream_), wrap it in one stream capture,
// and relaunch that single exec each token. The deterministic decode arena
// makes every token's buffer addresses identical, so the captured exec's baked
// pointers stay valid; per-token input/position/GDN-state are injected into
// those fixed buffers before each relaunch.

bool CommandEncoder::decode_capture_begin() {
  device_.make_current();
  g_decode_capturing.store(true, std::memory_order_relaxed);
  hipError_t e =
      hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
  if (e != hipSuccess) {
    g_decode_capturing.store(false, std::memory_order_relaxed);
    (void)hipGetLastError();
    static const bool dbg = std::getenv("MLX_PURE_DEBUG") != nullptr;
    if (dbg)
      fprintf(stderr, "[cap] BeginCapture failed: %s\n", hipGetErrorString(e));
    return false;
  }
  return true;
}

bool CommandEncoder::decode_capture_end_record(int slot) {
  device_.make_current();
  slot &= 1;
  static const bool dbg = std::getenv("MLX_PURE_DEBUG") != nullptr;
  hipGraph_t g = nullptr;
  hipError_t ee = hipStreamEndCapture(stream_, &g);
  g_decode_capturing.store(false, std::memory_order_relaxed);
  if (ee != hipSuccess || !g) {
    if (g)
      hipGraphDestroy(g);
    (void)hipGetLastError();
    if (dbg)
      fprintf(stderr, "[cap] EndCapture failed: %s\n", hipGetErrorString(ee));
    return false;
  }
  size_t nn = 0;
  hipGraphGetNodes(g, nullptr, &nn);
  if (dbg) {
    fprintf(stderr, "[cap] captured %zu nodes\n", nn);
    std::vector<hipGraphNode_t> nodes(nn);
    if (hipGraphGetNodes(g, nodes.data(), &nn) == hipSuccess) {
      int n_kernel = 0, n_memcpy = 0, n_memset = 0, n_event = 0, n_other = 0;
      for (auto& nd : nodes) {
        hipGraphNodeType ty;
        if (hipGraphNodeGetType(nd, &ty) != hipSuccess) {
          n_other++;
          continue;
        }
        switch (ty) {
          case hipGraphNodeTypeKernel:
            n_kernel++;
            break;
          case hipGraphNodeTypeMemcpy:
            n_memcpy++;
            break;
          case hipGraphNodeTypeMemset:
            n_memset++;
            break;
          case hipGraphNodeTypeEventRecord:
          case hipGraphNodeTypeWaitEvent:
            n_event++;
            break;
          default:
            n_other++;
            break;
        }
      }
      fprintf(
          stderr,
          "[cap] node types: kernel=%d memcpy=%d memset=%d event=%d other=%d\n",
          n_kernel,
          n_memcpy,
          n_memset,
          n_event,
          n_other);
    }
  }
  if (nn == 0) {
    hipGraphDestroy(g);
    return false;
  }
  hipGraphExec_t exec = nullptr;
  hipError_t ie = hipGraphInstantiate(&exec, g, nullptr, nullptr, 0);
  if (ie != hipSuccess) {
    if (dbg)
      fprintf(stderr, "[cap] Instantiate failed: %s\n", hipGetErrorString(ie));
    hipGraphDestroy(g);
    (void)hipGetLastError();
    return false;
  }
  if (decode_cap_exec_[slot])
    hipGraphExecDestroy(decode_cap_exec_[slot]);
  if (decode_cap_graph_[slot])
    hipGraphDestroy(decode_cap_graph_[slot]);
  decode_cap_graph_[slot] = g;
  decode_cap_exec_[slot] = exec;
  // Stream capture records WITHOUT executing — run the exec once to actually
  // compute the record token's logits/state.
  CHECK_HIP_ERROR(hipGraphLaunch(exec, stream_));
  if (hipError_t st = worker_->commit(stream_); st != hipSuccess) {
    set_device_error(st, "queuing the completion callback of a graph launch");
  }
  return true;
}

bool CommandEncoder::decode_capture_replay(int slot) {
  slot &= 1;
  if (!decode_cap_exec_[slot])
    return false;
  device_.make_current();
  CHECK_HIP_ERROR(hipGraphLaunch(decode_cap_exec_[slot], stream_));
  if (hipError_t st = worker_->commit(stream_); st != hipSuccess) {
    set_device_error(st, "queuing the completion callback of a graph replay");
  }
  return true;
}

void CommandEncoder::decode_capture_destroy() {
  for (int s = 0; s < 2; ++s) {
    if (decode_cap_exec_[s]) {
      hipGraphExecDestroy(decode_cap_exec_[s]);
      decode_cap_exec_[s] = nullptr;
    }
    if (decode_cap_graph_[s]) {
      hipGraphDestroy(decode_cap_graph_[s]);
      decode_cap_graph_[s] = nullptr;
    }
  }
}

void CommandEncoder::commit() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  temporary_ptrs_.clear();
  batch_bytes_ = 0;

  if (use_hip_graphs() && node_count_ > 0) {
    if (!from_nodes_.empty()) {
      CHECK_HIP_ERROR(hipGraphAddDependencies(
          build_graph_,
          from_nodes_.data(),
          to_nodes_.data(),
          from_nodes_.size()));
    }
    device_.make_current();

    hipGraphExec_t graph_exec = nullptr;
    bool build_graph_adopted =
        false; // build_graph_ became a slot's source graph
    std::shared_ptr<std::atomic<int>> inflight;
    ExecSlot* used_slot = nullptr;
    // Reuse a drained exec (inflight==0) for an identical kernel sequence. In
    // decode-mode the whole-forward graph recurs every token, so refresh the
    // cached exec's params in one hipGraphExecUpdate; otherwise (or if
    // ExecUpdate fails) reinstantiate into the slot. The slot owns the source
    // graph + arg Packs for the exec's life (CLR stores kernelParams by
    // pointer).
    static const bool prefill_replay_cu = [] {
      const char* e = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
      return e && e[0] == '1';
    }();
    const bool use_execupdate = graph_decode_mode() || prefill_replay_cu;
    auto& pool = exec_pool_[graph_nodes_key_ + ":" + graph_deps_key_];
    // For the stable decode topology, grow the pool to N execs (skip reuse
    // until then) so replay always finds a drained slot despite completion-flag
    // lag.
    static const size_t replay_slots = [] {
      const char* e = std::getenv("MLX_GRAPH_REPLAY_SLOTS");
      return e ? std::max<size_t>(2, std::atoi(e)) : 4;
    }();
    const std::string& grow_key =
        graph_decode_mode() ? decode_key_ : prefill_key_;
    const bool force_grow = use_execupdate && !grow_key.empty() &&
        (graph_nodes_key_ + ":" + graph_deps_key_) == grow_key &&
        pool.size() < replay_slots;
    for (auto& slot : pool) {
      if (force_grow)
        break;
      if (slot.inflight->load(std::memory_order_acquire) != 0) {
        continue;
      }
      bool refreshed = false;
      if (use_execupdate) {
        hipGraphExecUpdateResult ur;
        hipGraphNode_t en;
        if (hipGraphExecUpdate(slot.exec, build_graph_, &en, &ur) ==
                hipSuccess &&
            ur == hipGraphExecUpdateSuccess) {
          refreshed = true;
          build_graph_adopted = true; // exec now bound to build_graph_'s nodes
          slot.src_nodes = build_nodes_;
          if (slot.source_graph)
            hipGraphDestroy(slot.source_graph);
          slot.source_graph = build_graph_;
        } else {
          (void)hipGetLastError();
        }
      }
      if (refreshed) {
        graph_exec = slot.exec;
      } else {
        // Reinstantiate into this slot from the new build graph.
        hipGraphExecDestroy(slot.exec);
        CHECK_HIP_ERROR(
            hipGraphInstantiate(&slot.exec, build_graph_, nullptr, nullptr, 0));
        graph_exec = slot.exec;
        slot.src_nodes = build_nodes_;
        if (slot.source_graph)
          hipGraphDestroy(slot.source_graph);
        slot.source_graph = build_graph_;
        build_graph_adopted = true;
      }
      inflight = slot.inflight;
      used_slot = &slot;
      break;
    }
    if (graph_exec == nullptr) {
      CHECK_HIP_ERROR(
          hipGraphInstantiate(&graph_exec, build_graph_, nullptr, nullptr, 0));
      inflight = std::make_shared<std::atomic<int>>(0);
      pool.push_back({graph_exec, inflight, build_graph_, {}, build_nodes_});
      used_slot = &pool.back();
      build_graph_adopted = true;
    }
    inflight->store(1, std::memory_order_release);

    // Reclaim this chunk's deferred-free buffers once it has drained, bounding
    // graph-mode memory to a sliding window of chunks instead of a whole
    // forward (which OOMs a 32GB card). Free with a generation LAG so a buffer
    // is only released after the next few chunks have also launched — covers
    // cross-chunk / in-place references that a lag-0 free races
    // (use-after-free). Tunable via MLX_GRAPH_FREE_LAG. Opt-in
    // (MLX_GRAPH_FREE_LAG): per-chunk reclaim is currently racy (frees a buffer
    // the next chunk still references → UAF), so OFF by default — frees flush
    // safely at the per-token synchronize. The real fix is 100% buffer reuse
    // (deterministic per-forward addresses), not freeing.
    static const long free_lag = [] {
      const char* e = std::getenv("MLX_GRAPH_FREE_LAG");
      return e ? std::atol(e) : -1;
    }();
    uint64_t my_gen = graph_current_gen();
    graph_advance_gen();
    CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream_));
    if (free_lag >= 0 && static_cast<long>(my_gen) > free_lag) {
      uint64_t fg = my_gen - free_lag;
      add_completed_handler([fg]() { free_graph_generation(fg); });
    }
    // Reclaim this chunk's stream-ordered POOL buffers now, via hipFreeAsync
    // queued right after the launch on stream_ (retires after the graph; no
    // blocking drain). This is the common case on the discrete pool and keeps
    // the peak bounded without stalling the pipeline.
    free_graph_generation_async(my_gen);
    // Backstop ONLY for the non-stream-ordered remainder (unified/slab buffers,
    // rare on the discrete path): if that residual backlog still exceeds a cap,
    // drain + flush. MLX_GRAPH_DEFER_MAX_MB (default 2048; 0 disables).
    static const size_t defer_cap = [] {
      const char* e = std::getenv("MLX_GRAPH_DEFER_MAX_MB");
      return static_cast<size_t>(e ? std::atoll(e) : 2048) << 20;
    }();
    if (defer_cap && graph_deferred_bytes() > defer_cap) {
      (void)hipStreamSynchronize(stream_);
      flush_graph_deferred_frees();
    }
    // The completion handler fires after the stream drains this commit's
    // launch; clear inflight so the slot can be reused next token.
    add_completed_handler(
        [inflight]() { inflight->store(0, std::memory_order_release); });

    // Lock in the stable decode topology key once it recurs on two consecutive
    // tokens (the first decode token after prefill differs). Thereafter,
    // matching tokens replay (build-once) — re-point a drained slot's params,
    // no rebuild.
    if (graph_decode_mode() && used_slot && decode_key_.empty()) {
      std::string fk = graph_nodes_key_ + ":" + graph_deps_key_;
      if (fk == pending_decode_key_) {
        decode_key_ = fk;
      } else {
        pending_decode_key_ = std::move(fk);
      }
    }
    // Same for prefill: lock the full-size-chunk topology once it recurs (chunk
    // 2 matches chunk 1), then later chunks build-once/replay it.
    if (!graph_decode_mode() && used_slot && prefill_key_.empty()) {
      std::string fk = graph_nodes_key_ + ":" + graph_deps_key_;
      if (fk == pending_prefill_key_) {
        prefill_key_ = fk;
      } else {
        pending_prefill_key_ = std::move(fk);
      }
    }

    // Reset build state for the next chunk.
    from_nodes_.clear();
    to_nodes_.clear();
    graph_nodes_key_.clear();
    graph_deps_key_.clear();
    node_map_.clear();
    active_deps_.clear();
    active_outputs_.clear();
    last_node_ = nullptr;
    bytes_in_graph_ = 0;
    build_nodes_.clear();
    build_node_params_.clear();

    // The exec references the current build's Packs by pointer (CLR stores
    // kernelParams by pointer) and is relaunched in later tokens, so the Packs
    // must outlive the slot's next use. Move them into the slot, releasing its
    // prior Packs — the slot's prior launch has drained (only inflight==0
    // reused). source_graph was adopted above when ExecUpdate/instantiate bound
    // the exec to build_graph_; if not adopted, destroy it. Build next fresh.
    used_slot->packs = std::move(graph_node_args_);
    graph_node_args_.clear();
    if (!build_graph_adopted) {
      hipGraphDestroy(build_graph_);
    }
    CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
  }

  node_count_ = 0;

  // Put completion handlers in a batch. On a stream that has failed the
  // callback cannot be queued and the handlers never run; record that so
  // synchronize() and the event waiters fail instead of blocking on them.
  if (hipError_t st = worker_->commit(stream_); st != hipSuccess) {
    set_device_error(st, "queuing the completion callback of a commit");
  }
}

void CommandEncoder::synchronize() {
  using namespace std::chrono_literals;
  // Mirrors the Metal encoder: wait, then throw the stream's error. Every
  // HIP status is checked because a failed stream returns its fault from
  // each of these calls and never runs the completion handlers, so the
  // promise below would otherwise be waited on forever (lablup/mlxcel#1804).
  auto fail = [this](hipError_t st, const char* where) {
    set_device_error(st, where);
    release_inflight();
    error_.check();
    throw std::runtime_error(describe_device_error(st, where));
  };
  if (hipError_t st = hipStreamSynchronize(stream_); st != hipSuccess) {
    fail(st, "synchronizing the stream");
  }
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  add_completed_handler([p = std::move(p)]() { p->set_value(); });
  commit();
  if (error_.valid()) {
    error_.check(); // the callback could not be queued
  }
  // The handler fires right after the stream drains, so poll the future at a
  // fine grain and the stream (in case it faults meanwhile) about once a
  // millisecond.
  for (int polls = 1; f.wait_for(100us) != std::future_status::ready;
       polls++) {
    if (polls % 10 != 0) {
      continue;
    }
    hipError_t st = hipStreamQuery(stream_);
    if (st != hipSuccess && st != hipErrorNotReady) {
      fail(st, "waiting for the stream's completion handlers");
    }
  }
  if (hipError_t st = hipStreamSynchronize(stream_); st != hipSuccess) {
    fail(st, "synchronizing the stream");
  }
  error_.check();
  release_inflight();
  // Stream is fully drained. Non-cached (no-reuse) execs reference these Packs
  // until now; cached-exec Packs live in their ExecSlot (clr#138) and are NOT
  // in these vectors, so clearing here is safe.
  graph_node_args_.clear();
  graph_node_args_prev_.clear();
  if (use_hip_graphs())
    flush_graph_deferred_frees();
}

namespace {

// Host wait for |ev|. Blocking (the device is in blocking-sync mode, so the
// thread sleeps) unless MLX_ROCM_GPU_WATCHDOG_SECS is set, in which case it
// polls and gives up with kWatchdogExpired at the deadline, like the other
// host waits. A faulted stream ends either wait with its fault.
hipError_t wait_inflight_event(hipEvent_t ev) {
  const int secs = gpu_watchdog_seconds();
  if (secs <= 0) {
    return hipEventSynchronize(ev);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(secs);
  for (;;) {
    hipError_t st = hipEventQuery(ev);
    if (st != hipErrorNotReady) {
      return st;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      return kWatchdogExpired;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

} // namespace

// Bound the memory that committed, unfinished batches pin (see
// max_inflight_bytes). |committed_bytes| is what the batch just committed
// allocated. Records an event after it, forgets batches that have finished,
// and blocks on the oldest while the rest still exceed the budget. A failed
// wait is recorded as this stream's error, which the next synchronize or
// event wait throws, and ends the tracking.
void CommandEncoder::throttle_inflight(size_t committed_bytes) {
  if (inflight_budget_ == 0 || committed_bytes == 0 || error_.valid() ||
      g_decode_capturing.load(std::memory_order_relaxed) ||
      stream_capturing()) {
    return;
  }
  hipEvent_t done = nullptr;
  if (!spare_events_.empty()) {
    done = spare_events_.back();
    spare_events_.pop_back();
  } else if (
      hipEventCreateWithFlags(&done, hipEventDisableTiming) != hipSuccess) {
    (void)hipGetLastError();
    return; // untracked: this batch is bounded by its own size only
  }
  if (hipError_t st = hipEventRecord(done, stream_); st != hipSuccess) {
    spare_events_.push_back(done);
    set_device_error(st, "recording an in-flight command batch");
    release_inflight();
    return;
  }
  inflight_.push_back(InflightBatch{done, committed_bytes});
  inflight_bytes_ += committed_bytes;
  while (!inflight_.empty()) {
    hipError_t st = hipEventQuery(inflight_.front().done);
    if (st == hipErrorNotReady) {
      if (inflight_bytes_ <= inflight_budget_) {
        return;
      }
      st = wait_inflight_event(inflight_.front().done);
    }
    if (st != hipSuccess) {
      set_device_error(st, "waiting for an in-flight command batch");
      release_inflight();
      return;
    }
    inflight_bytes_ -= inflight_.front().bytes;
    spare_events_.push_back(inflight_.front().done);
    inflight_.pop_front();
  }
}

void CommandEncoder::release_inflight() {
  for (auto& b : inflight_) {
    spare_events_.push_back(b.done);
  }
  inflight_.clear();
  inflight_bytes_ = 0;
}

// Global flag: true while any stream on this process is recording a HIP graph.
// Lazy library inits (e.g. hipblasLtCreate) abort the process if first called
// during capture, so they consult this to defer to a non-capturing path.
std::atomic<bool> g_stream_capturing{false};
bool stream_capturing() {
  return g_stream_capturing.load(std::memory_order_relaxed);
}
void set_stream_capturing(bool v) {
  g_stream_capturing.store(v, std::memory_order_relaxed);
}

std::atomic<bool> g_graph_active{false};
bool graph_active() {
  return g_graph_active.load(std::memory_order_relaxed);
}
void set_graph_active(bool v) {
  g_graph_active.store(v, std::memory_order_relaxed);
}

// Decode-mode: a single-token forward accrues into ONE graph (no mid-forward
// commit) that is refreshed via hipGraphExecUpdate and launched once per token.
// Set by the generation loop for Lstep==1 steps; prefill leaves it off so its
// large intermediates stay bounded by the per-graph caps. Disable entirely with
// MLX_GRAPH_DECODE=0.
bool graph_decode_mode() {
  static const bool enabled = [] {
    const char* e = std::getenv("MLX_GRAPH_DECODE");
    return !(e && std::string(e) == "0");
  }();
  return enabled && g_graph_decode_mode.load(std::memory_order_relaxed);
}
void set_graph_decode_mode(bool v) {
  g_graph_decode_mode.store(v, std::memory_order_relaxed);
}

std::unordered_map<int, Device>& get_devices() {
  static std::unordered_map<int, Device> devices;
  return devices;
}

void ensure_device_flags(int device_index) {
  if (device_index < 0) {
    return;
  }
  // Lock-free fast path for the common indices: this runs on every unified
  // allocation and every host read of a unified buffer.
  static std::atomic<uint64_t> flagged_low{0};
  const bool low = device_index < 64;
  const uint64_t bit = low ? (uint64_t{1} << device_index) : 0;
  if (low && (flagged_low.load(std::memory_order_acquire) & bit)) {
    return;
  }
  static std::mutex mu;
  static std::vector<int> flagged_high;
  std::lock_guard<std::mutex> lock(mu);
  if (low ? (flagged_low.load(std::memory_order_relaxed) & bit) != 0
          : std::find(flagged_high.begin(), flagged_high.end(), device_index) !=
              flagged_high.end()) {
    return;
  }
  // Per index, not one process-wide bool: if device 0 were flagged first a
  // global gate would leave device 1 unflagged. hipSetDeviceFlags applies to
  // the current device, so switch only when the caller is on another one and
  // switch back after. Never iterate every device: creating a context or queue
  // on the other GPU of a multi-GPU host is what wedges the discrete GPU's
  // queue over a TB5 link, so touch only this device.
  int prev = -1;
  const bool have_prev = hipGetDevice(&prev) == hipSuccess;
  const bool switched = !have_prev || prev != device_index;
  if (switched && hipSetDevice(device_index) != hipSuccess) {
    // Could not bind the device, so the flags cannot be applied to it. Clear
    // the error and do not record the device as flagged, so a later call can
    // try again.
    (void)hipGetLastError();
    return;
  }
  const hipError_t flags_err = hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
  if (flags_err != hipSuccess) {
    // Not fatal (the device then keeps its default wait mode, as it would
    // have before this helper existed); clear the error so it does not
    // surface from an unrelated later call. Recorded as done either way so a
    // failing device is not retried on every allocation. Say so once on
    // stderr: without blocking-sync a regression of #1876 would otherwise be
    // silent.
    (void)hipGetLastError();
    std::fprintf(
        stderr,
        "[mlxcel rocm] hipSetDeviceFlags(hipDeviceScheduleBlockingSync) failed "
        "for device %d: %s\n",
        device_index,
        hipGetErrorString(flags_err));
  }
  if (switched && have_prev) {
    (void)hipSetDevice(prev);
  }
  if (low) {
    flagged_low.fetch_or(bit, std::memory_order_release);
  } else {
    flagged_high.push_back(device_index);
  }
}

void ensure_current_device_flags() {
  int current = -1;
  if (hipGetDevice(&current) == hipSuccess) {
    ensure_device_flags(current);
  }
}

Device& device(mlx::core::Device device) {
  auto& devices = get_devices();
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    // Bind this device (callers rely on it being current afterwards) and make
    // sure it is in blocking-sync mode before the Device and its streams are
    // constructed. The first unified allocation usually got there first, since
    // the allocator runs before the first stream exists; see
    // ensure_device_flags() for why the order matters.
    (void)hipSetDevice(device.index);
    ensure_device_flags(device.index);
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

CommandEncoder& get_command_encoder(Stream s) {
  // Bind the HIP current device to this stream's device. HIP's current device
  // is per-thread; everything that touches a stream goes through here (eval,
  // kernel launches, event record/wait, commit, completion callbacks). Without
  // binding, operations for a non-default GPU (--device 1) execute against
  // device 0 — the stream/event/kernel land on the wrong device and the queue
  // hangs. With HIP_VISIBLE_DEVICES the only device IS index 0 so the bug is
  // hidden.
  auto& d = device(s.device);
  d.make_current();
  return d.get_command_encoder(s);
}

void clear_all_encoders() {
  auto& devices = get_devices();
  for (auto& [idx, dev] : devices) {
    dev.clear_encoders();
  }
}

} // namespace mlx::core::rocm

namespace mlx::core {
long decode_inline_launch_count() {
  return rocm::g_inline_launches_.load(std::memory_order_relaxed);
}
// Full decode-step stream-capture bridge.
bool decode_capture_begin() {
  return rocm::get_command_encoder(default_stream(default_device()))
      .decode_capture_begin();
}
bool decode_capture_end_record(int slot) {
  return rocm::get_command_encoder(default_stream(default_device()))
      .decode_capture_end_record(slot);
}
bool decode_capture_replay(int slot) {
  return rocm::get_command_encoder(default_stream(default_device()))
      .decode_capture_replay(slot);
}
void decode_capture_destroy() {
  rocm::get_command_encoder(default_stream(default_device()))
      .decode_capture_destroy();
}
} // namespace mlx::core
