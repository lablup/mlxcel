// Copyright © 2025 Apple Inc.
// Modified by mlxcel: captured child graphs (cuBLASLt, cuDNN, and every other
// stream capture) are barriers inside an MLX CUDA graph (issue #2462).
// Modified from upstream MLX 7ca40033132dae444d45f7fc3daae57e24c13f6c mlx/backend/cuda/device.cpp

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/utils.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <future>
#include <unordered_set>

namespace mlx::core::cu {

// Defined in kernel_utils.cu
void* get_noop_kernel_ptr();

namespace {

bool use_cuda_graphs() {
  static bool use_graphs = env::get_var("MLX_USE_CUDA_GRAPHS", true);
  return use_graphs;
}

const char* save_cuda_graphs_dot_file() {
  static const char* filename = []() -> const char* {
    const char* env = std::getenv("MLX_SAVE_CUDA_GRAPHS_DOT_FILE");
    if (env && std::strlen(env) == 0) {
      return nullptr;
    }
    return env;
  }();
  return filename;
}

// [mlxcel #2462] Boolean environment flags with the convention
// docs/environment-variables.md uses: trimmed, case-insensitive; `0`,
// `false`, `off` and `no` are false, `1`, `true`, `on` and `yes` are true,
// unset or anything else is `default_value`. MLX's `env::get_var` is `atoi`,
// which would read `on` as 0.
bool mlxcel_env_flag(const char* name, bool default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  std::string value(raw);
  auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return default_value;
  }
  value = value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
  for (auto& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (value == "0" || value == "false" || value == "off" || value == "no") {
    return false;
  }
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    return true;
  }
  return default_value;
}

// [mlxcel #2462] Whether captured child graphs act as barriers. On by
// default; MLXCEL_CUDA_GRAPH_CAPTURE_BARRIER=0 (or false/off/no) restores
// upstream's data-only dependencies for A/B runs.
bool capture_barrier_enabled() {
  static bool enabled = mlxcel_env_flag("MLXCEL_CUDA_GRAPH_CAPTURE_BARRIER", true);
  return enabled;
}

// [mlxcel #2462] Per-encoder barrier bookkeeping, kept out of device.h so the
// header stays upstream's: the nodes added to the current graph that have no
// successor yet (open branches), the last barrier node, and whether the
// concurrent batch being collected holds a capture. The map is thread-local
// because an encoder is driven by one thread: the main thread's encoders
// live in `get_command_encoders`' static map and every other thread has its
// own, and mlxcel never calls `new_thread_unsafe_stream` (the only way an
// encoder from `get_global_command_encoders` is reached from a second
// thread). If that changes, this map needs a mutex, or the open and barrier
// nodes of one thread would point into a graph another thread reset. The map
// is leaked on purpose: `~CommandEncoder` runs `commit` and `clear_graph_state`
// during thread exit, after a thread_local with a destructor may be gone.
struct CaptureBarrierState {
  std::vector<std::pair<cudaGraphNode_t, std::string>> open;
  std::pair<cudaGraphNode_t, std::string> barrier{nullptr, ""};
  bool concurrent_has_capture{false};
};

std::unordered_map<const void*, CaptureBarrierState>& capture_barrier_states() {
  static thread_local auto* m =
      new std::unordered_map<const void*, CaptureBarrierState>();
  return *m;
}

// [mlxcel #2462] Set by `add_graph_node` around its `insert_graph_dependencies`
// call, so the barrier logic learns that a node is a captured child graph
// from the caller instead of sniffing the node type string.
thread_local bool pending_capture_node = false;

inline bool is_empty_dim(dim3 dim) {
  return (dim.x == 0 && dim.y == 0 && dim.z == 0) ||
      (dim.x == 1 && dim.y == 1 && dim.z == 1);
}

} // namespace

Device::Device(int device) : device_(device) {
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &compute_capability_major_, cudaDevAttrComputeCapabilityMajor, device_));
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &compute_capability_minor_, cudaDevAttrComputeCapabilityMinor, device_));
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &concurrent_managed_access_,
      cudaDevAttrConcurrentManagedAccess,
      device_));
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &host_native_atomic_, cudaDevAttrHostNativeAtomicSupported, device_));
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &managed_memory_, cudaDevAttrManagedMemory, device_));
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &memory_pools_, cudaDevAttrMemoryPoolsSupported, device_));
}

Device::~Device() = default;

void Device::make_current() {
  // We need to set/get current CUDA device very frequently, cache it to reduce
  // actual calls of CUDA APIs. Use -1 as sentinel so the first call on each
  // new thread always calls cudaSetDevice (which establishes the CUDA primary
  // context). Without this, device 0 would never get set on a new thread.
  static thread_local int current = -1;
  if (current != device_) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_));
    current = device_;
  }
}

CommandEncoder::CaptureContext::CaptureContext(CommandEncoder& enc) : enc(enc) {
  enc.device().make_current();
  if (!use_cuda_graphs()) {
    return;
  }
  CHECK_CUDA_ERROR(
      cudaStreamBeginCapture(enc.stream(), cudaStreamCaptureModeThreadLocal));
}

CommandEncoder::CaptureContext::~CaptureContext() {
  if (!use_cuda_graphs()) {
    enc.node_count_++;
    return;
  }

  graph.end_capture(enc.stream());
  if (discard) {
    return;
  }
  enc.add_graph_node(graph);
}

CommandEncoder::ConcurrentContext::ConcurrentContext(CommandEncoder& enc)
    : enc(enc) {
  enc.in_concurrent_ = true;
}

CommandEncoder::ConcurrentContext::~ConcurrentContext() {
  enc.in_concurrent_ = false;
  if (!use_cuda_graphs()) {
    return;
  }

  // Use an empty graph node for synchronization
  CommandEncoder::GraphNode empty{NULL, "E", std::to_string(enc.node_count_++)};
  CHECK_CUDA_ERROR(cudaGraphAddEmptyNode(&empty.node, enc.graph_, NULL, 0));

  // Insert the concurrent -> empty node dependencies
  for (auto& from : enc.concurrent_nodes_) {
    enc.from_nodes_.push_back(from.node);
    enc.to_nodes_.push_back(empty.node);
    enc.graph_deps_key_ += from.id;
    enc.graph_deps_key_ += "-";
    enc.graph_deps_key_ += empty.id;
    enc.graph_deps_key_ += "-";
  }

  // Insert the input -> concurrent node dependencies without updating output
  // nodes
  auto outputs = std::move(enc.active_outputs_);
  bool batch_has_capture = capture_barrier_enabled() &&
      capture_barrier_states()[&enc].concurrent_has_capture;
  enc.insert_graph_dependencies(std::move(enc.concurrent_nodes_));

  // Update output node to be the empty node
  for (auto o : outputs) {
    enc.node_map_.emplace(o, empty).first->second = empty;
  }

  // [mlxcel #2462] Every batch node now has the join node as a successor, so
  // the join node is the open branch; when the batch held a captured child
  // graph it is also the barrier every later node depends on.
  if (capture_barrier_enabled()) {
    auto& state = capture_barrier_states()[&enc];
    state.open.clear();
    state.open.push_back({empty.node, empty.id});
    if (batch_has_capture) {
      state.barrier = {empty.node, empty.id};
    }
  }
}

void CommandEncoder::insert_graph_dependencies(GraphNode node) {
  node.id = std::to_string(node_count_++);
  if (in_concurrent_) {
    // [mlxcel #2462] The batch inherits the capture flag of any node in it.
    if (pending_capture_node) {
      capture_barrier_states()[this].concurrent_has_capture = true;
      pending_capture_node = false;
    }
    concurrent_nodes_.push_back(std::move(node));
  } else {
    std::vector<GraphNode> nodes;
    nodes.push_back(std::move(node));
    insert_graph_dependencies(std::move(nodes));
  }
}

void CommandEncoder::insert_graph_dependencies(std::vector<GraphNode> nodes) {
  for (auto& node : nodes) {
    graph_nodes_key_ += node.node_type;
    graph_nodes_key_ += "-";
  }
  std::vector<GraphNode> deps;
  {
    // Dependencies must be added in the same order to produce a consistent
    // topology
    std::unordered_set<cudaGraphNode_t> set_deps;
    for (auto d : active_deps_) {
      if (auto it = node_map_.find(d); it != node_map_.end()) {
        auto [_, inserted] = set_deps.insert(it->second.node);
        if (inserted) {
          deps.push_back(it->second);
        }
      }
    }
    // [mlxcel #2462] A captured child graph (a cuBLASLt matmul, a cuDNN
    // convolution or attention, any other stream capture) joins every open
    // branch of the graph before it and gates every node added after it,
    // on top of the data dependencies above. Upstream orders a child graph
    // only by the buffers it declares, which lets its kernels run alongside
    // unrelated kernel nodes of the same graph. On GB10 (CUDA 13, driver
    // 580.178.04, MLX 243bdf09c) that overlap corrupted cuBLASLt f32 GEMMs
    // non-deterministically: about 40% of fresh processes returned wrong
    // hidden states from the first run of a 12-layer DeBERTa-v2 encoder, a
    // random layer each time, with every replay correct. Chaining every node
    // in program order removed it (0 of 20), chaining only edges that touch a
    // child graph removed it (0 of 20), chaining only kernel-kernel edges did
    // not (18 of 20), and this barrier removed it (0 of 30) while keeping
    // kernel nodes between two child graphs concurrent. The shared resource
    // inside cuBLASLt is not identified; the captured child graphs hold
    // split-K GEMMs with a workspace semaphore memset. Each barrier also
    // changes the dependency key, so the graph cache sees the barrier
    // topology.
    if (capture_barrier_enabled() && !nodes.empty()) {
      auto& state = capture_barrier_states()[this];
      // A single node is a capture when `add_graph_node` says so; a
      // concurrent batch (`ConcurrentContext`) is one when any node it
      // collected was, and its join node becomes the barrier afterwards
      // (`~ConcurrentContext`).
      bool is_capture = pending_capture_node || state.concurrent_has_capture;
      pending_capture_node = false;
      state.concurrent_has_capture = false;
      if (is_capture) {
        for (auto& [node, id] : state.open) {
          if (set_deps.insert(node).second) {
            deps.push_back(GraphNode{node, "K", id});
          }
        }
        state.open.clear();
        state.barrier = {nodes.back().node, nodes.back().id};
      } else if (
          state.barrier.first != nullptr &&
          set_deps.insert(state.barrier.first).second) {
        deps.push_back(
            GraphNode{state.barrier.first, "K", state.barrier.second});
      }
      for (auto& from : deps) {
        state.open.erase(
            std::remove_if(
                state.open.begin(),
                state.open.end(),
                [&](auto& open) { return open.first == from.node; }),
            state.open.end());
      }
      for (auto& node : nodes) {
        state.open.push_back({node.node, node.id});
      }
    }
  }
  active_deps_.clear();
  if (force_single_root_ && deps.empty()) {
    for (auto& node : nodes) {
      roots_.push_back(node.node);
    }
  }

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
}

// Can be tuned with MLX_MAX_OPS_PER_BUFFER, MLX_MAX_MB_PER_BUFFER
std::pair<int, int> get_graph_limits(Device& d) {
  auto cc =
      d.compute_capability_major() * 100 + d.compute_capability_minor() * 10;
  int ops = 20;
  int mb = 100;
  switch (cc) {
    case 800: // A100
      ops = 20;
      mb = 400;
      break;
    case 900: // H100
    case 1000: // B200
    case 1200: // Consumer Blackwell
      ops = 100;
      mb = 1000;
      break;
    case 1210: // DGX Spark
      ops = 20;
      mb = 25;
      break;
  }
  return {env::max_ops_per_buffer(ops), env::max_mb_per_buffer(mb)};
}

CommandEncoder::CommandEncoder(Device& d)
    : device_(d),
      stream_(d),
      graph_(d),
      worker_(std::make_shared<Worker>(d)),
      graph_cache_("MLX_CUDA_GRAPH_CACHE_SIZE", /* default_capacity */ 400),
      // DGX Spark / RTX Spark run memory-bound kernels slower in graphs that
      // have several root nodes.
      force_single_root_(
          d.compute_capability_major() == 12 &&
          d.compute_capability_minor() == 1) {
  std::tie(max_ops_per_graph_, max_mb_per_graph_) = get_graph_limits(d);
  worker_->start();
}

CommandEncoder::~CommandEncoder() {
  try {
    synchronize();
  } catch (...) {
    // Synchronizing can fail when the CUDA runtime is shutting down.
  }
  worker_->stop();
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_->add_task(std::move(task));
}

void CommandEncoder::set_input_array(const array& arr) {
  if (!use_cuda_graphs()) {
    return;
  }
  bytes_in_graph_ += arr.data_size();
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
}

void CommandEncoder::set_output_array(const array& arr) {
  if (!use_cuda_graphs()) {
    return;
  }

  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
  active_outputs_.push_back(id);
}

void CommandEncoder::add_kernel_node_raw(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    dim3 cluster_dim,
    uint32_t smem_bytes,
    void** params) {
  bool use_cluster = !is_empty_dim(cluster_dim);
  assert(!use_cluster || device_.compute_capability_major() >= 9);

  if (!use_cuda_graphs()) {
    node_count_++;
    cudaLaunchConfig_t config = {};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream();
    cudaLaunchAttribute attr = {};
    if (use_cluster) {
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = cluster_dim.x;
      attr.val.clusterDim.y = cluster_dim.y;
      attr.val.clusterDim.z = cluster_dim.z;
      config.attrs = &attr;
      config.numAttrs = 1;
    }
    CHECK_CUDA_ERROR(cudaLaunchKernelExC(&config, func, params));
    return;
  }

  cudaKernelNodeParams kernel_params = {0};
  kernel_params.func = func;
  kernel_params.gridDim = grid_dim;
  kernel_params.blockDim = block_dim;
  kernel_params.kernelParams = params;
  kernel_params.sharedMemBytes = smem_bytes;
  cudaGraphNode_t node = add_kernel_node_raw(kernel_params);
  if (use_cluster) {
    cudaKernelNodeAttrValue attr = {};
    attr.clusterDim.x = cluster_dim.x;
    attr.clusterDim.y = cluster_dim.y;
    attr.clusterDim.z = cluster_dim.z;
    CHECK_CUDA_ERROR(cudaGraphKernelNodeSetAttribute(
        node, cudaLaunchAttributeClusterDimension, &attr));
  }
}

void CommandEncoder::add_kernel_node_raw(
    CUfunction func,
    dim3 grid_dim,
    dim3 block_dim,
    dim3 cluster_dim,
    uint32_t smem_bytes,
    void** params) {
  bool use_cluster = !is_empty_dim(cluster_dim);
  assert(!use_cluster || device_.compute_capability_major() >= 9);

  if (!use_cuda_graphs()) {
    node_count_++;
    CUlaunchConfig config = {};
    config.gridDimX = grid_dim.x;
    config.gridDimY = grid_dim.y;
    config.gridDimZ = grid_dim.z;
    config.blockDimX = block_dim.x;
    config.blockDimY = block_dim.y;
    config.blockDimZ = block_dim.z;
    config.sharedMemBytes = smem_bytes;
    config.hStream = stream();
    CUlaunchAttribute attr = {};
    if (use_cluster) {
      attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attr.value.clusterDim.x = cluster_dim.x;
      attr.value.clusterDim.y = cluster_dim.y;
      attr.value.clusterDim.z = cluster_dim.z;
      config.attrs = &attr;
      config.numAttrs = 1;
    }
    CHECK_CUDA_ERROR(cuLaunchKernelEx(&config, func, params, nullptr));
    return;
  }

  CUDA_KERNEL_NODE_PARAMS kernel_params = {};
  kernel_params.func = func;
  kernel_params.gridDimX = grid_dim.x;
  kernel_params.gridDimY = grid_dim.y;
  kernel_params.gridDimZ = grid_dim.z;
  kernel_params.blockDimX = block_dim.x;
  kernel_params.blockDimY = block_dim.y;
  kernel_params.blockDimZ = block_dim.z;
  kernel_params.kernelParams = params;
  kernel_params.sharedMemBytes = smem_bytes;
  CUgraphNode node = add_kernel_node_raw(kernel_params);
  if (use_cluster) {
    CUlaunchAttributeValue attr = {};
    attr.clusterDim.x = cluster_dim.x;
    attr.clusterDim.y = cluster_dim.y;
    attr.clusterDim.z = cluster_dim.z;
    CHECK_CUDA_ERROR(cuGraphKernelNodeSetAttribute(
        node, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, &attr));
  }
}

cudaGraphNode_t CommandEncoder::add_kernel_node_raw(
    const cudaKernelNodeParams& params) {
  cudaGraphNode_t node;
  CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&node, graph_, NULL, 0, &params));
  insert_graph_dependencies(GraphNode{node, "K"});
  return node;
}

CUgraphNode CommandEncoder::add_kernel_node_raw(
    const CUDA_KERNEL_NODE_PARAMS& params) {
  CUgraphNode node;
  CHECK_CUDA_ERROR(cuGraphAddKernelNode(&node, graph_, NULL, 0, &params));
  insert_graph_dependencies(GraphNode{node, "K"});
  return node;
}

std::pair<std::string, bool> subgraph_to_key(cudaGraph_t graph) {
  // Constructs a key representing the nodes of a sub-graph.
  // Also checks if the sub-graph is updatable as CUDA graphs do not get
  // updated correctly if a kernel node getting updated has a different cluster
  // shape than the node it's being updated with.
  std::string key = "(";
  size_t num_nodes = 0;
  CHECK_CUDA_ERROR(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  if (num_nodes == 0) {
    return {key + ")", true};
  }
  bool is_updatable = true;
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  CHECK_CUDA_ERROR(cudaGraphGetNodes(graph, nodes.data(), &num_nodes));
  for (const auto& node : nodes) {
    if (!is_updatable) {
      break;
    }
    cudaGraphNodeType type;
    CHECK_CUDA_ERROR(cudaGraphNodeGetType(node, &type));
    switch (type) {
      case cudaGraphNodeTypeGraph: {
        // Try to be updatable for a structure like graph -> graph -> kernel
        cudaGraph_t child;
        CHECK_CUDA_ERROR(cudaGraphChildGraphNodeGetGraph(node, &child));
        auto [subkey, sub_is_updatable] = subgraph_to_key(child);
        is_updatable &= sub_is_updatable;
        key += subkey;
        break;
      }
      case cudaGraphNodeTypeHost:
        key += "H";
        break;
      case cudaGraphNodeTypeMemset:
        key += "M";
        break;
      case cudaGraphNodeTypeKernel: {
        cudaLaunchAttributeValue cluster_dim;
        CHECK_CUDA_ERROR(cudaGraphKernelNodeGetAttribute(
            node, cudaLaunchAttributeClusterDimension, &cluster_dim));
        // Only allow dim.x to be greater than 1
        if (cluster_dim.clusterDim.y > 1 || cluster_dim.clusterDim.z > 1) {
          is_updatable = false;
        } else {
          key += "K";
          key += std::to_string(cluster_dim.clusterDim.x);
        }
        break;
      }
      case cudaGraphNodeTypeWaitEvent:
        key += "W";
        break;
      case cudaGraphNodeTypeEventRecord:
        key += "R";
        break;
      default:
        is_updatable = false;
    }
  }
  key += ")";
  return {key, is_updatable};
}

void CommandEncoder::add_graph_node(cudaGraph_t child) {
  if (!use_cuda_graphs()) {
    node_count_++;
    CudaGraphExec graph_exec;
    graph_exec.instantiate(child);
    device_.make_current();
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, stream()));
    return;
  }
  cudaGraphNode_t node;
  auto [sub_graph_key, is_updatable] = subgraph_to_key(child);
  is_graph_updatable_ &= is_updatable;
  CHECK_CUDA_ERROR(cudaGraphAddChildGraphNode(&node, graph_, NULL, 0, child));
  pending_capture_node = true;
  insert_graph_dependencies(GraphNode{node, sub_graph_key});
}

void CommandEncoder::add_graph_node(
    cudaGraph_t child,
    const std::string& subgraph_key,
    bool is_updatable) {
  if (!use_cuda_graphs()) {
    node_count_++;
    CudaGraphExec graph_exec;
    graph_exec.instantiate(child);
    device_.make_current();
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, stream()));
    return;
  }
  is_graph_updatable_ &= is_updatable;
  cudaGraphNode_t node;
  CHECK_CUDA_ERROR(cudaGraphAddChildGraphNode(&node, graph_, NULL, 0, child));
  pending_capture_node = true;
  insert_graph_dependencies(GraphNode{node, subgraph_key});
}

bool CommandEncoder::needs_commit() {
  return (node_count_ > max_ops_per_graph_) ||
      ((bytes_in_graph_ >> 20) > max_mb_per_graph_);
}

void CommandEncoder::commit() {
  nvtx3::scoped_range r("CommandEncoder::commit");
  try {
    commit_impl();
  } catch (...) {
    // Clear pending CUDA error first.
    cudaGetLastError();
    // Clear states.
    clear_graph_state();
    node_count_ = 0;
    bytes_in_graph_ = 0;
    // Clear graph.
    try {
      graph_.reset();
    } catch (...) {
      // Destroying could fail.
      graph_.release();
    }
    try {
      graph_ = CudaGraph(device_);
    } catch (...) {
      // Keep the original error.
    }
    // Re-throw the error.
    throw;
  }
}

void CommandEncoder::synchronize() {
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  add_completed_handler([p = std::move(p)]() { p->set_value(); });
  commit();
  f.wait();
}

void CommandEncoder::commit_impl() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  if (use_cuda_graphs() && node_count_ > 0) {
    // Give a graph with several roots a single no-op root.
    if (roots_.size() > 1) {
      cudaKernelNodeParams params = {};
      params.func = get_noop_kernel_ptr();
      params.gridDim = dim3(1);
      params.blockDim = dim3(1);
      cudaGraphNode_t root;
      CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&root, graph_, NULL, 0, &params));
      for (auto node : roots_) {
        from_nodes_.push_back(root);
        to_nodes_.push_back(node);
      }
      graph_nodes_key_ += "R-";
    }
    if (!from_nodes_.empty()) {
#if CUDART_VERSION >= 13000
      CHECK_CUDA_ERROR(cudaGraphAddDependencies(
          graph_,
          from_nodes_.data(),
          to_nodes_.data(),
          nullptr, // edgeData
          from_nodes_.size()));
#else
      CHECK_CUDA_ERROR(cudaGraphAddDependencies(
          graph_, from_nodes_.data(), to_nodes_.data(), from_nodes_.size()));
#endif
    }

    device_.make_current();

    if (!is_graph_updatable_) {
      CudaGraphExec graph_exec;
      graph_exec.instantiate(graph_);
      CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, stream_));
    } else {
      auto graph_key = graph_nodes_key_ + ":" + graph_deps_key_;
      auto& graph_exec = graph_cache_[graph_key];

      if (graph_exec != nullptr) {
        cudaGraphExecUpdateResult update_result;
#if CUDART_VERSION >= 12000
        cudaGraphExecUpdateResultInfo info;
        cudaGraphExecUpdate(graph_exec, graph_, &info);
        update_result = info.result;
#else
        cudaGraphNode_t error_node;
        cudaGraphExecUpdate(graph_exec, graph_, &error_node, &update_result);
#endif // CUDART_VERSION >= 12000
        if (update_result != cudaGraphExecUpdateSuccess) {
          cudaGetLastError(); // reset error
          graph_exec.reset();
        }
      }
      if (graph_exec == nullptr) {
        graph_exec.instantiate(graph_);
      }

      CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, stream_));
    }

    // Save cuda graph to dot file
    if (const char* filename = save_cuda_graphs_dot_file(); filename) {
      static int count = 0;
      auto path = fmt::format("{}_{}.dot", filename, ++count);
      // [mlxcel] MLXCEL_CUDA_GRAPHS_DOT_VERBOSE=1 adds node parameters to
      // the dump, which is what shows which buffers concurrent nodes touch.
      static const unsigned dot_flags =
          mlxcel_env_flag("MLXCEL_CUDA_GRAPHS_DOT_VERBOSE", false)
          ? cudaGraphDebugDotFlagsVerbose
          : 0;
      CHECK_CUDA_ERROR(
          cudaGraphDebugDotPrint(graph_, path.c_str(), dot_flags));
    }

    // Reset state
    clear_graph_state();
    graph_ = CudaGraph(device_);
  }

  // Put completion handlers in a batch.
  worker_->commit(stream_);
  node_count_ = 0;
  bytes_in_graph_ = 0;
}

void CommandEncoder::clear_graph_state() {
  capture_barrier_states().erase(this);
  from_nodes_.clear();
  to_nodes_.clear();
  graph_deps_key_.clear();
  graph_nodes_key_.clear();
  node_map_.clear();
  active_deps_.clear();
  active_outputs_.clear();
  concurrent_nodes_.clear();
  roots_.clear();
  is_graph_updatable_ = true;
}

Device& device(int cuda_device) {
  // The devices are leak intentionally as user code may still be accessing
  // device after main thread teardown.
  static auto* devices = []() {
    auto* devices = new std::vector<Device>;
    int device_count = gpu::device_count();
    for (int i = 0; i < device_count; ++i) {
      devices->emplace_back(i);
    }
    return devices;
  }();
  return devices->at(cuda_device);
}

Device& device(mlx::core::Device d) {
  return device(d.index);
}

CommandEncoder& get_command_encoder(Stream s) {
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    auto& global_encoders = get_global_command_encoders();
    it = global_encoders.find(s.index);
    if (it == global_encoders.end()) {
      throw std::runtime_error(
          fmt::format(
              "There is no Stream(gpu, {}) in current thread.", s.index));
    }
  }
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  if (is_main_thread()) {
    static auto* encoders = new std::unordered_map<int, CommandEncoder>();
    return *encoders;
  } else {
    static thread_local std::unordered_map<int, CommandEncoder> encoders;
    return encoders;
  }
}

std::unordered_map<int, CommandEncoder>& get_global_command_encoders() {
  // encoders are leaked intentionally as they would synchronize on process
  // shutdown
  static auto* encoders = new std::unordered_map<int, CommandEncoder>();
  return *encoders;
}

} // namespace mlx::core::cu
