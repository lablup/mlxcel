// Copyright © 2025 Apple Inc.

// mlxcelverse: ROCm FFT through hipFFT (lablup/mlxcel#1825).
//
// Ported from mlx/backend/cuda/fft.cu. Two differences are worth knowing before
// reading it against the CUDA original.
//
// This hipFFT has no `hipfftXtMakePlanMany` or `hipfftXtExec`, the generic
// entry points the CUDA file uses to pass input, output and execution types
// separately. It has the typed API instead: `hipfftMakePlanMany64` takes one
// `hipfftType` (C2C, R2C or C2R) and each transform has its own exec function.
// MLX only ever asks for float32 and complex64 here, so the typed API covers
// the same ground and the type-selection helpers the CUDA file needs disappear.
//
// hipFFT work is not a HIP graph kernel node, so it goes through
// `CommandEncoder::launch_kernel`, which splits the accumulated graph, runs the
// call on the same stream ordered after it, and starts the next graph fresh.
// That is the same path the rocBLAS and hipBLASLt GEMMs take in matmul.cpp.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/lru_cache.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core {

namespace rocm {

// Scales an FFT result in place. hipFFT, like cuFFT, leaves the 1/n of an
// inverse transform to the caller.
template <typename T>
__global__ void scale_fft_output(T* out, T scale, size_t size) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = index; i < size; i += stride) {
    out[i] = out[i] * scale;
  }
}

} // namespace rocm

namespace {

void check_hipfft_error(const char* name, hipfftResult err) {
  if (err != HIPFFT_SUCCESS) {
    throw std::runtime_error(
        std::string(name) +
        " failed with code: " + std::to_string(static_cast<int>(err)) + ".");
  }
}

#define CHECK_HIPFFT_ERROR(cmd) check_hipfft_error(#cmd, (cmd))

enum class FFTTransformType : uint8_t {
  C2C = 0,
  R2C = 1,
  C2R = 2,
};

hipfftType hipfft_type(FFTTransformType transform_type) {
  switch (transform_type) {
    case FFTTransformType::C2C:
      return HIPFFT_C2C;
    case FFTTransformType::R2C:
      return HIPFFT_R2C;
    case FFTTransformType::C2R:
      return HIPFFT_C2R;
  }
  throw std::runtime_error("[FFT] Unsupported hipFFT transform type.");
}

// The backend's LRU cache is an unordered_map keyed by the type itself, so it
// needs std::hash. A string key avoids specializing std::hash for a type in an
// anonymous namespace; plan lookups happen once per shape, so its cost does not
// matter.
std::string make_plan_key(
    int device_id,
    FFTTransformType transform_type,
    int64_t n,
    int64_t batch) {
  return std::to_string(device_id) + ":" +
      std::to_string(static_cast<int>(transform_type)) + ":" +
      std::to_string(n) + ":" + std::to_string(batch);
}

struct HipFFTPlan {
  HipFFTPlan(int device_id, hipfftHandle handle, size_t workspace_size)
      : device_id(device_id), handle(handle), workspace_size(workspace_size) {}

  ~HipFFTPlan() {
    if (handle != 0) {
      try {
        hipfftDestroy(handle);
      } catch (...) {
      }
    }
  }

  HipFFTPlan(const HipFFTPlan&) = delete;
  HipFFTPlan& operator=(const HipFFTPlan&) = delete;

  int device_id;
  hipfftHandle handle;
  size_t workspace_size;
};

struct OrderedArray {
  array arr;
  std::vector<int> order;
};

// Deliberately much smaller than the CUDA file's 128. Measured on gfx1151 with
// ROCm 10.0.0: once enough hipFFT plans are alive at the same time, creating the
// next one blocks forever inside the library, with the GPU idle and the calling
// thread asleep. A probe cycling 26 distinct transform shapes reproduces it at
// capacity 16, 24, 32 and 128, and passes every time at 8. A plan also outlives
// its cache entry while an execution that captured it is still pending, so the
// live count can exceed the capacity and the cap has to leave headroom. The
// root cause inside rocFFT is not identified; this bounds it. Raise it with
// MLX_ROCM_FFT_CACHE_SIZE when measuring.
auto& fft_plan_cache() {
  static rocm::LRUBytesKeyCache<std::string, std::shared_ptr<HipFFTPlan>> cache(
      "MLX_ROCM_FFT_CACHE_SIZE",
      /* default_capacity */ 8);
  return cache;
}

std::mutex& fft_plan_cache_mutex() {
  static std::mutex m;
  return m;
}

int64_t input_embed(FFTTransformType transform_type, int64_t n) {
  return transform_type == FFTTransformType::C2R ? (n / 2 + 1) : n;
}

int64_t output_embed(FFTTransformType transform_type, int64_t n) {
  return transform_type == FFTTransformType::R2C ? (n / 2 + 1) : n;
}

std::shared_ptr<HipFFTPlan> get_fft_plan(
    rocm::CommandEncoder& encoder,
    FFTTransformType transform_type,
    int64_t n,
    int64_t batch) {
  auto key =
      make_plan_key(encoder.device().hip_device(), transform_type, n, batch);

  auto& cache = fft_plan_cache();
  // The cache is shared, and building a plan is not idempotent in cost, so the
  // check-build-insert runs under one lock.
  std::lock_guard<std::mutex> cache_lock(fft_plan_cache_mutex());
  if (auto entry = cache.get(key); entry.has_value()) {
    return *entry;
  }

  encoder.device().make_current();

  hipfftHandle handle = 0;
  size_t workspace_size = 0;
  try {
    CHECK_HIPFFT_ERROR(hipfftCreate(&handle));
    // The work area is allocated per execution from MLX's pool rather than by
    // hipFFT, so a cached plan holds no device memory of its own.
    CHECK_HIPFFT_ERROR(hipfftSetAutoAllocation(handle, 0));

    long long plan_n[1] = {n};
    long long inembed[1] = {input_embed(transform_type, n)};
    long long onembed[1] = {output_embed(transform_type, n)};
    CHECK_HIPFFT_ERROR(hipfftMakePlanMany64(
        handle,
        /* rank = */ 1,
        plan_n,
        inembed,
        /* istride = */ 1,
        /* idist = */ input_embed(transform_type, n),
        onembed,
        /* ostride = */ 1,
        /* odist = */ output_embed(transform_type, n),
        hipfft_type(transform_type),
        batch,
        &workspace_size));
  } catch (...) {
    if (handle != 0) {
      hipfftDestroy(handle);
    }
    throw;
  }

  auto plan = std::make_shared<HipFFTPlan>(
      encoder.device().hip_device(), handle, workspace_size);
  cache.put(key, plan);
  return plan;
}

std::vector<int> make_identity_order(int ndim) {
  std::vector<int> order(ndim);
  std::iota(order.begin(), order.end(), 0);
  return order;
}

std::vector<int> move_axis_to_back_permutation(int ndim, int axis_pos) {
  std::vector<int> perm;
  perm.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    if (i != axis_pos) {
      perm.push_back(i);
    }
  }
  perm.push_back(axis_pos);
  return perm;
}

std::vector<int> apply_permutation(
    const std::vector<int>& values,
    const std::vector<int>& perm) {
  std::vector<int> out(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    out[i] = values[perm[i]];
  }
  return out;
}

int find_axis_position(const std::vector<int>& order, int axis) {
  auto it = std::find(order.begin(), order.end(), axis);
  if (it == order.end()) {
    throw std::runtime_error("[FFT] Internal axis tracking mismatch.");
  }
  return static_cast<int>(it - order.begin());
}

OrderedArray prepare_input(
    const OrderedArray& current,
    int axis,
    bool allow_direct,
    rocm::CommandEncoder& encoder,
    Stream s) {
  int axis_pos = find_axis_position(current.order, axis);
  bool axis_last = axis_pos == static_cast<int>(current.order.size()) - 1;
  bool direct = allow_direct && axis_last && current.arr.flags().row_contiguous;

  if (direct) {
    return current;
  }

  array view = current.arr;
  std::vector<int> order = current.order;
  if (!axis_last) {
    auto perm = move_axis_to_back_permutation(current.arr.ndim(), axis_pos);
    view = transpose_in_eval(current.arr, perm);
    order = apply_permutation(current.order, perm);
  }

  array packed = contiguous_copy_gpu(view, s);
  encoder.add_temporary(packed);
  return {std::move(packed), std::move(order)};
}

void execute_fft(
    const array& in,
    array& out,
    FFTTransformType transform_type,
    bool inverse,
    rocm::CommandEncoder& encoder) {
  if (!in.flags().row_contiguous || in.strides(-1) != 1) {
    throw std::runtime_error("[FFT] Expected packed row-contiguous FFT input.");
  }

  int64_t n =
      transform_type == FFTTransformType::C2R ? out.shape(-1) : in.shape(-1);
  int64_t batch = in.shape().empty() ? 1 : in.size() / in.shape(-1);
  auto plan = get_fft_plan(encoder, transform_type, n, batch);

  encoder.set_input_array(in);
  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
  encoder.set_output_array(out);

  // The work area comes from the same async pool as every other buffer on this
  // encoder. A synchronous allocator::malloc here deadlocks: it waits on the
  // pool while the encoder still holds outstanding async frees.
  size_t ws_bytes = std::max<size_t>(plan->workspace_size, 1);
  array workspace({static_cast<int>(ws_bytes)}, uint8, nullptr, {});
  workspace.set_data(mlx::core::rocm::malloc_async(ws_bytes, encoder));
  encoder.add_temporary(workspace);

  void* in_ptr = const_cast<void*>(gpu_ptr<const void>(in));
  void* out_ptr = gpu_ptr<void>(out);
  void* ws_ptr = gpu_ptr<void>(workspace);
  int direction = inverse ? HIPFFT_BACKWARD : HIPFFT_FORWARD;
  hipfftHandle handle = plan->handle;

  // hipFFT is not a graph kernel node; launch_kernel splits the graph and runs
  // this on the encoder stream, ordered after what came before. The captured
  // plan keeps the cache entry alive for the duration of the call.
  encoder.launch_kernel([handle,
                         plan,
                         in_ptr,
                         out_ptr,
                         ws_ptr,
                         direction,
                         transform_type](hipStream_t stream) {
    CHECK_HIPFFT_ERROR(hipfftSetStream(handle, stream));
    CHECK_HIPFFT_ERROR(hipfftSetWorkArea(handle, ws_ptr));
    switch (transform_type) {
      case FFTTransformType::C2C:
        CHECK_HIPFFT_ERROR(hipfftExecC2C(
            handle,
            static_cast<hipfftComplex*>(in_ptr),
            static_cast<hipfftComplex*>(out_ptr),
            direction));
        break;
      case FFTTransformType::R2C:
        CHECK_HIPFFT_ERROR(hipfftExecR2C(
            handle,
            static_cast<hipfftReal*>(in_ptr),
            static_cast<hipfftComplex*>(out_ptr)));
        break;
      case FFTTransformType::C2R:
        CHECK_HIPFFT_ERROR(hipfftExecC2R(
            handle,
            static_cast<hipfftComplex*>(in_ptr),
            static_cast<hipfftReal*>(out_ptr)));
        break;
    }
  });
}

void restore_output_layout(const OrderedArray& current, array& out) {
  Strides out_strides(out.ndim());
  for (size_t i = 0; i < current.order.size(); ++i) {
    out_strides[current.order[i]] = current.arr.strides(i);
  }

  auto [data_size, row_contiguous, col_contiguous] =
      check_contiguity(out.shape(), out_strides);
  bool contiguous =
      current.arr.flags().contiguous && data_size == current.arr.data_size();

  out.copy_shared_buffer(
      current.arr,
      out_strides,
      {contiguous, row_contiguous, col_contiguous},
      current.arr.data_size());
}

void apply_inverse_scale(
    array& arr,
    const std::vector<size_t>& axes,
    const array& out,
    rocm::CommandEncoder& encoder) {
  if (axes.empty()) {
    return;
  }

  double scale = 1.0;
  for (auto axis : axes) {
    scale /= out.shape(axis);
  }

  size_t size = arr.data_size();
  constexpr int block_size = 256;
  int num_blocks = static_cast<int>(std::min<size_t>(
      (size + block_size - 1) / block_size, 65535));

  encoder.set_input_array(arr);
  encoder.set_output_array(arr);

  if (arr.dtype() == float32) {
    float scale_f = static_cast<float>(scale);
    encoder.add_kernel_node(
        &rocm::scale_fft_output<float>,
        dim3(num_blocks),
        dim3(block_size),
        0,
        gpu_ptr<float>(arr),
        scale_f,
        size);
  } else if (arr.dtype() == complex64) {
    hipFloatComplex scale_f =
        make_hipFloatComplex(static_cast<float>(scale), 0.0f);
    encoder.add_kernel_node(
        &rocm::scale_fft_output<hipFloatComplex>,
        dim3(num_blocks),
        dim3(block_size),
        0,
        gpu_ptr<hipFloatComplex>(arr),
        scale_f,
        size);
  } else {
    throw std::runtime_error("[FFT] Unsupported dtype for inverse scaling.");
  }
}

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  auto& in = inputs[0];

  if (out.size() == 0) {
    return;
  }

  auto order = make_identity_order(in.ndim());
  OrderedArray current{in, std::move(order)};

  std::vector<int> axis_sequence;
  axis_sequence.reserve(axes_.size());
  if (inverse_) {
    for (auto axis : axes_) {
      axis_sequence.push_back(static_cast<int>(axis));
    }
  } else {
    for (int i = static_cast<int>(axes_.size()) - 1; i >= 0; --i) {
      axis_sequence.push_back(static_cast<int>(axes_[i]));
    }
  }

  int real_axis = axes_.empty() ? -1 : static_cast<int>(axes_.back());

  for (size_t i = 0; i < axis_sequence.size(); ++i) {
    int axis = axis_sequence[i];
    bool step_real = real_ && axis == real_axis;
    auto transform_type = step_real
        ? (inverse_ ? FFTTransformType::C2R : FFTTransformType::R2C)
        : FFTTransformType::C2C;

    // hipFFT, like cuFFT, may overwrite its input for C2R, so the original
    // input is only used directly when it is donatable to the output.
    auto prepared = prepare_input(
        current,
        axis,
        /* allow_direct = */ transform_type != FFTTransformType::C2R ||
            is_donatable(in, out),
        encoder,
        s);

    Shape step_shape = prepared.arr.shape();
    if (step_real) {
      step_shape.back() = out.shape(axis);
    }

    Dtype step_dtype =
        transform_type == FFTTransformType::C2R ? float32 : complex64;
    array step_out(std::move(step_shape), step_dtype, nullptr, {});
    execute_fft(prepared.arr, step_out, transform_type, inverse_, encoder);
    encoder.add_temporary(step_out);

    current = {std::move(step_out), std::move(prepared.order)};
  }

  if (inverse_) {
    apply_inverse_scale(current.arr, axes_, out, encoder);
  }

  restore_output_layout(current, out);
}

} // namespace mlx::core
