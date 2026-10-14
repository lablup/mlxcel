// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/quantized/quantized.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core {

namespace rocm {

template <typename T, typename ScaleT, int BITS>
__global__ void fp_quantize_kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    ScaleT* __restrict__ scales,
    int64_t num_groups,
    int group_size) {
  // Grid capped by copy_grid_blocks (LOCAL_FIXES.md 46); the threads stride
  // the groups on a 64-bit counter, which also widens the group indexing.
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t group_idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       group_idx < num_groups;
       group_idx += stride) {
  const T* group_input = input + group_idx * group_size;

  // Find max absolute value in group (use float for computation)
  float max_abs = fabsf(static_cast<float>(group_input[0]));
  for (int i = 1; i < group_size; ++i) {
    max_abs = fmaxf(max_abs, fabsf(static_cast<float>(group_input[i])));
  }

  // Compute scale (symmetric quantization)
  float max_quant = static_cast<float>((1 << (BITS - 1)) - 1);
  float scale = max_abs / max_quant;

  // Avoid division by zero
  if (scale == 0.0f) {
    scale = 1.0f;
  }

  scales[group_idx] = static_cast<ScaleT>(scale);

  // Quantize values
  int64_t output_idx = group_idx * (group_size * BITS / 8);
  int group_bytes = group_size * BITS / 8;
  constexpr uint8_t mask = static_cast<uint8_t>((1u << BITS) - 1u);

  for (int i = 0; i < group_bytes; ++i) {
    output[output_idx + i] = 0;
  }

  int8_t min_val = -(1 << (BITS - 1));
  int8_t max_val = (1 << (BITS - 1)) - 1;

  for (int i = 0; i < group_size; ++i) {
    float val = static_cast<float>(group_input[i]);
    int quant_val = static_cast<int>(roundf(val / scale));
    quant_val = max(static_cast<int>(min_val), min(static_cast<int>(max_val), quant_val));

    int bit_index = i * BITS;
    int64_t byte_idx = output_idx + (bit_index >> 3);
    int bit_offset = bit_index & 0x7;

    uint32_t shifted =
        static_cast<uint32_t>(static_cast<uint8_t>(quant_val) & mask)
        << bit_offset;

    output[byte_idx] |= static_cast<uint8_t>(shifted & 0xFF);
    if (bit_offset + BITS > 8) {
      output[byte_idx + 1] |= static_cast<uint8_t>((shifted >> 8) & 0xFF);
    }
  }
  }
}

template <typename T, typename ScaleT, int BITS>
__global__ void fp_dequantize_kernel(
    const uint8_t* __restrict__ input,
    const ScaleT* __restrict__ scales,
    T* __restrict__ output,
    int64_t num_groups,
    int group_size) {
  // Grid capped by copy_grid_blocks (LOCAL_FIXES.md 46); the threads stride
  // the groups on a 64-bit counter, which also widens the group indexing.
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t group_idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       group_idx < num_groups;
       group_idx += stride) {
  float scale = static_cast<float>(scales[group_idx]);

  int64_t input_base = group_idx * (group_size * BITS / 8);
  T* group_output = output + group_idx * group_size;
  constexpr uint8_t mask = static_cast<uint8_t>((1u << BITS) - 1u);
  constexpr uint8_t sign_bit = static_cast<uint8_t>(1u << (BITS - 1));

  for (int i = 0; i < group_size; ++i) {
    int bit_index = i * BITS;
    int64_t byte_idx = input_base + (bit_index >> 3);
    int bit_offset = bit_index & 0x7;

    uint32_t packed = static_cast<uint32_t>(input[byte_idx]);
    if (bit_offset + BITS > 8) {
      packed |= static_cast<uint32_t>(input[byte_idx + 1]) << 8;
    }
    uint8_t uval = static_cast<uint8_t>((packed >> bit_offset) & mask);

    // Convert back to signed
    int8_t quant_val;
    if (uval & sign_bit) {
      quant_val = static_cast<int8_t>(uval | ~mask);
    } else {
      quant_val = static_cast<int8_t>(uval);
    }

    group_output[i] = static_cast<T>(static_cast<float>(quant_val) * scale);
  }
  }
}

// Optimized packed dequantize kernel
template <typename T, int BITS>
__global__ void fp_dequantize_packed_kernel(
    const uint8_t* __restrict__ input,
    const T* __restrict__ scales,
    T* __restrict__ output,
    int64_t size,
    int group_size) {
  constexpr int pack_factor = 8 / BITS;

  // Grid capped by copy_grid_blocks (LOCAL_FIXES.md 46); the threads stride
  // their packed chunks on a 64-bit counter.
  const int64_t stride = int64_t(blockDim.x) * gridDim.x * pack_factor;
  for (int64_t oindex = int64_t(blockIdx.x) * blockDim.x * pack_factor + threadIdx.x * pack_factor;
       oindex < size;
       oindex += stride) {
  size_t gindex = oindex / group_size;
  float scale = static_cast<float>(scales[gindex]);

  size_t idx = oindex / pack_factor;
  uint8_t val = input[idx];
  uint8_t mask = (1 << BITS) - 1;
  uint8_t sign_bit = static_cast<uint8_t>(1 << (BITS - 1));

  #pragma unroll
  for (int i = 0; i < pack_factor; ++i) {
    uint8_t uval = (val >> (BITS * i)) & mask;

    // Convert to signed
    int8_t quant_val;
    if (uval & sign_bit) {
      quant_val = static_cast<int8_t>(uval | ~mask);
    } else {
      quant_val = static_cast<int8_t>(uval);
    }

    output[oindex + i] = static_cast<T>(static_cast<float>(quant_val) * scale);
  }
  }
}

} // namespace rocm

void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  int64_t num_elements = w.size();
  int64_t num_groups = num_elements / group_size;

  int block_size = 256;
  uint32_t num_blocks = rocm::copy_grid_blocks(num_groups, block_size);
  
  enc.set_input_array(w);
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  
    #define LAUNCH_FP_QUANTIZE(T, ScaleT, BITS) \
      enc.add_kernel_node( \
          &rocm::fp_quantize_kernel<T, ScaleT, BITS>, \
          dim3(num_blocks), dim3(block_size), 0u, \
          gpu_ptr<T>(w), gpu_ptr<uint8_t>(wq), gpu_ptr<ScaleT>(scales), \
          num_groups, group_size)

    #define DISPATCH_BITS(T, ScaleT) \
      switch (bits) { \
        case 2: LAUNCH_FP_QUANTIZE(T, ScaleT, 2); break; \
        case 3: LAUNCH_FP_QUANTIZE(T, ScaleT, 3); break; \
        case 4: LAUNCH_FP_QUANTIZE(T, ScaleT, 4); break; \
        case 5: LAUNCH_FP_QUANTIZE(T, ScaleT, 5); break; \
        case 6: LAUNCH_FP_QUANTIZE(T, ScaleT, 6); break; \
        case 8: LAUNCH_FP_QUANTIZE(T, ScaleT, 8); break; \
        default: throw std::runtime_error("Unsupported bits for fp_quantize"); \
      }
    
    switch (w.dtype()) {
      case float32:
        DISPATCH_BITS(float, float);
        break;
      case float16:
        DISPATCH_BITS(__half, __half);
        break;
      case bfloat16:
        DISPATCH_BITS(hip_bfloat16, hip_bfloat16);
        break;
      default:
        throw std::runtime_error("Unsupported dtype for fp_quantize");
    }

    #undef DISPATCH_BITS
    #undef LAUNCH_FP_QUANTIZE
}

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s) {
  
  enc.set_input_array(wq);
  enc.set_input_array(scales);
  enc.set_output_array(w);
  
  // Use packed kernel for power-of-2 bits
  if (bits == 2 || bits == 4 || bits == 8) {
    int pack_factor = 8 / bits;
    int64_t size = w.size() / pack_factor;

    int block_size = 256;
    uint32_t num_blocks = rocm::copy_grid_blocks(
        (size + pack_factor - 1) / pack_factor, block_size);
    
      #define LAUNCH_FP_DEQUANTIZE_PACKED(T, BITS) \
        enc.add_kernel_node( \
            &rocm::fp_dequantize_packed_kernel<T, BITS>, \
            dim3(num_blocks), dim3(block_size), 0u, \
            gpu_ptr<uint8_t>(wq), gpu_ptr<T>(scales), gpu_ptr<T>(w), \
            w.size(), group_size)

      #define DISPATCH_BITS_PACKED(T) \
        switch (bits) { \
          case 2: LAUNCH_FP_DEQUANTIZE_PACKED(T, 2); break; \
          case 4: LAUNCH_FP_DEQUANTIZE_PACKED(T, 4); break; \
          case 8: LAUNCH_FP_DEQUANTIZE_PACKED(T, 8); break; \
          default: break; \
        }
      
      switch (w.dtype()) {
        case float32:
          DISPATCH_BITS_PACKED(float);
          break;
        case float16:
          DISPATCH_BITS_PACKED(__half);
          break;
        case bfloat16:
          DISPATCH_BITS_PACKED(hip_bfloat16);
          break;
        default:
          throw std::runtime_error("Unsupported dtype for fp_dequantize");
      }

      #undef DISPATCH_BITS_PACKED
      #undef LAUNCH_FP_DEQUANTIZE_PACKED
  } else {
    // Fallback for non-power-of-2 bits
    int64_t num_elements = w.size();
    int64_t num_groups = num_elements / group_size;

    int block_size = 256;
    uint32_t num_blocks = rocm::copy_grid_blocks(num_groups, block_size);
    
      #define LAUNCH_FP_DEQUANTIZE(T, ScaleT, BITS) \
        enc.add_kernel_node( \
            &rocm::fp_dequantize_kernel<T, ScaleT, BITS>, \
            dim3(num_blocks), dim3(block_size), 0u, \
            gpu_ptr<uint8_t>(wq), gpu_ptr<ScaleT>(scales), gpu_ptr<T>(w), \
            num_groups, group_size)

      #define DISPATCH_BITS(T, ScaleT) \
        switch (bits) { \
          case 3: LAUNCH_FP_DEQUANTIZE(T, ScaleT, 3); break; \
          case 5: LAUNCH_FP_DEQUANTIZE(T, ScaleT, 5); break; \
          case 6: LAUNCH_FP_DEQUANTIZE(T, ScaleT, 6); break; \
          default: throw std::runtime_error("Unsupported bits for fp_dequantize"); \
        }
      
      switch (w.dtype()) {
        case float32:
          DISPATCH_BITS(float, float);
          break;
        case float16:
          DISPATCH_BITS(__half, __half);
          break;
        case bfloat16:
          DISPATCH_BITS(hip_bfloat16, hip_bfloat16);
          break;
        default:
          throw std::runtime_error("Unsupported dtype for fp_dequantize");
      }

      #undef DISPATCH_BITS
      #undef LAUNCH_FP_DEQUANTIZE
  }
}

} // namespace mlx::core
