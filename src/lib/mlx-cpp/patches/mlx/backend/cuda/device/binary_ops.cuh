// Copyright © 2025 Apple Inc.
// Patched by mlxcel: added mixed-type overloads to arithmetic ops so that
// JIT-compiled fused kernels can operate on (float, __nv_bfloat16) or
// (__nv_bfloat16, float) pairs without template-deduction failures.
// Without this patch, Gemma 4 and Qwen 3.5 models crash on CUDA because
// their computation graphs produce mixed bf16/float intermediates in the
// compiled DAG.

#include "mlx/backend/cuda/device/unary_ops.cuh"

#include <cuda/std/array>

namespace mlx::core::cu {

// ---------------------------------------------------------------------------
// Mixed-type promotion helper: when T != U, compute in float.
// Used by arithmetic binary ops to handle bf16/float mixed operands in
// JIT fused kernels.
// ---------------------------------------------------------------------------
#define MLX_MIXED_BINARY_OP(OP_EXPR)                                         \
  template <                                                                  \
      typename T,                                                             \
      typename U,                                                             \
      typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, U>>>        \
  __device__ float operator()(T x, U y) {                                    \
    return OP_EXPR;                                                           \
  }

struct Add {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x + y;
  }
  MLX_MIXED_BINARY_OP(float(x) + float(y))
};

struct FloorDivide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      auto q = x / y;
      if constexpr (cuda::std::is_signed_v<T>) {
        if (x % y != 0 && (x < 0) != (y < 0)) {
          q -= 1;
        }
      }
      return q;
    } else if constexpr (is_complex_v<T>) {
      // Complex is not supported, simply make compiler happy.
      return x / y;
    } else {
      return cuda::std::floor(x / y);
    }
  }
  MLX_MIXED_BINARY_OP(cuda::std::floor(float(x) / float(y)))
};

struct Divide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x / y;
  }
  MLX_MIXED_BINARY_OP(float(x) / float(y))
};

struct Remainder {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      if constexpr (cuda::std::is_signed_v<T>) {
        auto r = x % y;
        if (r != 0 && (r < 0 != y < 0)) {
          r += y;
        }
        return r;
      } else {
        return x % y;
      }
    } else if constexpr (is_complex_v<T>) {
      return x % y;
    } else {
      T r = cuda::std::fmod(x, y);
      if (r != 0 && (r < 0 != y < 0)) {
        r = r + y;
      }
      return r;
    }
  }
};

struct Equal {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    using cuda::std::isnan;
    if constexpr (is_complex_v<T>) {
      return x == y ||
          (isnan(x.real()) && isnan(y.real()) && isnan(x.imag()) &&
           isnan(y.imag())) ||
          (x.real() == y.real() && isnan(x.imag()) && isnan(y.imag())) ||
          (isnan(x.real()) && isnan(y.real()) && x.imag() == y.imag());
    } else {
      return x == y || (isnan(x) && isnan(y));
    }
  }
};

struct Greater {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x <= y;
  }
};

struct LogAddExp {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (is_complex_v<T>) {
      if (cuda::std::isnan(x.real()) || cuda::std::isnan(x.imag()) ||
          cuda::std::isnan(y.real()) || cuda::std::isnan(y.imag())) {
        return {
            cuda::std::numeric_limits<float>::quiet_NaN(),
            cuda::std::numeric_limits<float>::quiet_NaN()};
      }
      auto max = x.real() > y.real() ? x : y;
      auto min = x.real() < y.real() ? x : y;
      auto min_real = min.real();
      auto max_real = max.real();
      if (!cuda::std::isfinite(min_real) && (min_real == max_real)) {
        if (min_real < 0) {
          return min;
        } else {
          return Log{}(Exp{}(min) + Exp{}(max));
        }
      } else {
        return Log1p{}(Exp{}(min - max)) + max;
      }
    } else {
      if (cuda::std::isnan(x) || cuda::std::isnan(y)) {
        return cuda::std::numeric_limits<T>::quiet_NaN();
      }
      T maxval = max(x, y);
      T minval = min(x, y);
      return (minval == -cuda::std::numeric_limits<T>::infinity() ||
              maxval == cuda::std::numeric_limits<T>::infinity())
          ? maxval
          : T(maxval + cuda::std::log1p(cuda::std::exp(minval - maxval)));
    }
  };
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return max(x, y);
    } else if constexpr (is_complex_v<T>) {
      if (cuda::std::isnan(x.real()) || cuda::std::isnan(x.imag())) {
        return x;
      }
      return x > y ? x : y;
    } else {
      if (cuda::std::isnan(x)) {
        return x;
      }
      return x > y ? x : y;
    }
  }
  // Mixed-type: promote to float, then return float
  template <
      typename T,
      typename U,
      typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, U>>>
  __device__ float operator()(T x, U y) {
    float xf = float(x), yf = float(y);
    if (cuda::std::isnan(xf)) {
      return xf;
    }
    return xf > yf ? xf : yf;
  }
};

struct Minimum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return min(x, y);
    } else if constexpr (is_complex_v<T>) {
      if (cuda::std::isnan(x.real()) || cuda::std::isnan(x.imag())) {
        return x;
      }
      return x < y ? x : y;
    } else {
      if (cuda::std::isnan(x)) {
        return x;
      }
      return x < y ? x : y;
    }
  }
  // Mixed-type: promote to float, then return float
  template <
      typename T,
      typename U,
      typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, U>>>
  __device__ float operator()(T x, U y) {
    float xf = float(x), yf = float(y);
    if (cuda::std::isnan(xf)) {
      return xf;
    }
    return xf < yf ? xf : yf;
  }
};

struct Multiply {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x * y;
  }
  MLX_MIXED_BINARY_OP(float(x) * float(y))
};

struct NotEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    if constexpr (is_complex_v<T>) {
      return x.real() != y.real() || x.imag() != y.imag();
    } else {
      return x != y;
    }
  }
};

struct Power {
  template <typename T>
  __device__ T operator()(T base, T exp) {
    if constexpr (cuda::std::is_integral_v<T>) {
      T res = 1;
      // Raising an integer to a negative power is undefined
      if constexpr (cuda::std::is_signed_v<T>) {
        if (exp < 0) {
          return 0;
        }
      }
      while (exp) {
        if (exp & 1) {
          res *= base;
        }
        exp >>= 1;
        base *= base;
      }
      return res;
    } else if constexpr (is_complex_v<T>) {
      return cuda::std::pow(base, exp);
    } else {
      return cuda::std::pow(base, exp);
    }
  }
  MLX_MIXED_BINARY_OP(cuda::std::pow(float(x), float(y)))
};

struct Subtract {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x - y;
  }
  MLX_MIXED_BINARY_OP(float(x) - float(y))
};

struct LogicalAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x && y;
  };
};

struct LogicalOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x || y;
  };
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x & y;
  };
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x | y;
  };
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x ^ y;
  };
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x << y;
  };
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x >> y;
  };
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T y, T x) {
    return cuda::std::atan2(y, x);
  }
  MLX_MIXED_BINARY_OP(cuda::std::atan2(float(x), float(y)))
};

struct DivMod {
  template <typename T>
  __device__ cuda::std::array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};

#undef MLX_MIXED_BINARY_OP

} // namespace mlx::core::cu
