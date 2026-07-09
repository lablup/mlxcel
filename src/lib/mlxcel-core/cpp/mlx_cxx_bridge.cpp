// Copyright 2025 mlx-lm-rs authors
// Direct C++ bridge implementation for MLX via cxx

#include "mlx_cxx_internal.h"

namespace mlx_cxx {

using namespace mlx::core;

// to_shape / to_dtype / from_dtype / gelu_tanh_approx live in
// mlx_cxx_internal.h so the split-out TUs (kernels/nemotron/ext) share them.

// Stream functions.
std::unique_ptr<MlxStream> default_stream() {
    return std::make_unique<MlxStream>(mlx::core::default_stream(mlx::core::default_device()));
}

std::unique_ptr<MlxStream> new_stream_on_device(bool gpu) {
    Device d = gpu ? Device::gpu : Device::cpu;
    return std::make_unique<MlxStream>(mlx::core::new_stream(d));
}

void synchronize_stream(const MlxStream& stream) {
    mlx::core::synchronize(stream.inner);
}

// Thread-local stream surface (mlx-vlm PR #1050).
//
// Each `ThreadLocalStream` registers a TLS slot keyed by a stream
// index. When a different thread first calls
// `stream_from_thread_local_stream` with the same handle, MLX
// transparently allocates a dedicated `Stream` for that thread on the
// same device. Synchronization through `synchronize(ThreadLocalStream)`
// targets the calling thread's resolved stream — exactly what the
// generation stream owners want so dispatch and sync stay paired.
std::unique_ptr<MlxThreadLocalStream> new_thread_local_stream_gpu() {
    return std::make_unique<MlxThreadLocalStream>(
        mlx::core::new_thread_local_stream(mlx::core::Device::gpu));
}

std::unique_ptr<MlxStream> stream_from_thread_local_stream(const MlxThreadLocalStream& tls) {
    return std::make_unique<MlxStream>(mlx::core::stream_from_thread_local_stream(tls.inner));
}

void synchronize_thread_local_stream(const MlxThreadLocalStream& tls) {
    mlx::core::synchronize(tls.inner);
}

// --- Multi-GPU device-index surface (epic #486, sub-issue #487) ---
//
// Portable across backends via MLX's `device_count` and the
// `Device(DeviceType, index)` constructor; no CUDA headers required. See
// the header comment for cross-backend semantics.
int32_t gpu_device_count() {
    int count = mlx::core::device_count(mlx::core::Device::gpu);
    // Clamp to >= 1: a CPU-only build reports 0 GPUs but `Device::gpu`
    // still resolves to the single default compute device, and callers
    // (and the acceptance criteria) expect at least one.
    return count >= 1 ? static_cast<int32_t>(count) : 1;
}

std::unique_ptr<MlxStream> new_stream_on_gpu_index(int32_t index) {
    return std::make_unique<MlxStream>(
        mlx::core::new_stream(Device(Device::gpu, index)));
}

void set_default_device_index(int32_t index) {
    mlx::core::set_default_device(Device(Device::gpu, index));
}

std::unique_ptr<MlxThreadLocalStream> new_thread_local_stream_gpu_index(int32_t index) {
    return std::make_unique<MlxThreadLocalStream>(
        mlx::core::new_thread_local_stream(Device(Device::gpu, index)));
}

std::unique_ptr<MlxArray> copy_array_to_stream(const MlxArray& a, const MlxStream& stream) {
    return std::make_unique<MlxArray>(mlx::core::copy(a.inner, stream.inner));
}

// Array factory functions.
std::unique_ptr<MlxArray> zeros(rust::Slice<const int32_t> shape, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::zeros(to_shape(shape), to_dtype(dtype)));
}

std::unique_ptr<MlxArray> zeros_stream(rust::Slice<const int32_t> shape, int32_t dtype, const MlxStream& stream) {
    return std::make_unique<MlxArray>(mlx::core::zeros(to_shape(shape), to_dtype(dtype), stream.inner));
}

std::unique_ptr<MlxArray> ones(rust::Slice<const int32_t> shape, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::ones(to_shape(shape), to_dtype(dtype)));
}

std::unique_ptr<MlxArray> ones_stream(rust::Slice<const int32_t> shape, int32_t dtype, const MlxStream& stream) {
    return std::make_unique<MlxArray>(mlx::core::ones(to_shape(shape), to_dtype(dtype), stream.inner));
}

std::unique_ptr<MlxArray> full_f32(rust::Slice<const int32_t> shape, float value, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::full(to_shape(shape), value, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> eye(int32_t n, int32_t m, int32_t k, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::eye(n, m, k, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> linspace(float start, float stop, int32_t num, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::linspace(start, stop, num, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> zeros_like(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::zeros_like(a.inner));
}

std::unique_ptr<MlxArray> ones_like(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::ones_like(a.inner));
}

std::unique_ptr<MlxArray> full_like(const MlxArray& a, float value) {
    return std::make_unique<MlxArray>(mlx::core::full_like(a.inner, value));
}

std::unique_ptr<MlxArray> from_slice_f32(rust::Slice<const float> data, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(array(data.data(), to_shape(shape)));
}

std::unique_ptr<MlxArray> from_slice_i32(rust::Slice<const int32_t> data, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(array(data.data(), to_shape(shape)));
}

std::unique_ptr<MlxArray> from_slice_u32(rust::Slice<const uint32_t> data, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(array(data.data(), to_shape(shape)));
}

std::unique_ptr<MlxArray> from_slice_i64(rust::Slice<const int64_t> data, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(array(data.data(), to_shape(shape)));
}

// Helper to create array from typed pointer
// Does NOT call eval - caller is responsible for ensuring data remains valid until eval
template<typename T>
static std::unique_ptr<MlxArray> make_array_typed(const uint8_t* data, const Shape& shape, Dtype dtype) {
    auto result = array(reinterpret_cast<const T*>(data), shape, dtype);
    return std::make_unique<MlxArray>(result);
}

// Create array from raw bytes with specified dtype
// Uses MLX's array constructor with typed pointer cast
// IMPORTANT: Caller must call eval() on the result before source data goes out of scope
std::unique_ptr<MlxArray> from_bytes(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape, int32_t dtype) {
    auto mlx_dtype = to_dtype(dtype);
    auto mlx_shape = to_shape(shape);

    // Create array from raw data with correct pointer type
    switch (dtype) {
        case 3:  // UINT32
            return make_array_typed<uint32_t>(data.data(), mlx_shape, mlx_dtype);
        case 4:  // UINT64
            return make_array_typed<uint64_t>(data.data(), mlx_shape, mlx_dtype);
        case 7:  // INT32
            return make_array_typed<int32_t>(data.data(), mlx_shape, mlx_dtype);
        case 8:  // INT64
            return make_array_typed<int64_t>(data.data(), mlx_shape, mlx_dtype);
        case 10:  // FLOAT32
            return make_array_typed<float>(data.data(), mlx_shape, mlx_dtype);
        case 9:  // FLOAT16: reinterpret 2-byte halfs, not per-byte uint8 casts
            return make_array_typed<mlx::core::float16_t>(data.data(), mlx_shape, mlx_dtype);
        case 12:  // BFLOAT16: reinterpret 2-byte bf16, not per-byte uint8 casts
            return make_array_typed<mlx::core::bfloat16_t>(data.data(), mlx_shape, mlx_dtype);
        case 1:  // UINT8
        default:
            return std::make_unique<MlxArray>(array(data.data(), mlx_shape, mlx_dtype));
    }
}

std::unique_ptr<MlxArray> from_bytes_nocopy(
    rust::Slice<const uint8_t> data,
    rust::Slice<const int32_t> shape,
    int32_t dtype) {
    auto mlx_dtype = to_dtype(dtype);
    auto mlx_shape = to_shape(shape);
    void* ptr = const_cast<uint8_t*>(data.data());
    auto noop_deleter = [](void*) {};
    return std::make_unique<MlxArray>(array(ptr, mlx_shape, mlx_dtype, noop_deleter));
}

// Helper function to convert float16 bits to float32
inline float f16_to_f32_bits(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;  // Zero
        } else {
            // Denormalized - convert to normalized
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);  // Inf/NaN
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);  // Normalized
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Helper function to convert bfloat16 bits to float32
inline float bf16_to_f32_bits(uint16_t h) {
    // bfloat16 is just the upper 16 bits of float32
    uint32_t f = static_cast<uint32_t>(h) << 16;
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Create half-precision array from raw bytes (bfloat16 or float16).
//
// Native bf16/fp16 mode (default): Creates arrays in their native dtype.
// The CUDA backend patches (#36-#39) ensure bf16 tensors compute natively
// without copy_v conversion overhead, halving memory usage for bf16 models.
//
// Legacy fp32 mode (MLX_BF16_NATIVE=0): Converts to fp32 at load time.
// Use when running on hardware/configurations without bf16 compute patches.
std::unique_ptr<MlxArray> from_bytes_f16(rust::Slice<const uint8_t> data, rust::Slice<const int32_t> shape, bool is_bfloat16) {
    auto mlx_shape = to_shape(shape);

    size_t num_elements = 1;
    for (auto s : mlx_shape) num_elements *= s;

    // Check for legacy fp32 conversion mode
    static bool use_native = []() {
        const char* env = std::getenv("MLX_BF16_NATIVE");
        return env == nullptr || std::string(env) != "0";
    }();

    if (use_native && is_bfloat16) {
        // Native bf16: create array directly with bfloat16 dtype.
        // SAFETY: bfloat16_t is a trivial struct wrapping uint16_t (2-byte aligned).
        // Safetensors data is memory-mapped with naturally aligned tensor offsets,
        // so the uint8_t* pointer is guaranteed to be 2-byte aligned for bf16 data.
        // MLX's array constructor copies the data immediately via std::copy.
        const mlx::core::bfloat16_t* bf16_src =
            reinterpret_cast<const mlx::core::bfloat16_t*>(data.data());
        return std::make_unique<MlxArray>(
            array(bf16_src, mlx_shape, mlx::core::bfloat16));
    }

    // fp16: keep native on Metal (macOS). On CUDA, fall through to fp32
    // conversion because CUDA SDPA doesn't support fp16 output with fp32 mask.
#ifdef __APPLE__
    if (!is_bfloat16) {
        const mlx::core::float16_t* f16_src =
            reinterpret_cast<const mlx::core::float16_t*>(data.data());
        return std::make_unique<MlxArray>(
            array(f16_src, mlx_shape, mlx::core::float16));
    }
#endif

    // Legacy fp32 conversion path
    std::vector<float> float_data(num_elements);
    const uint16_t* src = reinterpret_cast<const uint16_t*>(data.data());

    if (is_bfloat16) {
        for (size_t i = 0; i < num_elements; ++i) {
            float_data[i] = bf16_to_f32_bits(src[i]);
        }
    } else {
        for (size_t i = 0; i < num_elements; ++i) {
            float_data[i] = f16_to_f32_bits(src[i]);
        }
    }

    return std::make_unique<MlxArray>(array(float_data.data(), mlx_shape));
}

// Array property accessors.
rust::Vec<int32_t> array_shape(const MlxArray& arr) {
    rust::Vec<int32_t> result;
    const auto& shape = arr.inner.shape();
    for (auto s : shape) {
        result.push_back(s);
    }
    return result;
}

int32_t array_dtype(const MlxArray& arr) {
    return from_dtype(arr.inner.dtype());
}

size_t array_size(const MlxArray& arr) {
    return arr.inner.size();
}

size_t array_ndim(const MlxArray& arr) {
    return arr.inner.ndim();
}

size_t array_itemsize(const MlxArray& arr) {
    return arr.inner.itemsize();
}

size_t array_nbytes(const MlxArray& arr) {
    return arr.inner.nbytes();
}

// Array data access (scalar extraction).
float item_f32(const MlxArray& arr) {
    return const_cast<array&>(arr.inner).item<float>();
}

int32_t item_i32(const MlxArray& arr) {
    return const_cast<array&>(arr.inner).item<int32_t>();
}

int64_t item_i64(const MlxArray& arr) {
    return const_cast<array&>(arr.inner).item<int64_t>();
}

bool item_bool(const MlxArray& arr) {
    return const_cast<array&>(arr.inner).item<bool>();
}

rust::Vec<uint8_t> array_to_raw_bytes(const MlxArray& arr) {
    // Ensure the array is evaluated and contiguous
    auto a = mlx::core::contiguous(arr.inner);
    mlx::core::eval(a);

    size_t nbytes = a.nbytes();
    const auto* data = reinterpret_cast<const uint8_t*>(a.data<void>());

    rust::Vec<uint8_t> result;
    result.reserve(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
        result.push_back(data[i]);
    }
    return result;
}

// Same body as `array_to_raw_bytes`, but declared `-> Result<Vec<u8>>` on the
// Rust side so cxx wraps the call in a try/catch and converts any thrown MLX
// exception (an allocation failure making the contiguous copy, or any deferred
// error forced by the eval) into a Rust `Err` instead of letting it cross the
// FFI boundary uncaught (which aborts the process). The audio synthesis readback
// uses this so the whole contiguous + eval + copy-out step is recoverable at one
// fallible boundary.
rust::Vec<uint8_t> try_array_to_raw_bytes(const MlxArray& arr) {
    auto a = mlx::core::contiguous(arr.inner);
    mlx::core::eval(a);

    size_t nbytes = a.nbytes();
    const auto* data = reinterpret_cast<const uint8_t*>(a.data<void>());

    rust::Vec<uint8_t> result;
    result.reserve(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
        result.push_back(data[i]);
    }
    return result;
}

// Evaluation.
void eval(const MlxArray& arr) {
    const_cast<array&>(arr.inner).eval();
}

// Same as `eval`, but declared `-> Result<()>` on the Rust side so cxx wraps
// the call in a try/catch and converts any thrown MLX exception into a Rust
// `Err` instead of letting it cross the FFI boundary uncaught (which aborts the
// process). The body is otherwise identical to `eval`.
void try_eval(const MlxArray& arr) {
    const_cast<array&>(arr.inner).eval();
}

void eval_all(rust::Slice<const MlxArray* const> arrays) {
    std::vector<array> arrs;
    arrs.reserve(arrays.size());
    for (const auto* a : arrays) {
        arrs.push_back(a->inner);
    }
    mlx::core::eval(arrs);
}

// Element-wise binary operations.
std::unique_ptr<MlxArray> add(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::add(a.inner, b.inner));
}

std::unique_ptr<MlxArray> subtract(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::subtract(a.inner, b.inner));
}

std::unique_ptr<MlxArray> remainder(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::remainder(a.inner, b.inner));
}

std::unique_ptr<MlxArray> multiply(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::multiply(a.inner, b.inner));
}

std::unique_ptr<MlxArray> divide(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::divide(a.inner, b.inner));
}

std::unique_ptr<MlxArray> maximum(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::maximum(a.inner, b.inner));
}

std::unique_ptr<MlxArray> minimum(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::minimum(a.inner, b.inner));
}

// Element-wise unary operations.
std::unique_ptr<MlxArray> negative(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::negative(a.inner));
}

std::unique_ptr<MlxArray> abs(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::abs(a.inner));
}

std::unique_ptr<MlxArray> exp(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::exp(a.inner));
}

std::unique_ptr<MlxArray> log(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log(a.inner));
}

std::unique_ptr<MlxArray> sqrt(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sqrt(a.inner));
}

std::unique_ptr<MlxArray> rsqrt(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::rsqrt(a.inner));
}

std::unique_ptr<MlxArray> square(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::square(a.inner));
}

std::unique_ptr<MlxArray> sin(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sin(a.inner));
}

std::unique_ptr<MlxArray> cos(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::cos(a.inner));
}

std::unique_ptr<MlxArray> tanh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::tanh(a.inner));
}

std::unique_ptr<MlxArray> sigmoid(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sigmoid(a.inner));
}

std::unique_ptr<MlxArray> floor(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::floor(a.inner));
}

std::unique_ptr<MlxArray> ceil(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::ceil(a.inner));
}

std::unique_ptr<MlxArray> round(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::round(a.inner));
}

std::unique_ptr<MlxArray> sign(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sign(a.inner));
}

std::unique_ptr<MlxArray> reciprocal(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::reciprocal(a.inner));
}

// Trigonometric functions
std::unique_ptr<MlxArray> tan(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::tan(a.inner));
}

std::unique_ptr<MlxArray> sinh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sinh(a.inner));
}

std::unique_ptr<MlxArray> cosh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::cosh(a.inner));
}

std::unique_ptr<MlxArray> arcsin(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arcsin(a.inner));
}

std::unique_ptr<MlxArray> arccos(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arccos(a.inner));
}

std::unique_ptr<MlxArray> arctan(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arctan(a.inner));
}

std::unique_ptr<MlxArray> arctan2(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::arctan2(a.inner, b.inner));
}

std::unique_ptr<MlxArray> arcsinh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arcsinh(a.inner));
}

std::unique_ptr<MlxArray> arccosh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arccosh(a.inner));
}

std::unique_ptr<MlxArray> arctanh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::arctanh(a.inner));
}

std::unique_ptr<MlxArray> degrees(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::degrees(a.inner));
}

std::unique_ptr<MlxArray> radians(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::radians(a.inner));
}

// Mathematical/Special functions
std::unique_ptr<MlxArray> erf(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::erf(a.inner));
}

std::unique_ptr<MlxArray> erfinv(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::erfinv(a.inner));
}

std::unique_ptr<MlxArray> expm1(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::expm1(a.inner));
}

std::unique_ptr<MlxArray> log2(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log2(a.inner));
}

std::unique_ptr<MlxArray> log10(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log10(a.inner));
}

std::unique_ptr<MlxArray> logaddexp(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::logaddexp(a.inner, b.inner));
}

std::unique_ptr<MlxArray> power(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::power(a.inner, b.inner));
}

// Checks
std::unique_ptr<MlxArray> isnan(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::isnan(a.inner));
}

std::unique_ptr<MlxArray> isinf(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::isinf(a.inner));
}

std::unique_ptr<MlxArray> isfinite(const MlxArray& a) {
    // isfinite = !isnan && !isinf
    auto nan_check = mlx::core::isnan(a.inner);
    auto inf_check = mlx::core::isinf(a.inner);
    auto either = mlx::core::logical_or(nan_check, inf_check);
    return std::make_unique<MlxArray>(mlx::core::logical_not(either));
}

std::unique_ptr<MlxArray> isneginf(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::isneginf(a.inner));
}

std::unique_ptr<MlxArray> isposinf(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::isposinf(a.inner));
}

// Reduction operations.
std::unique_ptr<MlxArray> sum_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::sum(a.inner));
}

std::unique_ptr<MlxArray> sum_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::sum(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> mean_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::mean(a.inner));
}

std::unique_ptr<MlxArray> mean_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::mean(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> max_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::max(a.inner));
}

std::unique_ptr<MlxArray> max_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::max(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> min_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::min(a.inner));
}

std::unique_ptr<MlxArray> min_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::min(a.inner, axis, keepdims));
}

// Product reduction
std::unique_ptr<MlxArray> prod_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::prod(a.inner));
}

std::unique_ptr<MlxArray> prod_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::prod(a.inner, axis, keepdims));
}

// Variance and standard deviation
std::unique_ptr<MlxArray> var_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::var(a.inner));
}

std::unique_ptr<MlxArray> var_axis(const MlxArray& a, int32_t axis, bool keepdims, int32_t ddof) {
    return std::make_unique<MlxArray>(mlx::core::var(a.inner, axis, keepdims, ddof));
}

std::unique_ptr<MlxArray> std_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::std(a.inner));
}

std::unique_ptr<MlxArray> std_axis(const MlxArray& a, int32_t axis, bool keepdims, int32_t ddof) {
    return std::make_unique<MlxArray>(mlx::core::std(a.inner, axis, keepdims, ddof));
}

// Logsumexp
std::unique_ptr<MlxArray> logsumexp_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::logsumexp(a.inner));
}

std::unique_ptr<MlxArray> logsumexp_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::logsumexp(a.inner, axis, keepdims));
}

// All/any reductions
std::unique_ptr<MlxArray> all_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::all(a.inner));
}

std::unique_ptr<MlxArray> any_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::any(a.inner));
}

// Matrix operations.
std::unique_ptr<MlxArray> matmul(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::matmul(a.inner, b.inner));
}

// Same body as `matmul`, but declared `-> Result` on the Rust side so cxx wraps
// the call in a try/catch and converts MLX's eager shape-mismatch exception (and
// any other throw) into a Rust `Err` instead of letting it abort the process.
// MLX validates matmul shapes at graph-build time, so this catches the throw at
// op construction, not only at eval.
std::unique_ptr<MlxArray> try_matmul(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::matmul(a.inner, b.inner));
}

std::unique_ptr<MlxArray> transpose(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::transpose(a.inner));
}

std::unique_ptr<MlxArray> transpose_axes(const MlxArray& a, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::transpose(a.inner, axes_vec));
}

std::unique_ptr<MlxArray> reshape(const MlxArray& a, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(mlx::core::reshape(a.inner, to_shape(shape)));
}

// Shape operations.
std::unique_ptr<MlxArray> expand_dims(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::expand_dims(a.inner, axis));
}

// Multi-axis variant: mirrors Python's `mx.expand_dims(a, (ax0, ax1, ...))`,
// which ends up in the `expand_dims(array, std::vector<int>)` MLX overload.
// A single bridge call avoids the per-axis FFI + UniquePtr alloc overhead
// that was showing up in SwitchGeGLU decode profiles (first expand_dims
// taking ~0.8 ms because the preceding layer's graph had to sync, even
// though the second back-to-back call was ~0.02 ms).
std::unique_ptr<MlxArray> expand_dims_multi(
    const MlxArray& a,
    rust::Slice<const int32_t> axes
) {
    std::vector<int> ax;
    ax.reserve(axes.size());
    for (int32_t axis : axes) {
        ax.push_back(static_cast<int>(axis));
    }
    return std::make_unique<MlxArray>(mlx::core::expand_dims(a.inner, ax));
}

std::unique_ptr<MlxArray> squeeze(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::squeeze(a.inner));
}

std::unique_ptr<MlxArray> squeeze_axis(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::squeeze(a.inner, axis));
}

std::unique_ptr<MlxArray> broadcast_to(const MlxArray& a, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(mlx::core::broadcast_to(a.inner, to_shape(shape)));
}

// Flatten array
std::unique_ptr<MlxArray> flatten(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::flatten(a.inner));
}

std::unique_ptr<MlxArray> flatten_range(const MlxArray& a, int32_t start_axis, int32_t end_axis) {
    return std::make_unique<MlxArray>(mlx::core::flatten(a.inner, start_axis, end_axis));
}

// Move axis
std::unique_ptr<MlxArray> moveaxis(const MlxArray& a, int32_t source, int32_t destination) {
    return std::make_unique<MlxArray>(mlx::core::moveaxis(a.inner, source, destination));
}

// Pad array
std::unique_ptr<MlxArray> pad(const MlxArray& a, rust::Slice<const int32_t> pad_width, float pad_value) {
    // pad_width is pairs of [before, after] for each dimension
    // Convert to vector of pairs
    std::vector<std::pair<int, int>> width_pairs;
    for (size_t i = 0; i + 1 < pad_width.size(); i += 2) {
        width_pairs.push_back({pad_width[i], pad_width[i + 1]});
    }
    return std::make_unique<MlxArray>(mlx::core::pad(a.inner, width_pairs, array(pad_value)));
}

// Split array at indices - returns single concatenated result for simplicity
// For proper multi-output support, users should use slice
std::unique_ptr<MlxArray> split_at_indices(const MlxArray& a, rust::Slice<const int32_t> indices, int32_t axis) {
    // Convert to Shape (SmallVector<int>) which MLX expects
    Shape idx_shape(indices.begin(), indices.end());
    // This returns vector of arrays - we just return the first for simple use cases
    // Full split support would need different approach
    auto splits = mlx::core::split(a.inner, idx_shape, axis);
    if (splits.empty()) {
        return std::make_unique<MlxArray>(a.inner);
    }
    return std::make_unique<MlxArray>(std::move(splits[0]));
}

// Diagonal operations
std::unique_ptr<MlxArray> diag(const MlxArray& a, int32_t k) {
    return std::make_unique<MlxArray>(mlx::core::diag(a.inner, k));
}

std::unique_ptr<MlxArray> diagonal(const MlxArray& a, int32_t offset, int32_t axis1, int32_t axis2) {
    return std::make_unique<MlxArray>(mlx::core::diagonal(a.inner, offset, axis1, axis2));
}

// Type conversion.
std::unique_ptr<MlxArray> astype(const MlxArray& a, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::astype(a.inner, to_dtype(dtype)));
}

// Copy.
std::unique_ptr<MlxArray> copy(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::copy(a.inner));
}

// High-level operations for LLM inference.
std::unique_ptr<MlxArray> softmax(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::softmax(a.inner, axis));
}

std::unique_ptr<MlxArray> softmax_precise(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::softmax(a.inner, axis, true));
}

std::unique_ptr<MlxArray> log_softmax(const MlxArray& a, int32_t axis) {
    // log_softmax(x) = x - logsumexp(x)
    auto lse = mlx::core::logsumexp(a.inner, axis, true);
    return std::make_unique<MlxArray>(mlx::core::subtract(a.inner, lse));
}

std::unique_ptr<MlxArray> rms_norm(const MlxArray& x, const MlxArray& weight, float eps) {
    // RMS norm: x * weight / sqrt(mean(x^2) + eps)
    auto x_sq = mlx::core::square(x.inner);
    auto mean_sq = mlx::core::mean(x_sq, -1, true);
    auto norm = mlx::core::rsqrt(mlx::core::add(mean_sq, array(eps)));
    auto normalized = mlx::core::multiply(x.inner, norm);
    auto out = mlx::core::multiply(normalized, weight.inner);
    // Keep the high-level fallback dtype-compatible with MLX fast::rms_norm:
    // scalar eps arithmetic may promote bf16/f16 inputs to fp32, but model
    // layers expect the normalization result to stay on the activation dtype
    // to avoid extra memory traffic and copy kernels.
    if (out.dtype() != x.inner.dtype()) {
        out = mlx::core::astype(out, x.inner.dtype());
    }
    return std::make_unique<MlxArray>(out);
}

std::unique_ptr<MlxArray> layer_norm(const MlxArray& x, const MlxArray& weight,
                                     const MlxArray& bias, float eps) {
    // Layer norm: (x - mean) / sqrt(var + eps) * weight + bias
    auto mean = mlx::core::mean(x.inner, -1, true);
    auto centered = mlx::core::subtract(x.inner, mean);
    auto var = mlx::core::mean(mlx::core::square(centered), -1, true);
    auto norm = mlx::core::rsqrt(mlx::core::add(var, array(eps)));
    auto normalized = mlx::core::multiply(centered, norm);
    auto scaled = mlx::core::multiply(normalized, weight.inner);
    return std::make_unique<MlxArray>(mlx::core::add(scaled, bias.inner));
}

std::unique_ptr<MlxArray> concatenate(rust::Slice<const MlxArray* const> arrays, int32_t axis) {
    std::vector<array> arrs;
    arrs.reserve(arrays.size());
    for (const auto* a : arrays) {
        arrs.push_back(a->inner);
    }
    return std::make_unique<MlxArray>(mlx::core::concatenate(arrs, axis));
}

rust::Vec<std::unique_ptr<MlxArray>> split(const MlxArray& a, int32_t num_splits, int32_t axis) {
    auto splits = mlx::core::split(a.inner, num_splits, axis);
    rust::Vec<std::unique_ptr<MlxArray>> result;
    for (auto& s : splits) {
        result.push_back(std::make_unique<MlxArray>(std::move(s)));
    }
    return result;
}

std::unique_ptr<MlxArray> slice(const MlxArray& a,
                                rust::Slice<const int32_t> starts,
                                rust::Slice<const int32_t> stops) {
    Shape starts_shape(starts.begin(), starts.end());
    Shape stops_shape(stops.begin(), stops.end());
    return std::make_unique<MlxArray>(mlx::core::slice(a.inner, starts_shape, stops_shape));
}

std::unique_ptr<MlxArray> slice_update(const MlxArray& src,
                                        const MlxArray& update,
                                        rust::Slice<const int32_t> starts,
                                        rust::Slice<const int32_t> stops) {
    Shape starts_shape(starts.begin(), starts.end());
    Shape stops_shape(stops.begin(), stops.end());
    return std::make_unique<MlxArray>(mlx::core::slice_update(src.inner, update.inner, starts_shape, stops_shape));
}

std::unique_ptr<MlxArray> argmax(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::argmax(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> where_cond(const MlxArray& condition, const MlxArray& x, const MlxArray& y) {
    return std::make_unique<MlxArray>(mlx::core::where(condition.inner, x.inner, y.inner));
}

std::unique_ptr<MlxArray> greater(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::greater(a.inner, b.inner));
}

std::unique_ptr<MlxArray> less(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::less(a.inner, b.inner));
}

std::unique_ptr<MlxArray> equal(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::equal(a.inner, b.inner));
}

void random_seed(uint64_t seed) {
    mlx::core::random::seed(seed);
}

std::unique_ptr<MlxArray> random_categorical(const MlxArray& logits, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::random::categorical(logits.inner, axis));
}

// Transformer-specific high-level operations.
std::unique_ptr<MlxArray> rope_forward(
    const MlxArray& x,
    int32_t head_dim,
    float theta,
    int32_t offset,
    bool traditional
) {
    // Generate position-dependent rotation frequencies
    // freq = theta^(-2i/d) for i in [0, d/2)
    auto half_dim = head_dim / 2;
    std::vector<float> inv_freq;
    inv_freq.reserve(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        inv_freq.push_back(1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim));
    }
    auto freqs = array(inv_freq.data(), {half_dim});

    // Get sequence length from input shape
    auto shape = x.inner.shape();
    int seq_len = shape[shape.size() - 2];  // [..., seq_len, head_dim]

    // Create position indices
    std::vector<int> pos;
    pos.reserve(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        pos.push_back(offset + i);
    }
    auto positions = array(pos.data(), {seq_len});

    // Compute angles: pos * freq -> [seq_len, half_dim]
    auto pos_f = mlx::core::astype(positions, float32);
    pos_f = mlx::core::expand_dims(pos_f, 1);  // [seq_len, 1]
    freqs = mlx::core::expand_dims(freqs, 0);  // [1, half_dim]
    auto angles = mlx::core::matmul(pos_f, freqs);  // [seq_len, half_dim]

    // Compute cos and sin
    auto cos_vals = mlx::core::cos(angles);
    auto sin_vals = mlx::core::sin(angles);

    // Interleave or concatenate based on traditional flag
    if (traditional) {
        // Traditional: interleave cos, sin
        cos_vals = mlx::core::repeat(cos_vals, 2, -1);
        sin_vals = mlx::core::repeat(sin_vals, 2, -1);
    } else {
        // Modern: concatenate cos, sin
        cos_vals = mlx::core::concatenate({cos_vals, cos_vals}, -1);
        sin_vals = mlx::core::concatenate({sin_vals, sin_vals}, -1);
    }

    // Apply rotation
    auto x1 = mlx::core::slice(x.inner, {}, {}, {2});  // even indices
    auto x2 = mlx::core::slice(x.inner, {1}, {}, {2}); // odd indices

    // Rotate: x * cos + rotate(x) * sin
    auto rotated = mlx::core::concatenate({
        mlx::core::subtract(
            mlx::core::multiply(x.inner, cos_vals),
            mlx::core::multiply(
                mlx::core::concatenate({mlx::core::negative(x2), x1}, -1),
                sin_vals
            )
        )
    }, -1);

    return std::make_unique<MlxArray>(std::move(rotated));
}

std::unique_ptr<MlxArray> apply_rope(
    const MlxArray& x,
    const MlxArray& cos,
    const MlxArray& sin
) {
    // Split x into two halves
    int dim = x.inner.shape().back();
    int half_dim = dim / 2;
    auto x1 = mlx::core::slice(x.inner, {0, 0, 0, 0}, {-1, -1, -1, half_dim});
    auto x2 = mlx::core::slice(x.inner, {0, 0, 0, half_dim}, {-1, -1, -1, dim});

    // Rotate: [x1, x2] * cos + [-x2, x1] * sin
    auto rotated_x1 = mlx::core::subtract(
        mlx::core::multiply(x1, cos.inner),
        mlx::core::multiply(x2, sin.inner)
    );
    auto rotated_x2 = mlx::core::add(
        mlx::core::multiply(x2, cos.inner),
        mlx::core::multiply(x1, sin.inner)
    );

    return std::make_unique<MlxArray>(mlx::core::concatenate({rotated_x1, rotated_x2}, -1));
}

std::unique_ptr<MlxArray> scaled_dot_product_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask
) {
    // Compute attention scores: Q @ K^T * scale
    auto k_t = mlx::core::transpose(k.inner, {0, 1, 3, 2});
    auto scores = mlx::core::matmul(q.inner, k_t);
    scores = mlx::core::multiply(scores, array(scale));

    // Apply mask if provided
    if (mask != nullptr) {
        scores = mlx::core::add(scores, mask->inner);
    }

    // Softmax
    auto weights = mlx::core::softmax(scores, -1);

    // Apply attention: weights @ V
    return std::make_unique<MlxArray>(mlx::core::matmul(weights, v.inner));
}

std::unique_ptr<MlxArray> linear_forward(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray* bias
) {
    // x @ weight.T
    auto w_t = mlx::core::transpose(weight.inner);
    auto result = mlx::core::matmul(x.inner, w_t);

    if (bias != nullptr) {
        result = mlx::core::add(result, bias->inner);
    }

    return std::make_unique<MlxArray>(std::move(result));
}

std::unique_ptr<MlxArray> quantized_linear_forward(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray* linear_bias,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    bool is_affine = (mode.size() == 6 && std::memcmp(mode.data(), "affine", 6) == 0);
    array result = [&]() {
        if (is_affine) {
            if (biases) {
                return mlx::core::quantized_matmul(
                    x.inner, weight.inner, scales.inner, biases->inner,
                    true, group_size, bits);
            }
            return mlx::core::quantized_matmul(
                x.inner, weight.inner, scales.inner, std::nullopt,
                true, group_size, bits);
        }
        std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
        return mlx::core::quantized_matmul(
            x.inner, weight.inner, scales.inner, biases_opt,
            true, std::optional<int>(group_size), std::optional<int>(bits),
            std::string(mode.data(), mode.size()));
    }();

    if (linear_bias != nullptr) {
        result = mlx::core::add(result, linear_bias->inner);
    }

    return std::make_unique<MlxArray>(std::move(result));
}

std::unique_ptr<MlxArray> quantized_linear_forward_global_scale(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray* global_scale,
    const MlxArray* linear_bias,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    bool is_affine = (mode.size() == 6 && std::memcmp(mode.data(), "affine", 6) == 0);
    array result = [&]() {
        if (is_affine) {
            if (biases) {
                return mlx::core::quantized_matmul(
                    x.inner, weight.inner, scales.inner, biases->inner,
                    true, group_size, bits);
            }
            return mlx::core::quantized_matmul(
                x.inner, weight.inner, scales.inner, std::nullopt,
                true, group_size, bits);
        }
        std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
        return mlx::core::quantized_matmul(
            x.inner, weight.inner, scales.inner, biases_opt,
            true, std::optional<int>(group_size), std::optional<int>(bits),
            std::string(mode.data(), mode.size()));
    }();

    if (global_scale != nullptr) {
        auto dt = result.dtype();
        result = mlx::core::astype(mlx::core::multiply(result, global_scale->inner), dt);
    }

    if (linear_bias != nullptr) {
        result = mlx::core::add(result, linear_bias->inner);
    }

    return std::make_unique<MlxArray>(std::move(result));
}

std::unique_ptr<MlxArray> swiglu_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& up_proj,
    const MlxArray& down_proj
) {
    // gate = gate_proj(x)
    auto gate_t = mlx::core::transpose(gate_proj.inner);
    auto gate = mlx::core::matmul(x.inner, gate_t);

    // up = up_proj(x)
    auto up_t = mlx::core::transpose(up_proj.inner);
    auto up = mlx::core::matmul(x.inner, up_t);

    // SwiGLU: silu(gate) * up
    // silu(x) = x * sigmoid(x)
    auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
    auto activated = mlx::core::multiply(silu_gate, up);

    // down_proj(activated)
    auto down_t = mlx::core::transpose(down_proj.inner);
    return std::make_unique<MlxArray>(mlx::core::matmul(activated, down_t));
}

// Compiled swiglu activation using mlx::core::compile with shapeless=true
// This should provide kernel fusion like Python's @mx.compile(shapeless=True)
namespace {
    // Static compiled function - initialized once and reused
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_swiglu() {
        // Define the swiglu function: silu(gate) * x
        auto swiglu_fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& gate = inputs[0];
            const auto& x = inputs[1];
            // silu(gate) = gate * sigmoid(gate)
            auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
            return {mlx::core::multiply(silu_gate, x)};
        };
        // Compile with shapeless=true for kernel fusion
        return mlx::core::compile(swiglu_fn, true);
    }
}

// Compiled relu_squared: square(maximum(x, 0)) → single fused kernel
// Python equivalent: CompiledBroadcastMaximumSquare
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_relu_squared() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x = inputs[0];
            return {mlx::core::square(mlx::core::maximum(x, mlx::core::array(0)))};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_relu_squared(const MlxArray& x) {
    static auto compiled_fn = get_compiled_relu_squared();
    auto result = compiled_fn({x.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled silu: x * sigmoid(x) → single fused kernel
// Python equivalent: CompiledSigmoidMultiply
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_silu() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x = inputs[0];
            return {mlx::core::multiply(x, mlx::core::sigmoid(x))};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_silu(const MlxArray& x) {
    static auto compiled_fn = get_compiled_silu();
    auto result = compiled_fn({x.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

std::unique_ptr<MlxArray> compiled_swiglu_activation(
    const MlxArray& gate,
    const MlxArray& x
) {
    // Get or create the compiled function (thread-safe initialization)
    static auto compiled_fn = get_compiled_swiglu();

    // Call the compiled function
    auto result = compiled_fn({gate.inner, x.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled GptOss SwiGLU activation using the exact mlx-lm formulation:
//   x_glu = clip(x_glu, max=7)
//   x_linear = clip(x_linear, min=-7, max=7)
//   return x_glu * sigmoid(1.702 * x_glu) * (x_linear + 1)
// Used by: GptOss
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_gpt_oss_swiglu_activation() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x_linear_in = inputs[0];
            const auto& x_glu_in = inputs[1];

            auto pos_limit = mlx::core::array(7.0f);
            auto neg_limit = mlx::core::array(-7.0f);
            auto x_glu = mlx::core::minimum(x_glu_in, pos_limit);
            auto x_linear = mlx::core::maximum(x_linear_in, neg_limit);
            x_linear = mlx::core::minimum(x_linear, pos_limit);

            auto glu_scaled = mlx::core::multiply(mlx::core::array(1.702f), x_glu);
            auto out_glu = mlx::core::multiply(x_glu, mlx::core::sigmoid(glu_scaled));
            auto result = mlx::core::multiply(out_glu, mlx::core::add(x_linear, mlx::core::array(1.0f)));
            return {mlx::core::astype(result, x_linear_in.dtype())};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_gpt_oss_swiglu_activation(
    const MlxArray& x_linear,
    const MlxArray& x_glu
) {
    static auto compiled_fn = get_compiled_gpt_oss_swiglu_activation();
    auto result = compiled_fn({x_linear.inner, x_glu.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
// Used by: StarCoder2
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_gelu() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x = inputs[0];
            auto sqrt2 = array(std::sqrt(2.0f));
            auto half = array(0.5f);
            auto one = array(1.0f);
            auto erf_val = mlx::core::erf(mlx::core::divide(x, sqrt2));
            auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
            auto result = mlx::core::multiply(x, scale);
            return {mlx::core::astype(result, x.dtype())};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_gelu(const MlxArray& x) {
    static auto compiled_fn = get_compiled_gelu();
    auto result = compiled_fn({x.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled GELU approx: erf-based GELU (x * 0.5 * (1 + erf(x / sqrt(2))))
// Uses erf instead of tanh for numerical stability with bf16 inputs.
// Used by: legacy/tests
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_gelu_approx() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x = inputs[0];
            // Use erf-based GELU for numerical stability with bf16 inputs.
            // See gelu_approx() below for detailed explanation.
            auto sqrt2 = array(std::sqrt(2.0f));
            auto half = array(0.5f);
            auto one = array(1.0f);
            auto erf_val = mlx::core::erf(mlx::core::divide(x, sqrt2));
            auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
            auto result = mlx::core::multiply(x, scale);
            return {mlx::core::astype(result, x.dtype())};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_gelu_approx(const MlxArray& x) {
    static auto compiled_fn = get_compiled_gelu_approx();
    auto result = compiled_fn({x.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled GeGLU: gelu(gate) * x
// Used by: legacy/tests for precise GeGLU
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_geglu() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& gate = inputs[0];
            const auto& x = inputs[1];
            // gelu(gate) = gate * 0.5 * (1 + erf(gate / sqrt(2)))
            auto sqrt2 = array(std::sqrt(2.0f));
            auto half = array(0.5f);
            auto one = array(1.0f);
            auto erf_val = mlx::core::erf(mlx::core::divide(gate, sqrt2));
            auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
            auto gelu_gate = mlx::core::multiply(gate, scale);
            auto result = mlx::core::multiply(gelu_gate, x);
            return {mlx::core::astype(result, x.dtype())};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_geglu_activation(
    const MlxArray& gate,
    const MlxArray& x
) {
    static auto compiled_fn = get_compiled_geglu();
    auto result = compiled_fn({gate.inner, x.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled GeGLU using Python MLX's tanh-approx GELU.
// Used by: Gemma4
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_geglu_approx() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& gate = inputs[0];
            const auto& x = inputs[1];
            auto result = mlx::core::multiply(gelu_tanh_approx(gate), x);
            return {mlx::core::astype(result, x.dtype())};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_geglu_approx_activation(
    const MlxArray& gate,
    const MlxArray& x
) {
    static auto compiled_fn = get_compiled_geglu_approx();
    auto result = compiled_fn({gate.inner, x.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled gelu_topk: sparse GELU with dynamic threshold.
// Matches Python: gelu_approx(max(0, x - (mean + std * multiplier)))
// Used by: Gemma3n MLP layers with activation_sparsity > 0
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_gelu_topk() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x = inputs[0];
            const auto& std_multiplier = inputs[1];

            // mean and std along last axis
            auto mean = mlx::core::mean(x, /* axis */ -1, /* keepdims */ true);
            auto diff = mlx::core::subtract(x, mean);
            auto var = mlx::core::mean(mlx::core::square(diff), -1, true);
            auto stddev = mlx::core::sqrt(var);

            // cutoff = mean + std * multiplier
            auto cutoff = mlx::core::add(mean, mlx::core::multiply(stddev, std_multiplier));

            // zeroed = max(x - cutoff, 0)
            auto shifted = mlx::core::subtract(x, cutoff);
            auto zeroed = mlx::core::maximum(shifted, array(0.0f));

            // gelu_approx via erf
            auto sqrt2 = array(std::sqrt(2.0f));
            auto half = array(0.5f);
            auto one = array(1.0f);
            auto erf_val = mlx::core::erf(mlx::core::divide(zeroed, sqrt2));
            auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
            auto result = mlx::core::multiply(zeroed, scale);
            return {mlx::core::astype(result, x.dtype())};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_gelu_topk(
    const MlxArray& x,
    float std_multiplier
) {
    static auto compiled_fn = get_compiled_gelu_topk();
    auto mult = array(std_multiplier);
    auto result = compiled_fn({x.inner, mult});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled softcap: tanh(scores / cap) * cap
// Used by: Gemma2 attention with logit softcapping
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_softcap() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& scores = inputs[0];
            const auto& cap = inputs[1];
            auto scaled = mlx::core::divide(scores, cap);
            auto tanhed = mlx::core::tanh(scaled);
            auto result = mlx::core::multiply(tanhed, cap);
            return {mlx::core::astype(result, scores.dtype())};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_softcap(
    const MlxArray& scores,
    float cap
) {
    if (cap <= 0.0f) {
        return std::make_unique<MlxArray>(scores.inner);
    }
    static auto compiled_fn = get_compiled_softcap();
    auto cap_arr = array(cap);
    auto result = compiled_fn({scores.inner, cap_arr});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled clip_residual for float16 overflow prevention
// Used by: Gemma3 residual connections
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_clip_residual_f16() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x = inputs[0];
            const auto& y = inputs[1];
            auto x_f32 = mlx::core::astype(x, mlx::core::float32);
            auto y_f32 = mlx::core::astype(y, mlx::core::float32);
            auto sum = mlx::core::add(x_f32, y_f32);
            auto bound = array(65504.0f);
            auto neg_bound = mlx::core::negative(bound);
            auto clipped = mlx::core::clip(sum, neg_bound, bound);
            return {mlx::core::astype(clipped, mlx::core::float16)};
        };
        return mlx::core::compile(fn, true);
    }
}

std::unique_ptr<MlxArray> compiled_clip_residual(
    const MlxArray& x,
    const MlxArray& y
) {
    // Check if float16 (dtype code 9 in MLX enum order)
    if (x.inner.dtype() != mlx::core::float16) {
        return std::make_unique<MlxArray>(mlx::core::add(x.inner, y.inner));
    }
    static auto compiled_fn = get_compiled_clip_residual_f16();
    auto result = compiled_fn({x.inner, y.inner});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled softcap SDPA: Q@K^T * scale -> softcap -> mask -> softmax -> @V
// Fuses the entire attention score computation into a compiled graph
// Used by: Gemma2 attention with logit softcapping
namespace {
    // Compiled version without mask (single-token decode path)
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_softcap_sdpa_nomask() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& q = inputs[0];       // [B, H, 1, D]
            const auto& k = inputs[1];       // [B, H, S, D]
            const auto& v = inputs[2];       // [B, H, S, D]
            const auto& scale_arr = inputs[3];
            const auto& cap_arr = inputs[4];

            // scores = Q @ K^T
            auto k_t = mlx::core::transpose(k, {0, 1, 3, 2});
            auto scores = mlx::core::matmul(q, k_t);

            // scores = scores * scale
            scores = mlx::core::multiply(scores, scale_arr);

            // softcap: tanh(scores / cap) * cap
            scores = mlx::core::multiply(mlx::core::tanh(mlx::core::divide(scores, cap_arr)), cap_arr);

            // softmax
            auto probs = mlx::core::softmax(scores, -1);

            // probs @ V
            auto result = mlx::core::matmul(probs, v);
            return {mlx::core::astype(result, v.dtype())};
        };
        return mlx::core::compile(fn, true);  // shapeless=true
    }

    // Compiled version with mask (prefill path)
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_softcap_sdpa_masked() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& q = inputs[0];
            const auto& k = inputs[1];
            const auto& v = inputs[2];
            const auto& scale_arr = inputs[3];
            const auto& cap_arr = inputs[4];
            const auto& mask = inputs[5];

            auto k_t = mlx::core::transpose(k, {0, 1, 3, 2});
            auto scores = mlx::core::matmul(q, k_t);
            scores = mlx::core::multiply(scores, scale_arr);
            scores = mlx::core::multiply(mlx::core::tanh(mlx::core::divide(scores, cap_arr)), cap_arr);
            scores = mlx::core::add(scores, mask);
            auto probs = mlx::core::softmax(scores, -1);
            auto result = mlx::core::matmul(probs, v);
            return {mlx::core::astype(result, v.dtype())};
        };
        return mlx::core::compile(fn, true);
    }

    bool enable_softcap_gqa_decode_grouped_opt() {
        static const bool enabled = []() {
            // Backward compatibility for old rollback knob:
            // - MLXCEL_DISABLE_SOFTCAP_GQA_DECODE_GROUPED=1 -> force OFF
            // - MLXCEL_DISABLE_SOFTCAP_GQA_DECODE_GROUPED=0 -> force ON
            if (const char* v = std::getenv("MLXCEL_DISABLE_SOFTCAP_GQA_DECODE_GROUPED")) {
                return std::string_view(v) == "0";
            }
            if (const char* v = std::getenv("MLXCEL_ENABLE_SOFTCAP_GQA_DECODE_GROUPED")) {
                return std::string_view(v) != "0";
            }
            return false;
        }();
        return enabled;
    }
}

std::unique_ptr<MlxArray> compiled_softcap_sdpa(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    float softcap,
    const MlxArray* mask
) {
    auto scale_arr = array(scale);
    auto cap_arr = array(softcap);

    if (mask) {
        static auto compiled_fn = get_compiled_softcap_sdpa_masked();
        // Cast mask to Q's dtype if needed
        auto m = mask->inner;
        if (m.dtype() != q.inner.dtype()) {
            m = mlx::core::astype(m, q.inner.dtype());
        }
        auto result = compiled_fn({q.inner, k.inner, v.inner, scale_arr, cap_arr, m});
        return std::make_unique<MlxArray>(std::move(result[0]));
    } else {
        static auto compiled_fn = get_compiled_softcap_sdpa_nomask();
        auto result = compiled_fn({q.inner, k.inner, v.inner, scale_arr, cap_arr});
        return std::make_unique<MlxArray>(std::move(result[0]));
    }
}

// Softcap SDPA with GQA: do repeat_kv outside compiled graph, then call compiled SDPA
// This avoids shape issues in compiled functions while still fusing the attention math
// Used by: Gemma2 attention (GQA + softcap)
std::unique_ptr<MlxArray> compiled_softcap_sdpa_gqa(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    float softcap,
    int32_t n_rep,
    const MlxArray* mask
) {
    // Decode-specialized GQA softcap path:
    // Avoid materializing repeated K/V tensors when q_len == 1 and no mask.
    // This keeps K/V in [B, H_kv, S, D] and broadcasts over n_rep in matmul.
    if (!mask && n_rep > 1 && q.inner.shape(2) == 1 &&
        enable_softcap_gqa_decode_grouped_opt()) {
        auto q_shape = q.inner.shape();
        auto k_shape = k.inner.shape();
        int B = q_shape[0];
        int Hq = q_shape[1];
        int QL = q_shape[2];
        int D = q_shape[3];
        int Hk = k_shape[1];
        int S = k_shape[2];

        auto q_grouped = mlx::core::reshape(q.inner, {B, Hk, n_rep, QL, D});
        auto k_t = mlx::core::transpose(k.inner, {0, 1, 3, 2});
        k_t = mlx::core::reshape(k_t, {B, Hk, 1, D, S});
        auto scores = mlx::core::matmul(q_grouped, k_t);

        auto scale_arr = array(scale);
        auto cap_arr = array(softcap);
        scores = mlx::core::multiply(scores, scale_arr);
        scores = mlx::core::multiply(
            mlx::core::tanh(mlx::core::divide(scores, cap_arr)),
            cap_arr
        );

        auto probs = mlx::core::softmax(scores, -1);
        auto v_grouped = mlx::core::reshape(v.inner, {B, Hk, 1, S, D});
        auto ctx = mlx::core::matmul(probs, v_grouped);
        auto out = mlx::core::astype(mlx::core::reshape(ctx, {B, Hq, QL, D}), v.inner.dtype());
        return std::make_unique<MlxArray>(std::move(out));
    }

    // repeat_kv outside compiled graph (uses shape-dependent operations)
    auto do_repeat_kv = [n_rep](const array& x) -> array {
        if (n_rep == 1) return x;
        auto shape = x.shape();
        int B = shape[0], H = shape[1], S = shape[2], D = shape[3];
        auto expanded = mlx::core::reshape(x, {B, H, 1, S, D});
        auto broadcasted = mlx::core::broadcast_to(expanded, {B, H, n_rep, S, D});
        return mlx::core::reshape(broadcasted, {B, H * n_rep, S, D});
    };

    auto rk = do_repeat_kv(k.inner);
    auto rv = do_repeat_kv(v.inner);

    auto scale_arr = array(scale);
    auto cap_arr = array(softcap);

    if (mask) {
        static auto compiled_fn = get_compiled_softcap_sdpa_masked();
        auto m = mask->inner;
        if (m.dtype() != q.inner.dtype()) m = mlx::core::astype(m, q.inner.dtype());
        auto result = compiled_fn({q.inner, rk, rv, scale_arr, cap_arr, m});
        return std::make_unique<MlxArray>(std::move(result[0]));
    } else {
        static auto compiled_fn = get_compiled_softcap_sdpa_nomask();
        auto result = compiled_fn({q.inner, rk, rv, scale_arr, cap_arr});
        return std::make_unique<MlxArray>(std::move(result[0]));
    }
}

// Compiled GELU MLP forward: down_proj(gelu(gate_proj(x)) * up_proj(x))
// Compiles entire MLP into a single cached graph
// Used by: legacy/tests for precise GELU-gated models
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_qgelu_mlp() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            // inputs: x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b
            const auto& x = inputs[0];
            const auto& gate_w = inputs[1];
            const auto& gate_s = inputs[2];
            const auto& gate_b = inputs[3];
            const auto& up_w = inputs[4];
            const auto& up_s = inputs[5];
            const auto& up_b = inputs[6];
            const auto& down_w = inputs[7];
            const auto& down_s = inputs[8];
            const auto& down_b = inputs[9];

            int group_size = 64;
            int bits = 4;
            // gate = quantized_matmul(x, gate_w, gate_s, gate_b)
            auto gate = mlx::core::quantized_matmul(x, gate_w, gate_s, gate_b, true, group_size, bits);

            // up = quantized_matmul(x, up_w, up_s, up_b)
            auto up = mlx::core::quantized_matmul(x, up_w, up_s, up_b, true, group_size, bits);

            // GELU(gate): gate * 0.5 * (1 + erf(gate / sqrt(2)))
            auto sqrt2 = array(std::sqrt(2.0f));
            auto half = array(0.5f);
            auto one = array(1.0f);
            auto erf_val = mlx::core::erf(mlx::core::divide(gate, sqrt2));
            auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
            auto gelu_gate = mlx::core::multiply(gate, scale);

            // activated = gelu(gate) * up
            auto activated = mlx::core::multiply(gelu_gate, up);

            // down = quantized_matmul(activated, down_w, down_s, down_b)
            auto down = mlx::core::quantized_matmul(activated, down_w, down_s, down_b, true, group_size, bits);

            return {down};
        };
        return mlx::core::compile(fn, true);  // shapeless=true
    }
}

std::unique_ptr<MlxArray> compiled_gelu_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray* gate_biases,
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray* up_biases,
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray* down_biases,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    std::string mode_str(mode.data(), mode.size());

    // Compiled path only supports affine mode with group_size=64, bits=4, and biases present.
    // The compiled lambda hardcodes these quantization parameters for graph caching.
    if (mode_str == "affine" && group_size == 64 && bits == 4
        && gate_biases && up_biases && down_biases) {
        static auto compiled_fn = get_compiled_qgelu_mlp();

        auto result = compiled_fn({
            x.inner,
            gate_proj.inner, gate_scales.inner, gate_biases->inner,
            up_proj.inner, up_scales.inner, up_biases->inner,
            down_proj.inner, down_scales.inner, down_biases->inner
        });

        return std::make_unique<MlxArray>(std::move(result[0]));
    }

    // Non-compiled fallback for mxfp4/nvfp4/mxfp8 or missing biases
    std::optional<array> gb_opt = gate_biases ? std::optional(gate_biases->inner) : std::nullopt;
    std::optional<array> ub_opt = up_biases ? std::optional(up_biases->inner) : std::nullopt;
    std::optional<array> db_opt = down_biases ? std::optional(down_biases->inner) : std::nullopt;

    auto gate = mlx::core::quantized_matmul(
        x.inner, gate_proj.inner, gate_scales.inner, gb_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);
    auto up = mlx::core::quantized_matmul(
        x.inner, up_proj.inner, up_scales.inner, ub_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    // GELU(gate) * up
    auto sqrt2 = array(std::sqrt(2.0f));
    auto half = array(0.5f);
    auto one = array(1.0f);
    auto erf_val = mlx::core::erf(mlx::core::divide(gate, sqrt2));
    auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
    auto gelu_gate = mlx::core::multiply(gate, scale);
    auto activated = mlx::core::multiply(gelu_gate, up);

    auto down = mlx::core::quantized_matmul(
        activated, down_proj.inner, down_scales.inner, db_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    return std::make_unique<MlxArray>(std::move(down));
}

// Compiled quantized GeGLU MLP using Python MLX's tanh-approx GELU.
// Used by: Gemma, Gemma2, Gemma3, Gemma4
namespace {
    // Compiled affine-quantized GeGLU MLP, keyed on (group_size, bits, mode).
    //
    // The earlier version hard-coded group_size=64/bits=4, so any other
    // affine quantization (notably the group_size=64/bits=8 MLP weights that
    // Gemma 4 mixed-precision checkpoints carry) fell through to the
    // op-at-a-time fallback in `compiled_gelu_approx_mlp_forward`. That fallback
    // runs `gelu_tanh_approx` outside any compile window, emitting ~7 Multiply +
    // ~4 Broadcast + 2 Add + 1 Tanh per layer. On the CPU-bound GB10 decode path
    // (issue #680) that is the single largest source of the gemma-4 per-step MLX
    // primitive count. Compiling the whole gate/up/gelu/down chain collapses the
    // element-wise activation into one fused `Compiled` primitive for any affine
    // group_size/bits, so 8-bit MLPs get the same fusion 4-bit ones already had.
    //
    // The cache is keyed on (group_size, bits, mode) so each quantization
    // contributes at most one compiled graph. Leaked on purpose, like the other
    // compile caches in this file, so it outlives MLX's `thread_local`
    // CompilerCache at shutdown (mlx/compile.cpp: each thread re-traces once).
    static std::function<std::vector<array>(const std::vector<array>&)>&
    get_compiled_qgelu_approx_mlp(int group_size, int bits, const std::string& mode) {
        struct Key { int group_size; int bits; std::string mode; };
        struct KeyHash {
            size_t operator()(const Key& k) const noexcept {
                size_t h = std::hash<int>{}(k.group_size);
                h ^= std::hash<int>{}(k.bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<std::string>{}(k.mode) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };
        struct KeyEq {
            bool operator()(const Key& a, const Key& b) const noexcept {
                return a.group_size == b.group_size && a.bits == b.bits && a.mode == b.mode;
            }
        };

        static std::mutex& mu = *new std::mutex();
        static std::unordered_map<
            Key,
            std::function<std::vector<array>(const std::vector<array>&)>,
            KeyHash, KeyEq>& cache =
            *new std::unordered_map<
                Key,
                std::function<std::vector<array>(const std::vector<array>&)>,
                KeyHash, KeyEq>();

        std::lock_guard<std::mutex> lock(mu);
        Key key{group_size, bits, mode};
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        auto fn = [group_size, bits, mode]
            (const std::vector<array>& inputs) -> std::vector<array> {
            // inputs: x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b
            const auto& x = inputs[0];
            const auto& gate_w = inputs[1];
            const auto& gate_s = inputs[2];
            const auto& gate_b = inputs[3];
            const auto& up_w = inputs[4];
            const auto& up_s = inputs[5];
            const auto& up_b = inputs[6];
            const auto& down_w = inputs[7];
            const auto& down_s = inputs[8];
            const auto& down_b = inputs[9];

            auto gate = mlx::core::quantized_matmul(
                x, gate_w, gate_s, std::optional<array>(gate_b), true,
                std::optional<int>(group_size), std::optional<int>(bits), mode);
            auto up = mlx::core::quantized_matmul(
                x, up_w, up_s, std::optional<array>(up_b), true,
                std::optional<int>(group_size), std::optional<int>(bits), mode);

            auto activated = mlx::core::multiply(gelu_tanh_approx(gate), up);

            auto down = mlx::core::quantized_matmul(
                activated, down_w, down_s, std::optional<array>(down_b), true,
                std::optional<int>(group_size), std::optional<int>(bits), mode);

            return {down};
        };

        auto [iter, _] = cache.emplace(key, mlx::core::compile(fn, /*shapeless=*/true));
        return iter->second;
    }
}

std::unique_ptr<MlxArray> compiled_gelu_approx_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray* gate_biases,
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray* up_biases,
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray* down_biases,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    std::string mode_str(mode.data(), mode.size());

    // Compile the whole gate/up/gelu/down chain for affine quantization that
    // carries biases (all three projections), collapsing `gelu_tanh_approx`'s
    // ~14 element-wise ops per layer into one fused `Compiled` primitive.
    //
    // Two regimes (issue #680):
    //   * group_size=64/bits=4: compiled on every shape, exactly as before this
    //     change. Prefill and decode were already validated on this path.
    //   * any other affine group_size/bits (notably the group_size=64/bits=8
    //     MLP weights in Gemma 4 mixed-precision checkpoints): compile only the
    //     single-token DECODE call (`l == 1`). On GB10 the compiled 8-bit
    //     *prefill* GEMM measured ~8-9% slower and ~0.2-0.7 GB higher peak than
    //     the op-at-a-time path (the fused shapeless graph forces a
    //     decode-oriented qmm kernel onto the large prefill matmul), so prefill
    //     for these checkpoints stays on the op-at-a-time fallback below. Decode
    //     is single-token and the fused transient is a few KB, so it neither
    //     regresses peak nor prefill while removing the per-step element-wise ops.
    //
    // GB10 decode is weight-bandwidth-bound (the GPU is ~93-95% busy streaming
    // the 8-bit MLP weights), so this primitive-count cut is throughput-neutral
    // there; it still removes real CPU dispatch and helps op-count-bound backends.
    // `MLXCEL_COMPILED_QGELU_MLP=0` forces the fallback (A/B + escape hatch).
    static const bool compiled_qgelu_enabled = []() {
        const char* v = std::getenv("MLXCEL_COMPILED_QGELU_MLP");
        if (!v) {
            return true;  // default ON
        }
        std::string s(v);
        return !(s == "0" || s == "false" || s == "off" || s == "no");
    }();

    // `l == 1` single-token decode: x is [B, 1, hidden].
    const auto& x_shape = x.inner.shape();
    const bool is_single_token =
        x_shape.size() >= 2 && x_shape[x_shape.size() - 2] == 1;
    // The pre-#680 always-compiled case, preserved bit-for-bit.
    const bool legacy_compiled_shape = (group_size == 64 && bits == 4);

    if (compiled_qgelu_enabled && mode_str == "affine"
        && gate_biases && up_biases && down_biases
        && (legacy_compiled_shape || is_single_token)) {
        auto& compiled_fn = get_compiled_qgelu_approx_mlp(group_size, bits, mode_str);

        auto result = compiled_fn({
            x.inner,
            gate_proj.inner, gate_scales.inner, gate_biases->inner,
            up_proj.inner, up_scales.inner, up_biases->inner,
            down_proj.inner, down_scales.inner, down_biases->inner
        });

        return std::make_unique<MlxArray>(std::move(result[0]));
    }

    std::optional<array> gb_opt = gate_biases ? std::optional(gate_biases->inner) : std::nullopt;
    std::optional<array> ub_opt = up_biases ? std::optional(up_biases->inner) : std::nullopt;
    std::optional<array> db_opt = down_biases ? std::optional(down_biases->inner) : std::nullopt;

    auto gate = mlx::core::quantized_matmul(
        x.inner, gate_proj.inner, gate_scales.inner, gb_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);
    auto up = mlx::core::quantized_matmul(
        x.inner, up_proj.inner, up_scales.inner, ub_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    auto activated = mlx::core::multiply(gelu_tanh_approx(gate), up);

    auto down = mlx::core::quantized_matmul(
        activated, down_proj.inner, down_scales.inner, db_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    return std::make_unique<MlxArray>(std::move(down));
}

// Compiled GeGLU MLP with per-projection NVFP4 global-scale sidecars folded in
// (issue #698). Used by: Gemma 4 dense MLP loaded from ModelOpt NVFP4 triplets.
//
// The direct ModelOpt transcode (issue #693/#697) keeps `weight_scale_2` as a
// per-linear f32 `global_scale` sidecar. `UnifiedLinear::forward` applies it
// eagerly as `astype(multiply(qmm_out, s), qmm_out.dtype())` after each qmm. On
// the gemma-4 decode path (CUDA graphs disabled, #688) routing each projection
// through its own FFI call adds three extra element-wise dispatches per layer,
// so a sidecar-carrying MLP used to bypass the fused kernel entirely.
//
// This function folds those exact multiplies back into the fused graph at the
// mathematically correct points: the GeGLU is nonlinear, so the gate scale is
// applied to the gate product BEFORE `gelu_tanh_approx`; the up scale is a
// linear factor on the up product; the down scale multiplies the fused output.
// Each fold reproduces `apply_global_scale`'s op sequence byte-for-byte
// (`multiply` by the f32 scalar, then `astype` back to the activation dtype),
// so the fused result is bit-identical to the op-at-a-time sidecar path. A null
// scale pointer means no multiply for that projection (mixed sidecar sets stay
// exact for the projections that carry no scale).
namespace {
    // Scaled GeGLU MLP compiled cache, keyed on
    // (group_size, bits, mode, has_gate, has_up, has_down, shapeless). The
    // presence flags are part of the key because they change the emitted graph
    // structure (a missing scale emits no multiply/astype), so each sidecar
    // pattern compiles to its own graph. The shapeless bit separates the
    // decode-oriented graph from the shape-specific native-NVFP4 prefill graph
    // added for issue #705: a shapeless graph can force the single-token qmm
    // kernel onto large prefill matmuls, while a shape-specific graph lets MLX
    // choose the normal prefill qmm kernel and still fuse the sidecar scales and
    // GeGLU activation. Leaked on purpose like the sibling compile caches so it
    // outlives MLX's thread_local CompilerCache at shutdown.
    static std::function<std::vector<array>(const std::vector<array>&)>&
    get_compiled_qgelu_scaled_mlp(
        int group_size, int bits, const std::string& mode,
        bool has_gate, bool has_up, bool has_down, bool shapeless) {
        struct Key {
            int group_size; int bits; std::string mode;
            bool has_gate; bool has_up; bool has_down;
            bool shapeless;
        };
        struct KeyHash {
            size_t operator()(const Key& k) const noexcept {
                size_t h = std::hash<int>{}(k.group_size);
                h ^= std::hash<int>{}(k.bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<std::string>{}(k.mode) + 0x9e3779b9 + (h << 6) + (h >> 2);
                size_t flags = (size_t(k.has_gate) << 2)
                    | (size_t(k.has_up) << 1) | size_t(k.has_down);
                h ^= std::hash<size_t>{}(flags) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<bool>{}(k.shapeless) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };
        struct KeyEq {
            bool operator()(const Key& a, const Key& b) const noexcept {
                return a.group_size == b.group_size && a.bits == b.bits
                    && a.mode == b.mode && a.has_gate == b.has_gate
                    && a.has_up == b.has_up && a.has_down == b.has_down
                    && a.shapeless == b.shapeless;
            }
        };

        static std::mutex& mu = *new std::mutex();
        static std::unordered_map<
            Key,
            std::function<std::vector<array>(const std::vector<array>&)>,
            KeyHash, KeyEq>& cache =
            *new std::unordered_map<
                Key,
                std::function<std::vector<array>(const std::vector<array>&)>,
                KeyHash, KeyEq>();

        std::lock_guard<std::mutex> lock(mu);
        Key key{group_size, bits, mode, has_gate, has_up, has_down, shapeless};
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        auto fn = [group_size, bits, mode, has_gate, has_up, has_down]
            (const std::vector<array>& inputs) -> std::vector<array> {
            // inputs: x, gate_w, gate_s, up_w, up_s, down_w, down_s,
            //         [gate_gs?], [up_gs?], [down_gs?]
            // The optional global scales are appended in gate/up/down order and
            // consumed in the same order, so the running index only advances
            // when the matching presence flag is set.
            size_t idx = 0;
            const auto& x = inputs[idx++];
            const auto& gate_w = inputs[idx++];
            const auto& gate_s = inputs[idx++];
            const auto& up_w = inputs[idx++];
            const auto& up_s = inputs[idx++];
            const auto& down_w = inputs[idx++];
            const auto& down_s = inputs[idx++];

            auto gate = mlx::core::quantized_matmul(
                x, gate_w, gate_s, std::nullopt, true,
                std::optional<int>(group_size), std::optional<int>(bits), mode);
            if (has_gate) {
                auto dt = gate.dtype();
                gate = mlx::core::astype(
                    mlx::core::multiply(gate, inputs[idx++]), dt);
            }

            auto up = mlx::core::quantized_matmul(
                x, up_w, up_s, std::nullopt, true,
                std::optional<int>(group_size), std::optional<int>(bits), mode);
            if (has_up) {
                auto dt = up.dtype();
                up = mlx::core::astype(
                    mlx::core::multiply(up, inputs[idx++]), dt);
            }

            auto activated = mlx::core::multiply(gelu_tanh_approx(gate), up);

            auto down = mlx::core::quantized_matmul(
                activated, down_w, down_s, std::nullopt, true,
                std::optional<int>(group_size), std::optional<int>(bits), mode);
            if (has_down) {
                auto dt = down.dtype();
                down = mlx::core::astype(
                    mlx::core::multiply(down, inputs[idx++]), dt);
            }

            return {down};
        };

        auto [iter, _] = cache.emplace(key, mlx::core::compile(fn, /*shapeless=*/shapeless));
        return iter->second;
    }

    // Apply the sidecar to an eager (non-compiled) projection output, mirroring
    // `QuantizedWeight::apply_global_scale` exactly.
    static array apply_global_scale_eager(array out, const MlxArray* scale) {
        if (!scale) {
            return out;
        }
        auto dt = out.dtype();
        return mlx::core::astype(mlx::core::multiply(out, scale->inner), dt);
    }
}

std::unique_ptr<MlxArray> compiled_gelu_approx_mlp_forward_global_scale(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray* gate_global_scale,
    const MlxArray* up_global_scale,
    const MlxArray* down_global_scale,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    std::string mode_str(mode.data(), mode.size());
    const bool has_gate = gate_global_scale != nullptr;
    const bool has_up = up_global_scale != nullptr;
    const bool has_down = down_global_scale != nullptr;

    // Same escape hatch as the unscaled fused path: `MLXCEL_COMPILED_QGELU_MLP=0`
    // forces the eager fold. (The Rust-side `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE`
    // kill switch bypasses this function entirely and restores op-at-a-time.)
    static const bool compiled_qgelu_enabled = []() {
        const char* v = std::getenv("MLXCEL_COMPILED_QGELU_MLP");
        if (!v) {
            return true;
        }
        std::string s(v);
        return !(s == "0" || s == "false" || s == "off" || s == "no");
    }();

    // Compile gate:
    //   * single-token decode uses the existing shapeless fused graph.
    //   * native NVFP4 prefill (group_size=16/bits=4/mode=nvfp4) uses a
    //     shape-specific fused graph. This is the issue #705 prefill recovery:
    //     it keeps the sidecar folds and GeGLU activation in one graph without
    //     forcing the decode-oriented qmm kernel that regressed #680.
    //   * all other multi-token sidecar cases keep the eager fold fallback.
    const auto& x_shape = x.inner.shape();
    const bool is_single_token =
        x_shape.size() >= 2 && x_shape[x_shape.size() - 2] == 1;
    const bool legacy_compiled_shape = (group_size == 64 && bits == 4);
    const bool native_nvfp4_prefill =
        !is_single_token && mode_str == "nvfp4" && group_size == 16 && bits == 4;

    if (compiled_qgelu_enabled
        && (legacy_compiled_shape || is_single_token || native_nvfp4_prefill)) {
        const bool shapeless = !native_nvfp4_prefill;
        auto& compiled_fn = get_compiled_qgelu_scaled_mlp(
            group_size, bits, mode_str, has_gate, has_up, has_down, shapeless);
        std::vector<array> inputs = {
            x.inner,
            gate_proj.inner, gate_scales.inner,
            up_proj.inner, up_scales.inner,
            down_proj.inner, down_scales.inner
        };
        if (has_gate) inputs.push_back(gate_global_scale->inner);
        if (has_up) inputs.push_back(up_global_scale->inner);
        if (has_down) inputs.push_back(down_global_scale->inner);
        auto result = compiled_fn(inputs);
        return std::make_unique<MlxArray>(std::move(result[0]));
    }

    // Eager fold (prefill / compile disabled). NVFP4 carries no quant biases,
    // so the qmm bias operand is always null here.
    auto gate = mlx::core::quantized_matmul(
        x.inner, gate_proj.inner, gate_scales.inner, std::nullopt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);
    gate = apply_global_scale_eager(std::move(gate), gate_global_scale);

    auto up = mlx::core::quantized_matmul(
        x.inner, up_proj.inner, up_scales.inner, std::nullopt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);
    up = apply_global_scale_eager(std::move(up), up_global_scale);

    auto activated = mlx::core::multiply(gelu_tanh_approx(gate), up);

    auto down = mlx::core::quantized_matmul(
        activated, down_proj.inner, down_scales.inner, std::nullopt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);
    down = apply_global_scale_eager(std::move(down), down_global_scale);

    return std::make_unique<MlxArray>(std::move(down));
}

// Compiled GeGLU SwitchGLU MLP forward: down(gelu(gate_gqm(x)) * up_gqm(x))
// where gate/up/down each use `mx::core::gather_qmm` for per-expert
// routing. Mirrors the dense `compiled_qgelu_mlp` pattern but with
// `gather_qmm` instead of `quantized_matmul`, collapsing three
// `gather_qmm` calls + the GeGLU activation + the final `gather_qmm`
// into one compile window so MLX can schedule gate/up in parallel
// and fuse the intermediate element-wise ops.
//
// Only covers the decode-friendly no-sort path (sorted_indices=false).
// The sort path (prefill when n_tokens * top_k >= 64) stays on the
// separate-ops path inside `SwitchGeGLU::forward`.
//
// Used by: Gemma 4 26B-a4b SwitchGeGLU experts
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_switch_qgeglu() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            // inputs: x, gate_w, gate_s, gate_b, up_w, up_s, up_b,
            //         down_w, down_s, down_b, rhs_indices
            const auto& x = inputs[0];
            const auto& gate_w = inputs[1];
            const auto& gate_s = inputs[2];
            const auto& gate_b = inputs[3];
            const auto& up_w = inputs[4];
            const auto& up_s = inputs[5];
            const auto& up_b = inputs[6];
            const auto& down_w = inputs[7];
            const auto& down_s = inputs[8];
            const auto& down_b = inputs[9];
            const auto& rhs_indices = inputs[10];

            int group_size = 64;
            int bits = 4;
            bool transpose = true;
            bool sorted_indices = false;

            auto gate = mlx::core::gather_qmm(
                x, gate_w, gate_s, std::optional<array>(gate_b),
                std::nullopt, std::optional<array>(rhs_indices),
                transpose, group_size, bits, "affine", sorted_indices);

            auto up = mlx::core::gather_qmm(
                x, up_w, up_s, std::optional<array>(up_b),
                std::nullopt, std::optional<array>(rhs_indices),
                transpose, group_size, bits, "affine", sorted_indices);

            // GeGLU: Python mlx-lm uses nn.gelu_approx(gate) * up.
            auto activated = mlx::core::multiply(gelu_tanh_approx(gate), up);

            auto down = mlx::core::gather_qmm(
                activated, down_w, down_s, std::optional<array>(down_b),
                std::nullopt, std::optional<array>(rhs_indices),
                transpose, group_size, bits, "affine", sorted_indices);

            return {down};
        };
        // shapeless=false: `gather_qmm`'s `Primitive::output_shapes`
        // cannot infer result shape without concrete `rhs_indices`
        // dims, so we key the compile on concrete input shapes. Decode
        // passes through the same shapes on every layer, so the cache
        // still resolves to a single compiled graph after the first
        // call.
        return mlx::core::compile(fn, false);
    }
}

std::unique_ptr<MlxArray> compiled_switch_qgeglu_forward(
    const MlxArray& x,
    const MlxArray& gate_w,
    const MlxArray& gate_s,
    const MlxArray* gate_b,
    const MlxArray& up_w,
    const MlxArray& up_s,
    const MlxArray* up_b,
    const MlxArray& down_w,
    const MlxArray& down_s,
    const MlxArray* down_b,
    const MlxArray& rhs_indices,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    std::string mode_str(mode.data(), mode.size());

    // Fused compile only covers affine mode with group_size=64, bits=4,
    // biases present, sorted_indices=false — the decode-hot case for
    // Gemma 4 26B-a4b experts. Any other combination falls back to
    // three separate `gather_qmm` calls at the Rust layer.
    if (mode_str == "affine" && group_size == 64 && bits == 4
        && gate_b && up_b && down_b) {
        static auto compiled_fn = get_compiled_switch_qgeglu();
        auto result = compiled_fn({
            x.inner,
            gate_w.inner, gate_s.inner, gate_b->inner,
            up_w.inner, up_s.inner, up_b->inner,
            down_w.inner, down_s.inner, down_b->inner,
            rhs_indices.inner
        });
        return std::make_unique<MlxArray>(std::move(result[0]));
    }

    // Non-compiled fallback: three separate gather_qmm calls + tanh-approx
    // GeGLU. Same math as the compiled path, just without the compile window
    // (for non-affine quantization modes or missing biases).
    std::optional<array> gb_opt = gate_b ? std::optional(gate_b->inner) : std::nullopt;
    std::optional<array> ub_opt = up_b ? std::optional(up_b->inner) : std::nullopt;
    std::optional<array> db_opt = down_b ? std::optional(down_b->inner) : std::nullopt;
    std::optional<array> rhs_opt = std::optional(rhs_indices.inner);

    auto gate = mlx::core::gather_qmm(
        x.inner, gate_w.inner, gate_s.inner, gb_opt,
        std::nullopt, rhs_opt, true,
        std::optional<int>(group_size), std::optional<int>(bits),
        mode_str, false);
    auto up = mlx::core::gather_qmm(
        x.inner, up_w.inner, up_s.inner, ub_opt,
        std::nullopt, rhs_opt, true,
        std::optional<int>(group_size), std::optional<int>(bits),
        mode_str, false);

    auto activated = mlx::core::multiply(gelu_tanh_approx(gate), up);

    auto down = mlx::core::gather_qmm(
        activated, down_w.inner, down_s.inner, db_opt,
        std::nullopt, rhs_opt, true,
        std::optional<int>(group_size), std::optional<int>(bits),
        mode_str, false);

    return std::make_unique<MlxArray>(std::move(down));
}

// SwiGLU MLP forward for non-quantized (FP16/BF16) weights:
//   down_proj(silu(gate_proj(x)) * up_proj(x))
//
// Matmul operations run outside the compile boundary because
// mlx::core::compile with matmul+transpose can produce incorrect results
// when the compiled graph is reused across layers with different weights.
// The SwiGLU activation (silu(gate) * up) uses the compiled swiglu kernel
// which correctly fuses element-wise ops only.
//
// Used by: Llama, Qwen2, Qwen3, and all other SwiGLU FP models
std::unique_ptr<MlxArray> compiled_swiglu_mlp_forward_fp16(
    const MlxArray& x,
    const MlxArray& gate_weight,
    const MlxArray& up_weight,
    const MlxArray& down_weight,
    const MlxArray* gate_bias,
    const MlxArray* up_bias,
    const MlxArray* down_bias
) {
    // gate_proj(x) = x @ gate_w.T [+ bias]
    auto gate_t = mlx::core::transpose(gate_weight.inner);
    auto gate = mlx::core::matmul(x.inner, gate_t);
    if (gate_bias) gate = mlx::core::add(gate, gate_bias->inner);

    // up_proj(x) = x @ up_w.T [+ bias]
    auto up_t = mlx::core::transpose(up_weight.inner);
    auto up = mlx::core::matmul(x.inner, up_t);
    if (up_bias) up = mlx::core::add(up, up_bias->inner);

    // SwiGLU activation: silu(gate) * up — compiled element-wise fusion
    static auto compiled_act = get_compiled_swiglu();
    auto activated = compiled_act({gate, up});

    // down_proj(activated) = activated @ down_w.T [+ bias]
    auto down_t = mlx::core::transpose(down_weight.inner);
    auto down = mlx::core::matmul(activated[0], down_t);
    if (down_bias) down = mlx::core::add(down, down_bias->inner);

    return std::make_unique<MlxArray>(std::move(down));
}

// GELU MLP forward for non-quantized (FP16/BF16) weights:
//   down_proj(gelu(gate_proj(x)) * up_proj(x))
//
// Matmul operations run outside the compile boundary because
// mlx::core::compile with matmul+transpose can produce incorrect results
// when the compiled graph is reused across layers with different weights.
// The GeGLU activation (gelu(gate) * up) uses the compiled geglu kernel
// which correctly fuses element-wise ops only.
//
// Used by: Gemma, Gemma4 and other GELU-gated FP models
std::unique_ptr<MlxArray> compiled_gelu_mlp_forward_fp16(
    const MlxArray& x,
    const MlxArray& gate_weight,
    const MlxArray& up_weight,
    const MlxArray& down_weight,
    const MlxArray* gate_bias,
    const MlxArray* up_bias,
    const MlxArray* down_bias
) {
    // gate_proj(x) = x @ gate_w.T [+ bias]
    auto gate_t = mlx::core::transpose(gate_weight.inner);
    auto gate = mlx::core::matmul(x.inner, gate_t);
    if (gate_bias) gate = mlx::core::add(gate, gate_bias->inner);

    // up_proj(x) = x @ up_w.T [+ bias]
    auto up_t = mlx::core::transpose(up_weight.inner);
    auto up = mlx::core::matmul(x.inner, up_t);
    if (up_bias) up = mlx::core::add(up, up_bias->inner);

    // GeGLU activation: gelu(gate) * up — compiled element-wise fusion
    static auto compiled_act = get_compiled_geglu();
    auto activated = compiled_act({gate, up});

    // down_proj(activated) = activated @ down_w.T [+ bias]
    auto down_t = mlx::core::transpose(down_weight.inner);
    auto down = mlx::core::matmul(activated[0], down_t);
    if (down_bias) down = mlx::core::add(down, down_bias->inner);

    return std::make_unique<MlxArray>(std::move(down));
}

// Gemma3n dense MLP forward for non-quantized bf16 language MLP weights:
//   cast input to bf16 -> gate/up -> gelu_approx or gelu_topk -> down -> bf16.
//
// Matmul operations intentionally stay outside mx::compile for the same reason
// as compiled_swiglu_mlp_forward_fp16 / compiled_gelu_mlp_forward_fp16: compiled
// matmul+transpose graphs can reuse the wrong constants across layers. The
// element-wise activation still uses the cached compiled kernels, while the
// whole MLP is built through one C++ bridge call instead of four Rust calls.
std::unique_ptr<MlxArray> gemma3n_mlp_forward(
    const MlxArray& x,
    const MlxArray& gate_weight,
    const MlxArray& up_weight,
    const MlxArray& down_weight,
    const MlxArray* gate_bias,
    const MlxArray* up_bias,
    const MlxArray* down_bias,
    float activation_sparsity,
    float std_multiplier
) {
    auto x_mlp = x.inner.dtype() == mlx::core::bfloat16
        ? x.inner
        : mlx::core::astype(x.inner, mlx::core::bfloat16);

    auto gate_t = mlx::core::transpose(gate_weight.inner);
    auto gate = mlx::core::matmul(x_mlp, gate_t);
    if (gate_bias) gate = mlx::core::add(gate, gate_bias->inner);

    auto up_t = mlx::core::transpose(up_weight.inner);
    auto up = mlx::core::matmul(x_mlp, up_t);
    if (up_bias) up = mlx::core::add(up, up_bias->inner);

    auto hidden = [&]() {
        if (activation_sparsity > 0.0f) {
            static auto compiled_topk = get_compiled_gelu_topk();
            auto mult = array(std_multiplier);
            auto activated = compiled_topk({gate, mult});
            return mlx::core::multiply(activated[0], up);
        }
        static auto compiled_geglu_approx = get_compiled_geglu_approx();
        auto activated = compiled_geglu_approx({gate, up});
        return activated[0];
    }();

    auto down_t = mlx::core::transpose(down_weight.inner);
    auto down = mlx::core::matmul(hidden, down_t);
    if (down_bias) down = mlx::core::add(down, down_bias->inner);

    return std::make_unique<MlxArray>(mlx::core::astype(down, mlx::core::bfloat16));
}

std::unique_ptr<MlxArray> transformer_layer_forward(
    const MlxArray& x,
    const MlxArray& attn_norm_weight,
    const MlxArray& q_proj,
    const MlxArray& k_proj,
    const MlxArray& v_proj,
    const MlxArray& o_proj,
    const MlxArray& ffn_norm_weight,
    const MlxArray& gate_proj,
    const MlxArray& up_proj,
    const MlxArray& down_proj,
    const MlxArray* kv_cache_k,
    const MlxArray* kv_cache_v,
    int32_t n_heads,
    int32_t n_kv_heads,
    int32_t head_dim,
    float rope_theta,
    int32_t rope_offset,
    float norm_eps
) {
    auto batch_size = x.inner.shape()[0];
    auto seq_len = x.inner.shape()[1];

    // Pre-attention norm (RMS norm)
    auto x_sq = mlx::core::square(x.inner);
    auto mean_sq = mlx::core::mean(x_sq, -1, true);
    auto norm = mlx::core::rsqrt(mlx::core::add(mean_sq, array(norm_eps)));
    auto h = mlx::core::multiply(mlx::core::multiply(x.inner, norm), attn_norm_weight.inner);

    // Q, K, V projections
    auto q = mlx::core::matmul(h, mlx::core::transpose(q_proj.inner));
    auto k = mlx::core::matmul(h, mlx::core::transpose(k_proj.inner));
    auto v = mlx::core::matmul(h, mlx::core::transpose(v_proj.inner));

    // Reshape for multi-head attention
    q = mlx::core::reshape(q, {batch_size, seq_len, n_heads, head_dim});
    k = mlx::core::reshape(k, {batch_size, seq_len, n_kv_heads, head_dim});
    v = mlx::core::reshape(v, {batch_size, seq_len, n_kv_heads, head_dim});

    // Transpose to [batch, heads, seq, dim]
    q = mlx::core::transpose(q, {0, 2, 1, 3});
    k = mlx::core::transpose(k, {0, 2, 1, 3});
    v = mlx::core::transpose(v, {0, 2, 1, 3});

    // Apply RoPE (simplified - just use the existing values for now)
    // In production, would compute cos/sin and apply

    // Update KV cache if provided
    if (kv_cache_k != nullptr && kv_cache_v != nullptr) {
        k = mlx::core::concatenate({kv_cache_k->inner, k}, 2);
        v = mlx::core::concatenate({kv_cache_v->inner, v}, 2);
    }

    // Handle GQA (grouped query attention)
    if (n_kv_heads != n_heads) {
        int repeats = n_heads / n_kv_heads;
        k = mlx::core::repeat(k, repeats, 1);
        v = mlx::core::repeat(v, repeats, 1);
    }

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto k_t = mlx::core::transpose(k, {0, 1, 3, 2});
    auto scores = mlx::core::multiply(mlx::core::matmul(q, k_t), array(scale));

    // Causal mask
    auto kv_len = k.shape()[2];
    auto mask_val = array(-1e9f);
    // Create causal mask (lower triangular)
    std::vector<float> mask_data(seq_len * kv_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = rope_offset + i + 1; j < kv_len; ++j) {
            mask_data[i * kv_len + j] = -1e9f;
        }
    }
    auto mask = array(mask_data.data(), {1, 1, seq_len, kv_len});
    scores = mlx::core::add(scores, mask);

    auto weights = mlx::core::softmax(scores, -1);
    auto attn_out = mlx::core::matmul(weights, v);

    // Transpose back and reshape
    attn_out = mlx::core::transpose(attn_out, {0, 2, 1, 3});
    attn_out = mlx::core::reshape(attn_out, {batch_size, seq_len, n_heads * head_dim});

    // Output projection
    auto attn_result = mlx::core::matmul(attn_out, mlx::core::transpose(o_proj.inner));

    // Residual connection
    auto h2 = mlx::core::add(x.inner, attn_result);

    // Pre-FFN norm (RMS norm)
    auto h2_sq = mlx::core::square(h2);
    auto mean_sq2 = mlx::core::mean(h2_sq, -1, true);
    auto norm2 = mlx::core::rsqrt(mlx::core::add(mean_sq2, array(norm_eps)));
    auto h3 = mlx::core::multiply(mlx::core::multiply(h2, norm2), ffn_norm_weight.inner);

    // FFN (SwiGLU)
    auto gate = mlx::core::matmul(h3, mlx::core::transpose(gate_proj.inner));
    auto up = mlx::core::matmul(h3, mlx::core::transpose(up_proj.inner));
    // silu(x) = x * sigmoid(x)
    auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
    auto activated = mlx::core::multiply(silu_gate, up);
    auto ffn_out = mlx::core::matmul(activated, mlx::core::transpose(down_proj.inner));

    // Final residual
    return std::make_unique<MlxArray>(mlx::core::add(h2, ffn_out));
}

// Advanced indexing operations.
std::unique_ptr<MlxArray> take(const MlxArray& a, const MlxArray& indices, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::take(a.inner, indices.inner, axis));
}

std::unique_ptr<MlxArray> gather(
    const MlxArray& a,
    rust::Slice<const MlxArray* const> indices,
    rust::Slice<const int32_t> axes,
    rust::Slice<const int32_t> slice_sizes
) {
    std::vector<array> idx_vec;
    idx_vec.reserve(indices.size());
    for (const auto* idx : indices) {
        idx_vec.push_back(idx->inner);
    }

    std::vector<int> axes_vec(axes.begin(), axes.end());

    // Shape is SmallVector<int>, construct it from slice
    mlx::core::Shape shape_vec;
    for (auto s : slice_sizes) {
        shape_vec.push_back(s);
    }

    return std::make_unique<MlxArray>(mlx::core::gather(
        a.inner, idx_vec, axes_vec, shape_vec
    ));
}

std::unique_ptr<MlxArray> take_along_axis(const MlxArray& a, const MlxArray& indices, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::take_along_axis(a.inner, indices.inner, axis));
}

std::unique_ptr<MlxArray> put_along_axis(const MlxArray& a, const MlxArray& indices,
                                          const MlxArray& values, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::put_along_axis(a.inner, indices.inner, values.inner, axis));
}

std::unique_ptr<MlxArray> stack(rust::Slice<const MlxArray* const> arrays, int32_t axis) {
    std::vector<array> arrs;
    arrs.reserve(arrays.size());
    for (const auto* a : arrays) {
        arrs.push_back(a->inner);
    }
    return std::make_unique<MlxArray>(mlx::core::stack(arrs, axis));
}

std::unique_ptr<MlxArray> tile(const MlxArray& a, rust::Slice<const int32_t> reps) {
    std::vector<int> reps_vec(reps.begin(), reps.end());
    return std::make_unique<MlxArray>(mlx::core::tile(a.inner, reps_vec));
}

std::unique_ptr<MlxArray> repeat(const MlxArray& a, int32_t repeats, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::repeat(a.inner, repeats, axis));
}

std::unique_ptr<MlxArray> arange_f32(float start, float stop, float step) {
    return std::make_unique<MlxArray>(mlx::core::arange(start, stop, step));
}

std::unique_ptr<MlxArray> arange_i32(int32_t start, int32_t stop, int32_t step) {
    return std::make_unique<MlxArray>(mlx::core::arange(start, stop, step));
}

// Logical operations.
std::unique_ptr<MlxArray> logical_not(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::logical_not(a.inner));
}

std::unique_ptr<MlxArray> logical_and(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::logical_and(a.inner, b.inner));
}

std::unique_ptr<MlxArray> logical_or(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::logical_or(a.inner, b.inner));
}

std::unique_ptr<MlxArray> all_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::all(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> any_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::any(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> greater_equal(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::greater_equal(a.inner, b.inner));
}

std::unique_ptr<MlxArray> less_equal(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::less_equal(a.inner, b.inner));
}

std::unique_ptr<MlxArray> not_equal(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::not_equal(a.inner, b.inner));
}

// Activation functions.
std::unique_ptr<MlxArray> silu(const MlxArray& a) {
    // silu(x) = x * sigmoid(x)
    return std::make_unique<MlxArray>(mlx::core::multiply(a.inner, mlx::core::sigmoid(a.inner)));
}

std::unique_ptr<MlxArray> gelu(const MlxArray& a) {
    // gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    auto sqrt2 = array(std::sqrt(2.0f));
    auto half = array(0.5f);
    auto one = array(1.0f);
    auto erf_val = mlx::core::erf(mlx::core::divide(a.inner, sqrt2));
    auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
    return std::make_unique<MlxArray>(mlx::core::multiply(a.inner, scale));
}

std::unique_ptr<MlxArray> gelu_approx(const MlxArray& a) {
    // Use erf-based exact GELU instead of the original tanh approximation.
    //
    // The tanh approximation 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    // produced NaN for negative bf16 inputs in the SigLIP vision encoder MLP
    // because power(x, 3.0f) uses exp(3*log(x)) which is undefined for x<0.
    // Even with x*x*x the MLX lazy graph fuses operations that can overflow
    // bf16 intermediates within the large SigLIP computation graph (4096 patches).
    //
    // The erf-based GELU: x * 0.5 * (1 + erf(x / sqrt(2))) is numerically
    // stable for all inputs and matches Python nn.GELU(approx="precise")
    // output within floating-point tolerance.
    auto sqrt2 = array(std::sqrt(2.0f));
    auto half = array(0.5f);
    auto one = array(1.0f);
    auto erf_val = mlx::core::erf(mlx::core::divide(a.inner, sqrt2));
    auto scale = mlx::core::multiply(half, mlx::core::add(one, erf_val));
    return std::make_unique<MlxArray>(mlx::core::multiply(a.inner, scale));
}

std::unique_ptr<MlxArray> relu(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::maximum(a.inner, array(0.0f)));
}

std::unique_ptr<MlxArray> leaky_relu(const MlxArray& a, float negative_slope) {
    auto zero = array(0.0f);
    auto pos = mlx::core::maximum(a.inner, zero);
    auto neg = mlx::core::multiply(mlx::core::minimum(a.inner, zero), array(negative_slope));
    return std::make_unique<MlxArray>(mlx::core::add(pos, neg));
}

// Sorting and searching.
std::unique_ptr<MlxArray> argsort(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::argsort(a.inner, axis));
}

std::unique_ptr<MlxArray> argpartition(const MlxArray& a, int32_t kth, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::argpartition(a.inner, kth, axis));
}

std::unique_ptr<MlxArray> argmin(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::argmin(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> topk(const MlxArray& a, int32_t k, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::topk(a.inner, k, axis));
}

// Sort and partition
std::unique_ptr<MlxArray> sort(const MlxArray& a, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::sort(a.inner, axis));
}

std::unique_ptr<MlxArray> partition(const MlxArray& a, int32_t kth, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::partition(a.inner, kth, axis));
}

// Cumulative operations
std::unique_ptr<MlxArray> cummax(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::cummax(a.inner, axis, reverse, inclusive));
}

std::unique_ptr<MlxArray> cummin(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::cummin(a.inner, axis, reverse, inclusive));
}

std::unique_ptr<MlxArray> cumprod(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::cumprod(a.inner, axis, reverse, inclusive));
}

// Scatter operations
std::unique_ptr<MlxArray> scatter(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    // MLX scatter uses different signature - simplified version
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter(a.inner, idx_vec, updates.inner, {axis}));
}

std::unique_ptr<MlxArray> scatter_add(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter_add(a.inner, idx_vec, updates.inner, {axis}));
}

std::unique_ptr<MlxArray> scatter_max(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter_max(a.inner, idx_vec, updates.inner, {axis}));
}

std::unique_ptr<MlxArray> scatter_min(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter_min(a.inner, idx_vec, updates.inner, {axis}));
}

std::unique_ptr<MlxArray> scatter_prod(const MlxArray& a, const MlxArray& indices, const MlxArray& updates, int32_t axis) {
    std::vector<array> idx_vec = {indices.inner};
    return std::make_unique<MlxArray>(mlx::core::scatter_prod(a.inner, idx_vec, updates.inner, {axis}));
}

// Bitwise operations
std::unique_ptr<MlxArray> bitwise_and(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::bitwise_and(a.inner, b.inner));
}

std::unique_ptr<MlxArray> bitwise_or(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::bitwise_or(a.inner, b.inner));
}

std::unique_ptr<MlxArray> bitwise_xor(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::bitwise_xor(a.inner, b.inner));
}

std::unique_ptr<MlxArray> left_shift(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::left_shift(a.inner, b.inner));
}

std::unique_ptr<MlxArray> right_shift(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::right_shift(a.inner, b.inner));
}

// Linear algebra
std::unique_ptr<MlxArray> tensordot(const MlxArray& a, const MlxArray& b, int32_t axes) {
    return std::make_unique<MlxArray>(mlx::core::tensordot(a.inner, b.inner, axes));
}

std::unique_ptr<MlxArray> inner(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::inner(a.inner, b.inner));
}

std::unique_ptr<MlxArray> outer(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::outer(a.inner, b.inner));
}

std::unique_ptr<MlxArray> trace(const MlxArray& a, int32_t offset, int32_t axis1, int32_t axis2) {
    return std::make_unique<MlxArray>(mlx::core::trace(a.inner, offset, axis1, axis2));
}

// Roll (circular shift)
std::unique_ptr<MlxArray> roll(const MlxArray& a, int32_t shift, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::roll(a.inner, shift, axis));
}

// Nan handling
std::unique_ptr<MlxArray> nan_to_num(const MlxArray& a, float nan_val, float posinf_val, float neginf_val) {
    return std::make_unique<MlxArray>(mlx::core::nan_to_num(a.inner, nan_val, posinf_val, neginf_val));
}

// Stop gradient
std::unique_ptr<MlxArray> stop_gradient(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::stop_gradient(a.inner));
}

// 2D convolution
std::unique_ptr<MlxArray> conv2d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv2d(
        input.inner, weight.inner,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation_h, dilation_w},
        groups
    ));
}

// Same body as `conv2d`, but declared `-> Result` on the Rust side so cxx wraps
// the call in a try/catch and converts MLX's eager shape-mismatch exception (and
// any other throw) into a Rust `Err` instead of letting it abort the process.
// MLX validates conv shapes at graph-build time, so this catches the throw at op
// construction, not only at eval.
std::unique_ptr<MlxArray> try_conv2d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv2d(
        input.inner, weight.inner,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation_h, dilation_w},
        groups
    ));
}

std::unique_ptr<MlxArray> avg_pool2d(
    const MlxArray& input,
    int32_t kernel_h, int32_t kernel_w,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w
) {
    // Manual average pooling: input shape [B, H, W, C]
    // Use conv2d with a kernel of 1/(kH*kW) uniform weights for each channel
    // But MLX doesn't have nn::AvgPool2d in C++ ops, so we implement manually.
    //
    // For each output position, sum the kernel window and divide by kernel size.
    // We extract windows using as_strided or manual loops, but the simplest
    // approach for MLX is to use the unfold pattern.
    //
    // Actually, MLX Python's nn.AvgPool2d uses mlx.core ops internally.
    // The simplest C++ implementation: use a depthwise conv2d with uniform weights.
    auto& x = input.inner;
    auto shape = x.shape();
    int32_t channels = shape.back(); // [B, H, W, C] -> C is last dim

    // Create uniform kernel: shape [C, kH, kW, 1] for groups=C (depthwise)
    float kernel_val = 1.0f / (kernel_h * kernel_w);
    auto kernel = mlx::core::full({channels, kernel_h, kernel_w, 1}, kernel_val);

    // Depthwise conv2d with groups=C gives us the sum/count = average
    auto result = mlx::core::conv2d(
        x, kernel,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {1, 1},  // dilation
        channels  // groups = channels for depthwise
    );
    return std::make_unique<MlxArray>(std::move(result));
}

// MoE (Mixture of Experts) operations.
std::unique_ptr<MlxArray> gather_mm(
    const MlxArray& a,
    const MlxArray& b,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool sorted_indices
) {
    std::optional<array> lhs_opt = lhs_indices ? std::optional(lhs_indices->inner) : std::nullopt;
    std::optional<array> rhs_opt = rhs_indices ? std::optional(rhs_indices->inner) : std::nullopt;

    return std::make_unique<MlxArray>(mlx::core::gather_mm(
        a.inner, b.inner, lhs_opt, rhs_opt, sorted_indices
    ));
}

std::unique_ptr<MlxArray> gather_qmm(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray* lhs_indices,
    const MlxArray* rhs_indices,
    bool transpose,
    int32_t group_size,
    int32_t bits,
    bool sorted_indices,
    rust::Str mode
) {
    bool is_affine = (mode.size() == 6 && std::memcmp(mode.data(), "affine", 6) == 0);
    std::optional<array> lhs_opt = lhs_indices ? std::optional(lhs_indices->inner) : std::nullopt;
    std::optional<array> rhs_opt = rhs_indices ? std::optional(rhs_indices->inner) : std::nullopt;

    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
    if (is_affine) {
        // Omit mode parameter to use default "affine" path (avoids MLX dispatch overhead)
        return std::make_unique<MlxArray>(mlx::core::gather_qmm(
            x.inner, w.inner, scales.inner, biases_opt,
            lhs_opt, rhs_opt, transpose,
            std::optional<int>(group_size), std::optional<int>(bits),
            "affine", sorted_indices));
    }
    return std::make_unique<MlxArray>(mlx::core::gather_qmm(
        x.inner, w.inner, scales.inner, biases_opt,
        lhs_opt, rhs_opt, transpose,
        std::optional<int>(group_size), std::optional<int>(bits),
        std::string(mode.data(), mode.size()), sorted_indices));
}

std::unique_ptr<MlxArray> quantized_matmul(
    const MlxArray& x,
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    bool transpose,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    bool is_affine = (mode.size() == 6 && std::memcmp(mode.data(), "affine", 6) == 0);
    if (is_affine) {
        if (biases) {
            return std::make_unique<MlxArray>(mlx::core::quantized_matmul(
                x.inner, w.inner, scales.inner, biases->inner,
                transpose, group_size, bits));
        }
        return std::make_unique<MlxArray>(mlx::core::quantized_matmul(
            x.inner, w.inner, scales.inner, std::nullopt,
            transpose, group_size, bits));
    }
    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::quantized_matmul(
        x.inner, w.inner, scales.inner, biases_opt,
        transpose, std::optional<int>(group_size), std::optional<int>(bits),
        std::string(mode.data(), mode.size())));
}

std::unique_ptr<MlxArray> dequantize(
    const MlxArray& w,
    const MlxArray& scales,
    const MlxArray* biases,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
    std::string mode_str(mode.data(), mode.size());

    return std::make_unique<MlxArray>(mlx::core::dequantize(
        w.inner, scales.inner, biases_opt,
        std::optional<int>(group_size), std::optional<int>(bits),
        mode_str
    ));
}

// Embedding.
std::unique_ptr<MlxArray> embedding(const MlxArray& weight, const MlxArray& indices) {
    return std::make_unique<MlxArray>(mlx::core::take(weight.inner, indices.inner, 0));
}

std::unique_ptr<MlxArray> quantized_embedding(
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray& indices,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    // Save original indices shape for reshaping later
    auto idx_shape = indices.inner.shape();

    // Flatten indices: [batch, seq] -> [batch * seq]
    auto flat_indices = mlx::core::reshape(indices.inner, {-1});

    // Take rows from quantized weights using flattened indices
    auto w_indexed = mlx::core::take(weight.inner, flat_indices, 0);
    auto scales_indexed = mlx::core::take(scales.inner, flat_indices, 0);

    std::optional<mlx::core::array> biases_opt = std::nullopt;
    if (biases) {
        biases_opt = mlx::core::take(biases->inner, flat_indices, 0);
    }

    std::string mode_str(mode.data(), mode.size());

    // Dequantize with explicit optional wrapping
    auto result = mlx::core::dequantize(
        w_indexed,
        scales_indexed,
        biases_opt,
        std::optional<int>(group_size),
        std::optional<int>(bits),
        mode_str,
        std::nullopt,  // global_scale
        std::optional<Dtype>(scales.inner.dtype())
    );

    // Get hidden dimension from dequantized result
    auto hidden_dim = result.shape().back();

    // Reshape back to original batch/seq dims + hidden: [batch * seq, hidden] -> [batch, seq, hidden]
    Shape new_shape;
    for (size_t i = 0; i < idx_shape.size(); ++i) {
        new_shape.push_back(idx_shape[i]);
    }
    new_shape.push_back(hidden_dim);

    auto reshaped = mlx::core::reshape(result, new_shape);

    return std::make_unique<MlxArray>(reshaped);
}

// Fast operations (using MLX fast kernels).
std::unique_ptr<MlxArray> fast_rope(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    float base,
    float scale,
    int32_t offset
) {
    return std::make_unique<MlxArray>(mlx::core::fast::rope(
        x.inner, dims, traditional, base, scale, offset
    ));
}

std::unique_ptr<MlxArray> fast_rope_with_freqs(
    const MlxArray& x,
    int32_t dims,
    bool traditional,
    float scale,
    int32_t offset,
    const MlxArray& freqs
) {
    // When using custom freqs, pass nullopt for base (can't use both)
    return std::make_unique<MlxArray>(mlx::core::fast::rope(
        x.inner, dims, traditional, std::nullopt, scale, offset, freqs.inner
    ));
}

// Compiled ProportionalRoPE (Gemma 4 full-attention layers). Mirrors
// mlx-lm's full-head `mx.fast.rope(..., freqs=[finite..., inf...])`
// call inside one `mx::core::compile` graph. Covers the common case
// `last_dim == head_dim` (Gemma 4 Q/K inputs). Caller must
// short-circuit `rotated_dims <= 0` and the `last_dim > head_dim`
// tail case — those stay on the op-at-a-time path.
//
// Compile cache is keyed on `(head_dim, rotated_dims)`. Each model
// variant contributes at most one entry.
namespace {
    using CompiledFn =
        std::function<std::vector<array>(const std::vector<array>&)>;

    static CompiledFn& get_compiled_proportional_rope(int head_dim) {
        static std::mutex& mu = *new std::mutex();
        // Intentionally leaked: compiled function destruction calls back into
        // MLX's process-wide CompilerCache. This map is constructed before
        // that cache during first compile(), so normal static destruction would
        // tear it down after MLX and can crash at process exit.
        static std::unordered_map<int64_t, CompiledFn>& cache =
            *new std::unordered_map<int64_t, CompiledFn>();

        int64_t key = static_cast<int64_t>(head_dim);
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        auto fn = [head_dim]
            (const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x = inputs[0];       // rank-4 [B, n_heads, L, head_dim]
            const auto& freqs = inputs[1];
            const auto& offset = inputs[2];  // scalar int32 array

            auto out = mlx::core::fast::rope(
                x, head_dim, false, std::nullopt, 1.0f, offset, freqs);
            return {out};
        };

        auto [iter, _] = cache.emplace(
            key, mlx::core::compile(fn, /*shapeless=*/false));
        return iter->second;
    }
}

std::unique_ptr<MlxArray> compiled_proportional_rope(
    const MlxArray& x,
    const MlxArray& freqs,
    int32_t head_dim,
    int32_t rotated_dims,
    int32_t offset
) {
    // `offset` flows through as a scalar array so the same compiled
    // graph serves every decode step without recompilation.
    (void)rotated_dims;
    auto offset_arr = array(offset);
    auto& compiled_fn = get_compiled_proportional_rope(head_dim);
    auto result = compiled_fn({x.inner, freqs.inner, offset_arr});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled Gemma 4 Q-path with proportional RoPE. Folds
//   reshape → fast_rms_norm → transpose → full-head ProportionalRoPE
// into a single `mx::core::compile` graph, replacing four
// cxx-bridge calls with one fused subgraph. Used on Gemma 4 full-attention layers only
// (`rope_type == "proportional"`); sliding-attention layers stay
// on the op-at-a-time path since their standard `fast_rope` chain
// is already short.
namespace {
    static CompiledFn& get_compiled_q_path_proportional(
        int head_dim, int rotated_dims, int n_heads, float rms_eps
    ) {
        struct Key { int head_dim; int rotated_dims; int n_heads; float rms_eps; };
        struct KeyHash {
            size_t operator()(const Key& k) const noexcept {
                size_t h = std::hash<int>{}(k.head_dim);
                h ^= std::hash<int>{}(k.rotated_dims) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<int>{}(k.n_heads) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<float>{}(k.rms_eps) + 0x9e3779b9 + (h << 6) + (h >> 2);
                return h;
            }
        };
        struct KeyEq {
            bool operator()(const Key& a, const Key& b) const noexcept {
                return a.head_dim == b.head_dim && a.rotated_dims == b.rotated_dims
                    && a.n_heads == b.n_heads && a.rms_eps == b.rms_eps;
            }
        };

        static std::mutex& mu = *new std::mutex();
        // Intentionally leaked: compiled function destruction calls back into
        // MLX's process-wide CompilerCache. This map is constructed before
        // that cache during first compile(), so normal static destruction would
        // tear it down after MLX and can crash at process exit.
        static std::unordered_map<Key, CompiledFn, KeyHash, KeyEq>& cache =
            *new std::unordered_map<Key, CompiledFn, KeyHash, KeyEq>();
        std::lock_guard<std::mutex> lock(mu);
        Key key{head_dim, rotated_dims, n_heads, rms_eps};
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;

        auto fn = [head_dim, n_heads, rms_eps]
            (const std::vector<array>& inputs) -> std::vector<array> {
            const auto& q_in = inputs[0];      // [B, L, n_heads * head_dim]
            const auto& q_norm_w = inputs[1];  // [head_dim]
            const auto& freqs = inputs[2];
            const auto& offset = inputs[3];    // scalar int array

            auto in_shape = q_in.shape();
            int B = in_shape[0];
            int L = in_shape[1];

            auto q = mlx::core::reshape(q_in, {B, L, n_heads, head_dim});
            q = mlx::core::fast::rms_norm(q, q_norm_w, rms_eps);
            q = mlx::core::transpose(q, {0, 2, 1, 3});

            auto out = mlx::core::fast::rope(
                q, head_dim, false, std::nullopt, 1.0f, offset, freqs);
            return {out};
        };

        auto [iter, _] = cache.emplace(
            key, mlx::core::compile(fn, /*shapeless=*/false));
        return iter->second;
    }
}

std::unique_ptr<MlxArray> compiled_q_path_proportional(
    const MlxArray& q_proj_out,
    const MlxArray& q_norm_weight,
    const MlxArray& freqs,
    float rms_eps,
    int32_t n_heads,
    int32_t head_dim,
    int32_t rotated_dims,
    int32_t offset
) {
    auto offset_arr = array(offset);
    auto& compiled_fn = get_compiled_q_path_proportional(
        head_dim, rotated_dims, n_heads, rms_eps);
    auto result = compiled_fn(
        {q_proj_out.inner, q_norm_weight.inner, freqs.inner, offset_arr});
    return std::make_unique<MlxArray>(std::move(result[0]));
}

// Compiled Gemma 4 per-layer-input-gate chain (e2b / e4b variants).
// Collapses
//   gate = gate_proj(after_ffn)
//   gate = gelu_approx(gate)
//   gate = gate * per_layer_input
//   gate = proj(gate)
//   gate = post_norm(gate)
//   out  = after_ffn + gate
// into one `mx::core::compile` graph. Covers the common
// affine/gs=64/bits=4 quantized configuration; other modes fall
// back to the op-at-a-time path at the Rust layer.
namespace {
    using CompiledFn =
        std::function<std::vector<array>(const std::vector<array>&)>;

    // Eps is captured in the compile closure because `fast::rms_norm`
    // takes a scalar float (not an array), so it has to be a
    // compile-time constant. A single Gemma 4 variant uses one eps
    // value, so the cache is expected to hold one entry.
    static CompiledFn& get_compiled_per_layer_input_gate(float eps) {
        static std::mutex& mu = *new std::mutex();
        // Intentionally leaked for the same static-destruction ordering reason
        // as the proportional RoPE compiled-function caches above.
        static std::unordered_map<uint32_t, CompiledFn>& cache =
            *new std::unordered_map<uint32_t, CompiledFn>();
        uint32_t key;
        std::memcpy(&key, &eps, sizeof(float));
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;

        auto fn = [eps](const std::vector<array>& inputs) -> std::vector<array> {
            // inputs: after_ffn, per_layer_input,
            //         gate_w, gate_s, gate_b,
            //         proj_w,  proj_s,  proj_b,
            //         post_norm_w
            const auto& after_ffn = inputs[0];
            const auto& per_layer = inputs[1];
            const auto& gate_w = inputs[2];
            const auto& gate_s = inputs[3];
            const auto& gate_b = inputs[4];
            const auto& proj_w = inputs[5];
            const auto& proj_s = inputs[6];
            const auto& proj_b = inputs[7];
            const auto& post_norm_w = inputs[8];

            int group_size = 64;
            int bits = 4;

            auto gate = mlx::core::quantized_matmul(
                after_ffn, gate_w, gate_s, gate_b, true, group_size, bits);

            auto gelu_gate = gelu_tanh_approx(gate);

            auto gated = mlx::core::multiply(gelu_gate, per_layer);
            auto projected = mlx::core::quantized_matmul(
                gated, proj_w, proj_s, proj_b, true, group_size, bits);
            auto normed = mlx::core::fast::rms_norm(projected, post_norm_w, eps);
            auto combined = mlx::core::add(after_ffn, normed);
            return {combined};
        };
        auto [iter, _] = cache.emplace(
            key, mlx::core::compile(fn, /*shapeless=*/true));
        return iter->second;
    }
}

std::unique_ptr<MlxArray> compiled_per_layer_input_gate(
    const MlxArray& after_ffn,
    const MlxArray& per_layer_input,
    const MlxArray& gate_w,
    const MlxArray& gate_s,
    const MlxArray* gate_b,
    const MlxArray& proj_w,
    const MlxArray& proj_s,
    const MlxArray* proj_b,
    const MlxArray& post_norm_w,
    const MlxArray* gate_global_scale,
    const MlxArray* proj_global_scale,
    float post_norm_eps,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    std::string mode_str(mode.data(), mode.size());

    // Compiled path only supports affine / gs=64 / bits=4 / both biases
    // present. The NVFP4 global-scale sidecar is mutually exclusive with affine
    // quantization (it is emitted only by the ModelOpt NVFP4 transcode), so the
    // sidecar-carrying case never reaches the compiled branch; guarding on the
    // scale pointers makes that explicit and keeps the fold in the eager path
    // below, where the gate scale is applied before the GeGLU and the proj
    // scale on the projected output (issue #698).
    if (mode_str == "affine" && group_size == 64 && bits == 4
        && gate_b && proj_b && !gate_global_scale && !proj_global_scale) {
        auto& compiled_fn = get_compiled_per_layer_input_gate(post_norm_eps);
        auto result = compiled_fn({
            after_ffn.inner, per_layer_input.inner,
            gate_w.inner, gate_s.inner, gate_b->inner,
            proj_w.inner, proj_s.inner, proj_b->inner,
            post_norm_w.inner
        });
        return std::make_unique<MlxArray>(std::move(result[0]));
    }

    // Non-compiled fallback (covers NVFP4 with global-scale sidecars).
    std::optional<array> gb = gate_b ? std::optional(gate_b->inner) : std::nullopt;
    std::optional<array> pb = proj_b ? std::optional(proj_b->inner) : std::nullopt;
    auto gate = mlx::core::quantized_matmul(
        after_ffn.inner, gate_w.inner, gate_s.inner, gb,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);
    gate = apply_global_scale_eager(std::move(gate), gate_global_scale);

    auto gelu_gate = gelu_tanh_approx(gate);
    auto gated = mlx::core::multiply(gelu_gate, per_layer_input.inner);

    auto projected = mlx::core::quantized_matmul(
        gated, proj_w.inner, proj_s.inner, pb,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);
    projected = apply_global_scale_eager(std::move(projected), proj_global_scale);
    auto normed = mlx::core::fast::rms_norm(projected, post_norm_w.inner, post_norm_eps);
    auto combined = mlx::core::add(after_ffn.inner, normed);
    return std::make_unique<MlxArray>(std::move(combined));
}

std::unique_ptr<MlxArray> fast_rms_norm(
    const MlxArray& x,
    const MlxArray& weight,
    float eps
) {
    return std::make_unique<MlxArray>(mlx::core::fast::rms_norm(
        x.inner, weight.inner, eps
    ));
}

std::unique_ptr<MlxArray> fast_rms_norm_no_weight(
    const MlxArray& x,
    float eps
) {
    return std::make_unique<MlxArray>(mlx::core::fast::rms_norm(
        x.inner, std::nullopt, eps
    ));
}

std::unique_ptr<MlxArray> fast_layer_norm(
    const MlxArray& x,
    const MlxArray* weight,
    const MlxArray* bias,
    float eps
) {
    std::optional<array> weight_opt = weight ? std::optional(weight->inner) : std::nullopt;
    std::optional<array> bias_opt = bias ? std::optional(bias->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::fast::layer_norm(
        x.inner, weight_opt, bias_opt, eps
    ));
}

std::unique_ptr<MlxArray> fast_scaled_dot_product_attention(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask
) {
    // MLX SDPA accepts boolean masks (True=attend, False=mask) and additive
    // masks (0=attend, -inf=mask).  Boolean/integer masks must NOT be cast to
    // float, or their True/False semantics are lost and they become additive
    // 1.0/0.0 offsets.  Float masks are cast to Q's dtype to satisfy the
    // "must promote to output type" constraint.
    std::optional<array> mask_opt = std::nullopt;
    std::string mask_mode = "";
    if (mask) {
        auto m = mask->inner;
        if (mlx::core::issubdtype(m.dtype(), mlx::core::floating)) {
            // Additive mask: cast to Q's dtype
            if (m.dtype() != q.inner.dtype()) {
                m = mlx::core::astype(m, q.inner.dtype());
            }
        }
        // Boolean/integer masks are passed through unchanged — MLX
        // handles them natively.
        mask_opt = m;
        mask_mode = "array";
    }
    return std::make_unique<MlxArray>(mlx::core::fast::scaled_dot_product_attention(
        q.inner, k.inner, v.inner, scale, mask_mode, mask_opt
    ));
}

// Fast SDPA with optional sinks (per-head attention bias for first position)
// Used by: GptOss
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention_with_sinks(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale,
    const MlxArray* mask,
    const MlxArray* sinks
) {
    std::optional<array> mask_opt = std::nullopt;
    std::string mask_mode = "";
    if (mask) {
        auto m = mask->inner;
        if (mlx::core::issubdtype(m.dtype(), mlx::core::floating)) {
            // Additive mask: cast to Q's dtype
            if (m.dtype() != q.inner.dtype()) {
                m = mlx::core::astype(m, q.inner.dtype());
            }
        }
        // Boolean/integer masks are passed through unchanged — MLX
        // handles them natively.
        mask_opt = m;
        mask_mode = "array";
    }
    std::optional<array> sinks_opt = sinks ? std::optional(sinks->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::fast::scaled_dot_product_attention(
        q.inner, k.inner, v.inner, scale, mask_mode, mask_opt, sinks_opt
    ));
}

// SDPA with explicit causal masking for prefill
std::unique_ptr<MlxArray> fast_scaled_dot_product_attention_causal(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    float scale
) {
    return std::make_unique<MlxArray>(mlx::core::fast::scaled_dot_product_attention(
        q.inner, k.inner, v.inner, scale, "causal", std::nullopt
    ));
}

std::unique_ptr<MlxArray> paged_decode_attention_dense_compat(
    const MlxArray& q,
    rust::Slice<const MlxArray* const> cache_keys,
    rust::Slice<const MlxArray* const> cache_values,
    rust::Slice<const int32_t> kv_lens,
    rust::Slice<const int32_t> block_tables,
    rust::Slice<const int32_t> block_table_offsets,
    int32_t block_size,
    float scale
) {
    if (block_size <= 0) {
        throw std::invalid_argument("paged_decode_attention_dense_compat: block_size must be > 0");
    }

    const auto q_shape = q.inner.shape();
    if (q_shape.size() != 4 || q_shape[2] != 1) {
        throw std::invalid_argument("paged_decode_attention_dense_compat: expected q shape [B, H, 1, D]");
    }

    const size_t batch = static_cast<size_t>(q_shape[0]);
    if (cache_keys.size() != batch || cache_values.size() != batch || kv_lens.size() != batch) {
        throw std::invalid_argument("paged_decode_attention_dense_compat: batch metadata length mismatch");
    }
    if (block_table_offsets.size() != batch + 1) {
        throw std::invalid_argument("paged_decode_attention_dense_compat: block_table_offsets must have length B + 1");
    }

    std::vector<array> outputs;
    outputs.reserve(batch);

    for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const MlxArray* key_cache = cache_keys[batch_idx];
        const MlxArray* value_cache = cache_values[batch_idx];
        if (key_cache == nullptr || value_cache == nullptr) {
            throw std::invalid_argument("paged_decode_attention_dense_compat: null cache pointer");
        }

        const int32_t kv_len = kv_lens[batch_idx];
        if (kv_len <= 0) {
            throw std::invalid_argument("paged_decode_attention_dense_compat: kv_len must be > 0");
        }

        const int32_t table_begin = block_table_offsets[batch_idx];
        const int32_t table_end = block_table_offsets[batch_idx + 1];
        if (table_begin < 0 || table_end < table_begin ||
            static_cast<size_t>(table_end) > block_tables.size()) {
            throw std::invalid_argument("paged_decode_attention_dense_compat: invalid block table offsets");
        }

        std::vector<array> key_blocks;
        std::vector<array> value_blocks;
        key_blocks.reserve(table_end - table_begin);
        value_blocks.reserve(table_end - table_begin);

        for (int32_t table_idx = table_begin; table_idx < table_end; ++table_idx) {
            const int32_t logical_block = block_tables[table_idx];
            if (logical_block < 0) {
                throw std::invalid_argument("paged_decode_attention_dense_compat: block indices must be non-negative");
            }

            const int32_t block_start = logical_block * block_size;
            if (block_start >= kv_len) {
                continue;
            }
            const int32_t block_end = std::min(block_start + block_size, kv_len);

            key_blocks.push_back(mlx::core::slice(
                key_cache->inner,
                {0, 0, block_start, 0},
                {1, static_cast<int>(key_cache->inner.shape(1)), block_end,
                 static_cast<int>(key_cache->inner.shape(3))}
            ));
            value_blocks.push_back(mlx::core::slice(
                value_cache->inner,
                {0, 0, block_start, 0},
                {1, static_cast<int>(value_cache->inner.shape(1)), block_end,
                 static_cast<int>(value_cache->inner.shape(3))}
            ));
        }

        if (key_blocks.empty() || value_blocks.empty()) {
            throw std::invalid_argument("paged_decode_attention_dense_compat: empty visible block list");
        }

        array key_visible =
            key_blocks.size() == 1 ? key_blocks.front() : mlx::core::concatenate(key_blocks, 2);
        array value_visible = value_blocks.size() == 1
            ? value_blocks.front()
            : mlx::core::concatenate(value_blocks, 2);

        auto q_i = mlx::core::slice(
            q.inner,
            {static_cast<int>(batch_idx), 0, 0, 0},
            {static_cast<int>(batch_idx + 1), static_cast<int>(q_shape[1]), 1,
             static_cast<int>(q_shape[3])}
        );
        auto attn_i = mlx::core::fast::scaled_dot_product_attention(
            q_i, key_visible, value_visible, scale, "", std::nullopt
        );
        outputs.push_back(std::move(attn_i));
    }

    if (outputs.empty()) {
        throw std::invalid_argument("paged_decode_attention_dense_compat: empty batch");
    }
    if (outputs.size() == 1) {
        return std::make_unique<MlxArray>(std::move(outputs.front()));
    }
    return std::make_unique<MlxArray>(mlx::core::concatenate(outputs, 0));
}

std::unique_ptr<MlxArray> paged_decode_attention_rotating_compat(
    const MlxArray& q,
    rust::Slice<const MlxArray* const> cache_keys,
    rust::Slice<const MlxArray* const> cache_values,
    rust::Slice<const int32_t> kv_lens,
    rust::Slice<const int32_t> logical_starts,
    int32_t block_size,
    float scale
) {
    if (block_size <= 0) {
        throw std::invalid_argument("paged_decode_attention_rotating_compat: block_size must be > 0");
    }

    const auto q_shape = q.inner.shape();
    if (q_shape.size() != 4 || q_shape[2] != 1) {
        throw std::invalid_argument("paged_decode_attention_rotating_compat: expected q shape [B, H, 1, D]");
    }

    const size_t batch = static_cast<size_t>(q_shape[0]);
    if (cache_keys.size() != batch || cache_values.size() != batch ||
        kv_lens.size() != batch || logical_starts.size() != batch) {
        throw std::invalid_argument("paged_decode_attention_rotating_compat: batch metadata length mismatch");
    }

    std::vector<array> outputs;
    outputs.reserve(batch);

    for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const MlxArray* key_cache = cache_keys[batch_idx];
        const MlxArray* value_cache = cache_values[batch_idx];
        if (key_cache == nullptr || value_cache == nullptr) {
            throw std::invalid_argument("paged_decode_attention_rotating_compat: null cache pointer");
        }

        const auto key_shape = key_cache->inner.shape();
        const auto value_shape = value_cache->inner.shape();
        if (key_shape.size() != 4 || value_shape.size() != 4) {
            throw std::invalid_argument("paged_decode_attention_rotating_compat: cache tensors must have rank 4");
        }

        const int32_t kv_len = kv_lens[batch_idx];
        const int32_t logical_start = logical_starts[batch_idx];
        const int32_t buffer_len = static_cast<int32_t>(key_shape[2]);
        if (kv_len <= 0) {
            throw std::invalid_argument("paged_decode_attention_rotating_compat: kv_len must be > 0");
        }
        if (buffer_len <= 0) {
            throw std::invalid_argument("paged_decode_attention_rotating_compat: buffer_len must be > 0");
        }
        if (logical_start < 0 || logical_start >= buffer_len) {
            throw std::invalid_argument("paged_decode_attention_rotating_compat: logical_start out of range");
        }

        std::vector<array> key_blocks;
        std::vector<array> value_blocks;
        const int32_t block_count = (kv_len + block_size - 1) / block_size;
        key_blocks.reserve(block_count);
        value_blocks.reserve(block_count);

        for (int32_t logical_block = 0; logical_block < block_count; ++logical_block) {
            const int32_t logical_pos = logical_block * block_size;
            const int32_t logical_end = std::min(logical_pos + block_size, kv_len);
            const int32_t token_count = logical_end - logical_pos;
            if (token_count <= 0) {
                continue;
            }

            const int32_t physical_start = (logical_start + logical_pos) % buffer_len;
            const int32_t physical_end = physical_start + token_count;
            if (physical_end <= buffer_len) {
                key_blocks.push_back(mlx::core::slice(
                    key_cache->inner,
                    {0, 0, physical_start, 0},
                    {1, static_cast<int>(key_shape[1]), physical_end, static_cast<int>(key_shape[3])}
                ));
                value_blocks.push_back(mlx::core::slice(
                    value_cache->inner,
                    {0, 0, physical_start, 0},
                    {1, static_cast<int>(value_shape[1]), physical_end, static_cast<int>(value_shape[3])}
                ));
            } else {
                const int32_t tail_count = buffer_len - physical_start;
                const int32_t head_count = token_count - tail_count;

                auto key_tail = mlx::core::slice(
                    key_cache->inner,
                    {0, 0, physical_start, 0},
                    {1, static_cast<int>(key_shape[1]), buffer_len, static_cast<int>(key_shape[3])}
                );
                auto key_head = mlx::core::slice(
                    key_cache->inner,
                    {0, 0, 0, 0},
                    {1, static_cast<int>(key_shape[1]), head_count, static_cast<int>(key_shape[3])}
                );
                key_blocks.push_back(mlx::core::concatenate(std::vector<array>{key_tail, key_head}, 2));

                auto value_tail = mlx::core::slice(
                    value_cache->inner,
                    {0, 0, physical_start, 0},
                    {1, static_cast<int>(value_shape[1]), buffer_len, static_cast<int>(value_shape[3])}
                );
                auto value_head = mlx::core::slice(
                    value_cache->inner,
                    {0, 0, 0, 0},
                    {1, static_cast<int>(value_shape[1]), head_count, static_cast<int>(value_shape[3])}
                );
                value_blocks.push_back(mlx::core::concatenate(std::vector<array>{value_tail, value_head}, 2));
            }
        }

        if (key_blocks.empty() || value_blocks.empty()) {
            throw std::invalid_argument("paged_decode_attention_rotating_compat: empty visible block list");
        }

        array key_visible =
            key_blocks.size() == 1 ? key_blocks.front() : mlx::core::concatenate(key_blocks, 2);
        array value_visible = value_blocks.size() == 1
            ? value_blocks.front()
            : mlx::core::concatenate(value_blocks, 2);

        auto q_i = mlx::core::slice(
            q.inner,
            {static_cast<int>(batch_idx), 0, 0, 0},
            {static_cast<int>(batch_idx + 1), static_cast<int>(q_shape[1]), 1,
             static_cast<int>(q_shape[3])}
        );
        auto attn_i = mlx::core::fast::scaled_dot_product_attention(
            q_i, key_visible, value_visible, scale, "", std::nullopt
        );
        outputs.push_back(std::move(attn_i));
    }

    if (outputs.empty()) {
        throw std::invalid_argument("paged_decode_attention_rotating_compat: empty batch");
    }
    if (outputs.size() == 1) {
        return std::make_unique<MlxArray>(std::move(outputs.front()));
    }
    return std::make_unique<MlxArray>(mlx::core::concatenate(outputs, 0));
}

namespace {
bool sdpa_supports_fast_path_impl(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal
) {
    const int value_head_dim = v.shape(-1);
    const int query_head_dim = q.shape(-1);
    const int query_sequence_length = q.shape(2);
    const int key_sequence_length = k.shape(2);
    const int num_query_heads = q.shape(1);
    const int num_kv_heads = k.shape(1);
    const int gqa_factor = num_query_heads / std::max(num_kv_heads, 1);

    const bool sdpa_vector_supported_head_dim =
        query_head_dim == value_head_dim &&
        (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128 ||
         query_head_dim == 256);
    const bool sdpa_full_supported_head_dim =
        query_head_dim == value_head_dim &&
        (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128 ||
         query_head_dim == 256);
    const bool sdpa_full_supported_mask =
        !has_mask || has_arr_mask ||
        (query_sequence_length <= key_sequence_length && do_causal);

    const bool supports_sdpa_full =
        query_sequence_length > 8 && sdpa_full_supported_mask && sdpa_full_supported_head_dim;
    const bool supports_sdpa_vector =
        query_sequence_length <= 8 && query_sequence_length <= key_sequence_length &&
        sdpa_vector_supported_head_dim && (query_sequence_length * gqa_factor) <= 32;

    return supports_sdpa_full || supports_sdpa_vector;
}

bool sdpa_supports_nax_impl(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal
) {
#ifdef __APPLE__
    const bool supports_fast_path =
        sdpa_supports_fast_path_impl(q, k, v, has_mask, has_arr_mask, do_causal);
    const bool supports_full_path =
        q.shape(2) > 8 && q.shape(-1) == v.shape(-1) &&
        (q.shape(-1) == 64 || q.shape(-1) == 80 || q.shape(-1) == 128 ||
         q.shape(-1) == 256);
    return supports_fast_path && supports_full_path &&
        q.shape(3) != 80 &&
        (mlx::core::env::enable_tf32() || q.dtype() != mlx::core::float32);
#else
    (void)q;
    (void)k;
    (void)v;
    (void)has_mask;
    (void)has_arr_mask;
    (void)do_causal;
    return false;
#endif
}
} // namespace

bool sdpa_supports_fast_path(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal
) {
    return sdpa_supports_fast_path_impl(
        q.inner, k.inner, v.inner, has_mask, has_arr_mask, do_causal);
}

bool sdpa_supports_nax(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal
) {
    return sdpa_supports_nax_impl(
        q.inner, k.inner, v.inner, has_mask, has_arr_mask, do_causal);
}

// Fused QKV projection + reshape + transpose + RoPE (no cache, no SDPA)
// Returns new_k array ready for cache. Call this function twice (for k and v) or use fused version below.
// This reduces FFI overhead for the projection + reshape + transpose + RoPE chain.
std::unique_ptr<MlxArray> fused_qkv_project_and_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,
    int32_t num_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    bool apply_rope,
    rust::Str mode
) {
    auto batch_size = x.inner.shape()[0];
    auto seq_len = x.inner.shape()[1];

    // Projection
    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
    std::string mode_str(mode.data(), mode.size());
    auto proj = mlx::core::quantized_matmul(
        x.inner, weight.inner, scales.inner, biases_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    // Reshape to [batch, seq_len, n_heads, head_dim]
    proj = reshape(proj, {batch_size, seq_len, num_heads, head_dim});

    // Transpose to [batch, n_heads, seq_len, head_dim]
    proj = transpose(proj, {0, 2, 1, 3});

    // Apply RoPE if requested (Q and K need it, V doesn't)
    if (apply_rope) {
        proj = mlx::core::fast::rope(proj, rope_dims, false, rope_base, 1.0f, cache_offset);
    }

    return std::make_unique<MlxArray>(std::move(proj));
}

void fused_qkv_project_split_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    rust::Str mode,
    std::unique_ptr<MlxArray>& q_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out
) {
    using namespace mlx::core;

    auto batch_size = x.inner.shape()[0];
    auto seq_len = x.inner.shape()[1];

    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
    std::string mode_str(mode.data(), mode.size());
    auto proj = quantized_matmul(
        x.inner, weight.inner, scales.inner, biases_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    int q_cols = num_heads * head_dim;
    int kv_cols = num_kv_heads * head_dim;
    int qkv_cols = q_cols + (2 * kv_cols);

    auto proj_shape = proj.shape();
    if (proj_shape.size() != 3 || proj_shape[2] != qkv_cols) {
        throw std::runtime_error("fused_qkv_project_split_rope: unexpected projection shape");
    }

    auto q = slice(proj, {0, 0, 0}, {batch_size, seq_len, q_cols});
    auto k = slice(proj, {0, 0, q_cols}, {batch_size, seq_len, q_cols + kv_cols});
    auto v = slice(proj, {0, 0, q_cols + kv_cols}, {batch_size, seq_len, qkv_cols});

    q = reshape(q, {batch_size, seq_len, num_heads, head_dim});
    k = reshape(k, {batch_size, seq_len, num_kv_heads, head_dim});
    v = reshape(v, {batch_size, seq_len, num_kv_heads, head_dim});

    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    q = mlx::core::fast::rope(q, rope_dims, false, rope_base, 1.0f, cache_offset);
    k = mlx::core::fast::rope(k, rope_dims, false, rope_base, 1.0f, cache_offset);

    q_out = std::make_unique<MlxArray>(std::move(q));
    k_out = std::make_unique<MlxArray>(std::move(k));
    v_out = std::make_unique<MlxArray>(std::move(v));
}

void fused_qkv_project_split_norm_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,
    const MlxArray& q_norm_weight,
    const MlxArray& k_norm_weight,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    float rms_eps,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    rust::Str mode,
    std::unique_ptr<MlxArray>& q_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out
) {
    using namespace mlx::core;

    auto batch_size = x.inner.shape()[0];
    auto seq_len = x.inner.shape()[1];

    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
    std::string mode_str(mode.data(), mode.size());
    auto proj = quantized_matmul(
        x.inner, weight.inner, scales.inner, biases_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    int q_cols = num_heads * head_dim;
    int kv_cols = num_kv_heads * head_dim;
    int qkv_cols = q_cols + (2 * kv_cols);

    auto proj_shape = proj.shape();
    if (proj_shape.size() != 3 || proj_shape[2] != qkv_cols) {
        throw std::runtime_error("fused_qkv_project_split_norm_rope: unexpected projection shape");
    }

    auto q = slice(proj, {0, 0, 0}, {batch_size, seq_len, q_cols});
    auto k = slice(proj, {0, 0, q_cols}, {batch_size, seq_len, q_cols + kv_cols});
    auto v = slice(proj, {0, 0, q_cols + kv_cols}, {batch_size, seq_len, qkv_cols});

    q = reshape(q, {batch_size, seq_len, num_heads, head_dim});
    k = reshape(k, {batch_size, seq_len, num_kv_heads, head_dim});
    v = reshape(v, {batch_size, seq_len, num_kv_heads, head_dim});

    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    q = mlx::core::fast::rms_norm(q, q_norm_weight.inner, rms_eps);
    k = mlx::core::fast::rms_norm(k, k_norm_weight.inner, rms_eps);

    q = mlx::core::fast::rope(q, rope_dims, false, rope_base, 1.0f, cache_offset);
    k = mlx::core::fast::rope(k, rope_dims, false, rope_base, 1.0f, cache_offset);

    q_out = std::make_unique<MlxArray>(std::move(q));
    k_out = std::make_unique<MlxArray>(std::move(k));
    v_out = std::make_unique<MlxArray>(std::move(v));
}

namespace {
array apply_su_scaled_rope(
    array x,
    int32_t rope_dims,
    const array& rope_freqs,
    float rope_input_scale,
    int32_t cache_offset
) {
    if (rope_input_scale != 1.0f && rope_dims > 0) {
        auto x_shape = x.shape();
        Shape starts(x_shape.size(), 0);
        Shape stops(x_shape.begin(), x_shape.end());
        stops.back() = std::min<int32_t>(rope_dims, x_shape.back());
        auto rotary = slice(x, starts, stops);
        auto scale = array(rope_input_scale, x.dtype());
        rotary = multiply(rotary, scale);
        x = slice_update(x, rotary, starts, stops);
    }

    return mlx::core::fast::rope(
        x, rope_dims, false, std::nullopt, 1.0f, cache_offset, rope_freqs);
}
}

void fused_qkv_project_split_su_scaled_rope(
    const MlxArray& x,
    const MlxArray& weight,
    const MlxArray& scales,
    const MlxArray* biases,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    const MlxArray& rope_freqs,
    float rope_input_scale,
    int32_t cache_offset,
    int32_t group_size,
    int32_t bits,
    rust::Str mode,
    std::unique_ptr<MlxArray>& q_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out
) {
    using namespace mlx::core;

    auto batch_size = x.inner.shape()[0];
    auto seq_len = x.inner.shape()[1];

    std::optional<array> biases_opt = biases ? std::optional(biases->inner) : std::nullopt;
    std::string mode_str(mode.data(), mode.size());
    auto proj = quantized_matmul(
        x.inner, weight.inner, scales.inner, biases_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    int q_cols = num_heads * head_dim;
    int kv_cols = num_kv_heads * head_dim;
    int qkv_cols = q_cols + (2 * kv_cols);

    auto proj_shape = proj.shape();
    if (proj_shape.size() != 3 || proj_shape[2] != qkv_cols) {
        throw std::runtime_error(
            "fused_qkv_project_split_su_scaled_rope: unexpected projection shape");
    }

    auto q = slice(proj, {0, 0, 0}, {batch_size, seq_len, q_cols});
    auto k = slice(proj, {0, 0, q_cols}, {batch_size, seq_len, q_cols + kv_cols});
    auto v = slice(proj, {0, 0, q_cols + kv_cols}, {batch_size, seq_len, qkv_cols});

    q = reshape(q, {batch_size, seq_len, num_heads, head_dim});
    k = reshape(k, {batch_size, seq_len, num_kv_heads, head_dim});
    v = reshape(v, {batch_size, seq_len, num_kv_heads, head_dim});

    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    q = apply_su_scaled_rope(q, rope_dims, rope_freqs.inner, rope_input_scale, cache_offset);
    k = apply_su_scaled_rope(k, rope_dims, rope_freqs.inner, rope_input_scale, cache_offset);

    q_out = std::make_unique<MlxArray>(std::move(q));
    k_out = std::make_unique<MlxArray>(std::move(k));
    v_out = std::make_unique<MlxArray>(std::move(v));
}

void fused_causal_prefill_attention(
    const MlxArray& x,
    const MlxArray& qkv_weight,
    const MlxArray& qkv_scales,
    const MlxArray* qkv_biases,
    const MlxArray& o_weight,
    const MlxArray& o_scales,
    const MlxArray* o_biases,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t rope_dims,
    float rope_base,
    float scale,
    int32_t group_size,
    int32_t bits,
    rust::Str mode,
    std::unique_ptr<MlxArray>& output_out,
    std::unique_ptr<MlxArray>& k_out,
    std::unique_ptr<MlxArray>& v_out
) {
    using namespace mlx::core;

    auto batch_size = x.inner.shape()[0];
    auto seq_len = x.inner.shape()[1];

    std::optional<array> qkv_biases_opt = qkv_biases ? std::optional(qkv_biases->inner) : std::nullopt;
    std::optional<array> o_biases_opt = o_biases ? std::optional(o_biases->inner) : std::nullopt;
    std::string mode_str(mode.data(), mode.size());

    auto proj = quantized_matmul(
        x.inner, qkv_weight.inner, qkv_scales.inner, qkv_biases_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    int q_cols = num_heads * head_dim;
    int kv_cols = num_kv_heads * head_dim;
    int qkv_cols = q_cols + (2 * kv_cols);

    auto q = slice(proj, {0, 0, 0}, {batch_size, seq_len, q_cols});
    auto k = slice(proj, {0, 0, q_cols}, {batch_size, seq_len, q_cols + kv_cols});
    auto v = slice(proj, {0, 0, q_cols + kv_cols}, {batch_size, seq_len, qkv_cols});

    q = reshape(q, {batch_size, seq_len, num_heads, head_dim});
    k = reshape(k, {batch_size, seq_len, num_kv_heads, head_dim});
    v = reshape(v, {batch_size, seq_len, num_kv_heads, head_dim});

    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    q = mlx::core::fast::rope(q, rope_dims, false, rope_base, 1.0f, 0);
    k = mlx::core::fast::rope(k, rope_dims, false, rope_base, 1.0f, 0);

    auto attn = mlx::core::fast::scaled_dot_product_attention(
        q, k, v, scale, "causal", std::nullopt);

    attn = transpose(attn, {0, 2, 1, 3});
    attn = reshape(attn, {batch_size, seq_len, q_cols});

    auto output = quantized_matmul(
        attn, o_weight.inner, o_scales.inner, o_biases_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    output_out = std::make_unique<MlxArray>(std::move(output));
    k_out = std::make_unique<MlxArray>(std::move(k));
    v_out = std::make_unique<MlxArray>(std::move(v));
}

// Compiled operations (with kernel fusion).
// Compiled MoE expert forward with quantized weights
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)> get_compiled_qmoe_expert() {
        auto moe_expert_fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            // inputs: [x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b, params]
            const auto& x = inputs[0];
            const auto& gate_w = inputs[1];
            const auto& gate_s = inputs[2];
            const auto& gate_b = inputs[3];
            const auto& up_w = inputs[4];
            const auto& up_s = inputs[5];
            const auto& up_b = inputs[6];
            const auto& down_w = inputs[7];
            const auto& down_s = inputs[8];
            const auto& down_b = inputs[9];
            // params[0] = group_size, params[1] = bits
            int group_size = 64;  // hardcoded for common case
            int bits = 4;
            // gate = quantized_matmul(x, gate_w, gate_s, gate_b)
            auto gate = mlx::core::quantized_matmul(x, gate_w, gate_s, gate_b, true, group_size, bits);

            // up = quantized_matmul(x, up_w, up_s, up_b)
            auto up = mlx::core::quantized_matmul(x, up_w, up_s, up_b, true, group_size, bits);

            // SwiGLU: silu(gate) * up
            auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
            auto activated = mlx::core::multiply(silu_gate, up);

            // down = quantized_matmul(activated, down_w, down_s, down_b)
            auto down = mlx::core::quantized_matmul(activated, down_w, down_s, down_b, true, group_size, bits);

            return {down};
        };
        return mlx::core::compile(moe_expert_fn, true);  // shapeless=true
    }
}

std::unique_ptr<MlxArray> compiled_moe_expert_forward(
    const MlxArray& x,
    const MlxArray& gate_proj,
    const MlxArray& gate_scales,
    const MlxArray* gate_biases,
    const MlxArray& up_proj,
    const MlxArray& up_scales,
    const MlxArray* up_biases,
    const MlxArray& down_proj,
    const MlxArray& down_scales,
    const MlxArray* down_biases,
    int32_t group_size,
    int32_t bits,
    rust::Str mode
) {
    std::string mode_str(mode.data(), mode.size());

    // Compiled path only supports affine mode with group_size=64, bits=4, and biases present.
    // The compiled lambda hardcodes these quantization parameters for graph caching.
    if (mode_str == "affine" && group_size == 64 && bits == 4
        && gate_biases && up_biases && down_biases) {
        static auto compiled_fn = get_compiled_qmoe_expert();

        auto result = compiled_fn({
            x.inner,
            gate_proj.inner, gate_scales.inner, gate_biases->inner,
            up_proj.inner, up_scales.inner, up_biases->inner,
            down_proj.inner, down_scales.inner, down_biases->inner
        });

        return std::make_unique<MlxArray>(std::move(result[0]));
    }

    // Non-compiled fallback for mxfp4/nvfp4/mxfp8 or missing biases
    std::optional<array> gb_opt = gate_biases ? std::optional(gate_biases->inner) : std::nullopt;
    std::optional<array> ub_opt = up_biases ? std::optional(up_biases->inner) : std::nullopt;
    std::optional<array> db_opt = down_biases ? std::optional(down_biases->inner) : std::nullopt;

    auto gate = mlx::core::quantized_matmul(
        x.inner, gate_proj.inner, gate_scales.inner, gb_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);
    auto up = mlx::core::quantized_matmul(
        x.inner, up_proj.inner, up_scales.inner, ub_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    auto silu_gate = mlx::core::multiply(gate, mlx::core::sigmoid(gate));
    auto activated = mlx::core::multiply(silu_gate, up);

    auto down = mlx::core::quantized_matmul(
        activated, down_proj.inner, down_scales.inner, db_opt,
        true, std::optional<int>(group_size), std::optional<int>(bits), mode_str);

    return std::make_unique<MlxArray>(std::move(down));
}

// Memory management.
void clear_memory_cache() {
    mlx::core::clear_cache();
}

void async_eval(const MlxArray& arr) {
    mlx::core::async_eval(const_cast<array&>(arr.inner));
}

void async_eval_all(rust::Slice<const MlxArray* const> arrays) {
    std::vector<array> arrs;
    arrs.reserve(arrays.size());
    for (const auto* a : arrays) {
        arrs.push_back(a->inner);
    }
    mlx::core::async_eval(arrs);
}

void detach_all(rust::Slice<const MlxArray* const> arrays) {
    for (const auto* a : arrays) {
        const_cast<array&>(a->inner).detach();
    }
}

void synchronize_default() {
    mlx::core::synchronize();
}

void set_default_device(bool gpu) {
    mlx::core::set_default_device(gpu ? mlx::core::Device::gpu : mlx::core::Device::cpu);
}

size_t set_wired_limit(size_t limit) {
    return mlx::core::set_wired_limit(limit);
}

size_t get_wired_limit() {
    auto& info = mlx::core::device_info();
    auto it = info.find("max_recommended_working_set_size");
    if (it != info.end()) {
        return std::get<size_t>(it->second);
    }
    return 0;
}

// MLX runtime memory accounting (issue #55) — thin one-line forwarders to
// the canonical entry points in `mlx/memory.h`. The active allocator
// (Metal / CUDA / no-gpu CommonAllocator) decides what each value means;
// see the header comment in `mlx_cxx_bridge.h` for the cross-backend
// semantics.
size_t get_active_memory() {
    return mlx::core::get_active_memory();
}

size_t get_peak_memory() {
    return mlx::core::get_peak_memory();
}

size_t get_cache_memory() {
    return mlx::core::get_cache_memory();
}

size_t set_memory_limit(size_t limit) {
    return mlx::core::set_memory_limit(limit);
}

size_t get_memory_limit() {
    return mlx::core::get_memory_limit();
}

size_t set_cache_limit(size_t limit) {
    return mlx::core::set_cache_limit(limit);
}

void reset_peak_memory() {
    mlx::core::reset_peak_memory();
}

size_t gpu_max_memory_size() {
    auto& info = mlx::core::device_info();
    // Metal backend uses "max_recommended_working_set_size"
    // CUDA backend uses "total_memory"
    for (const auto& key : {"max_recommended_working_set_size", "total_memory"}) {
        auto it = info.find(key);
        if (it != info.end()) {
            return std::get<size_t>(it->second);
        }
    }
    return 0;
}

std::unique_ptr<MlxStream> new_gpu_stream() {
    return std::make_unique<MlxStream>(mlx::core::new_stream(mlx::core::Device::gpu));
}

// Optimized generation functions.
// Extract last token logits: logits[:, -1, :] -> [batch, vocab]
// This is the optimized path for sampling during generation
std::unique_ptr<MlxArray> slice_last_logits(const MlxArray& logits) {
    // logits shape: [batch, seq_len, vocab_size]
    auto shape = logits.inner.shape();
    if (shape.size() != 3) {
        // Already 2D, just return copy
        return std::make_unique<MlxArray>(logits.inner);
    }
    int32_t seq_len = shape[1];
    // Use slice to get [:, -1:, :] then squeeze
    auto sliced = mlx::core::slice(logits.inner, {0, seq_len - 1, 0}, shape);
    auto squeezed = mlx::core::squeeze(sliced, 1);
    return std::make_unique<MlxArray>(std::move(squeezed));
}

// Slice on the last dimension only: a[..., start:end]
// Useful for fused QKV/gate_up projections
std::unique_ptr<MlxArray> slice_last_dim(const MlxArray& a, int32_t start, int32_t end) {
    auto shape = a.inner.shape();
    int ndim = shape.size();

    // Build starts: [0, 0, ..., start]
    Shape starts(ndim, 0);
    starts[ndim - 1] = start;

    // Build stops: [dim0, dim1, ..., end]
    Shape stops;
    for (int i = 0; i < ndim - 1; i++) {
        stops.push_back(shape[i]);
    }
    stops.push_back(end);

    return std::make_unique<MlxArray>(mlx::core::slice(a.inner, starts, stops));
}

// Argmax on last axis for greedy sampling, returns scalar
std::unique_ptr<MlxArray> argmax_last_axis(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::argmax(a.inner, -1, false));
}

// Reshape token for next forward pass: [] or [batch] -> [batch, 1]
// This allows passing token array directly without extracting scalar value
std::unique_ptr<MlxArray> reshape_token_for_forward(const MlxArray& token) {
    auto shape = token.inner.shape();
    if (shape.empty()) {
        // Scalar -> [1, 1]
        return std::make_unique<MlxArray>(mlx::core::reshape(token.inner, {1, 1}));
    } else if (shape.size() == 1) {
        // [batch] -> [batch, 1]
        return std::make_unique<MlxArray>(mlx::core::reshape(token.inner, {static_cast<int32_t>(shape[0]), 1}));
    }
    // Already correct shape
    return std::make_unique<MlxArray>(token.inner);
}

// Async eval two arrays at once (for lookahead pipelining)
void async_eval_pair(const MlxArray& a, const MlxArray& b) {
    std::vector<array> arrs = {a.inner, b.inner};
    mlx::core::async_eval(arrs);
}

// Export a pair of unevaluated arrays as a DOT graph for profiling.
void export_to_dot_pair(rust::Str path, const MlxArray& a, const MlxArray& b) {
    std::ofstream os(std::string(path.data(), path.size()));
    if (!os.is_open()) {
        throw std::runtime_error("failed to open DOT export path");
    }
    std::vector<array> arrs = {a.inner, b.inner};
    mlx::core::export_to_dot(os, arrs);
}

// Set default stream for subsequent operations
void set_default_stream(const MlxStream& stream) {
    mlx::core::set_default_stream(stream.inner);
}

// Check whether the current default device is GPU
bool is_gpu_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu;
}

// Top-p (nucleus) filtering.
// Not compiled: MLX v0.31.x Scan primitive (cumsum) lacks output_shapes,
// which causes "CumSum cannot infer output shapes" when used inside
// mlx::core::compile with shapeless=true.
namespace {
    array top_p_filter(const array& x, float top_p) {
        auto probs = mlx::core::softmax(x, -1);
        auto sorted_indices = mlx::core::argsort(mlx::core::negative(probs), -1);
        auto sorted_probs = mlx::core::take_along_axis(probs, sorted_indices, -1);
        auto cum_probs = mlx::core::cumsum(sorted_probs, -1, false, true);
        auto shifted_cum = cum_probs - sorted_probs;
        auto mask = mlx::core::less_equal(shifted_cum, mlx::core::array(top_p));
        auto sorted_logits = mlx::core::take_along_axis(x, sorted_indices, -1);
        auto filtered_sorted = mlx::core::where(
            mask, sorted_logits, mlx::core::array(std::numeric_limits<float>::lowest()));
        auto unsort_indices = mlx::core::argsort(sorted_indices, -1);
        return mlx::core::take_along_axis(filtered_sorted, unsort_indices, -1);
    }
}

// Compiled min-p filtering fused into a single kernel.
namespace {
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_min_p_filter() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            const auto& x = inputs[0];
            const auto& min_p_arr = inputs[1];
            auto probs = mlx::core::softmax(x, -1);
            auto max_prob = mlx::core::max(probs, -1, true);
            auto threshold = mlx::core::multiply(max_prob, min_p_arr);
            auto mask = mlx::core::greater_equal(probs, threshold);
            return {mlx::core::where(mask, x,
                mlx::core::array(std::numeric_limits<float>::lowest()))};
        };
        return mlx::core::compile(fn, true);
    }
}

// Fused sampling: temperature scaling + top-k + top-p + min-p + categorical
// in a single function call to minimize FFI round-trips.
// Input: 2D logits [batch, vocab] (already sliced, penalties already applied)
// Returns sampled token
//
// Uses compiled (fused) kernels where supported:
// - Categorical sampling: temp scaling + random::categorical in one kernel
// - Min-p filtering: softmax + max + mask in one kernel
// - Top-p filtering: uncompiled (cumsum lacks output_shapes in MLX v0.31.x)
std::unique_ptr<MlxArray> fused_sample(
    const MlxArray& logits,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p
) {
    auto x = logits.inner;

    // Greedy path: argmax
    if (temperature == 0.0f || top_k == 1) {
        return std::make_unique<MlxArray>(mlx::core::argmax(x, -1, false));
    }

    // Temperature scaling
    if (temperature > 0.0f && temperature != 1.0f) {
        x = x / mlx::core::array(temperature);
    }

    // Top-k filtering: keep only the k highest-probability tokens
    // (Not compiled: argpartition has dynamic shape that breaks compile)
    if (top_k > 0) {
        auto neg_x = mlx::core::negative(x);
        auto indices = mlx::core::argpartition(neg_x, top_k - 1, -1);

        auto shape = indices.shape();
        int ndim = shape.size();
        Shape start(ndim, 0);
        Shape stop(shape.begin(), shape.end());
        start[ndim - 1] = top_k - 1;
        stop[ndim - 1] = top_k;

        auto kth_idx = mlx::core::slice(indices, start, stop);
        auto threshold = mlx::core::take_along_axis(x, kth_idx, -1);
        auto mask = mlx::core::greater_equal(x, threshold);
        x = mlx::core::where(mask, x, mlx::core::array(std::numeric_limits<float>::lowest()));
    }

    // Top-p (nucleus) filtering
    if (top_p > 0.0f && top_p < 1.0f) {
        x = top_p_filter(x, top_p);
    }

    // Min-p filtering — compiled kernel
    if (min_p > 0.0f && min_p < 1.0f) {
        static auto compiled_fn = get_compiled_min_p_filter();
        auto result = compiled_fn({x, mlx::core::array(min_p)});
        x = std::move(result[0]);
    }

    // Categorical sampling (not compiled — random ops need known shapes at trace time)
    return std::make_unique<MlxArray>(mlx::core::random::categorical(x, -1));
}

// SSM (State Space Model) primitives for Mamba/Jamba/Nemotron-H.
// Cumulative sum along axis
std::unique_ptr<MlxArray> cumsum(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::cumsum(a.inner, axis, reverse, inclusive));
}

// Lower triangular matrix (keeps elements on and below k-th diagonal)
std::unique_ptr<MlxArray> tril(const MlxArray& a, int32_t k) {
    return std::make_unique<MlxArray>(mlx::core::tril(a.inner, k));
}

// Upper triangular matrix (keeps elements on and above k-th diagonal)
std::unique_ptr<MlxArray> triu(const MlxArray& a, int32_t k) {
    return std::make_unique<MlxArray>(mlx::core::triu(a.inner, k));
}

// Clip values to range [a_min, a_max]
std::unique_ptr<MlxArray> clip(const MlxArray& a, const MlxArray& a_min, const MlxArray& a_max) {
    return std::make_unique<MlxArray>(mlx::core::clip(a.inner, a_min.inner, a_max.inner));
}

// log(1 + x) - numerically stable for small x, used for softplus
std::unique_ptr<MlxArray> log1p(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::log1p(a.inner));
}

// Numerically stable softplus activation: log(1 + exp(x))
// Uses logaddexp(x, 0) = log(exp(x) + exp(0)) which matches Python's mx.logaddexp(x, 0).
// This avoids float16 overflow that log1p(exp(x)) causes for x >= ~11.09
// (where exp(x) overflows float16's max of 65504).
// Used by: Mamba, Mamba2, SSM models for delta = softplus(dt_proj(delta))
std::unique_ptr<MlxArray> softplus(const MlxArray& a) {
    auto zero = mlx::core::zeros_like(a.inner);
    return std::make_unique<MlxArray>(mlx::core::logaddexp(a.inner, zero));
}

// Compiled gated-delta decode step kernel.
// Uses mlx::core::compile to fuse all operations into a single Metal kernel dispatch.
// This matches Python mlx-lm's gated_delta_kernel which uses a custom Metal kernel.
namespace {
    // Compiled version for scalar gate (g_ndim == 2): [B, H]
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_gated_delta_step_scalar_gate() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            using namespace mlx::core;
            const auto& q = inputs[0];      // [B, H, Dk]
            const auto& k = inputs[1];      // [B, H, Dk]
            const auto& v = inputs[2];      // [B, H, Dv]
            const auto& g = inputs[3];      // [B, H]
            const auto& beta = inputs[4];   // [B, H]
            const auto& state = inputs[5];  // [B, H, Dv, Dk]

            auto decay = expand_dims(expand_dims(g, -1), -1);  // [B,H,1,1]
            auto ns = multiply(state, decay);

            auto k_exp = expand_dims(k, -2);
            auto kv_mem = sum(multiply(ns, k_exp), -1, false);

            auto beta_exp = expand_dims(beta, -1);
            auto delta = multiply(subtract(v, kv_mem), beta_exp);
            auto delta_exp = expand_dims(delta, -1);
            ns = add(ns, multiply(k_exp, delta_exp));

            auto q_exp = expand_dims(q, -2);
            auto y = sum(multiply(ns, q_exp), -1, false);

            return {y, ns};
        };
        return mlx::core::compile(fn, true);
    }

    // Compiled version for per-dim gate (g_ndim == 3): [B, H, Dk]
    static std::function<std::vector<array>(const std::vector<array>&)>
    get_compiled_gated_delta_step_dim_gate() {
        auto fn = [](const std::vector<array>& inputs) -> std::vector<array> {
            using namespace mlx::core;
            const auto& q = inputs[0];
            const auto& k = inputs[1];
            const auto& v = inputs[2];
            const auto& g = inputs[3];      // [B, H, Dk]
            const auto& beta = inputs[4];
            const auto& state = inputs[5];

            auto decay = expand_dims(g, -2);  // [B,H,1,Dk]
            auto ns = multiply(state, decay);

            auto k_exp = expand_dims(k, -2);
            auto kv_mem = sum(multiply(ns, k_exp), -1, false);

            auto beta_exp = expand_dims(beta, -1);
            auto delta = multiply(subtract(v, kv_mem), beta_exp);
            auto delta_exp = expand_dims(delta, -1);
            ns = add(ns, multiply(k_exp, delta_exp));

            auto q_exp = expand_dims(q, -2);
            auto y = sum(multiply(ns, q_exp), -1, false);

            return {y, ns};
        };
        return mlx::core::compile(fn, true);
    }
}

// Fused gated-delta single-token decode step using compiled kernels.
// Uses mlx::core::compile for kernel fusion matching Python's gated_delta_kernel.
//
// Used by: Qwen3.5, Qwen3Next, KimiLinear (GatedDeltaNet T=1 decode)
void fused_gated_delta_decode_step(
    const MlxArray& q,
    const MlxArray& k,
    const MlxArray& v,
    const MlxArray& g,
    const MlxArray& beta,
    const MlxArray& state,
    int32_t q_dtype,
    std::unique_ptr<MlxArray>& output,
    std::unique_ptr<MlxArray>& new_state_out
) {
    using namespace mlx::core;

    std::vector<array> result;
    if (g.inner.ndim() == 2) {
        static auto compiled_fn = get_compiled_gated_delta_step_scalar_gate();
        result = compiled_fn({q.inner, k.inner, v.inner, g.inner, beta.inner, state.inner});
    } else {
        static auto compiled_fn = get_compiled_gated_delta_step_dim_gate();
        result = compiled_fn({q.inner, k.inner, v.inner, g.inner, beta.inner, state.inner});
    }

    auto y = std::move(result[0]);
    auto ns = std::move(result[1]);

    // Cast output to query dtype if needed
    auto target_dtype = static_cast<Dtype>(to_dtype(q_dtype));
    if (y.dtype() != target_dtype) {
        y = astype(y, target_dtype);
    }

    output = std::make_unique<MlxArray>(std::move(y));
    new_state_out = std::make_unique<MlxArray>(std::move(ns));
}

// 1D convolution with groups support (for depthwise conv when groups=channels)
std::unique_ptr<MlxArray> conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv1d(
        input.inner, weight.inner, stride, padding, dilation, groups
    ));
}

// Same body as `conv1d`, but declared `-> Result` on the Rust side so cxx wraps
// the call in a try/catch and converts MLX's eager shape-mismatch exception (and
// any other throw) into a Rust `Err` instead of letting it abort the process.
// MLX validates conv shapes at graph-build time, so this catches the throw at op
// construction, not only at eval.
std::unique_ptr<MlxArray> try_conv1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv1d(
        input.inner, weight.inner, stride, padding, dilation, groups
    ));
}

// Swap axes (convenient for SSM attention)
std::unique_ptr<MlxArray> swap_axes(const MlxArray& a, int32_t axis1, int32_t axis2) {
    return std::make_unique<MlxArray>(mlx::core::swapaxes(a.inner, axis1, axis2));
}

// Core ops additions.
std::unique_ptr<MlxArray> identity(int32_t n, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::identity(n, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> tri(int32_t n, int32_t m, int32_t k, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::tri(n, m, k, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> unflatten(const MlxArray& a, int32_t axis, rust::Slice<const int32_t> shape) {
    return std::make_unique<MlxArray>(mlx::core::unflatten(a.inner, axis, to_shape(shape)));
}

std::unique_ptr<MlxArray> as_strided(const MlxArray& a, rust::Slice<const int32_t> shape, rust::Slice<const int64_t> strides, size_t offset) {
    std::vector<size_t> strides_vec(strides.begin(), strides.end());
    return std::make_unique<MlxArray>(mlx::core::as_strided(a.inner, to_shape(shape), mlx::core::Strides(strides_vec.begin(), strides_vec.end()), offset));
}

std::unique_ptr<MlxArray> contiguous(const MlxArray& a, bool allow_col_major) {
    return std::make_unique<MlxArray>(mlx::core::contiguous(a.inner, allow_col_major));
}

std::unique_ptr<MlxArray> broadcast_arrays_get(rust::Slice<const MlxArray* const> arrays, size_t index) {
    std::vector<mlx::core::array> inputs;
    inputs.reserve(arrays.size());
    for (const MlxArray* p : arrays) {
        inputs.push_back(p->inner);
    }
    auto result = mlx::core::broadcast_arrays(inputs);
    return std::make_unique<MlxArray>(std::move(result.at(index)));
}

size_t broadcast_arrays_count(rust::Slice<const MlxArray* const> arrays) {
    std::vector<mlx::core::array> inputs;
    inputs.reserve(arrays.size());
    for (const MlxArray* p : arrays) {
        inputs.push_back(p->inner);
    }
    return mlx::core::broadcast_arrays(inputs).size();
}

std::unique_ptr<MlxArray> floor_divide(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::floor_divide(a.inner, b.inner));
}

std::unique_ptr<MlxArray> array_equal(const MlxArray& a, const MlxArray& b, bool equal_nan) {
    return std::make_unique<MlxArray>(mlx::core::array_equal(a.inner, b.inner, equal_nan));
}

std::unique_ptr<MlxArray> allclose(const MlxArray& a, const MlxArray& b, double rtol, double atol) {
    return std::make_unique<MlxArray>(mlx::core::allclose(a.inner, b.inner, rtol, atol));
}

std::unique_ptr<MlxArray> isclose(const MlxArray& a, const MlxArray& b, double rtol, double atol) {
    return std::make_unique<MlxArray>(mlx::core::isclose(a.inner, b.inner, rtol, atol));
}

std::unique_ptr<MlxArray> median_all(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::median(a.inner));
}

std::unique_ptr<MlxArray> median_axis(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::median(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> logcumsumexp(const MlxArray& a, int32_t axis, bool reverse, bool inclusive) {
    return std::make_unique<MlxArray>(mlx::core::logcumsumexp(a.inner, axis, reverse, inclusive));
}

std::unique_ptr<MlxArray> bitwise_invert(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::bitwise_invert(a.inner));
}

std::unique_ptr<MlxArray> real_part(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::real(a.inner));
}

std::unique_ptr<MlxArray> imag_part(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::imag(a.inner));
}

std::unique_ptr<MlxArray> conjugate(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::conjugate(a.inner));
}

std::unique_ptr<MlxArray> view(const MlxArray& a, int32_t dtype) {
    return std::make_unique<MlxArray>(mlx::core::view(a.inner, to_dtype(dtype)));
}

std::unique_ptr<MlxArray> kron(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::kron(a.inner, b.inner));
}

std::unique_ptr<MlxArray> addmm(const MlxArray& c, const MlxArray& a, const MlxArray& b, float alpha, float beta) {
    return std::make_unique<MlxArray>(mlx::core::addmm(c.inner, a.inner, b.inner, alpha, beta));
}

std::unique_ptr<MlxArray> block_masked_mm(
    const MlxArray& a,
    const MlxArray& b,
    int32_t block_size,
    const MlxArray* mask_out,
    const MlxArray* mask_lhs,
    const MlxArray* mask_rhs
) {
    std::optional<mlx::core::array> opt_mask_out = mask_out ? std::make_optional(mask_out->inner) : std::nullopt;
    std::optional<mlx::core::array> opt_mask_lhs = mask_lhs ? std::make_optional(mask_lhs->inner) : std::nullopt;
    std::optional<mlx::core::array> opt_mask_rhs = mask_rhs ? std::make_optional(mask_rhs->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::block_masked_mm(a.inner, b.inner, block_size, opt_mask_out, opt_mask_lhs, opt_mask_rhs));
}

std::unique_ptr<MlxArray> segmented_mm(const MlxArray& a, const MlxArray& b, const MlxArray& segments) {
    return std::make_unique<MlxArray>(mlx::core::segmented_mm(a.inner, b.inner, segments.inner));
}

std::unique_ptr<MlxArray> hadamard_transform(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::hadamard_transform(a.inner));
}

std::unique_ptr<MlxArray> number_of_elements(const MlxArray& a, rust::Slice<const int32_t> axes, bool inverted, int32_t dtype) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::number_of_elements(a.inner, axes_vec, inverted, to_dtype(dtype)));
}

// Convolution additions.
std::unique_ptr<MlxArray> conv3d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_d, int32_t stride_h, int32_t stride_w,
    int32_t padding_d, int32_t padding_h, int32_t padding_w,
    int32_t dilation_d, int32_t dilation_h, int32_t dilation_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv3d(
        input.inner, weight.inner,
        {stride_d, stride_h, stride_w},
        {padding_d, padding_h, padding_w},
        {dilation_d, dilation_h, dilation_w},
        groups
    ));
}

std::unique_ptr<MlxArray> conv_transpose1d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t output_padding,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv_transpose1d(
        input.inner, weight.inner, stride, padding, dilation, output_padding, groups
    ));
}

std::unique_ptr<MlxArray> conv_transpose2d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_h, int32_t stride_w,
    int32_t padding_h, int32_t padding_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t output_padding_h, int32_t output_padding_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv_transpose2d(
        input.inner, weight.inner,
        {stride_h, stride_w},
        {padding_h, padding_w},
        {dilation_h, dilation_w},
        {output_padding_h, output_padding_w},
        groups
    ));
}

std::unique_ptr<MlxArray> conv_transpose3d(
    const MlxArray& input,
    const MlxArray& weight,
    int32_t stride_d, int32_t stride_h, int32_t stride_w,
    int32_t padding_d, int32_t padding_h, int32_t padding_w,
    int32_t dilation_d, int32_t dilation_h, int32_t dilation_w,
    int32_t output_padding_d, int32_t output_padding_h, int32_t output_padding_w,
    int32_t groups
) {
    return std::make_unique<MlxArray>(mlx::core::conv_transpose3d(
        input.inner, weight.inner,
        {stride_d, stride_h, stride_w},
        {padding_d, padding_h, padding_w},
        {dilation_d, dilation_h, dilation_w},
        {output_padding_d, output_padding_h, output_padding_w},
        groups
    ));
}

// Einsum.
std::unique_ptr<MlxArray> einsum(rust::Str subscripts, rust::Slice<const MlxArray* const> operands) {
    std::string subscripts_str(subscripts.begin(), subscripts.end());
    std::vector<mlx::core::array> operands_vec;
    operands_vec.reserve(operands.size());
    for (const MlxArray* p : operands) {
        operands_vec.push_back(p->inner);
    }
    return std::make_unique<MlxArray>(mlx::core::einsum(subscripts_str, operands_vec));
}

// Linear algebra.
std::unique_ptr<MlxArray> linalg_norm(const MlxArray& a, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::linalg::norm(a.inner, axis, keepdims));
}

std::unique_ptr<MlxArray> linalg_norm_ord(const MlxArray& a, double ord, int32_t axis, bool keepdims) {
    return std::make_unique<MlxArray>(mlx::core::linalg::norm(a.inner, ord, axis, keepdims));
}

std::unique_ptr<MlxArray> linalg_norm_str(const MlxArray& a, rust::Str ord, int32_t axis, bool keepdims) {
    std::string ord_str(ord.begin(), ord.end());
    return std::make_unique<MlxArray>(mlx::core::linalg::norm(a.inner, ord_str, axis, keepdims));
}

std::unique_ptr<MlxArray> linalg_qr_q(const MlxArray& a) {
    auto [Q, R] = mlx::core::linalg::qr(a.inner);
    return std::make_unique<MlxArray>(std::move(Q));
}

std::unique_ptr<MlxArray> linalg_qr_r(const MlxArray& a) {
    auto [Q, R] = mlx::core::linalg::qr(a.inner);
    return std::make_unique<MlxArray>(std::move(R));
}

std::unique_ptr<MlxArray> linalg_svd_u(const MlxArray& a) {
    auto result = mlx::core::linalg::svd(a.inner);
    return std::make_unique<MlxArray>(std::move(result[0]));
}

std::unique_ptr<MlxArray> linalg_svd_s(const MlxArray& a) {
    auto result = mlx::core::linalg::svd(a.inner);
    return std::make_unique<MlxArray>(std::move(result[1]));
}

std::unique_ptr<MlxArray> linalg_svd_vt(const MlxArray& a) {
    auto result = mlx::core::linalg::svd(a.inner);
    return std::make_unique<MlxArray>(std::move(result[2]));
}

std::unique_ptr<MlxArray> linalg_inv(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::linalg::inv(a.inner));
}

std::unique_ptr<MlxArray> linalg_pinv(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::linalg::pinv(a.inner));
}

std::unique_ptr<MlxArray> linalg_cholesky(const MlxArray& a, bool upper) {
    return std::make_unique<MlxArray>(mlx::core::linalg::cholesky(a.inner, upper));
}

std::unique_ptr<MlxArray> linalg_solve(const MlxArray& a, const MlxArray& b) {
    return std::make_unique<MlxArray>(mlx::core::linalg::solve(a.inner, b.inner));
}

std::unique_ptr<MlxArray> linalg_solve_triangular(const MlxArray& a, const MlxArray& b, bool upper) {
    return std::make_unique<MlxArray>(mlx::core::linalg::solve_triangular(a.inner, b.inner, upper));
}

std::unique_ptr<MlxArray> linalg_lu_p(const MlxArray& a) {
    auto result = mlx::core::linalg::lu(a.inner);
    return std::make_unique<MlxArray>(std::move(result[0]));
}

std::unique_ptr<MlxArray> linalg_lu_l(const MlxArray& a) {
    auto result = mlx::core::linalg::lu(a.inner);
    return std::make_unique<MlxArray>(std::move(result[1]));
}

std::unique_ptr<MlxArray> linalg_lu_u(const MlxArray& a) {
    auto result = mlx::core::linalg::lu(a.inner);
    return std::make_unique<MlxArray>(std::move(result[2]));
}

std::unique_ptr<MlxArray> linalg_lu_factor_lu(const MlxArray& a) {
    auto [LU, pivots] = mlx::core::linalg::lu_factor(a.inner);
    return std::make_unique<MlxArray>(std::move(LU));
}

std::unique_ptr<MlxArray> linalg_lu_factor_pivots(const MlxArray& a) {
    auto [LU, pivots] = mlx::core::linalg::lu_factor(a.inner);
    return std::make_unique<MlxArray>(std::move(pivots));
}

std::unique_ptr<MlxArray> linalg_eig_values(const MlxArray& a) {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eig(a.inner);
    return std::make_unique<MlxArray>(std::move(eigenvalues));
}

std::unique_ptr<MlxArray> linalg_eig_vectors(const MlxArray& a) {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eig(a.inner);
    return std::make_unique<MlxArray>(std::move(eigenvectors));
}

std::unique_ptr<MlxArray> linalg_eigvals(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::linalg::eigvals(a.inner));
}

std::unique_ptr<MlxArray> linalg_eigh_values(const MlxArray& a) {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eigh(a.inner);
    return std::make_unique<MlxArray>(std::move(eigenvalues));
}

std::unique_ptr<MlxArray> linalg_eigh_vectors(const MlxArray& a) {
    auto [eigenvalues, eigenvectors] = mlx::core::linalg::eigh(a.inner);
    return std::make_unique<MlxArray>(std::move(eigenvectors));
}

std::unique_ptr<MlxArray> linalg_eigvalsh(const MlxArray& a) {
    return std::make_unique<MlxArray>(mlx::core::linalg::eigvalsh(a.inner));
}

std::unique_ptr<MlxArray> linalg_cross(const MlxArray& a, const MlxArray& b, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::linalg::cross(a.inner, b.inner, axis));
}

std::unique_ptr<MlxArray> linalg_tri_inv(const MlxArray& a, bool upper) {
    return std::make_unique<MlxArray>(mlx::core::linalg::tri_inv(a.inner, upper));
}

std::unique_ptr<MlxArray> linalg_cholesky_inv(const MlxArray& a, bool upper) {
    return std::make_unique<MlxArray>(mlx::core::linalg::cholesky_inv(a.inner, upper));
}

// FFT.
std::unique_ptr<MlxArray> fft(const MlxArray& a, int32_t n, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::fft::fftn(a.inner, {n}, {axis}));
}

std::unique_ptr<MlxArray> ifft(const MlxArray& a, int32_t n, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::fft::ifftn(a.inner, {n}, {axis}));
}

std::unique_ptr<MlxArray> rfft(const MlxArray& a, int32_t n, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::fft::rfftn(a.inner, {n}, {axis}));
}

std::unique_ptr<MlxArray> irfft(const MlxArray& a, int32_t n, int32_t axis) {
    return std::make_unique<MlxArray>(mlx::core::fft::irfftn(a.inner, {n}, {axis}));
}

std::unique_ptr<MlxArray> fft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::fftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> ifft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::ifftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> rfft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::rfftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> irfft2(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::irfftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> fftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::fftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> ifftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::ifftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> rfftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::rfftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> irfftn_axes(const MlxArray& a, rust::Slice<const int32_t> n, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::irfftn(a.inner, to_shape(n), axes_vec));
}

std::unique_ptr<MlxArray> fftshift(const MlxArray& a, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::fftshift(a.inner, axes_vec));
}

std::unique_ptr<MlxArray> ifftshift(const MlxArray& a, rust::Slice<const int32_t> axes) {
    std::vector<int> axes_vec(axes.begin(), axes.end());
    return std::make_unique<MlxArray>(mlx::core::fft::ifftshift(a.inner, axes_vec));
}

// Random.
std::unique_ptr<MlxArray> random_key(uint64_t seed) {
    return std::make_unique<MlxArray>(mlx::core::random::key(seed));
}

std::unique_ptr<MlxArray> random_split_key(const MlxArray& key, int32_t num) {
    return std::make_unique<MlxArray>(mlx::core::random::split(key.inner, num));
}

std::unique_ptr<MlxArray> random_uniform(float low, float high, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::uniform(
        mlx::core::array(low), mlx::core::array(high), to_shape(shape), to_dtype(dtype), opt_key
    ));
}

std::unique_ptr<MlxArray> random_normal(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::normal(to_shape(shape), to_dtype(dtype), opt_key));
}

std::unique_ptr<MlxArray> random_bernoulli_p(float p, rust::Slice<const int32_t> shape, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::bernoulli(mlx::core::array(p), to_shape(shape), opt_key));
}

std::unique_ptr<MlxArray> random_randint(int32_t low, int32_t high, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::randint(
        mlx::core::array(low), mlx::core::array(high), to_shape(shape), to_dtype(dtype), opt_key
    ));
}

std::unique_ptr<MlxArray> random_truncated_normal(float lower, float upper, rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::truncated_normal(
        mlx::core::array(lower), mlx::core::array(upper), to_shape(shape), to_dtype(dtype), opt_key
    ));
}

std::unique_ptr<MlxArray> random_gumbel(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::gumbel(to_shape(shape), to_dtype(dtype), opt_key));
}

std::unique_ptr<MlxArray> random_laplace(rust::Slice<const int32_t> shape, int32_t dtype, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::laplace(to_shape(shape), to_dtype(dtype), opt_key));
}

std::unique_ptr<MlxArray> random_permutation(int32_t x, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::permutation(x, opt_key));
}

std::unique_ptr<MlxArray> random_permutation_array(const MlxArray& a, int32_t axis, const MlxArray* key) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::permutation(a.inner, axis, opt_key));
}

std::unique_ptr<MlxArray> random_multivariate_normal(
    const MlxArray& mean,
    const MlxArray& cov,
    rust::Slice<const int32_t> shape,
    int32_t dtype,
    const MlxArray* key
) {
    std::optional<mlx::core::array> opt_key = key ? std::make_optional(key->inner) : std::nullopt;
    return std::make_unique<MlxArray>(mlx::core::random::multivariate_normal(
        mean.inner, cov.inner, to_shape(shape), to_dtype(dtype), opt_key
    ));
}

// Quantization additions.
std::unique_ptr<MlxQuantizedWeights> quantize_weights(const MlxArray& w, int32_t group_size, int32_t bits) {
    return std::make_unique<MlxQuantizedWeights>(
        mlx::core::quantize(w.inner, group_size, bits));
}

std::unique_ptr<MlxQuantizedWeights> quantize_weights_with_mode(const MlxArray& w, int32_t group_size, int32_t bits, rust::Str mode) {
    return std::make_unique<MlxQuantizedWeights>(
        mlx::core::quantize(
            w.inner,
            std::optional<int>(group_size),
            std::optional<int>(bits),
            std::string(mode.data(), mode.size())));
}

std::unique_ptr<MlxArray> quantized_weights_w(const MlxQuantizedWeights& weights) {
    return std::make_unique<MlxArray>(weights.weight);
}

std::unique_ptr<MlxArray> quantized_weights_scales(const MlxQuantizedWeights& weights) {
    return std::make_unique<MlxArray>(weights.scales);
}

bool quantized_weights_has_biases(const MlxQuantizedWeights& weights) {
    return weights.biases.has_value();
}

std::unique_ptr<MlxArray> quantized_weights_biases(const MlxQuantizedWeights& weights) {
    if (!weights.biases.has_value()) {
        return nullptr;
    }
    return std::make_unique<MlxArray>(*weights.biases);
}

std::unique_ptr<MlxArray> quantize_weights_w(const MlxArray& w, int32_t group_size, int32_t bits) {
    auto result = mlx::core::quantize(w.inner, group_size, bits);
    return std::make_unique<MlxArray>(std::move(result[0]));
}

std::unique_ptr<MlxArray> quantize_weights_scales(const MlxArray& w, int32_t group_size, int32_t bits) {
    auto result = mlx::core::quantize(w.inner, group_size, bits);
    return std::make_unique<MlxArray>(std::move(result[1]));
}

std::unique_ptr<MlxArray> quantize_weights_biases(const MlxArray& w, int32_t group_size, int32_t bits) {
    auto result = mlx::core::quantize(w.inner, group_size, bits);
    return std::make_unique<MlxArray>(std::move(result[2]));
}

}  // namespace mlx_cxx
