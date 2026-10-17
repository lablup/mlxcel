// Copyright © 2023-2024 Apple Inc.
//
// CUDA patch: bf16 type promotion preservation
//
// Modified from upstream MLX ce45c525 mlx/dtype.cpp
//
// Changes to the type_rules promotion table:
//   - bf16 + float32 → bf16  (was float32)
//   - float32 + bf16 → bf16  (was float32)
//
// Not changed: bf16 + float16 and float16 + bf16 stay at float32, as upstream
// (see the NOTE above the table). A bf16 model over an fp16 TurboQuant KV
// cache therefore attends in float32; `layers::tests` pins this through the
// `sdpa_attention_dtype` bridge (issue #2422). Because bf16 + float32 is
// patched, a left-to-right fold over three dtypes is order-dependent:
// result_type(bf16, fp16, fp16) is float32 but result_type(fp16, bf16, bf16)
// is bf16 (fp16 + bf16 -> float32, then float32 + bf16 -> bf16).
//
// Rationale: On CUDA, bf16 is the native compute type for LLM inference.
// The standard JAX promotion rules (bf16 + fp32 → fp32) trigger copy_v<bf16, float>
// conversion kernels at every binary operation boundary, causing ~58% GPU time
// overhead on 9B models. Since fp32 operands in the LLM compute graph are typically
// just scalars (normalization factors, constants), preserving bf16 avoids these
// costly type conversion round-trips with negligible precision impact.
//
// This patch is CUDA-only — applied via the overlay system in CMakeLists.txt
// only when MLX_BUILD_CUDA is set. Metal builds use the unmodified upstream file.

#include <cstdint>

#include "mlx/dtype.h"

namespace mlx::core {

namespace {

constexpr int num_types = 14;
constexpr int num_cats = 8;

constexpr Dtype::Kind type_kinds[num_types] = {
    Dtype::Kind::b, // bool_,
    Dtype::Kind::u, // uint8,
    Dtype::Kind::u, // uint16,
    Dtype::Kind::u, // uint32,
    Dtype::Kind::u, // uint64,
    Dtype::Kind::i, // int8,
    Dtype::Kind::i, // int16,
    Dtype::Kind::i, // int32,
    Dtype::Kind::i, // int64,
    Dtype::Kind::f, // float16,
    Dtype::Kind::f, // float32,
    Dtype::Kind::f, // float64,
    Dtype::Kind::V, // bfloat16,
    Dtype::Kind::c // complex64,
};

// Based on Jax type promotion rules with CUDA-specific bf16 preservation:
// https://jax.readthedocs.io/en/latest/type_promotion.html
//
// CUDA modifications (marked with /* PATCHED */):
//   bfloat16 row: float32 column changed from float32 → bfloat16
//
// NOTE: Only the bf16+fp32 promotion is changed (bf16 row, fp32 col and
// fp32 row, bf16 col). The fp16+bf16 promotions are left at float32
// to avoid breaking float16 models (e.g., quantized Qwen-MoE, DeepSeek).
// clang-format off
constexpr Dtype type_rules[num_types][num_types] = {
// bool       uint8      uint16     uint32     uint64     int8       int16      int32      int64      float16    float32   float64    bfloat16   complex64
  {bool_,     uint8,     uint16,    uint32,    uint64,    int8,      int16,     int32,     int64,     float16,   float32,  float64,   bfloat16,  complex64}, // bool
  {uint8,     uint8,     uint16,    uint32,    uint64,    int16,     int16,     int32,     int64,     float16,   float32,  float64,   bfloat16,  complex64}, // uint8
  {uint16,    uint16,    uint16,    uint32,    uint64,    int32,     int32,     int32,     int64,     float16,   float32,  float64,   bfloat16,  complex64}, // uint16
  {uint32,    uint32,    uint32,    uint32,    uint64,    int64,     int64,     int64,     int64,     float16,   float32,  float64,   bfloat16,  complex64}, // uint32
  {uint64,    uint64,    uint64,    uint64,    uint64,    float32,   float32,   float32,   float32,   float16,   float32,  float64,   bfloat16,  complex64}, // uint64
  {int8,      int16,     int32,     int64,     float32,   int8,      int16,     int32,     int64,     float16,   float32,  float64,   bfloat16,  complex64}, // int8
  {int16,     int16,     int32,     int64,     float32,   int16,     int16,     int32,     int64,     float16,   float32,  float64,   bfloat16,  complex64}, // int16
  {int32,     int32,     int32,     int64,     float32,   int32,     int32,     int32,     int64,     float16,   float32,  float64,   bfloat16,  complex64}, // int32
  {int64,     int64,     int64,     int64,     float32,   int64,     int64,     int64,     int64,     float16,   float32,  float64,   bfloat16,  complex64}, // int64
  {float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float16,   float32,  float64,   float32,   complex64}, // float16
  {float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,   float32,  float64,   bfloat16,  complex64}, // float32  /* PATCHED: bf16 col only */
  {float64,   float64,   float64,   float64,   float64,   float64,   float64,   float64,   float64,   float64,   float64,  float64,   float64,   complex64}, // float64
  {bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  bfloat16,  float32,   bfloat16, float64,   bfloat16,  complex64}, // bfloat16 /* PATCHED: fp32 col only, fp16 stays float32 */
  {complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64, complex64,complex64, complex64, complex64}, // complex64
};


constexpr bool subcategory_to_category[num_cats][num_cats] = {
// complexfloating floating inexact signedinteger unsignedinteger integer number generic
  {true,           false,   true,   false,        false,          false,  true,  true}, // complexfloating
  {false,          true,    true,   false,        false,          false,  true,  true}, // floating
  {false,          false,   true,   false,        false,          false,  true,  true}, // inexact
  {false,          false,   false,  true,         false,          true,   true,  true}, // signedinteger
  {false,          false,   false,  false,        true,           true,   true,  true}, // unsignedinteger
  {false,          false,   false,  false,        false,          true,   true,  true}, // integer
  {false,          false,   false,  false,        false,          false,  true,  true}, // number
  {false,          false,   false,  false,        false,          false,  false, true}, // generic
};

constexpr Dtype::Category type_to_category[num_types] = {
    Dtype::Category::generic, // bool_,
    Dtype::Category::unsignedinteger, // uint8,
    Dtype::Category::unsignedinteger, // uint16,
    Dtype::Category::unsignedinteger, // uint32,
    Dtype::Category::unsignedinteger, // uint64,
    Dtype::Category::signedinteger, // int8,
    Dtype::Category::signedinteger, // int16,
    Dtype::Category::signedinteger, // int32,
    Dtype::Category::signedinteger, // int64,
    Dtype::Category::floating, // float16,
    Dtype::Category::floating, // float32,
    Dtype::Category::floating, // float64,
    Dtype::Category::floating, // bfloat16,
    Dtype::Category::complexfloating, // complex64,
};

// clang-format on

} // namespace

Dtype promote_types(const Dtype& t1, const Dtype& t2) {
  return Dtype(
      type_rules[static_cast<int>(t1.val())][static_cast<int>(t2.val())]);
}

Dtype::Kind kindof(const Dtype& t) {
  return type_kinds[static_cast<int>(t.val())];
}

template class MLX_API TypeToDtype<bool>;
template class MLX_API TypeToDtype<uint8_t>;
template class MLX_API TypeToDtype<uint16_t>;
template class MLX_API TypeToDtype<uint32_t>;
template class MLX_API TypeToDtype<uint64_t>;
template class MLX_API TypeToDtype<int8_t>;
template class MLX_API TypeToDtype<int16_t>;
template class MLX_API TypeToDtype<int32_t>;
template class MLX_API TypeToDtype<int64_t>;
template class MLX_API TypeToDtype<float16_t>;
template class MLX_API TypeToDtype<float>;
template class MLX_API TypeToDtype<double>;
template class MLX_API TypeToDtype<bfloat16_t>;
template class MLX_API TypeToDtype<complex64_t>;

template <>
TypeToDtype<bool>::operator Dtype() {
  return bool_;
}

template <>
TypeToDtype<uint8_t>::operator Dtype() {
  return uint8;
}

template <>
TypeToDtype<uint16_t>::operator Dtype() {
  return uint16;
}

template <>
TypeToDtype<uint32_t>::operator Dtype() {
  return uint32;
}

template <>
TypeToDtype<uint64_t>::operator Dtype() {
  return uint64;
}

template <>
TypeToDtype<int8_t>::operator Dtype() {
  return int8;
}

template <>
TypeToDtype<int16_t>::operator Dtype() {
  return int16;
}

template <>
TypeToDtype<int32_t>::operator Dtype() {
  return int32;
}

template <>
TypeToDtype<int64_t>::operator Dtype() {
  return int64;
}

template <>
TypeToDtype<float16_t>::operator Dtype() {
  return float16;
}

template <>
TypeToDtype<float>::operator Dtype() {
  return float32;
}

template <>
TypeToDtype<double>::operator Dtype() {
  return float32;
}

template <>
TypeToDtype<bfloat16_t>::operator Dtype() {
  return bfloat16;
}

template <>
TypeToDtype<complex64_t>::operator Dtype() {
  return complex64;
}

bool issubdtype(const Dtype& a, const Dtype& b) {
  return a == b;
}

bool issubdtype(const Dtype::Category& cat, const Dtype& type) {
  return false;
}

bool issubdtype(const Dtype& type, const Dtype::Category& cat) {
  return issubdtype(type_to_category[static_cast<uint32_t>(type.val())], cat);
}

bool issubdtype(const Dtype::Category& a, const Dtype::Category& b) {
  return subcategory_to_category[static_cast<uint32_t>(a)]
                                [static_cast<uint32_t>(b)];
}

} // namespace mlx::core
