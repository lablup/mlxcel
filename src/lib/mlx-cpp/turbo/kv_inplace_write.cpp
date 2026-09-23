// Copyright 2025-2026 Lablup Inc.
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

#include "kv_inplace_write.h"

#include <mlx/backend/common/slicing.h>
#include <mlx/backend/gpu/copy.h>
#include <mlx/primitives.h>
#include <mlx/utils.h>

#include <memory>
#include <sstream>
#include <stdexcept>

namespace mlxcel::turbo {

namespace mx = mlx::core;

namespace {

// Output shares input 0's buffer; input 1 is copied into it at `start_`.
class InplaceSliceWrite : public mx::UnaryPrimitive {
 public:
  InplaceSliceWrite(mx::Stream stream, mx::Shape start)
      : mx::UnaryPrimitive(stream), start_(std::move(start)) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error(
        "[inplace_slice_write] GPU only; use slice_update on the CPU");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const auto& dst = inputs[0];
    const auto& rows = inputs[1];
    // Adopt dst's buffer unconditionally: this is the whole point, and the
    // caller guarantees no reader needs the overwritten region's old value.
    out.copy_shared_buffer(dst);
    if (rows.size() == 0) {
      return;
    }
    mx::Shape unit_strides(start_.size(), 1);
    auto [data_offset, out_strides] =
        mx::prepare_slice(out, start_, unit_strides);
    mx::copy_gpu_inplace(
        /* in = */ rows,
        /* out = */ out,
        /* data_shape = */ rows.shape(),
        /* i_strides = */ rows.strides(),
        /* o_strides = */ out_strides,
        /* i_offset = */ 0,
        /* o_offset = */ data_offset,
        mx::CopyType::GeneralGeneral,
        stream());
  }

  const char* name() const override {
    return "InplaceSliceWrite";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const InplaceSliceWrite&>(other);
    return start_ == o.start_;
  }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape()};
  }

 private:
  mx::Shape start_;
};

} // namespace

mx::array inplace_slice_write(
    const mx::array& dst,
    const mx::array& rows,
    const std::vector<int>& start) {
  if (dst.ndim() != rows.ndim() || start.size() != dst.ndim()) {
    throw std::invalid_argument(
        "[inplace_slice_write] dst, rows and start must share a rank");
  }
  if (dst.dtype() != rows.dtype()) {
    throw std::invalid_argument(
        "[inplace_slice_write] dst and rows must share a dtype");
  }
  for (size_t i = 0; i < start.size(); ++i) {
    if (start[i] < 0 || start[i] + rows.shape(i) > dst.shape(i)) {
      std::ostringstream msg;
      msg << "[inplace_slice_write] rows " << rows.shape() << " at start "
          << mx::Shape(start.begin(), start.end()) << " do not fit in "
          << dst.shape();
      throw std::invalid_argument(msg.str());
    }
  }
  auto stream = mx::default_stream(mx::default_device());
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument(
        "[inplace_slice_write] requires the GPU as the default device");
  }
  return mx::array(
      dst.shape(),
      dst.dtype(),
      std::make_shared<InplaceSliceWrite>(
          stream, mx::Shape(start.begin(), start.end())),
      {dst, rows});
}

} // namespace mlxcel::turbo
