// Copyright © 2025 Apple Inc.
// Modified by mlxcel: Added cutlass_gather_mm for general GatherMM with both
// lhs_indices and rhs_indices support. Synced to upstream c9aa5605, which
// folds in #3469's Accumulator C-type fix for cutlass half-type matmul.

#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/cuda/gemms/grouped_gemm_arch.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <stdexcept>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

using ProblemSize = cutlass::gemm::GemmCoord;

namespace cu {

namespace cg = cooperative_groups;

template <int N_READS>
__global__ void prepare_grouped_mm_data(
    const uint32_t* indices,
    size_t size,
    int group_count,
    int K,
    int N,
    int lda,
    int ldb,
    int item_size,
    int8_t* a_start,
    int8_t* b_start,
    int8_t* out_start,
    // [mlxcel #629] 64-bit batch strides: with many experts (e.g. 256) or
    // long prompts, `group * item_size * b_batch_stride` and
    // `offset * item_size * a_batch_stride` overflow int32 and produce wild
    // pointers (observed as an illegal memory access on minimax-m2-3bit,
    // E=256, dff*hidden ~ 4.7M elements).
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t out_batch_stride,
    ProblemSize* problem_sizes,
    int64_t* a_lds,
    int64_t* b_lds,
    int64_t* out_lds,
    void** a_ptrs,
    void** b_ptrs,
    void** out_ptrs) {
  auto block = cg::this_thread_block();

  // cumsum(histogram(indices)) - offset for each group.
  extern __shared__ uint32_t cum_histo[];

  int group = block.thread_rank();
  if (group < group_count) {
    cum_histo[group] = 0;
  }

  block.sync();

  // Since |indices| is sorted, the position where element changes would be its
  // cumulative histogram.
  size_t elems_per_block = block.num_threads() * N_READS;
  for (int r = 0; r < cuda::ceil_div(size, elems_per_block); ++r) {
    // TODO: Use vectorized read.
    for (int i = 0; i < N_READS; ++i) {
      size_t pos = r * elems_per_block + group * N_READS + i;
      if (pos >= size) {
        break;
      }
      auto elem = indices[pos];
      auto next = pos < size - 1 ? indices[pos + 1] : group_count;
      while (elem < next) {
        cum_histo[elem] = pos + 1;
        elem++;
      }
    }
  }

  block.sync();

  if (group < group_count) {
    // Fill shapes.
    int delta =
        group == 0 ? cum_histo[0] : cum_histo[group] - cum_histo[group - 1];
    problem_sizes[group] = {delta, N, K};
    a_lds[group] = lda;
    b_lds[group] = ldb;
    out_lds[group] = N;
    // Fill pointers (64-bit arithmetic, see [mlxcel #629] note above).
    int64_t offset = group == 0 ? 0 : cum_histo[group - 1];
    a_ptrs[group] = a_start + offset * item_size * a_batch_stride;
    b_ptrs[group] = b_start + group * item_size * b_batch_stride;
    out_ptrs[group] = out_start + offset * item_size * out_batch_stride;
  }
}

__global__ void prepare_segmented_mm_data(
    const uint32_t* segments,
    int num_segments,
    int M,
    int N,
    int lda,
    int ldb,
    int item_size,
    bool a_transposed,
    bool b_transposed,
    int8_t* a_start,
    int8_t* b_start,
    int8_t* out_start,
    ProblemSize* problem_sizes,
    int64_t* a_lds,
    int64_t* b_lds,
    int64_t* out_lds,
    void** a_ptrs,
    void** b_ptrs,
    void** out_ptrs) {
  int idx = cg::this_grid().thread_rank();
  if (idx >= num_segments)
    return;

  int64_t start = segments[2 * idx];
  int64_t end = segments[2 * idx + 1];
  int K_i = (end > start) ? static_cast<int>(end - start) : 0;

  problem_sizes[idx] = {M, N, K_i};
  a_lds[idx] = lda;
  b_lds[idx] = ldb;
  out_lds[idx] = N;

  // Offset into K dimension depends on layout:
  // A [M,K]: row-major offset = start, col-major offset = start * lda
  // B [K,N]: row-major offset = start * ldb, col-major offset = start
  int64_t a_offset = a_transposed ? start * lda : start;
  int64_t b_offset = b_transposed ? start : start * ldb;

  a_ptrs[idx] = a_start + a_offset * item_size;
  b_ptrs[idx] = b_start + b_offset * item_size;
  out_ptrs[idx] = out_start + static_cast<int64_t>(idx) * M * N * item_size;
}

// [mlxcel] General gather matmul: each output batch element i uses
// A[lhs_indices[i]] and B[rhs_indices[i]], all with the same M x N x K shape.
__global__ void prepare_gather_mm_general_data(
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int batch_size,
    int M,
    int N,
    int K,
    int64_t lda,
    int64_t ldb,
    int item_size,
    int8_t* a_start,
    int8_t* b_start,
    int8_t* out_start,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    ProblemSize* problem_sizes,
    int64_t* a_lds,
    int64_t* b_lds,
    int64_t* out_lds,
    void** a_ptrs,
    void** b_ptrs,
    void** out_ptrs) {
  int i = cg::this_grid().thread_rank();
  if (i >= batch_size)
    return;

  uint32_t lhs_idx = lhs_indices[i];
  uint32_t rhs_idx = rhs_indices[i];

  problem_sizes[i] = {M, N, K};
  a_lds[i] = lda;
  b_lds[i] = ldb;
  out_lds[i] = N;

  a_ptrs[i] = a_start + static_cast<int64_t>(lhs_idx) * item_size * a_batch_stride;
  b_ptrs[i] = b_start + static_cast<int64_t>(rhs_idx) * item_size * b_batch_stride;
  out_ptrs[i] = out_start + static_cast<int64_t>(i) * M * N * item_size;
}

} // namespace cu

namespace {

// Shared GEMM configuration for every type and arch.
template <typename T, typename ArchTag, int kAlignmentC>
struct CommonGemmConfiguration {
  using Element = T;
  using Arch = ArchTag;
  using Accumulator = std::conditional_t<(sizeof(T) < 4), float, T>;
  using EpilogueOutputOp = cutlass::epilogue::thread::
      LinearCombination<T, kAlignmentC, Accumulator, Accumulator>;
};

// Slow GEMM configuration as fallback.
template <
    typename T,
    typename Arch,
    int kAlignmentC = 1,
    bool kEnableTF32 = false,
    typename Enable = void>
struct GemmConfiguration : public CommonGemmConfiguration<T, Arch, 1> {
  using OpClass = cutlass::arch::OpClassSimt;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static const int kAlignmentAB = 1;
  static const int kStages = 2;
};

// Specialized GEMM configuration for sm80 and later.
template <typename T, typename Arch, int kAlignmentC>
struct GemmConfiguration<
    T,
    Arch,
    kAlignmentC,
    true,
    std::enable_if_t<Arch::kMinComputeCapability >= 80 && sizeof(T) <= 4>>
    : public CommonGemmConfiguration<T, cutlass::arch::Sm80, kAlignmentC> {
  using OpClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32 / sizeof(T)>;
  static const int kAlignmentAB = 1;
  static const int kStages = 2;
};

// Specialized GEMM configuration for tf32 on sm80.
template <int kAlignmentC>
struct GemmConfiguration<float, cutlass::arch::Sm80, kAlignmentC, true>
    : public CommonGemmConfiguration<float, cutlass::arch::Sm80, kAlignmentC> {
  using OpClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static const int kAlignmentAB = 1;
  static const int kStages = 3; // use SM80_CP_ASYNC
};

// [mlxcel #1544] The pre-Ampere arm of `dispatch_cutlass_arch` tags every part
// below compute capability 8.0 with `cutlass::arch::Sm70`, Turing included.
// That holds only while the configuration the arm selects stays SIMT: with
// `OpClassSimt` and `InstructionShape<1, 1, 1>` there is no MMA atom for the
// tag to choose, CUTLASS erases the tag, and one arm can serve 7.0 through
// 7.5. Give the pre-Ampere arm a tensor-core operator and the tag becomes
// load-bearing, at which point a Turing part tagged `Sm70` silently loses
// `m16n8k8` and Turing needs its own arm again. These two assertions fail the
// build on that day instead. The second one carries `kEnableTF32 = true`,
// which is the arm `MLX_ENABLE_TF32` reaches, and is what rules out a
// pre-Ampere tag ever selecting one of the two tensor-core specializations
// above: both are constrained on `Arch::kMinComputeCapability >= 80`.
static_assert(
    std::is_same_v<
        GemmConfiguration<float, cutlass::arch::Sm70, 1, false>::OpClass,
        cutlass::arch::OpClassSimt>,
    "pre-Ampere grouped GEMM is no longer SIMT: the architecture tag in "
    "dispatch_cutlass_arch became load-bearing, so Turing needs its own arm "
    "again (see gemms/grouped_gemm_arch.h)");
static_assert(
    std::is_same_v<
        GemmConfiguration<float, cutlass::arch::Sm70, 8, true>::OpClass,
        cutlass::arch::OpClassSimt>,
    "a pre-Ampere tag selected a tensor-core configuration under "
    "MLX_ENABLE_TF32 (see gemms/grouped_gemm_arch.h)");

// [mlxcel #1544] `cp.async` arrived with Ampere. The 3-stage pipeline above is
// commented "use SM80_CP_ASYNC" and is bound to `cutlass::arch::Sm80` by an
// explicit full specialization, so no pre-Ampere tag can reach it; a
// pre-Ampere configuration stays at the 2 stages `MmaPipelined` implements
// with ordinary global-to-shared copies. Asserted rather than reasoned about,
// because a 3-stage pipeline without `cp.async` is either a build failure or a
// silent serialization and neither announces itself.
static_assert(
    GemmConfiguration<float, cutlass::arch::Sm70, 8, true>::kStages == 2,
    "a pre-Ampere grouped GEMM configuration asked for a 3-stage pipeline, "
    "which needs the cp.async that Ampere introduced");

// Get direct access to kernel.
template <typename GemmKernel>
class GemmGroupedEncoder
    : public cutlass::gemm::device::GemmGrouped<GemmKernel> {
 public:
  void encode(cu::CommandEncoder& encoder) {
    encoder.add_kernel_node_ex(
        cutlass::Kernel<GemmKernel>,
        {static_cast<uint32_t>(this->params_.threadblock_count), 1, 1},
        {GemmKernel::kThreadCount, 1, 1},
        {},
        sizeof(typename GemmKernel::SharedStorage),
        this->params_);
  }
};

// Invoke the grouped GEMM of CUTLASS 2.x API, which supports small alignments.
template <typename GemmConfiguration>
void grouped_gemm_v2(
    bool a_transposed,
    bool b_transposed,
    int group_count,
    ProblemSize* problem_sizes,
    int64_t* a_lds,
    int64_t* b_lds,
    int64_t* out_lds,
    void* a_ptrs,
    void* b_ptrs,
    void* out_ptrs,
    cu::CommandEncoder& encoder) {
  dispatch_bool(a_transposed, [&](auto a_transposed_tag) {
    dispatch_bool(b_transposed, [&](auto b_transposed_tag) {
      using LayoutA = std::conditional_t<
          a_transposed_tag.value,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>;
      using LayoutB = std::conditional_t<
          b_transposed_tag.value,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>;
      using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
          typename GemmConfiguration::Element,
          LayoutA,
          cutlass::ComplexTransform::kNone,
          GemmConfiguration::kAlignmentAB,
          typename GemmConfiguration::Element,
          LayoutB,
          cutlass::ComplexTransform::kNone,
          GemmConfiguration::kAlignmentAB,
          // [mlxcel] Sync with upstream #3469: use Accumulator for the C
          // operand type so cutlass picks the correct half-precision GEMM
          // configuration (was Element pre-c9aa5605).
          typename GemmConfiguration::Accumulator,
          cutlass::layout::RowMajor,
          typename GemmConfiguration::Accumulator,
          typename GemmConfiguration::OpClass,
          typename GemmConfiguration::Arch,
          typename GemmConfiguration::ThreadblockShape,
          typename GemmConfiguration::WarpShape,
          typename GemmConfiguration::InstructionShape,
          typename GemmConfiguration::EpilogueOutputOp,
          cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
          GemmConfiguration::kStages>::GemmKernel;
      using GemmGrouped = GemmGroupedEncoder<GemmKernel>;

      static int threadblock_count = GemmGrouped::sufficient();
      typename GemmGrouped::Arguments args(
          problem_sizes,
          group_count,
          threadblock_count,
          {/* alpha */ 1, /* beta */ 0},
          reinterpret_cast<typename GemmGrouped::ElementA**>(a_ptrs),
          reinterpret_cast<typename GemmGrouped::ElementB**>(b_ptrs),
          reinterpret_cast<typename GemmGrouped::ElementC**>(out_ptrs),
          reinterpret_cast<typename GemmGrouped::ElementC**>(out_ptrs),
          a_lds,
          b_lds,
          out_lds,
          out_lds);

      GemmGrouped gemm;
      CHECK_CUTLASS_ERROR(gemm.initialize(
          args,
          allocate_workspace(encoder, gemm.get_workspace_size(args)),
          encoder.stream()));
      gemm.encode(encoder);
    });
  });
}

// [mlxcel #1544] The architecture decision itself lives in
// `gemms/grouped_gemm_arch.h` as a pure function of the compute capability
// major version, so it can be enumerated over every architecture on a host
// with no GPU (`grouped_gemm_arch_tests.rs`, through
// `cpp/grouped_gemm_arch_probe.cpp`). Upstream mapped every pre-Ampere part to
// `cutlass::arch::Sm75`, which names Turing and so described hardware that a
// compute capability 7.0 part does not have. That header records why the
// corrected tag is `Sm70` for the whole pre-Ampere arm, what it measured to
// establish that the retag moves no device code, and what has to change before
// Turing needs an arm of its own.
template <typename F>
void dispatch_cutlass_arch(cu::Device& device, F&& f) {
  switch (mlxcel::grouped_gemm_arch_for(device.compute_capability_major())) {
    case mlxcel::GroupedGemmArch::Sm70:
      f(type_identity<cutlass::arch::Sm70>{});
      return;
    case mlxcel::GroupedGemmArch::Sm80:
      f(type_identity<cutlass::arch::Sm80>{});
      return;
    case mlxcel::GroupedGemmArch::Sm90:
      f(type_identity<cutlass::arch::Sm90>{});
      return;
  }
}

// The signature every `grouped_gemm_v2` instantiation shares, named so the
// selection below can start from no kernel at all rather than from a
// placeholder instantiation.
using GroupedGemmFn = void (*)(
    bool,
    bool,
    int,
    ProblemSize*,
    int64_t*,
    int64_t*,
    int64_t*,
    void*,
    void*,
    void*,
    cu::CommandEncoder&);

GroupedGemmFn get_grouped_mm_funcion(Dtype dtype, int N, cu::Device& device) {
  // [mlxcel #1544] This was
  // `grouped_gemm_v2<GemmConfiguration<float, cutlass::arch::Sm75>>`, a
  // placeholder that reads as a pre-Ampere default and names Turing on parts
  // that are not Turing. It was never the value returned, because
  // `dispatch_float_types` throws on a non-float dtype and every float dtype
  // assigns `fun`, but it did force one more template instantiation into the
  // binary purely to serve as an initializer. `nullptr` says what the code
  // means and cannot be misread as an architecture decision.
  GroupedGemmFn fun = nullptr;
  dispatch_float_types(dtype, "grouped_gemm_v2", [&](auto type_tag) {
    using DataType = cutlass_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_cutlass_arch(device, [&](auto arch_tag) {
      using Arch = MLX_GET_TYPE(arch_tag);
      dispatch_bool(N % 8 == 0, [&](auto is_out_aligned) {
        constexpr int kAlignmentC = is_out_aligned ? 8 : 1;
        dispatch_bool(env::enable_tf32(), [&](auto kEnableTF32) {
          fun = grouped_gemm_v2<
              GemmConfiguration<DataType, Arch, kAlignmentC, kEnableTF32>>;
        });
      });
    });
  });
  // Unreachable while `dispatch_float_types` throws on every dtype it does not
  // dispatch, which is the contract today. Kept so that a future widening of
  // that dispatch surfaces as a named error instead of a null call.
  if (fun == nullptr) {
    throw std::runtime_error(
        "[grouped_gemm_v2] no grouped GEMM kernel was selected for the "
        "requested dtype");
  }
  return fun;
}

} // namespace

void cutlass_grouped_gemm_unaligned(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int group_count,
    const array& a,
    const array& b,
    const array& indices,
    array& out,
    cu::CommandEncoder& encoder) {
  int K = a.shape(-1);
  int N = b.shape(-1);

  // Prepare device pointers for matmul.
  int problem_sizes_nbytes =
      group_count * cuda::ceil_div(sizeof(ProblemSize), 8) * 8;
  int nbytes = problem_sizes_nbytes +
      group_count * (3 * sizeof(void*) + 3 * sizeof(int64_t));
  nbytes = cuda::ceil_div(nbytes, 256) * 256;
  array gemm_args(cu::malloc_async(nbytes, encoder), {nbytes}, int8);
  encoder.add_temporary(gemm_args);

  ProblemSize* problem_sizes = gpu_ptr<ProblemSize>(gemm_args);
  int64_t* a_lds = gpu_ptr<int64_t>(gemm_args) + problem_sizes_nbytes / 8;
  int64_t* b_lds = a_lds + group_count;
  int64_t* out_lds = b_lds + group_count;
  void** a_ptrs = reinterpret_cast<void**>(out_lds + group_count);
  void** b_ptrs = a_ptrs + group_count;
  void** out_ptrs = b_ptrs + group_count;

  // Fill the pointers by computing offsets from indices.
  constexpr int N_READS = 4;
  int n_threads = cuda::ceil_div(indices.size(), N_READS);
  n_threads = group_count < n_threads ? n_threads : group_count;
  dim3 block_dims(std::min(n_threads, 1024));
  dim3 num_blocks(1);

  encoder.set_input_array(indices);
  encoder.set_output_array(gemm_args);
  encoder.add_kernel_node_ex(
      cu::prepare_grouped_mm_data<N_READS>,
      num_blocks,
      block_dims,
      {},
      group_count * sizeof(uint32_t), // sizeof(cum_histo)
      gpu_ptr<uint32_t>(indices),
      indices.size(),
      group_count,
      K,
      N,
      lda,
      ldb,
      out.itemsize(),
      gpu_ptr<int8_t>(a),
      gpu_ptr<int8_t>(b),
      gpu_ptr<int8_t>(out),
      static_cast<int64_t>(a.shape(-2)) * a.shape(-1), // a_batch_stride
      static_cast<int64_t>(b.shape(-2)) * b.shape(-1), // b_batch_stride
      static_cast<int64_t>(out.shape(-2)) * out.shape(-1), // out_batch_stride
      problem_sizes,
      a_lds,
      b_lds,
      out_lds,
      a_ptrs,
      b_ptrs,
      out_ptrs);

  // Invoke grouped GEMM.
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(gemm_args);
  encoder.set_output_array(out);
  auto* fun = get_grouped_mm_funcion(a.dtype(), N, encoder.device());
  fun(a_transposed,
      b_transposed,
      group_count,
      problem_sizes,
      a_lds,
      b_lds,
      out_lds,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      encoder);
}

void cutlass_segmented_mm(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int num_segments,
    int M,
    int N,
    const array& a,
    const array& b,
    const array& segments,
    array& out,
    cu::CommandEncoder& encoder) {
  // Allocate grouped GEMM args on device.
  int problem_sizes_nbytes =
      num_segments * cuda::ceil_div(sizeof(ProblemSize), 8) * 8;
  int nbytes = problem_sizes_nbytes +
      num_segments * (3 * sizeof(void*) + 3 * sizeof(int64_t));
  nbytes = cuda::ceil_div(nbytes, 256) * 256;
  array gemm_args(cu::malloc_async(nbytes, encoder), {nbytes}, int8);
  encoder.add_temporary(gemm_args);

  ProblemSize* problem_sizes = gpu_ptr<ProblemSize>(gemm_args);
  int64_t* a_lds = gpu_ptr<int64_t>(gemm_args) + problem_sizes_nbytes / 8;
  int64_t* b_lds = a_lds + num_segments;
  int64_t* out_lds = b_lds + num_segments;
  void** a_ptrs = reinterpret_cast<void**>(out_lds + num_segments);
  void** b_ptrs = a_ptrs + num_segments;
  void** out_ptrs = b_ptrs + num_segments;

  // Build problem descriptions from segments on the GPU.
  int block_size = std::min(num_segments, 256);
  int num_blocks = cuda::ceil_div(num_segments, block_size);

  encoder.set_input_array(segments);
  encoder.set_output_array(gemm_args);
  encoder.add_kernel_node_ex(
      cu::prepare_segmented_mm_data,
      dim3(num_blocks),
      dim3(block_size),
      {},
      0,
      gpu_ptr<uint32_t>(segments),
      num_segments,
      M,
      N,
      static_cast<int>(lda),
      static_cast<int>(ldb),
      static_cast<int>(out.itemsize()),
      a_transposed,
      b_transposed,
      gpu_ptr<int8_t>(a),
      gpu_ptr<int8_t>(b),
      gpu_ptr<int8_t>(out),
      problem_sizes,
      a_lds,
      b_lds,
      out_lds,
      a_ptrs,
      b_ptrs,
      out_ptrs);

  // Dispatch grouped GEMM.
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(gemm_args);
  encoder.set_output_array(out);
  auto* fun = get_grouped_mm_funcion(a.dtype(), N, encoder.device());
  fun(a_transposed,
      b_transposed,
      num_segments,
      problem_sizes,
      a_lds,
      b_lds,
      out_lds,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      encoder);
}

// [mlxcel] General gather matmul: handles arbitrary lhs_indices and rhs_indices.
// Each output batch element i computes: out[i] = A[lhs_indices[i]] @ B[rhs_indices[i]]
// Uses the same CUTLASS grouped GEMM infrastructure as cutlass_grouped_gemm_unaligned,
// but with a simpler GPU-side pointer preparation kernel.
void cutlass_gather_mm(
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    int M,
    int N,
    int K,
    const array& a,
    const array& b,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    cu::CommandEncoder& encoder) {
  nvtx3::scoped_range r("cutlass_gather_mm");

  int batch_size = static_cast<int>(out.size() / (M * N));

  // Allocate grouped GEMM metadata on device.
  using ProblemSize = cutlass::gemm::GemmCoord;
  int problem_sizes_nbytes =
      batch_size * cuda::ceil_div(sizeof(ProblemSize), 8) * 8;
  int nbytes = problem_sizes_nbytes +
      batch_size * (3 * sizeof(void*) + 3 * sizeof(int64_t));
  nbytes = cuda::ceil_div(nbytes, 256) * 256;
  array gemm_args(cu::malloc_async(nbytes, encoder), {nbytes}, int8);
  encoder.add_temporary(gemm_args);

  ProblemSize* problem_sizes = gpu_ptr<ProblemSize>(gemm_args);
  int64_t* a_lds = gpu_ptr<int64_t>(gemm_args) + problem_sizes_nbytes / 8;
  int64_t* b_lds = a_lds + batch_size;
  int64_t* out_lds = b_lds + batch_size;
  void** a_ptrs = reinterpret_cast<void**>(out_lds + batch_size);
  void** b_ptrs = a_ptrs + batch_size;
  void** out_ptrs = b_ptrs + batch_size;

  // Compute batch strides for A and B (elements per expert matrix)
  int64_t a_batch_stride = a.shape(-2) * a.shape(-1);
  int64_t b_batch_stride = b.shape(-2) * b.shape(-1);

  // GPU-side pointer preparation: reads indices and sets up pointer arrays
  int block_size = std::min(batch_size, 256);
  int num_blocks = cuda::ceil_div(batch_size, block_size);

  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(gemm_args);
  encoder.add_kernel_node(
      cu::prepare_gather_mm_general_data,
      dim3(num_blocks),
      dim3(block_size),
      gpu_ptr<uint32_t>(lhs_indices),
      gpu_ptr<uint32_t>(rhs_indices),
      batch_size,
      M,
      N,
      K,
      lda,
      ldb,
      static_cast<int>(out.itemsize()),
      gpu_ptr<int8_t>(a),
      gpu_ptr<int8_t>(b),
      gpu_ptr<int8_t>(out),
      a_batch_stride,
      b_batch_stride,
      problem_sizes,
      a_lds,
      b_lds,
      out_lds,
      a_ptrs,
      b_ptrs,
      out_ptrs);

  // Invoke CUTLASS grouped GEMM.
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(gemm_args);
  encoder.set_output_array(out);
  auto* fun = get_grouped_mm_funcion(a.dtype(), N, encoder.device());
  fun(a_transposed,
      b_transposed,
      batch_size,
      problem_sizes,
      a_lds,
      b_lds,
      out_lds,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      encoder);
}

} // namespace mlx::core
