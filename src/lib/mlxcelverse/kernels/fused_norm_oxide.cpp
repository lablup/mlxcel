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

// Compiled only when the build sets MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1
// (Linux CUDA); contract in `fused_norm_oxide.h`, gate in `mlxcel-core/build.rs`.

#include "fused_norm_oxide.h"

#include <mlx/backend/cuda/allocator.h>
#include <mlx/backend/cuda/device.h>
#include <mlx/backend/cuda/jit_module.h>
#include <mlx/backend/gpu/copy.h>
#include <mlx/primitives.h>
#include <mlx/utils.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace mlxcel::turbo {

namespace {

// What each kernel parameter carries. The values are the wire format of
// `mlxcel_oxide_fused_norm_install`'s `roles` argument and are mirrored by
// `ArgRole` in `mlxcel-core/src/cuda_oxide_fused_norm.rs`, which derives the
// per-entry order from the artifact's manifest instead of assuming it.
enum Role : std::uint8_t {
    kXPtr = 0,
    kXLen = 1,
    kResidualPtr = 2,
    kResidualLen = 3,
    kWeightPtr = 4,
    kWeightLen = 5,
    kEps = 6,
    kWeightBias = 7,
    kDim = 8,
    kNormedPtr = 9,
    kNormedLen = 10,
    kNewResidualPtr = 11,
    kNewResidualLen = 12,
    kRoleCount = 13,
};

// Why a call took the existing port. `kTaken` means it did not. Mirrored by
// `Fallback` on the Rust side.
enum Fallback : int {
    kTaken = 0,
    kNotInstalled = 1,
    kDeviceNotGpu = 2,
    kDtypePair = 3,
    kElementLimit = 4,
    kShape = 5,
    kBlockDim = 6,
    kGridLimit = 7,
};

constexpr std::uint64_t kMaxElements = 0xffffffffull;
// One block per row, so the row count is the grid's x extent, which CUDA caps
// at 2^31-1.
constexpr std::uint64_t kMaxGridX = 0x7fffffffull;
constexpr int kWarp = 32;
constexpr int kMaxBlock = 1024;

struct OxideEntry {
    mlx::core::Dtype act;
    mlx::core::Dtype weight;
    std::string symbol;
    std::array<std::uint8_t, kRoleCount> roles;
};

// One validated artifact. Immutable once installed and shared with every lazy
// node built from it, so replacing or removing the installed artifact cannot
// pull the payload out from under a node that has not been evaluated yet.
struct OxideArtifact {
    std::string module_name;
    std::string payload;
    std::vector<std::string> symbols;
    int target_major;
    int target_minor;
    std::vector<OxideEntry> entries;
};

std::shared_mutex& artifact_mutex() {
    static std::shared_mutex mtx;
    return mtx;
}

std::shared_ptr<const OxideArtifact>& artifact_slot() {
    static std::shared_ptr<const OxideArtifact> slot;
    return slot;
}

// Lets the common "nothing installed" case skip the lock.
std::atomic<bool> g_installed{false};
std::atomic<std::uint64_t> g_oxide_routes{0};
std::atomic<std::uint64_t> g_native_routes{0};
std::atomic<std::uint64_t> g_launches{0};
std::atomic<int> g_last_fallback{kNotInstalled};

std::shared_ptr<const OxideArtifact> current_artifact() {
    if (!g_installed.load(std::memory_order_acquire)) {
        return nullptr;
    }
    std::shared_lock lock(artifact_mutex());
    return artifact_slot();
}

std::optional<mlx::core::Dtype> dtype_from_code(std::int32_t code) {
    switch (code) {
        case 0:
            return mlx::core::float32;
        case 1:
            return mlx::core::float16;
        case 2:
            return mlx::core::bfloat16;
        default:
            return std::nullopt;
    }
}

// Fused residual-add + RMSNorm through the cuda-oxide module.
//
// Inputs `(x, residual, weight)`, outputs `(normed, new_residual)`. Everything
// that touches the device happens in `eval_gpu`, on the stream MLX evaluates
// the node on, through that stream's command encoder.
class OxideFusedAddRmsNorm : public mlx::core::Primitive {
  public:
    OxideFusedAddRmsNorm(
        mlx::core::Stream stream,
        std::shared_ptr<const OxideArtifact> artifact,
        std::size_t entry,
        float eps,
        float weight_bias,
        std::uint32_t rows,
        std::uint32_t dim,
        std::uint32_t threads)
        : Primitive(stream),
          artifact_(std::move(artifact)),
          entry_(entry),
          eps_(eps),
          weight_bias_(weight_bias),
          rows_(rows),
          dim_(dim),
          threads_(threads) {}

    void eval_cpu(
        const std::vector<mlx::core::array>&,
        std::vector<mlx::core::array>&) override {
        throw std::runtime_error(
            "[fused_add_rms_norm] the cuda-oxide route only runs on the GPU.");
    }

    void eval_gpu(
        const std::vector<mlx::core::array>& inputs,
        std::vector<mlx::core::array>& outputs) override {
        namespace cu = mlx::core::cu;
        using mlx::core::array;

        auto& s = stream();
        auto& encoder = cu::get_command_encoder(s);
        auto& device = encoder.device();

        // The route was chosen against device 0's capability. A node evaluated
        // on a device of another architecture is an error, not a fallback.
        if (device.compute_capability_major() != artifact_->target_major ||
            device.compute_capability_minor() != artifact_->target_minor) {
            throw std::runtime_error(
                "[fused_add_rms_norm] the cuda-oxide artifact targets sm_" +
                std::to_string(artifact_->target_major) +
                std::to_string(artifact_->target_minor) +
                " but this node is evaluated on sm_" +
                std::to_string(device.compute_capability_major()) +
                std::to_string(device.compute_capability_minor()) + ".");
        }

        // The payload was validated against its manifest when it was
        // installed, so the module bypasses MLX's PTX disk cache: a hit there
        // would load bytes nobody compared to the validated payload. The
        // module is still memoized in memory under its content-keyed name.
        //
        // Load the module and resolve the entry before the encoder or the
        // allocator is touched, so a module that fails to load leaves no
        // half-registered node behind. A failure throws out of the evaluation;
        // nothing here retries on another implementation.
        const OxideEntry& entry = artifact_->entries[entry_];
        cu::JitModule& mod = cu::get_jit_module(
            device,
            artifact_->module_name,
            [this]() {
                return std::make_tuple(
                    true, artifact_->payload, artifact_->symbols);
            },
            /* use_disk_cache, upstream's flag, ignored by the overlay */ false,
            cu::DiskCache::Bypass);
        CUfunction kernel = mod.get_kernel(entry.symbol);

        for (auto& out : outputs) {
            out.set_data(cu::malloc_async(out.nbytes(), encoder));
        }

        // The kernel indexes each buffer as `rows * dim` (or `dim`) packed
        // elements, so a strided, broadcast or otherwise non-contiguous input
        // is copied first, the policy `fast::cuda_kernel` applies to the
        // existing port. Layout is a property of an evaluated array, so this
        // cannot be decided when the route is chosen.
        std::vector<array> copies;
        auto contiguous = [&copies, &s](const array& a) -> array {
            if (a.flags().row_contiguous) {
                return a;
            }
            copies.push_back(array(a.shape(), a.dtype(), nullptr, {}));
            mlx::core::copy_gpu(
                a, copies.back(), mlx::core::CopyType::General, s);
            return copies.back();
        };
        const array x = contiguous(inputs[0]);
        const array residual = contiguous(inputs[1]);
        const array weight = contiguous(inputs[2]);

        // Lengths are element counts and travel as 64-bit values. `KernelArgs`
        // stores `CUdeviceptr` (an unsigned 64-bit integer) by value, which is
        // exactly the `.param .u64` the entry declares; `dim` is `.u32`.
        auto length = [](const array& a) {
            return static_cast<CUdeviceptr>(a.size());
        };
        cu::KernelArgs args;
        for (std::uint8_t role : entry.roles) {
            switch (role) {
                case kXPtr:
                    args.append(x);
                    break;
                case kXLen:
                    args.append(length(x));
                    break;
                case kResidualPtr:
                    args.append(residual);
                    break;
                case kResidualLen:
                    args.append(length(residual));
                    break;
                case kWeightPtr:
                    args.append(weight);
                    break;
                case kWeightLen:
                    args.append(length(weight));
                    break;
                case kEps:
                    args.append(eps_);
                    break;
                case kWeightBias:
                    args.append(weight_bias_);
                    break;
                case kDim:
                    args.append(dim_);
                    break;
                case kNormedPtr:
                    args.append(outputs[0]);
                    break;
                case kNormedLen:
                    args.append(length(outputs[0]));
                    break;
                case kNewResidualPtr:
                    args.append(outputs[1]);
                    break;
                case kNewResidualLen:
                    args.append(length(outputs[1]));
                    break;
                default:
                    throw std::logic_error(
                        "[fused_add_rms_norm] unknown cuda-oxide argument role.");
            }
        }

        // Registration order of `CustomKernel::eval_gpu`. The encoder derives
        // the graph dependencies from these buffers; `gpu::eval` keeps inputs
        // and sibling outputs alive until the launch completes.
        encoder.set_input_array(x);
        encoder.set_input_array(residual);
        encoder.set_input_array(weight);
        encoder.set_output_array(outputs[0]);
        encoder.set_output_array(outputs[1]);
        for (const auto& t : copies) {
            encoder.add_temporary(t);
        }
        // One block per row, `fused_norm_cuda_block(dim, n_reads)` lanes
        // (`threads_`, computed by `fused_norm.cpp`), no dynamic shared
        // memory: the launch rule in the artifact's manifest.
        encoder.add_kernel_node_raw(
            kernel, dim3(rows_, 1, 1), dim3(threads_, 1, 1), {}, 0, args.args());
        g_launches.fetch_add(1, std::memory_order_relaxed);
    }

    DEFINE_NAME(OxideFusedAddRmsNorm)

  private:
    std::shared_ptr<const OxideArtifact> artifact_;
    std::size_t entry_;
    float eps_;
    float weight_bias_;
    std::uint32_t rows_;
    std::uint32_t dim_;
    std::uint32_t threads_;
};

std::nullopt_t take_existing_port(Fallback reason) {
    g_native_routes.fetch_add(1, std::memory_order_relaxed);
    g_last_fallback.store(reason, std::memory_order_relaxed);
    return std::nullopt;
}

} // namespace

std::optional<std::vector<mlx::core::array>> try_oxide_fused_add_rms_norm(
    const mlx::core::array& x,
    const mlx::core::array& residual,
    const mlx::core::array& weight,
    float eps,
    float weight_bias,
    std::size_t rows,
    int dim,
    int threads) {
    auto artifact = current_artifact();
    if (!artifact) {
        return take_existing_port(kNotInstalled);
    }
    if (mlx::core::default_device() != mlx::core::Device::gpu) {
        return take_existing_port(kDeviceNotGpu);
    }
    std::optional<std::size_t> entry;
    for (std::size_t i = 0; i < artifact->entries.size(); ++i) {
        if (artifact->entries[i].act == x.dtype() &&
            artifact->entries[i].weight == weight.dtype()) {
            entry = i;
            break;
        }
    }
    if (!entry) {
        return take_existing_port(kDtypePair);
    }
    if (rows == 0 || dim <= 0) {
        return take_existing_port(kShape);
    }
    // The contract the artifact was validated under (#2832) refuses more than
    // 2^32-1 elements; the kernel indexes with 64-bit offsets but was never
    // run past that.
    if (static_cast<std::uint64_t>(rows) >
        kMaxElements / static_cast<std::uint64_t>(dim)) {
        return take_existing_port(kElementLimit);
    }
    // Refuse before enqueue: a launch past the grid limit fails at
    // evaluation, where nothing can reroute it.
    if (static_cast<std::uint64_t>(rows) > kMaxGridX) {
        return take_existing_port(kGridLimit);
    }
    if (threads < kWarp || threads > kMaxBlock || threads % kWarp != 0) {
        return take_existing_port(kBlockDim);
    }

    auto s = mlx::core::to_stream(mlx::core::StreamOrDevice{});
    auto outputs = mlx::core::array::make_arrays(
        {x.shape(), x.shape()},
        {x.dtype(), x.dtype()},
        std::make_shared<OxideFusedAddRmsNorm>(
            s,
            std::move(artifact),
            *entry,
            eps,
            weight_bias,
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(dim),
            static_cast<std::uint32_t>(threads)),
        {x, residual, weight});
    g_oxide_routes.fetch_add(1, std::memory_order_relaxed);
    g_last_fallback.store(kTaken, std::memory_order_relaxed);
    return outputs;
}

} // namespace mlxcel::turbo

// Plain C control surface for `mlxcel-core/src/cuda_oxide_fused_norm.rs`. Kept
// out of the cxx bridge so a build without the feature carries none of it.
extern "C" {

// Install an artifact the caller has already validated against its manifests.
// `symbols` holds `entry_count` entry names joined by '\n'; `act_dtypes` and
// `weight_dtypes` hold `entry_count` dtype codes (0 f32, 1 f16, 2 bf16);
// `roles` holds 13 role bytes per entry, in kernel parameter order. Returns 0
// on success and a nonzero code when the description is malformed, in which
// case the previously installed artifact, if any, is left in place.
int mlxcel_oxide_fused_norm_install(
    const std::uint8_t* payload,
    std::size_t payload_len,
    const char* module_name,
    std::size_t module_name_len,
    std::int32_t target_major,
    std::int32_t target_minor,
    const char* symbols,
    std::size_t symbols_len,
    const std::int32_t* act_dtypes,
    const std::int32_t* weight_dtypes,
    const std::uint8_t* roles,
    std::size_t entry_count) noexcept {
    using namespace mlxcel::turbo;
    try {
        if (payload == nullptr || payload_len == 0 || module_name == nullptr ||
            module_name_len == 0 || symbols == nullptr ||
            act_dtypes == nullptr || weight_dtypes == nullptr ||
            roles == nullptr || entry_count == 0) {
            return 1;
        }
        auto artifact = std::make_shared<OxideArtifact>();
        artifact->module_name.assign(module_name, module_name_len);
        artifact->payload.assign(
            reinterpret_cast<const char*>(payload), payload_len);
        artifact->target_major = target_major;
        artifact->target_minor = target_minor;

        std::string joined(symbols, symbols_len);
        std::size_t start = 0;
        while (start <= joined.size()) {
            std::size_t end = joined.find('\n', start);
            if (end == std::string::npos) {
                end = joined.size();
            }
            if (end == start) {
                return 2;
            }
            artifact->symbols.push_back(joined.substr(start, end - start));
            start = end + 1;
        }
        if (artifact->symbols.size() != entry_count) {
            return 2;
        }

        for (std::size_t i = 0; i < entry_count; ++i) {
            auto act = dtype_from_code(act_dtypes[i]);
            auto weight = dtype_from_code(weight_dtypes[i]);
            if (!act || !weight) {
                return 3;
            }
            OxideEntry entry{*act, *weight, artifact->symbols[i], {}};
            std::array<bool, kRoleCount> seen{};
            for (std::size_t j = 0; j < kRoleCount; ++j) {
                std::uint8_t role = roles[i * kRoleCount + j];
                if (role >= kRoleCount || seen[role]) {
                    return 4;
                }
                seen[role] = true;
                entry.roles[j] = role;
            }
            artifact->entries.push_back(std::move(entry));
        }

        std::unique_lock lock(artifact_mutex());
        artifact_slot() = std::move(artifact);
        g_installed.store(true, std::memory_order_release);
        return 0;
    } catch (...) {
        return 5;
    }
}

// Remove the installed artifact. Later calls take the existing port; lazy
// nodes already built keep their own reference and still evaluate.
void mlxcel_oxide_fused_norm_uninstall() noexcept {
    using namespace mlxcel::turbo;
    std::unique_lock lock(artifact_mutex());
    g_installed.store(false, std::memory_order_release);
    artifact_slot().reset();
}

bool mlxcel_oxide_fused_norm_installed() noexcept {
    return mlxcel::turbo::g_installed.load(std::memory_order_acquire);
}

// Route markers. `route_count` and `native_route_count` are bumped when a
// call is routed (graph construction); `launch_count` when the oxide node
// enqueues its kernel in `eval_gpu`, which is the proof it reached the GPU.
std::uint64_t mlxcel_oxide_fused_norm_route_count() noexcept {
    return mlxcel::turbo::g_oxide_routes.load(std::memory_order_relaxed);
}

std::uint64_t mlxcel_oxide_fused_norm_native_route_count() noexcept {
    return mlxcel::turbo::g_native_routes.load(std::memory_order_relaxed);
}

std::uint64_t mlxcel_oxide_fused_norm_launch_count() noexcept {
    return mlxcel::turbo::g_launches.load(std::memory_order_relaxed);
}

// The `Fallback` code of the most recent routing decision (0 when it took the
// oxide route).
int mlxcel_oxide_fused_norm_last_fallback() noexcept {
    return mlxcel::turbo::g_last_fallback.load(std::memory_order_relaxed);
}

} // extern "C"
