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

// Standalone launch benchmark for the committed cuda-oxide fused residual-add
// + RMSNorm module (issue #2834).
//
// The #2832 harness that produced `results/bench-sm_121.json` needs the
// cuda-oxide compiler toolchain. This program needs only the CUDA driver API:
// it loads the committed PTX with `cuModuleLoadData`, launches an entry with
// the ABI and the launch rule the manifests state, and times the launches with
// CUDA events on the launch stream. No MLX, no allocator, no lazy evaluation,
// so its numbers are the kernel and the driver's launch path alone.
//
// `scripts/standalone-bench.sh` builds it, checks the PTX digest against the
// five manifests before running it, and wraps the run in `gpu-lock`. It is not
// part of any cargo build.
//
// The digest check proves that the PTX about to load is the PTX the in-tree
// manifests pin; it says nothing about who produced either (it is not an
// authenticity check). The gap between the check and the load is a local one:
// anyone who can rewrite the PTX file in that window can already run code as
// the same user.
//
// ABI (abi/*.schema.json): x_ptr, x_len, residual_ptr, residual_len,
// weight_ptr, weight_len, eps (f32), weight_bias (f32), dim (u32), normed_ptr,
// normed_len, new_residual_ptr, new_residual_len. Lengths are u64 element
// counts. Launch: grid = (rows, 1, 1), block = (fused_norm_threads(dim), 1, 1),
// no dynamic shared memory.

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& what) {
    std::fprintf(stderr, "oxide_standalone_bench: %s\n", what.c_str());
    std::exit(1);
}

void check(CUresult status, const char* what) {
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(status, &name);
        die(std::string(what) + " failed: " + (name ? name : "unknown"));
    }
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

enum class Dtype { F32, F16, BF16 };

Dtype parse_dtype(const std::string& s) {
    if (s == "f32") return Dtype::F32;
    if (s == "f16") return Dtype::F16;
    if (s == "bf16") return Dtype::BF16;
    die("unknown dtype " + s);
}

size_t elem_bytes(Dtype d) { return d == Dtype::F32 ? 4 : 2; }

// Round-to-nearest-even f32 -> bf16 bit pattern, finite inputs only.
uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

uint16_t f32_to_f16(float v) {
    _Float16 h = static_cast<_Float16>(v);
    uint16_t bits;
    std::memcpy(&bits, &h, 2);
    return bits;
}

std::vector<uint8_t> encode(const std::vector<float>& values, Dtype d) {
    std::vector<uint8_t> out(values.size() * elem_bytes(d));
    for (size_t i = 0; i < values.size(); i++) {
        if (d == Dtype::F32) {
            std::memcpy(&out[i * 4], &values[i], 4);
        } else {
            uint16_t bits = d == Dtype::BF16 ? f32_to_bf16(values[i]) : f32_to_f16(values[i]);
            std::memcpy(&out[i * 2], &bits, 2);
        }
    }
    return out;
}

// Deterministic values in a normal exponent range (LCG, uniform in [-1, 1)).
std::vector<float> lcg_values(uint32_t seed, size_t n, float center, float spread) {
    std::vector<float> out(n);
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        out[i] = center + spread * (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }
    return out;
}

// `fused_norm_threads` of kernels/fused_norm.cpp and the manifests' rule.
unsigned fused_norm_threads(unsigned dim) {
    unsigned threads = (dim + 3) / 4;
    threads = ((threads + 31) / 32) * 32;
    return std::clamp(threads, 32u, 1024u);
}

struct Shape {
    Dtype act;
    Dtype weight;
    std::string act_name;
    std::string weight_name;
    unsigned rows;
    unsigned dim;
};

// A whole-string unsigned decimal no larger than `max`. std::stoul accepts a
// leading '-' (and wraps it), ignores trailing text and throws on garbage; a
// benchmark argument must be none of those.
unsigned long parse_unsigned(const std::string& text, const std::string& what,
                             unsigned long max) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        die(what + ": expected a non-negative decimal integer, got '" + text + "'");
    }
    try {
        unsigned long value = std::stoul(text);
        if (value > max) die(what + ": " + text + " is above the limit " + std::to_string(max));
        return value;
    } catch (const std::exception&) {
        die(what + ": '" + text + "' is out of range");
    }
}

// act:weight:ROWSxDIM
Shape parse_shape(const std::string& spec) {
    std::stringstream ss(spec);
    std::string act, weight, dims;
    if (!std::getline(ss, act, ':') || !std::getline(ss, weight, ':') || !std::getline(ss, dims)) {
        die("shape " + spec + ": expected act:weight:ROWSxDIM");
    }
    auto x = dims.find('x');
    if (x == std::string::npos) die("shape " + spec + ": expected act:weight:ROWSxDIM");
    Shape s{parse_dtype(act), parse_dtype(weight), act, weight,
            static_cast<unsigned>(parse_unsigned(dims.substr(0, x), "shape " + spec + " rows", UINT_MAX)),
            static_cast<unsigned>(parse_unsigned(dims.substr(x + 1), "shape " + spec + " width", UINT_MAX))};
    if (s.rows == 0 || s.dim == 0) die("shape " + spec + ": zero rows or width");
    return s;
}

std::string entry_name(const Shape& s) {
    if (s.act == s.weight) return "fused_add_rms_norm_" + s.act_name;
    if (s.weight == Dtype::F32 && s.act != Dtype::F32) {
        return "fused_add_rms_norm_" + s.act_name + "_wf32";
    }
    die("no entry for activation " + s.act_name + " with weight " + s.weight_name);
}

double percentile(std::vector<double> sorted, double q) {
    size_t n = sorted.size();
    size_t idx = std::clamp<size_t>(static_cast<size_t>(std::ceil(n * q)), 1, n) - 1;
    return sorted[idx];
}

}  // namespace

int main(int argc, char** argv) {
    std::string ptx_path, csv_path, label;
    std::vector<Shape> shapes;
    int iters = 200, warmup = 20;
    for (int i = 1; i < argc; i++) {
        std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) die(flag + " needs a value");
            return argv[++i];
        };
        if (flag == "--ptx") ptx_path = value();
        else if (flag == "--csv") csv_path = value();
        else if (flag == "--label") label = value();
        else if (flag == "--iters") iters = static_cast<int>(parse_unsigned(value(), "--iters", 100000000));
        else if (flag == "--warmup") warmup = static_cast<int>(parse_unsigned(value(), "--warmup", 100000000));
        else if (flag == "--shape") shapes.push_back(parse_shape(value()));
        else die("unknown argument " + flag);
    }
    if (ptx_path.empty() || shapes.empty() || iters <= 0) {
        die("usage: --ptx FILE --shape act:weight:ROWSxDIM [--shape ...] [--iters N] "
            "[--warmup N] [--csv FILE] [--label TEXT]");
    }
    std::ifstream in(ptx_path, std::ios::binary);
    if (!in) die("cannot read " + ptx_path);
    std::string ptx((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ptx.push_back('\0');

    double t0 = now_ms();
    check(cuInit(0), "cuInit");
    CUdevice device;
    check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context;
    check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");
    double context_ms = now_ms() - t0;
    int cc_major = 0, cc_minor = 0;
    check(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
          "compute capability");
    check(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
          "compute capability");

    // Module load: the driver compiles the PTX here. Host wall clock, since no
    // device work exists yet for an event to bracket.
    t0 = now_ms();
    CUmodule module;
    check(cuModuleLoadData(&module, ptx.data()), "cuModuleLoadData");
    double module_load_ms = now_ms() - t0;

    CUstream stream;
    check(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
    CUevent ev_start, ev_stop;
    check(cuEventCreate(&ev_start, CU_EVENT_DEFAULT), "cuEventCreate");
    check(cuEventCreate(&ev_stop, CU_EVENT_DEFAULT), "cuEventCreate");

    std::printf("device sm_%d%d, context %.1f ms, module load %.1f ms (%zu PTX bytes)\n",
                cc_major, cc_minor, context_ms, module_load_ms, ptx.size() - 1);

    const char* header =
        "label,act,weight,rows,dim,grid,block,entry,iters,warmup,module_load_ms,context_ms,"
        "first_launch_us,warm_mean_us,sync_median_us,sync_p90_us,sync_min_us,"
        "launch_call_us,sync_roundtrip_us,bytes_moved,effective_gbps,first_in_process,output_ok\n";
    std::string rows_csv;
    bool first_in_process = true;
    bool all_ok = true;

    for (const Shape& s : shapes) {
        std::string entry = entry_name(s);
        CUfunction fn;
        check(cuModuleGetFunction(&fn, module, entry.c_str()), "cuModuleGetFunction");
        size_t n = static_cast<size_t>(s.rows) * s.dim;
        if (n > 0xffffffffull) die("more than 2^32-1 elements");
        size_t act_bytes = n * elem_bytes(s.act);
        size_t w_bytes = static_cast<size_t>(s.dim) * elem_bytes(s.weight);

        std::vector<float> xv = lcg_values(2834u + s.rows, n, 0.0f, 1.0f);
        std::vector<float> rv = lcg_values(905u + s.dim, n, 0.0f, 1.0f);
        std::vector<float> wv = lcg_values(7u, s.dim, 1.0f, 0.05f);
        std::vector<uint8_t> xb = encode(xv, s.act), rb = encode(rv, s.act), wb = encode(wv, s.weight);

        CUdeviceptr d_x, d_r, d_w, d_normed, d_newres;
        check(cuMemAlloc(&d_x, act_bytes), "cuMemAlloc");
        check(cuMemAlloc(&d_r, act_bytes), "cuMemAlloc");
        check(cuMemAlloc(&d_w, w_bytes), "cuMemAlloc");
        check(cuMemAlloc(&d_normed, act_bytes), "cuMemAlloc");
        check(cuMemAlloc(&d_newres, act_bytes), "cuMemAlloc");
        check(cuMemcpyHtoD(d_x, xb.data(), act_bytes), "cuMemcpyHtoD");
        check(cuMemcpyHtoD(d_r, rb.data(), act_bytes), "cuMemcpyHtoD");
        check(cuMemcpyHtoD(d_w, wb.data(), w_bytes), "cuMemcpyHtoD");
        check(cuMemsetD8(d_normed, 0, act_bytes), "cuMemsetD8");
        check(cuMemsetD8(d_newres, 0, act_bytes), "cuMemsetD8");

        uint64_t len = n, wlen = s.dim;
        float eps = 1e-5f, weight_bias = 0.0f;
        uint32_t dim = s.dim;
        void* params[13] = {&d_x, &len, &d_r, &len, &d_w, &wlen, &eps, &weight_bias, &dim,
                            &d_normed, &len, &d_newres, &len};
        unsigned block = fused_norm_threads(s.dim);
        auto launch = [&]() {
            check(cuLaunchKernel(fn, s.rows, 1, 1, block, 1, 1, 0, stream, params, nullptr),
                  "cuLaunchKernel");
        };
        auto timed = [&](int count) -> double {
            check(cuEventRecord(ev_start, stream), "cuEventRecord");
            for (int k = 0; k < count; k++) launch();
            check(cuEventRecord(ev_stop, stream), "cuEventRecord");
            check(cuEventSynchronize(ev_stop), "cuEventSynchronize");
            float ms = 0.0f;
            check(cuEventElapsedTime(&ms, ev_start, ev_stop), "cuEventElapsedTime");
            return static_cast<double>(ms) * 1000.0;
        };

        check(cuStreamSynchronize(stream), "cuStreamSynchronize");
        double first_us = timed(1);
        for (int k = 0; k < warmup; k++) launch();
        check(cuStreamSynchronize(stream), "cuStreamSynchronize");

        // Back-to-back launches between two events: the #2832 `warm` figure.
        double warm_mean_us = timed(iters) / iters;

        // One launch per event pair, each synchronized.
        std::vector<double> samples(iters);
        for (int k = 0; k < iters; k++) samples[k] = timed(1);
        std::sort(samples.begin(), samples.end());

        // Host cost of the launch call alone (enqueue, no wait), then of a
        // launch followed by a stream synchronize.
        check(cuStreamSynchronize(stream), "cuStreamSynchronize");
        double h0 = now_ms();
        for (int k = 0; k < iters; k++) launch();
        double launch_call_us = (now_ms() - h0) * 1000.0 / iters;
        check(cuStreamSynchronize(stream), "cuStreamSynchronize");
        h0 = now_ms();
        for (int k = 0; k < iters; k++) {
            launch();
            check(cuStreamSynchronize(stream), "cuStreamSynchronize");
        }
        double sync_roundtrip_us = (now_ms() - h0) * 1000.0 / iters;

        // The launch must have written both outputs: a finite, non-zero result.
        std::vector<uint8_t> normed(act_bytes), newres(act_bytes);
        check(cuMemcpyDtoH(normed.data(), d_normed, act_bytes), "cuMemcpyDtoH");
        check(cuMemcpyDtoH(newres.data(), d_newres, act_bytes), "cuMemcpyDtoH");
        bool nonzero_normed = std::any_of(normed.begin(), normed.end(), [](uint8_t b) { return b; });
        bool nonzero_newres = std::any_of(newres.begin(), newres.end(), [](uint8_t b) { return b; });
        bool exact = true;
        if (s.act == Dtype::F32) {
            // f32 residual sum is one IEEE add, so the host can check it exactly.
            for (size_t i = 0; i < n; i++) {
                float got;
                std::memcpy(&got, &newres[i * 4], 4);
                if (got != xv[i] + rv[i]) {
                    exact = false;
                    break;
                }
            }
        }
        bool ok = nonzero_normed && nonzero_newres && exact;
        all_ok = all_ok && ok;

        // Effective traffic as #2832 counted it: x and residual read,
        // new_residual written and re-read, normed written, weight read per row.
        double bytes_moved = 5.0 * act_bytes + static_cast<double>(s.rows) * w_bytes;
        double gbps = bytes_moved / (warm_mean_us * 1e-6) / 1e9;

        const char* row_format =
            "%s,%s,%s,%u,%u,%u,%u,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%.2f,%s,%s\n";
        auto format_row = [&](char* buf, size_t cap) {
            return std::snprintf(buf, cap, row_format, label.c_str(), s.act_name.c_str(),
                                 s.weight_name.c_str(), s.rows, s.dim, s.rows, block, entry.c_str(),
                                 iters, warmup, module_load_ms, context_ms, first_us, warm_mean_us,
                                 percentile(samples, 0.5), percentile(samples, 0.9),
                                 samples.front(), launch_call_us, sync_roundtrip_us, bytes_moved,
                                 gbps, first_in_process ? "true" : "false", ok ? "true" : "false");
        };
        // Sized by a first pass, so a long --label cannot truncate the row and
        // drop its newline.
        int needed = format_row(nullptr, 0);
        if (needed < 0) die("cannot format the CSV row");
        std::string line(static_cast<size_t>(needed) + 1, '\0');
        format_row(line.data(), line.size());
        line.resize(static_cast<size_t>(needed));
        rows_csv += line;
        std::printf("%-4s/%-4s %5u x %-5u grid %5u block %4u  first %8.2f us  warm mean %8.3f  "
                    "median %8.3f  p90 %8.3f  launch call %6.3f  launch+sync %7.3f  %7.2f GB/s%s\n",
                    s.act_name.c_str(), s.weight_name.c_str(), s.rows, s.dim, s.rows, block,
                    first_us, warm_mean_us, percentile(samples, 0.5), percentile(samples, 0.9),
                    launch_call_us, sync_roundtrip_us, gbps, ok ? "" : "  OUTPUT CHECK FAILED");
        first_in_process = false;

        cuMemFree(d_x);
        cuMemFree(d_r);
        cuMemFree(d_w);
        cuMemFree(d_normed);
        cuMemFree(d_newres);
    }

    if (!csv_path.empty()) {
        bool exists = static_cast<bool>(std::ifstream(csv_path));
        std::ofstream out(csv_path, std::ios::app);
        if (!out) die("cannot write " + csv_path);
        if (!exists) out << header;
        out << rows_csv;
    }
    cuEventDestroy(ev_start);
    cuEventDestroy(ev_stop);
    cuStreamDestroy(stream);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return all_ok ? 0 : 1;
}
