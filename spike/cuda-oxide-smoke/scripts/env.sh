#!/usr/bin/env bash
# Toolchain discovery for the cuda-oxide smoke package (issue #2831).
#
# Source this file; it exports the environment cuda-oxide needs and fails with
# an actionable message when a prerequisite is missing. It never changes the
# repository's Rust pin: the package's own rust-toolchain.toml selects 1.99.1
# when a cargo command runs inside spike/cuda-oxide-smoke.
#
# Required:
#   CUDA 13.0 toolkit (nvcc, cuobjdump, ptxas). Default search:
#     /usr/local/cuda-13.0, then /usr/local/cuda. Override with CUDA_HOME.
#   LLVM 21 with NVPTX, including clang headers and libclang.so. Search:
#     1. LLVM21_PREFIX, if set
#     2. /usr/lib/llvm-21 (apt.llvm.org layout)
#     3. an LLVM 21 tarball unpacked anywhere, via LLVM21_PREFIX
#   Rust 1.99.1 from rustup with rust-src and rustc-dev (see
#   rust-toolchain.toml; `rustup toolchain install 1.99.1` is enough).
#   cargo-oxide built from the pinned revision (see README.md).

_spike_env_error() {
    echo "cuda-oxide-smoke: $*" >&2
    return 1
}

if [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
    echo "cuda-oxide-smoke: source this file instead of executing it: . scripts/env.sh" >&2
    exit 2
fi

# --- CUDA toolkit -----------------------------------------------------------
if [ -z "${CUDA_HOME:-}" ]; then
    for candidate in /usr/local/cuda-13.0 /usr/local/cuda; do
        if [ -x "$candidate/bin/nvcc" ]; then
            CUDA_HOME="$candidate"
            break
        fi
    done
fi
[ -n "${CUDA_HOME:-}" ] || {
    _spike_env_error "no CUDA 13.0 toolkit found. Install it or set CUDA_HOME to a CUDA 13.0 tree (needs bin/nvcc, bin/ptxas, bin/cuobjdump)."
    return 1
}
for tool in nvcc ptxas cuobjdump; do
    [ -x "$CUDA_HOME/bin/$tool" ] || {
        _spike_env_error "CUDA toolkit at $CUDA_HOME is missing bin/$tool"
        return 1
    }
done

# --- LLVM 21 ----------------------------------------------------------------
_llvm_prefix=""
if [ -n "${LLVM21_PREFIX:-}" ]; then
    _llvm_prefix="$LLVM21_PREFIX"
elif [ -d /usr/lib/llvm-21 ]; then
    _llvm_prefix=/usr/lib/llvm-21
elif command -v llvm-config-21 >/dev/null 2>&1; then
    _llvm_prefix=$(
        cd "$(dirname "$(readlink -f "$(command -v llvm-config-21)")")/.." && pwd
    )
fi
[ -n "$_llvm_prefix" ] || {
    _spike_env_error "LLVM 21 not found. Set LLVM21_PREFIX to an LLVM 21 installation (needs bin/llc, bin/clang, bin/llvm-config, lib/libclang.so, lib/clang/21/include), or use the upstream cuda-oxide devcontainer."
    return 1
}
for header_dir in "$_llvm_prefix"/lib/clang/*/include; do
    [ -f "$header_dir/stddef.h" ] && break
done
[ -f "$_llvm_prefix/lib/libclang.so" ] || {
    _spike_env_error "libclang.so is missing under $_llvm_prefix/lib. Install libclang-21-dev (apt.llvm.org) or use an LLVM 21 release tarball."
    return 1
}
[ -f "${header_dir:-/nonexistent}/stddef.h" ] || {
    _spike_env_error "clang resource headers are missing under $_llvm_prefix/lib/clang/*/include. bindgen needs stddef.h; install libclang-common-21-dev or use the LLVM 21 release tarball."
    return 1
}
[ -x "$_llvm_prefix/bin/llc" ] || {
    _spike_env_error "llc is missing under $_llvm_prefix/bin. cuda-oxide needs LLVM 21 with the NVPTX target."
    return 1
}
# Fail early when the LLVM build has no NVPTX backend. Capture the version text
# before grepping so an early grep exit cannot SIGPIPE the producer.
_llc_version=$("$_llvm_prefix/bin/llc" --version)
if ! grep -q nvptx <<<"$_llc_version"; then
    _spike_env_error "LLVM at $_llvm_prefix has no NVPTX target; cuda-oxide cannot emit PTX."
    return 1
fi

export LLVM21_PREFIX="$_llvm_prefix"
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export CUDA_TOOLKIT_PATH="$CUDA_HOME"
export CUDA_OXIDE_LLC="$_llvm_prefix/bin/llc"
export LLVM_CONFIG_PATH="$_llvm_prefix/bin/llvm-config"
export LIBCLANG_PATH="$_llvm_prefix/lib"
export PATH="$_llvm_prefix/bin:$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH"

# The package is built by the pinned toolchain; verify it rather than letting a
# silently different compiler load an incompatible codegen backend.
if command -v rustc >/dev/null 2>&1; then
    _rustc_version=$(rustc --version | awk '{print $2}')
    case "$_rustc_version" in
        1.99.1*) ;;
        *)
            _spike_env_error "rustc $_rustc_version is active, but cuda-oxide pins 1.99.1. Run commands from spike/cuda-oxide-smoke so its rust-toolchain.toml applies, or run: rustup toolchain install 1.99.1 --component rust-src,rustc-dev,llvm-tools"
            return 1
            ;;
    esac
else
    _spike_env_error "rustc not found; install the pinned toolchain with rustup."
    return 1
fi

if ! command -v cargo-oxide >/dev/null 2>&1; then
    _spike_env_error "cargo-oxide not found. Build it from the pinned revision (see README.md, 'Build cargo-oxide')."
    return 1
fi

unset _llvm_prefix header_dir _rustc_version _llc_version
