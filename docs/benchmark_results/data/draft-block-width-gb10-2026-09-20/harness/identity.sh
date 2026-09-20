#!/usr/bin/env bash
# Environment identity for one #1797 sweep session.
#
# Records what the previous GB10 records could not. Until 2026-09-20 no harness
# in this tree wrote the kernel release or the NVIDIA driver version anywhere,
# and on that day a kernel upgrade (6.17.0-1029-nvidia -> 7.0.0-1019-nvidia)
# landed without its matching NVIDIA module. The GPU was dead on first boot and
# the driver was reinstalled at 580.178.04 (from 580.173.02). Nothing in the
# #1820 record or its harness could express that, which is why it had to be
# passed along by hand. Both values are captured here so the next person
# comparing across sessions has something to key on.
#
# The MLX pin, the toolchain, the effective CUDA architecture list and the
# binary hash are here for the same reason: a record that says only "GB10" does
# not say enough to know whether two numbers are comparable.
#
#   identity.sh <server-binary> [model dirs...]
set -uo pipefail
BIN=${1:-}
shift || true
REPO=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

printf 'date: %s\n' "$(date -Is)"
printf 'host: %s\n' "$(uname -n)"
printf 'kernel: %s\n' "$(uname -r)"
printf 'arch: %s\n' "$(uname -m)"
printf 'nvidia_driver: %s\n' \
  "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
printf 'gpu: %s\n' \
  "$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
# The architecture list the build was PINNED to, which is not what a default
# build produces here: build.rs auto-detects and yields `121a`, while
# .github/workflows/release.yml ships `90a;100;121` and every prior GB10 record
# pins plain `121`. The suffix does not change the decode kernels (only the
# NVFP4 weight-quantization converter keys on __CUDA_ARCH_SPECIFIC__); it is
# recorded because a measurement should describe the binaries users run.
printf 'MLX_CUDA_ARCHITECTURES: %s\n' "${MLX_CUDA_ARCHITECTURES:-unset (auto-detected)}"
# nvcc is not on the default PATH on this host; the CUDA install root is.
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
printf 'cuda_toolkit: %s\n' \
  "$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
printf 'mlx_pin: %s\n' \
  "$(sed -n 's/^ *GIT_TAG \([0-9a-f]\{40\}\).*/\1/p' "$REPO/src/lib/mlx-cpp/CMakeLists.txt" | head -1)"
# The rustc that BUILT the binary, which is not necessarily the one first on
# PATH here: this repo pins a toolchain and the build exports it ahead of the
# default. `BUILD_RUSTC` lets the caller state the one it used; both are
# printed so a mismatch is visible rather than silently recorded as the wrong
# one.
printf 'rustc_on_path: %s\n' "$(rustc --version 2>/dev/null)"
printf 'rustc_used_for_build: %s\n' \
  "$(${BUILD_RUSTC:-rustc} --version 2>/dev/null)"
printf 'git_commit: %s\n' "$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
printf 'git_dirty: %s\n' \
  "$([ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] && echo yes || echo no)"
printf 'uptime: %s\n' "$(uptime -p 2>/dev/null)"
printf 'mem_total_gib: %.1f\n' \
  "$(awk '/MemTotal/ {print $2 / 1048576}' /proc/meminfo)"

if [ -n "$BIN" ] && [ -x "$BIN" ]; then
  printf 'binary: %s\n' "$BIN"
  printf 'binary_sha256: %s\n' "$(sha256sum "$BIN" | cut -d' ' -f1)"
  printf 'binary_mtime: %s\n' "$(date -Is -r "$BIN")"
  # What the binary was actually compiled for is not readable from the file:
  # the first bare numeric string in a 100 MB binary is not the architecture
  # list. The binary prints it itself at startup, as
  # `compiled for [121] (cubin)` on the arch summary line, and that log line is
  # the evidence the record cites. `MLX_CUDA_ARCHITECTURES` above records what
  # the build was asked for; the two are compared in the record.
fi

for d in "$@"; do
  [ -d "$d" ] || continue
  printf 'model %s: bytes=%s model_type=%s quantization=%s\n' \
    "$d" \
    "$(du -sb "$d" 2>/dev/null | cut -f1)" \
    "$(python3 -c "import json,sys;c=json.load(open('$d/config.json'));print(c.get('model_type','?'))" 2>/dev/null)" \
    "$(python3 -c "
import json
c = json.load(open('$d/config.json'))
q = c.get('quantization')
if q:
    print(q.get('mode', 'affine (implicit)'))
else:
    qc = c.get('quantization_config') or {}
    groups = qc.get('config_groups') or {}
    fmts = sorted({g.get('format') for g in groups.values() if g.get('format')})
    print(','.join(fmts) or qc.get('quant_method') or 'none')
" 2>/dev/null)"
done
