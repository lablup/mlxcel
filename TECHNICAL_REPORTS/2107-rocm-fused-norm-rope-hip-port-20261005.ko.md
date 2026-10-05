# 기술 보고서: PR #2107 - fused_add_rms_norm과 fused_rope_qk_append를 HIP으로 port

**날짜**: 2026-10-05

**상태**: gfx1151 호스트에서 구현 및 측정 완료. head `c6592780`, origin/main `33c45053` 위, 머지 대기 중.

**언어**: C++ (HIP kernel source 두 개, kernel holder, port table, support predicate, launcher 거부), Rust (gate caching, eligibility 검사, parity test, CPU device gate test), `build.rs` (header 추적), Markdown (결과 문서, 환경 변수, 설치, README, 정확성 문서), CSV와 TSV (benchmark 행, logit trace)

**위험도**: 낮음에서 중간 (두 fusion 모두 모든 backend에서 기본값이 꺼져 있어 기본 decode 경로는 바뀌지 않습니다. 다만 predicate와 빈 입력 처리 변경은 Metal과 CUDA에도 적용되며, 두 backend는 개발 호스트에서 실행할 수 없었습니다)

## 요약

Issue #2063 (#1814의 일부, epic #1801)은 `fused_add_rms_norm`과 `fused_rope_qk_append` (#905)의 HIP port를 요청했습니다. 이 PR 이전에는 두 port table 모두 `.rocm = nullptr`였기 때문에, ROCm에서 `MLXCEL_FUSED_ADD_RMSNORM=1`과 `MLXCEL_FUSED_ROPE_APPEND=1`을 켜도 아무 표시 없이 MLX graph 경로가 유지되었습니다.

이 PR은 `fused_norm_hip.h`와 `fused_rope_append_hip.h`를 추가하고 두 `.rocm` slot을 채우며, 각 port가 대체하는 ROCm graph와 bit 단위로 일치하게 만듭니다. Fused RMSNorm에서 graph의 0 부호까지 그대로 유지합니다. 두 predicate는 이제 기본 device가 GPU인지도 확인하고, Rust gate는 port 답이 `true`일 때만 cache합니다. 빈 입력은 Rust gate와 launcher 양쪽에서 거부되어 graph로 갑니다.

gfx1151에서 decode는 분명하게 개선되지 않았습니다. Llama 3.1 8B는 +0.3% (잡음)였고, Qwen2.5 7B는 median 기준 +1.5%에서 +1.7%였지만 그 크기가 round 묶음 사이에서 off arm 자체의 편차만큼 흔들렸으며, `both`가 `add`보다 높지 않았습니다. 그래서 두 기본값은 Metal, CUDA와 마찬가지로 ROCm에서도 꺼진 상태로 둡니다. Fusion을 켠 teacher-forced trace는 끈 trace와 byte 단위로 같고, Metal 대비 decided mismatch는 0입니다.

## 1. 문제 정의

### 1.1 빈 slot 두 개

`fused_norm_ports()`와 `fused_rope_ports()`에는 Metal과 CUDA 항목만 있었습니다. Predicate가 `has_kernel_port`를 반환했으므로 ROCm에서는 false였고, opt-in 환경 변수는 효과가 없었습니다. Parity test인 `fused_norm_parity_tests.rs`와 `fused_rope_parity_tests.rs`는 predicate가 false이면 일찍 반환했기 때문에 ROCm에서는 아무것도 검증하지 않았습니다.

### 1.2 Port가 맞춰야 하는 것

#1814 port 요구 사항은 gfx1151에서 graph fallback과 기존 tolerance 안에서 일치할 것을 요구합니다. Decode profile (#2061)은 이 두 kernel이 대신할 수 있는 몫을 Llama 3.1 decode GPU 시간의 최대 0.83%에서 0.89%로 잡았습니다. 그래서 issue는 전후 측정도 요구했고, PR이 답해야 할 질문은 ROCm에서 fusion을 기본으로 켤지 여부였습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `fused_norm_hip.h` (신규, 161줄) | `FUSED_ADD_RMS_NORM_HIP_SOURCE`. CUDA 본문에서 port하고 overlay의 `rms_norm_kernel`을 재현하도록 조정 |
| `fused_rope_append_hip.h` (신규, 210줄) | `FUSED_ROPE_APPEND_HIP_SOURCE`. CUDA 본문에서 port하고 `rope.hip`을 재현하도록 조정 |
| `fused_norm.cpp` | `MLXCEL_BRIDGE_ROCM_BACKEND` 아래에서 `fast::hip_kernel`을 호출하는 `FusedNormKernelHolderHip`, `.rocm` getter, ROCm에서 `Threads`를 256으로 고정, predicate에 GPU device 조건, 빈 입력 거부 |
| `fused_rope_append.cpp` | `FusedRopeKernelHolderHip`, `.rocm` getter, predicate에 GPU device 조건, batch나 window가 0인 입력 거부 |
| `layers.rs` | `gpu_port_available` (device는 호출마다 읽고 port 답은 `true`만 cache), eligibility에 비어 있지 않은 입력 조건 추가, 두 기본값의 doc comment에 ROCm 결과 기록 |
| Parity test | Predicate가 false인 GPU backend에서는 skip 대신 실패, 두 kernel의 ROCm byte 동일성 test, 0 부호 test, norm test의 bf16 budget을 쓰는 RoPE dtype sweep |
| `tests/cpu_device_custom_kernel_gates.rs` | 두 predicate 모두 CPU device에서 거절해야 함 |
| Bridge와 FFI 문서 | `mlx_cxx_bridge.h`, `lib.rs`에 device 조건과 ROCm port 설명 |
| Build | `mlxcel-core/build.rs`가 두 header를 추적 |
| 문서와 데이터 | 신규 `docs/benchmark_results/rocm-fused-norm-rope-gfx1151-2026-10-05.md`, benchmark CSV 네 개, `benchmarks/logit_traces/rocm_gfx1151_3f0e51af/` 아래 trace TSV 12개와 metadata, `environment-variables.md`, `installation.md`, README, 정확성 문서 갱신 |

Branch에는 commit 여섯 개와 origin/main merge 하나가 있습니다. Port (`ef3831d9`), bit 단위 일치 수정 (`8ebf809f`), 0 부호 수정 (`3f0e51af`), 결과 문서 (`756d03d8`), review 수정 (`bea7d1ab`), merge (`9e6f095d`), 빈 입력 거부 (`c6592780`)입니다. origin/main 대비 diff는 38개 파일, 4748줄 추가와 99줄 삭제이며, 추가분 대부분은 trace TSV입니다. `src/**/*.cpp`의 `.rocm = nullptr` 줄은 두 개 줄어 origin/main `33c45053`의 7개에서 head의 5개가 됩니다 (branch의 원래 base 기준으로는 10개에서 8개).

## 3. Port: ROCm graph와 bit 단위로 일치

### 3.1 Metal kernel이 아니라 graph를 따름

각 HIP 본문은 CUDA 본문의 thread mapping, kernel 이름, 입력, 출력, grid, template argument를 그대로 유지합니다. Launch는 이미 CUDA launch가 있는 `.cpp` 파일에 남아 있어 `make verify-kernel-dtype-keys`의 `9 in scope` pin이 바뀌지 않습니다. 줄 단위 port에서 벗어나는 곳은 모두 Metal이나 CUDA kernel이 아니라 ROCm graph가 계산하는 값을 재현하기 위한 것입니다. 그 결과 ROCm에서 fusion을 켜도 출력의 bit가 하나도 바뀌지 않습니다.

**`fused_add_rms_norm`**은 `add`와 그 뒤의 overlay `rms_norm_kernel<T, 256, 4>`를 대체합니다. Port는 다음을 따릅니다.

- ROCm에서는 행 길이에서 block 크기를 정하지 않고 행마다 256 thread (`FUSED_NORM_ROCM_THREADS`, overlay의 `BLOCK_DIM`)로 launch합니다. 그래서 thread별 strided 합과 32-wide `__shfl_xor` fold가 graph의 reduction tree를 따릅니다.
- 행 길이를 `Dim` template 상수가 아니라 실행 시점에 `weight_shape[0]`에서 읽습니다. Graph의 kernel도 그렇게 받습니다. (`Dim`은 cache key에 남습니다.)
- `rms_norm_row`처럼 `1.0f / sqrtf(...)`를 씁니다.
- Graph의 `w * normalized`처럼 gain 곱셈을 `T`에서 합니다 (`gain * scaled`, 둘 다 `T`).

**`fused_rope_qk_append`**는 slice, reshape, transpose와 `fast_rope` 호출 두 개를 대체합니다. Port는 다음을 따릅니다.

- 각도를 `rope.hip`의 순서대로 `(scale * position) * inv_freq`, `inv_freq = exp2f(-d * log2(base))`로 계산하고, sine과 cosine을 `sincosf` 한 번으로 얻습니다.
- 두 회전 출력을 `#pragma clang fp contract(off)` 아래에서 명시적인 `fmaf` 호출로 쓰되, 해당 shape에서 graph가 쓰는 형태를 따릅니다. 한 sequence의 한 token (`B == 1 && L == 1`, batch-1 decode)이면 `rope_single_1d`, 그 밖에는 `rope`의 형태입니다. hipcc가 컴파일한 두 kernel이 두 번째 출력을 서로 다르게 fuse하기 때문입니다.
- f16에서는 각 fused 결과를 double로 만들고 (float 곱은 double에서 정확합니다) `__half`로 한 번만 변환합니다. Graph의 f16 instantiation은 한 번 반올림하는데, `(T)fmaf(...)`는 먼저 f32로 반올림하기 때문입니다. f32와 bf16은 `fmaf` 값을 그대로 씁니다.

### 3.2 0의 부호

마지막 불일치가 가장 찾기 어려웠습니다. 다른 수정을 모두 넣은 뒤에도 `MLXCEL_FUSED_ADD_RMSNORM=1`로 만든 Llama 3.1 decode trace는 128개 위치 중 5개에서 graph와 달랐습니다. Join 지점을 계측해 보니 영향을 받은 호출마다 원소 하나에서 port는 +0을, graph는 -0을 썼습니다. f16에서 underflow되는 정규화 원소이거나, 0인 weight에 음수 원소를 곱한 경우입니다. Port는 gain을 f32에서 곱하고 한 번 반올림했는데, f16, bf16, f32 모두 값은 같지만 hipRTC가 그 형태로 만든 코드는 0인 곱의 부호를 버렸습니다. Commit `3f0e51af`는 graph처럼 곱셈을 `T`로 옮기고 `fused_add_rms_norm_keeps_the_rocm_graph_sign_of_zero` (f32, f16, bf16에서 underflow 원소와 부호 있는 0 weight)를 추가했습니다. 이 변경 뒤 Llama 3.1 8B와 Qwen2.5 7B에서 fusion을 켠 trace가 끈 trace와 byte 단위로 같아졌습니다.

### 3.3 모든 세부 사항은 실패한 test에서 나옴

이 세부 사항은 코드를 읽어서 찾은 것이 아닙니다. hipRTC로 컴파일한 port는 식에 컴파일러의 선택 여지가 있는 곳마다 hipcc로 컴파일한 graph와 다르게 반올림했고, 각 차이는 실패한 test나 trace 불일치로 드러났습니다.

| 세부 사항 | 없을 때의 실패 |
|---|---|
| 실행 시점 행 길이 | 폭 4096에서 f32 행 4096개 중 44개가 normalizer 1 ulp 차이 (행 scale e^-8에서 e^8) |
| 행마다 256 thread | 행 크기에 맞춘 thread 수로는 norm byte 동일성 test 실패 |
| `T`에서 gain 곱셈 | -0 대신 +0, Llama 3.1 logit이 decode 위치 128개 중 5개에서 변함 |
| Shape별 FMA 형태 | 여러 token 형태만 맞춘 port는 batch-1 decode에서 실패. hipRTC에 맡기면 f32 원소 약 10개 중 1개가 1 ulp 이동 |
| f16 단일 반올림 | 약 2^13개 원소 중 1개 불일치. Head 48개의 186-token window에서는 항상 발생 |

PR의 negative check가 각 수정이 고정되어 있음을 확인합니다. 행 크기 thread 수, compile-time 행 길이, f32 gain 곱, RoPE FMA 형태 하나만 쓰기, f16 RoPE 결과의 f32 반올림, 8에서 시작하는 norm fold는 각각 parity test를 실패시킵니다. Commit된 test 외에도, 일회성 RoPE stress run 432개 case (batch 1에서 3, window 1에서 300 token, offset 131000까지, 전체와 부분 rotary dim, 값 최대 181배 scale)와 네 폭에서 4096행 norm run이 f32, f16, bf16에서 graph와 bit 단위로 일치했습니다. 이 run은 commit된 test가 아닙니다.

### 3.4 Wave guard

Norm 본문은 #1814가 shuffle 기반 port마다 요구하는 `__AMDGCN_WAVEFRONT_SIZE__`와 `__AMDGCN_WAVEFRONT_SIZE`에 대한 `#error` 검사 두 개를 유지합니다. HIP 7.15의 AMD clang 23은 gfx1151이나 gfx942에서 두 macro를 모두 정의하지 않으므로 이 guard는 효과가 없습니다. 각 fold를 32-lane 그룹 안에 묶어 두는 것은 `__shfl_xor`의 명시적 width 32와, `lane`과 `sg`가 `threadIdx.x % 32`와 `threadIdx.x / 32`라는 점입니다. 64-lane wavefront에서는 두 절반이 각각 `local_sums`로 reduce될 것이라는 추론이며, 실행해 보지는 않았습니다. RoPE 본문에는 shuffle도 thread 사이의 shared memory 읽기도 없으므로 어떤 wavefront 크기에서도 올바르고, #2064 sampler처럼 guard가 없습니다. 이는 issue의 acceptance 문구 ("both HIP sources carry the wave32 guard")에서 벗어나며, issue는 체크된 항목 옆에 그 이유를 기록했습니다.

## 4. Gating

### 4.1 두 predicate의 GPU device 조건

`fused_add_rms_norm_available()`과 `fused_rope_qk_append_available()`은 이제 `has_kernel_port`를 읽기 전에 `mlx::core::default_device()`가 GPU가 아니면 false를 반환합니다. Custom kernel은 GPU stream에서만 실행되고, CPU 기본 device (GPU build에서 `MLXCEL_DEVICE=cpu`)에서는 `eval_cpu`가 throw합니다. 다른 custom kernel gate에 같은 조건을 넣은 #2069를 따른 것입니다. `tests/cpu_device_custom_kernel_gates.rs`는 이제 두 predicate가 CPU device에서 거절하는지 확인합니다. 이 조건은 모든 backend에 적용되므로, Metal과 CUDA에서도 fusion을 켠 채 `MLXCEL_DEVICE=cpu`로 실행하면 첫 launch에서 throw하는 대신 graph 경로로 갑니다.

### 4.2 Rust gate는 true인 port 답만 cache

기존 Rust gate는 FFI predicate를 한 번 묻는 `OnceLock<bool>`이었습니다. Predicate에 device 조건이 들어가면 이 방식은 맞지 않습니다. Device는 바뀔 수 있으므로 (`MLXCEL_DEVICE`, `DefaultDeviceGuard`) 첫 답을 cache하면 그 답이 고정됩니다. PR은 `gpu_port_available`에서 검사를 나눕니다.

- Device 쪽 (`ffi::default_device_is_gpu()`)은 호출마다 읽습니다. MLX 기본 device를 한 번 FFI로 읽는 비용입니다.
- Port 쪽은 처음 `true`를 답할 때까지 묻고, 그 뒤에는 `OnceLock<()>`에 기록합니다.

첫 버전은 port 검사가 반환한 값을 무엇이든 cache했습니다. Review (`bea7d1ab`)에서 race가 발견되었습니다. C++ predicate가 device를 다시 읽기 때문에, Rust의 읽기와 C++의 읽기 사이에 다른 thread가 device를 옮기면 predicate가 `false`를 반환해 프로세스가 끝날 때까지 fusion이 꺼질 수 있었습니다. `true`만 cache하면 `false`는 다음 호출에서 다시 묻게 되므로 이 틈이 닫힙니다. Backend는 프로세스 중간에 바뀌지 않으므로 cache된 `true`는 계속 유효합니다.

## 5. 빈 입력 거부

마지막 commit `c6592780`은 security review에서 나왔습니다. 빈 입력이 크기 0인 grid로 fused launcher에 도달했고, 폭이 0인 행은 norm launcher에서 0으로 나누기 (`x.size() / dim`)를 일으켰습니다. 이제 두 단계에서 이런 shape을 거부합니다.

- **Rust eligibility.** `fused_add_rms_norm_eligible`은 마지막 차원이 양수이고 모든 차원이 양수일 것을 요구합니다. `FusedQKVLinear`는 fused RoPE 경로로 가기 전에 batch와 window가 양수일 것을 요구합니다. 둘 다 빈 배열을 처리할 수 있는 graph로 돌아갑니다.
- **C++ launcher.** `fused_add_rms_norm`은 빈 `x`나 마지막 차원 0에 대해, `fused_rope_qk_append`는 batch나 window 0에 대해 `std::invalid_argument`를 throw합니다. Rust gate를 거치지 않는 직접 호출자를 보호하며, bridge가 두 함수를 `Result`로 선언하므로 throw는 Rust에서 `Err`가 됩니다.

이 경우는 Metal과 CUDA에도 이미 존재했습니다. 그쪽에서도 이제 graph로 갑니다.

## 6. 정확성 근거

### 6.1 Test

gfx1151에서 `cargo test --release --features rocm -p mlxcel-core --lib -- --test-threads=1 fused_norm_parity_tests fused_rope_parity_tests`는 test 21개를 실행하며 모두 통과합니다.

- 기존 tolerance test (norm: normalized RMS / max 기준 f32 1e-6 / 1e-5, f16 2e-3 / 1.2e-2, bf16 1.6e-2 / 7e-2, RoPE 2e-3 / 1.2e-2)가 이제 ROCm에서 실행되며, GPU backend의 predicate가 false이면 skip하지 않고 실패합니다.
- `fused_add_rms_norm_is_byte_identical_to_the_rocm_graph`: 폭 128에서 4096의 f32, f16, bf16, 그리고 행 scale이 e^-8에서 e^8에 걸친 폭 4096의 1024행.
- `fused_add_rms_norm_keeps_the_rocm_graph_sign_of_zero`.
- `fused_rope_append_matches_graph_rope_every_dtype`: f32, f16, bf16에 대한 새 tolerance sweep. Review (`bea7d1ab`)에서 bf16에 norm test의 bf16 budget을 주었습니다. f16 budget은 2-sigma 원소에서 bf16 1 ulp 정도라, 두 경로가 다르게 contract하는 Metal이나 CUDA에서 반올림 하나만 뒤집혀도 실패하기 때문입니다. f32와 f16 budget은 그대로이며, issue의 tolerance도 바뀌지 않았습니다.
- `fused_rope_append_is_byte_identical_to_the_rocm_graph`: batch 1과 2, window 1에서 512 token, offset 131071까지, 두 convention.

### 6.2 Trace

Teacher-forced trace (`benchmarks/logit_traces/rocm_gfx1151_3f0e51af/`)에서, `MLXCEL_FUSED_ADD_RMSNORM=1`인 Llama 3.1 8B와 두 flag를 모두 켠 Qwen2.5 7B의 `w1`, `w8`, `w1ctx512` 모두 on/off 쌍이 byte 단위로 같습니다. Metal 대비 (`compare_logit_traces.py --decided 2.0`):

| 기준 | 비교 대상 | Top-1 불일치 | Decided mismatch |
|---|---|---|---|
| `metal_m1u_bec64748` Llama 3.1 `w8` | ROCm `addrms` `w8` | 0 / 640 | 0 / 230 |
| `metal_m5_d1128266` Qwen2.5 `w8` | ROCm `addrms-rope` `w8` | 4 / 640 (최대 차이 0.047) | 0 / 274 |

On과 off trace가 같으므로 이 행은 fusion을 끈 행과 같습니다. 8-token 생성의 `rocprofv3` kernel trace가 실제 실행되는 kernel을 확인합니다. Qwen2.5는 두 custom kernel을 모두 launch하고 graph RoPE kernel은 launch하지 않습니다. Llama 3.1은 norm port를 launch하고 `rope_single_freqs_1d` / `rope_freqs`를 유지합니다. `rope_scaling` table이 RoPE kernel을 우회하기 때문입니다.

### 6.3 Gate

PR 기준: `756d03d8`에서 `make verify-rocm` 통과, 147 suite, 11871 통과, 0 실패, 378 ignored. `bea7d1ab` 뒤에는 parity test (21개 통과), `cpu_device_custom_kernel_gates`, `dead_doc_pointers`, mlxcel-core lib과 test의 clippy, fmt가 통과했고, Qwen2.5-7B greedy 64-token 출력은 두 fusion을 끄든 켜든 같았습니다. `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`는 `9 in scope`와 pin이 그대로인 채 통과했습니다.

Orchestrator 검증: origin/main `33c45053`과 최신 상태인 head `c6592780`에서 `make verify-rocm`의 모든 단계가 통과했으며, test 11,873개 통과, 0 실패, 378 ignored, smoke OK입니다.

## 7. 측정과 두 기본값을 끈 채로 두는 결정

### 7.1 방법

`scripts/bench_decode.sh`를 pp512/tg128로, run마다 arm 하나, round마다 arm 순서를 돌려 가며, 모든 run을 `scripts/rocm_gpu_guard.sh --idle-secs 60`을 거쳐 실행했습니다. 44개 run 모두 첫 시도에 깨끗했습니다. 병렬 unit이 run 사이에 GPU를 쓰고 있었고 guard가 끝날 때까지 기다렸습니다. Off는 `MLXCEL_FUSED_ADD_RMSNORM=0 MLXCEL_FUSED_ROPE_APPEND=0`입니다. 원시 행은 `benchmarks/rocm_strixhalo-gfx1151_2026-10-05_fused-norm-rope-{off,add,rope,both}.csv`에 있습니다. 측정은 `3f0e51af`에서 했고, 그 뒤 commit은 gating, test, 빈 입력 거부, 문서를 바꿨을 뿐 kernel 본문은 바꾸지 않았습니다.

### 7.2 결과

| 모델 | Arm | Decode tok/s median | Off 대비 | Paired median (on - off) |
|---|---|---|---|---|
| Qwen2.5-7B (7 round) | off | 46.53 | | |
| | add | 47.23 | +1.5% | +1.48 (7개 중 6개 양수) |
| | rope | 46.80 | +0.6% | +0.33 (7개 중 6개) |
| | both | 47.32 | +1.7% | +0.95 (7개 중 6개) |
| Llama-3.1-8B (5 round) | off | 37.85 | | |
| | add | 37.95 | +0.3% | -0.01 (5개 중 2개) |
| Qwen3-30B-A3B (3 round) | off | 62.21 | | |
| | both | 62.42 | +0.3% | +0.06 (3개 중 2개) |

Prefill (512 token) median은 +0.4%에서 +4.3% 움직였지만, run별 편차도 그만큼 넓어서 (Qwen2.5 off만 해도 1565에서 1651 tok/s) prefill도 결과로 읽지 않습니다.

### 7.3 데이터가 기본값 전환을 뒷받침하지 않는 이유

- **Llama 3.1**은 norm port만 실행합니다 (layer마다 add + RMSNorm join 하나, `rope_scaling` 때문에 RoPE port는 우회). Median과 paired 차이 모두 0 근처입니다.
- **Qwen3-30B-A3B**는 두 kernel 모두 호출하지 않으므로 (`qwen3_moe.rs`에 자체 block이 있음) +0.3%는 대조군이며 잡음으로 읽습니다.
- **Qwen2.5 7B**는 로컬 모델 중 두 port에 모두 도달하는 유일한 모델이고, 결정은 이 숫자에 달려 있습니다. 모든 on arm이 7 round 중 6 round에서 off보다 높았지만, 세 가지 관찰 때문에 결과로 볼 수 없습니다.
  - **Round 묶음 사이의 drift.** Round 1에서 3은 +0.95에서 +3.5 tok/s였습니다. 약 두 시간 뒤 같은 binary로 실행한 round 4에서 7은 대부분 -0.4에서 +1이었고, round 4의 `both`는 -2.7, round 7의 `add`는 +2.7이었습니다. 이득의 크기가 측정 시점에 따라 달라집니다.
  - **`both`가 `add`를 넘지 않음.** 각 port가 독립적으로 시간을 줄인다면 둘 다 켠 쪽이 하나만 켠 쪽보다 높아야 합니다. Median은 47.32 대 47.23이고, paired median은 오히려 낮습니다 (+0.95 대 +1.48).
  - **Off arm의 편차가 이득만큼 넓음.** Off는 45.75에서 46.95 tok/s로, median 이득과 비슷한 폭입니다.

  Branch 첫 commit에서 했던 이전 run도 같은 그림이었습니다 (7 round: off 46.54, add 46.84, rope 46.95, both 47.58 tok/s).

- **개연성 논증.** Qwen2.5에서 1%에서 2%의 이득은 port가 없애는 dispatch로 보아 그럴듯합니다. RoPE port 하나가 layer마다 slice 세 개, reshape와 transpose 쌍 세 개, `fast_rope` 호출 두 개를 대신하고, #2099는 작은 op가 많은 경로에서는 GPU 시간 비중이 실제보다 작게 잡힌다는 것을 보였습니다. 결과 문서는 이 논증을 적으면서 동시에 이 데이터가 그것을 입증하지 않는다고 적습니다. 이 보고서도 그 구분을 유지합니다. 이 논증은 이득이 있을 수 있는 이유를 설명할 뿐, 이득이 측정되었다는 뜻이 아닙니다.

따라서 `FUSED_ADD_RMSNORM_DEFAULT`와 `FUSED_ROPE_APPEND_DEFAULT`는 Metal, CUDA와 마찬가지로 ROCm에서도 `false`로 두며, `layers.rs`의 doc comment에 ROCm 숫자를 기록했습니다. Port가 graph와 byte 단위로 같으므로 ROCm에서 opt-in해도 출력 측면의 비용은 없고, 자기 모델에서 이득을 측정한 배포는 두 flag를 켤 수 있습니다. 결과 문서는 ROCm 기본값을 켜려면 무엇이 필요한지 적어 둡니다. 조용한 GPU에서 더 긴 interleaved run (20쌍 이상)으로 Qwen2.5 계열 모델 (`rope_scaling`이 없어 두 port가 모두 실행됨)이 paired 차이가 일관되게 양수인 채 1% 이상 이득을 보이고, Llama 계열 모델은 손해가 없어야 합니다.

## 8. 기술적 선택과 그 이유

- **CUDA kernel이 아니라 ROCm graph에 맞춤.** Tolerance 안에서만 맞는 port라면 opt-in이 출력을 바꿉니다. Graph와 bit 단위로 맞추면 flag는 순수한 성능 선택이 되고, on/off trace를 byte 단위로 비교할 수 있습니다.
- **ROCm에서 build flag로 `Threads`를 256으로 고정.** ROCm build에는 Metal이나 CUDA backend가 없으므로, `fused_add3_layer_norm`에서처럼 build flag가 곧 backend입니다. `Threads`는 template argument이므로 cache key는 폭별로 유지됩니다.
- **Device는 호출마다 읽고, port 답은 true만 cache.** Device는 바뀔 수 있지만 backend는 바뀌지 않습니다. `false`를 cache하면 일시적인 읽기 하나가 프로세스 내내 fusion을 끌 수 있습니다.
- **빈 입력을 두 단계에서 거부.** Rust gate는 일반 호출자를 graph로 보내고, launcher의 throw는 직접 호출자를 보호합니다.
- **Shuffle하는 kernel에만 guard.** RoPE 본문에는 lane 간 연산이 없으므로, 거기 `#error`를 넣으면 wave64에서 올바른 kernel을 거부할 뿐입니다.
- **기본값은 끈 채로 유지.** 측정된 이득이 round 묶음 사이에서 안정적이지 않고, 둘 다 켠 arm이 독립적인 두 절약처럼 행동하지 않습니다.

## 9. 진행 과정 메모

Developer agent는 PR을 연 뒤 API 사용 한도에 걸려 멈췄습니다. 이후 orchestrator가 최종 gate (`c6592780`에서 `make verify-rocm`, 결과는 6.3절)를 실행하고, 그 검증 내용으로 PR 본문을 갱신하고, label (`status:done`, `type:performance`, `priority:medium`, `area:core`, `platform:linux`)을 설정했습니다.

## 10. 남은 위험과 검증하지 않은 것

- **Metal과 CUDA는 실행하지 않았습니다.** Kernel source와 table 항목은 그대로지만, predicate에 GPU device 조건이, launcher에 빈 입력 거부가 들어갔고, parity test는 predicate가 false이면 skip 대신 실패합니다.
- **Byte 동일성은 컴파일러에 의존합니다.** Port는 HIP 7.15의 gfx1151에서 hipcc가 overlay kernel에 대해 만든 코드를 재현합니다. hipcc, hipRTC, overlay의 `rms_norm.hip` / `rope.hip`이 바뀌면 동일성이 깨질 수 있으며, byte 동일성 test가 가장 먼저 보여 줄 것입니다.
- **Gemma와 IQuest Loop Coder**도 norm port를 호출하지만 이 호스트에 checkpoint가 없습니다. Gemma의 `(1 + w)` convention은 tolerance test로만 검증됩니다.
- **Wave64 (CDNA)는 테스트하지 않았습니다.** 현재 clang에서 norm guard는 효과가 없으므로 wave64 build는 컴파일 시점에 막히지 않으며, 그쪽의 정확성은 명시적 shuffle width에 기댑니다.
- **성능 질문은 열려 있습니다.** 측정은 공유 호스트에서 5에서 7 round로 짧았습니다. 권고는 기본값을 끈 채로 두라는 것이지, port에 이득이 없다는 것이 아닙니다.

## 11. 학습 포인트

- **두 컴파일러 사이의 bit 단위 일치는 읽기가 아니라 test로 찾습니다.** hipRTC와 hipcc는 같은 식에서 다른 선택 (contraction, compile-time 상수, 부호 있는 0)을 했습니다. 각각 실패한 test나 trace로 발견했고, 수정을 되돌리면 실패하는 test로 고정했습니다.
- **0의 부호는 실제 차이입니다.** 값이 같은 두 계산이라도 한쪽이 0의 부호를 잃으면 logit이 움직일 수 있습니다.
- **바뀔 수 없는 답만 cache합니다.** Device 검사와 port 검사를 나누고 `true`만 cache하면, hot path gate를 싸게 유지하면서 일시적인 답이 고정되는 것을 막습니다.
- **둘 다 켠 arm은 일관성 검사입니다.** `both`가 `add`를 넘지 않는다는 사실은 median 이득만큼 많은 것을 알려 주었습니다.
- **개연성과 근거를 구분합니다.** 없어진 dispatch에 대한 논증은 이득이 있을 수 있는 이유를 설명하며, 측정 결과가 아니라 그 자체로 기록됩니다.

Refs: #2063, #1814, #1801, #905, #2061, #2064, #2069, #2099, #1809.
