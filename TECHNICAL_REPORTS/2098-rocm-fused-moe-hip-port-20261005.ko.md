# 기술 보고서: PR #2098 - fused MoE decode kernel의 HIP port

**날짜**: 2026-10-05

**상태**: gfx1151 호스트에서 구현 및 측정 완료. head `eade24a1`, origin/main `57d8ed29` 위, PR open 상태로 머지 대기 중.

**언어**: C++ (`fast::hip_kernel`을 통한 HIP kernel source, port table, predicate), Rust (`switch_layers.rs`와 `nemotron_h.rs`의 backend gate, FFI, parity test), Bash (idle-GPU guard), Python (guard unit test), Markdown, TSV/CSV (logit trace, bench 결과)

**위험도**: 중간 (ROCm의 single-token MoE decode가 모든 SwitchGLU 계열에서 기본값으로 새 GPU kernel을 실행하고, ROCm의 Nemotron-H는 경로가 바뀝니다. Metal은 그대로이고, CUDA에서 바뀌는 동작은 `MLXCEL_FUSED_MOE_RELU2`가 panic 대신 사양하는 것 하나입니다. Wave64 장치는 시험하지 않았습니다)

## 요약

Issue #2065 (#1814의 항목 5, #2086 profile에서 2순위)는 두 fused decode-MoE kernel의 HIP port를 요구했습니다. 이 PR 이전에는 `moe_gateup_ports()`와 `moe_down_ports()`의 `.rocm`이 `nullptr`였고, 그 앞의 Rust gate 두 개가 모두 backend 전체 단위의 `custom_kernels_available()`을 읽었습니다. 이 값은 ROCm에서 false이므로 ROCm decode는 routed expert마다 `gather_qmm`을 실행했습니다.

이 PR은 CUDA kernel을 옮기면서 lane fold만 `__shfl_down(v, o, 32)`로 바꾼 `MOE_GATEUP_HIP_SOURCE`와 `MOE_DOWN_HIP_SOURCE`를 추가하고, 두 `.rocm` 칸을 채우며, backend 단위 gate를 port table을 읽는 predicate 두 개 (`fused_moe_kernels_available()`, `moe_down_kernel_available()`)로 바꿉니다. ROCm build에서는 threadgroup당 row 수 기본값 `MLXCEL_FUSED_MOE_SGY`가 8이 아니라 2입니다. gfx1151에서 8로는 port가 `gather_qmm`보다 느렸기 때문입니다.

gfx1151에서 Qwen3-30B-A3B-4bit decode는 median 61.29에서 62.51 tok/s로 올랐습니다 (+2.0%, 모든 after run이 모든 before run보다 높음). 이는 activation, combine, dispatch overhead만 회수할 수 있다는 #2086의 추정과 맞습니다. Nemotron-3-Nano는 +1.0% (noise 수준)이고, expert가 Dff 상한을 넘는 Mixtral은 변화가 없습니다. Logit trace는 모든 비교에서 decided position 불일치가 0이며, Qwen3의 single-token logit은 이제 ROCm `gather_qmm` (5/128)보다 Metal fused kernel에 더 가깝습니다 (top-1 불일치 1/128).

작업 중 발견한 두 가지도 고치거나 문서화했습니다. `scripts/rocm_gpu_guard.sh`가 명령 자신의 종료된 자식 프로세스를 외부 GPU 사용자로 세어 대부분의 측정 시도를 거부하고 있었고, #1814 port들이 지닌 wave32 `#error` guard는 AMD clang 23에서 작동하지 않습니다.

## 1. 문제 정의

### 1.1 ROCm에서 닿을 수 없던 fused MoE 경로

Fused two-kernel decode MoE 경로 (활성화된 intermediate를 쓰는 gate-up kernel, 이어서 score로 가중된 expert 출력을 합치는 down kernel)는 Metal과 CUDA port만 있었습니다. ROCm을 이 경로에서 막은 것은 서로 독립적인 두 가지였습니다.

- port table에 `.rocm` 항목이 없어서 launcher가 거부했습니다.
- `switch_layers.rs`의 `fused_moe_enabled()`와 `nemotron_h.rs`의 `use_fused` 결정이 모두 `custom_kernels_available()`을 읽었고, 이 값은 ROCm에서 의도적으로 false입니다 (#1803). Table만 채웠다면 gate는 열리지 않았을 것입니다.

#2086 profile은 Qwen3-30B-A3B decode GPU 시간의 46.8%를 fused MoE가 닿는 작업으로 보았지만, 그중 42.1 point는 이미 호스트 대역폭 (약 181 GB/s)에 가까운 expert GEMV였습니다. 그래서 fused kernel이 회수할 수 있는 양은 GPU 시간의 약 5%와 token당 약 430개의 dispatch로 추정했습니다.

### 1.2 낡은 문서

`README.md`, `docs/installation.md`, `docs/environment-variables.md`는 여전히 ROCm에서 affine MoE 모델이 abort하므로 `MLXCEL_FUSED_MOE=0`이 필요하다고 적고 있었습니다. 이는 이미 코드와 맞지 않았고 (backend 단위 gate가 ROCm을 `gather_qmm`에 두고 있었음), 이 PR 이후에는 kernel이 실제로 실행되므로 반대 방향으로 틀린 설명이 됩니다.

## 2. 변경 요약

Commit 일곱 개:

- **`d3801805`** `update(rocm): port the fused MoE decode kernels to HIP`: HIP source 두 개와 holder, 두 `.rocm` 칸, predicate 두 개와 FFI, `switch_layers.rs`와 `nemotron_h.rs`의 gate 변경, `MLXCEL_FUSED_MOE_RELU2` 사양, test gate와 새 SwiGLU parity case.
- **`721ff14a`** `docs(rocm): state what the fused MoE wave32 guard actually covers`: `hipcc -E -dM` 확인 후 주석과 env 문서 수정 (5절).
- **`4ab91f82`** `fix(bench): stop rocm_gpu_guard rejecting the command's own exited children`: guard 수정과 unit test (4절).
- **`359f67ed`** `update(rocm): default the fused MoE SGY to 2 on ROCm`.
- **`826dfa8c`** `fix(rocm): pick the fused MoE SGY default from the build flag`: `make verify-kernel-port-dispatch`가 거부한 runtime backend 비교를 대체.
- **`77afbb1d`** `docs(rocm): list the SSM update step among the HIP-ported kernels`: #2099 위로 rebase한 뒤의 README 문구.
- **`eade24a1`** `docs(rocm): publish the fused MoE decode and logit results on gfx1151`: `docs/benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md`, bench CSV, `benchmarks/logit_traces/rocm_gfx1151_77afbb1d/` 아래 trace.

### 2.1 Kernel

두 HIP source는 CUDA 본문에서 warp shuffle 하나만 바꾼 것입니다. HIP의 `__shfl_down_sync`는 mask를 무시하는 호환용 shim이므로, bitlinear port (#1862)를 따라 width를 명시한 native `__shfl_down(var, delta, 32)`를 씁니다. 입력, 출력, grid (`threadIdx.x`에 출력 row당 32-lane wavefront 하나, `threadIdx.y`에 block당 `sgy`개 row, `grid.z`에 expert slot)와 template arg가 같으므로 hipRTC cache key도 CUDA key와 같은 방식으로 `T`를 담습니다. `expf`와 `tanhf`는 정밀한 library 호출 그대로입니다 (hipRTC는 fast-math 없이 -O3로 compile).

Down kernel은 CUDA kernel의 6-bit 분기 (세 byte에 weight 네 개, `quantized.h`와 같은 배치), score를 f32에서 접는 f32 partial, 그리고 #886이 도입한 K-sum 뒤 activation dtype으로의 단일 반올림을 그대로 유지합니다. HIP down kernel 하나가 `moe_down_ports()`의 호출자 셋을 모두 담당합니다. `run_fused_moe_two_kernel`을 통한 SwitchGLU와 GeGLU, 그리고 `moe_down_kernel_fn()`을 통한 Nemotron-H의 opt-in squared-ReLU 경로입니다.

### 2.2 Port table을 읽는 gate

`fused_moe_kernels_available()`은 `has_kernel_port(moe_gateup_ports()) && has_kernel_port(moe_down_ports())`이고, `moe_down_kernel_available()`은 down table만 읽습니다. `select_kernel_port`가 같은 table을 읽으므로 gate와 dispatch는 서로 다르게 답할 수 없습니다 (#1801 규칙). `switch_layers.rs`는 모든 SwitchGLU 계열을 첫 번째 predicate로 막으며, 여기에는 qwen3_moe, qwen3_vl_moe, qwen3_next, gemma4가 포함됩니다. `nemotron_h.rs`는 두 번째를 읽습니다. 두 곳 모두 더 이상 `custom_kernels_available()`을 호출하지 않습니다. ROCm에 port가 없다고 적었던 `lib.rs`와 `gpu_backend.h`의 doc comment는 이제 backend 단위 검사는 ROCm을 여전히 `None`처럼 다루고, port된 kernel의 자체 predicate가 그 kernel을 연다고 설명합니다.

Nemotron-H의 ROCm 경로는 `forward_nonfused`에서 `fused_moe_forward`로 바뀝니다. 기본 분기는 여전히 routed expert에 `gather_qmm`을 쓰고, 바뀌는 것은 그 주변의 combine뿐입니다. 기본 분기가 custom kernel을 실행하지 않는데도 gate 조건을 남겨 둔 것은, 나중에 그 분기에 kernel이 추가되더라도 port가 없는 backend로 gate가 조용히 넓어지지 않게 하기 위해서입니다.

### 2.3 `MLXCEL_FUSED_MOE_RELU2`는 거부 대신 사양

Opt-in squared-ReLU 분기는 이제 `has_kernel_port(moe_fc1_relu2_ports())`와 `has_kernel_port(moe_down_ports())`도 요구합니다. fc1 kernel은 Metal port만 있으므로 (ROCm port는 #2069), ROCm과 CUDA에서 이 flag는 기본 `gather_qmm` 분기를 그대로 둡니다. CUDA에서는 `nemotron_h.rs`의 `.expect`에서 나던 panic이 사라지고, Metal은 그대로입니다.

### 2.4 Test gate

`fused_moe_parity_tests.rs`의 `gpu_backend_or_skip()`은 이제 GPU backend가 없을 때만 건너뜁니다. Metal, CUDA, ROCm에서는 `fused_moe_kernels_available()`을 assert하고 false면 실패하므로, port 누락은 건너뛰지 않고 보고됩니다. qwen3_moe와 qwen3_vl_moe의 positive control은 `custom_kernels_available()` 대신 이 predicate를 읽습니다. 새 case `fused_moe_swiglu_kernel_matches_references_every_down_width`는 Qwen3 shape에서 SwiGLU를 4/4, 8/8, 4/6 bit로 실행해 SwiGLU 분기와 6-bit down 분기에 닿습니다. 허용 오차는 바뀌지 않았습니다. HIP kernel이 `__shfl_down`으로 reduce하는지 source를 검사하는 assert도 있습니다.

## 3. 측정 결과

### 3.1 방법

gfx1151 (Radeon 8060S, RDNA 3.5), ROCm 10.0.0, HIP 7.15.26333, MLX pin `81ba1c6a`, overlay `75915908`. Before: origin/main `57d8ed29`. After: 그 main 위의 branch `77afbb1d`. `scripts/bench_decode.sh`를 pp512/tg128로, before와 after binary를 run마다 번갈아 세 round 실행했고, 모든 run은 수정된 guard를 거쳤으며 경합이 있던 시도는 거부 후 재실행했습니다.

작업 도중 main이 `57d8ed29` (#2099, HIP `ssm_update` port)로 이동했고, 이는 Nemotron-H decode를 바꿉니다. Branch를 rebase하고 baseline을 `57d8ed29`에서 다시 build한 뒤 decode와 Nemotron logit 측정을 모두 다시 했으므로, before와 after의 차이는 이 PR뿐입니다.

### 3.2 Decode 처리량

| 모델 | 경로 변화 | Before tok/s (median) | After tok/s (median) | 변화 |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | `gather_qmm`에서 fused HIP 쌍으로 | 61.29 / 61.05 / 61.83 (61.29) | 62.51 / 62.68 / 62.16 (62.51) | +2.0% |
| Nemotron-3-Nano-30B-A3B-4bit | `forward_nonfused`에서 `fused_moe_forward`로 (여전히 `gather_qmm`) | 74.30 / 74.62 / 74.37 (74.37) | 75.11 / 75.09 / 74.88 (75.09) | +1.0%, noise 수준 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 없음 (Dff 14336이 `MLXCEL_FUSED_MOE_MAX_DFF` 기본값 8192 초과) | 9.42 / 9.08 / 10.91 (9.42) | 9.51 / 9.57 (두 run) | +1.3%, noise (같은 코드) |

Qwen3의 향상은 일관되며 (모든 after run이 모든 before run보다 높음) #2086의 상한 안에 있습니다. Fused 쌍은 activation, weighted sum, index 생성 dispatch를 없애지만 expert weight는 똑같이 읽습니다. Mixtral의 세 번째 after run은 깨끗한 GPU 구간을 얻지 못했습니다. 실행하는 코드가 같으므로 9.08~10.91의 폭은 이 호스트에서 26 GB checkpoint가 보이는 noise입니다. 모델별 median에 대한 `compare_bench_csv.py --allow-commit-change`는 세 쌍에서 median 1.016x를 보고하며, 10% 넘게 움직인 것은 없습니다. Prefill은 fused 경로를 지나지 않습니다.

### 3.3 SGY 기본값

첫 구현은 CUDA kernel을 Metal의 SGY 기본값 8과 함께 그대로 옮겼습니다. Guard를 거친 한 쌍의 측정에서 Qwen3는 `gather_qmm` (`c0b71344`) 61.64 tok/s, fused 59.42였습니다. Port가 대체하려던 fallback보다 느렸습니다. 한 binary에서 `MLXCEL_FUSED_MOE_SGY`를 1, 2, 4, 8, 16, 32로 바꾼 sweep에서는 모든 반복에서 2가 가장 빨랐습니다 (약 62.3 tok/s, `gather_qmm` 약 61, SGY 1 약 61.2, 나머지 59.3~60.4). 이 sweep의 guard 시도 아홉 번은 모두 경합으로 거부되었으므로 (적어도 한 sample에 다른 unit의 GPU 프로세스가 보임) 참고용일 뿐이고, 근거는 새 기본값에서 측정한 3.2의 표입니다. SGY는 threadgroup 모양만 정하며, 출력이 이 값에 의존하지 않는다는 것은 `fused_moe_geglu_kernel_bitwise_invariant_across_sgy`가 고정합니다.

### 3.4 Logit

`compare_logit_traces.py --decided 2.0`, 새 trace는 `benchmarks/logit_traces/rocm_gfx1151_77afbb1d/`. Fused kernel은 single-token forward에서만 실행되므로 `w8` trace는 여기에 닿지 않습니다 (`w8`에서 Qwen3 `default`와 `fused0`은 동일). kernel을 실행하는 window는 `w1`이고, Nemotron-H `w1ctx512`는 state가 있는 decode를 다룹니다.

| 모델, window | 기준 | 후보 | Top-1 불일치 | Decided 불일치 |
|---|---|---|---|---|
| Qwen3-30B-A3B, w1 | Metal fused | ROCm fused (HIP) | 1 / 128 | 0 / 74 |
| Qwen3-30B-A3B, w1 | Metal `gather_qmm` | ROCm `gather_qmm` | 5 / 128 | 0 / 70 |
| Qwen3-30B-A3B, w1 | ROCm `gather_qmm` | ROCm fused | 3 / 128 | 0 / 74 |
| Qwen3-30B-A3B, w8 | Metal default | ROCm default | 17 / 640 | 0 / 319 |
| Nemotron-3-Nano, w1ctx512 | ROCm main | 이 branch | 3 / 128 | 0 / 68 |
| Nemotron-3-Nano, w1ctx512 | Metal M5 | 이 branch | 6 / 128 | 0 / 71 |
| Nemotron-3-Nano, w8 | ROCm `c5fe9a16` (`forward_nonfused`) | 이 branch | 20 / 640 | 0 / 286 |
| Nemotron-3-Nano, w8 | Metal M5 | 이 branch | 22 / 640 | 0 / 287 |

모든 행에서 decided position 불일치는 0입니다. Nemotron-H의 경로 변화는 undecided token 몇 개를 움직이며, 그 위치의 기준 gap은 0.5 이하입니다. `gather_qmm` 경로는 바뀌지 않았습니다. Qwen3 `fused0` `w1`과 Mixtral `w1` trace는 모든 data row에서 `rocm_gfx1151_bec64748`와 byte 단위로 같고, Qwen3 `fused0` `w8`은 `57d8ed29`에서 만든 trace와 byte 단위로 같습니다. Nemotron-3-Nano는 `MLXCEL_FUSED_MOE_RELU2=1` 유무와 관계없이 같은 40-token greedy text를 생성합니다.

## 4. 측정 중 발견한 guard 버그

`scripts/rocm_gpu_guard.sh` (#2086에서 추가)는 `/sys/class/kfd/kfd/proc`을 1 Hz로 sample하고, GPU를 쥔 프로세스 중 명령 자신이나 그 자손이 아닌 것이 있으면 시도를 거부합니다. KFD는 프로세스가 reap된 뒤 deferred work에서 그 항목을 지우므로, 잠깐 동안 항목이 `/proc` 디렉터리가 없는 pid를 가리킵니다. 그러면 다음 sample이 명령 자신의 종료된 자식 (`bench_decode.sh`의 `rocminfo` probe, 또는 bench binary 자체)을 보고, `descends_from`이 부모 관계를 읽지 못해 외부 프로세스로 셌습니다. 이 unit과 병렬로 돌던 #2067 unit 모두에서 대부분의 시도가 이렇게 거부되었습니다.

수정은 `/proc/<pid>`가 없는 holder를 건너뛰는 것입니다. 종료되고 reap된 프로세스는 GPU를 쓰고 있지 않습니다. `tests/test_rocm_decode_profile.py`의 `test_a_kfd_entry_left_by_an_exited_process_is_not_contention`은 가짜 KFD 디렉터리로 그 구간을 재현하며, 옛 guard에서는 실패하고 (exit 75, `CONTENDED`) 새 guard에서는 통과합니다. 동시에 시작한 guard 두 개가 서로를 거부하는 두 번째 현상은 코드가 아니라 운영으로 (더 긴 idle 구간, 대기) 대응했습니다.

## 5. 작동하지 않는 wave32 guard

#1814 HIP port들은 `__AMDGCN_WAVEFRONT_SIZE`와 `__AMDGCN_WAVEFRONT_SIZE__`에 대한 `#error` guard를 지닙니다. gfx942와 gfx1151에 대해 `hipcc -E -dM`을 실행해 보니 AMD clang 23 (HIP 7.15)은 두 철자 모두 정의하지 않으므로, guard는 발동하지 않고 wave64 target도 compile됩니다. 그런 장치에서 kernel을 올바르게 유지하는 것은 명시한 shuffle width입니다. `__shfl_down(v, o, 32)`는 64-lane wavefront에서도 각 32-lane fold를 한 row 안에 가둡니다. env 문서 초안은 wave64 target이 compile에 실패한다고 잘못 적었고, 이를 고쳐 이제 wave64는 시험하지 않았으며 `MLXCEL_FUSED_MOE=0`으로 그런 호스트를 `gather_qmm`에 둘 수 있다고 설명합니다. 이전의 bitlinear port도 마찬가지이며, 이 PR은 그것을 바꾸지 않습니다.

## 6. 기술적 선택과 그 이유

- **Backend가 아니라 port table로 gate합니다.** Backend 단위 predicate로는 "ROCm에 이 kernel은 있고 저 kernel은 없다"를 표현할 수 없는데, #1814는 port를 하나씩 채우므로 바로 그 상태를 만듭니다. `select_kernel_port`와 같은 table을 읽으면 gate와 dispatch가 구조적으로 일치합니다.
- **CUDA kernel을 재설계하지 않고 옮깁니다.** Geometry, template arg, 수치 처리 (6-bit 분기, f32 partial, 단일 최종 반올림)를 같게 두면 parity 범위와 #886 수정이 그대로 유효하고, 성능 작업은 따로 측정할 수 있는 단계로 남습니다.
- **ROCm SGY 기본값은 compile 시점에 고릅니다.** 첫 시도는 launcher에서 `gpu_kernel_backend()`를 runtime에 비교했고, `make verify-kernel-port-dispatch`는 `select_kernel_port` 밖의 runtime backend 비교를 거부합니다. ROCm build에는 Metal이나 CUDA backend가 없으므로 `MLXCEL_BRIDGE_ROCM_BACKEND`가 곧 backend이고 gate도 통과합니다. Metal과 CUDA는 8을 유지합니다.
- **Port 일부만 있을 때는 거부하지 않고 사양합니다.** `MLXCEL_FUSED_MOE_RELU2`는 fc1이나 down이 없으면 `gather_qmm`으로 돌아가며, 이것으로 CUDA의 잠재적 panic도 사라집니다.
- **Port가 없는 GPU backend에서는 건너뛰지 않고 실패합니다.** 이제 모든 GPU backend가 두 port를 가지므로, 거기서 predicate가 false라면 parity test가 보고해야 할 결함입니다.
- **Main이 움직이면 baseline을 다시 만듭니다.** #2099가 Nemotron-H decode를 바꿨으므로, 옛 baseline과 비교했다면 #2099의 효과가 이 PR의 것으로 잡혔을 것입니다.

## 7. 검증

gfx1151에서:

- 최종 rebase head에서 `make verify-rocm`: 146 suite, 11857 passed, 0 failed, 378 ignored (head `eade24a1`). Rebase 전 head `beb8f77c`에서는 146 suite, 11852 passed, 0 failed, 378 ignored.
- `fused_moe_parity_tests`: 5 passed. Fused 대 all-f32 기준의 normalized RMS 최대 3.0e-6, fused 대 `gather_qmm` 약 3.7e-3, `gather_qmm` 대 기준 약 3.3e-3. Negative control: down kernel의 lane fold를 16 대신 8에서 시작하면 두 기준 test가 모두 실패합니다 (normalized RMS 0.63과 0.70).
- `qwen3_moe::tests`와 `qwen3_vl_moe::tests`: 12 passed, positive control이 kernel을 실행.
- `switch_layers`와 `nemotron_h` lib test: 93 passed.
- `mlxcel-core`와 `mlxcel`에 대한 `cargo clippy -D warnings`: clean. Fast gate 통과, dtype-key pin은 범위 안에서 9로 변함없음.
- Guard unit test: 옛 guard에서 실패, 새 guard에서 통과.

정적 review에서 CRITICAL이나 HIGH는 없었습니다. MEDIUM은 wave64 우려였고 5절의 발견으로 답했습니다 (compile 실패는 없음, 시험하지 않음). LOW 항목은 고쳤습니다. `MLXCEL_FUSED_MOE_MAX_DFF` 행에 ROCm을 넣었고, `lib.rs`와 `gpu_backend.h`의 낡은 "ROCm has no ports" 주석을 다시 썼습니다.

## 8. 남은 위험과 검증하지 않은 부분

- **Metal과 CUDA는 실행하지 않았습니다.** 이 호스트에 하드웨어가 없습니다. `.metal`과 `.cuda` 항목과 source는 손대지 않았고 두 predicate는 그곳에서 true입니다. CUDA에서 보이는 변화는 `MLXCEL_FUSED_MOE_RELU2` 사양이고, SGY 기본값 변경은 ROCm build에만 compile됩니다.
- **Gemma 4와 Qwen3-Next는 실행하지 않았습니다.** 같은 gate를 거쳐 같은 kernel에 닿지만 (Gemma 4는 GeGLU) checkpoint가 없었습니다. GeGLU 분기는 parity test로만 덮여 있습니다.
- **Wave64 (CDNA)는 시험하지 않았습니다.** Compile 시점 guard는 이를 막지 못하며 (5절), 정확성은 명시한 shuffle width에 기대고 있습니다.
- **장치 하나.** 처리량과 SGY 선택은 모두 gfx1151에서 나왔습니다. SGY sweep 자체는 모든 시도가 경합 상태여서 참고용이고, 다른 RDNA나 CDNA 장치에는 다른 기본값이 맞을 수 있습니다.
- **Noise에 가까운 작은 향상.** Nemotron-H의 +1.0%는 run 간 편차 안에 있고, Mixtral은 after run이 두 개뿐입니다.
- **단순한 port.** Vectorized load나 shared memory의 `x` 없이 CUDA를 그대로 옮긴 것입니다. Expert GEMV가 대역폭에 묶여 있으므로 여유는 작을 것으로 보이지만 측정하지는 않았습니다.

권장 후속 작업: HIP MoE kernel의 vectorized load 또는 `x`의 shared memory 적재, 작동하지 않는 macro에 기대지 않는 모든 ROCm port용 wave size 검사 (예: port 선택 시 host 쪽 `warpSize` 검사), granite의 `block_sparse_moe`를 `forward_fused_kernel`로 연결 (#2086에서 언급), Nemotron-H loader의 stdout 출력 정리.

## 9. 학습 포인트

- **Gate가 열려야 port에 닿습니다.** 호출자가 backend 단위 predicate를 읽는 동안에는 `.rocm`을 채워도 아무것도 바뀌지 않았습니다. Port table 위의 kernel별 predicate가 port가 부분적으로만 있는 상태를 다룰 수 있게 합니다.
- **그대로 옮긴 port가 fallback보다 느릴 수 있습니다.** 다른 하드웨어에서 맞춘 CUDA launch 모양은 SGY를 8에서 2로 바꾸기 전까지 gfx1151에서 `gather_qmm`에 졌습니다. Before/after 측정은 port PR 안에 있어야 합니다.
- **코드에 닿는 trace window를 고릅니다.** Fused kernel은 single-token forward에서만 실행되므로 `w8` trace는 같은 경로끼리 비교했을 것입니다. 변화가 보이는 것은 `w1` (state가 있는 decode는 `w1ctx512`)입니다.
- **Compile 시점 guard가 실제로 발동할 수 있는지 확인합니다.** Wave32 `#error`는 보호처럼 보였지만, `hipcc -E -dM`은 그 macro가 정의되지 않는다는 것을 보여 주었습니다.
- **측정 도구에도 버그가 있습니다.** Guard의 잘못된 거부는 두 unit에 걸쳐 시간을 잡아먹었고, 원인은 실제 경합이 아니라 kernel의 정리 순서였습니다. 그 구간을 재현하는 regression test가 수정이 유지되도록 합니다.

Refs: #2065, #1814, #2086, #2099, #2069, #2067, #1862, #1803, #1885, #1801, #886.
