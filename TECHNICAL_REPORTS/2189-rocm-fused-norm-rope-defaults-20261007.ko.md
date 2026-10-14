# 기술 보고서: PR #2189 - ROCm에서 Fused Add-RMSNorm과 RoPE-Append를 기본으로 켜기

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 헤드 `9874c3c4`(origin/main `fc21784f` 위로 리베이스), PR 열림, 머지 대기. #2145를 닫음(#1801의 일부).

**언어**: Rust(mlxcel-core `layers.rs`; mlxcel `models/llama3.rs`, `models/llama3_tests.rs`), Python(`scripts/rocm_decode_profile.py`, `tests/test_rocm_decode_profile.py`), Markdown(README, `docs/environment-variables.md`, `docs/installation.md`, `docs/benchmarks.md`, 결과 페이지 두 개), CSV(벤치마크 파일 두 개)

**위험도**: 낮음. 이 PR이 켜는 두 HIP 포트는 대체하는 ROCm 그래프와 바이트 단위로 동일하므로, 트레이스를 뜬 모델에서는 기본값 변경이 logit을 하나도 바꿀 수 없습니다. Metal과 CUDA 빌드는 이전과 같은 `false` 상수를 컴파일합니다. 남은 위험은 로컬 체크포인트 없이 ROCm에서 기본으로 norm 포트를 타게 된 모델 계열(Gemma, IQuest Loop Coder)에 있습니다.

## 요약

PR #2107(#2063 해결)은 `fused_add_rms_norm`과 `fused_rope_qk_append`를 HIP로 포팅했지만 둘 다 opt-in으로 남겼습니다. gfx1151에서 디코드 이득(Llama 3.1 8B +0.3%, Qwen2.5 7B 최대 +1.7%)이 실행 간 편차 안에 있었기 때문입니다. 이후 메인테이너는 #2145에서 둘 다 ROCm에서만 기본으로 켜기로 결정했습니다. 근거는 비대칭입니다. 포트는 정확도 비용이 없고(on/off logit 트레이스 쌍이 모두 바이트 동일), 느려진 중앙값이 하나도 없었으므로, 이득을 노이즈와 구분할 수 없더라도 기본으로 켜서 잃을 것이 없습니다. 이 PR은 그 결정을 구현한 것이지 성능 향상 주장이 아닙니다.

로직 변경은 두 줄입니다. `src/lib/mlxcel-core/src/layers.rs`의 `FUSED_ADD_RMSNORM_DEFAULT`와 `FUSED_ROPE_APPEND_DEFAULT`가 `cfg!(feature = "rocm")`이 됩니다. 환경 변수는 여전히 양방향으로 덮어씁니다. 부작용 하나는 실제로 고쳐야 했습니다. Llama 3.1 `rope_scaling` 우회 안내는 RoPE 커널이 요청됐는지를 게이트에 물었기 때문에, 기본값이 켜지면 ROCm의 모든 Llama 3.1 실행에서 출력됐을 것입니다. 새 `layers::fused_flag_explicit_value`로 이 안내는 `MLXCEL_FUSED_ROPE_APPEND`가 명시적으로 참 값일 때만 출력됩니다.

기본값이 켜진 빌드를 gfx1151에서 다시 측정했습니다. `=0` 대비 Llama 3.1 -0.2%, Qwen2.5 +0.8%로 둘 다 노이즈입니다. `w8` logit 트레이스는 기본값과 두 변수 모두 `0`인 경우가 바이트 동일하고, `rocprofv3`는 기본 경로가 포팅된 커널을 디스패치함을 보여 줍니다. `9874c3c4`에서 유닛의 `make verify-rocm`은 OK를 출력했고 11,996 통과, 0 실패, 383 무시였습니다. 오케스트레이터는 머지 전에 같은 헤드에서 전체 게이트를 다시 실행했습니다.

## 1. 문제 정의

### 1.1 #2107 이후의 상태

`FUSED_ADD_RMSNORM_DEFAULT`와 `FUSED_ROPE_APPEND_DEFAULT`는 모든 백엔드에서 그냥 `false`였습니다. 문서 주석에 이유가 기록되어 있었습니다. Metal(M1 Ultra)에서는 op 수준 마이크로벤치와 디코드 측정 모두 fusion을 정당화하지 못했고, RoPE fusion에는 재현되는 패리티 미만 셀이 하나 있었습니다(hidden 8192, batch 1, 0.89x에서 0.94x). #2063의 ROCm 측정은 "확실한 이득이 아니므로 ROCm도 공통 기본값을 유지한다"는 결론과 함께 그 주석에 덧붙여졌습니다.

#2107의 ROCm 수치:

| 모델 | 구성 | Off | On | 변화 |
|---|---|---|---|---|
| Llama-3.1-8B | norm만(RoPE는 `rope_scaling` 때문에 우회) | 37.85 | 37.95 | +0.3% |
| Qwen2.5-7B | norm / RoPE / 둘 다 | 46.53 | 47.23 / 46.80 / 47.32 | +1.5% / +0.6% / +1.7% |
| Qwen3-30B-A3B | 대조군 | | | +0.3% |

Qwen2.5의 7개 라운드 중 6개에서 on이 off보다 높았지만, 이득의 크기는 off 쪽 편차만큼 실행마다 흔들렸습니다.

### 1.2 결정

#2145에서 메인테이너의 판단은 속도가 아니라 정확성에 근거합니다. 두 포트는 ROCm에서 그래프와 바이트 동일한 출력을 냅니다(`w1`, `w8`, `w1ctx512`의 모든 쌍, Metal 대비 판정 불일치 0/230과 0/274). 토큰을 바꿀 수 없고 중앙값 기준으로 한 번도 느리지 않았던 기본값은 그 백엔드에서 손해가 없으며, 기본으로 켜면 변수의 존재를 아는 사람만이 아니라 모든 ROCm 사용자가 포팅된 커널을 실행하게 됩니다. Metal은 상황이 다릅니다(RoPE fusion에 측정된 회귀 셀이 있고, Metal norm 커널은 반올림 한 단계만큼 그래프와 다를 수 있습니다). 그래서 Metal과 CUDA는 꺼진 상태로 둡니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlxcel-core/src/layers.rs` | 두 `FUSED_*_DEFAULT` 상수가 `cfg!(feature = "rocm")`이 됨. 문서 주석을 백엔드별로 다시 쓰고 #2107 수치와 결정을 기록. `fused_flag_enabled_from`은 새 공개 함수 `fused_flag_explicit_value(value) -> Option<bool>`에 위임. 새 테스트 `fused_905_defaults_are_on_for_rocm_builds_only`, `fused_flag_explicit_value_separates_a_setting_from_the_default` |
| `src/models/llama3.rs` | `fused_rope_launcher_requested`가 `MLXCEL_FUSED_ROPE_APPEND`에 대해 게이트 대신 새 `fused_rope_append_explicitly_requested(value)`를 사용. 안내와 런처 순서 문서 주석 갱신 |
| `src/models/llama3_tests.rs` | `the_rope_append_bypass_notice_needs_an_explicit_truthy_value` |
| `scripts/rocm_decode_profile.py`, `tests/test_rocm_decode_profile.py` | #2063 역할이 기본값으로 도달하는 것으로 집계됨. 테스트는 `()` 대신 기본 튜플을 검사 |
| 문서 | README ROCm 항목, `environment-variables.md` 기본값 열 "off (Metal, CUDA) / on (ROCm)", `installation.md`의 ROCm 행, `benchmarks.md`, `rocm-correctness-gfx1151-2026-09-30.md`의 메모, `rocm-fused-norm-rope-gfx1151-2026-10-05.md`에 재측정을 담은 Decision 섹션 |
| `benchmarks/` | `rocm_strixhalo-gfx1151_2026-10-07_fused-default-{on,off}.csv` |

커밋은 세 개입니다. 안내 수정과 테스트를 포함한 기본값 전환(`c1309a0e`), 재측정(`3f5783c6`), README, 문서, 디코드 프로파일 정렬(`9874c3c4`). 13개 파일, 197줄 추가, 65줄 삭제입니다.

## 3. 설계

### 3.1 상수에 `cfg!(feature = "rocm")`

```rust
pub(crate) const FUSED_ADD_RMSNORM_DEFAULT: bool = cfg!(feature = "rocm");
pub(crate) const FUSED_ROPE_APPEND_DEFAULT: bool = cfg!(feature = "rocm");
```

여기의 `rocm` feature는 mlxcel-core 자체의 것(`src/lib/mlxcel-core/Cargo.toml`)이고, 루트 `rocm` feature가 이를 켭니다. ROCm 빌드에는 Metal이나 CUDA 백엔드가 컴파일되지 않으므로 빌드 플래그가 곧 백엔드를 식별합니다. 이는 같은 이유로 C++ 쪽에서 `#ifdef MLXCEL_BRIDGE_ROCM_BACKEND` 아래 ROCm `MLXCEL_FUSED_MOE_SGY` 기본값을 2로 정한 PR #2098을 따른 것입니다.

이슈가 배제한 대안:

- **런타임 백엔드 비교**(`gpu_kernel_backend() == Rocm`) 또는 `metal_is_available() || cuda_is_available()` 게이트. `scripts/ci/check_kernel_port_dispatch.py`(`verify-kernel-port-dispatch` 게이트)가 커널 포트 디스패치 코드에서 이런 패턴을 거부하며, 컴파일러가 이미 아는 답을 위해 런타임 분기를 추가하게 됩니다.
- **별도의 ROCm 상수나 ROCm 전용 게이트 함수.** `OnceLock` 게이트와 파서를 중복시키고 두 백엔드가 어긋날 여지를 만듭니다. `cfg!`를 쓰면 fusion마다 상수 하나, 뒤집을 곳 하나가 유지되어 문서 주석이 말하는 "여기서 뒤집는다" 계약이 지켜집니다.

`#[cfg]` 항목 대신 `cfg!`를 쓰면 모든 빌드에서 두 값이 타입 검사되고, 테스트가 `cfg!(feature = "rocm")`과 직접 비교할 수 있어 각 백엔드의 테스트 실행이 자기 기본값을 검사합니다.

### 3.2 양방향 환경 변수 오버라이드

`fused_flag_enabled_from`의 계약은 그대로입니다. 모든 백엔드에서 `0`/`false`/`off`/`no`는 끄고 `1`/`true`/`on`/`yes`는 켭니다(대소문자 무시, 공백 제거). 설정되지 않았거나 인식할 수 없는 값은 컴파일된 기본값을 따르므로 오타가 디코드 그래프를 조용히 바꾸지 않습니다. ROCm에서는 `MLXCEL_FUSED_ADD_RMSNORM=0 MLXCEL_FUSED_ROPE_APPEND=0`으로 그래프로 돌아가고, Metal과 CUDA에서는 여전히 `=1`로 opt-in합니다. 게이트는 여전히 각 변수를 `OnceLock`에 한 번 읽으므로 설정은 첫 디코드 전에 되어 있어야 합니다.

리팩터링은 파서를 나눕니다. `fused_flag_explicit_value(value)`는 `Some(false)`, `Some(true)`, `None` 중 하나를 반환하고, `fused_flag_enabled_from`은 이제 `fused_flag_explicit_value(value).unwrap_or(default)`입니다. 동작은 동일하며, 호출자가 "켜져 있는가"와 별개로 "사용자가 설정했는가"를 물을 수 있도록 나눈 것입니다.

### 3.3 Llama 3.1 우회 안내

Llama 3.1의 `rope_scaling`은 주파수 테이블을 만들고, fused RoPE 커널은 이를 받을 수 없으므로 `Attention::forward`는 플래그와 상관없이 커널을 우회합니다. `report_fused_rope_bypass_once`는 우회 사유가 성립하고 어떤 변수가 fused RoPE 런처를 요청했을 때 stderr에 한 번 안내를 출력합니다. 그 문서 주석은 변수를 설정하지 않은 사용자는 안내를 보지 않는다고 약속했습니다.

이 PR 이전에 `fused_rope_launcher_requested`는 `MLXCEL_FUSED_ROPE_APPEND`에 대해 게이트(`fused_rope_append_enabled()`)에 물었습니다. 기본값이 꺼져 있을 때는 이것이 맞았습니다. 게이트는 사용자가 opt-in했을 때만 참이었고, 존재 여부 대신 게이트에 물었기 때문에 `=0`을 잃어버린 커널로 보고하지 않았습니다. ROCm 빌드에서 기본값이 켜지면 설정하지 않은 모든 실행에서 게이트가 참이므로, ROCm의 모든 Llama 3.1 실행이 안내를 출력했을 것입니다.

수정은 다른 질문을 던집니다. `fused_rope_append_explicitly_requested(value)`는 `fused_flag_explicit_value(value) == Some(true)`이므로 안내는 명시적인 참 값에서만 출력됩니다. 미설정, 인식 불가 값, `0`은 모든 백엔드에서 조용합니다. 목록의 나머지 두 변수(`MLXCEL_FUSED_CAUSAL_PREFILL`, `MLXCEL_ENABLE_FUSED_QKV_SPLIT_ROPE`)는 존재만으로 켜지는 변수이며 바뀌지 않았습니다.

한 가지 결과: ROCm에서 `MLXCEL_FUSED_ROPE_APPEND=maybe` 같은 인식 불가 값은 기본값(on)을 따르지만 안내를 출력하지 않습니다. 사용자가 커널을 분명히 요청한 것이 아니므로 의도와 맞습니다.

### 3.4 테스트

- `fused_905_defaults_are_on_for_rocm_builds_only`는 두 상수가 `cfg!(feature = "rocm")`과 같은지 확인한 뒤, `#[cfg(feature = "rocm")]`에서는 미설정이 on이고 `0`이 off인지, `#[cfg(not(...))]`에서는 미설정이 off이고 `1`이 on인지 확인합니다. 각 백엔드의 테스트 실행이 자기 기본값과 그 백엔드에서 중요한 방향의 오버라이드를 고정합니다.
- `fused_flag_explicit_value_separates_a_setting_from_the_default`는 `None`, 인식 불가 값(`""`, `maybe`, `2`), 인식되는 각 집합의 모든 철자, 공백과 대소문자를 다룹니다.
- `the_rope_append_bypass_notice_needs_an_explicit_truthy_value`는 기본 ROCm Llama 3.1 실행이 항상 겪는 미설정 경우와 거짓, 참 집합을 다룹니다.
- 기존 `fused_905_flags_follow_the_default_and_respect_explicit_values`와 패리티 테스트는 이미 리터럴이 아니라 상수를 읽으므로 수정이 필요 없었습니다.

### 3.5 디코드 프로파일 스크립트

`scripts/rocm_decode_profile.py`는 각 ROCm 커널 포트 유닛이 어떤 디코드 역할에 도달하는지 분류합니다. #2063에 대해 기본값은 `()`를 반환하고 역할은 opt-in으로만 나열했습니다. 이제 둘 다 같은 튜플을 반환합니다. `add_rms_join_post_attn`에, 체크포인트에 `rope_scaling` 테이블이 없으면 `rope_append`가 더해지며, 메모는 `cfg!` 상수와 #2145를 가리킵니다.

## 4. 재측정

gfx1151, `scripts/bench_decode.sh` pp512/tg128, 기본값(변수 없음) 대 `MLXCEL_FUSED_ADD_RMSNORM=0 MLXCEL_FUSED_ROPE_APPEND=0`, 라운드마다 순서 교대, 3회 중앙값. 모든 실행은 #2146의 호스트 전역 잠금과 함께 `scripts/rocm_gpu_guard.sh`를 거쳤고, 12회 모두 첫 시도에 깨끗했습니다. 측정은 첫 커밋의 리베이스 전 SHA(`57b59629`)에서 했습니다.

| 모델 | 기본값(on) | `=0`(off) | 변화 | 쌍별 차이(on - off), 라운드별 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct-4bit | 38.09 | 38.15 | -0.2% | +0.02 / -0.49 / -0.05 |
| Qwen2.5-7B-Instruct-4bit | 47.90 | 47.51 | +0.8% | +0.39 / +0.30 / -0.04 |

두 변화 모두 실행 간 노이즈 안에 있습니다. Llama 3.1은 norm 포트만 실행하며, -0.2%는 어느 쪽의 다른 모든 실행보다 높은 off 실행 하나(38.58)에서 나왔습니다. Qwen2.5는 두 포트를 모두 실행하며 세 라운드 중 두 번 off보다 약간 높았습니다. 어느 수치도 속도 향상 주장이 아니고, 애초에 속도 향상에 기대지 않은 결정에 반하는 것도 아닙니다.

Qwen2.5 prefill은 모든 라운드에서 기본값 쪽이 2.7%에서 6.9% 높았습니다(중앙값 1609.96 대 1544.08). #2107 측정에서는 보이지 않았고 세 라운드로는 편차와 구분할 수 없으므로, 결과 페이지는 결론 없이 기록만 합니다.

## 5. 검증

모두 gfx1151에서:

- **바이트 동일성.** 두 모델의 `logit_trace` `w8`, 기본값 대 두 변수 모두 `0`: 파일이 바이트 동일(`cmp`).
- **기본 경로가 커널에 도달함.** 변수 없이 8토큰 `mlxcel generate`에 `rocprofv3 --kernel-trace`: Qwen2.5는 `fused_add_rms_norm`과 `fused_rope_qk_append`를 각각 252회(28 레이어 x 9 forward) 디스패치했고 그래프 RoPE 커널은 없었습니다. 두 변수가 `0`이면 대신 `rope_single_1d`를 디스패치했습니다. Llama 3.1은 norm 포트를 288회(32 x 9) 디스패치했고 `rope_single_freqs_1d` / `rope_freqs`를 유지했습니다. 이것이 상수만이 아니라 바이너리에서 전환이 효과를 냈다는 증거입니다.
- **우회 안내.** Llama 3.1에서 기본값일 때 없음, `MLXCEL_FUSED_ROPE_APPEND=1`일 때 있음, `=0`일 때 없음.
- **대상 테스트.** `--features rocm`에서 `layers::tests::fused`, `fused_norm_parity_tests`, `fused_rope_parity_tests`: 32 통과. `models::llama3::`: 27 통과, 두 변수가 `0`일 때도 통과하는 `greedy_decode_is_token_identical_to_the_unfused_baseline` 포함. `tests.test_rocm_decode_profile`: 26 통과.
- **스크립트 게이트.** `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: 통과. `verify-kernel-port-dispatch` 통과는 런타임 백엔드 비교가 들어오지 않았다는 확인입니다.
- **유닛의 `make verify-rocm`** (`9874c3c4`, `fc21784f` 위로 리베이스): OK. Clippy와 ROCm 스모크 테스트 통과, 152개 테스트 바이너리에서 11,996 통과, 0 실패, 383 무시.
- **오케스트레이터의 `make verify-rocm`**: 머지 전에 같은 헤드에서 다시 실행.

검증하지 않음: Metal과 CUDA(이 호스트에 없음). 그 빌드에서는 두 상수가 이전과 똑같이 `false`로 평가되고, 안내 변경은 출력 조건을 좁히기만 합니다.

## 6. 기술적 선택과 그 이유

- **속도가 아니라 정확성 근거로 ROCm 기본값 켜기.** 바이트 동일하고 중앙값으로 한 번도 느리지 않은 포트는 손해가 없습니다. 이 결정은 fusion이 더 빠르다는 주장이 명시적으로 아닙니다.
- **Metal과 CUDA는 끈 채로 유지.** Metal 측정에는 RoPE fusion의 재현되는 회귀 셀이 있었고, Metal norm 커널은 잔차 합의 반올림만큼 그래프와 다를 수 있습니다. #2107도 이 PR도 그 근거를 바꾸지 않습니다.
- **빌드 feature가 곧 백엔드.** `cfg!(feature = "rocm")`은 PR #2098의 선례를 따르고, 런타임 분기를 추가하지 않으며, `verify-kernel-port-dispatch`를 통과시킵니다.
- **fusion마다 상수 하나, 파서 하나.** 게이트의 ROCm 전용 복제가 없으므로 두 백엔드가 파싱이나 캐싱에서 어긋날 수 없습니다.
- **"명시적으로 설정됨"과 "켜짐"을 분리.** 우회 안내는 앞의 질문이 필요하고 게이트는 뒤의 질문에 답하기 때문에 `fused_flag_explicit_value`가 생겼습니다. 기본값이 백엔드마다 달라지면 두 질문은 더 이상 같지 않습니다.
- **opt-in이 아니라 출시되는 기본값을 재측정.** 새 측정은 기본값을 `=0`과 비교합니다. 이제 ROCm 사용자가 마주하는 비교가 이것이며, 원래 #2107 수치 옆에 기록됩니다.

## 7. 남은 위험과 후속 작업

- **검증되지 않은 모델 계열.** Gemma와 IQuest Loop Coder도 `fused_add_rms_norm`을 호출하고(`src/models/gemma.rs`, `src/models/iquestloopcoder.rs`) 이제 ROCm에서 기본으로 norm 포트를 타지만, 이 호스트에는 둘 다 체크포인트가 없습니다. Gemma의 `(1 + w)` 가중치 규약은 logit 트레이스나 생성 실행이 아니라 `fused_norm_parity_tests`의 허용 오차 테스트로만 다뤄집니다. 텍스트 백본이 `Llama3Model`이나 Qwen2 경로인 VLM도 기본값을 물려받습니다. 트레이스를 뜬 Llama/Qwen2.5 코드 경로를 공유하지만 개별 실행은 하지 않았습니다. Gemma 체크포인트로 트레이스를 뜨는 것이 분명한 후속 작업입니다.
- **Wave64(CDNA) 미검증.** 모든 측정과 트레이스는 RDNA 3.5 gfx1151(wave32)에서 했습니다. 기본값은 이제 빌드가 지원하는 모든 ROCm 타깃에 적용됩니다.
- **Qwen2.5 prefill 변화.** 2.7%에서 6.9%의 prefill 증가는 설명되지 않았고 편차일 수 있습니다. 더 긴 측정으로 판가름할 수 있습니다.
- **이득은 노이즈 수준.** 이후 ROCm 변경으로 그래프 경로가 포트보다 빨라지면 기본값을 다시 봐야 합니다. `MLXCEL_FUSED_ADD_RMSNORM=0` / `MLXCEL_FUSED_ROPE_APPEND=0`은 여전히 차단 스위치이고, 상수는 여전히 뒤집을 유일한 곳입니다.

## 8. 학습 포인트

- **바이트 동일한 포트는 기본값의 질문을 바꾼다.** 대체 커널이 logit 하나도 바꿀 수 없다면 "확실히 더 빠른가"는 잘못된 기준이고 "한 번이라도 느린가"로 충분하며, 기본값은 그에 따를 수 있습니다.
- **게이트에 묶인 안내는 기본값이 바뀌면 의미가 바뀐다.** 게이트가 "사용자가 opt-in했다"를 뜻하는 동안에는 게이트에 묻는 것이 맞았습니다. 기본값을 뒤집자 게이트의 의미가 조용히 바뀌었고, 게이트에서 사용자 의도를 추론하던 코드는 명시적 값을 읽도록 바꿔야 했습니다.
- **기본 경로를 바이너리에서 증명하라.** 단위 테스트는 상수를 고정하지만, 기본 빌드가 실제로 포팅된 커널을 실행하고 Llama 3.1이 그래프 RoPE를 유지한다는 것을 보여 주는 것은 `rocprofv3` 디스패치 횟수(252와 288)입니다.
- **빌드 시점의 질문은 컴파일러에게 맡겨라.** 빌드 플래그가 이미 백엔드를 결정한다면 `cfg!`가 런타임 백엔드 검사보다 단순하고 안전하며, 저장소의 디스패치 게이트가 그 패턴을 강제합니다.
