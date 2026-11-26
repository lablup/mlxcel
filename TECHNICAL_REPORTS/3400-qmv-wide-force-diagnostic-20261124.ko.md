# 기술 보고서: PR #3400 - 15세대 미만에서 qmv_wide를 측정하는 `MLXCEL_QMV_WIDE=force`

**작성일**: 2026-11-24
**상태**: 열림 (머지 전 작성)
**위험도**: 낮음 (새 선택 값만 추가, 미설정·`0`·`1` 동작은 그대로)

## 요약

M1 Ultra에서 Gemma 4 12B를 MTP 어시스턴트와 함께 돌리면 설명형 프롬프트가 일반 디코딩보다 느렸다. 블록 4 검증 한 번이 일반 디코딩 약 3스텝에 해당하기 때문이다. 15세대 이상에서 여러 행짜리 양자화 행렬곱을 한 번에 처리하는 MLX `qmv_wide` 커널이 13세대에서도 도움이 되는지가 질문이었다. 업스트림은 15세대 미만의 affine 연산을 제외하고, `MLXCEL_QMV_WIDE`로는 qmv_wide를 끄는 것만 가능해서 답하려면 패치 후 재빌드가 필요했다. 이 PR은 이를 허용하는 `MLXCEL_QMV_WIDE=force`를 추가하고 답을 기록한다. M1 Ultra에서 강제로 켠 커널은 검증을 더 느리게 만들고(라운드당 약 68ms에서 94ms) 바이트 동일성을 깬다. 따라서 업스트림 기준이 그대로 맞다(#3398).

## 1. 문제 정의

Metal 오버레이 `quantized.cpp`의 `use_qmv_wide`는 `mode != "affine" || arch_gen >= 15`를 반환하며, affine qmv_wide는 15세대부터만 qmv보다 빠르다는 업스트림 주석이 붙어 있다. 그래서 M1(13세대)에서 4행 MTP 검증은 일반 `qmv`로 행마다 따로 돈다. Ultra 칩에서는 `qmv_mma`가 꺼져 있고 mlxcel의 `mlxcel_affine_qmm_t_rows`는 5행부터 맡기 때문이다. 이 기준값은 업스트림의 튜닝 상수라 pin을 올리면 낡을 수 있고, 14세대(M2)는 여기서 측정된 적이 없다.

## 2. 변경 요약

- `quantized.cpp` 오버레이: `mlxcel_qmv_wide_forced()`가 `MLXCEL_QMV_WIDE`를 한 번 읽어 `force`면 참을 반환하고, `use_qmv_wide`는 이 값이 설정되면 15세대 미만 affine도 받는다. 기존 플래그는 이미 `force`를 "켜짐"으로 읽으므로 exactness 게이트의 런타임 `mlxcel_set_qmv_wide`도 그대로 동작한다. 오버레이 헤더에 값과 측정 결과를 적었다.
- `overlays/metal/README.md`와 `src/lib/mlxcelverse/CMakeLists.txt`의 오버레이 메모에 유지할 로컬 변경으로 추가했다. MLX pin을 올릴 때 이 목록을 다시 적용한다.
- `docs/environment-variables.md`: 값, 적용 범위, 측정 결과, "서빙에 쓰지 말 것"을 추가했다.

## 3. 기술적 선택과 그 이유

**새 변수가 아니라 기존 변수의 값으로.** 이 판단은 이미 `MLXCEL_QMV_WIDE`가 맡고 있다. 변수를 하나 더 두면 같은 결정에 손잡이가 둘이 되고, 둘을 함께 줬을 때(예: `MLXCEL_QMV_WIDE=0`과 별도 force 변수)의 의미가 정해지지 않는다.

**qmv/qmv_wide 분기에만 적용.** `dispatch_qmm_t`에서는 `qmv_mma`와 `mlxcel_affine_qmm_t_rows`를 qmv 분기보다 먼저 시도한다. `force`는 이 둘의 행 범위를 건드리지 않으므로 업스트림이 제한한 커널만 정확히 측정한다.

**Rust 변경 없음.** `src/models/speculative_exactness.rs`의 `qmv_wide_pinned_by_operator()`는 값이 무엇이든 설정되어 있으면 운영자 고정으로 본다. 그래서 `force`에서는 게이트가 좁은 커널 재시도를 건너뛰고, 검사가 실패하면 MTP를 끈다. 진단용으로는 이것이 맞는 기본값이다. 출력이 바뀔 수 있는 경우는 `MLXCEL_MTP_ALLOW_INEXACT=1`을 함께 줬을 때뿐이고, 그때는 이미 큰 경고가 나온다.

## 4. 측정

M1 Ultra, `gemma-4-12b-it-4bit`(HF 리비전 `73bcf09`)와 `gemma-4-12b-it-assistant-bf16`, 블록 4, temperature 0, 400토큰 숫자 나열 프롬프트, 워밍업 후 값마다 1회:

| `MLXCEL_QMV_WIDE` | 검사 | MTP | 검증 ms / 100라운드 | 디코딩 tok/s |
|---|---|---|---:|---:|
| 미설정 | 통과 | 사용 | 6771 | 50.3 |
| `0` | 통과 | 사용 | 6768 | 50.2 |
| `1` | 통과 | 사용 | 6900 | 49.6 |
| `force` | 실패 (layer 0 attention) | 꺼짐 | 해당 없음 | 일반 디코딩 |
| `force` + `MLXCEL_MTP_ALLOW_INEXACT=1` | 무시하고 진행 | 사용 | 9397 | 37.1 |

임시 빌드로 번갈아 돌린 이전 측정에서는 강제 커널이 라운드당 91ms, 기존 커널이 68ms였다. 설명형 프롬프트는 26 대 33 tok/s였고 출력이 156번째 문자부터 달라졌다. `M = 1`은 `qmv_wide`를 타지 않으므로 일반 디코딩에는 영향이 없다.

## 5. 한계

- 14세대(M2)는 아직 측정하지 않았다. 이 PR로 변수 하나만 바꿔 돌리면 된다.
- 체크포인트 하나, 블록 크기 하나에서만 측정했다. 검사 결과는 연산 형태마다 다르므로 다른 모델은 어디서 달라지는지가 다를 수 있지만, qmv_wide가 빠른지 여부까지 달라질 근거는 없다.
- M1에서 MTP가 느린 문제 자체는 해결하지 않았다. 후보는 `mlxcel_affine_qmm_t_rows`를 4행까지 넓히는 것이고, 별도 측정이 필요하다.
