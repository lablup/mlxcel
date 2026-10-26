# PR #2631: bf16 addmm 및 vision oracle을 백엔드별 실제 계산 결과에 맞춤

**Date**: 2026-10-26
**Status**: Metal(M1 Ultra)에서 완료. ROCm, CUDA, Metal M3 이상 기대값은 소스 분석 기반이며 실측하지 않음
**Languages**: Rust, Markdown
**Risk Level**: Low (테스트만 변경)

## 요약

PR #2631(이슈 #2626)은 Apple Silicon에서 실패하던 vision 테스트 약 10개를 고칩니다. 처음에는 flaky로 기록했습니다. 직렬 실행에서도 대부분 같은 결과로 실패했고, 조사해 보니 백엔드에 따라 결과가 달라지는 oracle 세 종류와 실제 테스트 경합 하나였습니다. 프로덕션 코드에는 잘못이 없습니다. oracle은 ROCm gfx1151과 x86_64 Linux CPU에서 캡처한 값이었고(PR #2551, #2578), Metal은 그중 일부를 정당하게 다르게 계산합니다. 이제 각 oracle은 테스트가 감지할 수 있는 속성을 기준으로 기대값을 고릅니다. `addmm` 경로 회귀를 가릴 만한 허용 오차로 단정문을 느슨하게 만들지 않았습니다.

## 1. 문제 정의

`main`의 M1 Ultra에서 다음과 같이 실패했습니다.
- `--lib dinov2_layers::policy`는 매번 5개 모두 실패했습니다.
- `--lib vision::dense`는 병렬 실행과 `--test-threads=1` 모두에서 5개가 실패했습니다. 4개는 항상 같았고 나머지 1개는 실행마다 바뀌었습니다.

이 테스트들은 vision linear 레이어가 `mlxcel_core::addmm`을 거치는지 증명합니다. 그 증명은 bf16 halfway 사례에 기댑니다. 이 사례에서는 `x @ w.T + b`를 한 번에 반올림한 값과, 곱을 반올림한 뒤 합을 다시 반올림한 값이 다릅니다.

## 2. 원인

| 실패 묶음 | 원인 |
|---|---|
| addmm 경로 oracle (9개) | MLX의 Metal `AddMM::eval_gpu`는 `min(M, N) == 1`이면 `gemv_axbpy`로 보냅니다. 이 커널의 `GEMVKernel` epilogue(`gemv.h`)는 bias를 더하기 전에 F32 누산값을 bf16으로 캐스팅합니다. 그래서 Metal에서 한 행짜리 곱은 두 번 반올림되고, 결과가 bias를 따로 더한 것과 같습니다. 두 행 이상이면 steel GEMM이나 `gemv_wide`를 써서 한 번만 반올림합니다. CPU와 ROCm은 행 수와 무관하게 한 번만 반올림합니다. 같은 장비의 upstream `mlx` 0.32.4도 똑같이 갈립니다(1행: 1.9375, 2행 이상: 1.9296875). 따라서 mlxcel 결함이 아니라 MLX 동작입니다. 이 커널은 이전 MLX pin과 새 pin에서 바이트 단위로 같으므로 #2614 bump가 원인이 아닙니다. |
| Sapiens2 compiled RoPE | fixture 값은 ROCm에서 캡처했습니다. Metal과 CPU에서는 compiled 그래프와 eager 그래프의 결과가 서로 같고, 저장소의 생성 스크립트도 이 호스트에서는 fixture 작성을 거부합니다. |
| SAM 3D Body MHR CPU geometry | fixture는 glibc 환경에서 캡처했습니다. macOS의 MLX CPU는 atan2, sin, cos를 Apple 수학 라이브러리로 계산하므로, 초월함수 원소 388개 중 27개가 1~2 ULP 다릅니다. atan2 하나만 쓰는 원소는 이 호스트에서 upstream `mx.arctan2`가 mlxcel과 정확히 같은 값을 냅니다. |
| 실행마다 바뀐 다섯 번째 실패 | MHR 테스트가 CPU `DefaultDeviceGuard`를 잡고 있었습니다. `lock_default_device()`를 잡지 않은 디바이스 의존 테스트가 그 구간에 CPU에서 실행되기도 했습니다. MHR이 통과하면서 guard를 잡는 시간이 길어지자, MoGe3 refiner와 batched-decode wrapper 테스트의 flaky도 드러났습니다. 병렬 24회 중 13회 실패했고, MHR을 건너뛰면 24회 중 0회였습니다. |

## 3. 기술적 선택과 그 이유

### 3.1 두 행으로 경로를 증명하고, 한 행 값은 백엔드별로 고정

경로를 증명하려면 `addmm`과 `matmul + add`의 결과가 다른 shape가 필요합니다. Metal에서 한 행이면 두 결과가 같아서 oracle이 둘을 구분하지 못합니다. 그렇다고 단정문을 느슨하게 하면 실제 회귀도 가려집니다. 새 `src/vision/addmm_oracle_test_support.rs`는 같은 행 두 개로 각 oracle을 실행합니다. 이 shape에서는 모든 백엔드가 한 번만 반올림하고, bias를 따로 더하는 경로는 여전히 다른 값을 냅니다. 각 테스트는 한 행 값도 백엔드별로 고정합니다. Metal GPU에서는 bias를 따로 더한 값이고, 다른 백엔드에서는 fused 값입니다. 그래서 MLX의 커널 선택이 바뀌면 조용히 통과하지 않고 실패합니다.

### 3.2 Linux가 아닌 호스트에서만 초월함수 필드에 한계 적용

MHR oracle은 Linux에서, 그리고 모든 호스트의 초월함수가 아닌 필드에서 bit-exact를 유지합니다. macOS에서 다시 캡처하면 macOS에서도 bit-exact로 맞출 수 있지만, 그러려면 upstream sam3d_body 코드를 실행해야 합니다. 이번 실행에서는 그것이 허용되지 않았습니다. 그래서 Linux가 아닌 호스트에서는 초월함수 필드 네 개만 2 ULP 이내로 제한합니다. 한계를 둔 단정문은 이것 하나뿐입니다.

### 3.3 공용 harness가 아니라 각 테스트에서 잠금

`lock_default_device()`를 `batched_decode_harness` 안에 넣으면 교착 상태가 됩니다. 호출자가 이미 `mlx_test_guard()`를 잡고 있고, 이 잠금은 재진입할 수 없기 때문입니다. 그래서 디바이스에 의존하는 테스트마다 따로 잠금을 잡습니다.

## 4. 검증

모든 실행은 M1 Ultra에서 `--features metal,accelerate` release 빌드로 했습니다.
- `--lib vision::dense`: 두 모드 모두 3회 중 3회 192개 통과했고, 빌드된 바이너리로 40회 중 40회 통과했습니다.
- `--lib dinov2_layers::policy`: 두 모드 모두 3회 중 3회 5개 통과했습니다.
- 최신 `main`을 병합한 뒤 오케스트레이터가 `vision::`을 세 번 다시 실행했고, workspace clippy와 fmt도 실행했습니다.

`#[cfg(test)]` 코드와 테스트 fixture만 바뀌었으므로 실제 체크포인트 실행은 필요하지 않았습니다.

## 5. 학습 포인트

- flaky라는 판단은 받아들이기 전에 직렬 실행으로 확인해야 합니다. 이번에는 직렬 실행이 결정적인 백엔드 차이와 실제 경합 하나를 갈라냈습니다.
- 한 백엔드에서 캡처한 oracle에는 그 백엔드의 커널 선택이 들어 있습니다. 값이 일부 백엔드에서만 맞는다면 테스트에 어느 백엔드인지 적어야 합니다.
- Metal에서 한 행짜리 bf16 `addmm`은 두 번 반올림합니다. 양자화하지 않은 bf16 레이어의 프로덕션 decode(M = 1)에도 해당하고, Metal의 upstream MLX와 같은 동작입니다. 따라서 mlx-lm과의 parity는 유지되지만, M = 1에서 CPU나 ROCm 캡처와 bit-exact로 비교하면 맞지 않습니다.

## 6. 후속 조치와 제한 사항

- ROCm gfx1151, CUDA, Metal M3 이상(`gemv_wide`), x86_64가 아닌 Linux에서는 실행하지 않았습니다. 이 환경의 기대값은 소스를 읽고 정했습니다.
- upstream 코드로 macOS에서 MHR oracle을 다시 캡처하면 2 ULP 한계를 없앨 수 있습니다.

## 참고 자료

- 이슈 #2626, PR #2551, #2578, #2614
- `src/vision/addmm_oracle_test_support.rs`, `src/vision/encoders/dinov2_policy_tests.rs`, `src/vision/dense/sam3d_body/mhr_tests.rs`
