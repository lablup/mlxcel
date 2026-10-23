# PR #2577: MoGe3 밀집 기하 예측 추가

**작성일**: 2026-10-23
**상태**: 완료
**언어**: Rust, C++, Python, Markdown
**위험도**: 높음

## 요약

PR #2577은 공개된 MoGe3 ViT-L 및 ViT-g 단안 기하 모델을 mlxcel에 이식한다. 정지 이미지 한 장이 공유 DINOv2 인코더, 합성곱 디코더, 카메라 복원, 희소 3D 리파이너를 거쳐 원본 해상도의 포인트, 깊이, 유효성, 표면 법선, 정규화 카메라 내부 파라미터를 생성한다. 이 모델군은 `mlxcel predict`에 등록되며, 이후 SAM 3D Objects 통합에서 사용할 공개 `Moge3` 라이브러리 API를 제공하고, 업스트림의 모든 추론 옵션과 선택적 리파인 단계별 직렬화를 지원한다.

구현에는 정확한 정수 `searchsorted` 브리지, 한도가 있는 희소 연산, 엄격한 설정/체크포인트/입력 검증, 변경되지 않은 업스트림 소스와 exact-native MLX에 고정된 인증 참조 하네스가 포함된다. 집중 단위 테스트와 정수 경계 브리지 테스트는 통과했다. gfx1151 순차 패리티는 두 공개 체크포인트와 필요한 다섯 옵션 사례 모두에서 통과했고, 최대 오차는 명시된 임계값보다 충분히 작으며 마스크와 비유한 값의 지지도 정확히 일치한다. 추출된 최종 소스는 라이브러리 컴파일, guarded 단위 테스트 실행, named integration 링크, CLI 컴파일, 경고를 거부하는 clippy, 현재 HEAD PR CI를 통과했다. 실제 전체 모델 GPU 및 기본 CLI 근거는 검증된 추출 전 바이너리에 연결된다. 독립 검토에서 중간의 모듈 추출, iterator 수정, 포맷 변경이 연산 그래프를 보존함을 확인했으며, 최종 바이너리에는 컴파일/링크/단위 테스트 근거가 있지만 별도 수치 결과는 없다. 인증된 전체 해상도 Python CPU 참조는 완전한 상태로 복구됐지만 외부 wrapper는 종료 코드 2였고 Python 자식 종료 코드는 기록되지 않았다. 불변 Rust CPU 캡처는 확인된 CPU 백엔드에서 종료 코드 0으로 완료됐고 고정된 식별자 및 스키마 검증기를 통과했으며 변경하지 않은 임계값에서 참조 출력 17개 모두와 일치했다. CPU 수치 결과는 검증된 추출 전 바이너리와 보관된 소스에 연결되고, 연산 그래프가 같은 추출 및 정리 변경은 최종 소스 컴파일, 링크, 단위 테스트, clippy, 현재 HEAD CI 근거로 다룬다.

## 1. 문제 정의

MoGe3는 RGB 이미지 한 장에서 서로 연결된 다섯 기하 결과를 예측한다. 카메라 공간 포인트맵, 깊이, 표면 법선, 유효성, 카메라 내부 파라미터다. 이 PR 전에는 mlxcel의 밀집 예측 표면이 이 출력 유형들을 직렬화할 수 있었고 DINOv2 구현이 백본을 제공할 수 있었지만, `model_type: moge3`가 인식되지 않았고 희소 3D 리파인 경로가 없었다.

이 이식에는 정확성에 민감한 구성요소가 여러 개 있다.

- MoGe3는 Python의 ties-to-even 반올림으로 토큰 그리드를 정하고 `resolution_level` 또는 명시적 `num_tokens`를 받는다.
- 희소 풀링과 submanifold 합성곱은 정확한 정수 키, 중복을 처리하는 `searchsorted`, 좌표 순서, ceil-div 크기, 부모 인덱스 의미에 의존한다.
- 카메라 복원은 초점 거리와 깊이 이동량을 반복 계산한다. 고정 FOV 경로와 복원 FOV 경로는 구분되어야 한다.
- 공개 체크포인트에는 F32 DINOv2-L/14 또는 DINOv2-g/14 가중치가 있고, 형태 기반 전치 합성곱 레이아웃 변환이 필요할 수 있다.
- 마스킹된 포인트와 깊이는 양의 무한대이며 마스킹된 법선은 0이다. 유한값만 확인하는 비교기는 지지도 불일치를 잘못 통과시킬 수 있다.

이 중 어느 하나라도 잘못되면 그럴듯하지만 틀린 기하가 나올 수 있다. 따라서 수용 검증은 인증된 전체 해상도 참조를 사용하고 수치 지표를 계산하기 전에 스키마, 형태, dtype, 마스크, 모든 유한값/NaN/무한대 분류를 확인한다.

## 2. 변경 요약

| 영역 | 변경 내용 |
|---|---|
| 런타임 통합 | `moge3`를 `geometry` 작업과 정지 이미지 입력으로 밀집 라우팅 및 아키텍처 카탈로그에 등록한다. |
| 공개 API | 하위 소비자를 위해 `Moge3`, `Moge3Output`, `InferOptions`, 기본 `estimate`, 검증된 설정 가능 추론을 공개한다. |
| 백본 및 디코더 | 공유 DINOv2 인코더를 재사용하고 중간 그리드 네 개를 결합하며 neck, point, mask, normal, scale head를 구현한다. |
| 희소 리파인 | 밀집 복셀 볼륨을 할당하지 않고 좌표 기반 submanifold 3D 합성곱, 이웃 조회, 2x 평균 풀링, 최근접 업샘플링, 희소 3D U-Net을 추가한다. |
| 기하 | UV 생성, 초점/깊이 이동 복원, 정규화 내부 파라미터, 포인트/깊이 변환, 투영 제어, 마스킹, 단계별 복원을 구현한다. |
| 코어 브리지 | F32 정밀도 범위를 넘는 희소 키도 정확히 유지하도록 side 선택을 포함한 int64 `searchsorted`를 추가한다. |
| 가중치 로딩 | 체크포인트를 엄격히 로드하고 아키텍처와 텐서 형태를 검증하며 F32 실행을 유지하고 지원하는 ConvTranspose 레이아웃을 멱등적으로 정규화한다. |
| 자원 안전성 | 큰 그래프 생성 전에 이미지 픽셀, 토큰 범위, 디코더 픽셀, 희소 구성, 채널, 리파인 단계, 보존되는 단계별 출력을 제한한다. |
| 검증 | 작은 업스트림 수학 오라클, 고정된 실제 체크포인트 참조, 다섯 사례 Rust 패리티, fail-closed CLI 검증기를 추가한다. |
| 문서 | 체크포인트, 옵션, 출력 텐서, 전처리, 메모리 동작, TF32 제어, 지원하지 않는 입력 표면을 문서화한다. |

모듈 추출 후 작업 트리는 현재 `6ca04d17` 대비 39개 파일에서 6,812줄 추가와 9줄 삭제를 포함한다.

## 3. 아키텍처와 기술적 선택

### 3.1 공유 밀집 예측 및 DINOv2 기반 재사용

```text
RGB 이미지
  -> 검증된 [0,1] NHWC 전처리
  -> DINOv2-L/14 또는 DINOv2-g/14 중간 레이어
  -> UV 증강 + 투영된 특징 피라미드
  -> point / mask / normal / scale head
  -> 초점 및 깊이 이동 복원
  -> 선택적 희소 3D 리파인 반복
  -> 투영 + 마스킹 + 원본 크기 복원
  -> DenseOutput 및 Moge3Output
```

이 모델군은 의존 PR이 도입한 밀집 레지스트리, 출력 writer, 보간, DINOv2 인코더를 재사용한다. 모델별 합성곱, 기하, 희소 코드는 `src/vision/dense/moge3/` 아래에 둔다. 이 구성은 공유 동작을 중앙화하면서 업스트림 MoGe3 수학을 보존한다.

### 3.2 밀집 변환 없이 희소 좌표 의미 보존

희소 복셀은 정수 `(batch, row, column, depth-bin)` 좌표와 래스터 키를 사용한다. 이웃 조회는 정확한 int64 `searchsorted`를 사용하고 누락된 이웃은 추가된 0 행을 gather한다. 풀링은 부모 키를 정렬하고 세그먼트 ID를 만들며 세그먼트별 첫 행을 계산하고 셀 구성원 최대 여덟 개를 평균하며 최근접 업샘플링에 쓰는 부모 인덱스를 반환한다.

C++ scatter API는 update rank가 target rank와 index rank의 합이어야 한다. 첫 guarded 단위 테스트는 Python의 `.at[...]` 어댑터가 암묵적으로 확장하는 rank-one update를 드러냈다. Rust는 이제 가산 부모 할당과 최소 첫 위치 축약에 `[M,1]` update를 제공한다. 이는 MLX Python 바인딩이 삽입하는 형태와 같으며 인덱스, 값, 순서, dtype, rank-one 출력을 바꾸지 않는다.

### 3.3 공개 경계와 할당 경계에서 fail closed

`InferOptions::validate`는 JSON 파싱 경계와 공개 라이브러리 경계 모두에서 적용된다. 체크포인트 토큰 범위, `refine_steps <= 16`, 리파이너 가용성, 0도보다 크고 180도보다 작은 유한 `fov_x`를 확인한다. 설정 검증은 공개 백본, 중간 레이어 네 개, head 스키마, 채널 폭, 디코더 증가량, 희소 작업 메모리, 리파이너 토폴로지를 제한한다. 런타임 검증은 이미지 크기, 텐서 rank, 좌표 범위/순서/고유성, 출력 보존 예산을 포함한다.

정수 이미지는 원본 dtype 범위로 스케일한다. 직접 전달되는 부동소수점 입력은 `[0,1]` 범위의 유한 RGB여야 한다. 잘못된 요청은 큰 할당이나 추론 전에 컨텍스트가 있는 오류를 반환한다.

### 3.4 업스트림 비유한 값과 옵션 동작을 그대로 노출

기본 출력은 정확히 `points`, `depth`, `mask`, `normal`, `intrinsics`다. `return_per_step`은 `points_stepN`, `depth_stepN`, `intrinsics_stepN`을 추가한다. `force_projection`, `apply_mask`, 고정 `fov_x`, 0회 리파인, 명시적 `num_tokens`는 업스트림 의미를 유지한다.

마스킹을 사용하면 잘못된 포인트와 깊이는 양의 무한대가 되고 잘못된 법선은 0이 된다. 커밋된 비교기는 먼저 정확한 키 집합, 형태, dtype, 마스크, 유한값/NaN/양의 무한대/음의 무한대 분류를 요구하고, 이후 공유 유한 지지에서만 상대 RMS를 계산한다. 유한 지지가 비어 있으면 실패한다.

### 3.5 지표 수용 전에 참조 인증

픽스처 하네스는 변경되지 않은 mlx-vlm 리비전 `abd3b2d7b728cb5372d4e092260da325e5ded209`, MLX 0.32.5, 네이티브 라이브러리 SHA-256 `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094`, 바인딩 SHA-256 `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`에 고정된다. 비교 전에 소스 모듈, 체크포인트 설정/전처리기/가중치, 무손실 입력 이미지, 런타임 매핑, 옵션, 출력 스키마, 아티팩트 해시를 검증한다.

실제 Rust 테스트는 두 체크포인트를 로드하기 전에 ViT-L 및 ViT-g 체크포인트 리비전과 파일 해시를 독립적으로 인증한다. 참조 생성과 비교는 `python -O`에서도 누락되거나 불일치하는 receipt를 거부한다.

전체 해상도 ViT-L CPU 생성기는 17개 출력을 모두 저장하고 완전한 receipt를 기록한 뒤 receipt 부분집합과 정확히 일치하는 최종 JSON을 출력했다. 이후 외부 shell wrapper가 따옴표 오류로 종료 코드 2를 반환했고 Python 자식 종료 코드는 별도로 기록되지 않았다. SHA-256 `ededb85947995300045bcaef281cf1254af373a951a54d0d2eba53ef8ee3c282`인 복구 검증기는 추론 없이 완료됐으며 원본 status/resource 해시, 메모리 중단 아님, 최종 출력, 정규 식별자/옵션, F32 파라미터 607개, 텐서 스키마, 유한값/NaN/무한대 분류를 검증한다. 복구 기록은 `wrapper_exit: 2`, `python_exit: not independently captured`, 누락된 실행 시점 wrapper 해시를 그대로 보존한다.

대응하는 불변 Rust CPU 캡처는 종료 코드 0이었고 CPU 런타임을 인증했다. 버전 1 검증기 SHA-256 `bae73574ef629abbcf6cca7b0a611c7390d2f3f159d5e2e19fe993dbae209cea`는 비교 전에 보관된 리비전과 소스, 바이너리, 체크포인트, 이미지, 정확한 옵션과 명령, 출력 스키마, preview, 메타데이터, 값 분류를 인증했다. 커밋 경로의 `cpu-parity.json` 기록은 이 식별자들을 참조 및 Rust 출력 해시, 리소스 receipt, 비교기 해시, 17개 전체 지표, 실패 없음 결과에 연결한다.

## 4. 리뷰 결과

### 4.1 정확성과 통합

독립 구현 리뷰는 개발 중 토큰 그리드 반올림, 공개 옵션 검증, helper K 의미, 전처리 컨텍스트, 비유한 값 비교, 누락된 옵션 사례 범위의 여섯 문제를 발견했다. 여섯 항목 모두 수정되어 재검토되었다. 최종 정적 검토에서는 모델, 레지스트리, 출력, 인증된 실제 GPU 옵션 사례 다섯 개의 통합에 열린 정확성 문제가 없었다.

CLI 검증기도 리뷰 후 강화되었다. 자식 프로세스의 인식 가능한 `Runtime device:` 줄을 정확히 하나 요구하고, Rust receipt 생성이나 텐서 비교 전에 요청/참조 백엔드 불일치를 거부하며, 확인된 장치를 기록한다. 따라서 환경에서 GPU를 요청했지만 CPU로 fallback된 실행을 GPU로 잘못 인증할 수 없다. 기본 no-`--opts` CLI 경로는 희소 모듈 추출 직전 소스에서 통과했다. 추출된 최종 소스는 컴파일, 링크, 단위 테스트, clippy 게이트를 통과했지만 새 CLI 바이너리로 전체 모델 추론을 다시 실행하지는 않았다.

### 4.2 보안과 성능

보안/성능 리뷰에서는 수정된 소스에 남은 문제가 없었다. 할당을 결정하는 값은 그래프 구성 전에 제한되고, 희소 이웃 구성은 chunk 단위로 처리되며 보존 메모리와 최대 작업 메모리를 계산한다. 비유한 출력은 산술 결과를 신뢰하는 대신 분류하고, 설정, 체크포인트, 이미지, 런타임, 참조 식별자는 fail closed 방식으로 검증한다.

희소 구현은 밀집 3D 복셀 할당을 피한다. 아래 검증 시간은 공유 GPU guard 아래의 진단 실행 시간이며 성능 벤치마크가 아니다.

### 4.3 문서 범위

구현 범위 안에 남은 문서 누락은 없다.

- `docs/dense-prediction.md`는 모델군/작업 라우팅, 두 체크포인트, CLI 문법, 모든 옵션, 텐서 다섯 개의 계약, 전처리, 메모리 거부, TF32 제어, 명시적인 batch/video/server 제외를 다룬다.
- `docs/supported-models.md`는 아키텍처, 출력, 옵션, 런타임 범위를 기록한다.
- `tests/fixtures/moge3/README.md`는 provenance, 재생성, 엄격한 비교, 모든 GPU 사례, 복구된 CPU 참조의 한계, 완료된 CPU 패리티, 기본 no-`--opts` CLI 절차를 기록한다.

문서는 고정된 업스트림 구현의 양의 무한대 마스킹을 의도적으로 따르며, NaN 포인트를 언급한 오래된 이슈 문구를 바로잡는다.

## 5. 검증 근거

| 검사 | 보고서 작성 시점 결과 |
|---|---|
| 집중 라이브러리 컴파일 | 첫 시도는 iterator 비교 타입 오류 한 개와 `UniquePtr` borrow 오류 세 개로 종료 코드 101이었다. 수정 및 모듈 추출 후 최종 소스 라이브러리 no-run 빌드는 10분 57초에 종료 코드 0으로 통과했다. |
| MoGe3 집중 단위 테스트 | 첫 guarded 실행은 6개를 통과한 뒤 rank-one scatter update에서 종료 코드 134로 중단됐다. `[M,1]` 어댑터 수정 후 최종 소스 guarded 실행은 0.79초에 8개 모두 통과했고 실패는 없었다. |
| 정확한 int64 `searchsorted` | 지나치게 제한적인 `--exact` selector는 0개를 선택했으므로 결과에 포함하지 않았다. 수정된 필터는 의도한 경계/side 의미 테스트를 실행해 1개 통과, 0개 실패였다. |
| 인증 하네스 검사 | 엄격한 작은 오라클 재생성은 바이트 단위로 같았다. 강화된 식별자/스키마 음성 검증은 음성 사례 15개와 양성 비교기 계약 1개를 통과했다. 기존 GPU 참조는 원본 텐서나 receipt를 다시 쓰지 않고 재검증됐다. 복구된 CPU 참조는 예상한 17개 텐서를 모두 포함하고 receipt 및 정규 식별자와 일치하며, 이 전체 유효 마스크 이미지에서 모든 F32 출력의 NaN/+무한대/-무한대 개수가 0임을 기록한다. |
| 실제 GPU 패리티 | **gfx1151에서 `MLX_ENABLE_TF32=0`으로 통과.** 검증된 추출 전 바이너리의 순차 테스트 함수 두 개가 ViT-L/ViT-g 사례 다섯 개를 다뤘으며 89.99초에 2개 통과, 0개 실패였다. 시간은 진단값이다. |
| 출력 무결성 | 수치 지표 전에 모든 사례에서 정확한 출력 키 집합, 형태, F32/bool dtype, 마스크 값, 유한값/NaN/무한대 분류가 통과했다. |
| 추출 전 소스의 실제 CLI | **통과.** 빌드는 종료 코드 0이었다. Guarded 기본 실행은 실제 no-`--opts` 경로를 사용하고 확인된 GPU를 인증했으며, 텐서 5개를 출력하고 정확한 마스크/비유한 지지, preview, 메타데이터/intrinsics, `mlxcel arch` 검사를 통과했다. Depth/points 상대 RMS는 `2.224974e-5`/`2.033098e-5`, normal 상대 RMS는 `4.20353e-7`, intrinsics 최대 절대 오차는 `2.080202e-5`였다. |
| 최종 소스 integration 링크 | **통과.** 명명된 실제 모델 integration target은 6분 27초에 종료 코드 0으로 링크됐다. 이는 링크를 검증하며 체크포인트 실행을 다시 주장하지 않는다. |
| 최종 소스 CLI 빌드 | **통과.** 최종 CLI는 6분 18초에 종료 코드 0으로 컴파일됐고 SHA-256 `4fd8c932c768e3267dfab498b31686986b4b6fb95f52d6bb2039c97d1277d3ec`을 생성했다. 이 바이너리로 전체 모델 패리티를 실행하지 않았다. |
| 필수 clippy | **통과.** 최종 소스에서 `--lib --tests -- -D warnings`가 3분 5초에 종료 코드 0이었다. |
| 최종 소스 수치 근거 범위 | 독립 검토에서 희소 모듈 추출, iterator 수정, 포맷 변경이 연산 그래프를 보존함을 확인했다. 최종 소스는 컴파일, 링크, guarded 단위 테스트, clippy 게이트를 통과했다. 수치 수용 근거는 검증된 추출 전 바이너리에 고정되며 최종 바이너리의 별도 체크포인트 결과는 주장하지 않는다. |
| Python CPU 참조 | **완전한 참조로 복구됨.** 전체 해상도 텐서 17개와 receipt가 출력 SHA-256 `afbc0bffa5112996423dd11230cf066498a0d98372f67adb201e37dfbf393cb0`으로 인증됐다. 최종 생성기 출력 후 wrapper가 종료 코드 2를 반환했으며 Python 종료 코드는 알 수 없으므로 성공한 프로세스 종료를 주장하지 않는다. |
| CPU 패리티 | **CPU에서 `MLX_ENABLE_TF32=0`으로 통과.** 불변 Rust 캡처는 종료 코드 0이었고 고정된 검증기가 모든 식별자와 정확한 17-key 스키마를 인증했으며 엄격한 비교는 실패가 없었다. 최대 depth/points 상대 RMS는 `1.5590750464e-5`/`1.4222186927e-5`, normal 상대 RMS는 `2.8714204e-7`, intrinsics 최대 절대 오차는 `1.6570091248e-5`였고 마스크/비유한 지지는 정확했다. |
| 현재 HEAD PR CI | **`c90268bb`에서 통과.** 필수 PR 검사는 모두 green이며 범위에 해당하지 않는 작업은 skip됐다. 통과 항목에는 format, clippy, CPU-only link, OpenXLA feature compile, cargo-deny, Python tooling, binary assets, 프로젝트 계약 게이트가 포함된다. |

CLI 빌드는 9분 19초에 종료 코드 0으로 완료됐고 바이너리 SHA-256은 `8740d2a60082e5d58564343e081d3660e1026e555dc6da78cfb362fcbeda57d2`였다. 시간은 진단값이다. 첫 검증기 실행은 격리된 Python 경로에 Pillow가 없어 CLI 호출이나 출력 생성 전에 실패했다. 재시도는 exact MLX 경로를 앞에 유지하고 Pillow를 위한 읽기 전용 SAM 참조 site-packages를 추가해 종료 코드 0으로 완료됐다.

추출된 최종 소스는 6분 18초 빌드에서 SHA-256 `4fd8c932c768e3267dfab498b31686986b4b6fb95f52d6bb2039c97d1277d3ec`인 별도 CLI 바이너리를 생성했다. 이 결과는 빌드 근거만 제공한다. 이 보고서의 모든 전체 모델 수치 및 CLI 출력 주장은 이전 `8740d2a6...` 바이너리와 보관된 `de4b55de...` 소스에 연결된다.

실제 Rust CPU 캡처는 검증된 추출 전 바이너리의 불변 복사본과 보관된 커밋 `de4b55de47e7a65e9fff60bdcd00c4f74c90de3f`을 사용했다. 복구된 Python CPU 참조와 정확히 같은 ViT-L 체크포인트, 이미지, resolution-9, 3회 리파인, 단계별 출력 설정을 사용했다. 프로세스는 확인된 CPU 백엔드에서 종료 코드 0이었다. 리소스 receipt는 19,676.249865초, 최대 RSS 370,176 KiB, 최소 가용 메모리 19,148,304,384바이트, `stopped_for_memory: false`를 기록한다. 이 값은 실행 provenance이며 벤치마크 결과가 아니다. Rust 출력 SHA-256은 `b7ce67a1fac333a5297dea4c2b579072fbd789f0391d34e053cd8960d21c2998`이다. 이 캡처는 보관된 추출 전 소스에만 해당하며 최종 추출 소스의 수치 근거가 아니다.

Rust 캡처 검증기는 버전 1이며 SHA-256은 `bae73574ef629abbcf6cca7b0a611c7390d2f3f159d5e2e19fe993dbae209cea`다. 출력을 읽기 전에 보관된 리비전, 바이너리와 소스 아카이브 해시, 정규 ViT-L 체크포인트, 고정 이미지, 정확한 17-key 옵션 사례, CPU 런타임, TF32 설정, 명령을 고정해 검증한다. 종료 코드 0으로 통과했으며 유한값, NaN, 양의 무한대, 음의 무한대 개수를 따로 기록했다. 이어진 엄격한 비교는 17개 키 전체에서 실패가 없었다. 커밋된 `tests/fixtures/moge3/cpu-parity.json`은 인증된 Rust receipt, 리소스, 검증 및 비교기 해시, 전체 지표, Python 참조 SHA-256 `afbc0bff...`, Rust 출력 SHA-256 `b7ce67a1...`을 연결한다.

순차 GPU 실행에서 관측한 최대 오차는 다음과 같다.

| 출력 | 필수 GPU 임계값 | 최대 관측 오차 |
|---|---:|---:|
| Depth | relative RMS <= `5e-3` | `2.2250e-5` |
| Points | relative RMS <= `5e-3` | `2.0331e-5` |
| Normal | relative RMS <= `5e-3` | `4.20e-7` |
| Intrinsics | maximum absolute <= `1e-3` | `2.0802e-5` |
| Mask 및 비유한 지지 | exact | exact |

CPU 수용 실행에서 관측한 최대 오차는 다음과 같다.

| 출력 | 필수 CPU 임계값 | 최대 관측 오차 |
|---|---:|---:|
| Depth | relative RMS <= `1e-4` | `1.5590750464e-5` |
| Points | relative RMS <= `1e-4` | `1.4222186927e-5` |
| Normal | relative RMS <= `1e-4` | `2.8714204e-7` |
| Intrinsics | maximum absolute <= `1e-3` | `1.6570091248e-5` |
| Mask 및 비유한 지지 | exact | exact |

이전의 테스트 두 개 병렬 GPU 실행도 종료 코드 0이었고 진단 근거로 보존한다. 위 순차 실행이 수용 기록이며, 그 실행 시간을 벤치마크로 해석하면 안 된다.

## 6. 인증된 GPU 사례

| 사례 | 체크포인트 | 검증 범위 | 키 수 | 참조 SHA-256 |
|---|---|---|---:|---|
| `vitl-gpu-resolution9` | ViT-L | 복원 FOV, 3회 리파인, 단계별 출력, 투영, 마스크 | 17 | `8a52dfd2f1d4f6fa7e9e955fd4ab3366bf4cb66c9de1b19816306c6917280a61` |
| `vitg-gpu-resolution9` | ViT-g | 동일한 기본 전체 해상도 경로의 giant 백본 | 17 | `b3a52681524456798639eef2971422edbd6113ea7b6188bb7b019bcd25b09cd7` |
| `vitl-no-refine` | ViT-L | 리파인 0회, 복원 FOV, 투영 없음, 마스킹 | 5 | `c6151def796e489085ffbf2c791c7bbc8f1532ffe1735dd586a7795c5951756f` |
| `vitl-fixed-unmasked` | ViT-L | 고정 60도 FOV, 3회 리파인, 단계별 출력, 투영 및 마스크 없음 | 17 | `517044d2f70104e62a59f975d48bec52d37f9b818357de88ea74e09c97b640bf` |
| `vitl-fixed-projected` | ViT-L | level 0을 덮어쓰는 명시적 3,600 토큰, 고정 FOV, 투영, 마스크 없음 | 5 | `9e113455ff51aa588854088fa9282348b5246f840de2f510e2916f5060405d44` |

별도의 `vitl-default` 5-key 참조 SHA-256은 `97cbe3c12ca8553f7f177bc189193f8b29ace8d536a25ab8417d7851d74c31e4`다. 성공한 추출 전 no-`--opts` CLI 검증에서 이 참조를 사용했다. 독립 검토에서 이후의 추출, iterator 수정, 포맷 변경이 연산 그래프를 보존함을 확인했다. 최종 소스는 컴파일, 링크, guarded 단위 테스트, clippy 게이트를 통과했으며 최종 바이너리 자체에는 별도 전체 모델 수치 결과가 없다.

복구된 `vitl-cpu-resolution9` 참조는 완료된 Rust CPU 캡처와 같은 17-key ViT-L resolution-9/refine-3/단계별 출력 사례를 사용한다. 출력 SHA-256은 `afbc0bffa5112996423dd11230cf066498a0d98372f67adb201e37dfbf393cb0`이며 `tests/fixtures/moge3/cpu-reference-recovery.json`은 wrapper 실패, 알 수 없는 Python 종료 코드, 검증기/status/resource 해시, 정확한 옵션, 출력별 값 분류 개수를 기록한다. `tests/fixtures/moge3/cpu-parity.json`은 완료된 엄격한 비교와 그 불변 추출 전 범위를 기록한다.

## 7. 호환성과 한계

- **검증된 백엔드**: TF32를 끈 gfx1151 ROCm과 CPU. CPU 수치 근거는 불변 추출 전 바이너리 및 보관 소스에 고정된다.
- **검증하지 않은 백엔드**: 이 보고서에서는 Metal과 CUDA 실행을 수행하지 않았다.
- **체크포인트 범위**: 공개된 ViT-L 및 ViT-g 체크포인트 모두 인증된 GPU 추론을 통과했다.
- **입력 범위**: 정지 이미지 한 장. Batch CLI 입력, video, streaming, server route는 #2355 범위 밖이다.
- **변환 범위**: 공식 PyTorch `model.pt` 변환은 이슈 범위 밖이며 공개 MLX safetensors를 지원한다.
- **의존성**: 병합된 밀집 출력 및 DINOv2 기반을 사용한다. 새 제3자 의존성은 추가하지 않았다.

## 8. 후속 조치

### 완료 상태

- [x] 인증된 GPU 옵션 사례 다섯 개와 기본 CLI 경로를 검증했다.
- [x] 실제 Rust CPU 캡처, 고정 검증, 변경하지 않은 임계값의 엄격한 17-key 비교를 완료했다.
- [x] 최종 소스 컴파일, 링크, guarded 단위 테스트, 경고 거부 clippy, CLI 빌드, 현재 HEAD 필수 CI 검사를 통과했다.

이후 문서 전용 커밋에는 저장소의 일반 CI 검사를 실행해야 한다. 문서만 바뀌므로 모델 그래프는 변하지 않으며 체크포인트 실행을 반복할 필요는 없다.

### 지속적 안전장치

- 체크포인트와 전체 텐서 아티팩트는 무시되는 `.artifacts/moge3` 아래에 두고 stage하지 않는다.
- 참조 재생성 시 정확한 소스/런타임/체크포인트/이미지 해시와 엄격한 receipt 검사를 유지한다.
- 패리티 임계값을 바꾸지 않고 지표 전에 스키마와 값 분류를 비교한다.

## 9. 변경 통계와 커밋

| 항목 | 값 |
|---|---:|
| 변경 파일 수 | 39 |
| 추가 줄 수 | 6,812 |
| 삭제 줄 수 | 9 |
| 새 MoGe3 Rust 모듈 수 | 14 |
| 통과한 인증 GPU 사례 수 | 5 |
| 통과한 인증 CPU 사례 수 | 1 |

| 커밋 | 메시지 |
|---|---|
| `5d787774` | `feat: add MoGe3 dense geometry prediction` |
| `de4b55de` | `docs: record MoGe3 GPU parity evidence` |
| `c90268bb` | `refactor: split MoGe3 sparse pooling module` |

## 참고 자료

- PR #2577: `feat: add MoGe3 dense geometry prediction`
- Issue #2355: MoGe3 단안 기하
- Epic #2382: 밀집 예측 모델 제공
- 업스트림 mlx-vlm 리비전 `abd3b2d7b728cb5372d4e092260da325e5ded209`
- `docs/dense-prediction.md`: 공개 CLI 및 텐서 계약
- `tests/fixtures/moge3/README.md`: 참조 provenance, 생성, 비교, CLI 검증 절차
- `/tmp/epic-2382-moge3-real-gpu-sequential.log`: 순차 GPU 수용 로그
- `/tmp/epic-2382-moge3-cli-validate-retry.log`: 성공한 추출 전 CLI 검증 로그
- `/tmp/epic-2382-moge3-final-lib-no-run.log`: 최종 소스 라이브러리 빌드 로그
- `/tmp/epic-2382-moge3-final-units.log`: 최종 소스 guarded 단위 테스트 로그
- `/tmp/epic-2382-moge3-final-real-no-run.log`: 최종 소스 named integration 링크 로그
- `/tmp/epic-2382-moge3-final-cli-build.log`: 최종 소스 CLI 빌드 로그
- `/tmp/epic-2382-moge3-final-clippy-retry2.log`: 최종 소스 경고 거부 clippy 로그
- `tests/fixtures/moge3/cpu-reference-recovery.json`: 인증된 Python CPU 참조 복구 receipt
- `/tmp/epic-2382-moge-recover-cpu-reference.py`: SHA-256 `ededb85947995300045bcaef281cf1254af373a951a54d0d2eba53ef8ee3c282`인 복구 검증기
- `/tmp/epic-2382-moge-validate-rust-cpu.py`: 고정된 Rust CPU 캡처 검증기 버전 1, SHA-256 `bae73574ef629abbcf6cca7b0a611c7390d2f3f159d5e2e19fe993dbae209cea`
- `.artifacts/moge3/rust-cpu-capture/validation.json`: 인증된 Rust CPU 출력 receipt
- `.artifacts/moge3/rust-cpu-capture/cpu-comparison.json`: 엄격한 17-key CPU 비교 결과
- `tests/fixtures/moge3/cpu-parity.json`: 커밋 경로의 CPU 패리티 receipt 및 전체 지표 기록
