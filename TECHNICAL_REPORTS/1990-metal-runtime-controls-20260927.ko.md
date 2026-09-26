# 기술 보고서: PR #1990, 실제 빌드 백엔드와 Metal 런타임 제어 일치

**작성일**: 2026-09-27
**상태**: 구현 및 로컬 검증 완료. 머지 대기 중.
**언어**: Rust, C++, Markdown
**위험도**: 중간

## 요약

이 변경은 macOS plain release build에서 MTP 정확성 자동 복구가 동작하지 않던 문제를 수정한다. MLX의 CMake 설정은 기본적으로 Metal을 활성화했지만 C++ bridge는 Cargo의 선택적 `metal` feature가 있을 때만 실제 QMV 및 command-buffer 제어를 노출했다. 따라서 plain `cargo build --release` 바이너리는 Metal에서 실행되면서도 Rust에서 호출한 QMV setter는 아무 동작도 하지 않았다. Exactness gate는 `qmv_wide`를 끄고 재시도했다고 기록했지만 실제로는 wide kernel을 두 번 측정했다.

이제 빌드는 Metal backend 여부를 한 번만 결정하고 같은 값을 CMake와 C++ bridge에 사용한다. 실패한 probe는 최초 verdict와 narrow 재시도 verdict를 모두 보존한다. Narrow 재시도가 성공하면 해당 mode를 유지하고, 실패하거나 실행할 수 없으면 이전 wide mode를 복원한다.

Plain release 바이너리로 M5 Max에서 Gemma 4 31B QAT와 Qwen 3.8 MTP를 검증했다. 자동 재시도와 별도의 `MLXCEL_QMV_WIDE=0` 프로세스는 각 체크포인트에서 동일한 출력과 acceptance 통계를 만들었다.

## 1. 문제 정의

Issue #1187에서 추가된 자동 재시도는 block verification이 single-token chain과 다르면 `qmv_wide`를 끈다. PR #1987의 Gemma 4 작업은 다른 geometry에서 이 재시도를 사용했다. 사용자가 보고한 M5 Ultra 빌드에서 Gemma의 최초 probe는 layer 0 sliding attention에서 524,288 logit byte 중 240,460 byte가 달랐다. 재시도 로그는 wide kernel을 꺼도 실패했다고 표시했지만, `MLXCEL_QMV_WIDE=0`으로 시작한 새 프로세스는 통과했다.

원인은 서로 다른 두 build-time 판정이었다. CMake는 모든 macOS build에서 기본적으로 Metal을 활성화했다. Bridge는 Cargo의 선택적 `metal` feature가 있을 때만 실제 runtime-control 호출을 컴파일했다. 따라서 plain release server에는 Metal backend가 존재했지만 제어 함수는 stub이었다.

기존 진단은 실패한 재시도 결과를 boolean으로 축약했다. 이 과정에서 재시도의 divergence 위치와 byte count가 사라져 두 번째 arm이 실제로 실행됐는지 확인하기 어려웠다.

## 2. 기술적 검토

### 2.1 Backend 선택

`mlxcel-core/build.rs`는 native build host와 `MLXCEL_BUILD_METAL`에서 Metal 사용 여부를 결정한다. 같은 결과가 `MLX_BUILD_METAL`과 `MLXCEL_BRIDGE_METAL_BACKEND`을 모두 설정한다. Plain macOS build는 Metal을 기본으로 사용하며, 명시적인 `MLXCEL_BUILD_METAL=OFF`는 backend와 제어 기능을 함께 비활성화한다.

순수 unit test는 macOS 기본값, 허용되는 명시적 값, non-macOS 비활성화 정책, 잘못된 macOS 값의 명시적 오류를 검사한다. Live example은 QMV를 양방향으로 전환하고 Metal command-buffer budget을 round-trip한 뒤 종료 시 이전 process 상태를 복원한다.

### 2.2 정확성 진단 및 상태

재시도는 완전한 `BlockChainExactness` verdict를 반환한다. Decline snapshot과 warning log는 최초 결과와 narrow 재시도 결과를 분리해 보존하며, localized layer 정보와 서로 다른 byte 수를 포함한다.

정확한 재시도는 narrow QMV를 유지한다. Wide path를 다시 활성화하면 방금 측정한 결정을 무효화하기 때문이다. 실패하거나 `NotRun`인 재시도는 wide QMV를 복원한다. 운영자가 `MLXCEL_QMV_WIDE` 값을 고정하면 자동 재시도를 계속 건너뛰며 로그에 그 이유를 표시한다.

### 2.3 호환성 및 보안

Backend policy 변경은 build configuration과 기존 runtime control에만 적용된다. Linux를 포함한 non-Metal build의 Metal 제어는 계속 inert 상태다. Endpoint, 인증 동작, unsafe block, checkpoint format은 변경하지 않는다.

영향받는 구버전 바이너리에서는 `MLXCEL_QMV_WIDE=0`을 설정한 새 프로세스를 시작하는 우회 방법을 사용할 수 있다. 이 값은 process 단위로 초기화되고 Metal QMV dispatch에만 적용되며 verify throughput을 낮출 수 있다. Model exactness probe 자체는 우회하지 않는다.

## 3. 검증

| 검사 | 결과 |
|---|---|
| Plain-release runtime-control example | QMV false/true를 정확히 관찰했고 command-buffer budget 17 round-trip 통과 |
| Gemma 4 31B QAT 자동 재시도 | block size 4 통과, output 64 tokens, 28 rounds에서 33/84 accept |
| Gemma 4 31B QAT narrow 고정 | 통과, 자동 재시도와 출력 및 acceptance 동일 |
| Gemma 출력 동일성 | 268 bytes, SHA-256 `55ad7d86e23b6132e4b3b2af1d5f9cdda4ed552baf26690d76a410bfe823d66d` |
| Qwen 3.8 자동 재시도 | block size 3 통과, output 64 tokens, 26 rounds에서 35/52 accept |
| Qwen 3.8 narrow 고정 | 통과, 자동 재시도와 출력 및 acceptance 동일 |
| Qwen 출력 동일성 | 328 bytes, SHA-256 `3aca47725fce1ffda00d959c4d9758eccc9b2bd04daaf9c4f6af575f1e6add05` |
| Format 및 focused retry tests | 통과 |
| 전체 `make verify` workspace gate | 통과 |
| 운영자 pin focused suite | 환경 변수 미설정, 0 고정, 1 고정에서 각각 16/16 통과 |

Gemma 자동 실행은 알려진 wide-path 차이 240,460/524,288 byte를 먼저 재현했고 runtime에서 선택한 narrow path로 통과했다. 생성 결과는 explicit-Metal release baseline과도 일치했다.

이 검증은 정확성 확인이다. 단일 샘플 실행 시간은 throughput 근거로 사용하지 않는다.

전체 gate는 마지막 test-only 격리 수정 전에 통과했다. 이후 해당 focused suite를 다시 빌드해 세 가지 ambient 운영자 pin mode에서 모두 통과했다.

## 4. 변경 요약

| 항목 | 값 |
|---|---:|
| 구현 파일 | 8 |
| 주요 커밋 | `d939279f` |

| 분류 | 요약 |
|---|---|
| Build 정확성 | 하나의 Metal 판정이 MLX와 bridge compile을 함께 제어 |
| Runtime 동작 | Plain release build에서 QMV와 command-buffer 설정이 실제로 변경됨 |
| 진단 | 최초 verdict와 재시도 verdict를 분리된 완전한 결과로 유지 |
| 테스트 | Build policy, retry state, diagnostic, live bridge 검사 추가 |
| 문서 | Plain build 동작, 검증 example, 구버전 우회 방법 문서화 |

## 5. 학습 포인트

Runtime backend query만으로는 충분하지 않다. 제어 surface가 다른 조건으로 컴파일됐다면 runtime에서 backend를 감지해도 제어할 수 없다. Backend 생성과 backend 전용 bridge는 동일한 build 판정을 사용해야 한다.

재시도 진단은 각 arm의 측정 결과를 보존해야 한다. 재시도가 실패했다는 정보만 남기면 no-op 제어를 숨길 수 있다. 위치와 byte count를 함께 기록하면 두 번째 측정이 실제로 달라졌는지 알 수 있다.

## 6. 한계

수정 바이너리의 실제 모델 검증 범위는 M5 Max 한 대, Gemma 4 31B QAT, Qwen 3.8이다. 일반적인 성능 변화를 입증하지 않으며 narrow QMV가 모든 모델을 exact하게 만든다고 보장하지 않는다. 사용자의 원래 M5 Ultra 호스트에서는 수정 바이너리를 다시 실행하지 못했다. Cross-host compile과 실제 Metal-OFF/CUDA 실행도 검증 범위 밖이다. 각 model과 block size에 대한 exactness gate가 계속 최종 판단을 내린다.

## 7. 참고 자료

- Issue #1988: M5 Ultra의 QMV 자동 재시도 실패 진단
- Issue #1187: narrow QMV 정확성 자동 재시도
- PR #1987: Gemma 4 31B MTP 정확성
- `docs/installation.md`
- `src/lib/mlxcel-core/examples/metal_runtime_switch_probe.rs`
