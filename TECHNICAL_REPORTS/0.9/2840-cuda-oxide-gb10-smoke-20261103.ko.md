# PR #2840: GB10용 cuda-oxide 아티팩트 스모크 추가

**작성일**: 2026-11-03
**상태**: 리뷰 중 (PR 열림, #2831을 닫음)
**언어**: Rust, Python, Bash, Markdown
**위험도**: 낮음 (격리된 spike 패키지, 공유 변경은 CI 경로 필터 축소뿐)

## 요약

PR #2840은 GB10에서 cuda-oxide 커널을 빌드·실행하고, 남은 실험이 의존할 아티팩트 계약을 고정한다. 고정된 NVIDIA/cuda-rust 리비전으로 경계가 분명한 vector-add 커널을 컴파일해 PTX와 `sm_121` cubin으로 내보냈고, 인자 배치가 방출된 PTX 시그니처와 `cuobjdump` kparam 레코드에서 나온 manifest로 검증했다. 크기가 작은 경우, 홀수, 블록 정렬, 꼬리 크기를 아우르는 14개 디바이스 케이스가 호스트 참조와 비트 단위로 일치했고(총 69,913원소), Compute Sanitizer는 오류 0건을 보고했다. 패키지는 `spike/` 아래 자체 워크스페이스 루트로 두었으므로, 일반 mlxcel 빌드는 여전히 cuda-oxide 크레이트를 해석하지 않고 애플리케이션 Rust pin 1.97.1을 유지한다.

## 1. 문제 정의

타당성 검토(#2835)는 첫 불확실성이 cuda-oxide가 대상 GPU에서 올바른 커널을 만드는지이며, 이 질문을 추론 커널이나 MLX 어댑터보다 먼저 해결해야 한다고 결론지었다. 이 PR 전에는 oxide 커널을 컴파일하거나 실행한 적이 없었다. 대상에는 함정도 있었다. CuTe 프리뷰가 `sm_121a`를 거부하므로 `sm_100a`나 `sm_120a`용 아티팩트를 GB10 근거로 이름만 바꿀 수 없고, 방출된 타깃을 가정이 아니라 실측으로 보여야 했다.

이슈는 격리 패키지, 재현 가능한 빌드와 스모크 하네스, 명시적 ABI manifest가 붙은 소비 가능한 아티팩트, 클린 빌드 검사, `gpu-lock` 아래 실제 GPU 실행을 요구했다. 컴파일만 한 결과로는 게이트를 통과할 수 없다.

## 2. 기술적 선택과 이유

### 2.1 자체 워크스페이스 루트를 가진 spike 패키지

`spike/cuda-oxide-smoke/Cargo.toml`에는 빈 `[workspace]` 테이블이 있다. 기존 `spike/iree-ffi`, `spike/rust-emitter`와 같은 방식이다. 루트 워크스페이스, 루트 `Cargo.lock`, `cargo tree -p mlxcel` 어디에서도 cuda-oxide 크레이트가 나오지 않고, 애플리케이션 툴체인은 1.97.1에 머문다. 코드젠 백엔드는 rustc 플러그인이라 자기를 적재하는 컴파일러와 버전이 맞아야 하므로, spike는 자체 `rust-toolchain.toml`에서 1.99.1을 고정한다. `Cargo.lock`을 커밋한 것도 의도적이다. 이 패키지는 바이너리와 디바이스 아티팩트를 만들므로 의존성 해석 자체가 아티팩트 재현성의 일부다.

### 2.2 `.oxart` 컨테이너는 아티팩트가 아니다

cuda-oxide는 디바이스 코드를 호스트 바이너리 안의 `.oxart` 섹션에 넣는다. 이 섹션은 매직 헤더, 타깃 문자열, 페이로드 레코드, 엔트리 레코드를 가진 버전 있는 컨테이너다. `scripts/oxide_artifact.py`가 컨테이너를 파싱해 PTX 페이로드를 꺼내고, 컴파일러가 따로 남긴 `.ptx` sidecar가 내장 페이로드와 바이트 단위로 같지 않으면 진행하지 않는다. 커밋된 cubin은 그 PTX를 `ptxas`로 어셈블한 결과다.

manifest에는 ABI 버전, 엔트리 심볼, 소스 리비전, 컨테이너 필드, 두 다이제스트, 각 인자의 순번·폭·포인터 여부·바이트 오프셋·파라미터 공간, 실행 제약이 들어간다. 검증은 순환하지 않는다. 사람이 쓴 주장은 `abi/vecadd.schema.json`이고, `validate`는 커밋된 PTX를 다시 파싱하고 cubin에 `cuobjdump -elf`와 `-sass`를 돌려 그 결과를 스키마와 manifest 양쪽에 대조한다. 컨테이너 페이로드 다이제스트가 커밋된 PTX 다이제스트와 같은지, manifest 소스가 스키마 리비전과 같은지도 확인한다.

### 2.3 기본 `sm_121` 타깃, 이름만 바꾼 아키텍처 특화 타깃이 아님

GB10은 compute capability 12.1을 보고한다. 이 호스트에서 cuda-oxide의 자동 디바이스 힌트는 `sm_121a`를 내놓고, 명시적으로 빌드한 `sm_121a` 모듈로도 14개 케이스가 모두 통과했다. 커밋한 아티팩트는 감지된 capability에 맞는 이식 가능한 기본 타깃인 `sm_121`을 쓴다. PTX 지시자, cubin 헤더, 로더가 서로 일치하고 모듈은 실제 디바이스에서 실행됐다. 이 결과는 Tensor Core 지원에 대해 아무것도 말하지 않는다. CuTe 경로는 실행하지 않았다.

### 2.4 launch 제약은 자기 자신이 아니라 harness와 대조한다

방출된 PTX에는 `.reqntid`나 `.maxntid`가 없고 cubin에도 스레드 수 속성이 없어서, 블록 크기를 아티팩트에서 끌어낼 수 없다. `smoke.sh`와 `sanitize.sh`가 harness 출력을 캡처하고, `oxide_artifact.py check-launch`가 모든 PASS 줄의 block이 manifest 블록 차원과 같은지, grid가 `ceil_div(n, block)`과 같은지, RESULT 집계가 파싱한 줄과 맞는지 검사한다. grid나 block이 틀리면 검사가 실패한다.

### 2.5 CI 제외는 이 패키지로 한정

새 `.rs`와 `Cargo.toml` 파일이 `rust`, `deps`, `cpu_link` 경로 필터에 걸려 해당 잡이 실행됐고, 선재하는 러너 결함 때문에 실패했다. self-hosted Linux 러너가 `mlxcel-core`의 CMake 빌드에서 BLAS를 찾지 못한다. 첫 수정은 `spike/**`를 제외했지만, 그러면 기존 spike 패키지 두 개의 트리거까지 사라진다. 최종 변경은 `rust`, `deps`, `rocm`, `cpu_link` 네 필터의 제외를 모두 `!spike/cuda-oxide-smoke/**`로 좁혔다. 러너 결함 자체는 건드리지 않았고 핵심 PR에서 계속 드러난다. 예를 들어 main run 41584274906이 그렇다.

## 3. 검증

오케스트레이터는 구현 보고를 그대로 받지 않고 리뷰 수정 뒤 하네스를 다시 돌렸다.

| 검사 | 결과 |
|---|---|
| `scripts/build.sh sm_121` | manifest 유효, payload name `cuda_oxide_smoke.ptx`, PTX 8.8, 타깃 `sm_121`, `u64` 인자 6개 |
| `scripts/smoke.sh sm_121` | 14/14 케이스, 69,913원소, exact-bitwise, launch 검사 유효 |
| `scripts/sanitize.sh sm_121` | `ERROR SUMMARY: 0 errors`, launch 검사 유효 |
| `scripts/check-unsupported-target.sh` | exit 101, `expected sm_<capability> with an optional 'a' suffix` |
| `scripts/check-clean-build.sh` | 워크스페이스·루트 lockfile·`mlxcel` 의존성 그래프에 cuda-oxide 없음, `rustc 1.97.1` 단언 |
| `scripts/check-lint.sh` | fmt 통과, clippy `-D warnings` |
| `scripts/check-guards.py` | 잘못된 입력 8종: traceback·행 없이 깨끗한 비영 종료 |
| 아티팩트 재현 | 재빌드해도 PTX·cubin 다이제스트 동일, sidecar가 커밋 PTX와 일치 |
| 변조한 manifest | `c_ptr` 폭 32를 스키마 64와 비교해 거부 |
| 다이제스트 | PTX `737559830058e1d142a80c3ce7afc668d5326a4fb1b15bf020f2f03fc71381a8`, cubin `4ab363a33b5b66ab75ddbbf30b8891e22e34b06559f6508f9f1593d6cdb199a6` |

증거는 `spark-102`에서 기록했다. NVIDIA GB10, compute capability 12.1, 드라이버 580.178.04, CUDA 13.0 V13.0.88, LLVM/Clang 21.1.8, rustc 1.99.1, cargo-oxide 0.3.0, 소스 리비전 `b631530700a8a4a5f165456f2d71fe9195b323b5`다. 호스트가 툴체인을 직접 제공하므로 업스트림 devcontainer는 쓰지 않았고, 그 경로는 문서화된 폴백으로 남는다.

구현 리뷰 두 번과 보안 리뷰 한 번에서 결함이 나왔고, 수정이 모두 다룬다. 첫 라운드는 payload 레코드 struct 레이아웃을 잡았다. 이름 길이를 이름 오프셋의 상위 절반에서 읽어 manifest에 빈 payload name이 들어갔고, evidence는 그 결함을 포맷 속성으로 서술했다. 같은 라운드에서 실제 launch 대조, gpu-lock을 강제하는 `sanitize.sh`, 패키지 범위 CI 제외, rustc 버전 단언, SIGPIPE에 안전한 파이프 검사, cubin 서술 정정, provenance 검증을 더했다. 보안 라운드는 길이 0인 `.oxart` 번들이 무한 루프를 돌며 메모리를 늘리는 문제, manifest가 지정한 아티팩트 경로가 루트 밖으로 나가는 문제, 정렬 지수와 서브프로세스 대기의 무제한, 그리고 예외를 보고로 바꾸지 못하는 두 경로를 찾았다. launch 검사는 이제 파싱한 케이스 수와 원소 합계를 선언된 값과 비교하므로, 자기 집계는 맞지만 줄어든 실행도 실패한다. `scripts/check-guards.py`가 이 잘못된 입력을 모두 실제 코드에 넣어 깨끗한 비영 종료를 요구한다.

## 4. 변경 요약

| 항목 | 값 |
|---|---|
| 변경 파일 | 이 리포트 작성 전 24개 |
| 추가 줄 | +3088 |
| 삭제 줄 | -1 |
| 신규 패키지 | `spike/cuda-oxide-smoke/` (커널, 하네스, 아티팩트 도구, ABI 스키마, 스크립트, 커밋된 아티팩트) |
| 증거 | `docs/mlxcelverse/cuda-oxide-smoke-evidence.md`, 타당성 검토에서 연결 |
| 공유 변경 | `.github/workflows/ci.yml` 경로 필터, `scripts/ci/check_binary_assets.py`의 cubin 허용 |
| mlxcel 런타임 코드 | 없음 |
| 테스트 | 독립 하네스의 디바이스 케이스 14개, mlxcel 테스트 스위트 변경 없음 |

## 5. 후속 조치

- #2832는 이 게이트에 막혀 있었고, 이제 타깃·아티팩트·manifest 계약을 물려받는다.
- #2833과 #2834는 선행 이슈에 계속 막혀 있다.
- self-hosted Linux 러너의 `cargo-clippy`·`CPU-only link` BLAS 탐색 실패는 선재 결함이고 핵심 PR에 영향을 준다. 별도로 고쳐야 한다. 이 PR은 격리된 spike가 그 잡들의 입력으로 취급되지 않게 했을 뿐이다.
- 아티팩트 계약에는 자체 CI 게이트가 없다. `oxide_artifact.py validate`는 Python과 `cuobjdump`만 있으면 되므로, 계약이 계속 바뀔 거라면 저렴한 잡을 붙일 수 있다.
- 커밋된 cubin은 `ptxas`가 어셈블했다. cuda-oxide의 `--materialize-cubin` 생산자는 쓰지 않았고, 도입한다면 manifest 항목이 따로 필요하다.
