# PR #2786: N1X에서 수행한 Windows CUDA 런타임 번들 검증 기록

**작성일**: 2026-10-31 (머지 전 작성)
**상태**: 리뷰 중
**언어**: Markdown, Python (주석 하나)
**위험도**: 낮음 (런타임 코드 변경 없음)

## 요약

#2640은 Windows CUDA 런타임 번들을 추가했지만 개발 호스트에 NVIDIA GPU가 없어 빌드 검증만 하고 닫혔다. #2653에서 arm64 RTX Spark N1X에 툴킷 이름을 바꾼 채 번들을 실행했다. 콜드 JIT 캐시의 `mlxcel generate`와 `mlxcel-server` 채팅 요청이 번들 폴더와 `MLXCEL_CUDA_RUNTIME_DIR` 양쪽에서 동작했고, 목록에서 빠진 DLL이나 헤더는 없었다. 이 PR은 문서의 '빌드 검증만 함' 서술을 이 결과로 바꾸고, 실행 중 드러난 두 가지 공백을 문서에 적는다.

## 1. 문제 정의

`scripts/package_windows_cuda_runtime.py`의 DLL 목록과 `JIT_HEADERS`는 x64 DLL의 정적 분석과 NVRTC 단독 컴파일로 만든 것이었다. #2640의 인수 조건 여섯 개 중 네 개가 미확인이었고, 실제 실행이 어떤 DLL을 로드하는지와 장치 초기화 이후의 동작이 여기에 포함됐다.

## 2. 기술적 선택과 그 이유

### 2.1 cuDNN 경로를 일부러 실행

Gemma 4 E4B의 짧은 프롬프트로는 cuDNN DLL이 하나도 로드되지 않아서, 이 실행만으로는 목록의 cuDNN 쪽이 검증되지 않는다. Llama 3.1 8B의 3950토큰 프롬프트가 cuDNN 어텐션 경로에 도달했고 cuDNN DLL 10개 모두 번들에서 로드됐다. `listdlls`가 없어 프로세스 수명 동안 500 ms 간격으로 모듈 목록을 수집했으므로, 한 간격 안에 로드와 해제가 끝난 DLL은 놓칠 수 있다.

### 2.2 환경 변수 시험에는 빌드 트리 바이너리를 사용

번들 전체를 옮기면 바이너리가 옆에 있는 DLL을 찾으므로 변수가 디렉터리를 고르는 상황이 아니다. 그래서 툴킷을 치운 채 `target`의 바이너리를 옮긴 번들의 `bin`과 함께 실행해 `MLXCEL_CUDA_RUNTIME_DIR`만 남겼고, 출력된 런타임 줄로 이를 확인했다.

### 2.3 문서만 변경

빠진 파일이 없었으므로 목록이 미확인이라고 적힌 주석 외에 코드는 바꾸지 않았다. `scripts/server_chat_smoke.sh`는 Windows Python에서 `/tmp`를 열다 실패하는데, 번들 범위 밖이라 그대로 두고 서버 확인은 `curl`로 했다.

## 3. 변경 요약

- `docs/installation.md`: Windows 제한 표의 행, DLL 목록의 출처, 평문 이름 로드 결과, CCCL 라이선스 출처, Visual C++ 런타임과 `openblas.dll` 주의 사항.
- `scripts/package_windows_cuda_runtime.py`: 출처 주석.

## 4. 결과

| 확인 항목 | 결과 |
|---|---|
| 번들 폴더, 툴킷 없음 | generate 45.1 tok/s, 서버 채팅 응답 |
| `MLXCEL_CUDA_RUNTIME_DIR`, 툴킷 없음 | generate 44.8 tok/s, 서버 채팅 응답 |
| 툴킷 있음, 번들 없음 | generate 43.6 tok/s, 런타임 줄에 툴킷 경로 표시 |
| `cudnn_ops`, `nvrtc-builtins`, `cusparse` 제거 | 기대 경로를 알리고 멈춤, `cudnn_ops`의 종료 코드 127 |
| `cufft` 제거 | generate 성공. cuFFT는 처음 호출할 때 로드 |
| 목록에 있으나 로드되지 않음 | cuFFT, cuSOLVER, cuSPARSE |

평문 이름 로드 결과는 #2641에 올렸다. `nvcuda.dll`은 System32, 드라이버의 `nvcudart_hybrida64.dll`은 DriverStore 폴더에서 로드됐고 `zlibwapi.dll`, `cask_profile_interface.dll`, `nvtileiras64.dll`은 로드되지 않았다.

## 5. 트레이드오프와 남은 문제

- **미실행:** x64 GPU 호스트, cuFFT와 cuSOLVER 호출, 디스플레이 드라이버만 있는 머신. 검증 호스트에는 Visual Studio가 있고 `MSVCP140.dll`과 `VCRUNTIME140.dll`을 System32에서 로드했으므로, 번들이 드라이버만으로 동작한다는 주장은 아직 시험하지 않았다.
- **라이선스 대체 파일:** `NVIDIA/cccl` 태그의 `LICENSE`를 나중에 찾았기 때문에 검증한 번들에는 CCCL 헤더를 대신 넣었다. 그 파일을 쓴 패키징은 따로 확인했고, DLL 로드에는 영향이 없다.
- **디코이:** 이번 실행은 무엇이 로드되는지를 기록할 뿐, 검색 순서에 파일을 심을 수 있는지는 시험하지 않았다. 이 부분은 #2641에 남아 있다.
- **로컬 게이트:** `make verify-cuda-runtime-dir`에 필요한 `c++`가 이 호스트에 없다. CI의 overlay host test 작업에서 실행해 통과했다.
- **다른 매뉴얼:** 이슈의 문서 조건에는 이 저장소 밖의 매뉴얼도 포함된다. 별도 PR로 수정했고, 이 보고서를 쓴 시점에는 아직 열려 있다.

## 6. 학습 포인트

- 정적 DLL 목록은 아키텍처에 따라 다르다. x64 목록의 `nvcudart_hybrid64.dll`이 arm64에서는 `nvcudart_hybrida64.dll`로 로드된다.
- 목록의 절반을 로드하지 않은 번들 시험도 겉보기에는 통과와 같다. 첫 generate 실행이 cuDNN에 도달하지 않았다는 사실은 모듈 목록을 확인해서 드러났다.
- Windows에서 시험 스크립트의 `PATH`를 줄이면 `timeout`이 `TIMEOUT.EXE`로 잡힌다. 첫 실행은 이 때문에 `mlxcel`을 시작하지도 못하고 실패했다.
