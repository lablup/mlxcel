# PR #2648: Windows ROCm 번들 헤더를 HIP_PATH보다 우선하고 GPU 타깃 추가

**날짜**: 2026-10-27
**상태**: Windows(gfx1151)에서 완료. 나머지 여섯 타깃은 빌드만 검증
**언어**: C++, CMake, Python, Markdown
**위험도**: 낮음

## 요약

PR #2648(이슈 #2639)은 #2607의 Windows ROCm 런타임 번들에 남아 있던 빈틈 두 가지를 메운다.

- **헤더 우선순위:** 시스템 전역 `HIP_PATH`가 더는 번들의 JIT 헤더를 덮어쓰지 않는다. 환경 변수와 번들이 서로 다른 헤더를 가리키면 두 디렉터리를 모두 밝히는 줄을 stderr에 출력한다.
- **GPU 타깃:** 배포용 빌드의 일곱 타깃 목록을 정했다. SDK에 커널 데이터가 없는 타깃은 패키징 스크립트가 거부한다. 모든 바이너리는 디바이스 코드를 컴파일한 타깃 목록을 출력한다.

RAM 32 GB인 검증 호스트에서 일곱 타깃 바이너리는 링크 병렬도를 낮춰야 링크된다. 크기는 gfx1151 전용 바이너리의 3.9배다.

## 1. 문제 정의

런타임 번들은 ROCm 릴리스 하나로 처음부터 끝까지 실행하려고 만든 것이다. DLL은 한 디렉터리에서, hiprtc 헤더는 그 디렉터리의 `include\`에서 가져온다.

AMD HIP SDK 설치 프로그램은 `HIP_PATH`를 시스템 전체에 설정한다. #2607 이후로는 이 값이 번들 헤더보다 우선했기 때문에, HIP SDK가 설치된 머신에서는 ROCm 10.1 번들이 다른 릴리스의 헤더로 fused 커널을 컴파일했다. 번들이 막으려던 버전 혼합이 바로 이것인데, 그동안은 문서 경고뿐이었다.

이와 별개로, 검증 빌드는 gfx1151용 디바이스 코드만 컴파일했다. 그래서 번들이 다른 RDNA GPU에서는 실행되지 않았다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 헤더 순서 | `choose_rocm_include_dir()`가 `runtime_is_bundle`을 받는다. Windows 순서는 `ROCM_HOME`, `ROCM_PATH`, 번들 include(런타임 출처가 `MLXCEL_ROCM_RUNTIME_DIR`이나 실행 파일 디렉터리이고 `hip\hip_runtime.h`가 있을 때), `HIP_PATH`, 빌드 시점 SDK일 때의 런타임 include, 내장 include 순이다. Linux는 번들 플래그가 false이고 `HIP_PATH`를 읽지 않으므로 `-I` 문자열이 그대로다. |
| 알림 줄 | `runtime_dir.h`의 순수 함수 `rocm_include_dir_notice()`가 출력한다. `ROCM_HOME`/`ROCM_PATH`가 번들을 덮어쓰거나 번들이 `HIP_PATH`를 이기면 `[mlx-rocm] hiprtc headers: <chosen> (<source>); ignoring <other> from <source>`를 낸다. 두 값이 같은 디렉터리면 출력하지 않는다(구분자, 끝 구분자, ASCII 대소문자를 정규화해 비교). `delayload.cpp`는 `rocm_runtime_dir_source()`를 노출한다. |
| 타깃 로그 | overlay CMake가 `CMAKE_HIP_ARCHITECTURES`를 쉼표로 이어 `MLXCEL_ROCM_ARCHITECTURES`로 넘긴다. `device.cpp`는 bound-device 줄 옆에 `[mlx-rocm] compiled HIP targets: ...`를 한 번 출력한다. |
| 타깃 검증 | 목록이 이제 C 문자열 리터럴에도 들어가므로, HIP target ID 문법에 맞지 않는 `MLX_ROCM_ARCHITECTURES` 항목이 있으면 configure가 `Not a HIP target ID`로 실패한다. |
| 패키징 | `package_windows_rocm_runtime.py`가 `.kpack`이나 rocBLAS/hipBLASLt 데이터가 없는 `--arch`를 모두 밝히며, 아무것도 쓰기 전에 종료 코드 1로 끝난다. |
| 테스트 | 새 순서와 알림을 다루는 호스트 C++ 테스트 항목, 누락 타깃 패키징 테스트 5개를 추가했다. `tests/test_rocm_arch_validation.py`는 실제 CMake 검증 블록을 `cmake -P`로 실행한다. |
| 문서 | 새 순서와 다시 쓴 `HIP_PATH` 규칙, 그리고 'GPU targets for distributable builds' 절을 추가했다. 이 절에는 측정 크기, 커널 데이터 출처인 multiarch tarball, 링크 병렬도 안내가 들어 있다. 허용되는 target ID 형식도 적었다. `LOCAL_FIXES.md` 66번. |

## 3. 기술적 선택과 그 이유

### 3.1 번들은 HIP_PATH보다 우선하지만 ROCM_HOME, ROCM_PATH보다는 아래

`HIP_PATH`는 설치 프로그램이 머신의 모든 프로그램에 한 번 설정하는 값이다. `ROCM_HOME`과 `ROCM_PATH`는 사용자가 셸 하나나 실행 한 번을 위해 일부러 지정한다. 번들을 이길 수 있는 것은 의도적인 지정뿐이고, 그 경우에도 알림 줄이 남는다.

신뢰 경계는 보안 리뷰에서 확인했다. 번들 include는 실행 파일 디렉터리나 `MLXCEL_ROCM_RUNTIME_DIR`에 있다. 거기에 쓸 수 있는 사람은 이미 바이너리와 DLL을 바꿀 수 있으므로, 번들을 `HIP_PATH`보다 앞에 둬도 커널을 바꿀 수 있는 주체가 새로 생기지 않는다.

### 3.2 선택 함수를 순수하게 유지

`choose_rocm_include_dir()`와 새 알림 함수는 값을 받아 문자열을 돌려준다. 호스트 테스트는 `jit_module.cpp`에서 awk로 `choose_rocm_include_dir()`를 뽑아 쓰므로, HIP나 Windows 없이 C++17 컴파일러만 있으면 어디서든 돈다. 새 매개변수를 넣어도 이 추출은 그대로 동작한다.

### 3.3 정한 타깃 목록과 커널 데이터의 출처

목록은 RDNA3 `gfx1100;gfx1101;gfx1102`, RDNA3.5 `gfx1150;gfx1151`, RDNA4 `gfx1200;gfx1201`이다.

설치된 gfx1151 TheRock tarball은 디바이스 bitcode가 모든 타깃을 다루므로, MLX를 일곱 타깃 모두로 컴파일한다. 하지만 rocBLAS·hipBLASLt 커널 데이터와 `.kpack`은 gfx1151만 들어 있고, 계열별 tarball도 각자 자기 계열만 담고 있다. AMD의 `therock-dist-windows-multiarch-10.1.0.tar.gz`(4.9 GB)에는 일곱 타깃이 모두 있다. 이는 풀지 않고 `tar -t`를 스트리밍해 확인했다. 문서와 스크립트 오류 메시지는 이 파일을 안내한다.

여러 SDK 루트의 커널 데이터를 합치는 기능은 넣지 않았다. 누락 타깃을 밝히며 실패하기만 해도 반쪽짜리 번들은 막힌다.

### 3.4 configure 단계에서 타깃 목록 검증

목록의 각 항목은 `--offload-arch` 플래그가 되고, 이 PR부터는 C 문자열 리터럴에도 들어간다. 그래서 환경 변수에 따옴표가 들어 있으면 빌드에 코드를 넣을 수 있었다. 빌드 환경을 통제하는 사람만 설정할 수 있는 값이라 리뷰는 LOW로 분류했다.

`gfx[0-9a-f]+`처럼 단순한 검사는 `gfx90a:xnack+`나 `gfx11-generic` 같은 유효한 ID를 거부한다. 그래서 target ID 문법을 그대로 따랐다.

## 4. 검증

모든 실행은 Windows 11, Radeon 8060S(gfx1151), ROCm 10.1.0에서 했다.

- **`HIP_PATH` 미끼:** `hip_runtime.h`가 `#error`인 폴더를 `HIP_PATH`로 지정했다. 번들 실행은 번들 include로 커널 세 개를 JIT 컴파일해 올바른 결과를 냈고 알림 줄을 출력했다. 빌드 트리 바이너리(빌드 시점 SDK 출처)는 설계대로 미끼 오류에 걸렸다.
- **실행 시 `ROCM_PATH`:** 여전히 우선했고, 알림 줄이 무시한 번들 include를 밝혔다.
- **번들 실행(gfx1151 빌드):** Qwen3-0.6B-4bit(74~78 tok/s)와 Qwen3-1.7B-bf16(26.9 tok/s) `generate`가 "Paris"라고 답했다. 서버 chat, 스트리밍, `/v1/messages`도 정상이었고, ROCm 모듈은 모두 번들 `bin`에서 로드되었다. 검증 기능을 넣은 뒤 재빌드한 바이너리는 76.8 tok/s로 생성했다.
- **일곱 타깃:**
  - `-j 8` 빌드는 HIP 오브젝트 40개(각각 코드 오브젝트 일곱 개 포함)를 모두 컴파일한 뒤 메모리 부족으로 링크에 실패했다(`link.exe` 0xC0000142).
  - `cargo build -j 2 --bin mlxcel --bin mlxcel-server`로 7분 1초 만에 둘 다 링크했다.
  - gfx1151 전용 대비 크기는 `mlxcel.exe` 745,592,320 대 193,465,856바이트, `mlxcel-server.exe` 735,546,880 대 183,420,416바이트, 커널 아카이브 716.6 대 190.1 MiB다.
  - HIP 단계는 5분 8초 대 3분 6초였다(트리 상태가 다름).
  - gfx1151에서 일곱 타깃 바이너리는 타깃 일곱 개를 모두 출력했고 75 tok/s로 "Paris"를 생성했다.
- **패키징:** gfx1151 SDK로 gfx1100, gfx1201을 요청하면 종료 코드 1로 끝나며 두 타깃을 밝히고, 아무것도 쓰지 않는다.
- **검사:** 호스트 C++ 테스트(main에서는 실패), Python 테스트 24개, `cargo test -p mlxcel-mlx-pin --lib`(49개), fmt, clippy `-p mlxcel-core --features rocm -D warnings`, overlay drift와 records, 환경 변수 문서 검사가 모두 통과했다.

## 5. 남은 공백

- **실행해 보지 않은 타깃:** gfx1100, gfx1101, gfx1102, gfx1150, gfx1200, gfx1201은 컴파일만 되고 실행해 보지 않았다. 하드웨어가 없다.
- **빌드 시간:** 일곱 타깃의 깨끗한 전체 빌드 시간은 재지 않았다. HIP 단계 시간 두 개는 서로 다른 트리 상태에서 쟀다.
- **링크 메모리:** 일곱 타깃 링크는 32 GB 호스트에서 `-j 8`이 남기는 것보다 많은 메모리를 쓴다. 문서는 `-j 2`를 권하지만 강제하지는 않는다.
- **일곱 타깃 번들은 아직 없음:** 일곱 타깃 번들에는 4.9 GB multiarch SDK가 필요하고, 이것으로 처음부터 끝까지 패키징해 본 적은 없다.
- **Linux:** CMake 검증은 Linux 빌드에서도 돌지만 Linux 호스트에서 실행해 보지 않았다. Linux ROCm CI 잡은 멈춰 있다(#1811).
- **비ASCII 경로의 알림 줄:** `getenv`는 ANSI를, 런타임 include는 UTF-8을 돌려주므로 비ASCII 경로에서는 불필요한 알림 줄이 나올 수 있다. 선택되는 디렉터리는 바뀌지 않는다.

## 6. 학습 포인트

- **설치 프로그램이 정한 변수:** 머신 전체에 설정된 변수에는 사용자 의도가 담겨 있지 않다. 이를 실행별 지정과 같은 순위로 다루면 설치된 제품 하나가 다른 제품의 설정을 조용히 바꾼다.
- **타깃별 정적 라이브러리:** 크기가 타깃 수에 비례해 늘어난다. Windows에서는 컴파일보다 링크 단계가 먼저 메모리 한계에 닿으므로, 낮춰야 할 것은 링크 병렬도다.
- **소스 코드에 복사되는 값:** 생성 소스 텍스트 안으로 들어가는 빌드 입력은 실제 문법으로 검증해야 한다. 유효한 입력까지 거부하는 단순화된 패턴으로는 부족하다.
