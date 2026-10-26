# PR #2652: Windows 빌드 CI 잡 추가와 runtime-dir 호스트 테스트 실행

**날짜**: 2026-10-27
**상태**: 완료. 두 신규 잡이 PR에서 통과했고, `main`에서의 첫 실행은 머지 시점에 이루어짐
**언어**: GitHub Actions YAML, Python, Shell, Markdown
**위험도**: 낮음

## 요약

PR #2652(이슈 #2642)는 Windows 빌드 글루와 Windows 런타임 디렉터리 로직을 처음으로 CI에 올립니다.
- **`overlay-host-tests`**(ubuntu-latest, 약 20초, 무조건 실행): `make verify-rocm-overlay verify-cuda-runtime-dir`를 실행합니다. ROCm 오버레이 레코드 검사와 두 오버레이의 런타임 및 include 디렉터리 선택 호스트 테스트입니다.
- **`windows-build`**(windows-latest, 컴파일 전용, 캐시 적중 시 약 20분): `mlxcel-mlx-pin` lib 테스트, `tests/` Python 스위트, 다이제스트로 고정하고 캐시한 ROCm 10.1.0 TheRock tarball에 대한 `cargo check -p mlxcel-core --features rocm`를 실행합니다.
- **Windows 테스트 정리:** bash가 필요한 테스트 클래스는 `win32`에서 건너뛰고, Windows 콘솔에서 스위트를 깨던 인코딩 버그 두 건을 고쳤습니다.

## 1. 문제 정의

Windows CUDA, ROCm 빌드 경로(#2599, #2603, #2604)와 재배치 가능한 런타임 로직(#2607, #2648, #2651)은 수동으로만 검증되었습니다. 선택 로직의 호스트 테스트 두 개는 어떤 CI 잡도 호출하지 않는 `make verify-*` 타깃에서만 돌았고, 실제 Windows `build.rs`를 돌리는 곳이 없었습니다. include 순서 회귀나 MLX 핀 갱신으로 인한 빌드 파손이 초록불로 머지될 수 있었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 호스트 테스트 | 새 `overlay-host-tests` 잡. 잡이 실행하는 파일 일부(`rocm_overlay.py`, `allocator_ownership_test.*`, Makefile 타깃)가 어떤 경로 필터에도 걸리지 않고 실행이 수 초이므로 무조건 실행합니다. 레코드 검사는 자체 합성 git 히스토리를 만들기 때문에 얕은 클론으로 충분합니다. |
| Windows 잡 | 호스티드 러너의 새 `windows-build` 잡. `cargo check`는 빌드 스크립트 출력을 숨기므로, 마지막 단계에서 `mlx_rocm_kernels.lib` 존재를 확인해 HIP 빌드가 실제로 돌았음을 증명합니다. |
| SDK tarball | `gfx1151` TheRock 10.1.0 tarball(1.65 GB)을 직접 계산한 sha256으로 고정합니다. AMD가 체크섬을 공개하지 않기 때문입니다. 캐시 키는 버전, 패밀리, 다이제스트입니다. 다운로드 후와 매 복원 후 다이제스트를 검사하고, 검증 직후 캐시를 저장해 빌드가 실패해도 캐시가 남습니다. 압축 해제 시 composable-kernel 라이브러리, MIOpen, 테스트 플러그인을 제외합니다(8.4 GB 중 3.8 GB). |
| 게이팅 | 새 `windows_build` 필터: 루트와 코어 `build.rs`, `build_support/**`, `mlxcel-mlx-pin/**`, MLX 핀, ROCm 및 CUDA 오버레이의 `delayload.cpp`/`runtime_dir.h`/`jit_module.cpp`, runtime-dir 테스트 스크립트, 패키징 스크립트와 그 테스트. 호스티드 러너만, `contents: read`, 자격 증명 비보존. |
| 테스트 | 다섯 스위트(및 `test_bench_serving_concurrency.py`)의 bash 기반 클래스를 `win32`에서 건너뜁니다. 순수 Python 클래스는 Windows에서도 실행됩니다. 로캘 코드 페이지 읽기와 `insert_apache_header.py`의 이모지 출력 문제를 고쳤습니다. |
| 문서 | `docs/installation.md` 제한 표, `CONTRIBUTING.md` 잡 목록, `docs/mlxcelverse/rocm-overlay.md`. |

## 3. 기술적 선택과 그 이유

### 3.1 부분 검사 대신 전체 ROCm 검사

이슈는 Windows 검사의 깊이를 열어 두었습니다. 확정 전 임시 브랜치에서 측정했습니다. 캐시 적중 시 약 20분(그중 `cargo check` 12~17.5분), 미스 시 14분에 다운로드 5초, 캐시 항목 1.6 GB입니다. 이슈에 정한 기준(45분 미만, 4 GB 미만)에 들어가므로 configure만 하지 않고 MLX ROCm 빌드 전체를 컴파일합니다. CUDA 변형은 선택 사항이라 추가하지 않았습니다.

### 3.2 호스트 테스트 잡을 무조건 실행으로

첫 버전은 `overlay-host-tests`를 `rocm`과 `windows_build` 필터로 게이팅했습니다. 리뷰에서 잡이 실행하는 파일 중 두 필터 모두 매칭하지 않는 것이 발견되었습니다. 약 20초짜리 잡에서는 게이팅으로 아끼는 시간보다 놓치는 커버리지가 더 크므로, 이제 `python-tooling`처럼 모든 PR과 push에서 실행합니다.

### 3.3 게이트가 회귀를 잡는지 증명

한 커밋에서 `choose_rocm_include_dir()`의 `HIP_PATH`와 번들 분기를 바꿨고, CI가 `FAIL: include: bundle beats HIP_PATH`로 실패했으며, 다음 커밋에서 되돌렸습니다. 되돌린 뒤에는 PR diff가 게이팅 경로를 건드리지 않아 두 잡이 건너뛰어졌습니다. 이를 강제할 `ci:full` 레이블이 저장소에 없어서, 두 호스트 테스트 스크립트에 CI 잡 이름을 적는 주석을 넣어 diff에 남겼습니다. 스쿼시 커밋도 이 경로를 포함하므로 `windows-build`가 `main`에서 한 번 실행되어 `main` 범위의 캐시 항목을 만듭니다.

### 3.4 캐시 범위

PR은 자기 `refs/pull/N/merge` 캐시 범위에만 쓸 수 있고 매 복원 후 다이제스트를 검사하므로, 포크 PR이나 손상된 항목이 다른 실행에 다른 SDK를 넘길 수 없습니다.

## 4. 검증

- Windows 11 호스트: `python -m unittest discover -s tests -p 'test_*.py'`(188개 실행, 76개 건너뜀), `cargo test -p mlxcel-mlx-pin --lib`(49개), ROCm `clang++`로 두 호스트 테스트, actionlint 1.7.7 통과.
- CI: 의도적 파손은 실패(run 40406365843), 최종 트리는 두 잡 모두 통과(run 40415050086: 호스트 테스트 21초, Windows 캐시 적중 19분 50초).
- 보안 리뷰: 이벤트 데이터가 `run:` 스크립트로 들어가는 경로 없음, 읽기 전용 권한, 워크플로의 태그 방식과 같은 액션 고정.

## 5. 남은 공백

- **`ci:full` 레이블 부재:** `ci.yml`이 참조하지만 저장소에 정의되어 있지 않습니다. 관리자가 만들기 전까지는 `windows-build` 정의만 바꾸는 PR(예: 다이제스트 갱신)이 머지 전에 그 잡을 돌릴 수 없고, self-hosted 잡의 레이블 opt-in도 효과가 없습니다.
- **Windows Python 스위트가 게이팅됨:** Windows에서만 깨지는 `tests/` 변경은 게이팅 경로를 건드리는 다음 PR에서야 실패합니다. 스위트만 돌리는 작은 비게이팅 Windows 잡으로 해결할 수 있습니다.
- **부분적인 오버레이 커버리지:** 필터는 오버레이마다 세 파일만 나열하지만 검사는 HIP 오버레이 전체를 컴파일합니다. 넓히면 ROCm 오버레이 PR마다 20분 실행이 추가됩니다.
- **링크 단계 없음:** `cargo check`는 링크하지 않으므로 루트 `build.rs`의 delay-load 링크 인자는 여전히 검증되지 않습니다.
- **캐시 압박:** 저장소가 이미 10 GB 캐시 한도 중 약 10.5 GB를 쓰고 있어 LRU 축출로 SDK 항목이 빠질 수 있습니다. 미스는 실패가 아니라 다운로드 한 번이 늘 뿐입니다.
- **과도한 건너뛰기:** bash가 필요 없는 일부 테스트(Makefile 텍스트 검사, 하드웨어 태그 테스트 하나)도 Windows에서 건너뜁니다. Linux에서는 계속 실행됩니다.

## 6. 학습 포인트

- **값싼 잡을 게이팅하면 커버리지를 잃는다:** 경로 필터는 잡의 모든 입력을 나열해야 하며, 수 초짜리 잡에서는 필터가 아끼는 것보다 잃는 커버리지가 대체로 더 큽니다.
- **`cargo check`는 빌드 스크립트를 조용히 실행한다:** 빌드 스크립트가 반드시 만드는 산출물을 확인해야 합니다. 그렇지 않으면 건너뛴 네이티브 빌드가 성공처럼 보입니다.
- **부정을 증명하려면 게이트가 실제로 동작해야 한다:** 의도적 파손을 되돌린 뒤에는 실행이 초록색인지만이 아니라 잡이 실제로 돌았는지 확인해야 합니다.
