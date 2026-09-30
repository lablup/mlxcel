# 기술 보고서: PR #2078 - mlxcelverse ROCm 동기화, drift 검사, fork PR 패키지

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료 (`origin/main` `c5a71cfa` 위의 `update/issue-1813-mlxcelverse-sync` 브랜치, `53c8f122`). 머지 대기 중.

**언어**: Python, Bash (도구), Make, Markdown, git 패치. HIP 파일 하나에 주석만 바뀐 변경

**위험도**: 빌드에는 낮음 (컴파일되는 코드의 기능 변경 없음. overlay 변경은 `rope.hip`의 주석 되돌리기뿐). 절차 측면에서는 중간: `make verify`와 `make verify-rocm`에 새 게이트가 추가되어, ROCm retarget을 빠뜨린 pin bump는 어느 호스트에서든 실패한다.

## 요약

이슈 #1813 (에픽 #1801의 3단계)은 mlxcel ROCm overlay의 두 upstream, 즉 MLX pin (ml-explore/mlx)과 백엔드를 가져온 fork (NripeshN/mlx `rocm-support`)를 따라가기 위한 도구와 문서화된 절차, 그리고 overlay가 가진 fork 측 수정을 fork로 돌려보내는 일을 요구했다.

이 PR은 `scripts/mlxcelverse/rocm_overlay.py`를 추가한다. 하위 명령은 `records`, `drift`, `sync`, `retarget`, 그리고 `check_api_drift.sh`가 쓰는 `export-tree`/`api-report`다. 진입점 `sync_from_fork.sh`, `check_api_drift.sh`와 29개 검사로 된 합성 이력 테스트도 함께 들어간다. 핵심은 drift 검사다. 15개 core overlay 파일 각각을 "MLX pin에 fork가 그 파일에 가한 변경을 합친 것"으로 다시 만들고, 남는 차이(overlay의 residual)가 리뷰를 거친 기록 `patches-rocm/CORE_RESIDUAL.diff`와 일치해야 한다. 이 기록의 각 hunk에는 해당 `LOCAL_FIXES.md` 항목을 가리키는 note가 붙는다. 기존의 +/- 줄 수 비교와 달리, 이 방법은 bump 이전부터 이미 잘못되어 있던 overlay를 잡아낸다. 오프라인으로 도는 부분은 `make verify-rocm-overlay`가 되었고, 이 타깃은 `make verify`와 `make verify-rocm` 양쪽에 들어간다.

PR은 또한 `docs/mlxcelverse/upstream/` 아래에 fork용 PR 패키지 11개(패치, PR 노트, 재현 스크립트)를 준비한다. LOCAL_FIXES 항목 8, 10~14, 16~27을 다룬다. 하나도 제출되지 않았다. 메인테이너의 원칙은 패키지 준비는 자동화가, 제출은 본인이 손으로 한다는 것이다. 또 ml-explore/mlx의 AI 사용 정책(#4331) 때문에 각 `pr.md`는 완성된 설명이 아니라 메인테이너가 자기 말로 다시 쓸 노트다.

구현 세션에서 auto 모드 권한 검사가 fork 코드 빌드를 거부했다. 그래서 `check_api_drift.sh` 전체 실행, 패키지를 fork에 대고 컴파일하기, fork 빌드에서 재현 스크립트 실행은 하지 못했다. 이들은 제출 전 단계로 기록되어 있다.

## 1. 문제 정의

ROCm overlay(`src/lib/mlx-cpp/patches-rocm/`)는 `mlx/backend/rocm/` 아래 백엔드 파일 108개와, fork의 hook을 합친 MLX core 파일 15개의 전체 파일 사본으로 이루어진다. `UPSTREAM`은 커밋 세 개를 기록한다: fork 커밋 `75915908`, fork와 MLX의 merge base `39886de4`, core 파일을 retarget한 MLX pin `81ba1c6a`. 이 PR 이전에는:

- **두 upstream 어느 쪽에도 도구가 없었다.** `scripts/mlxcelverse/`는 존재하지 않았고, pin bump 절차는 `patches-rocm/README.md`의 세 줄과 `CONTRIBUTING.md`의 한 문단이 전부였다. #1801의 retarget은 upstream 커밋 360개를 넘으며 손으로 여섯 군데를 고쳐야 했다.
- **bump 검사는 잘못된 overlay를 볼 수 없었다.** 이전과 이후 upstream base에 대한 overlay의 +/- 줄 수 비교는 upstream 변경이 들어왔고 로컬 변경이 남았다는 것만 확인한다. overlay가 애초에 맞았는지는 알 수 없다. Metal overlay 하나가 upstream 수정의 절반만 가진 채 몇 달을 지냈고, 리뷰어가 한 줄씩 비교하고 나서야 발견되었다 (TECHNICAL_REPORTS/1772, 3절).
- **fork로 돌아간 것이 없었다.** `LOCAL_FIXES.md`는 27개 항목으로 늘었고 상당수가 "fork에도 해당, fork에 제안 예정"으로 표시되어 있었지만, upstream 링크가 달린 항목은 없었다.
- **기록이 어긋나 있었다.** README는 백엔드 파일을 106개라고 했고(실제 108개), 몇몇 LOCAL_FIXES 항목은 건드린 파일을 명시하지 않았으며, `rope.hip`은 어느 항목에도 기록되지 않은 주석 줄바꿈 차이만으로 fork와 달랐다.

## 2. 변경 요약

- **`scripts/mlxcelverse/rocm_overlay.py`** (신규, 약 1100줄): `records`, `drift`, `sync`, `retarget`, `export-tree`, `api-report`. git 객체는 두 remote를 모두 가진 캐시 저장소(`~/.cache/mlxcel/mlxcelverse/mlx.git`)에서 가져온다. 도구는 fetch만 하고 push는 하지 않는다.
- **`scripts/mlxcelverse/sync_from_fork.sh`**, **`check_api_drift.sh`** (신규): 이슈가 이름을 정한 진입점. `check_api_drift.sh`는 `build.rs`가 넘기는 ROCm 옵션으로 MLX를 단독 configure하고, 실패하는 translation unit을 모두 보도록 `make -k`로 빌드한 뒤 object들의 정의/미정의 심볼을 비교한다. libmlx가 정적 라이브러리라서 ROCm `eval_gpu`가 없는 primitive는 미정의 `mlx::core` 심볼로만 드러나기 때문이다.
- **`scripts/mlxcelverse/rocm_overlay_test.sh`** (신규): 임시 저장소에 만든 합성 upstream/fork 이력 위에서 29개 검사.
- **`src/lib/mlx-cpp/patches-rocm/CORE_RESIDUAL.diff`** (신규, 생성 파일): byte 단위로 재구성되지 않는 core 파일 4개의 residual과 파일별 note.
- **`Makefile`**: 새 `verify-rocm-overlay` 타깃, `verify`와 `verify-rocm`에 추가.
- **`docs/mlxcelverse/rocm-overlay.md`** (신규): 기록 파일, 도구, pin bump와 fork 동기화 절차, bump PR 본문과 보고서에 넣을 "ROCm overlay" 절 템플릿. `CONTRIBUTING.md`, `patches-rocm/README.md`, `docs/architecture.md`, `docs/README.md`, `src/lib/mlx-cpp/CMakeLists.txt`의 overlay 주석이 이 문서를 가리키거나 새 기록 파일을 나열한다.
- **`docs/mlxcelverse/upstream/`** (신규): 색인과 패키지 디렉터리 11개.
- **기록**: README 백엔드 수 106에서 108로. LOCAL_FIXES는 항목 2, 7, 11, 20의 파일을 명시하고 파일 명시 규칙을 적었다. `rope.hip` 주석은 fork 문구로 되돌렸다.

scripts와 docs 밖의 코드 diff는 `rope.hip` 주석 되돌리기와, 어떤 빌드도 컴파일하지 않는 `docs/` 아래 재현용 `.hip`(패키지 06)뿐이다.

## 3. drift 검사가 이미 잘못된 overlay를 잡는 방식

### 3.1 재구성

drift는 core 파일마다 overlay가 되어야 할 파일을 만든다.

```
reconstruct(path) = git merge-file  pin:path  fork-merge-base:path  fork:path
```

pin의 사본에 fork가 그 파일에 가한 변경을 합친 것이다. 3-way base는 fork의 upstream merge base이고, 충돌 표시는 그대로 둔다. fork의 자기 merge base 대비 diff를 pin 위에 적용하기 때문에, fork가 MLX를 머지하며 가져온 upstream 변경을 fork의 변경으로 착각하지 않는다. fork에 파일이 없으면 재구성 결과는 pin의 사본이다.

그다음 overlay를 재구성 결과와 diff한다. 남는 것은 overlay가 스스로 더한 것, 즉 충돌 해결과 로컬 수정뿐이다. `81ba1c6a`/`75915908` 기준으로 core 파일 15개 중 11개는 byte 단위로 일치한다. 나머지 4개의 residual은 모두 설명된다:

| Core 파일 | residual의 근거 |
|---|---|
| `mlx/backend/common/compiled.cpp` | 항목 1: upstream의 `negative_strides` 추적과 fork의 상수 입력 건너뛰기를 모두 유지 |
| `mlx/fast_primitives.h` | 항목 1: upstream의 `compile_options`와 fork의 `output_input_aliases`를 모두 유지 |
| `mlx/io/safetensors.cpp` | 항목 1: upstream의 빈 텐서 가드와 fork의 ROCm `staged_write`를 모두 유지 |
| `mlx/backend/gpu/primitives.cpp` | 항목 24: `DynamicSliceUpdate`가 `copy_gpu`로 복사하고 fork의 강제 donation을 제거 |

### 3.2 리뷰된 기록

residual은 `CORE_RESIDUAL.diff`로 커밋된다. 헤더에는 커밋 세 개와 모든 core overlay 파일의 SHA-256이 기록되고, residual이 있는 파일마다 해당 LOCAL_FIXES 항목을 적은 `# note <path>:` 줄이 필요하다. 다시 계산한 residual이 기록과 다르면 drift는 실패하고, 차이를 hunk 단위로 출력하며, 받아들이려면 `--write`가 필요하다. `--write`는 기존 note를 유지하고 note가 필요한 새 파일을 알려 준다.

잘못된 overlay는 이렇게 드러난다. residual은 "upstream도 fork도 하지 않는데 overlay가 하는 모든 것"이다. upstream 변경의 절반을 빠뜨린 overlay라면 residual에 upstream 줄을 지우는 hunk가 생기고, 검사를 통과하려면 리뷰어가 그 hunk를 보고 LOCAL_FIXES 항목과 연결하는 note를 써야 한다. 기록을 처음 생성한 것 자체가 기존 overlay에 대한 그 리뷰였다. residual의 모든 hunk가 항목 1과 24로 추적되었고, 절반만 적용된 upstream 변경은 발견되지 않았다. 이후로는:

- core 파일을 수정하면 해시가 바뀌므로, drift를 다시 돌려 리뷰할 때까지 `records`(오프라인)가 "changed since the residual was generated"로 실패한다.
- pin bump나 fork 동기화는 헤더의 커밋을 바꾸므로, 새 커밋 기준으로 residual을 다시 만들 때까지 `records`가 실패한다.

합성 테스트는 이 경우를 직접 다룬다. core 파일의 upstream 줄 하나를 변경 이전 형태로 되돌리고, `records`가 해시로 실패하는지, drift가 "core residual differs"로 실패하는지, drift 출력에 빠진 upstream 줄이 새 hunk로 나오는지 확인한다.

### 3.3 백엔드 파일과 overlay가 옮기지 않은 hook

백엔드 파일은 재구성할 pin 사본이 없으므로, drift는 각 파일을 fork 커밋과 비교한다. fork 대비 수정되었거나, 로컬에만 있거나(`hadamard.hip`, `fft.hip`), 빠진 파일은 모두 `LOCAL_FIXES.md`에 이름이 있어야 한다. 또한 overlay가 옮기지 않은 백엔드 밖 MLX 파일에 대한 fork의 변경을 나열하고, LOCAL_FIXES가 그 이유를 적도록 요구한다.

리뷰에서 첫 버전이 너무 느슨하다는 것이 드러났다. 파일 이름만 어디든 나오면 명시로 인정되어, `mlx/device.cpp`가 `mlx/backend/rocm/device.cpp`를, 최상위 `CMakeLists.txt`가 백엔드의 것을 명시한 것으로 처리되었다. 이제 짧은 이름은 독립적으로 나올 때만, 파일 이름만 쓴 경우는 같은 이름의 다른 overlay 파일이 없을 때만 인정된다(`bd5de658`, 이전 도구에서 실패하는 회귀 케이스 포함). PR이 기록을 건드려야 했던 이유도 이 규칙이다. LOCAL_FIXES 항목 2, 7, 11, 20이 파일을 명시하게 되었고, fork와의 차이가 주석 줄바꿈뿐이던 `rope.hip`은 로컬 수정으로 기록하는 대신 되돌렸다.

### 3.4 `sync --check`가 증명하지 않는 것

기록된 fork 커밋에 대한 `sync_from_fork.sh --check`는 `patches-rocm/`를 byte 단위로 재현하며, 이는 이슈의 수락 조건 중 하나였다. 이는 구조상 당연하다. `sync`는 로컬 수정을 이전 fork 커밋 대비 overlay 자신의 차이로 옮기므로, 같은 커밋으로 동기화하면 항등 변환이 된다. 기록되지 않은 로컬 변경은 잡을 수 없다. 절차 문서는 이를 명시하고, 그 역할은 drift가 한다고 적는다.

## 4. `make verify-rocm-overlay` 게이트

```make
verify-rocm-overlay:
	@python3 scripts/mlxcelverse/rocm_overlay.py records
	@bash scripts/mlxcelverse/rocm_overlay_test.sh
```

`records`는 저장소 안의 파일만 읽고, 테스트와 합쳐 약 2초 걸린다. `UPSTREAM`의 형식이 올바르고 빌드의 MLX pin(`scripts/ci/mlx_pinned_commit.sh`)을 가리키는지, README의 백엔드/core 파일 수와 core 파일 목록이 트리와 맞는지, LOCAL_FIXES 항목 1~N이 각각 한 번씩 있는지, `CORE_RESIDUAL.diff`가 현재 커밋과 현재 core 파일 해시로 생성되었고 residual마다 note가 있고 residual 없는 note가 없는지 확인한다. 이 브랜치에서는 백엔드 108개와 core 15개, LOCAL_FIXES 27개 항목, note가 있는 residual core 파일 4개, pin `81ba1c6a`를 보고한다.

이 타깃은 `make verify`와 `make verify-rocm` **양쪽**에 있다. pin bump는 보통 Mac에서 하는데, Mac에서는 `verify-rocm`이 돌지 않는다. `verify`에 게이트가 있으면 `rocm_overlay.py retarget` 없이 `GIT_TAG`만 바꾼 경우 AMD 빌드가 먼저 깨지는 대신 Mac에서 바로 실패한다. 온라인 검사(`drift`, `sync_from_fork.sh`, `check_api_drift.sh`)는 fetch나 빌드가 필요하므로 절차의 수동 단계로 남는다.

## 5. upstream 패키지 11개

`docs/mlxcelverse/upstream/` 아래 각 디렉터리에는 fork head `75915908`에 대한 `git format-patch`(작성자는 메인테이너, AI 표기 없음), `pr.md`(제목, 제출자용 헤더 표, fork PR 템플릿 형태의 본문 초안), 재현 스크립트가 있다. 2026-09-30 기준 fork head는 여전히 `UPSTREAM`이 기록한 `75915908`이고 열린 PR이 없으므로, 이 수정들 중 fork에 들어간 것은 없다.

| 패키지 | LOCAL_FIXES 항목 | `75915908`에 깨끗이 적용 | 상태 |
|---|---|---|---|
| 01 quantized matmul dispatch | 8, 10, 12, 13 | 예 | 수동 제출 준비 완료, 미제출 |
| 02 ArgReduce grid limit | 11 | 예 | 동일 |
| 03 Scatter 크기 인자와 좁은 인덱스 dtype | 14, 17 | 예 | 동일 |
| 04 SliceUpdate tail과 source donation | 21, 24 | 예 | 동일 |
| 05 Hadamard | 16 | 예 | 동일 |
| 06 첫 HIP 큐 이전의 blocking-sync | 20 | 예 | 동일 |
| 07 hipFFT를 통한 FFT | 18, 25 | 05, 06 이후에만 (`CMakeLists.txt`와 `primitives.cpp`에서 05와 인접 줄 충돌) | 05, 06 이후 준비 완료, 미제출 |
| 08 launch geometry helper | 19, 22 | 예 | 수동 제출 준비 완료, 미제출 |
| 09 CPU stream에서의 SDPA fallback | 23 | 예 | 동일 |
| 10 HIP header 의존성 | 26 | 예 | 동일 |
| 11 fine-grained 메모리 위 CPU stream BLAS | 27 | 예, 01~10 위에서도 | 동일 |

패키지 11은 브랜치를 항목 27(#2072, PR #2079)이 머지된 `c5a71cfa` 위로 rebase한 뒤 추가되었다. 색인에는 각 패치가 fork head에서 `git apply --check`를 통과하고(07은 05, 06 이후), 11개 모두 `git am`으로 순서대로 적용된다고 기록되어 있다.

패키지로 만들지 않은 항목과 이유도 색인에 있다. 1~6은 retarget 전용이라 fork가 새 MLX를 머지할 때 upstream으로 간다. 7(GPU fault 보고)은 ml-explore/mlx#3742의 `Event::error()` 위에 만들어졌는데 fork의 MLX base가 그보다 오래되었다. 9는 잘못된 결과를 내는 경로를 끄는 것인데 아직 근본 원인이 없다. 15(`SearchSorted`)는 fork의 base에 없는 primitive(ml-explore/mlx#4035)를 구현한다.

패키지는 fork가 리뷰할 단위로 묶었다(01은 모두 `quantized/qmm.hip`, 03과 04는 `indexing.hip`의 두 부분, 08은 `kernel_utils.hpp`의 grid helper 두 개). 각 패치는 수정을 만든 mlxcel 커밋을 fork 파일에 3-way 머지해 만들었고, retarget hunk와 항목 9는 빼고 mlxcel 고유의 주석은 다시 썼다.

## 6. 기여 상황과 AI 사용 정책

### 6.1 준비는 자동화가, 제출은 손으로

이슈 스레드와 색인 첫머리에 적힌 메인테이너의 지시는, fork 측 수정은 #1813 아래에서 NripeshN/mlx `rocm-support`에 대한 PR로 준비하되 제출은 수동으로 한다는 것이다. 자동화는 NripeshN/mlx나 ml-explore/mlx에 push, fork, 이슈/PR/댓글 작성을 하지 않는다. 준비 과정에서는 읽기 전용 fetch와 API 조회만 사용했다. 색인 끝에 수동 절차가 있다: fork, `git am`, 빌드, 적용 전후 재현 실행, `pre-commit`, PR 생성, 그리고 색인과 해당 LOCAL_FIXES 항목에 링크 기록.

### 6.2 fork

2026-09-30에 읽기 전용으로 확인한 내용:

- fork의 `CONTRIBUTING.md`는 MLX의 예전 일반 문구다(테스트, 성능 변경 시 벤치마크, API 변경 시 문서, 테스트 통과와 리뷰 1회, `pre-commit`). 이슈가 비활성화되어 있어 PR이 유일한 경로다.
- PR 템플릿은 "Proposed changes"와 네 칸짜리 체크리스트이고, AI나 자동화에 대한 언급은 없다. `pr.md` 초안은 이 형태를 따르고 체크리스트는 비워 두었다.
- PR 13개, 기여자 4명, 머지 8개, 마지막 머지는 2026-07-19. 공식 GitHub 리뷰 없이 머지된다. 제목은 `fix(rocm): ...` 형식이고 패치도 이를 따른다. fork는 두 달 넘게 조용해서 응답 시간은 알 수 없다.

### 6.3 ml-explore/mlx의 AI 사용 정책

fork의 브랜치는 ml-explore/mlx#2300의 head이므로, fork에 머지된 것은 결국 ml-explore 리뷰로 간다. 2026-08-20 ml-explore/mlx는 AI 사용 정책(#4331)을 도입했다. PR 템플릿은 "I understand it is strictly prohibited to use AI to write PR description"으로 시작하고 "AI usage disclosure" 칸을 추가했으며, `CONTRIBUTING.md`는 AI를 활용한 작업에도 기여자가 책임을 지고 글은 기여자 자신의 목소리로 쓰라고 요구한다. fork의 브랜치는 이보다 앞선다.

패치, 커밋 메시지, `pr.md` 본문은 AI의 도움을 받아 작성되었다. fork 자체 규칙은 이를 금지하지 않지만, fork 메인테이너가 upstream의 기대를 적용할 수 있다. 그래서 색인은 각 `pr.md`를 **메인테이너가 자기 말로 다시 쓸 노트**로 다루고, disclosure를 요구받으면 AI 도움을 밝히라고 권하며, 결정은 메인테이너에게 맡긴다. 모든 패키지가 제출이 아니라 "수동 제출 준비 완료" 상태인 이유이고, "scale-dispatch 수정이 upstream에 제안되고 링크가 기록됨"이라는 수락 조건이 이 PR 이후에도 열려 있는 이유다.

## 7. 기술적 선택과 그 이유

### fork 커밋이 아니라 fork의 merge base 기준으로 재구성

overlay를 fork의 core 파일 사본과 직접 diff하면 fork의 MLX base와 pin 사이의 모든 upstream 변경이 섞인다. fork의 자기 merge base 대비 delta를 pin 위에 머지하면 세 출처(upstream, fork, 로컬)가 분리되어 residual에는 로컬 부분만 남는다. core 파일에 대한 `sync`도 같은 생각이다: "pin + 이전 fork delta"에서 "pin + 새 fork delta"로 가는 3-way 머지.

### residual 0 규칙 대신 커밋되고 리뷰된 residual

항목 1과 24는 정당한 로컬 차이이므로 "overlay는 재구성 결과와 같아야 한다"는 규칙은 절대 통과하지 못한다. residual을 기록하고 파일마다 note를 요구하면 정당한 차이는 모두 명시되고, 새 차이는 리뷰에서 보이는 diff가 된다. 헤더의 파일 해시 덕분에 오프라인 `records` 검사가 아무것도 fetch하지 않고 core 파일 수정을 잡는다.

### 로컬 수정을 overlay 자신의 차이로 옮기기

`sync`는 백엔드 파일마다 이전 fork 커밋, overlay, 새 fork 커밋을 3-way 머지한다. 따라서 overlay와 이전 fork 커밋의 차이(로컬 수정 전체)가 로컬 전용 파일까지 포함해 자동으로 옮겨진다. 별도 패치 목록을 다시 적용하는 대안은 그 목록을 overlay와 손으로 맞춰야 한다. 대가는 3.4절에서 본 것처럼 `sync --check`가 기록되지 않은 변경에 대해 아무것도 증명하지 못한다는 점이고, 이는 drift의 파일 명시 규칙이 맡는다.

### 게이트에는 오프라인 검사만, 온라인 검사는 수동

`make verify`에는 `records`와 합성 테스트만 들어간다. 네트워크도 ROCm도 필요 없고 어느 호스트에서든 2초 정도면 끝난다. drift는 fetch를 하고, `check_api_drift.sh`는 AMD 호스트와 cold MLX 빌드가 필요하다. 이들을 게이트에 넣으면 `make verify`가 네트워크와 대부분의 기여자에게 없는 하드웨어에 의존하게 된다.

### fetch한 트리를 신뢰하지 않기

git은 `..` 트리 항목을 그대로 저장하고 fetch는 기본적으로 fsck를 하지 않는다. 그래서 악의적인 fork 커밋(또는 `--fork-url`)이 `sync`나 `export-tree`로 하여금 overlay 밖, 예컨대 `mlx/backend/rocm/../../../../.bashrc`에 쓰게 할 수 있었다. 보안 리뷰가 조작한 트리로 이를 재현했다. 이제 트리에서 온 경로에 대한 모든 쓰기와 삭제는 `safe_dest`를 거치고, 트리 목록은 `-z`를 쓰며, 옵션처럼 보이는 fetch 인자는 거부한다(`51b142fc`, `0119aee9`). 테스트는 `git mktree`로 그런 트리를 만들어 확인한다.

## 8. 검증

PR 작성자 (gfx1151):

- 새로 fetch한 캐시로 기록된 fork 커밋에 대해 `sync_from_fork.sh --check`를 실행했고 `patches-rocm/`가 byte 단위로 재현되었다. 이는 항등 변환이므로(3.4절) 작성자는 이전 fork 커밋 `53cbdf8c`(이후 34개 파일 변경)를 가리키는 사본에서 forward sync도 돌렸다. 30개 파일이 깨끗이 머지되었고 4개가 로컬 수정과의 실제 상호작용으로 표시되었다.
- 기록된 커밋에 대한 drift 통과: core 15개 중 11개 byte 단위 일치, residual은 항목 1과 24.
- 각 패키지의 `75915908`에 대한 `git apply --check`(07은 05, 06 이후)와 순차 `git am`. PR 본문 검증 줄은 "all ten"이라고 쓰는데, 패키지 11은 그 뒤에 추가되었고 해당 커밋은 단독으로도, 01~10 위에서도 적용된다고 기록한다.
- `cargo test --features rocm --test dead_doc_pointers` 통과.

오케스트레이터 검증 (gfx1151, `c5a71cfa` 위 `53c8f122`):

- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay` 통과. records 검사는 백엔드 108개와 core 15개, LOCAL_FIXES 27개 항목, note가 있는 residual core 파일 4개, pin `81ba1c6a`를 보고했다.
- `scripts/mlxcelverse/rocm_overlay_test.sh` 29개 검사 통과: 정상 overlay의 records와 drift, `sync --check`, forward sync, fork 충돌, pin보다 앞선 fork, `sync`와 `export-tree` 양쪽의 `..` 트리 항목, pin bump와 retarget, 이미 잘못된 overlay, 기록되지 않은 백엔드 변경(다른 곳의 같은 이름 파일로만 명시된 경우 포함), `.DS_Store`, 번호가 빠진 LOCAL_FIXES, 틀린 README 수, 엉뚱한 기록 파일, 합성 빌드 로그에 대한 `api-report`.
- 이 PR에서는 전체 테스트 스위트를 다시 돌리지 않았다. scripts와 docs 밖의 diff는 `rope.hip`의 주석 변경과 어떤 빌드도 컴파일하지 않는 `docs/` 아래 `.hip` 재현 파일뿐이라, 스위트는 바뀌지 않은 코드만 돌렸을 것이다. 게다가 호스트에서 GPU 측정 유닛 두 개가 돌고 있었다.
- 오케스트레이터는 유닛의 `c5a71cfa` rebase와 항목 27 커밋이 push되지 않은 것을 발견해 `--force-with-lease`로 push했고, PR 본문의 패키지 수를 고쳤다.

## 9. 학습 포인트

- **개수 검사는 이동을 확인할 뿐 정확성을 확인하지 않는다.** bump 전후의 +/- 줄 수 비교는 delta가 보존되었다는 것만 말해 준다. 정확성을 확인하려면 파일이 어떠해야 하는지에 대한 기준이 필요하고, 여기서는 그 기준을 만들 수 있다: pin + fork의 자기 base 대비 delta.
- **리뷰한 예외 목록 자체를 산출물로 만든다.** residual은 작고(파일 4개, LOCAL_FIXES 항목 2개) hunk마다 주석이 있어서, 매 bump마다 사람이 다시 해야 할 리뷰가 한 번 기록되고 무언가 바뀔 때만 다시 요구된다.
- **항등 검사는 항등 검사라고 밝혀야 한다.** 기록된 커밋에 대한 `sync --check`는 구조상 byte 단위로 일치한다. 절차 문서 첫 버전은 이것이 절차 밖에서 바뀐 overlay를 잡을 수 있다고 적었고, 리뷰가 이를 고쳤다. 쓸모 있는 동기화 증거는 이전 커밋에서의 forward sync가 되었다.
- **이름 매칭은 보안에 영향을 주는 휴리스틱이다.** 파일 이름만으로 "기록됨"을 인정하자 가장 자주 수정되는 백엔드 파일들이 조용히 면제되었다. 변경이 문서화되었는지 판단하는 검사도 변경이 허용되는지 판단하는 검사만큼 신중해야 한다.
- **fetch한 git 트리의 경로는 입력이다.** fetch만 하는 도구도 남의 트리에서 나온 경로로 파일을 쓴다. 그 경로는 신뢰하지 않아야 한다.

## 10. 검증하지 않은 것

구현 세션에서 auto 모드 권한 검사가 fork 코드 빌드를 거부했다. 그 결과:

- **`check_api_drift.sh`를 끝까지 실행하지 않았다.** "fork의 merge base + 현재 upstream에 대해 알려진 여섯 개의 깨짐을 나열한다"는 수락 조건은 확인되지 않았다. AMD 호스트에서 실행할 명령은 `scripts/mlxcelverse/check_api_drift.sh --rocm-from fork:75915908dfe5028335d318b10340313744fd3a8d`다. `api-report`는 테스트의 합성 빌드 로그로만 검증되었다.
- **어떤 패키지도 fork에 대고 컴파일하지 않았다.** fork의 MLX base는 mlxcel의 pin보다 오래되었으므로 각 패치가 적어도 거기서 컴파일되는지 확인해야 한다. 패키지 03은 `add_kernel_node`에 컴파일 타임 인자 크기 검사를 추가하는데, mlxcel이 컴파일한 적 없는 fork 호출 지점이 여기서 걸릴 수 있다.
- **fork 빌드에서 재현 스크립트를 실행하지 않았다**(패치 전후 모두). 스크립트는 각 수정을 검증한 mlxcel 테스트를 따른다. 패치 전 일부 케이스는 GPU 큐 fault를 일으키므로, 해당 스크립트는 케이스마다 타임아웃이 있는 자식 프로세스에서 실행한다.
- fork 템플릿이 요구하는 **`pre-commit run --all-files`**를 패치에 대해 돌리지 않았다.

색인은 이 네 가지를 각 PR 제출 전 단계로 적어 둔다. 별도로, Metal과 CUDA는 이 호스트에서 돌리지 못했다. PR은 Metal이나 CUDA 코드를 건드리지 않고, `verify-rocm-overlay`는 순수 Python과 git이다.

## 11. 남은 작업

- 각 제출 전: 패치를 적용한 fork 빌드, 적용 전후 재현 실행, `pre-commit`, `pr.md`를 메인테이너 자신의 말로 다시 쓰기, AI disclosure 결정.
- 각 제출 후: 색인에 링크를 기록하고 해당 LOCAL_FIXES 항목에 "Proposed upstream: <link>"를 추가. scale-dispatch 수정(패키지 01)에 대한 이슈의 수락 조건은 그때 충족된다.
- AMD 호스트에서 `check_api_drift.sh --rocm-from fork:75915908...`를 한 번 실행해 알려진 여섯 개의 깨짐을 확인.
- 항목 7은 fork가 #3742 이후의 MLX를 머지하면, 항목 15는 #4035 이후를 머지하면, 항목 9는 근본 원인이 밝혀지면 패키지로 만든다.
- fork 동기화 후에는 fork가 흡수한 패키지를 지우고 나머지를 새 head 기준으로 다시 만든다.

참고: #1813 (이 PR로 닫힘), #1801, #1802, PR #1818, PR #1819, #1811, PR #2079, #2072, TECHNICAL_REPORTS/1772, ml-explore/mlx#2300, ml-explore/mlx#4331, ml-explore/mlx#3742, ml-explore/mlx#4035.
