# 기술 보고서: PR #2059 - ROCm 정확도 매트릭스 행 추가, paged-attention 포트 술어, strided scan 그리드 수정

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: C++ (ROCm overlay 헤더, paged-attention 런처, cxx 브리지), Rust (프로덕션 게이트, 테스트), Python (CI 검사기), Markdown, TSV 트레이스

**위험도**: 낮음에서 중간. overlay 변경은 ROCm의 모든 strided scan 실행 그리드를 바꿉니다. 정확도 수정이지만 자주 쓰이는 기본 연산을 건드립니다. 프로덕션 게이트 변경은 구조상 현재 모든 백엔드에서 같은 답을 냅니다. Metal과 CUDA에서는 실행하지 않았습니다.

## 요약

이슈 #1809는 두 가지를 요구합니다. Metal 기준 대비 ROCm 정확도 매트릭스, 그리고 통과하는 `make verify-test-rocm` 게이트입니다. 첫 실행(#1826)은 체크포인트 네 개를 다뤘고, 게이트는 `origin/main` `4595b06f`에서 37개 테스트가 실패했습니다.

이 PR은 게이트 실패를 다른 작업에 속한 세 개만 남깁니다. paged-attention 테스트 34개는 해당 커널에 ROCm 포트가 없어서(#1814) 실패했을 뿐입니다. 이제 이 테스트들은 한 곳에서 건너뜁니다. 각 커널 자신의 `KernelPorts` 테이블을 읽는 술어를 쓰고, ROCm에서만 건너뛰므로 포트가 들어오는 순간 다시 실행됩니다. 이 커널 앞의 프로덕션 경로들도 같은 술어를 묻습니다.

또 매트릭스에 빠져 있던 다섯 행을 ROCm 쪽에서 추가합니다. 두 번째 dense 모델, sliding-window 모델, SSM 하이브리드 두 개, VLM 하나입니다. Metal 호스트에 접근할 수 없어 Metal 쪽은 대기 상태이며, 실행할 정확한 명령을 커밋했습니다. SSM 하이브리드를 실행하다가 실제 ROCm 결함을 찾았습니다. strided scan의 실행 그리드를 오래된 로컬 사본 `get_2d_grid_dims`로 계산해서, 필요한 블록의 최대 16배를 띄우고 배열 끝을 넘어 쓰다가 GPU fault를 냈습니다. 이를 고쳤고, 수정 없이는 fault가 나는 테스트로 덮었습니다.

끝으로 #2029가 남긴 질문을 정리합니다. Nemotron-H의 `use_fused` 가드는 opt-in 경로 `MLXCEL_FUSED_MOE_RELU2`에서는 실제 수정이고(가드 이전 빌드는 abort), 기본 경로에서는 방어적입니다(가드 이전 빌드도 생성함).

## 1. 문제 정의

### 게이트가 빨간 이유는 잘못된 숫자가 아니라 포트 부재였습니다

`make verify-test-rocm`은 37개 테스트에서 실패했습니다. 그중 34개는 fused paged-attention 커널(v1 decode, v2 partial, merge)을 다룹니다. 이 커널들은 Metal과 CUDA 포트가 있고 `.rocm = nullptr`입니다. 테스트는 커널을 실행해 런처의 거부를 받거나, 포트 부재로 먼저 거절하는 프로덕션 경로를 거쳐 테스트가 확인하려던 결정에 닿지 못했습니다. 수치 결함은 하나도 없었습니다. 실패가 전부 알려진 것인 빨간 게이트에서는 새 실패를 놓치기 쉽습니다.

건너뛸 근거가 될 포트 술어도 없었습니다. `paged_attention.cpp`에는 `KernelPorts` 테이블이 있지만 `*_available()`을 내보내지 않았고, 그래서 `ffi_tests.rs`가 dispatch 검사기의 `BACKEND_ENUMERATION_TODO`에 남아 있었습니다(#1814 계획 5항). 프로덕션 게이트는 정의상 Metal 또는 CUDA만 뜻하는 `custom_kernels_available()`을 물었기 때문에, 이후 HIP 포트가 들어와도 그 뒤에서 도달할 수 없었을 것입니다.

### 다섯 매트릭스 행은 한 번도 실행되지 않았습니다

sliding-window, SSM 하이브리드, VLM 행과 `Qwen2.5-7B-Instruct-4bit`는 #1803과 #1805에 막혀 있었고, 둘 다 이미 닫혔습니다. 호스트에는 해당 체크포인트가 없었습니다.

### Nemotron-H 가드는 실행이 아니라 추적으로만 확인됐습니다

PR #2029는 ROCm에서 Nemotron-H 체크포인트를 돌려보지 않은 채 Nemotron-H의 `use_fused`에 `&& custom_kernels_available()`을 더했고, 이 항을 방어적이라고 기록했습니다.

## 2. 변경 요약

- **포트 술어.** `src/lib/mlx-cpp/turbo/`의 `paged_attention_decode_available`, `paged_attention_v2_partial_available`, `paged_attention_merge_states_available`는 각각 자기 테이블에 대한 `has_kernel_port`를 반환합니다. 브리지는 `paged_attention_kernels_available`(세 커널 모두), `paged_attention_decode_available`, `paged_attention_v2_available`(partial과 merge), `paged_attention_merge_available`을 노출합니다.
- **프로덕션 게이트.** `PagedBlockPool`의 batched paged decode(`cache/paged.rs`)는 전체 커널 술어를, MLA split-KV(`mla/mod.rs`)는 merge 술어를, `paged_decode_backend`(`layers.rs`)는 v1 술어를, `run_sparse_decode`(`paged_v2/sparse.rs`)는 shape와 sparsity 거절 이후에 v2 술어를 묻습니다. sparse 경로에는 이전에 포트 검사가 없었습니다. ROCm에서는 매 스텝 매 레이어마다 입력을 만들고, 실행을 거부당하고, plan 거절로 보고했습니다.
- **테스트 건너뛰기.** `src/lib/mlxcel-core/src/test_support/kernel_ports.rs`에 `require_paged_attention_port!`, `require_paged_decode_port!`, `require_paged_v2_port!`, `require_paged_merge_port!`가 있습니다. 각 매크로는 백엔드가 ROCm이고 그 테스트가 필요로 하는 커널의 술어가 false일 때만 일찍 반환하며, `skipping <module>:<line>: ROCm has no <kernels> kernel port yet (lablup/mlxcel#1814) ...`를 프로세스 stderr에 직접 출력합니다. 실패하던 테스트 34개와, 예전에 Metal과 CUDA 밖에서 조용히 반환하던 `ffi_tests` 테스트 두 개가 이 매크로를 씁니다.
- **검사기.** `!metal_is_available() && !cuda_is_available()`로 쓰인 `ffi_tests` 게이트 두 개는 매크로와 GPU 백엔드가 없을 때의 반환으로 바뀌었고, `ffi_tests.rs`는 `BACKEND_ENUMERATION_TODO`에서 빠졌습니다.
- **scan 그리드 수정.** `patches-rocm/mlx/backend/rocm/kernel_utils.hpp`의 `get_2d_grid_dims(shape, strides, divisor)`는 CUDA 백엔드처럼 `mlx/backend/common/utils.cpp`의 `get_2d_grid_dims_common`에 위임합니다. `LOCAL_FIXES.md` 22항, `tests/rocm_strided_scan.rs`.
- **트레이스.** `benchmarks/logit_traces/rocm_gfx1151_c5fe9a16/`: 트레이스 16개(체크포인트 다섯 개의 w1, w8, w256, 그리고 `gemma-3-4b-it`의 `w8ctx1536`), METADATA, RUNS, SHA256SUMS, Metal 명령을 담은 README.
- **문서.** `docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md`, 2026-09-12 실행 문서와 `docs/installation.md`의 링크, `verify-rocm` 순서에 대한 Makefile 주석.

## 3. 기술적 결정

### 테이블 기준으로, ROCm에서만, 드러나게 건너뜁니다

사용자가 건 조건은 건너뛰기가 #1814를 명시하고, 눈에 보이는 한 곳에 있고, 포트가 들어오면 다시 나타나야 한다는 것이었습니다. 포트 테이블을 읽으면 세 번째 조건이 바로 충족됩니다. 술어와 dispatch가 같은 `port_for`를 쓰므로 `.rocm` 항목이 채워지면 술어가 true가 됩니다. ROCm으로 한정했기 때문에 Metal이나 CUDA에서 술어가 잘못 false를 내면 테스트가 통과하는 대신 실패합니다. `eprintln!` 대신 `std::io::stderr()`로 출력하는 것도 중요합니다. libtest는 통과한 테스트의 `eprintln!`을 가로채 버리기 때문에, 헬퍼의 첫 버전은 게이트 로그에 아무것도 남기지 않았습니다.

### 술어 하나가 아니라 커널별 술어

첫 버전은 모든 테스트에 전체 커널 술어 하나를 썼습니다. 리뷰에서 #1814가 커널을 하나씩 포팅할 수 있고, 그러면 v2만 쓰는 테스트가 틀린 메시지를 단 채 계속 건너뛰게 된다는 지적이 나왔습니다. 이제 각 테스트는 자신이 실행하는 커널을 지정합니다(v1, v2, merge, 또는 어느 경로든 탈 수 있는 batched decode의 경우 전체).

### scan 그리드는 이슈로 미루지 않고 고쳤습니다

이 수정 없이는 SSM 하이브리드 행이 트레이스를 만들 수 없었고, 원인은 upstream과의 작고 국소적인 차이였습니다. overlay가 한 오버로드에는 자체 구현을 들고 있었고, 옆의 다른 오버로드는 이미 공용 헬퍼에 위임하고 있었습니다. 위임하면 upstream 동작이 그대로 돌아옵니다. 공용 헬퍼는 각 차원과 남은 divisor의 gcd를 취합니다. 연속 입력이면 차원들의 곱이 `axis_size * stride`의 배수이므로 gcd 과정은 항상 divisor 1로 끝나고, 그리드는 정확히 `size / divisor`가 됩니다.

### scan 결함은 fault로 잡습니다

그리드가 커도 범위 안 출력은 올바릅니다. 여분 블록은 배열 끝 너머 메모리만 건드리기 때문입니다. 값 비교로는 보이지 않습니다. 그래서 테스트에는 옛 그리드가 25 MB 배열에서 16배를 넘치는 shape를 넣었습니다. 이 경우 매핑된 메모리를 벗어나 fault가 나며, gfx1151에서 헤더를 되돌리면 실제로 그렇습니다. 값 비교는 scan 자체를 확인하는 용도로 남겼고, f32로 저장한 작은 정수를 써서 GPU와 CPU가 비트 단위로 일치해야 합니다. 첫 버전의 입력 생성기가 0만 만든다는 점을 리뷰가 잡았고, 머지 전에 고쳤습니다.

### sliding-window shape 추가

표준 폭에서는 `gemma-3-4b-it`가 컨텍스트를 최대 520 토큰만 받으므로 1024 토큰 윈도 안에 머뭅니다. `MLXCEL_TRACE_START_TOKEN=1536`과 함께 쓰는 `w8ctx1536` shape는 평가하는 모든 chunk에 1536 토큰 컨텍스트를 주므로, sliding-window 마스크와 회전 캐시가 윈도 너머까지 실행됩니다.

### Metal 절반을 지어내지 않았습니다

Metal 호스트에 접근할 수 없었고, 이 체크포인트들의 Metal 트레이스는 저장소에 없습니다. ROCm 트레이스는 리비전, 커밋, 바이너리 해시와 함께 커밋했고, README에 정확한 명령과, Metal의 커널과 ROCm의 그래프를 비교하게 되는 쌍(하이브리드의 `w1`, Nemotron-H 전체)을 적었습니다.

## 4. 검증

모두 gfx1151(Radeon 8060S, ROCm 10.0.0 패키지, HIP 7.15.26333)에서 수행했습니다.

- paged-attention 테스트: 커널별 세분화 이후 `cargo test -p mlxcel-core --profile test-fast --features rocm --lib -- --test-threads=1 paged_v2 mla:: cache::paged ffi_tests::test_fused_paged` 결과 266개 통과, 0개 실패, 건너뛰기 줄 36개(`4595b06f`에서 실패하던 34개와, 예전에 조용히 반환하던 `ffi_tests` 두 개). mlxcel-core clippy `--lib --tests -D warnings`, 빠른 스크립트 게이트, `dead_doc_pointers`도 통과합니다.
- `tests/rocm_strided_scan.rs`: 수정 후 3개 통과. 옛 헤더에서는 `[64, 96, 32, 32]` 경우가 `strided_scan`에서 fault를 냅니다.
- 트레이스: 16개 모두 exit 0, NaN 없음(`RUNS.txt`). 수정 전에는 granite w8과 Nemotron-H w8, w256이 `strided_scan`에서 fault를 냈습니다.
- 생성: 체크포인트 다섯 개 모두 `-t 0`으로 일관된 텍스트를 생성하고, VLM 두 개는 `tests/fixtures/test_image_shapes.png`를 올바르게 설명합니다.
- Nemotron-H 가드: `680eb064`에서 기본 경로는 생성하고 `MLXCEL_FUSED_MOE_RELU2=1`은 `[metal_kernel] No Metal back-end.`로 abort합니다(exit 134). `4595b06f`에서는 둘 다 생성합니다.
- 리뷰 후속 변경 이전 브랜치(`bae40d24`)에서 `make verify-rocm`: versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy, ROCm smoke 통과. `verify-test-rocm`은 11736개 통과, 3개 실패, 377개 ignored였습니다. 세 개는 다른 작업에 속한 알려진 실패입니다. rebase한 브랜치의 최종 게이트 결과는 PR 본문에 기록합니다.

## 5. 배운 점

- **매트릭스 행은 크래시 테스트이기도 합니다.** SSM 하이브리드 행은 Metal 쪽이 대기 중이라 백엔드 간 수치를 내지 못했지만, 이 PR에서 가장 중요한 결함을 찾았습니다. 새 모델 계열을 한 백엔드에서 끝까지 돌리면, 어떤 단위 테스트도 그 shape로 조합해 본 적 없는 기본 연산이 실행됩니다.
- **upstream 헬퍼의 로컬 사본은 조용히 어긋납니다.** 한 오버로드는 위임하도록 바뀌었고 다른 하나는 그대로였습니다. 어떤 shape가 걸리는지는 헤드 수와 chunk 길이가 공유하는 인수에 달려 있어서, 짧은 프롬프트와 1토큰 폭은 동작했고 문제를 가렸습니다.
- **건너뛰기는 흔적이 보여야 합니다.** 통과한 테스트의 `eprintln!`은 libtest 캡처에 묻힙니다. 게이트 로그에서 보이지 않는 건너뛰기는 통과로 읽힙니다.
- **테스트의 단언만이 아니라 입력도 확인하십시오.** 첫 scan 테스트는 GPU와 CPU를 정확히 비교했지만 입력이 전부 0이었습니다. 수정 후 통과했고 수정 없이 fault를 냈으므로, 되돌리기 확인만으로는 값 비교가 아무것도 검사하지 않는다는 사실이 드러나지 않았습니다.
- **주장은 실행해서 확인하십시오.** Nemotron-H 가드는 추적만으로 방어적이라고 기록됐습니다. 두 빌드를 실제로 돌려보니, 추적이 따라가지 않은 경로에서 실제 수정이었습니다.

## 6. 주의 사항과 검증하지 않은 것

- **Metal과 CUDA는 실행하지 않았습니다.** 새 술어는 구조상 두 백엔드에서 true이므로(같은 테이블, 같은 `port_for`) 게이트와 건너뛰기가 바뀌지 않지만, 실행으로 확인하지는 않았습니다. overlay 수정은 Metal과 CUDA 빌드가 복사하지 않는 `patches-rocm/`에 있습니다.
- **다섯 행의 Metal 쪽**은 대기 중이며, 이 행들에 대한 백엔드 간 수치는 없습니다.
- **VLM 트레이스는 텍스트만 다룹니다.** 비전 타워는 이미지 생성 확인 두 번으로만 덮었습니다.
- **gfx1151만** 실행했고, strided scan 테스트의 되돌리기 확인은 이 호스트 할당기 배치에서 초과 접근이 내는 fault에 의존합니다.
- **트레이스는 다른 작업 단위의 프로세스와 GPU를 공유**하며 실행했습니다. 정확도 트레이스이며 시간 측정이 아닙니다.

## 7. 남은 작업

- #1809는 열린 채로 둡니다: 다섯 행의 Metal 쪽, 그리고 남은 게이트 실패 세 개(bf16 `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, #2037의 `gelu_approx_matches_mlx_nn_bit_for_bit`와 `family_order_is_exhaustive`).
- #1814: 건너뛰기 36개를 끝낼 paged-attention 포트와 나머지 포트 테이블.
- #1813: `LOCAL_FIXES.md` 22항을 fork에 제안.

참고: #1809, #1801, #1814, #1813, #1826, #2029, #2037, #1785.
