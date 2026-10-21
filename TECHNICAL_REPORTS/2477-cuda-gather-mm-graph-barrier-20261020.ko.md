# PR #2477: CUDA sorted gather_mm 배치 스트라이드와 캡처 그래프 배리어

**작성일**: 2026-10-20
**상태**: CUDA (GB10, sm_121)에서 구현 및 검증 완료, PR 열림

## 요약

#2462를 닫는다. GB10에서 조용히 잘못된 값을 내던 CUDA 오버레이 결함 두 개를 수정하고, 각각의 모델 수준 우회책을 제거한다. (1) 정렬된 `gather_mm` 분기(`gather_mm_rhs`)가 크기 1 축의 스트라이드를 grouped-GEMM leading dimension으로 읽어, per-expert bias add 같은 producer가 유닛 스트라이드를 넘겨주면 expert 그룹의 첫 행 이후가 어긋난 activation을 읽는다. (2) 새로 캡처한 큰 CUDA 그래프의 첫 실행이 fresh 프로세스의 약 40%에서 간헐적으로 잘못된 값을 반환한다(GLiNER2.5의 12레이어 DeBERTa-v2 인코더). 업스트림이 스트림 캡처 차일드 그래프를 선언한 버퍼만으로 정렬하기 때문이다. 우회책(`PrivacyFilterMoe.sort_routed`를 양자화 플레인으로 제한, `DebertaV2Encoder`의 레이어별 그래프 분할)을 제거한다.

## 문제 정의

두 결함 모두 PR #2460, #2461에서 우회책과 함께 "별도 이슈가 필요하다"는 표시와 함께 배포됐다. 스트라이드 결함 때문에 Privacy Filter의 dense 플레인(7.5k 토큰 입력)이 정렬되지 않은 gather를 쓰게 되어 8-bit forward의 약 1.5배가 걸렸고, f32 activation을 정렬된 `gather_mm`(`M == 1`)으로 넘기는 어떤 bf16 MoE든 같은 문제를 겪을 수 있었다. 그래프 결함은 레이어별 그래프 분할 뒤에 숨어 있었는데, 이는 인코더를 한 그래프로 캡처해 얻으려던 캡처 비용 절감을 다시 잃는 방법이다. 같은 트리거(스트림 캡처 차일드 그래프 사이의 kernel-kernel 엣지)는 여러 cuBLASLt matmul을 한 그래프로 캡처하는 모든 워크로드를 위협한다.

## 기술적 결정

- **A1: 레이아웃이 맞는 leading dimension.** `gather_mm_rhs`는 이제 `ensure_row_contiguous(a)`를 거치고 `a.strides(-2)`(크기 1 축의 스트라이드, donation 의존) 대신 `lda = a.shape(-1)`(배치 스트라이드)를 쓴다. 버그 바이섹션에서 매 순서마다 정확했던 `GatherQMM` grouped 경로와 업스트림 Metal과 일치한다. 업스트림 베이스 옆에 `[mlxcel #2462]` 표시를 뒀다.
- **A2: 캡처 차일드 그래프를 배리어로.** 새 오버레이 `mlx/backend/cuda/device.cpp`(CUDA 오버레이 23번째 파일, 업스트림 `243bdf09c` 기반): 캡처된 차일드 그래프는 그래프의 열린 분기 전부와 조인하고, 이후 추가되는 모든 노드를 게이트하며, 이전 캡처 차일드 그래프와 서로 체이닝된다. 이는 MLX가 버퍼에서 유도하는 데이터 의존성에 더해진다. `MLXCEL_CUDA_GRAPH_CAPTURE_BARRIER=0`은 A/B를 위해 업스트림의 데이터 전용 정렬을 복원하고, `MLXCEL_CUDA_GRAPHS_DOT_VERBOSE=1`은 그래프마다 verbose DOT 플래그로 덤프해 concurrent 노드가 어떤 버퍼를 만지는지 보여준다.
- **Concurrent 배치가 capture 플래그를 상속.** 노드에 캡처 차일드 그래프가 포함된 `ConcurrentContext` 배치는 조인 노드를 배리어로 만들고 이후 열린 분기 상태를 지운다. concurrent 섹션 아래에서 그룹화된 cuBLASLt matmul이 단일 노드와 같은 방식으로 이후 노드를 게이트한다.
- **환경 플래그 문법.** 두 스위치는 `docs/environment-variables.md`가 문서화한 규약(공백 제거, 대소문자 무시, `0`/`false`/`off`/`no`와 `1`/`true`/`on`/`yes`)으로 파싱한다. 새 헬퍼 `mlxcel_env_flag`를 쓴다. MLX의 `env::get_var`는 `atoi`라 `on`을 0으로 읽어 배리어가 기본으로 꺼졌을 것이다(리뷰 발견).
- **배리어 북키핑은 의도적 누수.** per-encoder 배리어 상태는 의도적으로 누수되는 `thread_local` 맵이다. `~CommandEncoder`는 스레드 종료 중에 `commit`과 `clear_graph_state`를 실행하는데, 파괴자가 있는 `thread_local`은 이미 사라졌을 수 있다. 주석에 그 이유와, 바뀌면 맵을 뮤텍스 아래로 옮겨야 한다는 `new_thread_unsafe_stream` 주의점을 기록했다.
- **우회책 제거.** `PrivacyFilterMoe.sort_routed`는 다시 무조건 true를 반환해 dense 플레인이 정렬된 경로를 쓰고, `DebertaV2Encoder`는 CUDA에서 레이어마다 그래프를 끝내지 않는다(`graph_break_per_layer` 제거). Metal은 그대로다.

## 변경 사항

- `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/matmul.cpp` (A1), `device.cpp` (신규, A2), `README.md` (23개 파일, device.cpp 출처와 pin-bump drift 검사).
- `src/lib/mlxcelverse/CMakeLists.txt` (오버레이 집합에 device.cpp).
- `src/models/openai_privacy_filter_layers.rs` (`sort_routed` 복원), `deberta_v2.rs` (`graph_break_per_layer` 제거), `switch_layers.rs` (테스트 모듈 선언), `openai_privacy_filter_tests.rs`, `deberta_v2_tests.rs`.
- 신규 테스트: `switch_layers_sorted_gather_layout_tests.rs` (3), `switch_layers_sorted_gather_history_tests.rs` (3), `deberta_v2_tests.rs::real_encoder_first_run_equals_replay`, `switch_layers_sorted_gather_test_support.rs` (공용 하네스: 대칭 RNG, `bf16_values`, reference).
- `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.{cpp,h}`, `src/lib/mlxcel-core/src/lib.rs` (`array_strides` 바인딩).
- `docs/environment-variables.md` (두 스위치 행과 문법), `docs/token-classification.md`, `CHANGELOG.md` (routed row를 정렬하는 전체 패밀리 목록).

## 검증

GB10, `--profile test-fast --features cuda`, 테스트 바이너리는 `gpu-lock` 아래 `--test-threads=1`, parity 실행은 `MLX_ENABLE_TF32=0`.

| 검사 | 결과 |
|---|---|
| 정렬 gather 레이아웃 테스트 (유닛 스트라이드 singleton 축, 전치 플레인, biased expert add) | 3/3, 수정 전 오버레이에서는 모두 실패(레이아웃 직접 구성) |
| 할당 히스토리 테스트 (fresh-process reference, 8-bit MoE 선행, 다른 정렬 shape) | 3/3 |
| 정렬 dense 경로의 Privacy Filter 체크포인트 게이트 (#2460에서 drift가 보였던 bf16, 8-bit, bf16 순서 포함) | 10/10 |
| `real_encoder_first_run_equals_replay`, 20 fresh 프로세스, 배리어 on | 20/20 (첫 encode == replay 비트 단위 일치) |
| 같음, 배리어 off + `CUDA_MODULE_LOADING=EAGER` | 13/20 FAILED, lazy 모듈 로딩이 트리거가 아니라 배리어가 수정한다는 뜻 |
| 디코드 A/B, 배리어 on vs off, 같은 바이너리, 팔당 인터리브 4라운드 | qwen3-moe-4bit ON 76.46 vs OFF 75.96 tok/s (+0.7%), phi-3.5-moe-4bit ON 43.21 vs OFF 42.61 (+1.4%); 페어 델타가 부호를 번갈고 ~1.5% 호스트 노이즈 플로어 안쪽이라 배리어는 MoE 디코드에서 측정 가능한 비용이 없음 |
| `5e29de95` 전체 워크스페이스 게이트 | build 0 (161 bin), clippy 0, doctest 0, 계약 검증 0; 161 bin 중 160 통과, `mllama_parity`가 알려진 1-ulp `ragged_multi_image_states_concatenate_real_rows` flake로 1회 실패 후 재실행 통과 (8/8); 이슈 #2467 |

EAGER 결과는 업스트림 리포트용으로 기록한다: 그래프를 프로그램 순서로 체이닝하면 손상이 사라지고, kernel-kernel 엣지만 체이닝하면 사라지지 않으며, EAGER 모듈 로딩은 이를 고치지 못한다.

## 후속 작업

- 위 트리거 데이터를 담은 MLX 업스트림 버그 리포트(선언된 버퍼만으로 정렬된 스트림 캡처 차일드 그래프가 GB10에서 무관한 kernel 노드와 겹친다).
- 오늘 이 분기에 유닛 스트라이드를 넘겨주는 producer는 GPT-OSS, Phixtral, Privacy Filter의 per-expert bias add다. CHANGELOG 항목이 routed row를 정렬하는 모든 패밀리를 나열하므로, 앞으로의 패밀리가 같은 패턴을 확인할 수 있다.
- GLiNER 헤드 자체의 `take` 호출은 3 프로세스 x 4라운드에서 깨끗했다. 배리어 수정과 일치하며 별도로 증명되지는 않았다.

## 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 18 |
| 추가 라인 | 1519 |
| 삭제 라인 | 73 |

| 커밋 | 메시지 |
|--------|---------|
| `f5e14115` | fix(cuda): take the batch stride in sorted gather_mm, make captures barriers |
| `5e29de95` | fix(cuda): parse barrier flags per docs, extend barriers to batches |

## 관련 항목

- 이슈 #2462; 우회 PR #2460 (sort_routed), #2461 (레이어별 그래프 분할).
- #2174(그래프 off에서도 지속되는 Llama fresh-process divergence)는 관련이 있지만 별개다.
- 게이트에서 본 1-ulp mllama parity flake는 이슈 #2467로, 무관하다.
