# PR #2933: 배치 엔진의 네이티브 4비트 GEMV

**작성일**: 2026-11-10
**상태**: 완료 (PR은 draft로 열려 있고 #2922를 닫음)

## 요약

ragged 연속 배칭 엔진이 이제 체크포인트의 packed 4비트 투영 가중치를 네이티브 HIP GEMV로 실행한다. 단일 시퀀스 디코드를 스텝당 24.6 ms에서 7.1 ms로 줄인 것과 같은 메커니즘이다. 기존 HSACO에 여섯 개의 export를 추가했다: `qmv4_multi_kernel`, `qmv4_multi_row_kernel`, 그리고 q/k/v와 gate/up을 융합한 multi-row 변형이다. 하나의 workgroup이 하나의 출력 행을 맡고, 각 weight word를 한 번만 dequantize해 활성화 행마다 하나씩 있는 accumulator에 누적하므로, 가중치 스트림을 요청마다 다시 읽지 않고 행마다 한 번만 읽는다. `gfx1151`의 llama-3.2-1b-instruct-4bit에서 B=8 ragged 디코드는 f16 경로의 59.8 tok/s에 대해 74.5와 76.7 tok/s를 기록했고, B=4는 65.6에 대해 60.0과 62.2 tok/s로 고정 8-accumulator 구조 때문에 약 5퍼센트 후퇴했으며 이는 후속 작업으로 기록했다. 모든 실행이 REFERENCE-EXACT PASS를 출력하고, 스텝당 GEMV launch 수는 B와 무관하게 113에서 65로 줄었으며, untied인 Qwen2.5-7B 체크포인트도 reference-exact다. 9개 파일, +982/-51.

## 1. 문제 정의

`XlaBatchEngine`은 여전히 f16으로 확장한 투영 가중치를 스트리밍하고 있었다. 단일 시퀀스 그래프는 #2890과 #2892 이후 packed 4비트 가중치를 사용했지만, 배치 그래프는 `ProjectionLayout::OutIn`과 f16 가중치로 emit되고(`src/lib/mlxcel-xla/src/iree.rs`, `src/lib/mlxcel-xla/src/emitter/model.rs`), `linear_seq_proj`는 해당 그래프에 packed operand가 등록되지 않아 f16 `dot_general`로 빠졌다. f16 확장 대신 체크포인트의 4비트 가중치를 읽으면 스텝당 스트리밍 바이트가 약 4분의 1로 줄어든다. 단일 디코드를 스텝당 24.6 ms에서 7.1 ms로 옮긴 레버이자 #2886이 지적한 배치와 단일 경로의 격차다.

커널 문제는 단일 디코드와 다르다. `qmv4_kernel`은 warp 하나를 출력 행 하나에 대응시키고, 기존 두 커널 모두 활성화 행 하나 `x[k]`를 읽는다. 배치 스텝에는 같은 가중치 행들을 필요로 하는 B개의 활성화 행이 있다. 요청마다 단일 행 커널을 호출하면 launch가 B배가 되고 가중치 스트림도 B번 읽혀 바이트 절감이 대부분 사라진다. 배치 경로에는 weight word를 한 번 읽어 모든 활성화 행에 적용하는 launch 하나가 필요하다.

## 2. 변경 내용

### 2.1 커널 (`src/lib/mlxcel-xla/csrc/qmv4.hip`)

- `qmv4_multi_kernel`은 warp-per-row 배치를 유지하며 상수 `K`, `N`, `B`와 ABI `(x[B,K], w[N,K/8], scales[N,K/64], biases[N,K/64], y[B,N])`를 갖는다. 각 lane은 weight word를 한 번 load하고 한 번 dequantize해 `B`개 accumulator에 곱한다. warp-shuffle reduction을 accumulator마다 반복하고 `y[b*N + row]`에 쓴다.
- `qmv4_multi_row_kernel`은 `K >= ROW_KERNEL_MIN_K`에서 쓰는 block-per-row 대응 커널로, 모든 B accumulator를 하나의 shared-memory 패스와 한 번의 barrier로 처리한다.
- 융합 `qkv`와 `gateup` multi-row 변형은 #2907의 cumulative-N ABI를 재사용한다. 행이 누적 출력 폭으로 자신의 set을 고르므로 q, k, v(그리고 gate, up)가 투영별 행별 launch가 아니라 그룹당 launch 하나를 유지한다.
- 행별 누적 순서는 단일 행 커널과 동일하다. 이것이 결과를 f16 경로와 수치적으로 같게 유지해 벤치마크가 근사치가 아니라 REFERENCE-EXACT를 출력하게 한다.

### 2.2 Emitter와 로더

- `emitter/builder.rs`에 rank-2 `[B, K]` 활성화를 1~3개 operand에 대해 처리하는 `native_qmv_rows`를 추가했다. rank-2 `linear_seq_proj`가 이를 통하고 `gate_up_proj`가 rank-2 또는 rank-1 형태를 고른다. `native_qmv`의 rank-1 assert는 완화하지 않고 유지했다.
- `emitter/model.rs`는 `OutIn` 레이아웃에 packed operand를 등록하고 q/k/v와 gate/up을 융합하며 레이아웃별 logical shape를 갖는다. `emit_decode_ragged_with_native`가 executable source를 ragged 모듈에 이어 붙이고, native 모듈을 넘기지 않으면 f16 그래프는 이전과 바이트 단위로 동일하다.
- `iree.rs`는 `load_inner`에서 native 모드를 `runtime_ffi_dimensions`와 가중치 업로드 이전에 한 번 결정한다. packed 경로와 f16 경로에서 인자 목록과 상주 가중치 집합이 다르기 때문이다. ragged gate는 dense 4비트 group-64 체크포인트와 packed 지원 레이아웃을 요구하고, 그 외에는 경고 한 번과 함께 f16에 머문다.
- `native_qmv.rs`는 ragged 그래프에 multi-row export를, 단일 시퀀스 그래프에 기존 단일 행 그룹을 선택한다. `module_parts`와 `ragged_module_parts`를 분리해 각 그래프가 자신이 dispatch하는 export만 embed한다.

### 2.3 폴백

`HIPCC`가 없거나 잘못된 컴파일러를 가리킬 때, 아키텍처가 지원되지 않을 때, 체크포인트가 4비트 group-64가 아닐 때 엔진은 이전 f16 그래프를 emit하고 로드당 경고 한 번을 남긴다. `HIPCC=/nonexistent/hipcc`와 빈 아티팩트 캐시로 검증했다: 경고 한 번, f16 경로, REFERENCE-EXACT PASS, 66.1 tok/s.

## 3. 측정

모든 실행은 `gfx1151`에서 `scripts/rocm_gpu_guard.sh --hold` 아래, 별도 표기가 없으면 llama-3.2-1b-instruct-4bit, A/B는 같은 바이너리로 수행했다.

| 설정 | 네이티브 4비트 | f16 경로 | 결과 |
|------|----------------|----------|------|
| B=8, 1회차 | 74.5 tok/s | 59.8 tok/s | REFERENCE-EXACT PASS |
| B=8, 2회차 | 76.7 tok/s | 59.8 tok/s | REFERENCE-EXACT PASS |
| B=4, 1회차 | 60.0 tok/s | 65.6 tok/s | REFERENCE-EXACT PASS |
| B=4, 2회차 | 62.2 tok/s | 65.6 tok/s | REFERENCE-EXACT PASS |
| B=4, untied Qwen2.5-7B-Instruct-4bit | 14.2 tok/s | 비교 안 함 | REFERENCE-EXACT PASS |

- ragged 디코드 스텝당 GEMV launch: 113에서 65로, B와 무관.
- 이슈의 B=8 기준선은 69~71 tok/s였고 이번 작업은 74.5~76.7을 측정했다.
- B=4 네이티브는 두 회차 모두 f16보다 약 5퍼센트 낮다. 커널이 B와 무관하게 8개 accumulator를 전제하므로 B=4 launch는 필요한 두 배의 accumulator 연산을 한다. 이슈의 승인 기준은 B=8이고 모든 실행이 reference-exact이므로, 이 사실을 숨기지 않고 후속 작업으로 남겼다.

## 4. 기술적 선택과 그 이유

### 4.1 B별 특화 대신 고정 8 accumulator

**컨텍스트:** ABI가 지원 B 값(4와 8)을 하나의 export 그룹으로 처리해야 한다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| B별 특화 (`MAX_B` 변형 또는 컴파일 타임 B 전환) | 불필요한 accumulator 연산 제거, B=4 후퇴 회복 | export, pipeline layout, 동기화할 코드가 늘어남 |
| **선택: 런타임 B와 고정 8 accumulator** | 모든 지원 B에 하나의 export 그룹, 단순한 ABI와 테스트 | B=4에서 약 5퍼센트 후퇴 |

**선택 이유:** 이슈의 목표는 multi-row 형태가 이기는 B=8이고, `y[b*N + row]`는 `b < B`에 대해서만 인덱싱되므로 잘못된 행이 잘못된 결과를 만들 수 없다. 특화는 그래프를 바꾸지 않는 국소적 후속 작업이다.

### 4.2 그래프별 export embed

단일 시퀀스 그래프와 ragged 그래프는 서로 다른 export 집합을 dispatch한다. HSACO 전체를 두 모듈에 embed하면 모든 vmfb가 쓰지 않는 export를 나른다. `module_parts`와 `ragged_module_parts`가 각 그래프에 필요한 그룹만 emit하며, 대가는 하나의 아티팩트에 대한 두 source builder다. 보안 리뷰는 공유 아티팩트 캐시 문제를 별도로 제기했다(후속 조치 참고).

### 4.3 cumulative-N 융합 ABI 재사용

배치 엔진은 그렇지 않으면 q/k/v와 gate/up launch 수를 B배로 늘린다. multi-output 형태를 같은 커널에 접어 넣으면 ragged launch 수가 어떤 B에서도 65로 유지되고, #2907의 단일 시퀀스 융합 계약과 어긋날 두 번째 계약도 생기지 않는다.

## 5. 검증과 한계

- 최종 tip에서 `cargo test -p mlxcel-xla --lib`: 399 passed, 0 failed, 19 ignored.
- 표적 emitter 테스트 `native_ragged_decode_emits_multi_row_dispatches`와 B=4 launch 구조 테스트가 새 dispatch 형태와 폴백을 덮는다.
- 최종 tip에서 `cargo fmt --all -- --check`, `cargo check -p mlxcel-xla --features iree --lib --tests`, `cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings`가 깨끗하다. `iree` feature는 루트 크레이트에서 빌드해야 하며, 크레이트 단독으로는 링크할 수 없다.
- `hipcc --genco -O3 --offload-arch=gfx1151 csrc/qmv4.hip`가 컴파일되고 결과 HSACO에 여섯 개 multi-row 커널을 포함한 열두 개 심볼이 들어 있다.
- 검증하지 않은 것: native 모드에서 B>1인 OpenXLA serve worker의 end-to-end 경로. 벤치 하네스가 같은 엔진을 쓰지만 워커의 admission 경로는 이번 실행에서 거치지 않았다.
- `mistral-7b-instruct-v0.3-4bit`는 두 번째 체크포인트 확인에 쓸 수 없었다. OpenXLA emitter가 가중치 처리 이전에 `model_type = "mistral"`을 거부하기 때문이다. untied 확인에는 Qwen2.5-7B-Instruct-4bit를 사용했다.

## 6. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 9 |
| 추가 라인 | +982 |
| 삭제 라인 | -51 |
| 추가 테스트 | 표적 emitter/커널 테스트 2개 |

| 카테고리 | 요약 |
|----------|------|
| 성능 | ragged 엔진용 multi-row packed GEMV와 융합 multi-row q/k/v, gate/up |
| 정확성 | B=4/B=8, tied/untied 체크포인트에서 reference-exact, 폴백 동작 불변 |
| 문서 | README, 환경 변수 문서, CHANGELOG |

## 7. 후속 조치

1. B=4 후퇴(두 회차 모두 약 5퍼센트)를 회복할 B별 accumulator 특화.
2. 단일 시퀀스 native gate는 지원되지 않는 packed 레이아웃에 경고하지 않는다. ragged gate는 이제 경고한다.
3. 공유 `<temp>/mlxcel-xla-qmv4` HSACO 캐시가 world-accessible이고 키에 hipcc 버전이 없다. 선행 문제이며 저장소 전반의 패턴으로, 이 PR에서 바꾸지 않았다.
