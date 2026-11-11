# PR #2957: C 진입점에서 prefill 위치 상한 검사

**작성일**: 2026-11-11
**상태**: 완료 (#2954 종료)

## 요약

PR #2936은 decode 좌표를 Rust 래퍼와 C decode 진입점에서 제한했지만, 네 개의 prefill C 진입점은 호출자가 준 위치를 검사 없이 그래프로 넘기고 있었다. 방출된 prefill 그래프는 그 값으로 `[context_capacity, rot]` RoPE 테이블을 인덱싱하고 아무것도 clamp하지 않으므로, `[0, context_capacity)` 밖의 위치는 그래프가 보고할 수 있는 오류가 아니라 device out-of-bounds read다. 이 PR은 공유 `xla_check_prefill_positions` 헬퍼를 추가해 `xla_llama_prefill`, `xla_llama_prefill_slot_logits`, `xla_llama_prefill_diagnostics_slot`, 공유 embeddings prefill 구현에서 IREE 호출 전에 검사하고, 누락된 NULL 검사를 보강하며, C 진입점을 직접 구동하는 실제 device 테스트로 경계를 증명한다. 4개 파일, +283/-2.

## 1. 문제 정의

decode는 #2936에서 고쳐졌다. `mrope_decode_coordinate`와 `mrope_slot_coordinates`가 typed `MropeCoordinateError::OutOfRange`를 반환하고, C decode 진입점이 상한을 반복해 직접 C 호출자가 Rust 가드를 우회할 수 없다. prefill은 별도 진입점 집합이고 그 PR의 리뷰가 decode로 범위를 정했기 때문에 남았다.

prefill 그래프는 버퍼의 모든 위치를 읽는다. 처음 `real_len`만 읽는 것이 아니라 패딩 꼬리도 같은 RoPE gather에 도달하기 때문이다. `xla_llama_prefill_slot_logits`는 바로 그 이유로 모든 토큰 id를 이미 검사하지만, 어떤 prefill 진입점도 위치 값을 검사하지 않았다. `xla_llama_prefill`은 추가로 `c`, `tokens`, `positions`, `out_token`을 NULL 검사 없이 역참조했고, `xla_llama_prefill_diagnostics_slot`은 context와 출력은 검사했지만 `tokens`와 `positions`는 검사하지 않았다.

위험은 검증되지 않은 정수로 인한 RoPE 테이블 밖 device read다. Rust API는 보통 정규 위치를 합성하므로 이 결함은 C 호출자, 또는 명시적 위치로 C 경계에 도달하는 Rust 호출자에게만 열려 있고, 가드가 보호해야 하는 대상이 바로 그 호출자다.

## 2. 변경 사항

### 2.1 공유 위치 가드 (`src/lib/mlxcel-xla/csrc/xla_iree.c`)

`xla_check_prefill_positions(caller, positions, count, axes, context_capacity)`는 axis-major M-RoPE 레이아웃(`axis * count + index`)의 모든 axis와 index를 순회하고, `[0, context_capacity)` 밖의 첫 값에서 호출자, axis, index, 값, capacity를 출력하며 1을 반환한다. `real_len < count`여도 모든 `count` 항목을 검사하므로 패딩 꼬리가 나쁜 값으로 그래프에 도달할 수 없다.

헬퍼는 네 곳에서 IREE 호출 전에 호출된다.

- `xla_llama_prefill`: `lp` 위치, axis 1 또는 3.
- `xla_llama_prefill_slot_logits`: `lp` 위치, axis 1 또는 3.
- `xla_llama_prefill_diagnostics_slot`: `lp` 위치, axis 1 또는 3.
- `xla_llama_prefill_embeddings_impl`: 아홉 개 `xla_llama_prefill_embeddings*` 래퍼 뒤의 공유 구현으로, descriptor의 `context_capacity` 개수와 명시적 `position_mode`를 사용한다.

### 2.2 NULL 검사

`xla_llama_prefill`은 context를 만지기 전에 `c`, `tokens`, `positions`, `out_token`을 검증한다. `xla_llama_prefill_diagnostics_slot`은 기존 계약 검사에 `tokens`와 `positions`를 추가한다.

### 2.3 test-support 전달자와 device 테스트 하나

공개 Rust API는 정규 위치를 합성하고 `prepare_for_mode`가 이미 범위 밖 명시 위치를 거부하므로, 일반 래퍼로는 device 테스트가 C 경계에 도달할 수 없다. `MLXCEL_XLA_TEST_SUPPORT` 전달자 `xla_test_prefill`과 `xla_test_prefill_slot_logits`가 호출자 위치로 실제 진입점을 호출하고, Rust에는 `IreeLlama::test_prefill_with_positions`와 `IreeRaggedLlama::test_prefill_slot_logits_with_positions`로 노출된다.

`tests/xla_ragged_hardening.rs`의 새 `#[ignore]` 테스트 `prefill_c_entry_points_reject_out_of_range_positions`는 같은 토큰에 범위 밖 값 하나를 넣어 실패를 기대하고, 모든 값이 범위 안일 때 같은 토큰으로 성공을 기대한다.

- 1D: 마지막 패딩 꼬리 슬롯을 -1, 그다음 `CAPACITY`로 두고 `real_len`은 3으로 두어 꼬리까지 검사됨을 증명한다.
- 3행 M-RoPE: 세 번째 axis의 첫 슬롯을 -1, 첫 axis의 패딩 꼬리를 `CAPACITY`로 두어 모든 axis가 검사됨을 증명한다.

## 3. 기술적 선택과 그 이유

### 3.1 네 개의 인라인 루프 대신 헬퍼 하나

같은 axis-major 루프를 네 번 복사하면 서로 어긋난다. 헬퍼가 레이아웃 주석, 상한, 메시지 형식을 한 번만 담고, 호출 지점은 다른 값만 넘긴다: 호출자 이름, 개수, axis 수, capacity.

### 3.2 Rust가 아니라 C에서 검사

Rust 래퍼가 이미 범위 밖 명시 위치를 거부하지만, C API는 export되어 Rust 없이도 도달한다. #2936 리뷰가 decode에 세운 규칙을 prefill이 따른다. 가드는 직접 호출될 수 있는 경계에 있고, Rust 검사는 더 이른 친절한 실패로 남는다.

### 3.3 공개 API 확장 대신 test-support 전달자

C 가드를 테스트하려고 Rust prefill API에 "명시 위치" 매개변수를 추가하면 안정적인 시그니처가 바뀐다. 두 전달자는 `MLXCEL_XLA_TEST_SUPPORT`에서만 컴파일되고 프로덕션이 쓰는 같은 C 함수를 호출하므로, 공개 표면을 넓히지 않고 실제 경계를 테스트한다.

### 3.4 패딩 꼬리까지 검사

`real_len`은 시퀀스 중 의미 있는 부분을 제한할 뿐 그래프가 만지는 버퍼의 양을 제한하지 않는다. RoPE gather는 `positions` 버퍼 전체를 소비하므로 처음 `real_len`만 제한하면 `real_len..lp` 구간에 같은 out-of-bounds read가 남는다. 루프는 전체 개수를 덮어 기존 토큰 범위 검사와 일치한다.

## 4. 검증

환경: AMD Strix Halo(`gfx1151`), 고정 IREE ROCm 배포 `iree-rocm-3.12.0rc20260721`. 모든 cargo와 GPU 명령은 표준 ROCm 환경으로 `scripts/rocm_gpu_guard.sh --hold` 아래에서 실행했다.

- `cargo fmt --all -- --check`: 통과.
- `cargo clippy -p mlxcel-xla --features iree --lib --tests -- -D warnings`: 통과.
- `cargo test -p mlxcel-xla --lib validation::tests::emit_all_reproduces_registered_goldens`: 1 통과. golden은 수정되지 않았다(`git diff --stat src/lib/mlxcel-xla/assets`가 비어 있음).
- `cargo check --features xla-iree,test-utils --test xla_ragged_hardening`: 통과.
- `cargo test --release --features xla-iree,test-utils --test xla_ragged_hardening -- --ignored --nocapture`: 5 통과, 0 실패(새 테스트와 #2936 게이트 4개). 거부 줄은 호출자, axis, index, 값, capacity를 명시한다. 예: `xla_llama_prefill: position[axis=0 index=7]=-1 outside [0,8)`, `xla_llama_prefill_slot_logits: position[axis=2 index=0]=-1 outside [0,8)`.
- 유닛 worktree의 병합 전 계약 게이트: 프로젝트 XLA lib 테스트와 `llama_compat_manifest`, kernel-port-dispatch와 native-env-writes 감사.

## 5. 한계

- C 가드는 위치를 제한하고 토큰 id는 제한하지 않는다. `xla_llama_prefill`에는 `vocab` 매개변수가 없어 embedding index를 검사할 수 없고, `xla_llama_prefill_slot_logits`는 기존 토큰 범위 검사를 유지한다. 향후 `xla_llama_prefill` 토큰 가드는 vocabulary가 필요하며 이 이슈 범위 밖의 시그니처 결정이다.
- device 테스트는 1D `xla_llama_prefill`과 M-RoPE `xla_llama_prefill_slot_logits` 경계를 다룬다. diagnostics와 embeddings 변형은 같은 헬퍼와 같은 axis 규칙을 공유하지만 그 래퍼를 device에서 별도로 실행하지는 않는다.
- test-support 전달자는 `MLXCEL_XLA_TEST_SUPPORT`에서만 컴파일되고 device suite는 `tests/xla_ragged_hardening.rs` 관례대로 일반 CI에서 `#[ignore]`다.

## 6. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 4 |
| 추가 라인 | +283 |
| 삭제 라인 | -2 |
| 커밋 | 1 |
| device 게이트 | 5 통과 |

| 파일 | 변경 |
|------|------|
| `src/lib/mlxcel-xla/csrc/xla_iree.c` | 공유 위치 가드, 호출 4곳, NULL 검사, test-support 전달자 2개 |
| `src/lib/mlxcel-xla/src/iree.rs` | test-support extern와 래퍼 메서드 |
| `src/lib/mlxcel-xla/src/lib.rs` | test-support에서 `IreeLlama` 재노출 |
| `tests/xla_ragged_hardening.rs` | 실제 device 경계 테스트 |
