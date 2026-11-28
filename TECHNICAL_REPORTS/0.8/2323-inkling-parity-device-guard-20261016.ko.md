# PR #2323: Inkling 정확 일치 테스트가 디바이스 가드를 잡도록 수정

**날짜**: 2026-10-16
**상태**: 머지 전 CUDA(GB10)에서 구현 및 검증 완료, 이슈 #1622는 오케스트레이터의 머지 후 반복 실행 결과가 게시될 때까지 열린 상태 유지
**위험도**: 낮음 (테스트 전용 변경: 테스트 세 개가 기존 가드를 잡고 테스트 헬퍼 하나가 불일치 보고를 반환함, 프로덕션 코드 변경 없음)

## 요약

이슈 #1622는 멀티스레드 전체 lib 스위트에서 `vision::inkling_vl::tests::mixed_prefill_scatter_order_is_normalized_text_then_image_then_audio`가 간헐적으로 실패하는 문제다. 단독 실행에서는 통과한다. 이 PR은 원인을 MLX의 프로세스 전역 기본 디바이스를 둘러싼 테스트 하네스 경합으로 분류한다. scatter, merge, tower 경로의 결함이 아니고, 프로덕션 관점의 부하 민감 수치 문제도 아니다. 이슈의 인수 조건에 전체 `--lib` 20회 실행이 포함되어 있고 이는 머지 후 오케스트레이터가 수행하므로 PR은 `Refs #1622`를 쓴다.

1. `vision::merge_tests::cpu_device` 같은 형제 테스트는 `lock_default_device()` 아래에서 기본 디바이스를 CPU로 옮긴다. Inkling-VL의 허용 오차 0 parity 테스트는 락을 잡지 않았으므로, 병렬 libtest 실행에서는 비교의 한쪽이 CPU에서, 다른 쪽이 GPU에서 만들어질 수 있었다. 두 백엔드는 마지막 비트에서 다르고(`0xbfcd2ef1` 대 `0xbfcd2ef0`), 비교는 rtol = atol = 0인 `allclose`를 쓴다.
2. `vision::inkling_vl`의 정확 일치 테스트 두 개와 `models::inkling::tiny_tests::mtp_partial_accept_restores_then_replays_kv_and_conv_state_exactly`가 이제 기본 디바이스 락을 잡는 `mlx_test_guard()`를 보유한다.
3. `arrays_equal`은 단순 bool 대신 불일치 보고(shape, 처음 다른 flat 인덱스와 두 값 및 비트 패턴, 불일치 개수, 최대 차이)를 반환한다. 이슈가 요구한 사항이다.

## 문제 정의

이 테스트는 하나의 호출 순서가 한 프로세스 안에서 두 번 호출되어도 비트 단위로 같은 결과를 내는지 검증한다. `expected`는 `prepare_input_embeddings` 다음에 `merge_audio_embeddings`를 호출한 결과이고, `actual`은 같은 두 호출로 이루어진 `prepare_input_embeddings_with_audio`다. 따라서 scatter 순서 결함이라면 단독 실행을 포함해 매번 실패해야 하는데 그렇지 않았다. 단 한 번의 관찰(2026-09-04, `cargo test --workspace`, 7722개 통과 1개 실패)은 불리언 외에 데이터를 남기지 않았다. 그래서 원인이 하네스 경합일 수도, tower의 부하 민감 수치일 수도, 드문 scatter 결함일 수도 있었고, 각각 수정 방향이 다르다.

`vision::merge::tests`의 테스트별 CPU 가드는 관찰 하루 전인 2026-09-03에 #1595로 들어왔다. MLX 기본 디바이스는 프로세스 전역 static이므로 이 변경이 경합을 가능하게 했다. #2058(llmjp_vl, #2060에서 수정)과 같은 부류다.

### 증거

| 확인 | 결과 |
|------|------|
| 같은 입력의 CPU 대 GPU | 비트 단위로 다름: embed RMSNorm이 인덱스 0에서 `0xbfcd2ef1` 대 `0xbfcd2ef0`, audio merge만 CPU에서 돌리면 audio 행이 달라짐, 최대 절댓값 차이 1.6e-7 |
| 한 프로세스에서 1000회 반복(두 호출 경로 교대) + 모델 50개 새로 생성 | 불일치 0 |
| 별도 프로세스 25개 | 25개 모두 통과, 모든 프로브 프로세스의 출력 해시 동일 |
| `cpu_device()`처럼 디바이스를 바꾸는 프로브 스레드, 가드 없음 | 1000회 중 882회 실패(첫 실행은 400회 중 175회), 모두 인덱스 0의 마지막 비트 차이 |
| 같은 프로브, `mlx_test_guard()` 보유 | 1000회 중 0회 실패 |
| `vision::`를 `--test-threads=1`로 실행 | 25회 모두 통과(549개 통과, 0개 실패) |
| 2026-09-04 이후 Nightly verify (Metal, `lablup-macmini-12`) | 49회, 48회에서 테스트가 실행되어 48회 모두 통과, 1회는 테스트 전에 실패 |

프로세스 내 반복과 프로세스 간 반복은 scatter, merge, tower 경로의 비결정성을 배제한다. fixture는 이름으로 키가 지정되어 HashMap 순서가 개입하지 않고, 이 경로는 맵을 순회하지 않는다. nightly 기록도 맞는다. 모든 nightly가 경합을 막는 `--test-threads=1`로 실행되고, 멀티스레드 `--workspace` 실행에서만 실패가 나왔다.

## 기술적 선택과 그 이유

- `serial_test`를 추가하지 않고 기존 `mlx_test_guard()`를 잡는다. 이슈는 분류 전에 `#[serial]`, `--test-threads=1` 변경, 완화된 허용 오차를 금지하며, 이 가드는 형제 테스트가 이미 쓰는 락이라 의존성이 늘지 않는다.
- 허용 오차는 정확히 0으로 유지한다. 이 비교는 scatter 순서 오류를 잡기 위한 것이고, CPU/GPU 차이는 하네스 산물이지 드리프트를 허용할 이유가 아니다.
- `image_prefill_keeps_hmlp_prepared_embeddings_on_classic_wrapper_path`와 MTP 롤백 테스트도 가드한다. 둘 다 두 계산을 정확히 비교하며 같은 노출을 가진다. 실제로 실패가 관찰된 것은 mixed-prefill 테스트뿐이다.
- 이슈의 재사용 지침대로 두 번째 헬퍼를 만들지 않고 `arrays_equal`을 그 자리에서 확장한다. 호스트 복사본에 `!=`를 써서 `equal_nan = false`인 `allclose`와 맞춘다. NaN은 무엇과도 같지 않고 -0.0은 0.0과 같다.
- 디바이스를 바꾸는 프로브는 커밋하지 않았다. 경합을 재현하지만 타이밍에 의존하므로 회귀 테스트가 아니라 증거다.

## 변경 사항

- `src/vision/inkling_vl.rs`: `arrays_equal`이 불일치 보고를 담은 `Result<(), String>`을 반환하고, 정확 일치 테스트 두 개가 `mlx_test_guard()`를 보유하며 보고와 함께 panic한다.
- `src/models/inkling/tiny_tests.rs`: `mtp_partial_accept_restores_then_replays_kv_and_conv_state_exactly`가 `mlx_test_guard()`를 보유한다.

## 검증

- CUDA(GB10, `--profile test-fast`): `models::inkling`(28개)과 `vision::inkling_vl`(5개)이 통과하고, clippy `-D warnings`, fmt, 계약 검사 5개가 깨끗하다.
- 되돌림 확인: 프로브 스레드는 가드 없는 비교를 1000회 중 882회, 가드를 잡은 비교를 1000회 중 0회 실패시킨다.
- 테스트 코드만 바뀌었으므로 실제 체크포인트는 쓰지 않았다.

## 미검증 사항과 후속 조치

- CUDA에서는 멀티스레드 `vision::` 실행을 측정할 수 없다. 수정 유무와 상관없이 6회 모두 76~293개 테스트 뒤에 `cudaStreamEndCapture ... failed`로 중단되어 이 테스트에 도달하지 못했다. 따라서 수정은 end-to-end 멀티스레드 통과가 아니라 프로브로 입증되었다.
- 인수 조건의 전체 `--lib` 20회 반복은 오케스트레이터에게 남겨 둔다. 이슈 #1622는 그 결과가 게시될 때까지 열어 둔다.
- 허용 오차 0으로 두 계산을 비교하는 새 테스트는 같은 이유로 `mlx_test_guard()`를 잡아야 한다. 불일치 보고 덕분에 재발 시 원인을 한눈에 구분할 수 있다. 인덱스 0의 마지막 비트 차이는 디바이스를, 순열이나 부분 쓰기는 scatter를 가리킨다.
