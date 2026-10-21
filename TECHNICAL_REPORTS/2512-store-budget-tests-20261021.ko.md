# PR #2512: store_budget helper 단위 테스트

**Date**: 2026-10-21
**Status**: Completed (PR open)

## 요약

PR #2512(#1662 해결)는 `src/server/store_budget.rs`의 순수 helper 두 개에 단위 테스트 9개를 추가한다. 테스트 파일은 `src/server/store_budget_tests.rs`다. 이 PR 전에는 두 helper 모두 테스트가 없었다. 두 helper는 메모리 안의 Responses 저장소와 Conversations 저장소의 크기를 제한한다. 그래서 어느 한쪽이 조용히 바뀌면 메모리 계산이나 eviction 순서가 달라진다. 이 PR은 production 코드를 바꾸지 않는다. 리뷰와 보안 점검에서 고칠 사항은 나오지 않았다.

## 1. 고정한 동작

| Helper | 테스트가 고정하는 계약 |
|---|---|
| `serialized_json_len_saturating` | scalar, escape 문자와 비ASCII 문자열, 빈 컨테이너와 중첩 컨테이너, unsized `&str`에서 `serde_json::to_vec(..).len()`과 같은 값을 낸다. 직렬화가 실패하면 `usize::MAX`를 반환한다. 오류를 내는 `Serialize` 구현과 tuple key를 쓰는 `HashMap`으로 확인한다. |
| `LruKey` `Ord` | `last_accessed`, `sequence`, `id` 순서로 비교한다. 단계마다 그 단계만 결과를 정하도록 만든 테스트가 하나씩 있다. 전체 정렬 테스트로 eviction 순서를 확인한다. 모든 쌍에서 `partial_cmp`가 `cmp`와 같고, 같은 key는 `Equal`이다. |

## 2. 기술적 선택

- **결정적인 시간값.** 모든 key는 기준 `Instant` 하나에 `Duration`을 더해 만든다. 따라서 결과가 시계 해상도에 좌우되지 않는다.
- **saturating add는 직접 테스트하지 않는다.** `usize::MAX` 바이트에 도달하려면 그만큼을 실제로 할당해야 한다. private struct `SaturatingByteCounter`는 그대로 두고 public helper를 거쳐서만 검증한다.
- **tuple key 테스트는 전제부터 확인한다.** 먼저 그 map에서 `serde_json::to_vec`이 실제로 실패하는지 확인한다. 나중에 serde_json이 이런 key를 받아들이게 되더라도 테스트가 엉뚱한 이유로 통과하지 않는다.
- **CHANGELOG 항목은 없다.** 앞선 테스트 전용 PR(#2481, #2482)도 항목을 추가하지 않았다.

## 3. 검증

| 항목 | 결과 |
|---|---|
| `server::store_budget_tests` | 9개 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | 문제 없음 |
| `cargo fmt --check` | main을 merge한 뒤 문제 없음 |

모든 실행은 macOS에서 `MLXCEL_BUILD_METAL=0 --profile test-fast --lib`로 했다.

**CI 참고.** PR을 열 때 main은 #2505가 남긴 check 4개에서 실패했다. 실패한 check는 `cargo-fmt`, `kernel port dispatch`, `env var docs`, `python tooling tests`다. #2516이 이를 고쳤고, PR branch는 마지막 CI 전에 main을 merge했다.

## 4. 남은 항목

이 PR에 남은 항목은 없다.
