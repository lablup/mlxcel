# PR #2756: classic draft-model 경로 제거

**작성일**: 2026-11-01 (머지 전 작성, 당시 PR은 열려 있었음)
**상태**: 리뷰 중
**언어**: Rust
**위험도**: 중간 (v0.8.0에 예고한 CLI와 API 제거, #2768과 함께 내야 함)

## 요약

v0.8.0에 예고한 대로 배치 기반 엔진 밖에 남은 마지막 디코드 루프 `SpeculativeGenerator`를 제거한다(#2176). MTP, DFlash, prompt lookup은 엔진에서 계속 동작하고, 일반 모델 드래프터는 #2768로 돌아온다.

## 1. 문제 정의

classic 루프는 엔진의 디코드 경로와 중복되었고 자체 수락 규칙과 테스트를 갖고 있었다. v0.8.0-beta.1 변경 기록에서 제거 예정으로 공지되어 있었다.

## 2. 기술적 검토 사항

- **라우팅.** 명시했거나 자동 감지된 MTP 드래프터, 오프라인 경로가 있는 타깃의 DFlash 디렉터리는 이전과 같이 라우팅된다. `generate`에서 그 밖의 경우는 드래프터를 읽기 전에 실패하며 대안을 안내한다.
- **호환성 깨짐.** `generate`의 `--num-draft-tokens`, `MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT`, `MLXCEL_SPECULATIVE_ACCEPT_DIAG`가 없어졌다. `ServerConfig::num_draft_tokens`는 `--draft-max`와 `/props`를 위해 남는다.
- **서버.** `Classic` 디스패치가 없어져, round loop가 없는 종류는 일반 디코드로 넘어가지 않고 시작 시 실패한다.
- **테스트.** 수락 판단 검증은 스크립트 드래프터를 쓰는 엔진 토큰 전용 루프로 옮겼다. 제거된 루프만 고정하던 테스트는 삭제했다.

## 3. 기술적 선택과 그 이유

`speculative/stochastic_accept.rs`의 modified rejection sampling은 이 루프가 유일한 호출자였으므로 이식하지 않고 제거한다.

## 4. 변경 요약

| 항목 | 값 |
|---|---|
| 변경 파일 | 44개 |
| 변경 줄 | +1287 / -4800 |
| 추가 테스트 | 19개 |

마지막 main 병합 전 결합 스택의 워크스페이스 게이트: 14874개 통과, 0개 실패, 490개 무시.

## 5. 후속 조치

- 이 PR 직후 #2768을 머지한다. 이 PR만으로는 일반 모델 드래프터가 사라진다.
- 미검증: 오프라인 DFlash 실행의 성공 경로. 오프라인 경로는 Laguna뿐인데 로컬에 Laguna DFlash 드래프터가 없어 fixture 테스트로만 고정되어 있다.
