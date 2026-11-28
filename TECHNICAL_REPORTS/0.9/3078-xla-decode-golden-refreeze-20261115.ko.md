# 기술 보고서: PR #3078 - 번들 xla decode golden 재생성

**작성일**: 2026-11-15
**상태**: 완료
**위험도**: 낮음

## 요약

#3048 이후 main에서 `mlxcel-xla` lib 테스트 3개가 실패해 왔다. decode KV 캐시를 module global로 옮긴 뒤 번들 `decode.mlir` 자산을 다시 생성하지 않았기 때문이다. 이 PR은 등록된 15개 계열의 `decode.mlir`를 crate 자체의 `ArchFixture::emit_all`로 재생성한다. prefill과 ragged-logits 자산은 바뀌지 않는다.

## 1. 문제 정의

#3048에서 단일 시퀀스 decode가 KV 캐시를 module global(`@kcache` / `@vcache`)에 두도록 바뀌어, decode `@main`은 더 이상 캐시를 인자로 받거나 반환하지 않는다. emitter는 이에 맞춰 바뀌었지만 커밋된 golden은 옛 시그니처 그대로였다. 그 결과 `emitter::tests::from_json_reproduces_bundled_assets_byte_for_byte`, `validation::tests::emit_all_reproduces_registered_goldens`, `validation::tests::registered_fixtures_are_byte_exact`가 실패했다. llama-3.2-1b에서는 2번째 줄에서 처음 어긋났고, 줄 수는 golden 1350줄, 생성 결과 1361줄이었다. PR 작업 중 `cargo test --workspace`를 돌리는 job이 없어 CI가 잡지 못했다(#3075).

## 3. 기술적 선택과 그 이유

기준은 golden이 아니라 emitter다. 시그니처 변경은 #3048이 의도한 것이고, 재생성 결과도 decode 모듈만 달라진다. 재생성은 validation 테스트가 비교에 쓰는 바로 그 `ArchFixture::emit_all`로 했기 때문에, 자산은 손으로 고친 것이 아니라 테스트 대상 코드 경로에서 나온다. `assets/llama-3.2-1b/README.md`에는 새 decode 입력을 반영했다.

## 7. 변경 요약

| 항목 | 값 |
|------|-------|
| 변경 파일 | 16개 (`decode.mlir` 15개, README 1개) |
| 소스 변경 | 없음 |
| 테스트 | `cargo test -p mlxcel-xla --lib -- emitter:: validation::`: 220 통과, 0 실패, 17 무시 |

관련: #3048 (module-global decode KV), #3075 (이슈).
