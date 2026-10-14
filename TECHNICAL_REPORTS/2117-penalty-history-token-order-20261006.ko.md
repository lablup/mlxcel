# 기술 보고서: PR #2117 - 페널티가 걸린 스텝을 방금 읽은 토큰까지 포함해 샘플링

**작성일**: 2026-10-06

**상태**: GB10에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Rust(디코드 루프, 공유 테스트 픽스처, 테스트)

**위험도**: 낮음. 출력이 바뀌는 것은 이력을 읽는 샘플러를 쓸 때뿐이고, 그때도 문서화된 의미로 바뀐다. 페널티가 없는 경로는 순서와 출력, 처리량이 그대로다.

## 요약

`CxxGenerator`는 `mlxcel generate`, `run`, 대화형 chat 뒤에서 도는 디코드 루프다. 이 생성기의 파이프라인 루프 네 개는 토큰 t를 `token_history`에 넣기 전에 스텝 t+1을 샘플링했고, 그래서 repetition·frequency·presence 페널티와 DRY는 늘 한 토큰 늦은 이력을 봤다(#2090). 이제 공용 헬퍼가 스텝 t+1의 샘플을 만들기 전에 토큰 t를 읽어 기록하며, 그동안 스텝 t+1의 forward는 이미 디바이스에서 돈다. 페널티 없는 출력과 처리량은 그대로이고 페널티 경로의 처리량도 변하지 않았다.

## 1. 문제 정의

파이프라인 루프는 스텝 t의 아직 계산되지 않은 샘플로 스텝 t+1을 만들어 GPU가 스텝 사이에 쉬지 않게 한다. forward에는 그 지연 토큰만 있으면 되지만, 이력을 읽는 샘플러는 토큰 t가 호스트의 `token_history`에 들어 있어야 한다. 네 루프(`generate_streaming`, `generate_streaming_with_embeddings`, `generate_with_stats_and_embeddings`, `generate_with_stats`) 모두 반복의 맨 앞에서 스텝 t+1 샘플을 만들고 맨 뒤에서 토큰 t를 넣었다. 첫 디코드 토큰은 프롬프트만 보고 샘플링됐고, `--repeat-last-n 3`에서는 창이 한 칸 밀려 같은 토큰이 바로 뒤에 다시 나와도 페널티를 받지 않을 수 있었다. mlx-lm(`generate_step`은 logits processor를 부르기 전에 소비한 토큰을 이어 붙인다)과 llama.cpp(샘플마다 `common_sampler_accept`)는 둘 다 그 토큰을 포함한다.

이 버그는 PR #2074를 하드닝하다 찾았다. 검증 위치를 순서대로 샘플링하는 prompt lookup이 페널티 아래에서 `CxxGenerator`와 달랐고, 호스트 쪽 시뮬레이션으로 확인하니 prompt lookup은 현재 이력과, `CxxGenerator`는 한 토큰 늦은 이력과 정확히 맞았다.

## 2. 변경 요약

- `generate.rs`의 `sample_next_step(next_logits, y, sampling, needs_history, token_history, sampler_state)`: 이력 경로에서는 `async_eval(next_logits)`를 부르고 `y`를 읽어 넣은 다음 샘플을 만들며, 읽은 토큰을 돌려줘 루프가 다시 쓰게 한다. 이력이 필요 없는 경로에서는 예전 `sample_token_optimized` 결과를 그대로 돌려준다.
- 각 루프는 `current_token: Option<i32>`를 들고, 헬퍼가 읽지 않았을 때만 `y`를 읽고 헬퍼가 넣지 않았을 때만 `token_history`에 넣는다.
- `test_support::induction`: `InductionModel`, `lcg_tokens`, `sequential_reference`를 `prompt_lookup_tests.rs`에서 옮겨 두 테스트 파일이 함께 쓴다.
- `generate_history_tests.rs`: 네 진입점을 순차 기준 디코더와 비교한다. 3토큰 repetition 창, 전체 이력 카운트(증분 `SamplerState`), frequency/presence, DRY, 페널티 없음.
- prompt lookup의 페널티 테스트가 다시 `CxxGenerator`를 기준 디코더와 대조한다.
- 페널티 경로에서 진단 도구가 무엇을 보는지 헬퍼에 문서화했다.

## 3. 기술적 선택과 그 이유

**forward를 먼저 제출하고 그다음 읽는다.** 이슈가 제안한 방식이다. 스텝 t+1의 forward는 지연 토큰만 필요하므로 호스트가 읽기 전에 제출하면 읽는 동안에도 디바이스가 일하고, 호스트 이력이 필요한 샘플러만 기다린다. 페널티 경로를 아예 동기로 돌리는 대안은 그 겹침을 잃고, 디바이스 쪽 지연 이력(mlx-lm 방식)은 호스트 `&[i32]`를 받는 `SamplerState`와 DRY를 다시 써야 한다.

**네 루프에 헬퍼 하나.** 순서 규칙은 함수 하나에만 있고, 루프는 기존의 읽기·넣기 자리로 `current_token`을 넘기기만 한다.

**되돌려서 확인한 판별 테스트.** 예전 순서로 되돌리면 repetition 창, frequency/presence, 전체 이력 카운트, DRY 테스트와 prompt lookup 페널티 테스트가 실패한다. 처음 쓴 전체 이력 테스트는 repetition 페널티를 써서 버그가 있어도 통과했다. 작은 어휘의 토큰이 모두 이력에 들어가면 페널티 집합이 더는 바뀌지 않기 때문이다. 그래서 등장할 때마다 값이 움직이는 카운트 기반 페널티로 바꿨다.

## 4. 검증

- `generate::history_tests`와 `speculative::prompt_lookup`: 44개 통과.
- GB10에서 한 트리로 빌드한 수정 전후 release 바이너리(`generate.rs`만 교체), greedy, 3회 중앙값, 교차 실행: 페널티 없는 처리량은 수정 전의 0.990x~1.007x, 페널티 처리량은 0.997x~1.010x다. 페널티 없는 출력은 다섯 경우 중 넷에서 같았다. 나머지 하나(Qwen3-1.7B, 800토큰 이야기)는 바꾸지 않은 바이너리로도 실행마다 결과가 달라서(세 번 실행이 문자 1300~1351에서 서로 갈라진다) 문자 1432에서의 전후 차이는 이번 변경에 대해 아무것도 말해 주지 않는다.
- 서버 배치 스케줄러: 영향 없음. lookahead는 `config_supports_fused_batch_except_bias`를 통과한 행만 받는데 이 함수가 `needs_token_history`를 거부하고, 행 단위 경로는 토큰을 샘플링한 틱 안에서 읽어 넣으며, prefill은 디코드 전에 첫 토큰을 넣는다.

## 5. 남은 위험과 검증하지 못한 것

- **GB10에서 plain greedy 디코딩이 실행마다 재현되지 않는다**(적어도 긴 Qwen3-1.7B 응답 하나). 원래 있던 문제이고 추적하는 이슈가 없으며, 여기서 원인은 조사하지 않았다.
- **페널티 경로의 진단 도구**(디코드 그래프 내보내기, astype 집계, Metal 캡처, 파이프라인 프로파일)는 이제 forward 없이 샘플러만 보고, 호스트 대기를 샘플 시간으로 센다. 헬퍼에 문서화했다.
- **`CxxGenerator`는 mirostat·adaptive-p 상태를 싣지 않는다.** 지금은 도달할 수 없다. `generate`와 `chat`은 둘 다 꺼 두고, 서버는 `CxxGenerator`를 쓰지 않는다.
- **`SpeculativeGenerator`**는 각 초안을 그 라운드의 앞선 초안이 빠진 이력으로 샘플링한다(이번 범위 밖).
- Metal은 측정하지 않았다. 변경은 백엔드와 무관한 호스트 로직이다.

## 6. 학습 포인트

- **파이프라인 루프에서는 소비자마다 호스트에 무엇이 필요한지 확인한다.** forward는 지연 토큰으로 돌 수 있었고 샘플러는 그럴 수 없었다. 제출을 나누면 겹침을 지키면서 의미도 바로잡을 수 있다.
- **회귀 테스트는 버그에서 실패해야 한다.** 여기서도 다섯 개 중 하나가 샘플러를 바꾸기 전까지 예전 순서에서 통과했고, 일부러 되돌려 봐야 알 수 있었다.
- **전후 패리티에는 자기 자신과의 패리티가 기준선으로 필요하다.** 스스로도 재현되지 않는 greedy 응답으로는 차이를 변경 탓으로 돌릴 수 없다.

## 7. 관련 항목

- 이슈 #2090, PR #2074(발견한 곳), PR #2092(prompt lookup 정책, 같은 테스트 픽스처).
