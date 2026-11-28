# PR #3303: eos 범위, Z1T 시퀀스, K2-Horizon reasoning 분리 보완

**작성일**: 2026-11-23
**상태**: PR 열림 (#3299 종료). 좁은 단위 및 라우트 테스트 통과, M3 Ultra 호스트에서 실제 체크포인트 `k2-horizon-0.9b` (서버, 세 effort 모두, 스트리밍과 비스트리밍), `limite-1b-violetto`, `z1t-0-mlx` (generate 스모크) 확인
**위험도**: Low

## 요약

최근 모델 포팅의 후속 작업 세 가지를 함께 처리한다. Limite와 Xing4.0은 음수이거나 `i32`에 들어가지 않는 `eos_token_id`를 다른 토큰으로 감싸지 않고 거부한다. Z1T는 준비되지 않은 시퀀스 id에 대해 panic 대신 model-owned 상태를 지연 생성한다. IFM K2-Horizon의 `reasoning_effort`별 think 쌍 세 개를 모든 reasoning 마커 테이블에 등록해, reasoning이 `content`가 아니라 `reasoning_content`로 간다.

## 1. 문제 정의

### 1.1 `eos_token_id` 축소 변환

`LimiteConfig::eos_token_ids`와 Xing4.0의 `parse_eos`는 `as_i64()` 값을 `as i32`로 변환했다. `4294967298`은 `2`가 되어 잘못된 토큰에서 생성이 멈췄고, 샘플링된 id와 절대 같을 수 없는 `-1`도 받아들였다.

### 1.2 Z1T 미준비 시퀀스 id

`Z1t::forward_rows_for_sequence`는 `with_existing_sequence_state`를 써서, 스케줄러가 `prepare_sequence_state`를 호출하지 않은 id에서 `Z1T missing model-owned sequence state`로 panic했다. 형제 패밀리 32개는 `with_or_create_sequence_state`를 쓴다.

### 1.3 K2-Horizon reasoning이 `content`에 남음

템플릿은 생성 프롬프트를 `<ifm|think>\n`, `<ifm|think_fast>\n`, `<ifm|think_faster>\n` (high, medium, low)로 끝낸다. 모델은 trace와 짝이 맞는 닫는 태그만 출력한다. 어떤 마커 테이블도 이 쌍을 몰라 trace와 닫는 태그가 `content`로 갔다.

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 잘못된 config로 인한 잘못된 정지 토큰 | Medium | Low |
| 미준비 Z1T 시퀀스 id로 인한 서버 panic | High | Low |
| K2-Horizon 클라이언트가 `content`에서 reasoning과 태그를 봄 | Medium | High |

## 2. 기술적 검토 사항

### 2.1 보안 관점

새 stream-filter delimiter는 `<think>`와 마찬가지로 모든 패밀리의 출력에 적용된다. 여섯 마커 중 어느 것도 다른 항목의 prefix나 substring이 아니고, `max_delim_len`은 테이블에서 계산되어 커지지 않으므로 partial-match 보류 창은 그대로다. K2가 아닌 모델이 `<ifm|think>` 문자열을 그대로 출력하면 reasoning 마커로 처리되는데, 이는 다른 모든 마커와 같은 노출이다.

### 2.2 성능 관점

primed 감지는 요청당 한 번 최대 세 쌍의 벡터를 만든다. 예산 추적기의 close 비교는 토큰당 두 칸짜리 `Option` 비교가 늘어난다. 둘 다 측정할 수 없는 수준이다.

### 2.3 호환성/의존성 관점

- breaking change 없음. `ThinkingTokenIds`에 필드가 추가되었고 struct literal은 `ThinkingTokenIds::new(open, close)`로 바뀌었다.
- 해석된 open이 `<ifm|think>`가 아닌 모든 패밀리에서 `primeable_pairs()`는 해석된 쌍 하나만 반환하므로 primed 감지는 변하지 않는다.
- 잘못된 `eos_token_id`를 가진 체크포인트 config는 이제 값을 명시한 오류로 로드에 실패한다. 이전에는 잘못된 정지 id로 로드되었다.

## 3. 기술적 선택과 그 이유

### 3.1 K2-Horizon effort 변형의 위치

`ThinkingMarkers`는 해석된 쌍 하나만 가진다. 세 open이 모두 vocab에 있지만 tokenizer는 하나만 보고할 수 있다.

| 대안 | 장점 | 단점 |
|------|------|------|
| `ThinkingMarkers`에 변형 필드 추가 | 데이터가 마커와 함께 이동 | 모든 struct literal 변경, 다른 패밀리에서는 항상 빈 필드 |
| Kimi K3처럼 K2 전용 마커 함수 | K3와 같은 형태 | 여전히 쌍 하나, medium과 low를 감지하지 못함 |
| **선택: 해석된 open을 키로 하는 `THINK_EFFORT_VARIANTS` 테이블** | primed 감지와 예산 resolver가 같은 테이블을 읽음, 다른 패밀리 영향 없음 | 마커 probe 옆에 조회가 하나 더 생김 |

`prompt_primed_open_close_marker`는 프롬프트가 끝나는 open에 해당하는 쌍의 close를 반환하므로, 스트리밍 `ThinkingDelimiterEcho`와 `--reasoning-format none`이 올바른 쌍을 되돌려 준다.

### 3.2 예산 close

스케줄러는 시작 시 모델당 `ThinkingTokenIds` 하나를 해석하며 요청별 primed 변형을 알지 못한다. `alt_closes` 덕분에 추적기가 `</ifm|think_fast>`와 `</ifm|think_faster>`를 close로 인식한다. 이것이 없으면 medium effort에서 예산이 걸린 요청은 답변까지 reasoning으로 세다가 답변 중간에 close를 강제로 넣는다. 강제 close 자체는 `</ifm|think>`로 남는다. 모든 필터가 이를 받아들이며, primed 변형을 `ThinkingState`로 전달하는 일은 후속 작업으로 남긴다.

### 3.3 Xing4.0 검증 위치

`Xing40Config`는 `eos_token_ids: Vec<i32>`를 저장하므로 `validate()`는 원본 JSON 값을 볼 수 없다. `parse_eos`가 역직렬화에서 범위 오류를 반환하고(이미 `validate()`가 실행되는 위치), `validate()`는 코드에서 만든 config의 음수 id를 거부한다. 필드 타입을 유지하고, 같은 파일의 동시 RoPE 변경이 건드리지 않는 `parse_eos`와 `validate` 안에서만 수정한다.

## 4. 구현 상세

- `src/loading/mod.rs`: `checked_eos_token_ids(value) -> Result<Vec<i32>, String>`, `i32::try_from` 사용, 음수 거부.
- `src/models/limite_config.rs`, `src/models/xing4_0.rs`: helper 호출부.
- `src/models/z1t.rs`: `with_or_create_sequence_state(seq_id, || self.fresh_states(), ..)`.
- `src/tokenizer/thinking.rs`, `src/tokenizer/mod.rs`: `THINK_EFFORT_VARIANTS`, `ThinkingMarkers::primeable_pairs`, `SINGLE_TOKEN_PAIRS`에 `<ifm|think>` 추가.
- `src/reasoning_display.rs`: `primeable_pairs` 기반 primed 감지.
- `src/server/thinking_budget.rs`: `alt_closes`, `is_close`, K2-Horizon resolver 분기.
- `src/server/tool_calls/stream_filter.rs`, `parser.rs`, `src/server/routes/chat.rs`: delimiter 행, canonical pair, marker pair, unclosed-primed close 목록.

## 5. 학습 포인트

prompt-primed reasoning 블록은 출력이 아니라 렌더링된 프롬프트에서 감지된다. 필터는 thinking 상태에서 시작하고 출력에는 close만 있다. 여러 open 중 하나를 고르는 템플릿이라면 primed 검사에 모든 open이, 각 소비자(stream filter, parser, unclosed 검사, budget)에 모든 close가 있어야 한다. 그렇지 않으면 해석된 쌍이 아닌 변형에서 분리가 조용히 실패한다.

## 7. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 21 |
| 추가 라인 | 약 600 |
| 삭제 라인 | 약 90 |
| 추가 테스트 | 9 |

| 카테고리 | 요약 |
|----------|------|
| 정확성 | eos 범위 검사, Z1T 지연 상태, 다섯 테이블과 unclosed-close 목록에 K2-Horizon reasoning 분리 |
| 테스트 | Limite와 Xing4.0 범위 테스트, Z1T fresh-id logits 일치, 실제 `/v1/chat/completions` 핸들러를 포함한 K2-Horizon 테스트 6개 |
| 문서 | `docs/supported-models.md`에서 제한 사항 제거 |

## 8. 후속 조치

- primed 변형의 close id를 `ThinkingState`에 전달해, medium이나 low effort의 예산이 `</ifm|think_fast>` 또는 `</ifm|think_faster>`를 강제하도록 한다.

## 부록 A. 테스트 결과

- Metal, GPU lock 하에서: `models::limite` 15, `models::xing4_0` 17, `models::z1t` 14, `coverage_tests` 18 통과.
- `server::k2_horizon_reasoning_tests` 6, `server::tool_calls` 502, `server::thinking_budget` 46, `thinking` 188, `tokenizer::` 143, `server::routes::chat` 73, `server::chat_request` 131, `primed` 62, `loading::` 428 통과.
- `mlxcel-server -m models/k2-horizon-0.9b`: high와 medium은 `reasoning_content`에 trace, `content`에 `42`를 반환. low는 즉시 닫혀 `reasoning_content`가 비어 있음. 어떤 응답에도 `ifm|think` 문자열 없음.
- `limite-1b-violetto`, `z1t-0-mlx`에서 `mlxcel generate` 스모크로 로드와 생성 확인.
